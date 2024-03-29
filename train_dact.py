import json
import os
import time

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from generate_data import generate_data_jit

from others import datetime_str, human_readable_time
from solvers.dact.agent.ppo import PPO, train_batch
from solvers.dact.options import get_options
from solvers.dact.problems.problem_tsp import TSP, _TSPDataset
from solvers.dact.utils.logger import log_to_tb_val
from utils.data_utils import load_dataset


def validate(rank, problem, agent, val_data_path, tb_logger, distributed=False, _id=None):
    opts = agent.opts
    # Validate mode
    agent.eval()
    problem.eval()

    data = load_dataset(val_data_path)[: opts.val_size]
    val_dataset = _TSPDataset(data)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.val_size, shuffle=False, num_workers=0, pin_memory=True)

    s_time = time.time()

    for batch_id, batch in enumerate(val_dataloader):
        assert batch_id < 1
        best_value, cost_hist, best_hist, r, rec_history, _best_sol = agent.rollout(
            problem, opts.val_m, batch, do_sample=True, record=False, show_bar=rank == 0
        )

    initial_cost = cost_hist[:, 0]
    time_used = torch.tensor([time.time() - s_time])
    costs_history = cost_hist
    search_history = best_hist
    reward = r

    # log to tb
    if not opts.no_tb:
        log_to_tb_val(
            tb_logger,
            time_used,
            initial_cost,
            best_value,
            reward,
            costs_history,
            search_history,
            batch_size=opts.val_size,
            val_size=opts.val_size,
            dataset_size=len(val_dataset),
            T=opts.T_max,
            show_figs=opts.show_figs,
            epoch=_id,
        )

    return best_value.mean(), torch.std(best_value) / np.sqrt(opts.val_size)


if __name__ == "__main__":

    dir_prefix = "/data/yzhang"
    save_dir = "runs/dact"
    tmp_data_dir = "dact_tmpfiles"
    if len(dir_prefix) > 0:
        save_dir = os.path.join(dir_prefix, save_dir)
        tmp_data_dir = os.path.join(dir_prefix, tmp_data_dir)

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opts = get_options("")
    opts.distributed = False
    opts.use_cuda = True
    opts.device = "cuda"

    opts.seed = seed

    opts.graph_size = 100
    opts.val_dataset = "data/clust100_seed1005.pkl"
    opts.max_grad_norm = 0.1
    opts.Xi_CL = 2
    opts.checkpoint_epochs = 10
    opts.batch_size = 500
    opts.data_type = "clust"
    # opts.resume = True
    # opts.load_path = "runs/dact/n100_20220604_095550/epoch-100.pt"

    tsp_problem = TSP(
        p_size=opts.graph_size,
        step_method=opts.step_method,
        init_val_met=opts.init_val_met,
        with_assert=opts.use_assert,
        P=opts.P,
    )
    agent = PPO(tsp_problem.NAME, tsp_problem.size, opts)

    if opts.load_path is not None:
        agent.load(opts.load_path)

    if opts.resume:
        epoch_resume = int(os.path.basename(opts.load_path).split("-")[1].split(".")[0])
        agent.opts.epoch_start = epoch_resume + 1
        print(f"loaded {opts.load_path}")

    run_name = f"n{opts.graph_size}_{opts.data_type}_{datetime_str()}"
    save_dir = os.path.join(save_dir, run_name)
    opts.run_name = run_name
    opts.save_dir = save_dir
    print("saved to", save_dir)
    tb_logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "opts.json"), "w") as w:
        w.write(json.dumps(vars(opts), indent=4))

    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device)

    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        pbar = tqdm.tqdm(
            total=(opts.K_epochs) * (opts.epoch_size // opts.batch_size) * (opts.T_train // opts.n_step),
            desc=f"Epoch {epoch + 1}/{opts.epoch_end} [data generation]",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            leave=False,
        )

        data_seed = np.random.randint(1000000)
        data, duration = generate_data_jit(
            tmp_data_dir,
            opts.epoch_size,
            opts.graph_size,
            opts.data_type,
            seed=data_seed,
        )
        if epoch == opts.epoch_start:
            pbar.write(f"generated {opts.epoch_size} instances (time={human_readable_time(duration)})")

        training_dataset = _TSPDataset(data)
        # prepare training data
        training_dataloader = DataLoader(
            training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)
        lr_str = "lr: actor {:.3e}, critic {:.3e}".format(
            agent.optimizer.param_groups[0]["lr"], agent.optimizer.param_groups[1]["lr"]
        )
        pbar.set_description(f"Epoch {epoch + 1}/{opts.epoch_end} [train] ({lr_str})")
        t1 = time.time()

        for batch_id, batch in enumerate(training_dataloader):
            train_batch(0, tsp_problem, agent, epoch, step, batch, tb_logger, opts, pbar)
            step += 1
        
        t2 = time.time()
        pbar.close()
        duration = human_readable_time(t2 - t1)

        agent.lr_scheduler.step()

        # save new model after one epoch
        if not opts.no_saving and (
            (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.epoch_end - 1
        ):
            agent.save(epoch)

        # validate the new model
        best_value, value_std = validate(0, tsp_problem, agent, opts.val_dataset, tb_logger, _id=epoch)
        print(f"Epoch {epoch + 1}/{opts.epoch_end} [val] obj value: {best_value:.4f} +- {value_std:.4f}, duration={duration} ({lr_str})")
