import json
import os
import time

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from generate_data import generate_tsp_data

from others import DotDict, datetime_str, human_readable_time
from solvers.dact.agent.ppo import PPO, train_batch
from solvers.dact.problems.problem_tsp import TSP, _TSPDataset
from solvers.dact.utils.logger import log_to_tb_val
from utils.data_utils import load_dataset

graph_size = 100
T_max = 1500
num_data_aug = 1
model_path = None
val_dataset_path = "data/clust100_seed1005.pkl"

opts = DotDict(
    {
        ## Overall settings
        "problem": "tsp",
        "graph_size": graph_size,
        "dummy_rate": 0.5,
        "step_method": "2_opt",
        "init_val_met": "random",
        "no_cuda": False,
        "no_tb": True,
        "show_figs": False,
        "no_saving": True,
        "use_assert": False,
        "no_DDP": True,
        ## DACT parameters
        "v_range": 6.0,
        "DACTencoder_head_num": 4,
        "DACTdecoder_head_num": 4,
        "critic_head_num": 6,
        "embedding_dim": 64,
        "hidden_dim": 64,
        "n_encode_layers": 3,
        "normalization": "layer",
        ## Training parameters
        "RL_agent": "ppo",
        "gamma": 0.999,
        "K_epochs": 3,
        "eps_clip": 0.1,
        "T_train": 200,
        "n_step": 4,
        "best_cl": False,
        "Xi_CL": 2,
        "batch_size": 200,
        "epoch_end": 200,
        "epoch_size": 4000,
        "lr_model": 0.0001,
        "lr_critic": 3e-05,
        "lr_decay": 0.985,
        "max_grad_norm": 0.2,
        ## Inference and validation parameters
        "T_max": T_max,
        "eval_only": False,
        "val_size": 200,  # 1000
        "val_dataset": val_dataset_path,
        "val_m": num_data_aug,
        ## Resume and load models
        "load_path": model_path,
        "resume": None,
        "epoch_start": 0,
        ## Logs/output settings
        "no_progress_bar": False,
        "log_dir": "logs",
        "log_step": 50,
        "output_dir": "outputs",
        "run_name": "run_name",
        "checkpoint_epochs": 1,
    }
)

opts.world_size = 1
opts.distributed = False
opts.use_cuda = True
opts.P = 1e10
opts.device = "cuda"


def generate_data(opts):
    data_seed = np.random.randint(1000000)
    t1 = time.time()
    data = generate_tsp_data(
        "dact_tmpfiles",
        opts.epoch_size,
        opts.graph_size,
        distribution="clust",
        seed=data_seed,
        save=False,
        quiet=True,
    )
    t2 = time.time()
    return data, t2 - t1


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

    print("best value {:7.4f} +- {:6.4f}".format(best_value.mean(), torch.std(best_value) / np.sqrt(opts.val_size)))

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


if __name__ == "__main__":

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opts.seed = seed

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
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        print("Resuming after {}".format(epoch_resume))
        agent.opts.epoch_start = epoch_resume + 1

    save_dir = f"runs/dact/n{opts.graph_size}_{datetime_str()}"
    print("saved to", save_dir)
    opts.save_dir = save_dir
    tb_logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "opts.json"), "w") as w:
        w.write(json.dumps(dict(opts.items()), indent=4))

    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device)

    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):

        # Training mode
        print(
            "Training {}/{} actor lr={:.3e} critic lr={:.3e}".format(
                epoch + 1, opts.epoch_end, agent.optimizer.param_groups[0]["lr"], agent.optimizer.param_groups[1]["lr"]
            )
        )

        data, duration = generate_data(opts)
        if epoch == opts.epoch_start:
            print(f"generated {opts.epoch_size} instances (time={human_readable_time(duration)})")

        # training_dataset = tsp_problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size)
        training_dataset = _TSPDataset(data)
        # prepare training data
        training_dataloader = DataLoader(
            training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)
        pbar = tqdm.tqdm(
            total=(opts.K_epochs) * (opts.epoch_size // opts.batch_size) * (opts.T_train // opts.n_step),
            desc=f"Epoch {epoch + 1}/{opts.epoch_end} training",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            leave=False,
        )
        for batch_id, batch in enumerate(training_dataloader):
            train_batch(0, tsp_problem, agent, epoch, step, batch, tb_logger, opts, pbar)
            step += 1
        pbar.close()

        agent.lr_scheduler.step()

        # save new model after one epoch
        if not opts.no_saving and (
            (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.epoch_end - 1
        ):
            agent.save(epoch)

        # validate the new model
        validate(0, tsp_problem, agent, opts.val_dataset, tb_logger, _id=epoch)
