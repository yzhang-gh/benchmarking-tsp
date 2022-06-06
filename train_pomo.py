import json
import math
import os
import time

import torch
import tqdm
import numpy as np
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.utils.tensorboard import SummaryWriter

from generate_data import generate_tsp_data
from others import datetime_str, human_readable_time
from solvers.pomo.TSP.POMO.TSPEnv import TSPEnv
from solvers.pomo.TSP.POMO.TSPModel import TSPModel
from solvers.pomo.utils.utils import AverageMeter

save_dir = "/data0/zhangyu/runs/pomo"

env_params = {
    "problem_size": 100,
    "pomo_size": 100,
}

model_params = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}

optimizer_params = {
    "optimizer": {"lr": 1e-4, "weight_decay": 1e-6},
    "scheduler": {
        "milestones": [
            3001
        ],
        "gamma": 0.1,
    },
}

trainer_params = {
    "use_cuda": True,
    "cuda_device_num": 0,
    "epochs": 3100,
    "train_episodes": 100000,
    "train_batch_size": 64,
    "logging": {
        "model_save_interval": 20,
        "img_save_interval": 100,
        "log_image_params_1": {"json_foldername": "log_image_style", "filename": "style_tsp_100.json"},
        "log_image_params_2": {"json_foldername": "log_image_style", "filename": "style_loss_1.json"},
    },
    "model_load": {
        "enable": False,  # enable loading pre-trained model
        # 'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.
    },
}


def _train_one_epoch(epoch, training_data, pbar=None):
    score_AM = AverageMeter()
    loss_AM = AverageMeter()

    num_data = len(training_data)
    num_seen_data = 0
    loop_cnt = 0
    while num_seen_data < num_data:

        remaining = num_data - num_seen_data
        batch_size = min(trainer_params["train_batch_size"], remaining)

        avg_score, avg_loss = _train_one_batch(training_data[num_seen_data : num_seen_data + batch_size])

        score_AM.update(avg_score, batch_size)
        loss_AM.update(avg_loss, batch_size)
        pbar.update(1)

        num_seen_data += batch_size

        # Log First 10 Batch, only at the first epoch
        if epoch == start_epoch:
            loop_cnt += 1
            if loop_cnt <= 10:
                pbar.write(
                    "Epoch {:d} [train] ({:5.1f}%) score={:.4f}, 'loss'={:.4f}".format(
                        epoch, 100.0 * num_seen_data / num_data, score_AM.avg, loss_AM.avg
                    )
                )

    scheduler.step()

    return score_AM.avg, loss_AM.avg


def _train_one_batch(batch_data):
    # Prep
    ###############################################
    model.train()
    env.load_problems(batch_data, device=device)
    reset_state, _reward, _done = env.reset()
    model.pre_forward(reset_state)

    prob_list = torch.zeros(size=(len(batch_data), env.pomo_size, 0), device=device)
    # shape: (batch, pomo, 0~problem)

    # POMO Rollout
    ###############################################
    state, reward, done = env.pre_step()
    while not done:
        selected, prob = model(state)
        # shape: (batch, pomo)
        state, reward, done = env.step(selected)
        prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

    # Loss
    ###############################################
    advantage = reward - reward.float().mean(dim=1, keepdims=True)
    # shape: (batch, pomo)
    log_prob = prob_list.log().sum(dim=2)
    # size = (batch, pomo)
    loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
    # shape: (batch, pomo)
    loss_mean = loss.mean()

    # Score
    ###############################################
    max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
    score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

    # Step & Return
    ###############################################
    model.zero_grad()
    loss_mean.backward()
    optimizer.step()
    return score_mean.item(), loss_mean.item()


def generate_data():
    data_seed = np.random.randint(1000000)
    t1 = time.time()
    data = generate_tsp_data(
        "pomo_tmpfiles",
        trainer_params["train_episodes"],
        env_params["problem_size"],
        distribution="clust",
        seed=data_seed,
        save=False,
        quiet=True,
    )
    t2 = time.time()
    return data, t2 - t1


if __name__ == "__main__":
    # cuda
    if trainer_params["use_cuda"]:
        cuda_device_num = trainer_params["cuda_device_num"]
        torch.cuda.set_device(cuda_device_num)
        device = torch.device("cuda", cuda_device_num)
    else:
        device = torch.device("cpu")

    seed = 1234
    trainer_params["seed"] = seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Main Components
    env = TSPEnv(device, **env_params)
    model = TSPModel(device, **model_params)
    optimizer = Optimizer(model.parameters(), **optimizer_params["optimizer"])
    scheduler = Scheduler(optimizer, **optimizer_params["scheduler"])

    run_name = f"n{env_params['problem_size']}_{datetime_str()}"
    save_dir = os.path.join(save_dir, run_name)
    print("saved to", save_dir)
    writer = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "opts.json"), "w") as w:
        w.write(json.dumps([env_params, model_params, optimizer_params, trainer_params], indent=4))

    # Restore
    start_epoch = 1
    model_load = trainer_params["model_load"]
    if model_load["enable"]:
        checkpoint_fullname = "{path}/epoch-{epoch}.pt".format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"loaded {checkpoint_fullname}")

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        start_epoch = model_load["epoch"] + 1
        scheduler.last_epoch = model_load["epoch"] - 1

        # resume random state for date generation
        # FIXME would be better to save and load random state
        for i in range(start_epoch):
            _data_seed = np.random.randint(1000000)

    model.to(device)

    end_epoch = trainer_params["epochs"]
    for epoch in range(start_epoch, end_epoch + 1):
        pbar = tqdm.tqdm(
            total=math.ceil(trainer_params["train_episodes"] // trainer_params["train_batch_size"]),
            desc=f"Epoch {epoch}/{end_epoch} [data generation]",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            leave=False,
        )

        data, duration = generate_data()
        if epoch == start_epoch:
            pbar.write(f"generated {trainer_params['train_episodes']} instances (time={human_readable_time(duration)})")

        data = torch.tensor(data, dtype=torch.float)  # numpy ndarray's default dtype is double

        pbar.set_description(f"Epoch {epoch}/{end_epoch} [train]")
        t1 = time.time()

        score, loss = _train_one_epoch(epoch, data, pbar)

        t2 = time.time()
        pbar.close()
        duration = human_readable_time(t2 - t1)
        print(f"Epoch {epoch}/{end_epoch} [train] {score=:.4f}, 'loss'={loss:.4f}, duration={duration}")

        writer.add_scalar("train_score", score, epoch)
        writer.add_scalar("train_loss", loss, epoch)

        all_done = epoch == trainer_params["epochs"]
        model_save_interval = trainer_params["logging"]["model_save_interval"]

        if all_done or (epoch % model_save_interval) == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint_dict, f"{save_dir}/epoch-{epoch}.pt")
