import os
from abc import ABC, abstractmethod
from datetime import datetime
from multiprocessing import Pool
import time

import numpy as np
import torch
import tqdm
from problems.tsp.tsp_baseline import read_tsplib

from .dact.agent.ppo import PPO
from .dact.problems.problem_tsp import TSP
from .nlkh._swig_test import infer_SGN, method_wrapper
from .nlkh.net.sgcn_model import SparseGCNModel
from .pomo.TSP.POMO.TSPEnv import TSPEnv
from .pomo.TSP.POMO.TSPModel import TSPModel


class BaseSovler(ABC):
    @abstractmethod
    def solve(self, problems):
        pass


class PomoSolver(BaseSovler):
    """POMO"""

    def __init__(self, opts):
        # save arguments
        self.env_params, self.model_params, self.tester_params = opts

        # CUDA
        if self.tester_params["use_cuda"]:
            cuda_device_num = self.tester_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
        else:
            device = torch.device("cpu")
        self.device = device

        # ENV and MODEL
        self.env = TSPEnv(device, **self.env_params)
        self.model = TSPModel(device, **self.model_params)

        # Restore
        model_load = self.tester_params["model_load"]
        checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)

    def solve(self, problems, seed=None):
        num_problems = len(problems)
        num_solved = 0

        tours = []
        scores = []
        while num_solved < num_problems:

            remaining = num_problems - num_solved
            batch_size = min(self.tester_params["test_batch_size"], remaining)

            tours_batch, scores_batch, _, _ = self._solve_batch(problems[num_solved : num_solved + batch_size])
            # shape: (batch, tour_len), scalar
            tours.append(tours_batch)
            scores.append(scores_batch)

            num_solved += batch_size

        return torch.vstack(tours), torch.vstack(scores), 0

    def _solve_batch(self, problems):
        """see pomo/TSP/TSPTester.py:_test_one_batch"""

        ## Augmentation
        if self.tester_params["augmentation_enable"]:
            aug_factor = self.tester_params["aug_factor"]
        else:
            aug_factor = 1

        ## Ready
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(problems, aug_factor, self.device)
            reset_state, _reward, _done = self.env.reset()
            self.model.pre_forward(reset_state)

        ## POMO Rollout
        state, _reward, done = self.env.pre_step()
        while not done:
            selected, _prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        batch_size = problems.shape[0]
        num_cities = problems.shape[1]
        reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)
        tours = self.env.selected_node_list.reshape(aug_factor, batch_size, self.env.pomo_size, num_cities)
        # shape: (augmentation, batch, pomo, tour_len)

        no_aug_reward = reward[0]
        # shape: (batch, pomo)
        max_pomo_reward, max_pomo_indices = no_aug_reward.max(dim=1)
        # shape: (batch,), (batch,)
        no_aug_scores = -max_pomo_reward  # negative sign to make positive value
        no_aug_tours = tours[0][torch.arange(batch_size), max_pomo_indices]  # NOTE advanced indexing
        # shape: (batch, tour_len)

        max_pomo_reward, max_pomo_indices = reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch), (augmentation, batch)
        aug_tours = tours[torch.arange(aug_factor).reshape(aug_factor, 1), torch.arange(batch_size), max_pomo_indices]
        # shape: (augmentation, batch, tour_len)
        max_aug_pomo_reward, max_aug_indices = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,), (batch,)
        aug_scores = -max_aug_pomo_reward  # negative sign to make positive value
        aug_tours = aug_tours[max_aug_indices, torch.arange(batch_size)]
        # shape: (batch, tour_len)

        return no_aug_tours, no_aug_scores, aug_tours, aug_scores


class DactSolver(BaseSovler):
    """DACT"""

    def __init__(self, opts):
        self.opts = opts

    def solve(self, problems, seed=1234):
        t1 = time.time()

        opts = self.opts
        opts.seed = seed

        # Figure out what's the problem
        self.tsp_problem = TSP(
            p_size=opts.graph_size,
            step_method=opts.step_method,
            init_val_met=opts.init_val_met,
            with_assert=opts.use_assert,
            P=opts.P,
        )
        self.agent = PPO(self.tsp_problem.NAME, self.tsp_problem.size, opts)

        if opts.load_path is not None:
            self.agent.load(opts.load_path)

        self.agent.eval()
        self.tsp_problem.eval()

        t2 = time.time()
        env_init_duration = t2 - t1

        batch = {"coordinates": problems}

        best_value, cost_hist, best_hist, r, rec_history, best_sol = self.agent.rollout(
            self.tsp_problem, opts.val_m, batch, do_sample=True, show_bar=True
        )

        return best_sol, best_value, env_init_duration


class NlkhSolver(BaseSovler):
    """NeuroLKH"""

    def __init__(self, opts):
        self.opts = opts

        net = SparseGCNModel()
        saved = torch.load(opts.model_path)
        net.load_state_dict(saved["model"])
        net.to(opts.device)
        self.net = net

    def solve(self, problems, seed=1234):
        opts = self.opts
        data_size = len(problems)
        graph_size = problems.shape[1]
        max_trials = opts.max_trials

        t1 = time.time()

        ## feature generation
        with Pool(opts.parallelism) as pool:
            feats = list(
                tqdm.tqdm(
                    pool.imap(method_wrapper, [("FeatGen", problem, graph_size) for problem in problems]),
                    desc="feature generation",
                    total=data_size,
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                )
            )
        feats = list(zip(*feats))
        edge_index, edge_feat, inverse_edge_index, feat_runtime = feats
        feat_runtime = np.sum(feat_runtime)
        edge_index = np.concatenate(edge_index)
        edge_feat = np.concatenate(edge_feat)
        inverse_edge_index = np.concatenate(inverse_edge_index)

        t2 = time.time()

        with torch.no_grad():
            candidate_Pi = infer_SGN(
                self.net, problems, edge_index, edge_feat, inverse_edge_index, batch_size=opts.batch_size
            )

        invec = np.concatenate([problems.reshape(data_size, -1) * 1000000, candidate_Pi[:data_size]], 1)

        if invec.shape[1] < max_trials * 2:
            invec = np.concatenate([invec, np.zeros([invec.shape[0], max_trials * 2 - invec.shape[1]])], 1)
        else:
            invec = invec.copy()

        ## call LKH with `invec`
        run_name = f"nlkh_tmpfiles/{datetime.now():%Y%m%d_%H%M%S}"
        os.makedirs(run_name, exist_ok=True)
        num_digits = len(str(data_size - 1))
        with Pool(opts.parallelism) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(
                        method_wrapper,
                        [
                            (
                                "NeuroLKH",
                                invec[i],
                                graph_size,
                                os.path.join(run_name, f"{i:0{num_digits}d}.tour"),
                                seed,
                                max_trials,
                            )
                            for i in range(data_size)
                        ],
                    ),
                    desc="call LKH",
                    total=data_size,
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                )
            )

        t3 = time.time()

        # results = np.array(results).reshape(data_size, -1, 2)[:, :max_trials, :]
        # dataset_objs = results[:, :, 0].mean(0)
        # dataset_runtimes = results[:, :, 1].sum(0)

        tours = [read_tsplib(os.path.join(run_name, f"{i:0{num_digits}d}.tour")) for i in range(data_size)]

        t4 = time.time()

        feat_duration = t2 - t1
        read_tour_duration = t4 - t3

        return torch.tensor(tours, dtype=torch.long), None, read_tour_duration
