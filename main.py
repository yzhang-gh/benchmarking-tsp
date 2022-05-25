import numpy as np
import os
import time
import torch
from solvers.solver_options import get_dact_solver_options, get_nlkh_solver_options, get_pomo_solver_options

from solvers.solvers import DactSolver, NlkhSolver, PomoSolver
from utils.data_utils import generate_seed, load_dataset

Testset_Size = 500
Graph_Size = 100


def get_costs(problems, tours):
    # Check that tours are valid, i.e. contain 0 to n -1
    assert (
        torch.arange(tours.size(1), out=tours.data.new()).view(1, -1).expand_as(tours) == tours.data.sort(1)[0]
    ).all(), "Invalid tour"

    # Gather city coordinates in order of tour
    coords = problems.gather(1, tours.unsqueeze(-1).expand_as(problems))

    # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
    rolled_coords = coords.roll(-1, dims=1)
    tour_lens = torch.linalg.norm(coords - rolled_coords, dim=2).sum(1)
    return tour_lens


def info(text):
    return f"\033[94m{text}\033[0m"


def human_readable_time(seconds):
    if seconds < 60:
        return f"{seconds:5.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:5.2f}m"
    else:
        return f"{seconds / 3600:5.2f}h"


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = "data"
    dataset_id = f"rue{Graph_Size}_seed{generate_seed(Graph_Size, 'rue', 'test')}"
    testfile_name = os.path.join(data_dir, dataset_id)

    # EUC 2D TSP problems
    rue_problems = load_dataset(testfile_name)
    cluster_problems = None

    rue_problems = rue_problems[:Testset_Size]
    rue_problems = torch.tensor(rue_problems, dtype=torch.float)  # numpy ndarray's default dtype is double

    print(f"Loaded '{testfile_name}'. {Testset_Size=}")

    pomo_options_list = [
        get_pomo_solver_options(Graph_Size, "pretrained/pomo/saved_tsp100_model2_longTrain", 3100, 1),
        get_pomo_solver_options(Graph_Size, "pretrained/pomo/saved_tsp100_model2_longTrain", 3100, 8),
    ]

    dact_options_list = [
        get_dact_solver_options(Graph_Size, "pretrained/dact/tsp100-epoch-195.pt", 1, 1000),
        get_dact_solver_options(Graph_Size, "pretrained/dact/tsp100-epoch-195.pt", 1, 5000),
        get_dact_solver_options(Graph_Size, "pretrained/dact/tsp100-epoch-195.pt", 4, 5000),
    ]

    nlkh_options_list = [get_nlkh_solver_options("pretrained/nlkh/neurolkh.pt")]

    for solver_name, solver_class, opts_list in [
        ("POMO", PomoSolver, pomo_options_list),
        ("DACT", DactSolver, dact_options_list),
        ("NeuroLKH", NlkhSolver, nlkh_options_list),
    ]:

        print(f"== {solver_name} ==")

        for opts in opts_list:
            
            if solver_name == "POMO":
                print(f"data_augmentation={opts[2]['aug_factor']}")
            elif solver_name == "DACT":
                print(f"data_augmentation={opts['val_m']}, max_steps={opts['T_max']}")
            elif solver_name == "NeuroLKH":
                print(f"max_trials={opts['max_trials']}")

            solver = solver_class(opts)

            seeds = [1, 2, 3]
            if solver_name == "POMO":
                seeds = [1]

            for seed in seeds:

                np.random.seed(seed)
                torch.manual_seed(seed)

                t_start = time.time()

                tours, scores, time_fix = solver.solve(rue_problems, seed)

                t_end = time.time()
                duration = t_end - t_start - time_fix
                duration = human_readable_time(duration)

                tours = tours.to("cpu")
                costs = get_costs(rue_problems, tours)
                len = costs.mean().item()

                reported_len = -1
                if scores is not None:
                    scores = scores.to("cpu")
                    reported_len = scores.mean().item()

                print(info(f"{seed=}, {reported_len=:.6f}, {len=:.6f}, {duration=:.2f} (+{time_fix:.2f}s)"))
                # assert (torch.div((scores - costs), torch.minimum(scores, costs)).abs() < 2e-7).all()
