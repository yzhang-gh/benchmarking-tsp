import json
import os
import time

import numpy as np
import torch

from others import DotDict, bold, human_readable_time, info
from solvers.solver_options import get_dact_solver_options, get_nlkh_solver_options, get_pomo_solver_options
from solvers.solvers import DactSolver, NlkhSolver, PomoSolver
from utils.data_utils import load_dataset

Testset_Size = 100
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


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = "../test_data"
    distribution = "clust"
    dataset_id = f"{distribution}{Graph_Size}"
    testfile_name = os.path.join(data_dir, dataset_id)

    result_dir = "test_results"

    # EUC 2D TSP problems
    problems = load_dataset(testfile_name)
    problems = problems[:Testset_Size]
    problems = torch.tensor(problems, dtype=torch.float)  # numpy ndarray's default dtype is double

    print(f"Loaded '{testfile_name}'. {Testset_Size=}")

    pomo_options_list = [
        get_pomo_solver_options(Graph_Size, "/data0/zhangyu/runs/pomo/n50_clust_20220608_231808", 800, 1),
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/saved_tsp100_model2_longTrain", 3100, 8),
    ]

    dact_options_list = [
        get_dact_solver_options(Graph_Size, "/data0/zhangyu/runs/dact/n50_clust_20220608_232711/epoch-40.pt", 1, 1500),
        # get_dact_solver_options(Graph_Size, "/data0/zhangyu/runs/dact/n50_clust_20220608_232711/epoch-50.pt", 1, 1500),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/tsp100-epoch-195.pt", 1, 5000),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/tsp100-epoch-195.pt", 4, 5000),
    ]

    nlkh_options_list = [get_nlkh_solver_options("pretrained/nlkh/neurolkh.pt", 100)]

    for solver_name, solver_class, opts_list in [
        ("POMO", PomoSolver, pomo_options_list),
        ("DACT", DactSolver, dact_options_list),
        ("NeuroLKH", NlkhSolver, nlkh_options_list),
    ]:

        print(bold(f"== {solver_name} =="))

        for opts in opts_list:

            if solver_name == "POMO":
                print(f"data_augmentation={opts[2]['aug_factor']}")
            elif solver_name == "DACT":
                print(f"data_augmentation={opts['val_m']}, max_steps={opts['T_max']}")
            elif solver_name == "NeuroLKH":
                print(f"max_trials={opts['max_trials']}")

            solver = solver_class(opts)

            seeds = np.random.randint(1000000, size=10)
            if solver_name == "POMO":
                seeds = seeds[:1]

            for seed in seeds:

                np.random.seed(seed)
                torch.manual_seed(seed)

                t_start = time.time()

                tours, scores = solver.solve(problems, seed)

                t_end = time.time()
                duration = t_end - t_start
                duration = human_readable_time(duration)

                tours = tours.to("cpu")
                costs = get_costs(problems, tours)
                len = costs.mean().item()

                reported_len = -1
                if scores is not None:
                    scores = scores.to("cpu")
                    reported_len = scores.mean().item()

                print(info(f"{seed=}, {reported_len=:.6f}, {len=:.6f}, {duration=!s}"))
                # assert (torch.div((scores - costs), torch.minimum(scores, costs)).abs() < 2e-7).all()

                run_name = f"{solver_name}/{dataset_id}/seed{seed}"
                save_dir = os.path.join(result_dir, run_name)
                os.makedirs(save_dir, exist_ok=True)

                # save test results
                with open(os.path.join(save_dir, "solver_opts.json"), "w") as w:
                    if type(opts) == DotDict:
                        opts = {k: v for k, v in opts.items()}
                    w.write(json.dumps(opts, indent=4))
                np.savetxt(os.path.join(save_dir, "tours.txt"), tours.numpy(), fmt="%d")
                with open(os.path.join(save_dir, "out.json"), "w") as w:
                    w.write(
                        json.dumps(
                            {"test_size": Testset_Size, "testfile_name": testfile_name, "duration": t_end - t_start},
                            indent=4,
                        )
                    )
