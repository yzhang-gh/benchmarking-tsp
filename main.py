import json
import os
import time

import numpy as np
import torch

from others import DotDict, bold, datetime_str, human_readable_time, info
from solvers.solver_options import get_dact_solver_options, get_nlkh_solver_options, get_pomo_solver_options
from solvers.solvers import DactSolver, NlkhSolver, PomoSolver
from utils.data_utils import load_dataset
from utils.file_utils import load_tsplib_file

Testset_Size = 10000
Graph_Size = 100
Test_Distribution = "rue"
Num_Runs = 10


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


def calc_tsp_int_length(int_coords, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    assert len(tour) == len(int_coords)
    sorted_locs = np.array(int_coords)[np.concatenate((tour, [tour[0]]))]
    return int(np.round(np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1)).sum())


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = "../test_data"
    dataset_id = f"{Test_Distribution}{Graph_Size}"
    testfile_name = os.path.join(data_dir, dataset_id)

    result_dir = f"test_{datetime_str()}_n{Graph_Size}_{Test_Distribution}"
    print(f"will save to {result_dir}")

    # EUC 2D TSP problems
    problems = load_dataset(testfile_name)
    problems = problems[:Testset_Size]
    problems = torch.tensor(problems, dtype=torch.float)  # numpy ndarray's default dtype is double

    print(f"Loaded '{testfile_name}'. {Testset_Size=}")

    pomo_options_list = [
        get_pomo_solver_options(Graph_Size, "/data0/zhangyu/runs/pomo/n50_clust_20220608_231808", 800, 1),
        get_pomo_solver_options(Graph_Size, "/data0/zhangyu/runs/pomo/n50_clust_20220608_231808", 800, 8),
    ]

    dact_options_list = [
        get_dact_solver_options(Graph_Size, "/data0/zhangyu/runs/dact/n50_clust_20220608_232711/epoch-40.pt", 1, 1500),
        # get_dact_solver_options(Graph_Size, "/data0/zhangyu/runs/dact/n50_clust_20220608_232711/epoch-50.pt", 1, 1500),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/tsp100-epoch-195.pt", 1, 5000),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/tsp100-epoch-195.pt", 4, 5000),
    ]

    nlkh_options_list = [
        # get_nlkh_solver_options("pretrained/nlkh/neurolkh.pt", num_runs=1, max_trials=Graph_Size, parallelism=32),
        # get_nlkh_solver_options("pretrained/nlkh/neurolkh_m.pt", num_runs=1, max_trials=Graph_Size, parallelism=32),
        get_nlkh_solver_options("pretrained/nlkh/clust-epoch-307.pt", num_runs=1, max_trials=Graph_Size, parallelism=32),
    ]

    for solver_name, solver_class, opts_list in [
        ("POMO", PomoSolver, pomo_options_list),
        ("DACT", DactSolver, dact_options_list),
        ("NeuroLKH", NlkhSolver, nlkh_options_list),
    ]:

        print(bold(f"== {solver_name} =="))

        for i_opts, opts in enumerate(opts_list):

            if solver_name == "POMO":
                print(f"data_augmentation={opts[2]['aug_factor']}, model={opts[2]['model_load']['path']}")
            elif solver_name == "DACT":
                print(f"data_augmentation={opts.val_m}, max_steps={opts.T_max}, model={opts.load_path}")
            elif solver_name == "NeuroLKH":
                print(f"num_runs={opts.num_runs}, max_trials={opts.max_trials}, model={opts.model_path}, parallelism={opts.parallelism}")

            solver = solver_class(opts)

            seeds = np.random.randint(1000000, size=Num_Runs).tolist()
            if solver_name == "POMO":
                seeds = seeds[:1]

            gaps_multi_runs = []
            duration_multi_runs = []
            for seed in seeds:

                np.random.seed(seed)
                torch.manual_seed(seed)

                t_start = time.time()

                tours, scores = solver.solve(problems, seed)

                t_end = time.time()
                duration = t_end - t_start
                duration_multi_runs.append(duration)
                duration = human_readable_time(duration)

                tours = tours.to("cpu")
                costs = get_costs(problems, tours)
                tour_len = costs.mean().item()

                reported_len = -1
                if scores is not None:
                    scores = scores.to("cpu")
                    reported_len = scores.mean().item()

                # print(info(f"{seed=}, {reported_len=:.6f}, {tour_len=:.6f}, {duration=!s}"))
                # assert (torch.div((scores - costs), torch.minimum(scores, costs)).abs() < 2e-7).all()

                # integer length
                num_digits = len(str(len(problems) - 1))
                lens = {}
                for i_tsp, (tour, float_len) in enumerate(zip(tours, costs.tolist())):
                    f = os.path.join(data_dir, dataset_id, f"{i_tsp:0{num_digits}d}.tsp")
                    coords = load_tsplib_file(f)
                    int_len = calc_tsp_int_length(coords, tour)
                    lens[f] = [float_len, int_len]

                run_name = f"{solver_name}/{dataset_id}_opts_{i_opts}/seed{seed}"
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
                with open(os.path.join(save_dir, "tour_lens.json"), "w") as w:
                    w.write(json.dumps(lens, indent=4))

                # optimity gap
                data_type = "cluster" if Test_Distribution == "clust" else "rue"
                optim_sol_file = f"../optim_sol/testing_set_optimum_{data_type}_{Graph_Size}.json"
                with open(optim_sol_file) as r:
                    optim_sol = json.loads(r.read())
                gaps = []
                for (k1, v1), (k2, v2) in zip(optim_sol.items(), lens.items()):
                    assert k1.split("/")[-1] == k2.split("/")[-1]
                    v1 = int(v1)
                    assert v1 <= v2[1]
                    gap = (v2[1] - v1) / v1
                    gaps.append(gap)
                gaps = np.array(gaps)
                gaps_multi_runs.append(gaps)

                print(
                    info(
                        f"{seed=}, {reported_len=:.6f}, {tour_len=:.6f}, gap={gaps.mean():.5%}, {duration=!s}"
                    )
                )

            gaps_multi_runs = np.stack(gaps_multi_runs)
            print(
                info(
                    f"gaps_multi_runs={gaps.mean():.5%}, std={gaps_multi_runs.std(axis=0).mean():.5%}, "
                    + f"avg duration={np.mean(duration_multi_runs):.4f}s"
                )
            )

            np.savetxt(
                os.path.join(result_dir, f"{solver_name}/{dataset_id}_opts_{i_opts}_multi_run_gaps.txt"),
                gaps_multi_runs,
            )
