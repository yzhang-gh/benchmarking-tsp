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

    Testset_Size = 10000
    Graph_Size = 100
    Test_Distribution = "rue"
    Num_Runs = 10  # number of different random seeds

    data_dir = "../test_data"
    dataset_id = f"{Test_Distribution}{Graph_Size}"
    testfile_name = os.path.join(data_dir, dataset_id)

    result_dir = f"test_{datetime_str()}_n{Graph_Size}_{Test_Distribution}"
    print(f"will save to {result_dir}")

    # TSP problems
    problems = load_dataset(testfile_name)
    problems = problems[:Testset_Size]
    problems = torch.tensor(problems, dtype=torch.float)  # numpy ndarray's default dtype is double

    print(f"Loaded '{testfile_name}', {Testset_Size=}, {Num_Runs=} (num of different random seeds)")

    pomo_options_list = [
        # # rue 50
        get_pomo_solver_options(Graph_Size, "pretrained/pomo/n50_rue_downloaded", 1000, 1),
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/n50_rue_downloaded", 1000, 8),
        # # rue 100
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/n100_rue_longtrain_downloaded", 3100, 1),
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/n100_rue_longtrain_downloaded", 3100, 8),
        # # clust 50
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/n50_clust_20220613_213856", 1000, 1),
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/n50_clust_20220613_213856", 1000, 8),
        # # clust 100
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/n100_clust_20220613_214521", 900, 1),
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/n100_clust_20220613_214521", 900, 8),
        # mix 100
        # get_pomo_solver_options(Graph_Size, "pretrained/pomo/n100_mix_20220619_215128", 1766, 8),
    ]

    dact_options_list = [
        # # rue 50
        get_dact_solver_options(Graph_Size, "pretrained/dact/n50_rue_downloaded/epoch-198.pt", 1, T_max=5000, test_batch_size=10000),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n50_rue_downloaded/epoch-198.pt", 1, T_max=5000, test_batch_size=4096),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n50_rue_downloaded/epoch-198.pt", 4, T_max=5000, test_batch_size=4096),
        # # rue 100
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n100_rue_downloaded/epoch-195.pt", 1, T_max=5000, test_batch_size=4096),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n100_rue_downloaded/epoch-195.pt", 4, T_max=5000, test_batch_size=1024),
        # # clust 50
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n50_clust_20220608_232711/epoch-199.pt", 1, T_max=5000, test_batch_size=10000),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n50_clust_20220608_232711/epoch-199.pt", 1, T_max=5000, test_batch_size=4096),
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n50_clust_20220608_232711/epoch-199.pt", 4, T_max=5000, test_batch_size=4096),
        # # clust 100
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n100_clust_20220607_221925/epoch-110.pt", 1, T_max=5000, test_batch_size=4096),
        # mix 100
        # get_dact_solver_options(Graph_Size, "pretrained/dact/n100_mix_20220608_232923/epoch-110.pt", 1, T_max=5000, test_batch_size=4096),
    ]

    nlkh_options_list = [
        get_nlkh_solver_options("pretrained/nlkh/rue/neurolkh.finetuned.rue1000.pt", num_runs=1, max_trials=Graph_Size, batch_size=256, parallelism=32),
        # get_nlkh_solver_options("pretrained/nlkh/rue/neurolkh.pt", num_runs=1, max_trials=Graph_Size, batch_size=2500, parallelism=32),
        # get_nlkh_solver_options("pretrained/nlkh/mix/neurolkh_m.pt", num_runs=1, max_trials=Graph_Size, parallelism=32),
        # get_nlkh_solver_options("pretrained/nlkh/clust/clust-epoch-307.finetuned.clust1000.pt", num_runs=1, max_trials=Graph_Size, batch_size=256, parallelism=32),
        # get_nlkh_solver_options("pretrained/nlkh/clust/clust-epoch-307.pt", num_runs=1, max_trials=Graph_Size, batch_size=2500, parallelism=32),
    ]

    for solver_name, solver_class, opts_list in [
        ("POMO", PomoSolver, pomo_options_list),
        ("DACT", DactSolver, dact_options_list),
        ("NeuroLKH", NlkhSolver, nlkh_options_list),
    ]:

        print(bold(f"== {solver_name} =="))

        for i_opts, opts in enumerate(opts_list):

            if solver_name == "POMO":
                opts_model = opts[2]['model_load']
                model_path = f"{opts_model['path']}/epoch-{opts_model['epoch']}.pt"
                print(f"data_augmentation={opts[2]['aug_factor']}, {model_path=}")
            elif solver_name == "DACT":
                model_path = opts.load_path
                print(f"data_augmentation={opts.val_m}, max_steps={opts.T_max}, {model_path=}")
            elif solver_name == "NeuroLKH":
                model_path = opts.model_path
                print(
                    f"num_runs={opts.num_runs}, max_trials={opts.max_trials}, {model_path=}, "
                    + f"batch_size={opts.batch_size}, parallelism={opts.parallelism}"
                )
            model_id = f"opts_{i_opts}_model_" + model_path.split("/")[-2]

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

                # integer length
                num_digits = 4
                lens = {}
                for i_tsp, (tour, float_len) in enumerate(zip(tours, costs.tolist())):
                    f = os.path.join(data_dir, dataset_id, f"{i_tsp:0{num_digits}d}.tsp")
                    coords = load_tsplib_file(f)
                    int_len = calc_tsp_int_length(coords, tour)
                    lens[f] = [float_len, int_len]

                run_name = f"{solver_name}/{model_id}/seed{seed}"
                save_dir = os.path.join(result_dir, run_name)
                os.makedirs(save_dir, exist_ok=True)

                # save test results
                with open(os.path.join(save_dir, "solver_opts.json"), "w") as w:
                    if type(opts) == DotDict:
                        opts = opts.todict()
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
                        f"{seed=}, {reported_len=:.6f}, {tour_len=:.6f}, gap={gaps.mean():.6%}, {duration=!s}"
                    )
                )

            gaps_multi_runs = np.stack(gaps_multi_runs)
            print(
                info(
                    f"gaps_multi_runs={gaps_multi_runs.mean():.6%}, std={gaps_multi_runs.std(axis=0).mean():.6%}, "
                    + f"avg duration={np.mean(duration_multi_runs):.4f}s"
                )
            )

            np.savetxt(
                os.path.join(result_dir, f"{solver_name}/{model_id}_multi_run_gaps.txt"),
                gaps_multi_runs,
            )
