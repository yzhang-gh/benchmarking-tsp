"""Largely modified from
https://github.com/yzhang-gh/NeuroLKH/blob/f09d06634d4ff840ee2650776d3de3f9e0f32200/tsplib_test.py
"""
import argparse
import json
import os
import re
import time
from multiprocessing import Pool
from subprocess import check_call

import numpy as np
import torch
import tqdm
from torch.autograd import Variable
from others import datetime_str, info

from solvers.nlkh.net.sgcn_model import SparseGCNModel


def write_para(tsp_filename, model_id, para_filename, optim, seed=1234):
    with open(para_filename, "w") as f:
        f.write(
            f"PROBLEM_FILE = {data_dir}/{tsp_filename}\n"
            + "RUNS = 1\n"
            + f"OPTIMUM = {optim}\n"
            + f"TIME_LIMIT = {Time_Limit}\n"
            + f"SEED = {seed}\n"
        )

        if model_id.startswith("lkh_"):  # lkh tuned
            with open("pretrained/lkh/" + model_id.replace("lkh_", "") + ".para") as r:
                f.write(r.read())
        else:
            f.write("MOVE_TYPE = 5\n" + "PATCHING_C = 3\n" + "PATCHING_A = 2\n")

        if "nlkh" in model_id:
            f.write(f"CANDIDATE_FILE = {out_dir}/candidate/{model_id}/{tsp_filename}.txt\n")


def infer_SGN_write_candidate(net, tsp_problem_files, model_id):
    sgn_runtimes = {}
    for tsp_problem_file in tsp_problem_files:
        t1 = time.time()

        with open(f"{data_dir}/{tsp_problem_file}", "r") as f:
            lines = f.readlines()
            # if not ("EDGE_WEIGHT_TYPE : EUC_2D\n" in lines or "EDGE_WEIGHT_TYPE: EUC_2D\n" in lines):
            #     print(info(tsp_problem_file), "not EUC_2D")

            n_nodes = None
            coord_line_start_index = None
            for i_l, l in enumerate(lines):
                if l.startswith("DIMENSION"):
                    n_nodes = int(l.split(" ")[-1].strip())
                if l.startswith("NODE_COORD_SECTION"):
                    coord_line_start_index = i_l + 1

            x = []
            for i in range(n_nodes):
                line = [float(_) for _ in lines[coord_line_start_index + i].strip().split()]
                assert len(line) == 3
                assert line[0] == i + 1
                x.append([line[1], line[2]])
            x = np.array(x)
            scale = max(x[:, 0].max() - x[:, 0].min(), x[:, 1].max() - x[:, 1].min()) * (1 + 2 * 1e-4)
            x = x - x.min(0).reshape(1, 2)
            x = x / scale
            x = x + 1e-4
            if x[:, 0].max() > x[:, 1].max():
                x[:, 1] += (1 - 1e-4 - x[:, 1].max()) / 2
            else:
                x[:, 0] += (1 - 1e-4 - x[:, 0].max()) / 2
            x = x.reshape(1, n_nodes, 2)
        n_edges = 20
        batch_size = 1
        node_feat = x
        dist = x.reshape(batch_size, n_nodes, 1, 2) - x.reshape(batch_size, 1, n_nodes, 2)
        dist = np.sqrt((dist**2).sum(-1))
        edge_index = np.argsort(dist, -1)[:, :, 1 : 1 + n_edges]
        edge_feat = dist[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
        inverse_edge_index = -np.ones(shape=[batch_size, n_nodes, n_nodes], dtype="int")
        inverse_edge_index[
            np.arange(batch_size).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)
        ] = (np.arange(n_edges).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_edges)
        inverse_edge_index = inverse_edge_index[
            np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index
        ]
        edge_index_np = edge_index

        node_feat = Variable(
            torch.FloatTensor(node_feat).type(torch.cuda.FloatTensor), requires_grad=False
        )  # B x N x 2
        edge_feat = Variable(torch.FloatTensor(edge_feat).type(torch.cuda.FloatTensor), requires_grad=False).view(
            batch_size, -1, 1
        )  # B x 20N x 1
        edge_index = Variable(torch.LongTensor(edge_index).type(torch.cuda.LongTensor), requires_grad=False).view(
            batch_size, -1
        )  # B x 20N
        inverse_edge_index = Variable(
            torch.FloatTensor(inverse_edge_index).type(torch.cuda.LongTensor), requires_grad=False
        ).view(
            batch_size, -1
        )  # B x 20N
        candidate_test = []
        label = None
        edge_cw = None

        y_edges, _, y_nodes = net.forward(node_feat, edge_feat, edge_index, inverse_edge_index, label, edge_cw, n_edges)
        y_edges = y_edges.detach().cpu().numpy()
        y_edges = y_edges[:, :, 1].reshape(batch_size, n_nodes, n_edges)
        y_edges = np.argsort(-y_edges, -1)
        edge_index = np.array(edge_index_np)
        candidate_index = edge_index[
            np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), y_edges
        ]
        candidate_test.append(candidate_index[:, :, :5])
        candidate_test = np.concatenate(candidate_test, 0)

        candidate_file = os.path.join(out_dir, f"candidate/{model_id}/{tsp_problem_file}.txt")
        os.makedirs(os.path.dirname(candidate_file), exist_ok=True)
        with open(candidate_file, "w") as f:
            f.write(str(n_nodes) + "\n")
            for j in range(n_nodes):
                line = str(j + 1) + " 0 5"
                for _ in range(5):
                    line += " " + str(candidate_test[0, j, _] + 1) + " 1"
                f.write(line + "\n")
            f.write("-1\nEOF\n")

        t2 = time.time()
        sgn_runtimes[tsp_problem_file] = t2 - t1

    return sgn_runtimes


Run_Result_Regex = re.compile(r"Run \d+: Cost = (\d+),.*")


def read_results(log_filename):
    """returns (num_of_successes, avg_gap, min_tour_len, avg_tour_len, min_trials, avg_trials, lkh_time)"""
    with open(log_filename, "r") as f:
        lines = f.readlines()
        successes = int(lines[-7].split(" ")[-2].split("/")[0])
        cost_min = float(lines[-6].split(",")[0].split(" ")[-1])
        cost_avg = float(lines[-6].split(",")[1].split(" ")[-1])
        trials_min = float(lines[-4].split(",")[0].split(" ")[-1])
        trials_avg = float(lines[-4].split(",")[1].split(" ")[-1])
        time = float(lines[-3].split(",")[1].split(" ")[-2])

        optim = 0
        gaps = []
        for l in lines:
            if l.startswith("OPTIMUM = "):
                optim = int(l.split("=")[-1].strip())
            ## it is okay as the `OPTIMUM` line is always before the `Run` line
            if matches := Run_Result_Regex.match(l):
                tour_len = int(matches.group(1))
                gaps.append((tour_len - optim) / optim)

        if len(gaps) == 0:
            print(info(log_filename), "0 run")
            avg_gap = -0.01
        else:
            avg_gap = sum(gaps) / len(gaps)

        return successes, avg_gap, cost_min, cost_avg, trials_min, trials_avg, time


def evaluate(args):
    """returns (model_id, tsp_problem_file, num_of_successes, avg_gap, min_tour_len, avg_tour_len, min_trials, avg_trials, lkh_time)"""
    return _evaluate(*args)


def _evaluate(model_id, tsp_problem_file, optim, seed):
    """returns (model_id, tsp_problem_file, num_of_successes, avg_gap, min_tour_len, avg_tour_len, min_trials, avg_trials, lkh_time)"""
    para_filename = os.path.join(out_dir, f"{model_id}/para/{tsp_problem_file}.{seed}.para")
    log_filename = os.path.join(out_dir, f"{model_id}/log/{tsp_problem_file}.{seed}.log")
    os.makedirs(os.path.dirname(para_filename), exist_ok=True)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    write_para(tsp_problem_file, model_id, para_filename, optim, seed)

    # t1 = time.time()
    with open(log_filename, "w") as f:
        check_call([LKH_Execuatable, para_filename], stdout=f)
    # t2 = time.time()

    print(f"{model_id:10}, {tsp_problem_file:20}, {seed=}")

    return model_id, tsp_problem_file, *read_results(log_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=30)
    opts = parser.parse_args()

    Time_Limit = 3600

    data_dir = "data_large"
    out_dir = f"test_{datetime_str()}_large"

    LKH_Execuatable = "solvers/nlkh/LKH"

    with open(os.path.join(data_dir, "testset_indices.txt")) as r:
        tsp_problem_files = r.read().strip().split("\n")
    tsp_problem_files = tsp_problem_files[: opts.n_samples]

    with open(os.path.join(data_dir, "testset_optims.json")) as r:
        optims = json.loads(r.read())

    nlkh_models = {
        "nlkh_rue": "pretrained/nlkh/neurolkh.pt",
        "nlkh_mix": "pretrained/nlkh/neurolkh_m.pt",
        "nlkh_clust": "pretrained/nlkh/clust-epoch-307.pt",
    }

    model_ids = [*nlkh_models.keys(), "lkh", "lkh_rue", "lkh_mix", "lkh_clust"]
    print(f"{model_ids=}")

    ## write nlkh candidates
    nlkh_sgn_runtimes = {}
    first_time = True
    for model_id, model_path in nlkh_models.items():
        print(f"writing {model_id} generated candidates")

        net = SparseGCNModel()
        net.load_state_dict(torch.load(model_path)["model"])
        net.cuda()

        with torch.no_grad():
            # the time may be inaccurate for the very first inference because the GPU is idle
            # shouldn't be a problem as the difference is slight
            if first_time:
                infer_SGN_write_candidate(net, tsp_problem_files, model_id)
                first_time = False
            sgn_runtimes = infer_SGN_write_candidate(net, tsp_problem_files, model_id)
        nlkh_sgn_runtimes[model_id] = sgn_runtimes

    args_list = [
        [model_id, tsp_problem_file, optims[tsp_problem_file]]
        for model_id in model_ids
        for tsp_problem_file in tsp_problem_files
    ]

    num_seeds = 10
    seeds = np.random.randint(1000000, size=num_seeds)
    seeds = sorted(seeds)  ## in order to see the progress
    print("seeds", seeds)
    args_list = [args + [seed] for seed in seeds for args in args_list]

    with Pool(66) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(evaluate, args_list),
                total=len(args_list),
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            )
        )

    with open(os.path.join(out_dir, "nlkh_sng_runtimes.json"), "w") as w:
        w.write(json.dumps(nlkh_sgn_runtimes))

    results = sorted(results, key=lambda x: tsp_problem_files.index(x[1]) * 10 + model_ids.index(x[0]))
    with open(os.path.join(out_dir, "results.csv"), "w") as w:
        w.write("\n".join([", ".join(map(str, res)) for res in results]))

    print(
        "{:10} {:9} {:10} {:10} {:10} {:10} {:10} {:8}".format(
            "Model", "Successes", "Gap", "Best", "Average", "Trials_Min", "Trials_Avg", "LKH_Time"
        )
    )
    current_problem = ""
    for results_grouped_by_seed in [results[n : n + num_seeds] for n in range(0, len(results), num_seeds)]:
        first_res = results_grouped_by_seed[0]
        assert all(e[0] == first_res[0] for e in results_grouped_by_seed)
        assert all(e[1] == first_res[1] for e in results_grouped_by_seed)
        res = zip(*results_grouped_by_seed)
        res = [
            func(e)
            for e, func in zip(
                res,
                [
                    lambda x: x[0],             ## model_id
                    lambda x: x[0],             ## tsp_problem_file
                    lambda x: sum(x),           ## num_successes
                    lambda x: sum(x) / len(x),  ## optim_gap
                    lambda x: min(x),           ## best_solution (min_tour_len)
                    lambda x: sum(x) / len(x),  ## avg_tour_len
                    lambda x: min(x),           ## min_trials
                    lambda x: sum(x) / len(x),  ## avg_trials
                    lambda x: sum(x) / len(x),  ## avg_lkh_time
                ],
            )
        ]

        if res[1] != current_problem:
            current_problem = res[1]
            print(f"== {current_problem} ==")
        print("{:10} {:>9d} {:10.6%} {:10.1f} {:10.1f} {:10.1f} {:10.1f} {:8.2f}".format(res[0], *res[2:]))
