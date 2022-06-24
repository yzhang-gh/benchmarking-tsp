"""Modified from
https://github.com/yzhang-gh/NeuroLKH/blob/f09d06634d4ff840ee2650776d3de3f9e0f32200/tsplib_test.py
"""
import argparse
import os
import pickle
import time
from multiprocessing import Pool
from subprocess import check_call

import numpy as np
import torch
import tqdm
from torch.autograd import Variable

from solvers.nlkh.net.sgcn_model import SparseGCNModel


def write_para(instance_name, method, para_filename, opt_value):
    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = tsplib_data/" + instance_name + ".tsp\n")
        f.write("MOVE_TYPE = 5\nPATCHING_C = 3\nPATCHING_A = 2\nRUNS = 10\n")
        f.write("OPTIMUM = " + str(opt_value) + "\n")
        if method == "NeuroLKH":
            f.write("SEED = 1234\n")
            f.write(f"CANDIDATE_FILE = {out_dir}/candidate/{instance_name}.txt\n")
        else:
            assert method == "LKH"


def read_feat(feat_filename):
    edge_index = []
    edge_feat = []
    inverse_edge_index = []
    with open(feat_filename, "r") as f:
        lines = f.readlines()
        for line in lines[:-1]:
            line = line.strip().split()
            for i in range(20):
                edge_index.append(int(line[i * 3]))
                edge_feat.append(int(line[i * 3 + 1]) / 1000000)
                inverse_edge_index.append(int(line[i * 3 + 2]))
    edge_index = np.array(edge_index).reshape(1, -1, 20)
    edge_feat = np.array(edge_feat).reshape(1, -1, 20)
    inverse_edge_index = np.array(inverse_edge_index).reshape(1, -1, 20)
    runtime = float(lines[-1].strip())
    return edge_index, edge_feat, inverse_edge_index, runtime


def write_candidate_pi(dataset_name, instance_name, candidate, pi):
    n_node = candidate.shape[0]
    with open("result/" + dataset_name + "/candidate/" + instance_name + ".txt", "w") as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " 0 5"
            for _ in range(5):
                line += " " + str(int(candidate[j, _]) + 1) + " " + str(_ * 100)
            f.write(line + "\n")
        f.write("-1\nEOF\n")
    with open("result/" + dataset_name + "/pi/" + instance_name + ".txt", "w") as f:
        f.write(str(n_node) + "\n")
        for j in range(n_node):
            line = str(j + 1) + " " + str(int(pi[j]))
            f.write(line + "\n")
        f.write("-1\nEOF\n")


def method_wrapper(args):
    if args[0] == "LKH":
        return solve_LKH(*args[1:])
    elif args[0] == "NeuroLKH":
        return solve_NeuroLKH(*args[1:])
    else:
        raise NotImplementedError


def solve_LKH(instance_name, opt_value):
    para_filename = os.path.join(out_dir, f"LKH_para/{instance_name}.para")
    log_filename = os.path.join(out_dir, f"LKH_log/{instance_name}.log")
    write_para(instance_name, "LKH", para_filename, opt_value)
    with open(log_filename, "w") as f:
        check_call([LKH_Execuatable, para_filename], stdout=f)
    return read_results(log_filename)


def solve_NeuroLKH(instance_name, opt_value):
    para_filename = os.path.join(out_dir, f"NeuroLKH_para/{instance_name}.para")
    log_filename = os.path.join(out_dir, f"NeuroLKH_log/{instance_name}.log")
    write_para(instance_name, "NeuroLKH", para_filename, opt_value)
    with open(log_filename, "w") as f:
        check_call([LKH_Execuatable, para_filename], stdout=f)
    return read_results(log_filename)


def infer_SGN_write_candidate(net, instance_names):
    for instance_name in instance_names:
        with open(f"tsplib_data/{instance_name}.tsp", "r") as f:
            lines = f.readlines()
            assert lines[4] == "EDGE_WEIGHT_TYPE : EUC_2D\n"
            assert lines[5] == "NODE_COORD_SECTION\n"
            n_nodes = int(lines[3].split(" ")[-1])
            x = []
            for i in range(n_nodes):
                line = [float(_) for _ in lines[6 + i].strip().split()]
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
        with open(os.path.join(out_dir, f"candidate/{instance_name}.txt"), "w") as f:
            f.write(str(n_nodes) + "\n")
            for j in range(n_nodes):
                line = str(j + 1) + " 0 5"
                for _ in range(5):
                    line += " " + str(candidate_test[0, j, _] + 1) + " 1"
                f.write(line + "\n")
            f.write("-1\nEOF\n")


def read_results(log_filename):
    results = []
    with open(log_filename, "r") as f:
        lines = f.readlines()
        successes = int(lines[-7].split(" ")[-2].split("/")[0])
        cost_min = float(lines[-6].split(",")[0].split(" ")[-1])
        cost_avg = float(lines[-6].split(",")[1].split(" ")[-1])
        trials_min = float(lines[-4].split(",")[0].split(" ")[-1])
        trials_avg = float(lines[-4].split(",")[1].split(" ")[-1])
        time = float(lines[-3].split(",")[1].split(" ")[-2])
        return successes, cost_min, cost_avg, trials_min, trials_avg, time


def eval_dataset(instance_names, method, opts, opt_values):
    os.makedirs(os.path.join(out_dir, method + "_para"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, method + "_log"), exist_ok=True)

    if method == "NeuroLKH":
        os.makedirs(os.path.join(out_dir, "candidate"), exist_ok=True)

        net = SparseGCNModel()
        net.cuda()
        saved = torch.load(opts.model_path)
        net.load_state_dict(saved["model"])
        sgn_start_time = time.time()
        with torch.no_grad():
            infer_SGN_write_candidate(net, instance_names)
        sgn_runtime = time.time() - sgn_start_time
        with Pool(os.cpu_count()) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(
                        method_wrapper,
                        [
                            ("NeuroLKH", instance_names[i], opt_values[instance_names[i]])
                            for i in range(len(instance_names))
                        ],
                    ),
                    total=len(instance_names),
                    desc="NeuroLKH",
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                )
            )
    else:
        assert method == "LKH"
        feat_runtime = 0
        sgn_runtime = 0
        with Pool(opts.parallelism) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(
                        method_wrapper,
                        [("LKH", instance_names[i], opt_values[instance_names[i]]) for i in range(len(instance_names))],
                    ),
                    total=len(instance_names),
                    desc="LKH",
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                )
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="pretrained/nlkh/neurolkh.pt")
    parser.add_argument("--n_samples", type=int, default=5)
    opts = parser.parse_args()

    opts.parallelism = 32
    out_dir = "test_results_tsplib"

    LKH_Execuatable = "solvers/nlkh/LKH"

    instance_names = [
        "kroB150",
        "rat195",
        "pr299",
        "d493",
        "rat575",
        "pr1002",
        "u1060",
        "vm1084",
        "pcb1173",
        "rl1304",
        "rl1323",
        "nrw1379",
        "fl1400",
        "fl1577",
        "vm1748",
        "u1817",
        "rl1889",
        "d2103",
        "u2152",
        "pcb3038",
        "fl3795",
        "fnl4461",
        "rl5915",
        "rl5934",
    ]
    instance_names = instance_names[: opts.n_samples]
    with open("tsplib_data/opt.pkl", "rb") as f:
        opt_values = pickle.load(f)
    lkh_results = eval_dataset(instance_names, "LKH", opts, opt_values)
    neurolkh_r_results = eval_dataset(instance_names, "NeuroLKH", opts, opt_values)

    opts.model_path = "pretrained/nlkh/neurolkh_m.pt"
    neurolkh_m_results = eval_dataset(instance_names, "NeuroLKH", opts, opt_values)
    print("Successes    Best Average Trials_Min Trials_Avg Time")
    for i in range(len(lkh_results)):
        print(f"----- {instance_names[i]} -----")
        print("{:>9d} {:7.1f} {:7.1f} {:10.1f} {:10.1f} {:4.1f}".format(*tuple(lkh_results[i])))
        print("{:>9d} {:7.1f} {:7.1f} {:10.1f} {:10.1f} {:4.1f}".format(*tuple(neurolkh_r_results[i])))
        print("{:>9d} {:7.1f} {:7.1f} {:10.1f} {:10.1f} {:4.1f}".format(*tuple(neurolkh_m_results[i])))
