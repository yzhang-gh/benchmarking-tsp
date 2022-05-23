import argparse
import os
from time import time

import numpy as np

from generate_data import generate_tsp_data
from solve import solve_dataset
from utils.data_utils import generate_seed, save_dataset

Num_Processes = os.cpu_count() - 3


def generate_data_and_extract_feat(data_dir, graph_size, mode="train", distributions=["rue", "clust"], seed=None):

    if mode == "val":
        n_samples = 1000
    elif mode == "train":
        n_samples = (20 * 200 // graph_size) * 125
    else:
        raise NotImplementedError(f"{mode=} not supported")

    n_neighbours = 20

    data_dir = os.path.join(data_dir, mode)
    print(f"{mode=}, will save data to {data_dir}")

    for distribution in distributions:
        if not seed:
            seed = generate_seed(graph_size, distribution, mode)

        if distribution == "clust":
            num_clusts = 7
        else:
            num_clusts = None

        # t_start = time()

        dataset = generate_tsp_data(data_dir, n_samples, graph_size, distribution, mode, seed, num_clusts=num_clusts)
        dataset_id = f"{distribution}{graph_size}_seed{seed}"
        results = solve_dataset(data_dir, dataset_id)
        optim_tours = [tour for _tour_len, tour, _time in results]

        dist = dataset.reshape(n_samples, graph_size, 1, 2) - dataset.reshape(n_samples, 1, graph_size, 2)
        dist = np.sqrt((dist**2).sum(-1))  # 10000 x 100 x 100
        edge_index = np.argsort(dist, -1)[:, :, 1 : 1 + n_neighbours]
        edge_feat = dist[np.arange(n_samples).reshape(-1, 1, 1), np.arange(graph_size).reshape(1, -1, 1), edge_index]

        optim_tours = np.array(optim_tours)  # n_samples x graph_size
        label = np.zeros(shape=[n_samples, graph_size, graph_size], dtype="bool")
        label[np.arange(n_samples).reshape(-1, 1), optim_tours, np.roll(optim_tours, 1, -1)] = True
        label[np.arange(n_samples).reshape(-1, 1), np.roll(optim_tours, 1, -1), optim_tours] = True
        label = label[np.arange(n_samples).reshape(-1, 1, 1), np.arange(graph_size).reshape(1, -1, 1), edge_index]

        inverse_edge_index = -np.ones(shape=[n_samples, graph_size, graph_size], dtype="int")
        inverse_edge_index[
            np.arange(n_samples).reshape(-1, 1, 1), edge_index, np.arange(graph_size).reshape(1, -1, 1)
        ] = (np.arange(n_neighbours).reshape(1, 1, -1) + np.arange(graph_size).reshape(1, -1, 1) * n_neighbours)
        inverse_edge_index = inverse_edge_index[
            np.arange(n_samples).reshape(-1, 1, 1), np.arange(graph_size).reshape(1, -1, 1), edge_index
        ]

        feat = {
            "node_feat": dataset,  # n_samples x graph_size x 2
            "edge_feat": edge_feat,  # n_samples x graph_size x n_neighbours
            "edge_index": edge_index,  # n_samples x graph_size x n_neighbours
            "label": label,  # n_samples x graph_size x n_neighbours
            "inverse_edge_index": inverse_edge_index,
        }

        # t_end = time()

        save_dataset(feat, os.path.join(data_dir, dataset_id + ".feat.pkl"))

        # if mode == "train":
        #     with open(f"walltime.txt", "a") as w:
        #         w.write(f"n={graph_size} {(t_end - t_start) / 60:.2f}m ({n_samples} instances)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="train", help="Generate training or validation datasets")
    cli_opts = parser.parse_args()

    dataset_type = cli_opts.type
    assert dataset_type in ["train", "val"]

    data_dir = "/data0/zhangyu/data_nlkh"
    distributions = ["clust"]
    train_graph_sizes = range(101, 501)
    val_graph_sizes = (100, 200, 500)
    seed = None

    graph_sizes = train_graph_sizes if dataset_type == "train" else val_graph_sizes
    arg_list = [
        (data_dir, graph_size, dataset_type, distributions, seed) for graph_size in graph_sizes
    ]

    print(f"Generating NeuroLKH data, {dataset_type=}, {distributions=}, {graph_sizes=}, {seed=}")

    for args in arg_list:
        generate_data_and_extract_feat(*args)
