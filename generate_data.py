import argparse
import os
import re
import subprocess

import numpy as np

from utils.data_utils import save_dataset
from utils.file_utils import load_tsplib_file


def generate_tsp_data(data_dir, dataset_size, graph_size, distribution="rue", seed=1234):
    print(f"Generating {distribution}{graph_size} ({dataset_size} instances)")

    if distribution == "rue":
        return np.random.uniform(size=(dataset_size, graph_size, 2))

    elif distribution == "clust":
        if graph_size in [50, 500]:
            min_num_clusts = max_num_clusts = 5
        elif graph_size in [100, 1000]:
            min_num_clusts = max_num_clusts = 10

        # assert (
        #     dataset_size % (max_num_clusts - min_num_clusts + 1) == 0
        # ), f"{min_num_clusts=}, {max_num_clusts=}, {dataset_size=}"

        # create nested folders to better organize tmp files
        tmp_data_dir = os.path.join(data_dir, f"{distribution}{graph_size}")
        tmp_data_dir = re.sub(r"(?!^)/", "_tmpfiles/", tmp_data_dir, count=1)
        os.makedirs(tmp_data_dir, exist_ok=True)

        data_size_per_clust = dataset_size // (max_num_clusts - min_num_clusts + 1)
        # Rscript call_netgen.R point_num clu.lower clu.upper num_per_clust seed data_dir
        cmd = (
            f"Rscript call_netgen.R {graph_size} {min_num_clusts} {max_num_clusts}"
            f" {data_size_per_clust} {seed} '{tmp_data_dir}'"
        )
        print(cmd)
        subprocess.run(cmd, shell=True)

        tsp_instances = []
        for i in range(dataset_size):
            filename = f"{distribution}{graph_size}_seed{seed}_{i}.tsp"
            tsp_instances.append(load_tsplib_file(os.path.join(tmp_data_dir, filename)))
        data = np.stack(tsp_instances)
        # rescale, from {1, 2, ..., 1000000} to [0, 1)
        data = (data - 1) / (1000000 - 1)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="data", help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument(
        "--data_distribution", type=str, default="all", help="Distributions to generate for problem, default 'all'."
    )
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset (default 10000)")
    parser.add_argument(
        "--graph_sizes",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Sizes of problem instances (default 50, 100)",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed (default 1234)")

    opts = parser.parse_args()

    if opts.data_distribution == "all":
        distributions = ["rue", "clust"]
    else:
        assert opts.data_distribution in ["rue", "clust", "clustered"], "Unknown data distribution"
        distributions = ["clust" if opts.data_distribution == "clustered" else opts.data_distribution]

    np.random.seed(opts.seed)

    data_dir = opts.data_dir
    dataset_size = opts.dataset_size
    seed = opts.seed
    print(f"{data_dir=}, {distributions}x{opts.graph_sizes} (graph size), {dataset_size=}, {seed=}")

    for distribution in distributions:
        for graph_size in opts.graph_sizes:

            dataset = generate_tsp_data(data_dir, opts.dataset_size, graph_size, distribution, seed)

            filename = f"{distribution}{graph_size}_seed{opts.seed}.pkl"
            save_dataset(dataset, os.path.join(data_dir, filename))
