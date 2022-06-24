import argparse
import os
import subprocess
import time

import numpy as np

from utils.data_utils import downscale_tsp_coords, generate_seed, save_dataset, upscale_tsp_coords
from utils.file_utils import load_tsplib_file


def generate_tsp_data(
    data_dir,
    dataset_size,
    graph_size,
    distribution="rue",
    mode="train",
    seed=None,
    num_clusts=None,
    save=True,
    quiet=False,
):
    """
    mode: This is used to generate the default random seed if seed is not provided.
    """
    if not seed:
        seed = generate_seed(graph_size, distribution, mode)

    np.random.seed(seed)

    if not quiet:
        print(f"Generating {distribution}{graph_size} ({dataset_size} instances), {seed=}")

    dataset_id = f"{distribution}{graph_size}_seed{seed}"

    if distribution == "rue":
        dataset = np.random.uniform(size=(dataset_size, graph_size, 2))
        ## for consistency, upscale to TSPLib range {1, 2, ..., 1000000} first, will normalize later
        dataset = upscale_tsp_coords(dataset)

    elif distribution == "clust":
        if not num_clusts:
            if graph_size in [50, 500]:
                min_num_clusts = max_num_clusts = 5
            elif graph_size in [100, 1000]:
                min_num_clusts = max_num_clusts = 10
            else:
                raise NotImplementedError()
        else:
            min_num_clusts = max_num_clusts = num_clusts

        # create subfolders named `dataset_id` to store `.tsp` and other tmp files
        os.makedirs(os.path.join(data_dir, dataset_id), exist_ok=True)

        data_size_per_clust = dataset_size  # // (max_num_clusts - min_num_clusts + 1)
        # Rscript call_netgen.R point_num clu.lower clu.upper num_per_clust seed data_dir
        cmd = (
            f"Rscript call_netgen.R {graph_size} {min_num_clusts} {max_num_clusts}"
            f" {data_size_per_clust} {seed} '{os.path.join(data_dir, dataset_id)}'"
        )
        if not quiet:
            print("  " + cmd)
        subprocess.run(cmd, shell=True, check=True)

        num_digits = len(str(dataset_size - 1))
        tsp_instances = []
        for i in range(dataset_size):
            tsp_instances.append(load_tsplib_file(os.path.join(data_dir, dataset_id, f"{i:0{num_digits}d}.tsp")))
        dataset = np.stack(tsp_instances)

    ## rescale, from {1, 2, ..., 1000000} to [0, 1)
    dataset = downscale_tsp_coords(dataset)

    if save:
        save_dataset(dataset, os.path.join(data_dir, dataset_id))
        print(f"Saved to {os.path.join(data_dir, dataset_id)}.pkl")

    return dataset


def generate_data_jit(tmp_data_dir, dataset_size, graph_size, distribution="rue", seed=None, num_clusts=None):
    """wrapper function for `generate_tsp_data` with `save=False` and `quiet=True` by default"""
    assert distribution in ["rue", "clust", "mix"]
    t1 = time.time()

    if distribution == "mix":
        assert dataset_size % 2 == 0
        data1 = generate_tsp_data(
            tmp_data_dir,
            dataset_size // 2,
            graph_size,
            distribution="rue",
            seed=seed,
            save=False,
            quiet=True,
        )
        data2 = generate_tsp_data(
            tmp_data_dir,
            dataset_size // 2,
            graph_size,
            distribution="clust",
            seed=seed,
            num_clusts=num_clusts,
            save=False,
            quiet=True,
        )
        data = np.vstack([data1, data2])
        np.random.shuffle(data)

    else:
        data = generate_tsp_data(
            tmp_data_dir,
            dataset_size,
            graph_size,
            distribution=distribution,
            seed=seed,
            num_clusts=num_clusts,
            save=False,
            quiet=True,
        )

    t2 = time.time()
    return data, t2 - t1


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
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="train",
        help="Dataset type (default 'train'). This is used to generate the default random seed if seed is not provided.",
    )
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed (default None)")

    opts = parser.parse_args()

    if opts.data_distribution == "all":
        distributions = ["rue", "clust"]
    else:
        assert opts.data_distribution in ["rue", "clust", "clustered"], "Unknown data distribution"
        distributions = ["clust" if opts.data_distribution == "clustered" else opts.data_distribution]

    data_dir = opts.data_dir
    dataset_size = opts.dataset_size
    seed = opts.seed
    print(f"{data_dir=}, {distributions}x{opts.graph_sizes} (graph size), {dataset_size=}, {seed=}")

    mode = opts.type

    for distribution in distributions:
        for graph_size in opts.graph_sizes:
            generate_tsp_data(data_dir, opts.dataset_size, graph_size, distribution, mode, seed)
