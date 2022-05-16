import argparse
import os

import numpy as np

from utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size, distribution="rue"):
    if distribution == "rue":
        return np.random.uniform(size=(dataset_size, tsp_size, 2))
    elif distribution == "clustered":
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default="data", help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument(
        "--data_distribution", type=str, default="all", help="Distributions to generate for problem, default 'all'."
    )

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument(
        "--graph_sizes",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Sizes of problem instances (default 50, 100)",
    )
    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (
        len(opts.problems) == 1 and len(opts.graph_sizes) == 1
    ), "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        "tsp": ["rue", "clustered"],
    }
    if opts.data_distribution == "all":
        distributions = ["rue", "clustered"]
    else:
        assert opts.data_distribution in ["rue", "clustered"], "Unknown data distribution"
        distributions = [opts.data_distribution]

    problem = "tsp"
    for distribution in distributions:
        for graph_size in opts.graph_sizes:

            datadir = os.path.join(opts.data_dir, problem)
            os.makedirs(datadir, exist_ok=True)

            if opts.filename is None:
                filename = os.path.join(
                    datadir,
                    opts.name,
                    f"{distribution}{graph_size}_seed{opts.seed}.pkl",
                )
            else:
                filename = check_extension(opts.filename)

            assert opts.f or not os.path.isfile(
                check_extension(filename)
            ), "File already exists! Try running with -f option to overwrite."

            np.random.seed(opts.seed)
            dataset = generate_tsp_data(opts.dataset_size, graph_size, distribution)

            print(dataset[0])

            save_dataset(dataset, filename)
