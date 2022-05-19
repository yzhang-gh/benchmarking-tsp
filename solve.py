import os
import time
from multiprocessing import Pool

from tqdm import tqdm

from problems.tsp.tsp_baseline import solve_concorde_log
from utils.data_utils import load_dataset, save_dataset

Save_Int_Distance = False

num_cpus = os.cpu_count() - 4
executable = os.path.abspath("problems/tsp/concorde/concorde/TSP/concorde")


def run_func(args):
    return solve_concorde_log(executable, *args, disable_cache=True, int_distance=Save_Int_Distance)


def solve_dataset(data_dir, dataset_id, save_int_distance=False):
    assert save_int_distance == Save_Int_Distance

    dataset = load_dataset(f"{data_dir}/{dataset_id}")

    print(f"Solving {dataset_id}.pkl... {save_int_distance=}")

    num_digits = len(str(len(dataset) - 1))

    t_start = time.time()

    with Pool(num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(
                    run_func,
                    [
                        (os.path.join(data_dir, dataset_id), f"{i:0{num_digits}d}", data)
                        for i, data in enumerate(dataset)
                    ],
                ),
                total=len(dataset),
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
            )
        )

    t_end = time.time()
    print(f"Used {(t_end - t_start) / 60:.2f}m")

    failed = [str(i) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))

    save_dataset(results, os.path.join(data_dir, f"{dataset_id}.sol{'.int' if save_int_distance else ''}.pkl"))

    return results


if __name__ == "__main__":

    for distribution in ["rue"]:#, "clust"
        graph_size = 100
        seed = 1024
        data_dir = "data"
        dataset_id = f"{distribution}{graph_size}_seed{seed}"

        solve_dataset(data_dir, dataset_id, False)
