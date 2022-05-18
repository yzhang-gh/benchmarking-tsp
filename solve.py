import os
from multiprocessing import Pool

from tqdm import tqdm

from problems.tsp.tsp_baseline import solve_concorde_log
from utils.data_utils import load_dataset, save_dataset

num_cpus = os.cpu_count() // 4 * 3

use_int_distance = False

executable = os.path.abspath("problems/tsp/concorde/concorde/TSP/concorde")
distribution = "clust"
graph_size = 100
seed = 1234
directory = f"data_tmpfiles/{distribution}{graph_size}"
file_basename = f"{distribution}{graph_size}_seed{seed}"
dataset = load_dataset(f"data/{file_basename}")

print(f"{use_int_distance=}")
print(f"Solving {file_basename}.pkl")


def run_func(args):
    return solve_concorde_log(executable, *args, disable_cache=True, int_distance=use_int_distance)


with Pool(num_cpus) as pool:
    results = list(
        tqdm(
            pool.imap(
                run_func,
                [(directory, f"{file_basename}_{i}", data) for i, data in enumerate(dataset)],
            ),
            total=len(dataset),
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )
    )

failed = [str(i) for i, res in enumerate(results) if res is None]
assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))

save_dataset(results, f"data/{file_basename}.sol{'.int' if use_int_distance else ''}.pkl")
