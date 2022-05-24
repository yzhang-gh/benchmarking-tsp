import numpy as np
import os
import pickle
import time
import torch
from torch.utils.data import DataLoader, Dataset
from solvers.solver_options import get_dact_solver_options, get_nlkh_solver_options, get_pomo_solver_options

from solvers.solvers import DactSolver, NlkhSolver, PomoSolver
from utils.data_utils import generate_seed, load_dataset

Testset_Size = 1
Graph_Size = 500


# class TSPDataset(Dataset):
#     def __init__(self, data=None, file_path=None, num_samples=1000000):
#         super(TSPDataset, self).__init__()

#         if file_path is not None:
#             assert os.path.splitext(file_path)[1] == ".pkl"
#             assert data == None

#             with open(file_path, "rb") as f:
#                 data = pickle.load(f)
#             data = torch.tensor(data[:num_samples])

#         self.data = [self.make_instance(d) for d in data]
#         self.size = len(self.data)

#     def make_instance(self, args):
#         """used by DACT"""
#         return {"coordinates": torch.FloatTensor(args)}

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         return self.data[idx]


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


def info(text):
    return f"\033[94m{text}\033[0m"


if __name__ == "__main__":
    # torch.set_printoptions(10)

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = "data"
    dataset_id = f"rue{Graph_Size}_seed{generate_seed(Graph_Size, 'rue', 'test')}"
    testfile_name = os.path.join(data_dir, dataset_id)

    # EUC 2D TSP problems
    rue_problems = load_dataset(testfile_name)
    cluster_problems = None

    rue_problems = rue_problems[:Testset_Size]
    rue_problems = torch.tensor(rue_problems, dtype=torch.float)  # numpy ndarray's default dtype is double

    print(f"Loaded '{testfile_name}'. {Testset_Size=}")

    pomo_solver = PomoSolver(
        *get_pomo_solver_options(Graph_Size, "pretrained/pomo/saved_tsp100_model2_longTrain", 3100, seed, 1)
    )
    dact_solver = DactSolver(get_dact_solver_options(Graph_Size, "pretrained/dact/tsp100-epoch-195.pt", seed, 1, 1500))
    nlkh_solver = NlkhSolver(get_nlkh_solver_options("pretrained/nlkh/neurolkh.pt"))

    for solver_name, solver in [("POMO", pomo_solver), ("DACT", dact_solver), ("NeuroLKH", nlkh_solver)]:

        if not solver_name == "NeuroLKH":
            continue

        print(f"== {solver_name} ==")

        for seed in [1, 2, 3, 4, 5]:#

            np.random.seed(seed)
            torch.manual_seed(seed)

            t_start = time.time()

            tours, scores = solver.solve(rue_problems, seed)
            
            t_end = time.time()
            duration = t_end - t_start

            tours = tours.to("cpu")
            costs = get_costs(rue_problems, tours)
            len = costs.mean().item()

            reported_len = -1
            if scores:
                scores = scores.to("cpu")
                reported_len = scores.mean().item()

            print(info(f"{solver_name} {seed=}, {reported_len=:.8f}, {len=:.8f}, {duration=:.2f}s"))
            # assert (torch.div((scores - costs), torch.minimum(scores, costs)).abs() < 2e-7).all()
