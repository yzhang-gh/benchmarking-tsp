import glob
import pickle
import json
import os
import subprocess
from problems.tsp.tsp_baseline import write_tsplib
import numpy as np

from utils.data_utils import downscale_tsp_coords, generate_seed, load_dataset, save_dataset
from utils.file_utils import load_tsplib_file


def collect_optim_tour_len(data_dir, dataset_id):
    print(f"collecting optim tour len from {data_dir}/{dataset_id}")
    optim_tour_len = {}

    with open(f"{data_dir}/{dataset_id}.sol.int.pkl", "rb") as rb:
        data = pickle.load(rb)
        num_digits = len(str(len(data) - 1))
        for i, (l, t, d) in enumerate(data):
            optim_tour_len.update({f"{data_dir}/{dataset_id}/{i:0{num_digits}d}.tsp": str(l)})

        with open(f"optim_tour_len_{dataset_id}.json", "w") as w:
            w.write(json.dumps(optim_tour_len, indent=4))


def pkl_to_tsp(dataset_id):
    print(dataset_id, "to .tsp")
    dataset = load_dataset(f"../test_data/{dataset_id}")
    num_digits = len(str(len(dataset) - 1))
    dot_tsp_file_dir = f"../test_data/{dataset_id}_2"

    count = 0
    for i, data in enumerate(dataset):
        tsp_file_name = f"{i:0{num_digits}d}"

        assert len(tsp_file_name) <= 8, "Concorde will produce tmp files using only the first 8 chars of filename"

        problem_filename = os.path.join(dot_tsp_file_dir, "{}.tsp".format(tsp_file_name))
        if not os.path.exists(problem_filename):
            write_tsplib(problem_filename, data, name=tsp_file_name)
            count += 1
    print(f"{count} files written to {dot_tsp_file_dir}")


def collect_dot_tsp_files(dataset_id, dest_folder):
    os.makedirs(f"../{dest_folder}/{dataset_id}", exist_ok=True)
    cmd = f"cp data/{dataset_id}/*.tsp ../{dest_folder}/{dataset_id}/"
    print(cmd)
    subprocess.run(cmd, shell=True)


def dot_tsp_files_to_pkl(dataset_id):
    print(dataset_id)
    data = []
    files = glob.glob(f"../test_data/{dataset_id}/*")
    files = sorted(files)
    num_digits = len(str(len(files) - 1))
    for i, f in enumerate(files):
        assert f.endswith(f"{i:0{num_digits}d}.tsp"), f"{i=}, {f=}"
        coords = load_tsplib_file(f)
        data.append(coords)
    data = np.stack(data)
    assert (data > 0).all()
    data = downscale_tsp_coords(data)
    save_dataset(data, f"../test_data/{dataset_id}")


for distribution in ["rue", "clust"]:
    for graph_size in [50, 100, 500, 1000]:
        seed = generate_seed(graph_size, distribution, "test")
        # seed = 1234
        # data_dir = "data"
        dataset_id = f"{distribution}{graph_size}"#_seed{seed}

        # collect_optim_tour_len(data_dir, dataset_id)
        pkl_to_tsp(dataset_id)
        # collect_dot_tsp_files(dataset_id, "train_data")
        # dot_tsp_files_to_pkl(dataset_id)
