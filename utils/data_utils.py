import os
import pickle

import numpy as np


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


Possible_Distributions = ["rue", "clust"]


def upscale_tsp_coords(coords):
    """[0, 1) to {1, 2, ..., 1000000}
    This assumes [0, 1) has already been discretized to {0, 0.000001, ..., 0.999998, 0.999999}
    """
    int_coords = np.round(coords * 1000000) + 1
    int_coords = int_coords.astype(int)
    return int_coords


def downscale_tsp_coords(coords):
    """{1, 2, ..., 1000000} to [0, 1)
    In fact, {0, 0.000001, ..., 0.999998, 0.999999}
    """
    coords = (coords - 1) / 1000000
    return coords


def generate_seed(graph_size, distribution, mode="train"):
    seed = graph_size * 10 + Possible_Distributions.index(distribution)
    if mode == "val":
        seed += 4
    elif mode == "test":
        seed += 8

    return seed
