import os
import pickle


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


def generate_seed(graph_size, distribution, mode="train"):
    seed = graph_size * 10 + Possible_Distributions.index(distribution)
    if mode == "val":
        seed += 4
    elif mode == "test":
        seed += 8

    return seed
