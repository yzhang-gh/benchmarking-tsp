# Benchmarking TSP

(Introduction)

## Install

```bash
git clone https://github.com/yzhang-gh/benchmarking-tsp.git
cd benchmarking-tsp
# clone neural solvers
git submodule update --init --recursive
```

### Environment

Python 3.8

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install tqdm tensorboard

# === POMO specific ===
conda install pytz

# === NeuroLKH specific ===
conda install scikit-learn cython swig
# PyConcorde
git clone https://github.com/jvkersch/pyconcorde.git
cd pyconcorde
pip install -e .
# use `pip install -e . --no-build-isolation --no-binary :all:`
# if you get the 'ValueError: numpy.ndarray size changed, ...' error

# === Clustered TSP data generation ===
conda install r r-essentials
R
# in the R session
> install.packages("devtools", repos='...') # select a mirror from https://cran.r-project.org/mirrors.html
> Sys.setenv(TAR = "/bin/tar")
> devtools::install_github("jakobbossek/netgen")
> q()  # quit
```

## Test

### Pretrained Models

- POMO
  https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver/TSP/POMO/result
- DACT
  https://github.com/yining043/VRP-DACT/tree/main/pretrained
- NeuroLKH
  https://github.com/liangxinedu/NeuroLKH/tree/main/pretrained

## Train

## Others

### TSP Data Generation

There are two types of TSP data:

Transformation

coordinate precision 0.000001

### TSP Data Organization

```
dataset_id := <distribution><graph_size>_seed<seed>
# e.g., rue50_seed1234

data/
├── <dataset_id>.pkl      # shape: (dataset_size, graph_size, 2)
├── <dataset_id>.sol.pkl
├── ...
│
└── <dataset_id>/         # tmp files
    ├── xxxx.tsp
    ├── xxxx.sol
    ├── xxxx.log
    └── ...
```

<!-- #### Default Random Seed

train

```
NeuroLKH
<graph_size> * 10 + 0/1 (rue/clustered)

LKH, GA-EAX
1234
```

test

```
<graph_size> * 10 + 8/9 (rue/clustered)
``` -->

<details>
<summary>Abbreviations Used in the Source Code</summary>

| Abbr. | Meaning                  |
| ----- | ------------------------ |
| rue   | random uniform Euclidean |
| clust | cluster(ed)              |
| feat  | feature                  |

</details>

## Acknowledgements

**Compared neural solvers**

- [DACT](https://github.com/yining043/VRP-DACT), Ma et al., *NeurIPS*, 2021
- [NeuroLKH](https://github.com/liangxinedu/NeuroLKH), Xin et al., *NeurIPS*, 2021
- [POMO](https://github.com/yd-kwon/POMO), Kwon et al., *NeurIPS*, 2020

<!--  -->

**Others**

- [Attention, Learn to Solve Routing Problems!](https://github.com/wouterkool/attention-learn-to-route), Kool et al., *ICLR*, 2019
- [netgen](https://github.com/jakobbossek/netgen), generating random (clustered) networks in R
