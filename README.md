# How Good is Neural Combinatorial Optimization?

This repository contains the source code for paper: How Good is Neural Combinatorial Optimization?

**Table of Contents**

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Test](#test)
- [Training](#training)
- [Tuning](#tuning)
- [Others](#others)
- [Acknowledgements](#acknowledgements)

## Directory Structure

```
benchmarking-tsp/
├── problems/
├── solvers/              # NCO solvers, LKH, EAX and LKH(tuned)
├── utils/
│
├── data_large/      ──┐  # TSPLIB, National, etc.
├── data_test/         │
│                      │
├── pretrained/        │
│   ├── dact/          ├── our generated/used datasets, trained models
│   ├── pomo/          │   and test results, provided as *.tar.gz
│   ├── lkh/           │
│   └── nlkh/          │
│                      │
└── test_results/    ──┘
```

[data.tar.gz](https://drive.google.com/file/d/1LNjGdwlEAgwEI15X_OM1aA8zrVg7IIcN/view?usp=sharing), [pretrained.tar.gz](https://drive.google.com/file/d/1Bf3mF53VCiQm5IcQO97NBnffQw6QM-Ek/view?usp=sharing), and [test_results.tar.gz](https://drive.google.com/file/d/1Copgz_pjYtiuSsi-Bqt7EOCgKePbJGr8/view?usp=sharing)

## Installation

```bash
git clone https://github.com/yzhang-gh/benchmarking-tsp.git
cd benchmarking-tsp
# clone neural solvers
git submodule update --init --recursive
```

### Environment

The code requires at least Python 3.8. [Conda](https://docs.conda.io/en/latest/index.html) is recommended for managing the packages.

```bash
conda create -n tsp python=3.8
conda activate tsp
```

Install packages

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

Build NeuroLKH

```bash
cd solvers/nlkh
bash setup.sh
```

Build LKH

```bash
cd solvers/nlkh
make
```

Build EAX

```bash
cd solvers/EAXrestart/src
g++ -o ../bin/GA-EAX-restart -O3 main.cpp environment.cpp cross.cpp evaluator.cpp indi.cpp randomize.cpp kopt.cpp sort.cpp -lm
```

## Test

### Test on synthetic TSP problems

```bash
python test.py
```

The TSP problem settings can be changed with variables `Test_Distribution`, `Graph_Size`, `Testset_Size`, and `Num_Runs`. Change `pomo_options_list`, `dact_options_list`, and `nlkh_options_list` to use different pretrained models.

### Test on TSPLIB/National/VLSI datasets

Make LKH first

```bash
cd solvers/nlkh
make
```

```bash
# cd ../..
python test_tsplib_nlkh.py
```

### Test energy consumption

You need to install [PowerJoular](https://github.com/joular/powerjoular) first (which requires administrative privileges).

```bash
python test_energy.py
```

Remember to change `Python_Abs_Path` and `test_script` accordingly.

### Pretrained models

You can download our pretrained models as provided in the Directory Structure section above. Some of them are obtained from the original authors.

- POMO
  https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver/TSP/POMO/result
- DACT
  https://github.com/yining043/VRP-DACT/tree/main/pretrained
- NeuroLKH
  https://github.com/liangxinedu/NeuroLKH/tree/main/pretrained

## Training

Please check `train_pomo.py`, `train_dact.py`, and `solvers/nlkh/train.py` (need to generate training data with `generate_data_nlkh.py` first).

## Tuning

The Sequential Model-based Algorithm Configuraiton ([SMAC](https://github.com/automl/SMAC3)) is used to tune LKH:

- Configuration scenario file: SMAC_scenario.txt
- Parameter configuration space file: LKH_pcs.txt
- Solver wrapper of LKH and the execution monitoring tool (Runsolver): the same as in [CEPS](https://github.com/senshineL/CEPS).
- Time budgets for tuning LKH: 24 hours for `rue/clustered/mix-100`, and 96 hours for `rue/clustered/mix-500/1000`.

## Others

<details>
<summary><strong>Abbreviations Used in the Source Code</strong></summary>

| Abbr.  | Meaning                  |
| ------ | ------------------------ |
| rue    | random uniform Euclidean |
| clust  | cluster(ed)              |
| feat   | feature                  |
| opt(s) | option(s)                |
| optim  | optimum                  |
| sol    | solution                 |

</details>

### TSP Data Generation

There are two types of TSP data: `rue` and `clustered`. For `rue`, the nodes (cities) are randomly and uniformly distributed within a unit square where the coordinates $x, y \in [0, 1)$.

<!-- In order to be consistent with the TSPLIB 95 format

Transformation

coordinate precision 0.000001 -->

<!-- ### TSP Data Organization

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
``` -->

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

## Acknowledgements

**Compared neural solvers**

- [DACT](https://github.com/yining043/VRP-DACT), Ma et al., *NeurIPS*, 2021
- [NeuroLKH](https://github.com/liangxinedu/NeuroLKH), Xin et al., *NeurIPS*, 2021
- [POMO](https://github.com/yd-kwon/POMO), Kwon et al., *NeurIPS*, 2020

<!--  -->

**Others**

- [Attention, Learn to Solve Routing Problems!](https://github.com/wouterkool/attention-learn-to-route), Kool et al., *ICLR*, 2019
- [netgen](https://github.com/jakobbossek/netgen), generating random (clustered) networks in R
