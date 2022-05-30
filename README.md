
## Environment

Python 3.8

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install tqdm

# === DACT specific ===
pip install opencv-python tensorboard_logger

# === NeuroLKH specific ===
conda install scikit-learn cython swig
# PyConcorde
git clone https://github.com/jvkersch/pyconcorde.git
cd pyconcorde
pip install -e .
# use `pip install -e . --no-build-isolation --no-binary :all:`
# if you get the 'ValueError: numpy.ndarray size changed, ...' error

# === Data generation ===
conda install r r-essentials
R
# in the R session
> install.packages("devtools", repos='...') # select a mirror from https://cran.r-project.org/mirrors.html
> Sys.setenv(TAR = "/bin/tar")
> devtools::install_github("jakobbossek/netgen")
> q()  # quit

# Training
conda install tensorboard

# === POMO specific ===
conda install pytz
```

## Data Organization and Naming

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

### Default Random Seed

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
```

## Pretrained Models

- POMO
  https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver/TSP/POMO/result
- DACT
  https://github.com/yining043/VRP-DACT/tree/main/pretrained
- NeuroLKH
  https://github.com/liangxinedu/NeuroLKH/tree/main/pretrained

## Abbreviation

clust --- cluster
feat --- feature

## Acknowledgements

https://github.com/wouterkool/attention-learn-to-route

netgen

---

bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
