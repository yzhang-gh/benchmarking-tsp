
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
> install.packages("devtools")
> Sys.setenv(TAR = "/bin/tar")
> devtools::install_github("jakobbossek/netgen")
```

## Pretrained Models

- POMO
  https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver/TSP/POMO/result
- DACT
  https://github.com/yining043/VRP-DACT/tree/main/pretrained
- NeuroLKH
  https://github.com/liangxinedu/NeuroLKH/tree/main/pretrained

## Acknowledgements

https://github.com/wouterkool/attention-learn-to-route
