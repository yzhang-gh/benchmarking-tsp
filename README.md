
## Environment

Python 3.8

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install tqdm

# DACT specific
pip install opencv-python tensorboard_logger

# NeuroLKH specific
conda install scikit-learn cython swig
# PyConcorde
git clone https://github.com/jvkersch/pyconcorde.git
cd pyconcorde
pip install -e .
# use `pip install -e . --no-build-isolation --no-binary :all:`
# if you get the 'ValueError: numpy.ndarray size changed, ...' error
```

## Acknowledgements

https://github.com/wouterkool/attention-learn-to-route
