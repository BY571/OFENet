# OFENet

PyTorch implementation of the [OFENet Paper](https://arxiv.org/abs/2003.01629).

## Work in progress - working now but still not as good as the paper performance

If you might be interested in the work check out the notebook. Currently it is not working as described in the paper, if you find errors or bugs feel free to let me know or correct them. 

# Environment Setup

1. run: `conda create -n OFENet python=3.7`
2. enter the environment with `conda activate OFENet`
3 run the installation the requirement.txt file with: `pip install -r requirement.txt`

# To run 

To run one experiment simply type: `python sac_ofenet.py`

All results are logged with tensorboard to check them type: `tensorboard --logdir=runs`

## TODO:
- fix hyperparameter saving bug
- OFENet training got worse, should have final loss for halfcheetah of 0.005 but currently has 0.12. Before the changes that made ofenet work loss was as paper loss or even lower. (problem might be batch norm?)
- Create plots for halfcheetah and other env
- add target-dim loading or values in a table in the readme

