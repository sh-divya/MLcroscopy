from torchvision.transforms import (
    Normalize,
    RandomCrop,
    ColorJitter,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    RandomRotation,
    Grayscale,
    Lambda
)

import torch.nn as nn
from mlcroscopy.utils.misc import Norm1D

XRF_Z = {
    "mean": (378035.5, 5707524.33, 24599.7, 5361.4),
    "std": (362643.5, 4208579.6, 4208579.6, 2608.0),
}

MICRO_Z = {
    "mean": [166.0630, 151.4283, 124.3986],
    "std": [29.7448, 28.4966, 30.6221]
}

EXPT_Z = {
    "mean": (5.6500e-01, 7.3500e+02, 7.9500e+02, 5.0840e-06, 1.6995e+01),
    "std": (3.3639e-01, 5.7779e+02, 5.9187e+02, 7.9433e-07, 2.2152e+00)
}

TRANSFROMS_KEY = {
    "oneD": lambda x: Norm1D(x["mean"], x["std"]),
    "norm": lambda x: Normalize(x["mean"], x["std"]),
    "crop": lambda x: RandomCrop(x),
    "horizontal": RandomHorizontalFlip(),
    "vertical": RandomVerticalFlip(),
    "color": lambda x: ColorJitter(**x),
    "spin": lambda x: RandomRotation(x)
}

LOSS_OPTIONS = {
    "l1": nn.L1Loss,
    "l2": nn.MSELoss,
    "smoothl1": nn.SmoothL1Loss,
}