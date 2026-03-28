import copy

from mlcroscopy.utils.constants import TRANSFROMS_KEY
from mlcroscopy.utils.misc import get_all_kv

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torchvision.transforms import Compose
from torch.utils.data import Subset
from sklearn.model_selection import KFold


def weights_init(m):
    # Skip layers where all parameters are frozen
    if not any(p.requires_grad for p in m.parameters()):
        return
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def make_model(config):
    model_name = config["model"]["name"]
    optim = config["model"]["optim"]
    kwargs = {}
    for p in ["batch_norm", "droprate"]:
        if p in optim:
            kwargs[p] = optim[p]

    model = instantiate(config["model"]["arch"], **kwargs).to(torch.float)
    if model_name in ["mlp", "concat", "wide_light", "wide"]:
        model.apply(weights_init)
    return model


def make_datasets(config):
    data = copy.copy(config["task"])

    transform_opt = ["micro", "xrf", "expt"]
    kwargs = {}

    for t in data["transforms"]:
        if t in transform_opt:
            kwargs[t + "_transform"] = make_transforms(data["transforms"][t])


    data.pop("transforms")
    data.pop("name")

    train_data = instantiate(
        data,
        root=config["data_path"],
        split_fptr="train.idx",
        **kwargs
    )
    test_data = instantiate(
        data,
        root=config["data_path"],
        split_fptr="test.idx",
        **kwargs
    )

    return train_data, test_data


def make_crossval_splits(train_data, config, seed=12345):
    folds = config.get("folds", 0)
    if folds > 0:
        all_idx = list(range(len(train_data)))
        splits = KFold(n_splits=folds, shuffle=True, random_state=seed)
        all_splits = [k for k in splits.split(all_idx)]
        train_folds, val_folds = [], []
        for i in range(folds):
            t, v = all_splits[i]
            train_folds.append(Subset(train_data, t))
            val_folds.append(Subset(train_data, v))
        return train_folds, val_folds
    else:
        return [train_data], None


def make_transforms(augs):
    transforms = []
    for i in get_all_kv(augs):
        k, v = i
        try:            
            t = TRANSFROMS_KEY[k]
            transforms.append(t(v))
        except KeyError:
            k, v = i
            transforms.append(TRANSFROMS_KEY[v])
        except ValueError:
            transforms.append(TRANSFROMS_KEY[i])

    return Compose(transforms)
