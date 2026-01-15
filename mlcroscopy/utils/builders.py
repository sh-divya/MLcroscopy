from mlcroscopy.utils.constants import TRANSFROMS_KEY
from mlcroscopy.utils.misc import get_all_kv

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torchvision.transforms import Compose
from torch.utils.data import Subset
from sklearn.model_selection import KFold


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def make_model(config):
    model_name = config["model"]["name"]
    optim = config["model"]["optim"]
    bn = optim.get("batch_norm", False)
    drop = optim.get("drop_rate", None)

    model = instantiate(config.model.arch, batch_norm=bn, droprate=drop).to(torch.float)
    if model_name in ["mlp", "concat", "wide_light", "wide"]:
        model.apply(weights_init)
    elif model_name in ["dino_light"]:
        pass
    return model


def make_datasets(config):
    data_type = config["task"]["name"]
    data = config["task"]

    microt = None
    xrft = None
    exptt = None

    try:
        microt = make_transforms(data["transforms"]["micro"])
        xrft = make_transforms(data["transforms"]["xrf"])
    except KeyError:
        exptt = make_transforms(data["transforms"])

    train_data = instantiate(
        data,
        root=config["base_data"],
        split_fptr="train.idx",
        micro_transform=microt,
        xrf_transform=xrft,
        expt_transform=exptt,
    )
    test_data = instantiate(
        data,
        root=config["base_data"],
        split_fptr="test.idx",
        micro_transform=microt,
        xrf_transform=xrft,
        expt_transform=exptt,
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
    for k, v in get_all_kv(augs):
        t = TRANSFROMS_KEY[k]
        transforms.append(t(v))

    return Compose(transforms)
