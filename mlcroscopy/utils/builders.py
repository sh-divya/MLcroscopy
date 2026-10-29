import copy

from mlcroscopy.utils.constants import TRANSFROMS_KEY
from mlcroscopy.utils.misc import get_all_kv

import torch
import torch.nn as nn
from hydra.utils import instantiate
from torchvision.transforms import Compose
from torch.utils.data import Subset
from sklearn.model_selection import GroupKFold


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
    train_data = make_dataset(config, "train.idx")
    test_data = make_dataset(config, "test.idx")

    return train_data, test_data


def make_dataset(config, split_fptr):
    """Build one configured task dataset from an index-file path."""
    data = copy.copy(config["task"])
    kwargs = make_transforms(data.pop("transforms", {}))
    data.pop("name", None)

    return instantiate(
        data,
        root=config["data_path"],
        split_fptr=split_fptr,
        **kwargs,
    )


def make_crossval_splits(train_data, config, seed=12345):
    folds = config.get("folds", 0)
    if folds > 1:
        all_idx = list(range(len(train_data)))
        triplet_ids = train_data.csv.loc[train_data.idx, "triplet"].to_numpy()
        splits = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        all_splits = list(splits.split(all_idx, groups=triplet_ids))
        train_folds, val_folds = [], []
        for i in range(folds):
            t, v = all_splits[i]
            train_folds.append(Subset(train_data, t))
            val_folds.append(Subset(train_data, v))
        return train_folds, val_folds
    else:
        return [train_data], [None]


def make_transforms(augs):
    transform_opt = {"micro", "xrf", "expt"}
    if any(k in transform_opt for k in augs):
        return {
            f"{k}_transform": make_transforms(v)
            for k, v in augs.items()
            if k in transform_opt
        }

    transforms = []
    for k, v in get_all_kv(augs, stop_keys=TRANSFROMS_KEY.keys()):
        if k in TRANSFROMS_KEY:
            transforms.append(TRANSFROMS_KEY[k](v))
        elif v in TRANSFROMS_KEY:
            transforms.append(TRANSFROMS_KEY[v])
        else:
            raise KeyError(f"Transform {k}: {v} not available")

    return Compose(transforms)
