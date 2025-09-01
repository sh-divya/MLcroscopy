from mlcroscopy.data.dataset import ScopyTensor, OnlyExpt
from mlcroscopy.models.archs import ScopyResNet, ExptMLP


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import (
    Normalize,
    RandomCrop,
    ColorJitter,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    Compose,
    Grayscale,
)


from sklearn.model_selection import KFold

XRF_Z = {
    "mean": (378035.5, 5707524.33, 24599.7, 5361.4),
    "std": (362643.5, 4208579.6, 4208579.6, 2608.0),
}

MICRO_Z = {"mean": 153.62, "std": 27.63}

transforms_key = {
    "norm": lambda x: Normalize(x["mean"], x["std"]),
    "crop": lambda x: RandomCrop(x),
    "flip": {"horizontal": RandomHorizontalFlip(), "vertical": RandomVerticalFlip()},
    "color": lambda x: ColorJitter(**x),
}


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def make_model(config):
    model_name = config["model"]["name"]
    model = config["model"]["arch"]
    optim = config["model"]["optim"]
    if model_name == "wide":
        bn = optim["batch_norm"]
        drop = optim["drop_rate"]

        model = ScopyResNet(
            model["pred_head_len"],
            model["pred_head_width"],
            model["pred_hidden_len"],
            model["micro_inp_width"],
            model["micro_out_width"],
            model["micro_res_n"],
            model["xrf_conv_width"],
            model["xrf_res_len"],
            bn,
            drop,
        ).to(torch.float)
    elif model_name == "mlp":
        bn = optim["batch_norm"]
        model = ExptMLP(model["hidden_dim"], model["hidden_layers"], bn).to(torch.float)
    else:
        raise ValueError(f"{model_name} not available")

    model.apply(weights_init)
    return model


def make_datasets(config):
    data_type = config["task"]["name"]
    data = config["task"]
    if data_type == "scratch":
        microt = make_transforms("micro", data["transforms"]["micro"])
        xrft = make_transforms("xrf", data["transforms"]["xrf"])
        train_data = ScopyTensor(
            data["base_data"],
            "train.idx",
            micro_transform=microt,
            xrf_transform=xrft,
            ytransform=data["transforms"]["target"],
        )
        # train_loader = DataLoader(train_data, config["model"]["optim"]["batch_size"], shuffle=True)

        test_data = ScopyTensor(
            data["base_data"],
            "test.idx",
        )
        # test_loader = DataLoader(test_data, config["model"]["optim"]["batch_size"], shuffle=False)
        return train_data, test_data
    elif data_type == "expt":
        train_data = OnlyExpt(data["base_data"], "train.idx", data["target"])
        test_data = OnlyExpt(data["base_data"], "test.idx")
        return train_data, test_data
    else:
        raise ValueError(f"{data_type} not available")


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


def make_transforms(src, augs):
    transforms = []
    for a, v in augs.items():
        t = transforms_key[a]
        # if a == 'norm':
        #     if src == "micro":
        #         t = t(MICRO_Z["mean"], MICRO_Z["std"])
        #     elif src == "xrf":
        #         t = t(XRF_Z["mean"], XRF_Z["std"])
        #     else:
        #         raise KeyError(f"Normalization for {src} not available")
        # elif isinstance(transforms_key[a], list):
        #     t = [i() for i in t]
        # elif a == "color":
        #     t = t(brightness=(0.2, 1.2), contrast=(0.2, 1.2))
        # else:
        #     t = t()
        if a == "flip":
            for i in v:
                transforms.append(t[i])
        else:
            transforms.append(t(v))

    return Compose(transforms)
