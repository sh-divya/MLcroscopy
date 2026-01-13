from mlcroscopy.data.dataset import ScopyTensor, OnlyExpt, ImgPlusExpt
from mlcroscopy.models.archs import (
    ScopyResNet,
    ScopyResLight,
    ExptMLP,
    BothTensorandExpt,
    ScopyDino
)
from mlcroscopy.utils.constants import XRF_Z, MICRO_Z, EXPT_Z, TRANSFROMS_KEY

import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


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
    elif model_name == "wide_light":
        bn = optim["batch_norm"]
        drop = optim["drop_rate"]
        model = ScopyResLight(
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
    elif model_name == "dino_light":
        bn = optim["batch_norm"]
        drop = optim["drop_rate"]
        model = ScopyDino(model["pred_hidden_len"],
                          model["pred_head_width"],
                          model["xrf_conv_width"],
                          model["xrf_res_len"],
                          drop, bn, model["name"]).to(torch.float)
    elif model_name == "concat":
        bn = optim["batch_norm"]
        drop = optim["drop_rate"]
        model = BothTensorandExpt(
            model["pred_head_len"],
            model["pred_head_width"],
            model["pred_hidden_len"],
            model["micro_inp_width"],
            model["micro_out_width"],
            model["micro_res_n"],
            model["xrf_conv_width"],
            model["xrf_res_len"],
            model["expt_hidden_len"],
            bn,
            drop,
        ).to(torch.float)
    elif model_name == "mlp":
        bn = optim["batch_norm"]
        model = ExptMLP(model["hidden_dim"], model["hidden_layers"], bn).to(torch.float)
    else:
        raise ValueError(f"{model_name} not available")

    if model_name in ["mlp", "concat", "wide_light", "wide"]:
        model.apply(weights_init)
    elif model_name in ["dino_light"]:
        pass
    return model


def make_datasets(config):
    data_type = config["task"]["name"]
    data = config["task"]
    if data_type in ["scratch", "dino"]:
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
        xstats= {"mean": torch.tensor(EXPT_Z["mean"]), "std": torch.tensor(EXPT_Z["std"])}
        train_data = OnlyExpt(data["base_data"], "train.idx", xstats, data["target"])
        test_data = OnlyExpt(data["base_data"], "test.idx", xstats=EXPT_Z)
        return train_data, test_data
    elif data_type == "both":
        microt = make_transforms("micro", data["transforms"]["micro"])
        xrft = make_transforms("xrf", data["transforms"]["xrf"])
        train_data = ImgPlusExpt(
            data["base_data"],
            "train.idx",
            micro_transform=microt,
            xrf_transform=xrft,
            ytransform=data["transforms"]["target"],
        )
        test_data = ImgPlusExpt(
            data["base_data"],
            "test.idx",
        )
        return train_data, test_data
    # elif data_type == "dino":
    #     microt = make_transforms("micro", data["transforms"]["micro"])
    #     xrft = make_transforms("xrf", data["transforms"]["xrf"])

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
        t = TRANSFROMS_KEY[a]
        # if a == 'norm':
        #     if src == "micro":
        #         t = t(MICRO_Z["mean"], MICRO_Z["std"])
        #     elif src == "xrf":
        #         t = t(XRF_Z["mean"], XRF_Z["std"])
        #     else:
        #         raise KeyError(f"Normalization for {src} not available")
        # elif isinstance(TRANSFORMS_KEY[a], list):
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
