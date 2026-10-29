from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
import wandb

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import random
import os

ROOT = Path(__file__).parent.parent.parent


class Norm1D:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, x):
        return (x - self.mean) / self.std
    
    def denorm(self, x):
        return (x * self.std) + self.mean


def get_all_kv(d, stop_keys=None):
    stop_keys = set(stop_keys or [])
    for k, v in d.items():
        if k in stop_keys:
            yield (k, v)
        elif isinstance(v, dict):
            yield from get_all_kv(v, stop_keys=stop_keys)
        elif isinstance(v, list):
            for i in v:
                yield (k, i)
        else:
            yield (k, v)

def plot_data_split(
    train, val, folds=3, split_idx=0, fig_axes=None, xmins=(0.0, 0.0), xmax=(2.0, 2.0)
):
    if not fig_axes:
        figure, axes = plt.subplots(2, folds)
        axes[0][0].set_ylabel("FE (Eth)")
        axes[1][0].set_ylabel("FE (CO)")
        figure.suptitle("Values in Splits")

        for i, ax in enumerate(axes):
            for a in ax:
                a.set_ylim(xmins[i], xmax[i])
    else:
        figure, axes = fig_axes

    y1 = list(zip(*[j.tolist() for _, j in val]))
    y2 = list(zip(*[j.tolist() for _, j in train]))

    axes[0][split_idx].hist(y1[0], 10, density=True, alpha=0.5, label="Val")
    axes[1][split_idx].hist(y1[1], 10, density=True, alpha=0.5, label="Val")

    axes[0][split_idx].hist(y2[0], 10, density=True, alpha=0.5, label="Train")
    axes[1][split_idx].hist(y2[1], 10, density=True, alpha=0.5, label="Train")

    return figure, axes


def set_seeds(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_expt_choices():
    hydra_cfg = HydraConfig.get()
    hydra_cfg = OmegaConf.to_container(hydra_cfg.runtime.choices)

    return hydra_cfg["model"], hydra_cfg["task"], hydra_cfg["log"]


def slurm_job_id():
    return os.environ.get("SLURM_JOB_ID", "default")


def logger(log_cfg, tag=0):
    if not tag:
        tag = slurm_job_id()
    # suff = str(tag) if tag else ""
    name = log_cfg.get("name", tag)
    # this line is unnecesary i think
    if not name:
        name = tag
    if log_cfg["type"] == "online":
        logObj = WandbLogger(
            project=log_cfg["project"],
            name=name,
            entity=log_cfg["user"],
            tags=[tag],
            save_dir=log_cfg.get("results", "wandb"),
            settings=wandb.Settings(init_timeout=180)
        )
    elif log_cfg["type"] == "debug":
        logObj = DummyLogger()
    else:
        raise ValueError(f"{log_cfg['type']} logger not available")

    return logObj

def get_successive_crops(sample, crop_size, device):
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        raise ValueError("Expected sample to be a tuple of (inputs, target)")

    inputs, target = sample
    if not isinstance(inputs, (tuple, list)) or len(inputs) == 0:
        raise ValueError("Expected inputs to contain the microscopy tensor first")

    img_tensor = inputs[0].to(device, dtype=torch.float16)
    other_inputs = tuple(x.to(device, dtype=torch.float16).unsqueeze(0) for x in inputs[1:])
    target = target.to(device, dtype=torch.float16)

    if img_tensor.ndim != 3:
        raise ValueError(
            f"Expected image tensor with shape (C, H, W), got {tuple(img_tensor.shape)}"
        )

    _, height, width = img_tensor.shape
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"Crop size {crop_size} is larger than image size {(height, width)}"
        )

    row_starts = list(range(0, height - crop_size + 1, crop_size))
    col_starts = list(range(0, width - crop_size + 1, crop_size))

    if row_starts[-1] != height - crop_size:
        row_starts.append(height - crop_size)
    if col_starts[-1] != width - crop_size:
        col_starts.append(width - crop_size)

    crops = []
    for top in row_starts:
        for left in col_starts:
            crop = img_tensor[:, top : top + crop_size, left : left + crop_size]
            crop_inputs = (crop.unsqueeze(0), *other_inputs)
            crops.append((crop_inputs, target))

    return crops