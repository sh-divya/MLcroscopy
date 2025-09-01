from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

import numpy as np
import random
import os


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
        )
    elif log_cfg["type"] == "debug":
        logObj = DummyLogger()
    else:
        raise ValueError(f"{log_cfg['type']} logger not available")

    return logObj
