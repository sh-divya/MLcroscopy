import hydra
from omegaconf import DictConfig, OmegaConf

from mlcroscopy.utils.misc import get_expt_choices, logger, set_seeds
from mlcroscopy.utils.builders import *
from mlcroscopy.models.pl_module import MultiRegress
from mlcroscopy.data.dataset import ScopyLightning

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    cfg = OmegaConf.to_container(cfg)
    model, task, log = get_expt_choices()
    cfg["model"]["name"] = model
    cfg["task"]["name"] = task
    cfg["log"]["type"] = log

    train(cfg)


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seeds(cfg["seed"])
    train_data, test_data = make_datasets(cfg)

    train_hp = cfg["model"]["optim"]
    splits = make_crossval_splits(train_data, train_hp, cfg["seed"])
    best_val_mae = []
    best_val_mse = []

    for i, (t, v) in enumerate(zip(*splits)):
        logObj = logger(cfg["log"])
        model = make_model(cfg).to(torch.float32)
        loss = nn.MSELoss()
        light_data = ScopyLightning(
            t, v, test_data, train_hp["batch_size"], train_hp["num_workers"]
        )
        module = MultiRegress(model, loss, train_hp["learning-rate"], device)

        if cfg["model"]["optim"].get("early_stopping", False):
            callbacks = [
                pl.callbacks.early_stopping.EarlyStopping(
                    monitor="Val MAE",
                    patience=cfg["model"]["optim"]["scheduler"]["patience"],
                    mode="min",
                )
            ]
        else:
            callbacks = []

        name = f"{cfg['log']['name']}"
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                save_top_k=1,
                monitor="valid_MAE",
                mode="min",
                dirpath=cfg["log"]["results"],
                filename=name + "-{epoch:02d}-{valid_MAE:.2f}",
            )
        )
        # if cfg["log"]["type"] == "online":
        #     logObj.watch(module, log='all', log_freq=25)
        trainer = pl.Trainer(
            max_epochs=train_hp["epochs"],
            logger=logObj,
            log_every_n_steps=1,
            min_epochs=1,
            default_root_dir=cfg["log"]["results"],
        )
        trainer.fit(module, light_data)

        # if cfg["log"]["type"] == "online":
        #     logObj.experiment.unwatch(module)

        logObj.experiment.finish()
        if v:
            best_val_mae.append(module.best_mae.item())
            best_val_mse.append(module.best_mse.item())

    if cfg.get("test", False):
        trainer.test(module, test_data, ckpt_path="best", verbose=False)
    else:
        if best_val_mae:
            print(f"Average val MAE across folds:{np.mean(best_val_mae)}")
            print(f"Average val MSE across folds:{np.mean(best_val_mse)}")

    return np.mean(best_val_mae)


if __name__ == "__main__":
    main()
