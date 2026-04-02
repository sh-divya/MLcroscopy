import hydra
from omegaconf import DictConfig, OmegaConf

from mlcroscopy.utils.misc import (
    get_expt_choices,
    logger,
    set_seeds,
    plot_data_split,
    ROOT,
)
from mlcroscopy.utils.builders import *
from mlcroscopy.utils.constants import LOSS_OPTIONS
from mlcroscopy.models.pl_module import MultiRegress
from mlcroscopy.data.dataset import ScopyLightning

import numpy as np
import torch.nn as nn
from pathlib import Path
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

    xmins, xmax = train_data.stats()
    train_hp = cfg["model"]["optim"]
    splits = make_crossval_splits(train_data, train_hp, cfg["seed"])

    img_path = cfg["log"].get("data_img", None)
    try:
        img_path = Path(img_path) / f"{cfg['log']['name']}.png"
    except TypeError:
        pass

    best_val_mae = []
    best_val_mse = []
    metric_name = f"Val {cfg['log']['metric'][0]} {cfg['log']['metric'][1]}"
    fig_axes = None

    for i, (t, v) in enumerate(zip(*splits)):
        if v is not None:
            has_val = True
        else:
            has_val = False
        cfg["model"]["has_val"] = has_val
        logObj = logger(cfg["log"])

        if img_path:
            fig_axes = plot_data_split(
                t, v, train_hp["folds"], i, fig_axes, xmins.values, xmax.values
            )
        model = make_model(cfg).to(torch.float32)
        loss = LOSS_OPTIONS[cfg["model"]["optim"].get("loss", "l2")]()

        light_data = ScopyLightning(
            t, v, test_data, train_hp["batch_size"], train_hp["num_workers"]
        )

        hparams_to_save = {"model": cfg["model"], "task": cfg["task"]}
        module = MultiRegress(
            model, loss, train_hp["learning-rate"], device, hparams=hparams_to_save
        )

        if train_hp.get("early_stopping", False) and has_val:
            callbacks = [
                pl.callbacks.early_stopping.EarlyStopping(
                    monitor=metric_name,
                    patience=train_hp["scheduler"]["patience"],
                    mode="min",
                )
            ]
        else:
            callbacks = []

        name = f"{cfg['log']['name']}"
        if cfg["log"]["results"] is not None and has_val:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    save_top_k=1,
                    monitor=f"total_val_{cfg['log']['metric'][0].lower()}_{cfg['log']['metric'][1].lower()}",
                    mode="min",
                    dirpath=Path(cfg["log"]["results"]) / name,
                    filename=name + "-{epoch:02d}-{valid_MAE:.2f}",
                )
            )
        else:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    save_last=True,
                    monitor=f"total_train_{cfg['log']['metric'][1]}",
                    dirpath=Path(cfg["log"]["results"]) / name,
                    filename=name + "-last",
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
            accelerator=device,
            callbacks=callbacks,
            limit_val_batches=0 if not has_val else None,
        )
        trainer.fit(module, light_data)

        # if cfg["log"]["type"] == "online":
        #     logObj.experiment.unwatch(module)

        logObj.experiment.finish()
        if has_val:
            if cfg["log"]["metric"][0] == "Eth":
                best_val_mae.append(module.best_eth_mae.item())
                best_val_mse.append(module.best_eth_mse.item())
            if cfg["log"]["metric"][0] == "CO":
                best_val_mae.append(module.best_co_mae.item())
                best_val_mse.append(module.best_co_mse.item())

    if img_path:
        figure, axes = fig_axes
        handles, labels = axes[0][-1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right")
        figure.tight_layout()
        figure.savefig(str(img_path))

    if cfg.get("test", False):
        trainer.test(module, test_data, ckpt_path="best", verbose=False)
    else:
        if len(best_val_mae) > 0:
            print(
                f"Average val {cfg['log']['metric'][0]} MAE across folds:{np.mean(best_val_mae)}"
            )
            print(
                f"Average val {cfg['log']['metric'][0]} MSE across folds:{np.mean(best_val_mse)}"
            )
            return np.mean(best_val_mae)

    print("HERE")
    print(f"Final training MAE:", trainer.callback_metrics["total_train_MAE"])
    print(f"Final training MSE:", trainer.callback_metrics["total_train_MSE"])


if __name__ == "__main__":
    main()
