import string
import random
import copy
import warnings
from functools import partial

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from run import train
from mlcroscopy.utils.misc import get_expt_choices

warnings.filterwarnings("ignore", ".*does not have many workers.*")

COUNT = 0


def objective(trial, cfg, base_cfg):
    global COUNT
    new_cfg = copy.deepcopy(base_cfg)

    try:
        arch_cfg = cfg["model"]["arch"]
        for k, v in arch_cfg.items():
            if len(v) > 0:
                new_cfg["model"]["arch"][k] = trial.suggest_categorical(k, v)
            else:
                new_cfg["model"]["arch"][k] = v[0]
    except KeyError:
        pass

    try:
        optim_cfg = cfg["model"]["optim"]
        for k, v in optim_cfg.items():
            if k == "lr":
                low = optim_cfg["lr"]["min"]
                high = optim_cfg["lr"]["max"]
                new_cfg["model"]["optim"][k] = trial.suggest_float("lr", low, high, log=True)
            elif k == "scheduler":
                new_cfg["model"]["optim"][k]["name"] = trial.suggest_categorical(k, v["name"])
            else:
                if len(v) > 0:
                    new_cfg["model"]["optim"][k] = trial.suggest_categorical(k, v)
                else:
                    new_cfg["model"]["optim"][k] = v[0]
    except KeyError:
        pass

    new_cfg["log"]["name"] = new_cfg["log"]["name"] + str(COUNT)
    # print(new_cfg["log"]["name"])
    COUNT = int(COUNT) + 1
    # print(COUNT)
    # raise Exception
    return train(new_cfg)


@hydra.main(version_base=None, config_path="config", config_name="sweep")
def run(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    model, task, log = get_expt_choices()
    cfg["model"]["name"] = model
    cfg["task"]["name"] = task
    cfg["log"]["type"] = log
    prms = cfg.pop("sweep")["parameters"]
    study = optuna.create_study(direction="minimize")
    study.optimize(partial(objective, cfg=prms, base_cfg=cfg), n_trials=cfg["trials"], timeout=None)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    run()
