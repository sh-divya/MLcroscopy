import string
import random
import copy
import warnings
from functools import partial

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf, ListConfig

from run import train
from mlcroscopy.utils.misc import get_expt_choices, get_all_kv

warnings.filterwarnings("ignore", ".*does not have many workers.*")


COUNT = 0


def objective(trial, cfg, base_cfg):
    global COUNT
    new_cfg = copy.deepcopy(base_cfg)

    try:
        arch_cfg = cfg["model"]["arch"]
        for k, v in arch_cfg.items():
            print(k)
            print(v)
            if isinstance(v, (dict, DictConfig)):
                for p, q in v.items():
                    choice = trial.suggest_categorical(f"{k}_{p}", list(range((len(q)))))
                    new_cfg["model"]["arch"][k][p] = q[choice]
            elif isinstance(v, (list, ListConfig)):
                choice = trial.suggest_categorical(k, list(range(len(v))))
                new_cfg["model"]["arch"][k] = v[choice]
    except KeyError:
        pass

    try:
        optim_cfg = cfg["model"]["optim"]
        for k, v in optim_cfg.items():
            if k == "lr":
                low = optim_cfg["lr"]["min"]
                high = optim_cfg["lr"]["max"]
                new_cfg["model"]["optim"]["learning-rate"] = trial.suggest_float("lr", low, high, log=True)
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
    COUNT = int(COUNT) + 1

    print(new_cfg["log"]["name"])
    # print(new_cfg["model"]["arch"]["micro_kwargs"]["layers"])
    # print(new_cfg["model"]["arch"]["xrf_kwargs"]["layers"])
    # print(new_cfg["model"]["arch"]["expt_kwargs"])
    # print(new_cfg["model"]["arch"]["pred_head_kwargs"])
    # print(new_cfg["model"]["arch"]["merge"])
    print("-----------")
    
    return 1
    # return train(new_cfg)


@hydra.main(version_base=None, config_path="config", config_name="sweep")
def run(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    model, task, log = get_expt_choices()
    cfg["model"]["name"] = model
    cfg["task"]["name"] = task
    cfg["log"]["type"] = log
    prms = cfg.pop("sweep")["parameters"]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())
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
