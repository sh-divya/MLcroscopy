# import string
# import random
import pprint
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


def _set_by_path(d, path, value):
    """Set value in nested dict `d` following the list `path`.

    Creates intermediate dicts if needed.
    """
    cur = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _iter_sweep(node, path=None):
    """Recursively iterate over sweep specification `node`.

    Yields tuples (path_list, leaf) where leaf is one of:
    - list: categorical choices
    - dict with 'min' and 'max': numeric range
    - dict with other structure: treated as categorical options (the dict itself)

    `path` is the list of keys from the root sweep parameters to this leaf.
    """
    if path is None:
        path = []

    # Lists -> categorical choices
    if isinstance(node, (list, ListConfig)):
        yield (path, list(node))
        return

    # Dict-like nodes: either numeric range or nested dict to descend
    if isinstance(node, (dict, DictConfig)):
        # numeric range specified with 'min' and 'max'
        if "min" in node and "max" in node:
            yield (path, node)
            return

        # otherwise descend into keys
        for k, v in dict(node).items():
            yield from _iter_sweep(v, path + [str(k)])
        return

    # Fallback: single scalar value (not sweep); ignore
    return


def objective(trial, cfg, base_cfg):
    """Objective wrapper used by Optuna.

    This function applies sampled choices into a deep copy of `base_cfg`
    following sweep rules in `cfg` and then (normally) would call
    `train(new_cfg)`. For now it returns a placeholder value.
    """
    global COUNT
    new_cfg = copy.deepcopy(base_cfg)

    for path, leaf in _iter_sweep(cfg):

        param_name = "sweep__" + "__".join(path) if len(path) > 0 else "sweep__root"

        # Handle categorical lists
        if isinstance(leaf, list):
            # Use indices for sampling to preserve original element types
            idx = trial.suggest_categorical(param_name, list(range(len(leaf))))
            chosen = leaf[int(idx)]
            # set the value in the new config at the same nested path
            _set_by_path(new_cfg, path, chosen)

        # Handle numeric ranges specified as dicts with min/max
        elif isinstance(leaf, dict) and "min" in leaf and "max" in leaf:
            low = float(leaf["min"])
            high = float(leaf["max"])
            # Use log scale if requested in the sweep entry
            log = bool(leaf.get("log", True))
            val = trial.suggest_float(param_name, low, high, log=log)
            _set_by_path(new_cfg, path, val)

        # Handle dict leaves that are actually choices (e.g., mapping to lists)
        elif isinstance(leaf, dict):

            options = list(leaf.keys())
            if len(options) > 0:
                choice = trial.suggest_categorical(param_name, options)
                _set_by_path(new_cfg, path, leaf[choice])

    new_cfg["log"]["name"] = new_cfg["log"].get("name", "run") + str(COUNT)
    COUNT = int(COUNT) + 1

    print(new_cfg["log"]["name"])
    # print(new_cfg["model"]["arch"]["micro_kwargs"])
    # print(new_cfg["model"]["arch"]["xrf_kwargs"])
    # print(new_cfg["model"]["arch"]["pred_head_kwargs"])
    pprint.pprint(new_cfg["model"]["arch"])
    print("-----------")
    # return 1
    return train(new_cfg)


@hydra.main(version_base=None, config_path="config", config_name="sweep")
def run(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    model, task, log = get_expt_choices()
    cfg["model"]["name"] = model
    cfg["task"]["name"] = task
    cfg["log"]["type"] = log
    prms = cfg.pop("sweep")["parameters"]
    study = optuna.create_study(direction="minimize") #, sampler=optuna.samplers.RandomSampler())
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
