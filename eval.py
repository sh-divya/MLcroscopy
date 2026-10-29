"""Evaluate a trained MLcroscopy model on one or more dataset splits."""

import json
from pathlib import Path
from typing import Any, Mapping

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from mlcroscopy.models.pl_module import MultiRegress
from mlcroscopy.utils.builders import make_dataset, make_model
from mlcroscopy.utils.constants import LOSS_OPTIONS
from mlcroscopy.utils.misc import set_seeds


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move a possibly nested model input to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _split_filename(split: str) -> str:
    """Allow either short split names (``train``) or explicit index files."""
    return split if Path(split).suffix else f"{split}.idx"


def adapt_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove the Lightning module's leading ``model.`` from weight keys."""
    return {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
    }


def _load_module(cfg: dict, checkpoint_path: Path, device: torch.device):
    model_cfg = cfg["model"]
    model_cfg["has_val"] = False
    model = make_model(cfg).to(torch.float32)
    loss = LOSS_OPTIONS[model_cfg["optim"].get("loss", "l2")]()
    module = MultiRegress(
        model,
        loss,
        model_cfg["optim"]["learning-rate"],
        device.type,
        hparams={"model": model_cfg, "task": cfg["task"]},
    )


    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = adapt_state_dict_keys(state_dict)
    # print(module.model.state_dict().keys())
    # raise Exception
    incompatible = module.model.load_state_dict(state_dict, strict=cfg["strict"])
    if not cfg["strict"] and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "Checkpoint loaded non-strictly: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return module.to(device).eval()


@torch.inference_mode()
def _evaluate(module, dataset, cfg: dict, device: torch.device) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"] or cfg["model"]["optim"]["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    predictions, targets = [], []
    for inputs, target in loader:
        output = module.model(_move_to_device(inputs, device))
        predictions.append(output.detach().cpu())
        targets.append(target.detach().cpu())

    if not predictions:
        raise ValueError("The requested split is empty")

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    errors = predictions - targets
    target_names = getattr(dataset, "targets", None)
    if not target_names or len(target_names) != targets.shape[-1]:
        target_names = [f"target_{i}" for i in range(targets.shape[-1])]

    result = {
        "samples": len(dataset),
        "overall": {
            "MAE": errors.abs().mean().item(),
            "MSE": errors.square().mean().item(),
            "RMSE": errors.square().mean().sqrt().item(),
        },
        "targets": {},
    }
    for i, name in enumerate(target_names):
        target_error = errors[:, i]
        result["targets"][name] = {
            "MAE": target_error.abs().mean().item(),
            "MSE": target_error.square().mean().item(),
            "RMSE": target_error.square().mean().sqrt().item(),
        }
    return result


def evaluate(cfg: dict) -> dict:
    set_seeds(cfg["seed"])
    requested_device = cfg["device"]
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    checkpoint_path = Path(to_absolute_path(cfg["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    module = _load_module(cfg, checkpoint_path, device)
    results = {}
    for split in cfg["splits"]:
        dataset = make_dataset(cfg, _split_filename(split))
        results[split] = _evaluate(module, dataset, cfg, device)
        print(f"{split}:\n{json.dumps(results[split], indent=2)}")

    if cfg.get("output_path"):
        output_path = Path(to_absolute_path(cfg["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Saved metrics to {output_path}")
    return results


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg: DictConfig):
    choices = HydraConfig.get().runtime.choices
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["model"]["name"] = choices["model"]
    cfg["task"]["name"] = choices["task"]
    evaluate(cfg)


if __name__ == "__main__":
    main()
