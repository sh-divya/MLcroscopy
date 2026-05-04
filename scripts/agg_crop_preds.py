import json
from pathlib import Path

import torch
import numpy as np

from mlcroscopy import ScopyPredictor
from mlcroscopy.data.dataset import ImgPlusExpt, OnlyExpt
from mlcroscopy.utils.builders import make_transforms


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


def get_avg_pred(model, batch):
    predy = []
    truey = []

    for inputs, y in batch:
        pred = model(inputs).detach().cpu().numpy().reshape(-1)
        true = y.detach().cpu().numpy().reshape(-1)

        predy.append(pred)
        truey.append(true)

    predy = np.stack(predy, axis=0)
    truey = np.stack(truey, axis=0)
    avg_pred = predy.mean(axis=0)

    return {
        "pred_values": predy,
        "true_values": truey,
        "avg_pred": avg_pred,
    }


if __name__ == "__main__":
    ckpt_path = "/network/scratch/d/divya.sharma/runs/ada/halberd_final/last.ckpt"
    device = "cuda"
    data_path = "/network/scratch/d/divya.sharma/ada/data"
    split = "train"
    results_dir = Path("/network/scratch/d/divya.sharma/ada/MLcroscopy/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"agg_crop_preds_{split}.json"


    tmpObj = ScopyPredictor(ckpt_path, device=device)
    tmpObj.model.eval()
    crop_size = tmpObj.task_params["transforms"]["micro"]["crop"]
    t = make_transforms(tmpObj.get_data_proc())
    data = ImgPlusExpt(root=data_path, split_fptr=f"{split}.idx", ytransform=False, **t)

    rows = []
    for sample_idx, sample in enumerate(data):
        img_batch = get_successive_crops(sample, crop_size, device)
        pred_info = get_avg_pred(tmpObj, img_batch)
        true_values = pred_info["true_values"][0].tolist()
        pred_values = pred_info["pred_values"].tolist()
        avg_pred = pred_info["avg_pred"].tolist()

        rows.append(
            {
                "sample_idx": sample_idx,
                "true_0": true_values[0],
                "true_1": true_values[1],
                "pred_values_0": [pred[0] for pred in pred_values],
                "pred_values_1": [pred[1] for pred in pred_values],
                "avg_pred_0": avg_pred[0],
                "avg_pred_1": avg_pred[1],
            }
        )

    with open(results_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved aggregated crop predictions to {results_path}")
