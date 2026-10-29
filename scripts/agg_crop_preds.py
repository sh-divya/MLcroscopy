import json
from pathlib import Path

import torch
import numpy as np

from mlcroscopy import ScopyPredictor
from mlcroscopy.data.dataset import ImgPlusExpt, OnlyExpt
from mlcroscopy.utils.builders import make_transforms
from mlcroscopy.utils.misc import get_successive_crops


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
