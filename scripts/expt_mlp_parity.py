import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mlcroscopy import ScopyPredictor
from mlcroscopy.data.dataset import OnlyExpt
from mlcroscopy.utils.builders import make_transforms


TARGET_LABELS = {
    0: "FE (Eth)",
    1: "FE (CO)",
}


def get_r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if np.isclose(ss_tot, 0.0):
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def get_axis_limits(records, idx):
    vals = []
    for record in records:
        vals.append(record[f"true_{idx}"])
        vals.append(record[f"pred_{idx}"])

    vals = np.asarray(vals, dtype=float)
    vmin = vals.min()
    vmax = vals.max()
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.05
    return vmin - pad, vmax + pad


def make_parity_plot(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for idx, ax in enumerate(axes):
        true_vals = [record[f"true_{idx}"] for record in records]
        pred_vals = [record[f"pred_{idx}"] for record in records]
        lims = get_axis_limits(records, idx)
        r2 = get_r2_score(true_vals, pred_vals)

        ax.scatter(true_vals, pred_vals, alpha=0.7, s=28)
        ax.plot(lims, lims, linestyle="--", color="black", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(TARGET_LABELS[idx])
        ax.set_xlabel(f"True {TARGET_LABELS[idx]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[idx]}")
        ax.text(
            0.05,
            0.95,
            f"$R^2$ = {r2:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.suptitle("ExptMLP Parity Plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ckpt_path = "/network/scratch/d/divya.sharma/runs/ada/lf_old_final/last.ckpt"
    device = "cuda"
    data_path = "/network/scratch/d/divya.sharma/ada/data"
    split = "train"

    results_dir = Path("/network/scratch/d/divya.sharma/ada/MLcroscopy/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = results_dir / f"expt_mlp_preds_{split}.json"
    plot_path = Path("/network/scratch/d/divya.sharma/ada/MLcroscopy/figures")
    plot_path = plot_path / f"expt_mlp_parity_{split}.png"

    predictor = ScopyPredictor(ckpt_path, device=device)
    predictor.model.eval()

    transforms = make_transforms(predictor.get_data_proc())
    data = OnlyExpt(
        root=data_path,
        split_fptr=f"{split}.idx",
        ytransform=False,
        **transforms,
    )

    rows = []
    for sample_idx, sample in enumerate(data):
        x, y = sample
        x = x.to(device, dtype=torch.float16).unsqueeze(0)
        y = y.to(device, dtype=torch.float16)

        pred = predictor(x).detach().cpu().numpy().reshape(-1)
        true = y.detach().cpu().numpy().reshape(-1)

        rows.append(
            {
                "sample_idx": sample_idx,
                "true_0": float(true[0]),
                "true_1": float(true[1]),
                "pred_0": float(pred[0]),
                "pred_1": float(pred[1]),
            }
        )

    with open(results_json_path, "w") as f:
        json.dump(rows, f, indent=2)

    make_parity_plot(rows, plot_path)
    print(f"Saved predictions to {results_json_path}")
    print(f"Saved parity plot to {plot_path}")
