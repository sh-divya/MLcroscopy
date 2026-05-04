import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

TARGET_LABELS = {
    0: "FE (Eth)",
    1: "FE (CO)",
}


def load_results(results_path):
    with open(results_path, "r") as f:
        return json.load(f)


def get_axis_limits(records, idx):
    true_vals = [record[f"true_{idx}"] for record in records]
    avg_preds = [record[f"avg_pred_{idx}"] for record in records]
    crop_preds = []
    for record in records:
        crop_preds.extend(record[f"pred_values_{idx}"])

    all_vals = np.asarray(true_vals + avg_preds + crop_preds, dtype=float)
    vmin = all_vals.min()
    vmax = all_vals.max()
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.05
    return vmin - pad, vmax + pad


def add_parity_line(ax, lims):
    ax.plot(lims, lims, linestyle="--", color="black", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def get_r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if np.isclose(ss_tot, 0.0):
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def get_gaussian_nll(y_true, y_pred, y_std, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    y_var = np.maximum(y_std ** 2, eps)
    nll = 0.5 * (np.log(2 * np.pi * y_var) + ((y_true - y_pred) ** 2) / y_var)
    return float(np.mean(nll))


def make_crop_level_plot(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.get_cmap("tab20", max(len(records), 1))

    for idx, ax in enumerate(axes):
        for sample_num, record in enumerate(records):
            true_val = record[f"true_{idx}"]
            preds = record[f"pred_values_{idx}"]
            x_vals = [true_val] * len(preds)
            y_vals = preds

            ax.scatter(
                x_vals,
                y_vals,
                alpha=0.7,
                s=20,
                color=cmap(sample_num),
                label=f"idx {record['sample_idx']}",
            )

        lims = get_axis_limits(records, idx)
        add_parity_line(ax, lims)
        ax.set_title(TARGET_LABELS[idx])
        ax.set_xlabel(f"True {TARGET_LABELS[idx]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[idx]}")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Sample Index",
        )

    fig.suptitle("Parity Plot by Crop")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_errorbar_plot(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for idx, ax in enumerate(axes):
        true_vals = []
        avg_preds = []
        pred_stds = []

        for record in records:
            preds = np.asarray(record[f"pred_values_{idx}"], dtype=float)
            true_vals.append(record[f"true_{idx}"])
            avg_preds.append(record[f"avg_pred_{idx}"])
            pred_stds.append(preds.std())

        lims = get_axis_limits(records, idx)
        ax.errorbar(
            true_vals,
            avg_preds,
            yerr=pred_stds,
            fmt="o",
            alpha=0.8,
            capsize=3,
        )
        add_parity_line(ax, lims)
        r2 = get_r2_score(true_vals, avg_preds)
        # nll = get_gaussian_nll(true_vals, avg_preds, pred_stds)
        ax.set_title(TARGET_LABELS[idx])
        ax.set_xlabel(f"True {TARGET_LABELS[idx]}")
        ax.set_ylabel(f"Average Predicted {TARGET_LABELS[idx]}")
        ax.text(
            0.05,
            0.95,
            f"$R^2$ = {r2:.3f}",
            # f"$R^2$ = {r2:.3f}\nNLL = {nll:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.suptitle("Parity Plot with Crop Error Bars")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option(
    "--results-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the aggregated crop prediction JSON file.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save parity plots. Defaults to the JSON file's parent.",
)
def run(results_path, output_dir):
    records = load_results(results_path)
    if output_dir is None:
        output_dir = results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # stem = results_path.stem
    stem = "agg_halberd_train"
    crop_plot_path = output_dir / f"{stem}_parity_crops.png"
    errorbar_plot_path = output_dir / f"{stem}_parity_errorbars.png"

    make_crop_level_plot(records, crop_plot_path)
    make_errorbar_plot(records, errorbar_plot_path)

    print(f"Saved crop-level parity plot to {crop_plot_path}")
    print(f"Saved error-bar parity plot to {errorbar_plot_path}")


if __name__ == "__main__":
    run()
