import json
from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from mlcroscopy import ScopyPredictor
from mlcroscopy.data.dataset import ImgPlusExpt, OnlyExpt
from mlcroscopy.utils.builders import make_transforms
from mlcroscopy.utils.misc import get_successive_crops

def get_avg_embed(modelObj, batch):

    micro_embed_avg = []
    xrf_embeds = []
    expt_inputs = []
    for inputs, _ in batch:
        micro_x, xrf_x, expt_x = inputs
        print("Micro", micro_x.shape)
        print("XRF", xrf_x.shape)
        print("Expt", expt_x.shape)
        micro_embed_avg.append(modelObj.model.get_micro_embed(micro_x))

    expt_inputs = expt_x.detach().cpu().numpy()
    micro_embed_avg = torch.cat(micro_embed_avg, dim=0)
    micro_embed_avg = torch.mean(micro_embed_avg, dim=0).detach().cpu().numpy()
    xrf_embeds = modelObj.model.get_xrf_embed(xrf_x).detach().cpu().numpy()

    return micro_embed_avg, xrf_embeds.reshape(-1), expt_inputs.reshape(-1)

if __name__ == '__main__':
    trained_models = {
        "Dino": "/network/scratch/d/divya.sharma/runs/ada/halberd_embed_full/last.ckpt",
        "Classic": "/network/scratch/d/divya.sharma/runs/ada/axe_embed_full/last.ckpt"
    }

    device = "cuda"
    data_path = "/network/scratch/d/divya.sharma/ada/data/gde_ada"
    split = "train"
    results_dir = Path("/network/scratch/d/divya.sharma/ada/MLcroscopy/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, ckpt in trained_models.items():

        tmpObj = ScopyPredictor(ckpt, device=device)
        tmpObj.model.eval()
        crop_size = tmpObj.task_params["transforms"]["micro"]["crop"]
        t = make_transforms(tmpObj.get_data_proc())
        data = ImgPlusExpt(root=data_path, split_fptr=f"{split}.idx", ytransform=False, **t)

        embeds = {
            "micro": [],
            "xrf": [],
            "expt": []
        }

        for sample_idx, sample in enumerate(data):
            img_batch = get_successive_crops(sample, crop_size, device)
            print(name)
            tmp = get_avg_embed(tmpObj, img_batch)
            embeds["micro"].append(tmp[0])
            embeds["xrf"].append(tmp[1])
            embeds["expt"].append(tmp[2])

        # pca_figs = {mod: plt.subplots() for mod in ["micro", "xrf", "expt"]}
        # tsne_figs = {mod: plt.subplots() for mod in ["micro", "xrf", "expt"]}
        # for mod, emb in embeds.items():
        #     pca = PCA(n_components=2)
        #     new_x = pca.fit_transform(emb)

        #     fig1, ax = pca_figs[mod]
        #     ax.scatter(new_x[:, 0], new_x[:, 1])
        #     ax.set_xlabel("PCA Component 1")
        #     ax.set_ylabel("PCA Component 2")
        #     fig1.suptitle(f"PCA model:{name} on {mod}")

        #     tsne = TSNE(n_components=2)
        #     new_x = tsne.fit_transform(np.array(emb))

        #     fig2, ax = tsne_figs[mod]
        #     ax.scatter(new_x[:, 0], new_x[:, 1])
        #     ax.set_xlabel("TSNE Component 1")
        #     ax.set_ylabel("TSNE Component 2")
        #     fig2.suptitle(f"TSNE model:{name} on {mod}")

        #     fig1.savefig(results_dir / f"{name}_pca_{mod}.png")
        #     fig2.savefig(results_dir / f"{name}_tsne_{mod}.png")



