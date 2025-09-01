import umap
import torch
import click
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import os
import sys
from pathlib import Path, PosixPath

MLPath = Path(__file__).parent.parent
sys.path.append(str(MLPath))
os.environ["TORCH_HOME"] = str(MLPath.parent / "checkpoints")

from mlcroscopy.data import embeddings

def reduced_dist(data, source, model, size_handler):
    img_path = MLPath.parent / "figures"

    umap_red = umap.UMAP()
    data_umap = umap_red.fit_transform(data)

    tsne_red = TSNE(n_components=2, perplexity=5, init="random", learning_rate="auto")
    data_tsne = tsne_red.fit_transform(data)

    plt.scatter(data_umap[:, 0], data_umap[:, 1], label="UMAP")
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], label="TSNE")

    plt.xlabel("Dim 0")
    plt.ylabel("Dim 1")

    plt.title(f"Reduced Dimensions of {source} images")
    plt.legend()

    plt.savefig(str(img_path / f"{source}_{model}_{size_handler}_reduced.png"))



def gen_embeds(data_path, source, model, loader, size_handler):
    exp_path = data_path / "Characterization_data.xlsx"
    model_name, model = list(model.items())[0]
    write_path = data_path / source /f"{model_name}_{size_handler}_embeds.csv"
    if write_path.is_file():
        embeds = pd.read_csv(write_path, index_col=0).values
        y = pd.read_excel(exp_path, index_col=0, skiprows=[1])["FE (CO)"].values
    else:
        model.eval()
        for x, y in loader:
            with torch.no_grad():
                if isinstance(x, dict):
                    embeds = []
                    for k in x:
                        partial = model(x[k]).reshape(x[k].shape[0], -1).numpy()
                        embeds.append(partial)
                    embeds = np.concatenate(embeds, axis=-1)
                else:
                    embeds = model(x).reshape([x.shape[0], -1]).numpy()
        df = pd.DataFrame(embeds)
        df.to_csv(write_path, index=True)

    return embeds, y

@click.command()
@click.option("--data_path")
@click.option("--source")
@click.option("--model_name")
@click.option("--size_handler")
def run(data_path, source, model_name, size_handler):
    if isinstance(data_path, PosixPath):
        pass
    else:
        data_path = Path(data_path)
    model, loader = embeddings.pretrained_model_loader(
        data_path, source, model_name, size_handler
    )
    model = {model_name: model}
    gen_embeds(data_path, source, model, loader, size_handler)


if __name__ == "__main__":

    # temp = dataset.XRF("/network/scratch/d/divya.sharma/ada/data", None)
    # print(temp[0])
    data_path = "/network/scratch/d/divya.sharma/ada/data"
    source = "XRF"
    model_name = "Inception"
    size_handler = "downscale"

    model, loader = embeddings.pretrained_model_loader(
        data_path,
        source,
        model_name,
        size_handler,
    )

    # for item, y in loader:
    #     print(type(item))
    #     raise Exception


    model = {model_name: model}
    embeds, expt = gen_embeds(Path(data_path), source, model, loader, size_handler)
    # reduced_dist(embeds, source, model_name, size_handler)

