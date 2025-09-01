import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from matplotlib import pyplot as plt

# Adapted from https://github.com/igrega348/CO2-catalysis

BASE_PATH = Path(__file__).parent.parent


def load_data(fptr="Characterization_data.xlsx", data_path=None):
    if data_path:
        data_path = Path(data_path)
    else:
        data_path = BASE_PATH.parent / "data"

    file_path = data_path / fptr
    df = pd.read_excel(file_path, skiprows=[1], index_col=0)
    df = df[
        [
            "AgCu Ratio",
            "Naf vol (ul)",
            "Sust vol (ul)",
            "Catalyst mass loading",
            "FE (Eth)",
            "FE (CO)",
        ]
    ]
    df = df.sort_values(by=["AgCu Ratio", "Naf vol (ul)"])
    df = df.dropna()
    df["FE (CO)"] = df["FE (CO)"] / 100
    df["FE (Eth)"] = df["FE (Eth)"] / 100

    dens_Ag = 10490  # kg/m^3
    dens_Cu = 8935  # kg/m^3
    dens_avg = (1 - df["AgCu Ratio"]) * dens_Cu + df["AgCu Ratio"] * dens_Ag
    mass = df["Catalyst mass loading"] * 1e-6  # kg
    area = 1.85**2  # cm^2
    A = area * 1e-4  # m^2
    thickness = (mass / dens_avg) / A  # m
    df.insert(3, column="Zero_eps_thickness", value=thickness)

    df["triplet"] = np.arange(len(df)) // 3
    gen = np.random.default_rng(2)
    order = gen.permutation(30)
    new_df = pd.DataFrame()
    for i in order:
        new_df = pd.concat([new_df, df[df["triplet"] == i]])
    new_df.reset_index(drop=True, inplace=True)
    df = new_df
    df.to_csv(data_path / "triplet.csv", index=False)
    triplet_df = df.groupby("triplet").mean()
    triplet_df = triplet_df.reset_index()


    triplet_df["Eth_bins"] = pd.cut(triplet_df["FE (Eth)"], bins=10, labels=False)
    triplet_df["CO_bins"] = pd.cut(triplet_df["FE (CO)"], bins=10, labels=False)

    # print(triplet_df.iloc[:, -2:])
    x_train, y_train, x_test, y_test = iterative_train_test_split(
        triplet_df.iloc[:, :-2].values, triplet_df.iloc[:, -2:].values, test_size=0.33
    )

    x_train_idx = x_train[:, 0]
    x_test_idx = x_test[:, 0]

    # print(df)
    x_train = df[df['triplet'].isin(x_train_idx)]
    x_test = df[df['triplet'].isin(x_test_idx)]

    # raise Exception
    with open(data_path / "train.idx", "w+") as fobj:
        fobj.write("\n".join([str(i) for i in x_train.index]))

    with open(data_path / "test.idx", "w+") as fobj:
        fobj.write("\n".join([str(i) for i in x_test.index]))

    fig, axs = plt.subplots(2)
    fig.suptitle("Train/ Test split of triplets")
    axs[0].hist(x_train["FE (Eth)"], bins=10, label="train", density=True)
    axs[0].hist(x_test["FE (Eth)"], bins=10, label="test", density=True)
    axs[0].set_xlabel("FE (Eth)")
    axs[1].hist(x_train["FE (CO)"], bins=10, density=True)
    axs[1].hist(x_test["FE (CO)"], bins=10, density=True)
    axs[1].set_xlabel("FE (CO)")
    fig.legend()
    fig.tight_layout()
    fig.savefig(data_path / "split_hist.png")
    

if __name__ == "__main__":
    load_data()
