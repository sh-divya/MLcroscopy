import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).parent.parent.parent / "data"

all_data = pd.read_csv(DATA_PATH / "triplet.csv")
y = all_data
# y = all_data.groupby("triplet")
out = {"eth": "FE (Eth)", "co": "FE (CO)"}

y_mean = y.mean()[["FE (Eth)", "FE (CO)"]]
y_std = y.std()[["FE (Eth)", "FE (CO)"]]
y_median = y.median()[["FE (Eth)", "FE (CO)"]]


xticks = y_mean.index
x = [2 * i for i in xticks]

# for o in out:
#     yerr = y_std[out[o]].values
#     plt.figure()
#     plt.bar(x, y_mean[out[o]], yerr=yerr, capsize=5, width=1.2)
#     plt.xticks(x, xticks)
#     fig = plt.gcf()
#     fig.set_size_inches(20, 10)
#     plt.title(f"{out[o]}_triplets")
#     plt.savefig(f"figures/triplet_{o}.png")



zeros = np.zeros_like(y[out['eth']])
mean_eth = np.full_like(y[out['eth']], y_mean[out['eth']])
mean_co = np.full_like(y[out['co']], y_mean[out['co']])
uniform_eth = np.random.uniform(y.min()[out['eth']], y.max()[out['eth']], y[out['eth']].shape)
uniform_co = np.random.uniform(y.min()[out['co']], y.max()[out['co']], y[out['eth']].shape)
normal = np.random.normal(y_mean[out['eth']], y_std[out['eth']], y[out['eth']].shape)
normal_co = np.random.normal(y_mean[out['co']], y_std[out['co']], y[out['co']].shape)
unit_normal = np.random.normal(0.5, 0.5, y[out['eth']].shape)
median_eth = np.full_like(y[out['eth']], y_median[out['eth']])
median_co = np.full_like(y[out['co']], y_median[out['co']])

targets_eth = {
    "zeros": zeros,
    "mean": mean_eth,
    "uniform": uniform_eth,
    "unit_normal": unit_normal,
    "median": median_eth
}

print("------Eth--------")
for t, v in targets_eth.items():
    mae = np.mean(np.abs(y[out['eth']] - v))
    mse = np.mean(np.square(y[out['eth']] - v))
    print(f"{t}_mae", mae)
    print(f"{t}_mse", mse)


targets_co = {
    "zeros": zeros,
    "mean": mean_co,
    "uniform": uniform_co,
    "unit_normal": unit_normal,
    "median": median_co
}

print("------CO--------")
for t, v in targets_co.items():
    mae = np.mean(np.abs(y[out['eth']] - v))
    mse = np.mean(np.square(y[out['eth']] - v))
    print(f"{t}_mae", mae)
    print(f"{t}_mse", mse)