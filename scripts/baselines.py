import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).parent.parent.parent / "data"

all_data = pd.read_csv(DATA_PATH / "triplet.csv")
y = all_data[["FE (Eth)", "FE (CO)"]].values
y_eth = y[:, 0]
y_co = y[:, 1]

plt.hist(y_eth, 20, alpha=0.5)
plt.hist(y_co, 20, alpha=0.5)
plt.savefig("figures/hist.png")
raise Exception
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y_median = np.median(y)

zeros = np.zeros_like(y)
mean = np.full_like(y, y_mean)
uniform = np.random.uniform(y.min(), y.max(), y.shape)
normal = np.random.normal(y_mean, y_std, y.shape)
unit_normal = np.random.normal(0.5, 0.5, y.shape)
median = np.full_like(y, y_median)

targets = {
    "zeros": zeros,
    "mean": mean,
    "uniform": uniform,
    "unit_normal": unit_normal,
    "median": median
}

for t, v in targets.items():
    mae = np.mean(np.abs(y - v))
    mse = np.mean(np.square(y - v))
    print(f"{t}_mae", mae)
    print(f"{t}_mse", mse)