import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

performance = {
    "zeros": {"MAE": 0.14, "MSE": 0.04},
    "Mean": {"MAE": 0.1, "MSE": 0.02},
    "median": {"MAE": 0.1, "MSE": 0.02},
    "MLP": {"MAE": 0.09, "MSE": 0.02},
    "High-fidelity": {"MAE": 0.09, "MSE": 0.01}
}

df = pd.DataFrame(performance).T
df["Predictor"] = df.index
df = df.reset_index(drop=True)
tidy = df.melt(id_vars="Predictor", var_name="Metric").rename(columns=str.title)
# print(tidy)

fig, ax = plt.subplots(figsize=(10, 10))
g = sns.barplot(tidy, x="Predictor", y="Value", hue="Metric", ax=ax)
fig.savefig("figures/tmp.png")

