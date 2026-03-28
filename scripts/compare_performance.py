import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

performance_eth = {
    "zeros": {"MAE": 0.15, "MSE": 0.03},
    "Mean": {"MAE": 0.07, "MSE": 0.007},
    "median": {"MAE": 0.063, "MSE": 0.007},
    "MLP": {"MAE": 0.032, "MSE": 0.002},
    "Best": {"MAE": 0.043, "MSE": 0.003}

}

df = pd.DataFrame(performance_eth).T
df["Predictor"] = df.index
df = df.reset_index(drop=True)
tidy = df.melt(id_vars="Predictor", var_name="Metric").rename(columns=str.title)
# print(tidy)

fig, ax = plt.subplots(figsize=(16, 8))
g = sns.barplot(tidy, x="Predictor", y="Value", hue="Metric", ax=ax)
ax.axhline(y=0.02, color='red', linestyle='--', linewidth=2, label='GP + Phys')
plt.legend()
plt.tight_layout()
fig.savefig("figures/baselines.png")

