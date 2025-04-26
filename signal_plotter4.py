import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow

df = pd.read_csv("data/lucas_skipped_min.csv")
signal = df.iloc[10, 1:].values.reshape(-1, 1)
scaler = MinMaxScaler()
normalized_signal = scaler.fit_transform(signal).flatten()

x = list(range(len(normalized_signal)))

colors = np.zeros((len(x), 3))
for i in range(len(x)):
    if i < len(x) // 2:
        colors[i] = np.array([0, 2 * i / len(x), 1 - 2 * i / len(x)])
    else:
        colors[i] = np.array([2 * (i - len(x) // 2) / len(x), 1 - 2 * (i - len(x) // 2) / len(x), 0])

fig, ax = plt.subplots(figsize=(20, 2.5))

ax.scatter(x, normalized_signal, marker='o', c=colors, s=30)
ax.set_xlabel("Band", fontsize=20, fontweight='bold')
ax.set_ylabel("Intensity", fontsize=20, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])


plt.subplots_adjust(top=0.8)
plt.savefig("imp_base.png", transparent=True)
