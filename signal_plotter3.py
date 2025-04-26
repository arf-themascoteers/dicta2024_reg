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

fig, ax = plt.subplots(figsize=(10, 2.5))

ax.scatter(x, normalized_signal, marker='o', c=colors, s=40)
ax.set_xlabel("Band", fontsize=25, fontweight='bold')
ax.set_ylabel("Intensity", fontsize=22, fontweight='bold', labelpad=15)
ax.set_xticks([])
ax.set_yticks([])

ax.spines['top'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)

plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust top and bottom padding
plt.savefig("imp_base.png", transparent=True)
plt.show()
