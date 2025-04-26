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

ax.scatter(x, normalized_signal, marker='o', c=colors, s=10)
# ax.set_xlabel("Band", fontsize=20)
# ax.set_ylabel("Intensity", fontsize=20)
ax.set_xticks([])
ax.set_yticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

annotations = {10: "Band 1", 30: "Band 2", 50: "Band 3"}
highlighted_points = [10, 30, 50]

for point in highlighted_points:
    ax.scatter(point, normalized_signal[point], color=colors[point], s=100, edgecolor='black', zorder=5)

for point, label in annotations.items():
    ax.annotate(label, (point, normalized_signal[point]), textcoords="offset points", xytext=(0,20), ha='center', fontsize=18)

arrowprops_red = dict(facecolor='red', edgecolor='red', shrink=0.05, width=1, headwidth=5)
arrowprops_green = dict(facecolor='green', edgecolor='green', shrink=0.05, width=1, headwidth=5)

point = highlighted_points[0]
y = normalized_signal[point]+0.2
ax.annotate('', xy=((point-5)/len(x), y), xytext=((point-2)/len(x), y), xycoords='axes fraction', arrowprops=arrowprops_red)
ax.annotate('', xy=((point+9)/len(x), y), xytext=((point+6)/len(x), y), xycoords='axes fraction', arrowprops=arrowprops_green)

point = highlighted_points[1]
y = normalized_signal[point]+0.15
ax.annotate('', xy=((point-6)/len(x), y), xytext=((point-3)/len(x), y), xycoords='axes fraction', arrowprops=arrowprops_green)
ax.annotate('', xy=((point+7)/len(x), y), xytext=((point+4)/len(x), y), xycoords='axes fraction', arrowprops=arrowprops_red)


point = highlighted_points[2]
y = normalized_signal[point]+0.17
ax.annotate('', xy=((point-8)/len(x), y), xytext=((point-5)/len(x), y), xycoords='axes fraction', arrowprops=arrowprops_red)
ax.annotate('', xy=((point+6)/len(x), y), xytext=((point+3)/len(x), y), xycoords='axes fraction', arrowprops=arrowprops_green)

red_arrow = Line2D([0, 1], [0, 0], color="red", lw=2, label="Deterioration", marker=">", markersize=10, markerfacecolor="red", markeredgecolor="red")
green_arrow = Line2D([0, 1], [0, 0], color="green", lw=2, label="Improvement", marker=">", markersize=10, markerfacecolor="green", markeredgecolor="green")

ax.legend(handles=[red_arrow, green_arrow], loc="upper left", fontsize=18,bbox_to_anchor=(0.3, 0.3), ncols=2)

plt.subplots_adjust(top=0.8)
plt.savefig("imp_dummy.png", transparent=True)
