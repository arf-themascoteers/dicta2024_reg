import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

array = np.random.rand(5, 10)

plt.figure(figsize=(10, 6))
sns.heatmap(array, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Proportion of selected bands'})
for x in range(1, array.shape[1]):
    plt.axvline(x, color='white', lw=2)
plt.xlabel("Lower dimensional Size")
plt.ylabel("Spectral region (nm)")
#plt.yticks(ticks=np.arange(len(range_labels)) + 0.5, labels=range_labels, rotation=0)
#plt.xticks(ticks=np.arange(df.shape[1]) + 0.5, labels=[8, 16, 32, 64, 128, 256, 512], rotation=0)
plt.tight_layout()
plt.savefig("heatmap.png", dpi=600)
plt.show()
