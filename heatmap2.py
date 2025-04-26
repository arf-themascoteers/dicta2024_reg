import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

df = pd.read_csv("back_results/v12_ds_v12_ds_indian_pines_30.csv")
array = df.filter(regex='^weight_').to_numpy()

array[array>0]=1

for i in range(0,len(array)):
    print(np.linalg.norm(array[i], ord=0))

plt.figure(figsize=(5, 3))
sns.heatmap(array, cmap=['black', 'white'], cbar=False)
plt.xlabel('Band')
plt.ylabel('Training iteration')

legend_elements = [
    Patch(facecolor='white', edgecolor='black', label='Nonzero weight'),
    Patch(facecolor='black', edgecolor='black', label='Zero weight')
]
plt.legend(
    handles=legend_elements,
    bbox_to_anchor=(0.5, 1.05),
    loc='center',
    frameon=False,
    ncol=2,
)
step = 10
plt.yticks(
    ticks=np.arange(0, array.shape[0], step),
    labels=(np.arange(0, array.shape[0], step) * 10)
)
plt.tight_layout()
plt.savefig("heatmap2.png")
plt.show()