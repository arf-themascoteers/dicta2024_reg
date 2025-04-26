import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

paths = {
    '8_7_weights/v0_weight/v0_weight_v0_weight_indian_pines_5_weights.csv' : "BS-Net-Classifier [12]",
    '8_7_weights/v2_weight/v2_weight_v2_weight_indian_pines_5_weights.csv' : "V2 (aggregated weighting)",
}

fig, axes = plt.subplots(1, 2, figsize=(20, 3))

for path in paths:
    data = pd.read_csv(path).values[:, 1:]
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    cv = stds / means

    axes[0].plot(cv, linestyle='-', label=paths[path])
    axes[1].plot(means, linestyle='-', label=paths[path])

axes[0].set_xlabel('Training Interation',fontsize=16)
axes[0].set_ylabel('CV',fontsize=16)
axes[0].set_title('CV of weights across the samples',fontsize=16)
axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes[0].legend(fontsize=13)

axes[1].set_xlabel('Training Interation',fontsize=16)
axes[1].set_ylabel('Mean weight',fontsize=16)
axes[1].set_title('Mean weight across the samples',fontsize=16)
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes[1].legend(fontsize=13)
plt.subplots_adjust(bottom=0.2)
os.makedirs("stored_figs", exist_ok=True)
plt.savefig('stored_figs/weightplot2.png')