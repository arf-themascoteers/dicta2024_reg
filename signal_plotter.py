import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import random

random_numbers = np.random.uniform(0, 0.009, 73)
s = "0.7240707278251648|0.5384008288383484|0.37546420097351074|0.4389619529247284|0.5712578296661377|0.47818446159362793|0.43329864740371704|0.30297690629959106|0.25595995783805847|0.5037322044372559|0.5425720810890198|0.6274012327194214|0.4166421592235565|0.3911632299423218|0.11710941791534424|0.01257567759603262|0.25536853075027466|0.3540898263454437|0.6634802222251892|0.3597012162208557|0.36382630467414856|0.009531293995678425|0.5005761981010437|0.6245270371437073|0.23290863633155823|0.19399063289165497|0.10882052034139633|0.7594400644302368|0.29831603169441223|0.6214621663093567"
s = s.split("|")
s = [float(i) for i in s]
print(min(s))

rn  = random_numbers.tolist()
s = s+ rn
random.shuffle(s)
print(len(s))

df = pd.read_csv("data/paviaU.csv")

signal = df.iloc[10,:-1]
x = list(range(len(signal)))
print(len(signal))

signal2 = np.multiply(signal, s).tolist()
print(signal2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
plt.subplots_adjust(top=0.6, bottom=0.4)

sc1 = ax1.scatter(x, signal, c=signal, cmap='viridis', s=2)
sc2 = ax2.scatter(x, signal2, c=signal2, cmap='viridis', s=2)
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])

ax1.set_xlabel('Band')
ax1.set_ylabel('Original Reflactance')
ax2.set_xlabel('Band')
ax2.set_ylabel('Recalibrated Reflactance')

#fig.colorbar(sc1, ax=ax1)
#fig.colorbar(sc2, ax=ax2)

ax1.set_title('Original data')
ax2.set_title('Recalibrated data')

plt.show()