import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22


def get_l0(file):
    df = pd.read_csv(file)
    return df['l0_s'].tolist()

y = []


y.append(get_l0("lucas_results/v9_lucas_v9_lucas_512.csv"))

labels = [
"LUCAS",

]


x = list(range(len(y[0])))

plt.figure(figsize=(10,6))

for i in range(len(y)):
    plt.plot(x, y[i], label=labels[i])

plt.axhline(y=512, color='black', linestyle='--', linewidth=1, label="Target maximum size")
plt.xlabel('Epoch')
plt.ylabel('$k_{active}$')
plt.ylim([0,5000])
legend = plt.legend(    bbox_to_anchor=(0.4, 1.2),loc='center',frameon=True,ncol=2,)
plt.tight_layout()
plt.savefig("l0_b_post.png", bbox_inches='tight', pad_inches=0.1)

plt.show()

