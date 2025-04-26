import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 500)
sigmoid = 1 / (1 + np.exp(-x))
derivative = sigmoid * (1 - sigmoid)

plt.plot(x, sigmoid, label='Sigmoid', color="#00354A")
plt.plot(x, derivative, label='Derivative', color="#97BF0D")
plt.legend(fontsize=21,frameon=False)
plt.axis("off")
plt.xticks([])
plt.yticks([])
plt.show()
