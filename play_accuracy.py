from sklearn.metrics import accuracy_score
import numpy as np


y_true = np.array([0,1,1,0])
y_pred = np.array([1,1,1,2])

acurracy = accuracy_score(y_true, y_pred)

print(acurracy)
