from sklearn.svm import SVR
from ds_manager import DSManager
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def evaluate_train_test_pair(alg, train_x, test_x, train_y, test_y, scaler_y):
    alg.fit(train_x, train_y)
    y_pred = alg.predict(test_x)
    return calculate_metrics(test_y, y_pred, scaler_y)


def evaluate_dataset(alg, dataset):
    return evaluate_train_test_pair(alg,*dataset.get_a_fold(),dataset.scaler_y)


def calculate_3_metrics(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    std_dev = np.std(y_test, ddof=1)
    rpd = std_dev / rmse
    return r2, rmse, rpd


def calculate_metrics(y_test, y_pred, scaler_y):
    r2, rmse, rpd = calculate_3_metrics(y_test, y_pred)
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    r2_o, rmse_o, rpd_o = calculate_3_metrics(y_test_original, y_pred_original)

    return r2, rmse_o, rpd_o

alg = SVR(C=10, epsilon=0.01, kernel='rbf', gamma='scale')
dataset = DSManager(name="lucas")
r2, rmse, rpd = evaluate_dataset(alg,dataset)
print(r2, rmse, rpd)
