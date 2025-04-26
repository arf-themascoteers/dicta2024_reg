from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_train_test_pair(train_x, test_x, train_y, test_y, scaler_y):
    evaluator_algorithm = get_metric_evaluator()
    evaluator_algorithm.fit(train_x, train_y)
    y_pred = evaluator_algorithm.predict(test_x)
    return calculate_metrics(test_y, y_pred, scaler_y)


def evaluate_dataset(dataset, transform):
    return evaluate_split(*dataset.get_a_fold(),dataset.scaler_y, transform)


def evaluate_split(train_x, test_x, train_y, test_y, scaler_y, transform=None):
    if transform is not None:
        train_x = transform.transform(train_x)
        test_x = transform.transform(test_x)
    return evaluate_train_test_pair(train_x, test_x, train_y, test_y, scaler_y)


def convert_to_numpy(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return t


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



def get_metric_evaluator():
    gowith = "sv"

    if gowith == "rf":
        return RandomForestRegressor()
    elif gowith == "sv":
        return SVR(C=10, epsilon=0.01, kernel='rbf', gamma='scale')
    else:
        return MLPRegressor(max_iter=2000)