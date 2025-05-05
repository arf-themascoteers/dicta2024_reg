import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DSManager:
    def __init__(self, name):
        self.name = name
        dataset_path = f"data/{self.name}.csv"
        df = pd.read_csv(dataset_path)
        self.data = df.to_numpy()

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.data[:,0:-1] = self.scaler_X.fit_transform(self.data[:,0:-1])
        self.data[:,-1] = self.scaler_y.fit_transform(self.data[:,-1].reshape(-1,1)).ravel()

        self.data = df.to_numpy()
        print(f"{self.name}: Total bs train samples", len(self.data))

    def get_name(self):
        return self.name

    def get_train_data(self):
        return self.data

    def get_train_x_y(self):
        return self.get_train_x(), self.get_train_y()

    def get_train_x(self):
        return self.data[:,0:-1]

    def get_train_y(self):
        return self.data[:, -1]

    def get_k_folds(self):
        folds = 1
        for i in range(folds):
            seed = 40 + i
            yield self.get_a_fold(seed)

    def get_a_fold(self, seed=50):
        return train_test_split(self.data[:,0:-1], self.data[:,-1], test_size=0.95, random_state=seed)

    def __repr__(self):
        return self.get_name()

    @staticmethod
    def get_dataset_names():
        return [
            "lucas",
            "lucas_min",
        ]

