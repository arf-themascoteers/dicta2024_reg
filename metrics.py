class Metrics:
    def __init__(self,time, r2, rmse, rpd, selected_features, selected_weights):
        self.time = time
        self.r2 = r2
        self.rmse = rmse
        self.rpd = rpd
        self.selected_features = selected_features
        self.selected_weights = selected_weights

