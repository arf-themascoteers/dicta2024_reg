from algorithm import Algorithm
import torch
import torch.nn as nn
import math
import attn_report


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 0.1

    def forward(self, X):
        X = torch.where(X < self.k, 0, X)
        return X


class ZhangNet(nn.Module):
    def __init__(self, bands):
        super().__init__()

        self.bands = bands
        self.weighter = nn.Sequential(
            nn.Linear(self.bands, 512),
            nn.ReLU(),
            nn.Linear(512, self.bands),
            nn.Sigmoid()
        )
        self.classnet = nn.Sequential(
            nn.Linear(self.bands, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
        )
        self.sparse = Sparse()
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        channel_weights = self.weighter(X)
        channel_weights = torch.mean(channel_weights, dim=0)
        sparse_weights = self.sparse(channel_weights)
        reweight_out = X * sparse_weights
        output = self.classnet(reweight_out)
        output = output.reshape(-1)
        return channel_weights, sparse_weights, output


class Algorithm_v2(Algorithm):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose):
        super().__init__(target_size, dataset, tag, reporter, verbose)
        self.criterion = torch.nn.MSELoss()
        self.zhangnet = ZhangNet(self.dataset.get_train_x().shape[1]).to(self.device)
        self.total_epoch = 500
        self.epoch = -1
        self.X_train = torch.tensor(self.dataset.get_train_x(), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.dataset.get_train_y(), dtype=torch.float32).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0

        for epoch in range(self.total_epoch):
            optimizer.zero_grad()
            channel_weights, sparse_weights, y_hat = self.zhangnet(self.X_train)
            deciding_weights = channel_weights
            mean_weight, all_bands, selected_bands = self.get_indices(deciding_weights)
            self.set_all_indices(all_bands)
            self.set_selected_indices(selected_bands)
            self.set_weights(mean_weight)
            mse_loss = self.criterion(y_hat, self.y_train)
            l1_loss = self.l1_loss(channel_weights)
            lambda_value = self.get_lambda(epoch+1)
            loss = mse_loss + lambda_value*l1_loss
            if self.epoch%10 == 0:
                attn_report.report_stats(channel_weights, sparse_weights, epoch, mse_loss, l1_loss.item(), lambda_value,loss)
            loss.backward()
            optimizer.step()

        print(self.get_name(),"selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))
        return self.zhangnet, self.selected_indices

    def get_indices(self, deciding_weights):
        mean_weights = deciding_weights
        if len(mean_weights.shape) > 1:
            mean_weights = torch.mean(mean_weights, dim=0)

        corrected_weights = mean_weights
        if torch.any(corrected_weights < 0):
            corrected_weights = torch.abs(corrected_weights)

        band_indx = (torch.argsort(corrected_weights, descending=True)).tolist()
        return mean_weights, band_indx, band_indx[: self.target_size]

    def l1_loss(self, channel_weights):
        return torch.norm(channel_weights, p=1) / torch.numel(channel_weights)

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch/self.total_epoch)



