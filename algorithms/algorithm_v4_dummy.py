from algorithm import Algorithm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import attn_handler


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
        sparse_weights = self.sparse(channel_weights)
        reweight_out = X * sparse_weights
        output = self.classnet(reweight_out)
        output = output.reshape(-1)
        return channel_weights, sparse_weights, output


class Algorithm_v4_dummy(Algorithm):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose):
        super().__init__(target_size, dataset, tag, reporter, verbose)
        self.criterion = torch.nn.MSELoss()
        self.zhangnet = ZhangNet(self.dataset.get_train_x().shape[1]).to(self.device)
        self.total_epoch = 10

        self.X_train = torch.tensor(self.dataset.get_train_x(), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.dataset.get_train_y(), dtype=torch.float32).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0
        for epoch in range(self.total_epoch):
            epoch = epoch
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, sparse_weights, y_hat = self.zhangnet(X)
                deciding_weights = channel_weights
                mean_weight, all_bands, selected_bands = attn_handler.get_indices(deciding_weights, self.target_size)
                self.set_all_indices(all_bands)
                self.set_selected_indices(selected_bands)
                self.set_weights(mean_weight)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                lambda_value = self.get_lambda(epoch+1)
                loss = mse_loss + lambda_value*l1_loss
                if batch_idx == 0 and epoch%10 == 0:
                    attn_handler.report_stats(self, channel_weights, sparse_weights, epoch, mse_loss, l1_loss.item(), lambda_value,loss)
                loss.backward()
                optimizer.step()

        print(self.get_name(),"selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))
        return self.zhangnet, self.selected_indices

    def l1_loss(self, channel_weights):
        return torch.norm(channel_weights, p=1) / torch.numel(channel_weights)

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch/self.total_epoch)



