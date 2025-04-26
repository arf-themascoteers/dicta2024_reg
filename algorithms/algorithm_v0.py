from algorithm import Algorithm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import csv
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
            nn.Conv1d(1,8,kernel_size=8, stride=4),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(8, 16, kernel_size=16, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(16, 32, kernel_size=8, stride=4),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Flatten(start_dim=1),
            nn.Linear(64,1)
        )
        self.sparse = Sparse()
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        channel_weights = self.weighter(X)
        sparse_weights = self.sparse(channel_weights)
        reweight_out = X * sparse_weights
        reweight_out = reweight_out.reshape(reweight_out.shape[0],1,reweight_out.shape[1])
        output = self.classnet(reweight_out)
        output = output.reshape(-1)
        return channel_weights, sparse_weights, output


class Algorithm_v0(Algorithm):
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
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0

        for epoch in range(self.total_epoch):
            self.epoch = epoch
            grad_norms = []
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
                if batch_idx == 0 and self.epoch%10 == 0:
                    attn_handler.report_stats(channel_weights, sparse_weights, epoch, mse_loss, l1_loss.item(), lambda_value,loss)
                loss.backward()
                grad_norm = torch.abs(self.zhangnet.weighter[2].weight.grad)
                grad_norms.append(grad_norm)
                optimizer.step()

            grad_norms = torch.cat(grad_norms, dim=0)
            mean_grad = torch.mean(grad_norms)
            with open('v0_grad_norm.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([mean_grad.item()])


        print(self.get_name(),"selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))
        return self.zhangnet, self.selected_indices

    def l1_loss(self, channel_weights):
        return torch.norm(channel_weights, p=1) / torch.numel(channel_weights)

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch/self.total_epoch)



