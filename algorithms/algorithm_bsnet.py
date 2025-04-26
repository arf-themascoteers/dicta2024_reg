from algorithm import Algorithm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import attn_handler


class BSNetFC(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.bands = bands
        self.weighter = nn.Sequential(
            nn.BatchNorm1d(self.bands),
            nn.Linear(self.bands, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.channel_weight_layer = nn.Sequential(
            nn.Linear(128, self.bands),
            nn.Sigmoid()
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.bands, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.bands),
            nn.BatchNorm1d(self.bands),
            nn.Sigmoid()
        )

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        channel_weights = self.weighter(X)
        channel_weights = self.channel_weight_layer(channel_weights)
        reweight_out = X * channel_weights
        output = self.encoder(reweight_out)
        return channel_weights, output


class Algorithm_bsnet(Algorithm):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose):
        super().__init__(target_size, dataset, tag, reporter, verbose)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        x,y = self.dataset.get_train_x_y()
        self.bsnet = BSNetFC(x.shape[1]).to(self.device)
        self.X_train = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y, dtype=torch.int32).to(self.device)

    def get_selected_indices(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(self.bsnet.parameters(), lr=0.00002)
        X_train = self.X_train
        dataset = TensorDataset(X_train, X_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0
        for epoch in range(100):
            for batch_idx, (X, y) in enumerate(dataloader):
                if X.shape[0] == 1:
                    continue
                optimizer.zero_grad()
                channel_weights, y_hat = self.bsnet(X)
                deciding_weights = channel_weights
                mean_weight, all_bands, selected_bands = attn_handler.get_indices(deciding_weights, self.target_size)
                self.set_all_indices(all_bands)
                self.set_selected_indices(selected_bands)
                self.set_weights(mean_weight)

                mse_loss = self.criterion(y_hat, y)
                l1_loss = torch.norm(channel_weights, p=1)/torch.numel(channel_weights)
                loss = mse_loss + l1_loss * 0.01
                if batch_idx == 0 and epoch%10 == 0:
                    attn_handler.report_stats(self, channel_weights, channel_weights, epoch, mse_loss, l1_loss.item(), 0.01,loss)
                loss.backward()
                optimizer.step()
            print(f"Epoch={epoch} MSE={round(mse_loss.item(), 5)}, L1={round(l1_loss.item(), 5)}, LOSS={round(loss.item(), 5)}")

        print(self.get_name(),"selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))
        return self.bsnet, self.selected_indices
