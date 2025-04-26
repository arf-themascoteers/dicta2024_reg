import torch
import train_test_evaluator

def report_stats(channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda1, loss):
    mean_weight = channel_weights
    means_sparse = sparse_weights

    if len(mean_weight.shape) > 1:
        mean_weight = torch.mean(mean_weight, dim=0)
        means_sparse = torch.mean(means_sparse, dim=0)

    min_cw = torch.min(mean_weight).item()
    min_s = torch.min(means_sparse).item()
    max_cw = torch.max(mean_weight).item()
    max_s = torch.max(means_sparse).item()
    avg_cw = torch.mean(mean_weight).item()
    avg_s = torch.mean(means_sparse).item()

    l0_cw = torch.norm(mean_weight, p=0).item()
    l0_s = torch.norm(means_sparse, p=0).item()

    mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)

    r2, rmse, rpd = 0, 0, 0

    if self.verbose:
        r2, rmse, rpd = train_test_evaluator.evaluate_dataset(self.dataset, self)

    self.reporter.report_epoch(epoch, mse_loss, l1_loss, lambda1, loss,
                               r2, rmse, rpd,
                               min_cw, max_cw, avg_cw,
                               min_s, max_s, avg_s,
                               l0_cw, l0_s,
                               selected_bands, means_sparse)