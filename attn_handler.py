import torch
import train_test_evaluator


def get_indices(deciding_weights, target_size):
    mean_weights = deciding_weights
    if len(mean_weights.shape) > 1:
        mean_weights = torch.mean(mean_weights, dim=0)

    corrected_weights = mean_weights
    if torch.any(corrected_weights < 0):
        corrected_weights = torch.abs(corrected_weights)

    band_indx = (torch.argsort(corrected_weights, descending=True)).tolist()
    return mean_weights, band_indx, band_indx[: target_size]

def report_stats(model,channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda1, loss):
    if not model.verbose:
        return

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

    mean_weight, all_bands, selected_bands = get_indices(channel_weights, model.target_size)
    r2, rmse, rpd = train_test_evaluator.evaluate_dataset(model.dataset, model)
    model.reporter.report_epoch(epoch, mse_loss, l1_loss, lambda1, loss,
                               r2, rmse, rpd,
                               min_cw, max_cw, avg_cw,
                               min_s, max_s, avg_s,
                               l0_cw, l0_s,
                               selected_bands, means_sparse)

    PRINT = True
    if PRINT:
        m = [epoch, mse_loss, l1_loss, lambda1, loss, r2, rmse, rpd, min_cw, max_cw, avg_cw, min_s, max_s, avg_s, l0_cw, l0_s]
        names = ['epoch', 'mse_loss', 'l1_loss', 'lambda1', 'loss', 'r2', 'rmse', 'rpd', 'min_cw', 'max_cw',
                 'avg_cw', 'min_s', 'max_s', 'avg_s', 'l0_cw', 'l0_s']
        m = [round(x, 3) if isinstance(x, float) else x for x in m]
        print(''.join(name.ljust(10) for name in names))
        print(''.join(str(v).ljust(10) for v in m))
