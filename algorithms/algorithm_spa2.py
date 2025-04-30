import numpy as np
from numba import njit
from algorithm import Algorithm

@njit
def fast_spa(X, num_bands, mask, verbose=False):
    X = X[:, mask]
    X = np.ascontiguousarray(X)
    n_samples, n_bands = X.shape
    selected = []
    proj = np.zeros(n_samples)

    for i in range(num_bands):
        norms = np.empty(n_bands)
        for j in range(n_bands):
            if j in selected:
                norms[j] = -1
                continue
            col = X[:, j]
            norms[j] = np.linalg.norm(col - proj)
        idx = np.argmax(norms)
        selected.append(idx)
        v = X[:, idx]
        v = np.ascontiguousarray(v)
        proj += (v @ proj) / (v @ v + 1e-8) * v
    return np.array(selected)


def compute_vip_scores(X, y):
    coef = np.abs(np.corrcoef(X, y, rowvar=False)[-1, :-1])
    return coef / np.max(coef)


class Algorithm_spa2(Algorithm):
    def __init__(self, target_size: int, dataset, tag, reporter, verbose):
        super().__init__(target_size, dataset, tag, reporter, verbose)

    def get_selected_indices(self):
        X = self.dataset.get_train_x()
        y = self.dataset.get_train_y()

        vip_scores = compute_vip_scores(X, y)
        mask = vip_scores > 0.3

        if self.verbose:
            print(f"Running SPA on {np.sum(mask)} features...")

        selected = fast_spa(X, self.target_size, mask, verbose=self.verbose)

        if self.verbose:
            print(f"Selected indices: {selected}")

        self.set_selected_indices(selected)
        return self, selected

    def is_cacheable(self):
        return False
