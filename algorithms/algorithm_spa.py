from algorithm import Algorithm
from auswahl import SPA, VIP


class Algorithm_spa(Algorithm):
    def __init__(self, target_size:int, dataset, tag, reporter, verbose):
        super().__init__(target_size, dataset, tag, reporter, verbose)

    def get_selected_indices(self):
        vip = VIP()
        selector = SPA(n_features_to_select=self.target_size)
        vip.fit(self.dataset.get_train_x(), self.dataset.get_train_y())
        mask = vip.vips_ > 0.3
        selector.fit(self.dataset.get_train_x(), self.dataset.get_train_y(), mask=mask)
        feature_ranking = selector.get_support(indices=True)
        self.set_selected_indices(feature_ranking)
        return self, feature_ranking

    def is_cacheable(self):
        return False