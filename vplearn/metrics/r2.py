from vplearn.metrics import _Metric

import numpy as np

class R2(_Metric):
    def __init__(self, ground_truth: np.ndarray, predict: np.ndarray):
        super().__init__()
        self._check_fit_input_format(ground_truth, predict)

        self.ground_truth = ground_truth
        self.predict = predict
        self.r2 = self._compute()

    def _compute(self) -> float:
        ss_res = np.sum((self.ground_truth - self.predict) ** 2)
        ss_tot = np.sum((self.ground_truth - np.mean(self.ground_truth)) ** 2)
        return 1 - ss_res / ss_tot

    def __float__(self) -> float:
        return float(self.r2)

    def __repr__(self) -> str:
        return f"R2({self.r2:.6f})"
    