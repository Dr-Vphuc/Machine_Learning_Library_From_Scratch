from vplearn.metrics import _Metric

import numpy as np

class MAE(_Metric):
    def __init__(self, ground_truth: np.ndarray, predict: np.ndarray):
        super().__init__()
        self._check_fit_input_format(ground_truth, predict)

        self.ground_truth = ground_truth
        self.predict = predict
        self.mae = self._compute()

    def _compute(self) -> float:
        return np.mean(abs(self.ground_truth - self.predict))

    def __float__(self) -> float:
        return float(self.mae)

    def __repr__(self) -> str:
        return f"MAE({self.mae:.6f})"
    