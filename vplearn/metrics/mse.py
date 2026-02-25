from vplearn.metrics import _Metric

import numpy as np

class MSE(_Metric):
    def __init__(self, ground_truth: np.ndarray, predict: np.ndarray):
        super().__init__()
        self._check_fit_input_format(ground_truth, predict)

        self.ground_truth = ground_truth
        self.predict = predict
        self.mse = self._compute()

    def _compute(self) -> float:
        return np.mean((self.ground_truth - self.predict) ** 2)

    def __float__(self) -> float:
        return float(self.mse)

    def __repr__(self) -> str:
        return f"MSE({self.mse:.6f})"
    