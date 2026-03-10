from vplearn.metrics import _Metric

import numpy as np

class Precision(_Metric):
    def __init__(self, ground_truth: np.ndarray, predict: np.ndarray):
        super().__init__()
        self._check_fit_input_format(ground_truth, predict)

        self.ground_truth = ground_truth
        self.predict = predict
        self.precision = self._compute()

    def _compute(self) -> float:
        true_positive = np.sum((self.ground_truth == 1) & (self.predict == 1))
        false_positive = np.sum((self.ground_truth == 0) & (self.predict == 1))
        return true_positive / (true_positive + false_positive)

    def __float__(self) -> float:
        return float(self.precision)

    def __repr__(self) -> str:
        return f"Precision({self.precision:.6f})"