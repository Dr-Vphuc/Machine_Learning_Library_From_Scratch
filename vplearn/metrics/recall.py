from vplearn.metrics import _Metric

import numpy as np

class Recall(_Metric):
    def __init__(self, ground_truth: np.ndarray, predict: np.ndarray):
        super().__init__()
        self._check_fit_input_format(ground_truth, predict)

        self.ground_truth = ground_truth
        self.predict = predict
        self.recall = self._compute()

    def _compute(self) -> float:
        true_positive = np.sum((self.ground_truth == 1) & (self.predict == 1))
        false_negative = np.sum((self.ground_truth == 1) & (self.predict == 0))
        return true_positive / (true_positive + false_negative)

    def __float__(self) -> float:
        return float(self.recall)

    def __repr__(self) -> str:
        return f"Recall({self.recall:.6f})"