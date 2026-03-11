from vplearn.metrics import _Metric

import numpy as np

class Precision(_Metric):
    def __init__(self, ground_truth: np.ndarray, predict: np.ndarray):
        super().__init__()
        self._check_fit_input_format(ground_truth, predict)

        self.ground_truth = ground_truth
        self.predict = predict
        self.classes: np.ndarray = np.unique(ground_truth)
        self.precision_per_class: dict = {}
        self.precision: float = self._compute()

    def _compute(self) -> float:
        precisions: list[float] = []

        for _class in self.classes:
            true_positive: int = np.sum((self.ground_truth == _class) & (self.predict == _class))
            false_positive: int = np.sum((self.ground_truth != _class) & (self.predict == _class))
            denominator: int = true_positive + false_positive

            p: float = true_positive / denominator if denominator > 0 else 0.0
            self.precision_per_class[_class] = p
            precisions.append(p)

        return float(np.mean(precisions))

    def __float__(self) -> float:
        return float(self.precision)

    def __format__(self, format_spec: str) -> str:
        return format(self.precision, format_spec)

    def __repr__(self) -> str:
        return f"Precision({self.precision:.6f})"