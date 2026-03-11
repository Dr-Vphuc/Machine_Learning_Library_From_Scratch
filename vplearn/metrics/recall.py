from vplearn.metrics import _Metric

import numpy as np

class Recall(_Metric):
    def __init__(self, ground_truth: np.ndarray, predict: np.ndarray):
        super().__init__()
        self._check_fit_input_format(ground_truth, predict)

        self.ground_truth = ground_truth
        self.predict = predict
        self.classes: np.ndarray = np.unique(ground_truth)
        self.recall_per_class: dict = {}
        self.recall: float = self._compute()

    def _compute(self) -> float:
        recalls: list[float] = []

        for _class in self.classes:
            true_positive: int = np.sum((self.ground_truth == _class) & (self.predict == _class))
            false_negative: int = np.sum((self.ground_truth == _class) & (self.predict != _class))
            denominator: int = true_positive + false_negative

            r: float = true_positive / denominator if denominator > 0 else 0.0
            self.recall_per_class[_class] = r
            recalls.append(r)

        return float(np.mean(recalls))

    def __float__(self) -> float:
        return float(self.recall)

    def __format__(self, format_spec: str) -> str:
        return format(self.recall, format_spec)

    def __repr__(self) -> str:
        return f"Recall({self.recall:.6f})"