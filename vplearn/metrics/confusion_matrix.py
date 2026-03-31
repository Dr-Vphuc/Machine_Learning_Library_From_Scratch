from vplearn.metrics import _Metric

import numpy as np

class ConfusionMatrix(_Metric):
    def __init__(self, ground_truth: np.ndarray, predict: np.ndarray):
        super().__init__()
        self._check_fit_input_format(ground_truth, predict)

        self.ground_truth = ground_truth
        self.predict = predict
        self.confusion_matrix = self._compute()

    def _compute(self) -> np.ndarray:
        classes = np.unique(self.ground_truth)
        n_classes = len(classes)
        
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for true_label, pred_label in zip(self.ground_truth, self.predict):
            true_idx = np.where(classes == true_label)[0][0]
            pred_idx = np.where(classes == pred_label)[0][0]
            matrix[true_idx][pred_idx] += 1
        
        return matrix

    def __repr__(self) -> str:
        return f"ConfusionMatrix:\n{self.confusion_matrix}"