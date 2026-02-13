from vplearn._model import ModelBaseModel

from typing import Literal

import pandas as pd
import numpy as np

class LogisticRegression(ModelBaseModel):
    def __init__(
        self,
        eta: float  = 0.05 , 
        check_w_after: float = 20,
        max_count: int = 10000,
        tol: float = 1e-4,
        activate: Literal['auto', 'sigmoid', 'softmax']  = "auto",
        threshold: float  = 0.5,
        auto_label = bool
    ):
        super().__init__()
        self.eta = eta
        self.check_w_after = check_w_after
        self.max_count = max_count
        self.tol = tol
        self.activate = activate
        if not (0 < threshold < 1):
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.auto_label = auto_label
        
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._check_fit_input_format(X, y)
        
        X_train = self._convert_to_numpy(X)
        y_train = self._convert_to_numpy(y)
        
        self.n_classes = self._get_number_of_classes(y_train)
        
        self.d = X.shape[1]
        self.w_init = np.random.randn(self.d, 1)
        
        if self.activate is None or self.activate == 'auto':
            if self.n_classes == 2:
                self.coefficents = self._logistic_sigmoid_regression(X_train, y_train)
            elif self.n_classes > 2:
                pass
        if self.activate == 'sigmoid':
            if self.n_classes > 2:
                raise ValueError("Sigmoid activation is only used for binary classification.")
            self.coefficents = self._logistic_sigmoid_regression(X_train, y_train)
        elif self.activate == 'softmax':
            pass
    
    def predict(self, X: pd.DataFrame):
        self._check_predict_input_format(X)
        
        X = self._correct_predict_input_format(X)
        
        return self._predict_sigmoid_logistic_class(X)
    
    def _get_number_of_classes(self, y: pd.Series) -> int:
        return len(np.unique(y).tolist())
    
    def _sigmoid(self, s):
        return 1/(1 + np.exp(-s))

    def _logistic_sigmoid_regression(self, X, y):
        w = [self.w_init]        
        N = X.shape[0]
        d = X.shape[1]
        count = 0
        while count < self.max_count:
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[i, :].reshape(d, 1)
                yi = y[i]
                zi = self._sigmoid(np.dot(w[-1].T, xi))
                w_new = w[-1] + self.eta * (yi - zi) * xi
                count += 1
                if count % self.check_w_after == 0:
                    if np.linalg.norm(w_new - w[-self.check_w_after]) < self.tol:
                        return w[-1]
                w.append(w_new)
        return w[-1]
    
    def _predict_sigmoid_logistic_class(self, X: np.ndarray) -> np.ndarray:
        preds = []
        confidents = []
        
        for x_row in X:
            confident = self._sigmoid(np.dot(self.coefficents.T, x_row))
            confidents.append(confident)
            
            pred = 1 if confident > self.threshold else 0
            preds.append(pred)
        
        self.confidents = confidents
        return np.array(preds)