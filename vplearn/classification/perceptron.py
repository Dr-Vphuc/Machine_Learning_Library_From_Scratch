from vplearn._model import ModelBaseModel

import pandas as pd
import numpy as np

class Perceptron(ModelBaseModel):
    def __init__(
        self,
        eta: float  = 1 , 
        check_w_after: float = 20,
        max_scan_time: int = 100,
        tol: float = 1e-4,
        ):
        super().__init__()
        self.w_init = None
        self.eta = eta
        self.check_w_after = check_w_after
        self.max_scan_time = max_scan_time
        self.tol = tol
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._check_fit_input_format(X, y)
        
        X_train = self._convert_to_numpy(X)
        y_train = self._convert_to_numpy(y)
        
        self.d = X.shape[1]
        self.w_init = np.random.randn(self.d, 1)
        
        self.max_count = X.shape[0] * self.max_scan_time
        
        self.coefficents, self.m = self._perceptron(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_predict_input_format(X)
        
        X = self._correct_predict_input_format(X)
        
        return self._predict_perceptron_class(X)
    
    def _h(self, w, x):
        return np.sign(np.dot(w.T, x))
    
    def _has_converged(self, X, y, w):
        return np.array_equal(self._h(w, X.T), y) 
    
    def _perceptron(self, X: np.ndarray, y):
        w = [self.w_init]
        N = X.shape[0]
        
        count = 0
        mis_points = []
        while count < self.max_count:
            mix_id = np.random.permutation(N)
            for i in range(N):
                xi = X[mix_id[i], :].reshape(self.d, 1)
                yi = y[0, mix_id[i]]
                count += 1
                if self._h(w[-1], xi)[0] != yi:
                    mis_points.append(mix_id[i])
                    w_new = w[-1] + self.eta * yi * xi
                    if count % self.check_w_after == 0:
                        if np.linalg.norm(w_new - w[-self.check_w_after]) < self.tol:
                            return w[-1]
                    w.append(w_new)
                    
            if self._has_converged(X, y, w[-1]):
                break
        return (w[-1], mis_points)
    
    def _predict_perceptron_class(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        
        for x_row in X:
            pred = self.w[0] + np.dot(x_row, self.w[1:])
            preds.append(pred)
        
        return np.array(preds)