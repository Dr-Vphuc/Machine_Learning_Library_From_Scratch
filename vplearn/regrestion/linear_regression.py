from vplearn._model import ModelBaseModel

from typing import Tuple

import pandas as pd
import numpy as np

class LinearRegrestion(ModelBaseModel):
    def __init__(self):
        super().__init__()
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        self._check_fit_input_format(X, y)
        
        self.X_train = self._convert_to_numpy(X)
        self.y_train = self._convert_to_numpy(y)
    
        X_bars = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        Q, R = self.qr_householder(X_bars)
        R_pinv = np.linalg.pinv(R)
        A = np.dot(R_pinv, Q.T)
        
        
        _ = np.dot(A, y).T.tolist()
        self.coefficents = _[0]
        
        return np.dot(A, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_predict_input_format(X)
        
        X = self._correct_predict_input_format(X)
        
        return self._predict_linear_regression(X)
    
    def qr_householder(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        M = A.shape[0]
        N = A.shape[1]
        
        Q = np.identity(M)
        R = np.copy(A)
        
        for n in range(N):
            x = A[n:, n]
            k = x.shape[0]
            ro = -np.sign(x[0]) * np.linalg.norm(x)
            
            e = np.zeros(k)
            e[0] = 1
            v = (1 / (x[0] - ro)) * (x - (ro * e))
            
            for i in range(N):
                R[n:, i] = R[n:, i] - (2 / (v@v)) * ((np.outer(v, v)) @ R[n:, i])
                
            for i in range(M):
                Q[n:, i] = Q[n:, i] - (2 / (v@v)) * ((np.outer(v, v)) @ Q[n:, i])
                
        return Q.transpose(), R
    
    def _predict_linear_regression(self, X: np.ndarray) -> np.ndarray:
        return self.coefficents[0] + np.dot(X, self.coefficents[1:])