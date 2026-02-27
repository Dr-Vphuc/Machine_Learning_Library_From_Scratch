from vplearn._model import ModelBaseModel

from typing import Union

import pandas as pd
import numpy as np

from cvxopt import matrix, solvers

class SVM(ModelBaseModel):
    def __init__(self):
        super().__init__()
    
    def fit(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        self._check_fit_input_format(X, y)
        
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)
        
    def predict(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        self._check_predict_input_format(X)
        
        X = self._correct_predict_input_format(X)
        
        preds = []
        
        for x_row in X:
            pred = np.sign(self.w.T.dot(x_row) + self.w0)
            preds.append(pred)
        
        return np.array(preds)
    
    def _find_X0_X1(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unique_classes = np.unique(y)
        X0 = X[y == unique_classes[0]]
        X1 = X[y == unique_classes[1]]
        return X0, X1
    
    def _find_hard_margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X0, X1 = self._find_X0_X1(X, y)
        # build P ~ K
        V = np.concatenate((X0.T, - X1.T), axis = 1)
        P = matrix(V.T.dot(V)) # P ~ K in slide see definition of V, K near eq (8)
        q = matrix(-np.ones((2 * N, 1))) # all-one vector
        # build A, b, G, h
        G = matrix(-np.eye(2 * N)) # for all lambda_n >= 0! note that we solve -g(lambda) ->
        min
        h = matrix(np.zeros((2 * N, 1)))
        A = matrix(y) # the equality constrain is actually y^T lambda = 0
        b = matrix(np.zeros((1, 1)))
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        l = np.array(sol['x']) # lambda
        epsilon = 1e-6 # just a small number, greater than 1e-9, to filter values of lambda
        S = np.where(l > epsilon)[0]
        VS = V[:, S]
        XS = X[:, S]
        yS = y[:, S]
        lS = l[S]
        # calculate w and b
        self.w = VS.dot(lS)
        self.w0 = np.mean(yS.T - self.w.T.dot(XS))