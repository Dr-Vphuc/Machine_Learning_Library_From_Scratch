from vplearn._model import ModelBaseModel

from typing import Literal

import pandas as pd
import numpy as np
from scipy import sparse

class LogisticRegression(ModelBaseModel):
    def __init__(
        self,
        eta: float  = 0.05 , 
        check_w_after: float = 20,
        max_scan_time: int = 100,
        tol: float = 1e-4,
        activate: Literal['auto', 'sigmoid', 'softmax']  = "auto",
        threshold: float  = 0.5
    ):
        super().__init__()
        self.eta = eta
        self.check_w_after = check_w_after
        self.max_scan_time = max_scan_time
        self.tol = tol
        self.activate = activate
        if not (0 < threshold < 1):
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._check_fit_input_format(X, y)
        
        X_train = self._convert_to_numpy(X)
        y_train = self._convert_to_numpy(y)
        
        self.n_classes = self._get_number_of_classes(y_train)
        
        self.max_count = X.shape[0] * self.max_scan_time
        
        self.d = X.shape[1]
        self.w_init = np.random.randn(self.d, 1)
        
        if self.activate is None or self.activate == 'auto':
            if self.n_classes == 2:
                self._activate_func = 'sigmoid'
                self.coefficents = self._logistic_sigmoid_regression(X_train, y_train)
            elif self.n_classes > 2:
                self._activate_func = 'softmax'
                self.coefficents = self._softmax_regression(X_train, y_train)
        if self.activate == 'sigmoid':
            if self.n_classes > 2:
                raise ValueError("Sigmoid activation is only used for binary classification.")
            self._activate_func = 'sigmoid'
            self.coefficents = self._logistic_sigmoid_regression(X_train, y_train)
        elif self.activate == 'softmax':
            self._activate_func = 'softmax'
            self.coefficents = self._softmax_regression(X_train, y_train)
    
    def predict(self, X: pd.DataFrame):
        self._check_predict_input_format(X)
        
        X = self._correct_predict_input_format(X)
        
        return (
            self._predict_sigmoid_logistic_class(X)
            if self._activate_func == 'sigmoid'
            else self._predict_softmax_logistic_class(self.w, X)
            )
    
    def _get_number_of_classes(self, y: np.ndarray) -> int:
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

    def _convert_labels(self, y, C):
        """
        convert 1d label to a matrix label: each column of this
        matrix coresponding to 1 element in y. In i-th column of Y,
        only one non-zeros element located in the y[i]-th position,
        and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return

        [[1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]]
        """
        Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
        return Y

    def _softmax_stable(self, Z):
        """
        Compute softmax values for each sets of scores in Z.
        each column of Z is a set of score.
        """
        e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
        A = e_Z / e_Z.sum(axis = 0)
        return A

    def _softmax(self, Z):
        """
        #Compute softmax values for each sets of scores in V.
        #each column of V is a set of score.
        """
        e_Z = np.exp(Z)
        A = e_Z / e_Z.sum(axis = 0)
        return A

    def _softmax_regression(self, X, y):
        W = [self.w_init]
        C = self._get_number_of_classes(y)
        Y = self._convert_labels(y, C)
        N = X.shape[0]
        d = X.shape[1]

        count = 0        
        while count < self.max_count:            
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[i, :].reshape(d, 1)
                yi = Y[:, i].reshape(C, 1)
                ai = self._softmax(np.dot(W[-1].T, xi))
                W_new = W[-1] + self.eta * xi.dot((yi - ai).T)
                count += 1
                if count % self.check_w_after == 0:
                    if np.linalg.norm(W_new - W[-self.check_w_after]) < self.tol:
                        return W[-1]
                W.append(W_new)
        return W[-1]

    def _cost(self, X, Y, W):
        A = self._softmax(W.T.dot(X))
        return -np.sum(Y*np.log(A))

    def _predict_softmax_logistic_class(self, w, X) -> np.ndarray:
        """
        predict output of each columns of X
        Class of each x_i is determined by location of max probability
        Note that class are indexed by [0, 1, 2, ...., C-1]
        """
        w = np.array(w)
        X = X.T
        A = self._softmax_stable(w.T.dot(X))
        return np.argmax(A, axis = 0)