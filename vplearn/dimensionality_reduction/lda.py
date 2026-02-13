from vplearn.dimensionality_reduction import DimensionalityReduction

from typing import Union

import pandas as pd
import numpy as np

class LDA(DimensionalityReduction):
    def __init__(self):
        super().__init__()
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        self._check_fit_input_format(X)
        
        X = self._convert_to_numpy(X)
        X_scaled = (X - X.mean()) / X.std(ddof=0)

        y = self._convert_to_numpy(y)
        classes, y_enc = np.unique(y, return_inverse=True)

        d = X_scaled.shape[1]
        C = len(classes)

        m = X_scaled.mean(axis=0, keepdims=True).T
        
        S_W = np.zeros((d, d), dtype=float)
        S_B = np.zeros((d, d), dtype=float)

        for c in range(C):
            Xc = X_scaled[y_enc == c]
            n_c = Xc.shape[0]
            if n_c == 0:
                continue

            mc = Xc.mean(axis=0, keepdims=True).T
            
            XcT = Xc.T
            SWc = (XcT - mc @ np.ones((1, n_c))) @ (XcT - mc @ np.ones((1, n_c))).T
            S_W += SWc

            a = (mc - m)
            S_B += n_c * (a @ a.T)

        reg = 1e-3
        S_W_reg = S_W + reg * np.eye(d)

        M = np.linalg.pinv(S_W_reg) @ S_B
        M = 0.5 * (M + M.T)

        eig_vals, eig_vecs = np.linalg.eigh(M)
        
        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        
        self.eig_vecs = eig_vecs
        self.Xc = X_scaled - X_scaled.mean(axis=0, keepdims=True)
        
    
    def transforms(
        self,
        k: int
    ) -> np.ndarray:
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        
        X_lda = self.Xc @ self.eig_vecs[:, :k]
        self.components_ = X_lda
        
        return X_lda
    
    def fit_transform(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        k: int
    ) -> np.ndarray:
        self.fit(X, y)
        return self.transforms(k)
