from vplearn.dimensionality_reduction import DimensionalityReduction

from typing import Union

import pandas as pd
import numpy as np

class PCA(DimensionalityReduction):
    def __init__(self):
        super().__init__()
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> None:
        self._check_fit_input_format(X)
        
        # X = X.fillna(X.mean())
        X = self._convert_to_numpy(X)
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

        X_corr = (1 / (len(X_scaled))) * X_scaled.T.dot(X_scaled)

        eig_vals, eig_vecs = np.linalg.eig(X_corr)
        
        idx = np.argsort(eig_vals)[::-1]
        
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        var_ratio = eig_vals / np.sum(eig_vals)
        
        self.explained_variance_ = eig_vals
        self.explained_variance_ratio_ = var_ratio
        self.X_scaled = X_scaled
        self.eig_vecs = eig_vecs
        
    
    def transforms(
        self,
        k: int
    ) -> np.ndarray:
        X_pca = self.X_scaled @ self.eig_vecs[:, :k]
        self.components_ = X_pca
        
        return X_pca
    
    def fit_transform(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        k: int
    ) -> np.ndarray:
        self.fit(X)
        return self.transforms(k)