from vplearn.dimensionality_reduction import DimensionalityReduction

from typing import Union

import pandas as pd
import numpy as np

class PCA(DimensionalityReduction):
    def __init__(self, n_components: int = None):
        super().__init__()
        self.n_components_ = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.std_ = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> None:
        self._check_fit_input_format(X)
        
        # X = X.fillna(X.mean())
        X = self._convert_to_numpy(X)
        
        # Lưu mean và std để dùng cho transform
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        
        X_scaled = (X - self.mean_) / self.std_

        X_corr = (1 / (len(X_scaled))) * X_scaled.T.dot(X_scaled)

        eig_vals, eig_vecs = np.linalg.eig(X_corr)
        
        idx = np.argsort(eig_vals)[::-1]
        
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        var_ratio = eig_vals / np.sum(eig_vals)
        
        self.explained_variance_ = eig_vals[:self.n_components_]
        self.explained_variance_ratio_ = var_ratio[:self.n_components_]
        self.components_ = eig_vecs[:, :self.n_components_]
        
    
    def transform(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        X = self._convert_to_numpy(X)
        
        # Scale X sử dụng mean/std từ training data
        X_scaled = (X - self.mean_) / self.std_
        
        # Transform bằng cách nhân với components
        X_pca = X_scaled @ self.components_
        
        return X_pca
    
    def fit_transform(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        self.fit(X)
        return self.transform(X)