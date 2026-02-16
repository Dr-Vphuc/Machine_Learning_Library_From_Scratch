from vplearn._model import InstanceBaseModel

import pandas as pd
import numpy as np

class Kmeans(InstanceBaseModel):
    def __init__(self):
        super().__init__()
        
    def fit(self, X: pd.DataFrame, k: int):
        self._check_fit_input_format(X)
        
        X = self._convert_to_numpy(X)
        if not isinstance(k, int):
            raise TypeError('k must be an integer')
        
        self.k = k
        
        self.centers, self.labels = self._fit_k_means(X, k)
    
    def predict(self, X: pd.DataFrame):
        distances = self._cdist(X, self.centers)
        preds = np.argmin(distances, axis=1)
        
        return preds
    
    def _fit_k_means(self, X: pd.DataFrame, k: int):
        centers = [self._init_centers(X, k)]
        
        while True:
            labels = self._assign_labels(X, centers[-1])
            new_centers = self._update_centers(X, labels, k)
            if self._has_converged(centers[-1], new_centers):
                break
            centers.append(new_centers)
            
        return np.array(centers[-1]), labels
        
    def _init_centers(self, X: np.ndarray, k: int) -> np.ndarray:
        """Initialize centers using K-means++ algorithm"""
        n_samples = X.shape[0]
        centers = []
        
        first_idx = np.random.randint(n_samples)
        centers.append(X[first_idx])
        
        for _ in range(k - 1):
            # Tính khoảng cách từ mỗi điểm đến center gần nhất
            distances = np.array([
                np.min([np.sum(np.square(x - c)) for c in centers])
                for x in X
            ])
                        
            probabilities = distances / distances.sum()
            next_idx = np.random.choice(n_samples, p=probabilities)
            centers.append(X[next_idx])
        
        self.init_centers = np.array(centers)
        return np.array(centers)
    
    def _assign_labels(self, X, centers):
        D = self._cdist(X, centers)
        
        return np.argmin(D, axis = 1)
    
    def _cdist(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        X_sq = np.sum(np.square(X), axis=1, keepdims=True)  
        C_sq = np.sum(np.square(centers), axis=1)           
        cross = X @ centers.T                      

        distances = np.sqrt(X_sq + C_sq - 2 * cross)
        return distances
    
    def _update_centers(self, X: np.ndarray, labels: np.ndarray, k: int):
        centers = np.zeros((k, X.shape[1]))
        for k in range(k):
            Xk = X[labels == k, :]
            centers[k,:] = np.mean(Xk, axis = 0)
        return centers
    
    def _has_converged(self, centers: np.ndarray, new_centers: np.ndarray) -> bool:
        return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))
        