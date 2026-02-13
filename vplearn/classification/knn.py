from vplearn._model import InstanceBaseModel

import pandas as pd
import numpy as np

class KNN(InstanceBaseModel):
    def __init__(self):
        super().__init__()
        
    def fit(self, X: pd.DataFrame, y: pd.Series, k: int) -> None:
        self._check_fit_input_format(X, y)
        
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        
        self.X_train = self._convert_to_numpy(X)
        self.y_train = self._convert_to_numpy(y)
        self.k = k
    
    def predict(self, X: pd.DataFrame):
        self._check_predict_input_format(X)
        
        X = self._correct_predict_input_format(X)
        
        return self._predict_knn_values(X)
    
    def _predict_knn_values(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        
        for x_row in X:
            k_closest_idxes = self._get_k_closest(x_row)
            k_closest_points_y = self.y_train[k_closest_idxes]
            
            values, counts = np.unique(k_closest_points_y, return_counts=True)
            most_common = values[counts.argmax()]
            
            preds.append(most_common)
            
        return np.array(preds)
    
    def _get_k_closest(self, x: np.ndarray) -> np.ndarray:
        
        distances = np.sum((self.X_train - x)**2, axis=1)
        idxes = np.argpartition(distances, self.k)[:self.k]
        
        return idxes
    
    
    