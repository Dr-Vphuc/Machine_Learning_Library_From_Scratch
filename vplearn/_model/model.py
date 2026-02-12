from typing import Union

import pandas as pd
import numpy as np

class Model:
    def __init__(seft):
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass
    
    def _check_fit_input_format(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> bool:
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError("X must be np.ndarray or pd.DataFrame/Series")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be np.ndarray or pd.Series")
        
        return True
    
    def _check_predict_input_format(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> bool:
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError("X must be np.ndarray or pd.DataFrame/Series")
        
        return True
    
    def _correct_predict_input_format(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            try:
                X_np = X.to_numpy()
                if X_np.ndim == 1:
                    if X.name:
                        X_np = X_np.reshape(-1, 1)
                    elif X.columns:
                        X_np = X_np.reshape(1, -1)
            except:
                raise 
        
        return X_np  