import pandas as pd
import numpy as np

class Model:
    def __init__(seft):
        pass
    
    def fit():
        pass
    
    def predict():
        pass
    
    def _check_fit_input_format(
        self, 
        X: pd.DataFrame | pd.Series | np.ndarray, 
        y: pd.Series | np.ndarray
    ) -> bool:
        if not isinstance(X, pd.DataFrame | pd.Series | np.ndarray):
            raise TypeError("X must be np.ndarray or pd.DataFrame/Series")
        if not isinstance(y, pd.DataFrame | np.ndarray):
            raise TypeError("y must be np.ndarray or pd.Series")
        
        return True
    
    def _check_predict_input_format(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray
    ) -> bool:
        if not isinstance(X, pd.DataFrame | pd.Series | np.ndarray):
            raise TypeError("X must be np.ndarray or pd.DataFrame/Series")
        
        return True