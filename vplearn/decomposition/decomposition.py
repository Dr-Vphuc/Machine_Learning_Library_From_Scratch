from typing import Union

import pandas as pd
import numpy as np

class Decomposition:
    def __init__(self):
        pass
    
    def fit(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> None:
        pass
    
    def transforms(self):
        pass
    
    def _check_fit_input_format(
        self, 
        X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> bool:
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
            raise TypeError("X must be np.ndarray or pd.DataFrame/Series")
        
        return True
    
    def _convert_to_numpy(
        self, 
        target: Union[pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        if isinstance(target, np.ndarray):
            return target
                
        return target.to_numpy()