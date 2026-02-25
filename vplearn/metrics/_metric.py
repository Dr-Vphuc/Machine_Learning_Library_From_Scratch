from typing import Union

import pandas as pd
import numpy as np

class _Metric():
    def __init__(self):
        pass
    
    def _compute(self):
        pass
    
    def _check_fit_input_format(
        self, 
        ground_truth: Union[np.ndarray], 
        predict: Union[np.ndarray] = None
    ) -> bool:
        if not isinstance(ground_truth, np.ndarray):
            raise TypeError("ground_truth must be np.ndarray")
        if not isinstance(predict, (np.ndarray)):
            raise TypeError("predict must be np.ndarray")
        
        return True