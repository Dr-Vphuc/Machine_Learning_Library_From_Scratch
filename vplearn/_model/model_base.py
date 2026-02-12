from vplearn._model import Model

from typing import Union

import pandas as pd
import numpy as np

class ModelBaseModel(Model):
    def __init__(self):
        super().__init__()
        
    def _convert_to_numpy(
        self, 
        target: Union[pd.DataFrame, pd.Series]
    ) -> np.ndarray:
        if isinstance(target, np.ndarray):
            return target
        # Self-note : Lưu lại các tên trường
        return target.to_numpy()