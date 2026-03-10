from typing import Union

import numpy as np
import pandas as pd

def train_test_split(
    X: Union[np.ndarray, pd.DataFrame, pd.Series], 
    y: Union[np.ndarray, pd.Series], 
    test_size=0.2, 
    random_state=None
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test