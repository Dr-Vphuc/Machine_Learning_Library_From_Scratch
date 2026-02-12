from vplearn._model import InstanceBaseModel

import pandas as pd
import numpy as np
from scipy.stats import norm

class GaussianNB(InstanceBaseModel):
    def __init__(self):
        super().__init__()
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._check_fit_input_format(X, y)    
        
        X = self._convert_to_dataframe(X)
        y = self._convert_to_dataframe(y)
        
        self.mu_list, self.std_list, self.pi_list = self._compute_data_features(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_predict_input_format(X)
        
        self._correct_predict_input_format(X)
        
        return self._predict_gaussian_nb_class(X)
    
    def _get_X_of_the_class(self, X: pd.DataFrame, y: pd.Series, _class: int) -> pd.DataFrame:
        df = pd.concat([X, y], axis=1)
        df = df[df.iloc[:, -1] == _class]
        
        return df.iloc[:, :-1]
    
    def _compute_data_features(self, X: pd.DataFrame, y: pd.Series) \
    -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        
        data_size = y.shape[0]
        n_class = len(pd.unique(y))
        
        mu_list = []
        std_list = []
        pi_list = []
        
        for _class in range(n_class):
            X_c = self._get_X_of_the_class(X, y, _class)
            
            mu_list_of_c = np.mean(X_c, axis=0)
            mu_list.append(mu_list_of_c)
            
            std_list_of_c = np.std(X_c, axis=0)
            std_list.append(std_list_of_c)
            
            pi_c = X_c.shape[0] / data_size
            pi_list.append(pi_c)
        
        return np.array(mu_list), np.array(std_list), np.array(pi_list)
    
    def _predict_gaussian_nb_class(self, X: np.ndarray) -> np.ndarray:
        n_classes = self.mu_list.shape[0]
        n_features = X.shape[1]
        predict = []
        
        for x_row in X:
            scores_list = []
            for c in range(n_classes):
                score = 1
                for i in range(n_features):
                    score *= norm.pdf(x = x_row[i], loc = self.mu_list[c][i], scale = self.std_list[c][i])
                score *= self.pi_list[c]
                scores_list.append(score)
            predict.append(np.argmax(scores_list))
            
        return np.array(predict)
    