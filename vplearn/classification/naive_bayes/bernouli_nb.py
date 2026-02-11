from vplearn._model import Model

import pandas as pd
import numpy as np

class BernouliNB(Model):
    def __init__(self):
        self.laplace_smoothing_alpha = 1
        super().__init__()
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be np.ndarray or pd.DataFrame")
        if not isinstance(y, pd.Series) and not isinstance(y, pd.DataFrame):
            raise TypeError("y must be np.ndarray or pd.DataFrame/Series")
        
        self._compute_lambda(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            raise TypeError("X must be np.ndarray or pd.DataFrame")
        try:
            X = X.to_numpy()
            X = X.reshape(1,-1)
        except:
            raise TypeError("Can not convert pd.DataFrame to np.ndarray")
        
        return self._predict_bnb_class(X)
    
    def _compute_mnb_features(self, X:pd.DataFrame, y:pd.Series) -> None:
        self.p_c = y.value_counts(normalize=True).to_dict()
        
        df = X.copy()
        df['class'] = y
        self.N_b = df.groupby('class').sum().sum(axis=1).to_dict()
        
    def _compute_lambda(self, X:pd.DataFrame, y:pd.Series) -> np.ndarray:
        lambda_list = []
        self.classes = list(y.unique().tolist())
        
        self._compute_mnb_features(X, y)
        
        for _class in self.classes:
            
            X_c = pd.DataFrame(X[y == _class])
            X_c_np = X_c.to_numpy()
            
            numerators = np.sum(X_c_np, axis=0) + self.laplace_smoothing_alpha
            denominator = X_c.shape[0] + 2 * self.laplace_smoothing_alpha
            
            lambda_c = numerators / denominator
            lambda_list.append(lambda_c)
            
        lambda_list = np.array(lambda_list)
        self.lambda_list = lambda_list
        
        print(f"lambda list : {lambda_list}")
        
    def _predict_bnb_class(self, X: np.ndarray) -> np.ndarray:
        pred = []
        
        for x_row in X:
            scores = []
            for class_idx, _class in enumerate(self.classes):
                score = self.p_c[_class]
                for feature_idx, x_i in enumerate(x_row):
                    _ = (self.lambda_list[class_idx, feature_idx] * x_i) \
                    + ((1 - self.lambda_list[class_idx, feature_idx]) * (1 - x_i))
                    print(f"({self.lambda_list[class_idx, feature_idx] * x_i} + {(1 - self.lambda_list[class_idx, feature_idx]) * (1 - x_i)})", end=',')
                    print(_)
                    score *= _
                scores.append(score)
                print(score)
            max_score_class_idx = np.argmax(scores)
            max_score_class = self.classes[max_score_class_idx]
            pred.append(max_score_class)
        
        return np.array(pred)