import pandas as pd
import numpy as np

class Encoding:
    def __init__(self):
        pass
    
    def label_encoding(self, y: pd.DataFrame) -> pd.DataFrame:
        classes = pd.unique(y)
        label_mapping = {}
        
        for idx, _class in enumerate(classes):
            label_mapping[_class] = idx
            
        return y.map(label_mapping)
    
    def categorical_encoding(self, X: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
        for feature in categorical_features:
            X[feature] = self.label_encoding(X[feature])
        
        return X

    def _word_with_idx(self, v: set) -> dict:
        result = {}
        for idx, key in enumerate(v):
            result[key] = idx
        
        return result

    def bag_of_words_encoding(self, X: list[str], v: set, phrase: bool) -> pd.DataFrame:
        vector_list = []
        vector_size = len(v)
        d = self._word_with_idx(v)
        
        if phrase:
            for d in X:
                tokens = d.strip().split(' ')
                vector = np.zeros(vector_size)
                idx_list = []
                
                for token in tokens:
                    idx_list.append(d[token])
                    
                np.add.at(vector, idx_list, 1)
                vector_list.append(vector)
            return np.array(vector_list)
        else:
            raise TypeError('Data does not pharse processed is not support yet.')
        
    def one_hot_encoding(self, X: pd.DataFrame, categorical_features: list[str], y: str = 'class') -> pd.DataFrame:
        for feature in categorical_features:
            one_hot = pd.get_dummies(X[feature], prefix=feature, drop_first=True).astype(int)
            X = pd.concat([X, one_hot], axis=1)
            X.drop(feature, axis=1, inplace=True)
            
        if y in X.columns:
            predict_var = X[y]
            X.drop(y, axis=1, inplace=True)
            X = pd.concat([X, predict_var], axis=1)
        return X