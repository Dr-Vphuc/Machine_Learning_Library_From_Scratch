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