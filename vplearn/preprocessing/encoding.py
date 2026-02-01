import pandas as pd

class Encoding:
    def __init__(self):
        pass
    
    def label_encoding(self, y: pd.DataFrame) -> pd.DataFrame:
        classes = pd.unique(y)
        label_mapping = {}
        
        for idx, _class in enumerate(classes):
            label_mapping[_class] = idx
            
        return y.map(label_mapping)
