from vplearn._model import InstanceBaseModel
from typing import Literal

import numpy as np
import pandas as pd

class MutinomialNB(InstanceBaseModel):
    def __init__(
        self,
        laplace_smoothing_alpha: float = 1,
        type: Literal['document', 'category'] = 'category',
    ) -> None:
        self.laplace_smoothing_alpha = laplace_smoothing_alpha
        self.type = type
        super().__init__()
        
    def fit(self, X: pd.DataFrame, y:pd.Series) -> None:
        self._check_fit_input_format(X, y)    
        
        X = self._convert_to_dataframe(X)
        y = self._convert_to_dataframe(y)
        
        if self.type == 'document':
            self._compute_document_lambda(X, y)
        else:
            self._compute_category_params(X, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_predict_input_format(X)
        
        X = self._correct_predict_input_format(X)
        
        if self.type == 'document':
            return self._predict_mnb_document_class(X)
        else:
            return self._predict_mnb_category_class(X)
        
    def _compute_mnb_document_features(self, X:pd.DataFrame, y:pd.Series) -> None:
        self.p_c = y.value_counts(normalize=True).to_dict()
        
        df = X.copy()
        df['class'] = y
        self.N_b = df.groupby('class').sum().sum(axis=1).to_dict()
        
    def _compute_document_lambda(self, X:pd.DataFrame, y:pd.Series) -> np.ndarray:
        lambda_list = []
        self.classes = list(y.unique().tolist())
        n_features = X.shape[1]
        
        self._compute_mnb_document_features(X, y)
        
        for _class in self.classes:
            denominator = n_features + self.N_b[_class]
            
            X_c = pd.DataFrame(X[y == _class])
            X_c_np = X_c.to_numpy()
            
            numerators = np.sum(X_c_np, axis=0) + self.laplace_smoothing_alpha
            
            lambda_c = numerators / denominator
            lambda_list.append(lambda_c)
            
        lambda_list = np.array(lambda_list)
        self.lambda_list = lambda_list
        
        return lambda_list
        
    def _predict_mnb_document_class(self, X:pd.DataFrame) -> np.ndarray:
        pred = []
        
        for x_row in X:
            scores = []
            for class_idx, _class in enumerate(self.classes):
                score = self.p_c[_class]
                for feature_idx, x_i in enumerate(x_row):
                    score *= self.lambda_list[class_idx,feature_idx] ** x_i
                scores.append(score)
            max_score_class_idx = np.argmax(scores)
            max_score_class = self.classes[max_score_class_idx]
            pred.append(max_score_class)

        return np.array(pred)
    
    # ---- Category type methods ----
    
    def _compute_category_params(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Tính các tham số cần thiết cho Multinomial NB dạng category:
        - p_c: xác suất tiên nghiệm P(k) cho mỗi lớp
        - classes: danh sách các lớp
        - num_nomials_per_feature: số giá trị phân biệt M của mỗi feature
        - X_by_class: dữ liệu huấn luyện chia theo từng lớp
        """
        self.classes = list(y.unique().tolist())
        
        # P(k) = Nk / N
        self.p_c = y.value_counts(normalize=True).to_dict()
        
        # Số giá trị phân biệt M cho mỗi feature
        self.num_nomials_per_feature: dict[str, int] = {
            col: X[col].nunique() for col in X.columns
        }
        
        # Lưu dữ liệu huấn luyện theo từng lớp để dùng khi predict
        self.X_by_class: dict = {
            _class: X[y == _class] for _class in self.classes
        }
    
    def _pxik_feature_per_class(
        self, 
        X_col: pd.Series, 
        xi: object, 
        num_nomials_M: int,
    ) -> float:
        """Tính xác suất P(xi | y = k) theo Laplace smoothing.
        
        P(xi | y=k) = (|{xm in class k : x_im = xi}| + alpha) / (Nk + M * alpha)
        
        Args:
            X_col: cột feature của các mẫu thuộc lớp k
            xi: giá trị feature cần tính xác suất
            num_nomials_M: số giá trị phân biệt M của feature tương ứng
            
        Returns:
            Xác suất P(xi | y = k)
        """
        count: int = (X_col == xi).sum()
        return (count + self.laplace_smoothing_alpha) / (
            len(X_col) + num_nomials_M * self.laplace_smoothing_alpha
        )
    
    def _predict_mnb_category_class(self, X: pd.DataFrame) -> np.ndarray:
        """Dự đoán lớp cho dữ liệu dạng category.
        
        y = argmax_k P(k) * prod_i P(xi | k)
        Sử dụng log để tránh underflow.
        
        Args:
            X: dữ liệu đầu vào (pd.DataFrame), mỗi hàng là một mẫu cần dự đoán
            
        Returns:
            Mảng numpy chứa nhãn lớp dự đoán cho từng mẫu
        """
        pred: list = []
        feature_cols: list[str] = list(self.X_by_class[self.classes[0]].columns)
        
        for _, x_row in X.iterrows():
            scores: np.ndarray = np.array(
                [np.log(float(self.p_c[_class])) for _class in self.classes]
            )
            
            for k, _class in enumerate(self.classes):
                X_k: pd.DataFrame = self.X_by_class[_class]
                for i, col in enumerate(feature_cols):
                    M: int = int(self.num_nomials_per_feature[col])
                    scores[k] += np.log(
                        self._pxik_feature_per_class(X_k[col], x_row.iloc[i], M)
                    )
            
            max_score_class_idx: int = int(np.argmax(scores))
            pred.append(self.classes[max_score_class_idx])
        
        return np.array(pred)

