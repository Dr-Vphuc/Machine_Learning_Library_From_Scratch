from vplearn.classification.naive_bayes import BernouliNB

import pandas as pd

x1 = (1, 1, 1, 0, 0, 0, 0, 0, 0)
x2 = (1, 1, 0, 1, 1, 0, 0, 0, 0)
x3 = (0, 1, 0, 0, 1, 1, 0, 0, 0)
x4 = (0, 1, 0, 0, 0, 0, 1, 1, 1)

X_train = pd.DataFrame([x1, x2, x3, x4])
X_train = X_train.to_numpy()
y_train = pd.Series(['B', 'B', "B", "N"])
X_test = pd.DataFrame(
    [
        (1, 1, 0, 0, 0, 0, 0, 1, 1)
        # (1, 1, 0, 0, 0, 0, 0, 0, 0)
    ]
)  
# X_test = pd.DataFrame([1, 0, 0, 1, 0, 0, 0, 1, 0])

bnb = BernouliNB()
bnb.fit(X_train, y_train)
pred = bnb.predict(X_test)
print(pred)
