from vplearn.classification.naive_bayes import MutinomialNB

import pandas as pd

x1 = (2, 1, 1, 0, 0, 0, 0, 0, 0)
x2 = (1, 1, 0, 1, 1, 0, 0, 0, 0)
x3 = (0, 1, 0, 0, 1, 1, 0, 0, 0)
x4 = (1, 0, 0, 0, 0, 0, 1, 1, 1)
# print([x1, x2, x3, x4])
X_train = pd.DataFrame([x1, x2, x3, x4])
X_train = X_train.to_numpy()
y_train = pd.Series([1, 1, 1, 0])
X_test = pd.DataFrame([(2, 0, 0, 1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, 0, 0, 1, 1)])
mnb = MutinomialNB()
mnb.fit(X_train, y_train)
pred = mnb.predict(X_test)
print(pred)