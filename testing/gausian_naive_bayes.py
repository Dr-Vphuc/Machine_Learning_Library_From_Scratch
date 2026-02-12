import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import seaborn as sns

from vplearn.classification.naive_bayes import GaussianNB
from vplearn.preprocessing import Encoding

data = sns.load_dataset('iris')
train = pd.concat([data[:40], data[50:90], data[100:140]])
test = pd.concat([data[40:50], data[90:100], data[140:150]])

ec = Encoding()

X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]
y_train = ec.label_encoding(y_train)
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]
y_test = ec.label_encoding(y_test)

# print(y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)
print(pred)