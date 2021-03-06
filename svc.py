import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# read dataset file
wine = pd.read_csv('winequality-red.csv')

# make quality scores to binary class. (0 is bad, 1 is good)
bins = (2.5, 5.9, 8.9)
group_names = [0, 1] # bad good
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)

print(wine['quality'].value_counts())

# X is feature, y is quality label
X = wine.drop('quality', axis=1)
y = wine['quality']

# make training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# make support vector classifier and fit & predict
svc = SVC(kernel='rbf', C=1.4)
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

# get classification report between prediction and test label
print(classification_report(y_test, pred_svc))

