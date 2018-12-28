'''
A starter's train and test datasets generated for playing with linear regression.
# Tags: Adoption; GetStarted; LinearRegressionData
'''

from sygmoid.imports import *
from sklearn.model_selection import train_test_split

X = []
y = []
for line in open('01-dummy-lr-data.csv'):
    d1, d2 = line.split(',')
    X.append(float(d1))
    y.append(float(d2))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.savetxt('02a-train-lr.csv', np.c_[X_train,y_train])
np.savetxt('02b-test-lr.csv', np.c_[X_test,y_test])
