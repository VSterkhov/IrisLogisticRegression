
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris
from logistic_regression import LogisticRegression

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

irisBunch = load_iris()
data = np.c_[irisBunch.data, irisBunch.target]
columns = np.append(irisBunch.feature_names, ["target"])
irisDF = pd.DataFrame(data, columns=columns)
irisDF = irisDF[irisDF['target']!=0]

targetData = pd.get_dummies(irisDF.loc[:, irisBunch.feature_names])
Y = irisDF['target']
Y[Y==1]=0
Y[Y==2]=1
X_train, X_test, Y_train, Y_test = train_test_split(targetData, Y, test_size=0.2)

Y_train = Y_train.array
Y_test = Y_test.array

logReg = LogisticRegression(X_train)

w, b, losses = logReg.train(X_train, Y_train, X_train.shape[0], 100, 0.01, RMSprop=False)

plt.title('Gradient descent')
plt.plot(losses)
plt.ylabel('losses')
plt.xlabel('epoches')
plt.show()

pred = logReg.predict(X_test)
acc = logReg.accuracy(Y_test, pred)

print('Accuracy Y_test with predictions: ', acc)

w, b, losses = logReg.train(X_train, Y_train, X_train.shape[0], 100, 0.01, RMSprop=True)

plt.title('RMSprop')
plt.plot(losses)
plt.ylabel('losses')
plt.xlabel('epoches')
plt.show()


pred = logReg.predict(X_test)
acc = logReg.accuracy(Y_test, pred)

print('Accuracy Y_test with predictions: ', acc)