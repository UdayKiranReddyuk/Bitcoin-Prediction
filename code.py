 
import numpy as np
import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/Project AIML/bitcoin.csv")
df.head()
 
df.drop(['Date'], axis=1, inplace=True)

predictionDays = 30

df['Prediction'] = df[['Price']].shift(-predictionDays)

df.head()

df.tail() 

x = np.array(df.drop(['Prediction'], axis=1)) 

x = x[:len(df)-predictionDays]
print(x)

y = np.array(df['Prediction'])

y = y[:-predictionDays]
print(y)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2)

predictionDays_array = np.array(df.drop(['Prediction'], axis=1))[-predictionDays:]
print(predictionDays_array)
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
svr_rbf.fit(xtrain, ytrain)
svr_rbf_confidence = svr_rbf.score(xtest,ytest)
print('SVR_RBF accuracy :',svr_rbf_confidence)

svm_prediction = svr_rbf.predict(xtest)

svm_prediction = svr_rbf.predict(predictionDays_array)
print(svm_prediction)
print()

print(df.tail(predictionDays))print(svm_prediction)
