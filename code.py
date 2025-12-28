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
# Get all the values except last 'n' rows
y = y[:-predictionDays]
print(y)
# Split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2)
# set the predictionDays array equal to last 30 rows from the original data set
predictionDays_array = np.array(df.drop(['Prediction'], axis=1))[-predictionDays:]
print(predictionDays_array)
from sklearn.svm import SVR
# Create and Train the Support Vector Machine (Regression) using radial basis function
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
svr_rbf.fit(xtrain, ytrain)
svr_rbf_confidence = svr_rbf.score(xtest,ytest)
print('SVR_RBF accuracy :',svr_rbf_confidence)
# print the predicted values
svm_prediction = svr_rbf.predict(xtest)
# Print the model predictions for the next 30 days
svm_prediction = svr_rbf.predict(predictionDays_array)
print(svm_prediction)
print()
#Print the actual price for bitcoin for last 30 days
print(df.tail(predictionDays))print(svm_prediction)
