import pandas as pd
from sklearn.metrics import f1_score
from joblib import load

knn = load('knn.joblib') 
knn1 = load('knn1.joblib') 

X_test = pd.read_csv('X_test.csv', index_col='index')
y_test_pd = pd.read_csv('y_test.csv', index_col='index')

y_test = y_test_pd['churn']

model = knn.predict(X_test)


X_test1 = pd.read_csv('X_test1.csv', index_col='index')
y_test_pd1 = pd.read_csv('y_test1.csv', index_col='index')

y_test1 = y_test_pd1['churn']

model1 = knn1.predict(X_test1)

print('Model test f1-score is: ',f1_score(y_test1, model1))