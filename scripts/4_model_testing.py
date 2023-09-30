import pandas as pd
from sklearn.metrics import f1_score
from joblib import load

knn = load('knn.joblib') 


X_test = pd.read_csv('X_test.csv', index_col='index')
y_test_pd = pd.read_csv('y_test.csv', index_col='index')

y_test = y_test_pd['churn']

model = knn.predict(X_test)


print('Model test f1-score is: ',f1_score(y_test, model))