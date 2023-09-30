import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline#импортирую бибилотеку трубы
from sklearn.impute import SimpleImputer# Одномерный импутер для заполнения пропущенных значений с помощью простых стратегий. 


X_train = pd.read_csv('X_train.csv', index_col='index')
X_test = pd.read_csv('X_test.csv', index_col='index')


standart_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    
    ('scaler', StandardScaler())
])

X_train = standart_pipe.fit_transform(X_train)# проводим трансформирование
X_test = standart_pipe.fit_transform(X_test)# проводим трансформирование

X_train = pd.DataFrame(X_train)#иначе в csv не запихать
X_test = pd.DataFrame(X_test)

X_test.to_csv('X_test.csv', index_label='index')
X_train.to_csv('X_train.csv', index_label='index')