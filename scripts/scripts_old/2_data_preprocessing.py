import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline#импортирую бибилотеку трубы
from sklearn.impute import SimpleImputer# Одномерный импутер для заполнения пропущенных значений с помощью простых стратегий. 
from sklearn.model_selection import train_test_split
df = pd.read_csv('train.csv', index_col='index')
df1 = pd.read_csv('train_mod1.csv', index_col='index')


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['churn'],axis=1), df['churn'], stratify=df['churn'], test_size=0.2,random_state=0
    )
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    df1.drop(['churn'],axis=1), df1['churn'], stratify=df1['churn'], test_size=0.2,random_state=0
    )  

standart_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    
    ('scaler', StandardScaler())
])

X_train = standart_pipe.fit_transform(X_train)# проводим трансформирование
X_test = standart_pipe.fit_transform(X_test)# проводим трансформирование

X_train = pd.DataFrame(X_train)#иначе в csv не запихать
X_test = pd.DataFrame(X_test)

X_test.to_csv('X_test.csv', index_label='index')
y_test.to_csv('y_test.csv', index_label='index') 
X_train.to_csv('X_train.csv', index_label='index')
y_train.to_csv('y_train.csv', index_label='index')
               
               #повторяем второй датасет

X_train1 = standart_pipe.fit_transform(X_train1)# проводим трансформирование
X_test1 = standart_pipe.fit_transform(X_test1)# проводим трансформирование
X_train1 = pd.DataFrame(X_train1)#иначе в csv не запихать
X_test1 = pd.DataFrame(X_test1)

X_test1.to_csv('X_test1.csv', index_label='index')
y_test1.to_csv('y_test1.csv', index_label='index') 
X_train1.to_csv('X_train1.csv', index_label='index')
y_train1.to_csv('y_train1.csv', index_label='index') 
print('data_preprocessing исполнен')