import pandas as pd
from sklearn.model_selection import train_test_split
#в данном скрипте просто готвлю данные
df = pd.read_csv('https://raw.githubusercontent.com/m6129/UrFU_2022_python/main/Dolganov/2_%D1%81%D0%B5%D0%BC%D0%B5%D1%81%D1%82%D1%80/train.csv')


X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['churn'],axis=1), df['churn'], stratify=df['churn'], test_size=0.2,random_state=0
)

X_train = pd.DataFrame(X_train)#иначе в csv не запихать
X_test = pd.DataFrame(X_test)

X_test.to_csv('X_test.csv', index_label='index')
y_test.to_csv('y_test.csv', index_label='index') 
X_train.to_csv('X_train.csv', index_label='index')
y_train.to_csv('y_train.csv', index_label='index')