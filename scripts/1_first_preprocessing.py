import pandas as pd

X_train = pd.read_csv('/home/ml-srv/mlops/data/X_train.csv', index_col='index')
X_test = pd.read_csv('/home/ml-srv/mlops/data/X_test.csv', index_col='index')

X_train = X_train.drop(['state'],axis=1)
X_test = X_test.drop(['state'],axis=1)

X_train['international_plan'] = X_train['international_plan'].map({'yes': 1, 'no': 0})#провожу перекодировку признаков
X_train['voice_mail_plan'] = X_train['voice_mail_plan'].map({'yes': 1, 'no': 0})
X_train['area_code'] = X_train['area_code'].map({'area_code_415': 1, 'area_code_408': 0,'area_code_510':3})

X_test['international_plan'] = X_test['international_plan'].map({'yes': 1, 'no': 0})#провожу перекодировку признаков
X_test['voice_mail_plan'] = X_test['voice_mail_plan'].map({'yes': 1, 'no': 0})
X_test['area_code'] = X_test['area_code'].map({'area_code_415': 1, 'area_code_408': 0,'area_code_510':3})

X_train.to_csv('/home/ml-srv/mlops/data/X_train.csv', index_label='index')
X_test.to_csv('/home/ml-srv/mlops/data/X_test.csv', index_label='index')

print('first_preprocessing исполнен')