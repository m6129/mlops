import pandas as pd


df = pd.read_csv('train.csv')
df = df.drop(['state'],axis=1)

df['international_plan'] = df['international_plan'].map({'yes': 1, 'no': 0})#провожу перекодировку признаков
df['voice_mail_plan'] = df['voice_mail_plan'].map({'yes': 1, 'no': 0})
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})
df['area_code'] = df['area_code'].map({'area_code_415': 1, 'area_code_408': 0,'area_code_510':3})

df.to_csv('train_mod1.csv', index_label='index')

print('first_preprocessing исполнен')