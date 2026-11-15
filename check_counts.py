import pandas as pd

df = pd.read_csv('training_setA.csv')
features = ['HR', 'MAP', 'Resp', 'Lactate', 'Age', 'HospAdmTime', 'ICULOS']
label = 'SepsisLabel'
df_clean = df.dropna(subset=features + [label])
print('Rows after dropna:', len(df_clean))
print('Patients after dropna:', df_clean['Patient_ID'].nunique())