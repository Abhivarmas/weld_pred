
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

df=pd.read_excel('//content//drive//MyDrive//Colab Notebooks//WeldRight Dataset.xlsx')

df['Defect'].value_counts()

df.columns

df1=df[['Current', 'Humidity', 'Temperature', 'Flow', 'Job Temp', 'Voltage', 'Defect']]

df2=df1[df1['Defect']=='No Defect']
df2=df2.sample(n=5000)

df3=df1[df1['Defect']=='Tungsten Inclusion']
df3=df3.sample(n=5000,replace=True,ignore_index=True)

df4=df1[df1['Defect']=='Porosity']
df4=df4.sample(n=5000,replace=True,ignore_index=True)

df_f=pd.concat([df2,df3,df4])

df_f=df_f.sample(frac=1)

df_f['Defect'].unique()

df_f['Defect']=df_f['Defect'].map({'No Defect':0, 'Tungsten Inclusion':1, 'Porosity':2})

df_f.isnull().sum()

df_f.dropna(inplace=True)

df_f['Current'].isnull().sum()

x=df_f.iloc[:,:-1].values
y=df_f.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

print(y_train.shape)
print(y_test.shape)

print(x_train.shape)
print(x_test.shape)

from sklearn.ensemble import RandomForestClassifier
ran_cl=RandomForestClassifier(n_estimators=400,criterion='gini',max_depth=21,min_samples_split=2,min_samples_leaf=2)
ran_cl.fit(x_train,y_train)
y_pred1=ran_cl.predict(x_test)

y_pred1_train=ran_cl.predict(x_train)

acc2=accuracy_score(y_pred1,y_test)
acc3=accuracy_score(y_pred1_train,y_train)


from sklearn.metrics import f1_score
f_score= f1_score(y_pred1,y_test,average='weighted')

import pickle

pickle.dump(ran_cl,open('saved_metal_model','wb'))

model=pickle.load(open('saved_metal_model','rb'))
model.predict(x_test)

