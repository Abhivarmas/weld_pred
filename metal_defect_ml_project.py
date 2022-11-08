#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[ ]:


df=pd.read_excel('//content//drive//MyDrive//Colab Notebooks//WeldRight Dataset.xlsx')


# In[ ]:


df


# In[ ]:


df['Defect'].value_counts()


# In[ ]:


df.columns


# In[ ]:


df1=df[['Current', 'Humidity', 'Temperature', 'Flow', 'Job Temp', 'Voltage', 'Defect']]


# In[ ]:


df2=df1[df1['Defect']=='No Defect']
df2=df2.sample(n=5000)


# In[ ]:


df3=df1[df1['Defect']=='Tungsten Inclusion']
df3=df3.sample(n=5000,replace=True,ignore_index=True)


# In[ ]:


df4=df1[df1['Defect']=='Porosity']
df4=df4.sample(n=5000,replace=True,ignore_index=True)


# In[ ]:


df_f=pd.concat([df2,df3,df4])


# In[ ]:


df_f=df_f.sample(frac=1)


# In[ ]:


df_f


# In[ ]:


df_f['Defect'].unique()


# In[ ]:


df_f.shape


# In[ ]:


df_f['Defect']=df_f['Defect'].map({'No Defect':0, 'Tungsten Inclusion':1, 'Porosity':2})


# In[ ]:


df_f.isnull().sum()


# In[ ]:


df_f.dropna(inplace=True)


# In[ ]:


df_f['Current'].isnull().sum()


# In[ ]:


df_f


# In[ ]:


x=df_f.iloc[:,:-1].values
y=df_f.iloc[:,-1].values


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)


# In[ ]:


print(y_train.shape)
print(y_test.shape)


# In[ ]:


print(x_train.shape)
print(x_test.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
ran_cl=RandomForestClassifier(n_estimators=400,criterion='gini',max_depth=21,min_samples_split=2,min_samples_leaf=2)
ran_cl.fit(x_train,y_train)
y_pred1=ran_cl.predict(x_test)


# In[ ]:


y_pred1_train=ran_cl.predict(x_train)


# In[ ]:


acc2=accuracy_score(y_pred1,y_test)
acc2


# In[ ]:


acc3=accuracy_score(y_pred1_train,y_train)
acc3


# In[ ]:


from sklearn.metrics import f1_score
f_score= f1_score(y_pred1,y_test,average='weighted')
f_score


# In[ ]:


import pickle

pickle.dump(ran_cl,open('saved_metal_model','wb'))







# In[ ]:


model=pickle.load(open('saved_metal_model','rb'))


# In[ ]:


model.predict(x_test)

