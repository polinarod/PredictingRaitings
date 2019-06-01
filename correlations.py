
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


features=pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\features.csv')
features.head()


# In[4]:


corr=features.corr(method='pearson')
corr.head()


# In[8]:


indexes=corr.index
corr_feat=[]
corr_index=[]
corr_columns=[]
for index in corr.index:
    for column in corr.index:
        if index != column and abs(corr.loc[index,column])>=0.9:
            corr_feat.append((index,column))
            corr_index.append(index)
            corr_columns.append(column)
            
print(len(corr_feat))
corr_feat


# In[9]:


features_clean=features.drop(columns=corr_columns)
features_clean.shape


# In[12]:


corr.loc['NumMoves','FullMoves'],corr.loc['FullMoves','NumMoves']


# In[5]:


features=pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\features_new.csv')
features.head()


# In[6]:


corr=features.corr(method='pearson')
corr.head()


# In[7]:


indexes=corr.index
corr_feat=[]
corr_index=[]
corr_columns=[]
for index in corr.index:
    for column in corr.index:
        if index < column and abs(corr.loc[index,column])>=0.9:
            corr_feat.append((index,column))
            corr_index.append(index)
            corr_columns.append(column)
            
print(len(corr_feat))
corr_feat


# In[9]:


features_corr=features_new.drop(columns=corr_columns)
features_corr.shape


# In[10]:


corr1=features_corr.corr(method='pearson')
corr1.head()


# In[14]:


corr_feat1=[]
corr_index1=[]
corr_columns1=[]
for index in corr1.index:
    for column in corr1.columns:
        if index < column and abs(corr1.loc[index,column])>=0.8:
            corr_feat1.append((index,column))
            corr_index1.append(index)
            corr_columns1.append(column)
            
print(len(corr_feat1))
corr_feat1


# In[15]:


features_corr1=features_corr.drop(columns=corr_columns1)
features_corr1.shape


# In[16]:


features_corr1.to_csv(r'C:\Users\Asus\PredictingRatings\data\features_corr.csv', sep=',',index=False)

