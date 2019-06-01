
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn import model_selection as ms
from scipy.stats import pearsonr

from sklearn.dummy import DummyRegressor


# In[3]:


elos=pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\clean_elos.csv')
elos.head()


# In[4]:


features=pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\features.csv')

print(features.info())
print(features.shape)
features.describe()


# In[6]:


target_white=elos['WhiteElo']
target_black=elos['BlackElo']
target_mean=elos['MeanElos']
target_diff=elos['DiffElos']
target_sum=elos['SumElos']


# In[7]:


def find_mean_error(pred,elos):
    er=0
    count=0
    for ind, val in enumerate(pred):
        er+=abs(elos.iloc[ind,0]-val[0])
        er+=abs(elos.iloc[ind,1]-val[1])
        count+=2
    return round(er/count,3)

def find_abs_error(pred,elos):
    er=0
    for ind, val in enumerate(pred):
        er+=abs(elos.iloc[ind,0]-val[0])
        er+=abs(elos.iloc[ind,1]-val[1])
    return round(er,3)

def find_root_mean_squared_error(pred,elos):
    er=0
    count=0
    for ind, val in enumerate(pred):
        er+=(elos.iloc[ind,0]-val[0])**2
        er+=(elos.iloc[ind,1]-val[1])**2
        count+=2
    return round((er/count)**0.5,3)


# In[20]:


#mean, median, quantile (0-1)
start = time.time()
dw = DummyRegressor('mean')
db = DummyRegressor('mean')
dw.fit(features, target_white)
db.fit(features, target_black)
white_pred = dw.predict(features)
black_pred =db.predict(features)
full_time=round((time.time() - start)/2,3)
print(full_time, 'c')


# In[21]:


print('Для белых:')
maew=mae(target_white,white_pred)
print ('MAE = {:.3f}'.format(maew))
rmsew = (mse(target_white, white_pred))**0.5
print ('RMSE = {:.3f}'.format(rmsew))
corr_coefw = pearsonr(target_white, white_pred)
print("Correlation coefficient = {:.3f}".format(corr_coefw[0]))
print('\nДля черных:')
maeb=mae(target_black,black_pred)
print ('MAE = {:.3f}'.format(maeb))
rmseb = (mse(target_black,black_pred))**0.5
print ('RMSE = {:.3f}'.format(rmseb))
corr_coefb = pearsonr(target_black, black_pred)
print("Correlation coefficient = {:.3f}".format(corr_coefb[0]))


# In[22]:


pred_elos=list(zip(white_pred,black_pred))
find_abs_error(pred_elos,elos),find_mean_error(pred_elos,elos),find_root_mean_squared_error(pred_elos,elos)


# In[23]:


pred_elos

