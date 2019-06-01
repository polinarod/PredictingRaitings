
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn


# In[2]:


elos=pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\clean_elos.csv')
#elos=pd.read_csv(r'C:\Users\vznam\Downloads\PredictingRatings-master\data\clean_elos.csv')
elos.head()


# In[3]:


features=pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\features.csv')
#features=pd.read_csv(r'C:\Users\vznam\Downloads\PredictingRatings-master\data\features.csv')

print(features.info())
print(features.shape)
features.describe()


# #### Поиск пропущенных значений

# In[6]:


from sklearn.impute import MissingIndicator

indicator = MissingIndicator(missing_values=np.NaN)
indicator = indicator.fit_transform(features)

indicator


# In[7]:


features.isnull().values.any()


# #### Стандартизация
# Стандартизация - это преобразование, которое центрирует данные путем удаления среднего значения каждого объекта, а затем масштабирует его путем деления (непостоянных) объектов на их стандартное отклонение . После стандартизации данных среднее значение будет равно нулю, а стандартное отклонение - одному.
# 
# Стандартизация может кардинально улучшить производительность моделей. Например, многие элементы, используемые в целевой функции алгоритма обучения (например, ядро RBF машин опорных векторов или регуляризаторы l1 и l2 линейных моделей), предполагают, что все объекты сосредоточены вокруг нуля и имеют дисперсию в том же порядке. Если у признака есть отклонение, которое на несколько порядков больше, чем у других, оно может доминировать в целевой функции и сделать оценщика неспособным учиться на других признаках правильно, как ожидалось.
# 
# В зависимости от ваших потребностей и данных, Sklearn предоставляет набор инструментов для масштабирования : StandardScaler , MinMaxScaler , MaxAbsScaler и RobustScaler .

# In[8]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
fscale=scaler.fit_transform(features)


# In[9]:


len(fscale), len(fscale[0])


# In[10]:


fscale[:5]


# In[11]:


features_scale = pd.DataFrame(fscale)
features_scale.head()


# In[12]:


from sklearn.preprocessing import RobustScaler
robust = RobustScaler(quantile_range = (0.1,0.9))
frobust=robust.fit_transform(features)


# In[13]:


len(frobust), len(frobust[0])


# In[14]:


frobust[:5]


# In[15]:


features_robust = pd.DataFrame(frobust)
features_robust.head()


# In[16]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
fminmax=minmax.fit_transform(features)


# In[17]:


fminmax[:5]


# In[16]:


features_minmax = pd.DataFrame(fminmax)
features_minmax.head()


# #### Нормализация
# Нормализация - это процесс масштабирования отдельных образцов для получения единичной нормы . В основных терминах вам необходимо нормализовать данные, когда алгоритм прогнозирует на основе взвешенных отношений, сформированных между точками данных. Масштабирование входных данных в единичные нормы является обычной операцией для классификации или кластеризации текста .
# 
# Одно из ключевых различий между масштабированием (например, стандартизацией) и нормализацией заключается в том, что нормализация является построчной операцией , в то время как масштабирование является столбцовой операцией.
# Хотя существует много других способов нормализации данных, sklearn предоставляет три нормы (значение, с которым сравниваются отдельные значения): l1 , l2 и макс . При создании нового экземпляра класса Normalizer вы можете указать желаемую норму в параметре norm .
# 
# Ниже формула для имеющихся норм обсуждаются и реализуются в коде Python - где результат представляет собой список знаменателей для каждого образца набора данных X  .

# In[18]:


from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
fnorm=normalizer.fit_transform(features)


# In[19]:


features_norm = pd.DataFrame(fnorm)
features_norm.head()


# #### mutual_info_regression
# Оценивает взаимную информацию для непрерывной целевой переменной.
# 
# Взаимная информация (MI)  между двумя случайными переменными является неотрицательным значением, которое измеряет зависимость между переменными. Она равен нулю тогда и только тогда, когда две случайные переменные являются независимыми, а более высокие значения означают более высокую зависимость.
# 
# Функция основана на непараметрических методах, основанных на оценке энтропии по расстояниям k-ближайших соседей.

# In[21]:


from sklearn.feature_selection import mutual_info_regression

mutual_info=mutual_info_regression(features,elos.SumElos)


# In[22]:


print(len(mutual_info))
mutual_info


# In[31]:


def get_dependency(info):
    for ind,val in enumerate(info):
        if val > 0.5:
            print(ind)


# In[24]:


mutual_info_mean=mutual_info_regression(features,elos.MeanElos)
print(len(mutual_info_mean))
mutual_info_mean


# In[25]:


mutual_info_diff=mutual_info_regression(features,elos.DiffElos)
print(len(mutual_info_diff))
mutual_info_diff


# In[32]:


print('SumElos')
get_dependency(mutual_info)
print('MeanElos')
get_dependency(mutual_info_mean)
print('DiffElos')
get_dependency(mutual_info_diff)


# #### f-regression
# Одномерные линейные регрессионные тесты.
# 
# Линейная модель для тестирования индивидуального эффекта каждого из множества регрессоров. Эта функция оценки используется в процедуре выбора функции, а не в процедуре выбора отдельно стоящей функции.
# 
# 2 этапа:
# 
# 1. Вычисляется корреляция между каждым регрессором и целью
# 2. Она преобразуется в F-оценку, а затем в р-значение.

# In[39]:


from sklearn.feature_selection import f_regression

pval_mean=f_regression(features,elos.MeanElos)[1]
pval_sum=f_regression(features,elos.SumElos)[1]
pval_diff=f_regression(features,elos.DiffElos)[1]
pval_white=f_regression(features,elos.WhiteElo)[1]
pval_black=f_regression(features,elos.BlackElo)[1]


# #### проверка коэффициентов регрессии
# 
# Вычисление индексов тех признаков, у которых p_value > 0.05, то есть они вероятно являются незначимыми

# In[78]:


def get_significance(pvals):
    res=[]
    for ind,val in enumerate(pvals):
        if val >= 0.05:
            res.append(ind)
    return res


# In[79]:


print('MeanElos')
ind_rej_mean=get_significance(pval_mean)
print(len(ind_rej_mean))
print('SumElos')
ind_rej_sum=get_significance(pval_sum)
print(len(ind_rej_sum))
print('DiffElos')
ind_rej_diff=get_significance(pval_diff)
print(len(ind_rej_diff))
print('WhiteElos')
ind_rej_white=get_significance(pval_white)
print(len(ind_rej_white))
print('BlackElos')
ind_rej_black=get_significance(pval_black)
print(len(ind_rej_black))


# In[80]:


features.columns


# In[81]:


for ind,val in enumerate(ind_rej_sum):
    print(features.columns[val])


# In[82]:


for ind,val in enumerate(ind_rej_mean):
    print(features.columns[val])


# In[83]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn import model_selection as ms
from scipy.stats import pearsonr

from warnings import simplefilter

rand_st=4
#simplefilter("ignore") 


# In[84]:


target_white=elos['WhiteElo']
target_black=elos['BlackElo']
target_mean=elos['MeanElos']
target_diff=elos['DiffElos']
target_sum=elos['SumElos']


# #### Baseline до удаления незначимых признаков

# In[85]:


start10 = time.time()
lr_10 = LinearRegression()
lr_10.fit(features, target_sum)
sum_pred = lr_10.predict(features)
full_time10=round(time.time() - start10,3)
print(full_time10, 'c')


# In[86]:


plt.figure(figsize=(8, 8))
plt.scatter(sum_pred,target_sum, color = 'b',alpha=0.5)
plt.xlabel('Модель 10 - predictions')
plt.ylabel('Сумма рейтингов (target)')
# Линия тренда
z = np.polyfit(sum_pred,target_sum, 1)
p = np.poly1d(z)
plt.plot(sum_pred,p(sum_pred),"r")
plt.show()


# In[87]:


mae10=mae(target_sum,sum_pred)
print ('MAE = {:.3f}'.format(mae10))
rmse10 = (mse(target_sum, sum_pred))**0.5
print ('RMSE = {:.3f}'.format(rmse10))
corr_coef10 = pearsonr(target_sum,sum_pred)
print("Correlation coefficient = {:.3f}".format(corr_coef10[0]))


# In[88]:


start9 = time.time()
lr_9 = LinearRegression()
lr_9.fit(features, target_diff)
diff_pred = lr_9.predict(features)
full_time9=round(time.time() - start9,3)
print(full_time9, 'c')


# In[89]:


plt.figure(figsize=(8, 8))
plt.scatter(diff_pred,target_diff, color = 'b',alpha=0.5)
plt.xlabel('Модель 9 - predictions')
plt.ylabel('Разница в рейтинге (target)')
# Линия тренда
z = np.polyfit(diff_pred,target_diff, 1)
p = np.poly1d(z)
plt.plot(diff_pred,p(diff_pred),"r")
plt.show()


# In[90]:


mae9=mae(target_diff,diff_pred)
print ('MAE = {:.3f}'.format(mae9))
rmse9 = (mse(target_diff, diff_pred))**0.5
print ('RMSE = {:.3f}'.format(rmse9))
corr_coef9 = pearsonr(target_diff,diff_pred)
print("Correlation coefficient = {:.3f}".format(corr_coef9[0]))


# In[91]:


def find_abs_error(pred,elos):
    er=0
    for ind, val in enumerate(pred):
        er+=elos.iloc[ind,0]-val[0]
        er+=elos.iloc[ind,1]-val[1]
    return round(er,5)

def find_mean_error(pred,elos):
    er=0
    count=0
    for ind, val in enumerate(pred):
        er+=elos.iloc[ind,0]-val[0]
        er+=elos.iloc[ind,1]-val[1]
        count+=2
    return round(er/count,5)


# In[93]:


white_elos1 = (sum_pred + diff_pred)/2
black_elos1 = (sum_pred-diff_pred)/2
pred_elos1=list(zip(white_elos1,black_elos1))
len(pred_elos1),pred_elos1[:5]


# In[94]:


find_abs_error(pred_elos1,elos),find_mean_error(pred_elos1,elos)


# #### Пробуем удалить незначимые по гипотезе выше признаки

# In[96]:


cols_reject=[]
for ind,val in enumerate(ind_rej_sum):
    cols_reject.append(features.columns[val])

len(cols_reject),features.shape


# In[98]:


features_new=features.drop(columns=cols_reject,inplace=False)


# In[99]:


features_new.shape


# In[100]:


start10 = time.time()
lr_10 = LinearRegression()
lr_10.fit(features_new, target_sum)
sum_pred = lr_10.predict(features_new)
full_time10=round(time.time() - start10,3)
print(full_time10, 'c')


# In[101]:


plt.figure(figsize=(8, 8))
plt.scatter(sum_pred,target_sum, color = 'b',alpha=0.5)
plt.xlabel('Модель 10 - predictions')
plt.ylabel('Сумма рейтингов (target)')
# Линия тренда
z = np.polyfit(sum_pred,target_sum, 1)
p = np.poly1d(z)
plt.plot(sum_pred,p(sum_pred),"r")
plt.show()


# In[102]:


mae10=mae(target_sum,sum_pred)
print ('MAE = {:.3f}'.format(mae10))
rmse10 = (mse(target_sum, sum_pred))**0.5
print ('RMSE = {:.3f}'.format(rmse10))
corr_coef10 = pearsonr(target_sum,sum_pred)
print("Correlation coefficient = {:.3f}".format(corr_coef10[0]))


# In[103]:


start9 = time.time()
lr_9 = LinearRegression()
lr_9.fit(features_new, target_diff)
diff_pred = lr_9.predict(features_new)
full_time9=round(time.time() - start9,3)
print(full_time9, 'c')


# In[104]:


plt.figure(figsize=(8, 8))
plt.scatter(diff_pred,target_diff, color = 'b',alpha=0.5)
plt.xlabel('Модель 9 - predictions')
plt.ylabel('Разница в рейтинге (target)')
# Линия тренда
z = np.polyfit(diff_pred,target_diff, 1)
p = np.poly1d(z)
plt.plot(diff_pred,p(diff_pred),"r")
plt.show()


# In[105]:


mae9=mae(target_diff,diff_pred)
print ('MAE = {:.3f}'.format(mae9))
rmse9 = (mse(target_diff, diff_pred))**0.5
print ('RMSE = {:.3f}'.format(rmse9))
corr_coef9 = pearsonr(target_diff,diff_pred)
print("Correlation coefficient = {:.3f}".format(corr_coef9[0]))


# In[106]:


def find_abs_error(pred,elos):
    er=0
    for ind, val in enumerate(pred):
        er+=elos.iloc[ind,0]-val[0]
        er+=elos.iloc[ind,1]-val[1]
    return round(er,5)

def find_mean_error(pred,elos):
    er=0
    count=0
    for ind, val in enumerate(pred):
        er+=elos.iloc[ind,0]-val[0]
        er+=elos.iloc[ind,1]-val[1]
        count+=2
    return round(er/count,5)


# In[107]:


white_elos1 = (sum_pred + diff_pred)/2
black_elos1 = (sum_pred-diff_pred)/2
pred_elos1=list(zip(white_elos1,black_elos1))
len(pred_elos1),pred_elos1[:5]


# In[108]:


find_abs_error(pred_elos1,elos),find_mean_error(pred_elos1,elos)


# In[26]:


from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel_fit = sel.fit(features)
transform=sel_fit.transform(features)
features_new=pd.DataFrame(transform)
features_new.shape


# In[29]:


inv_trans=sel_fit.inverse_transform(features_new)
inv_features=pd.DataFrame(inv_trans)
reject_features=inv_features.columns[(inv_features== 0).all()]
reject_features


# In[32]:


for ind,val in enumerate(reject_features):
    print(features.columns[val])


# ### Новое:
# Поиск корреляций между признаками и целевой переменной

# In[4]:


from scipy.stats import pearsonr

correlations = {}

target = elos['MeanElos']
features_name=features.columns.tolist()

for f in features_name:
    data_temp = features.copy()
    data_temp['MeanElos']=target
    x1 = data_temp[f].values
    x2 = data_temp['MeanElos'].values
    key = f + ' vs ' + 'MeanElos'
    correlations[key] = pearsonr(x1,x2)[0]
    
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index].head()


# In[17]:


correlations = {}

target = elos['DiffElos']
features_name=features.columns.tolist()

for f in features_name:
    data_temp = features.copy()
    data_temp['DiffElos']=target
    x1 = data_temp[f].values
    x2 = data_temp['DiffElos'].values
    key = f + ' vs ' + 'DiffElos'
    correlations[key] = pearsonr(x1,x2)[0]
    if f == 'NumMoves':
        print(key,correlations[key] )
    
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index].head()


# In[6]:


plt.plot(features.Result,elos['DiffElos'])
plt.xlabel('Result')
plt.ylabel('DiffELos')
plt.show()


# In[15]:


correlations = {}

target = elos['SumElos']
features_name=features.columns.tolist()

for f in features_name:
    data_temp = features.copy()
    data_temp['SumElos']=target
    x1 = data_temp[f].values
    x2 = data_temp['SumElos'].values
    key = f + ' vs ' + 'SumElos'
    correlations[key] = pearsonr(x1,x2)[0]
    
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index].head()


# In[8]:


correlations = {}

target = elos['WhiteElo']
features_name=features.columns.tolist()

for f in features_name:
    data_temp = features.copy()
    data_temp['WhiteElo']=target
    x1 = data_temp[f].values
    x2 = data_temp['WhiteElo'].values
    key = f + ' vs ' + 'WhiteElo'
    correlations[key] = pearsonr(x1,x2)[0]
    
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index].head()


# In[9]:


correlations = {}

target = elos['BlackElo']
features_name=features.columns.tolist()

for f in features_name:
    data_temp = features.copy()
    data_temp['BlackElo']=target
    x1 = data_temp[f].values
    x2 = data_temp['BlackElo'].values
    key = f + ' vs ' + 'BlackElo'
    correlations[key] = pearsonr(x1,x2)[0]
    
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index].head()

