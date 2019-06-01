
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics.cluster import silhouette_score
import pickle


# In[3]:


features=pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\features_corr.csv')
#features=pd.read_csv(r'C:\Users\vznam\Downloads\PredictingRatings-master\data\features_corr.csv')
features.head()


# In[4]:


elos=pd.read_csv(r'C:\Users\Asus\PredictingRatings\data\clean_elos.csv')
#elos=pd.read_csv(r'C:\Users\vznam\Downloads\PredictingRatings-master\data\clean_elos.csv')
elos.head()


# In[5]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(elos)
y_kmeans = kmeans.predict(elos)


# In[6]:


with open('kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans,f) 


# In[7]:


len(y_kmeans),y_kmeans


# In[8]:


silhouette_kmeans = silhouette_score(elos,y_kmeans)
silhouette_kmeans


# In[9]:


kmeans


# In[11]:


kmeans1 = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=5000,
       n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=4, tol=0.0001, verbose=0)
                 
kmeans1.fit(elos)
y_kmeans1 = kmeans1.predict(elos)
silhouette_kmeans1 = silhouette_score(elos,y_kmeans1)
silhouette_kmeans1        


# In[24]:


dbskan = DBSCAN(eps=3, min_samples=2).fit(elos)
y_dbskan = dbskan.labels_


# In[25]:


silhouette_dbskan = silhouette_score(elos,y_dbskan)
silhouette_dbskan


# In[13]:


agg = AgglomerativeClustering(n_clusters=4)
agg.fit(elos)


# In[20]:


y_agg = agg.labels_

agg.n_clusters_


# In[21]:


silhouette_agg = silhouette_score(elos,y_agg)
silhouette_agg


# In[12]:


elos['Class'] = y_kmeans
elos.head()


# In[14]:


elos_1=elos[elos['Class']==0]
elos_1.shape


# In[15]:


elos_2=elos[elos['Class']==1]
elos_2.shape


# In[16]:


elos_3=elos[elos['Class']==2]
elos_3.shape


# In[19]:


elos_4=elos[elos['Class']==3]
elos_4.shape


# In[20]:


elos_1.shape[0]+elos_2.shape[0]+elos_3.shape[0]+elos_4.shape[0]


# In[25]:


elos_1.to_csv(r'C:\Users\Asus\PredictingRatings\data\elos1.csv',sep=',',index=False)
elos_2.to_csv(r'C:\Users\Asus\PredictingRatings\data\elos2.csv',sep=',',index=False)
elos_3.to_csv(r'C:\Users\Asus\PredictingRatings\data\elos3.csv',sep=',',index=False)
elos_4.to_csv(r'C:\Users\Asus\PredictingRatings\data\elos4.csv',sep=',',index=False)


# In[22]:


features_1=features[features.index.isin(elos_1.index)]
features_1.shape


# In[23]:


features_2=features[features.index.isin(elos_2.index)]
features_3=features[features.index.isin(elos_3.index)]
features_4=features[features.index.isin(elos_4.index)]


# In[28]:


features_1.to_csv(r'C:\Users\Asus\PredictingRatings\data\features1.csv',sep=',',index=False)
features_2.to_csv(r'C:\Users\Asus\PredictingRatings\data\features2.csv',sep=',',index=False)
features_3.to_csv(r'C:\Users\Asus\PredictingRatings\data\features3.csv',sep=',',index=False)
features_4.to_csv(r'C:\Users\Asus\PredictingRatings\data\features4.csv',sep=',',index=False)


# In[27]:


features_1.shape

