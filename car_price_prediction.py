#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('C:/Users/sourabh/Desktop/car prediction/car data.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[8]:


df['Years_old'] = 2020-df['Year']


# In[10]:


df.drop(['Car_Name','Year'],axis=1,inplace=True)


# In[11]:


df.head()


# In[12]:


df1 = pd.get_dummies(df,drop_first=True)


# In[13]:


df1.head()


# In[15]:


#plots
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


sns.pairplot(df1)


# In[17]:


#get correlations of each features in dataset
corrmat = df1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df1[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[18]:


X = df1.iloc[:,1:]
Y = df1.iloc[:,0]


# In[19]:


X.head()


# In[20]:


Y.head()


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[23]:


from sklearn.ensemble import RandomForestRegressor
# First create the base model to tune
rf = RandomForestRegressor()


# In[26]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np


# In[27]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[28]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[29]:


# Random search of parameters, using 5 fold cross validation
#using diff hyperparams
rf_random_cv = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[31]:


rf_random_cv.fit(X_train,Y_train)


# In[32]:


rf_random_cv.best_params_


# In[34]:


Y_pred = rf_random_cv.predict(X_test)


# In[35]:


from sklearn import metrics


# In[36]:


print('MAE:', metrics.mean_absolute_error(Y_test, Y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


# In[37]:


df_new = pd.DataFrame({'Actual':Y_test,"Predicted":Y_pred})
df_new.head()


# In[39]:


from sklearn.metrics import r2_score
R2 = r2_score(Y_test,Y_pred)
R2


# In[41]:



import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random_cv, file)


# In[ ]:




