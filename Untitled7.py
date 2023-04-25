#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pandas.read_excel("new_citation_synthetic_data2.xlsx")
df


# In[4]:


df.iloc[:, 3:-1]


# In[5]:


df['mean'] = df.iloc[:, 3:17].mean(axis=1)
df['stddev'] = df.iloc[:, 3:17].std(axis=1)

for i in range(len(df)) :
    for col in df.iloc[:, 3:17].columns:
        if df.loc[i, col] < (df.loc[i, "mean"] - 5 * df.loc[i, "stddev"]):
            df.loc[i, col] = df.loc[i, "mean"]


# In[6]:


df.iloc[:, 3:18] # the data set is being changed


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.nunique()


# In[10]:


df_temp = df.copy()

df_temp.drop(columns=["Paper ID","Journal ID","Year of Publishing", "mean", "stddev"], inplace=True, axis=1)
df_temp.info()


# In[11]:


# running models

X, y = df_temp.iloc[:, 0:14], df_temp.iloc[:, 14]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=5)


# In[12]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
class CompareModels:
    def __init__(self):
        import pandas as pd
        self._models = pd.DataFrame(
            data=['R^2', 'RMSE', 'MAE'],
            columns=['Model']
        ).set_index(keys='Model')
        
    def add(self, model_name, y_test, y_pred):
        import numpy as np
        self._models[model_name] = np.array(
            object=[
                r2_score(y_true=y_test, y_pred=y_pred), # R^2
                np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)), # RMSE
                mean_absolute_error(y_true=y_test, y_pred=y_pred), #MAE
                
            ]
        )
        
    def R2AndRMSE(y_test, y_pred):
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        return r2_score(y_true=y_test, y_pred=y_pred), np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)),mean_squared_error(y_true=y_test, y_pred=y_pred), mean_absolute_error(y_true=y_test, y_pred=y_pred)
    
    @property
    def models(self):
        return self._models
    
    @models.setter
    def models(self, _):
        print('Cannot perform such task.')
    
    def show(self, **kwargs):
        import matplotlib.pyplot as plt
        kwargs['marker'] = kwargs.get('marker', 'X')
        self._models.plot(**kwargs)
        plt.xticks(range(len(self._models)), self._models.index)
        plt.xlabel('')
        plt.axis('auto')
        plt.show()


# In[13]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
# prepare model

model.fit(x_train, y_train)
LinearRegression()
y_pred_lr = model.predict(x_test)

# checking accuracy and error
lr_R2, lr_RMSE, lr_MSE, lr_MAE = CompareModels.R2AndRMSE(y_test=y_test, y_pred=y_pred_lr)

print("R2: ",lr_R2)
print("RMSE: ",lr_RMSE)
print("MSE: ",lr_MSE)
print("MAE: ",lr_MAE)


# In[14]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error 
from IPython.core.pylabtools import figsize
plot = CompareModels()
plot.add('Linear Regression', y_test=y_test, y_pred=y_pred_lr)
plot.show(figsize = (5,5))


# In[15]:


from sklearn.preprocessing import PolynomialFeatures
p=PolynomialFeatures()
X_poly=p.fit_transform(x_test)
model.fit(X_poly, y_test)
y_pred_lr_pf=model.predict(X_poly)

# checking accuracy and error
lr_pf_R2, lr_pf_RMSE, lr_pf_MSE, lr_pf_MAE = CompareModels.R2AndRMSE(y_test=y_test, y_pred=y_pred_lr_pf)

print("R2: ",lr_pf_R2)
print("RMSE: ",lr_pf_RMSE)
print("MSE: ",lr_pf_MSE)
print("MAE: ",lr_pf_MAE)


# In[16]:


plot = CompareModels()
plot.add(model_name='Linear Regression With Polynomial Features', y_test=y_test, y_pred=y_pred_lr_pf)
plot.show(figsize=(5, 5))


# In[17]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=13, n_estimators=60, random_state=56, n_jobs=-1)

model.fit(x_train,y_train)

y_pred_rf = model.predict(x_test)

# checking accuracy and error
rf_R2, rf_RMSE, rf_MSE, rf_MAE = CompareModels.R2AndRMSE(y_test=y_test, y_pred=y_pred_rf)
print("R2: ",rf_R2)
print("RMSE: ",rf_RMSE)
print("MSE: ",rf_MSE)
print("MAE: ",rf_MAE)


# In[18]:


plot = CompareModels()
plot.add('Random Forest', y_test=y_test, y_pred=y_pred_rf)
plot.show(figsize = (5,5))


# In[19]:


from sklearn.svm import SVR
svr_linear = SVR(kernel='linear')
svr_linear.fit(x_train, y_train)
svr_linear.score(x_test, y_test)


# In[20]:


y_pred_svr = svr_linear.predict(x_test)


# In[21]:


# checking accuracy and error
svr_R2, svr_RMSE, svr_MSE, svr_MAE = CompareModels.R2AndRMSE(y_test=y_test, y_pred=y_pred_svr)
print("R2: ",svr_R2)
print("RMSE: ",svr_RMSE)
print("MSE: ",svr_MSE)
print("MAE: ",svr_MAE)


# In[22]:


plot = CompareModels()
plot.add('SVR', y_test, y_pred_svr)
plot.show(figsize=(5, 5))


# In[23]:


plot = CompareModels()
plot.add('Linear Regression', y_test=y_test, y_pred=y_pred_lr)
plot.add('Linear Regression With Polynomial Features', y_test=y_test, y_pred=y_pred_lr_pf)
plot.add('Random Forest', y_test=y_test, y_pred=y_pred_rf)
plot.add('SVR', y_test, y_pred_svr)
plot.show(figsize=(10, 7))


# In[30]:


# R2 score comparison
import matplotlib.pyplot as plt

x = np.array(["Linear Regression","Linear Regression With Polynomial Features","Random Forest", "Support Vector Machine"])
y = np.array([lr_R2,lr_pf_R2,rf_R2, svr_R2])
fig = plt.figure(figsize=(10, 5))
#plt.plot(x,y)
plt.bar(x, y,color='maroon')
plt.axis('auto')
plt.xlabel("Model Type")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison")
plt.show()


# In[32]:


# RMSE Comparison
x = np.array(["Linear Regression","Linear Regression With Polynomial Features", "Random Forest", "Support Vector Machine"])
y = np.array([lr_RMSE,lr_pf_RMSE, rf_RMSE, svr_RMSE])
fig = plt.figure(figsize=(10, 5))
plt.bar(x, y,color='maroon')
plt.axis('auto')
plt.xlabel("Model Type")
plt.ylabel("RMSE")
plt.title("RMSE Comparison")
plt.show()


# In[34]:


# MSE Comparison
x = np.array(["Linear Regression","Linear Regression With Polynomial Features","Random Forest", "Support Vector Machine"])
y = np.array([lr_MSE,lr_pf_MSE, rf_MSE, svr_MSE])
fig = plt.figure(figsize=(10, 5))
plt.bar(x, y,color='maroon')
plt.axis('auto')
plt.xlabel("Model Type")
plt.ylabel("MSE")
plt.title("MSE Comparison")
plt.show()


# In[35]:


# MAE Comparison
x = np.array(["Linear Regression","Linear Regression With Polynomial Features", "Random Forest", "Support Vector Machine"])
y = np.array([lr_MAE,lr_pf_MAE,rf_MAE, svr_MAE])
fig = plt.figure(figsize=(10, 5))
plt.bar(x, y,color='maroon')
plt.axis('auto')
plt.xlabel("Model Type")
plt.ylabel("MAE")
plt.title("MAE Comparison")
plt.show()


# In[ ]:




