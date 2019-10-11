
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset=pd.read_csv('D:/MLAI/project1/diabetes.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 8].values


# In[3]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=123)


# In[4]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[12]:


y_pred = regressor.predict(X_test)


# In[6]:


regressor.score(X_test,y_test)


# In[7]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=6)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly, y)


# In[15]:


lin_reg_2.score(X_poly,y)

