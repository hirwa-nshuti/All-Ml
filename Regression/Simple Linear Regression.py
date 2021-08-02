#!/usr/bin/env python
# coding: utf-8

# A simple model using Scikit Learn to predict the CO2 Emmision of cars with different parameters.

# ## Importing the Required Libraries to use

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## About the dataset

# ### `FuelConsumption.csv`:
# 
# We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.
# -   **MODELYEAR** e.g. 2014
# -   **MAKE** e.g. Acura
# -   **MODEL** e.g. ILX
# -   **VEHICLE CLASS** e.g. SUV
# -   **ENGINE SIZE** e.g. 4.7
# -   **CYLINDERS** e.g 6
# -   **TRANSMISSION** e.g. A6
# -   **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# -   **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# -   **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# -   **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0
# 

# ## Loading the Dataset

# In[2]:


df = pd.read_csv("FuelConsumption.csv")

df.head()


# In[3]:


# summarize the data
df.describe()


# ### Feature selection

# In[4]:


data = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
data.head(10)


# In[5]:


viz = data[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# In[6]:


x=(data.drop(["CO2EMISSIONS"],1)).values
y=(data["CO2EMISSIONS"]).values
x.shape


# ### Creating training and test data from our dataset

# In[7]:


from sklearn.model_selection import train_test_split
#Splitting the data into training and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
print('Train Set', x_train.shape, y_train.shape)
print('Test Set', x_test.shape, y_test.shape)


# ### Building the model

# In[8]:


from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(x_train, y_train)
#The Coefficients
print('Coefficients: ', model.coef_)
print('The Intercept: ', model.intercept_)


# ### Drawing Regression Line

# In[9]:


# Reshaping y array to fit the line
y = np.repeat(y[..., np.newaxis], 3, -1)
y.shape


# In[10]:


# Plotting the fit line
regressionLine = model.coef_*x+model.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, regressionLine)
plt.show()


# ### Model Evaluation

# In[11]:


from sklearn.metrics import r2_score
y_hat=model.predict(x_test)
print('Mean absolute error: %.2f' %np.mean(np.absolute(y_hat - y_test)) )
print('Residual sum of squares(MSE): %.2f' %np.mean((y_hat - y_test)**2))
print('R_2 score: %.2f' %r2_score(y_hat, y_test))

