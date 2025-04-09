#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[7]:


#load the shampoo sales dataset
data=pd.read_csv('shampoo (1).xls',usecols=[1],names=["sales"],header=0)


# In[8]:


data


# In[9]:


## convert to time series format 
data.index =pd.date_range(start='1901-01', periods=len(data), freq='M')
# convert the data set into a time series format with a monthly frequancy
data


# In[4]:


#visualize the data 
plt.figure(figsize=(10, 5))
plt.plot(data, marker= 'o',linestyle="-")
plt.title("shampoo sales Over time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.show()


# In[5]:


#check stationarity using ADF test
result = adfuller(data['sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] <= 0.05:
    print('The data is non-stationary. Differencing is required.')
else:
    print('The data is stationary.')


# In[6]:


#differencing  to make te data sationary 
data_diff=data.diff().dropna()


# In[7]:


# plot AFC and PACF to determine p,q values
fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # Remove the extra '-' before the assignment operator
plot_acf(data_diff, lags=15, ax=axes[0])
plot_pacf(data_diff, lags=15, method='ywm', ax=axes[1])
axes[0].set_title("ACF Plot")
axes[1].set_title("PACF Plot")
plt.show()


# In[8]:


# Fit ARIMA model (p,d,q) = (5,1,0) based on ACF/PACF analysis
model =ARIMA(data, order=(5,1,0))
model_fit = model.fit()


# In[9]:


#Print model summary
print(model_fit.summary())
#Display AR coefficients, standard errors, p values


# In[10]:


# Forecasting future values 
forecast_step =12 #predict next 12 month 
forecast =model_fit.forecast(steps=forecast_step)
#forecast 12 future time points (next 12 month )
#forecast ()generates predicted values based


# In[11]:


plt.figure(figsize=(10, 5))
plt.plot(data, label='Actual Sales')
plt.plot(pd.date_range(start=data.index[-1], periods=forecast_step+1, freq='M')[1:], forecast, label='Forecast', color='red')
plt.title("Shampoo Sales Forecast using ARIMA")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()


# # Walk forward ARIMA

# In[12]:


df = pd.read_csv("shampoo (1).xls",header=0, parse_dates=[0])


# In[13]:


data = df['Sales'].values


# In[14]:


train_size =int(len(data) *0.8)
train, test =data[:train_size], data[train_size:]


# In[15]:


# walk forward validation 
history = train.tolist() #Convert train set toa list for dynamic update
predictions= []
for t in test:
    #fit AR model
    model = ARIMA(data, order=(5,1,0)) #using last 7days for adtoregression
    model_fit = model.fit()
    
    # predict next value
    y_pred = model_fit.predict(start=len(history),end=len(history))[0]
    predictions.append(y_pred)
    # update history with actual observation
    history.append(t)


# In[16]:


# evaluate performance 
from sklearn.metrics import mean_squared_error
rmse =np.sqrt(mean_squared_error(test, predictions))
print(f'Walk-Forward Validation RMSE: {rmse:.4f}')


# In[17]:


#Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(test, label='Actual Temperature', marker='o')
plt.plot(predictions, label='Predicted Temperature', marker='x', linestyle='dashed')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.title('AR Model - Walk Forward Validation')
plt.legend()
plt.show()


# In[ ]:




