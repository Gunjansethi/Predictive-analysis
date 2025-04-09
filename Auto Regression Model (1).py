#!/usr/bin/env python
# coding: utf-8

# In[1]:


# auto correlation: which means it includes indirect relation also.
# partial auto correlation: 155min pe itna temperature hai.
#     so in python we use partial bcoz hm dusre kis factor pe nhi ja rhe hai.


# In[3]:


#### Interview questions

# we will go to the PACF not go to the ACF
# bcoz isme hm past wale ka correlation check kr rhe hai.

# for more search on gpt or ask from mam


# In[4]:


# ADF test,
# what is ACF 
# what is PACF


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


# In[6]:


df = pd.read_csv('daily-min-temperatures.xls', parse_dates=['Date'], index_col='Date')


# In[7]:


X=df.values
print('Shape of data /t',df.shape)
print('Original Dataset:\n',df.head())
print('After Extracting only temperatures:\n',X)


# In[8]:


df.plot()


# In[9]:


df[:200].plot()


# In[10]:


# No evident 


# # ADF Test

# In[11]:


from statsmodels.tsa.stattools import adfuller

dftest = adfuller(df['Temp'], autolag = 'AIC')

print("1. ADF : ",dftest[0])
print("2. P-Value :", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Usd for ADF Regression andd critical values Calculation : ", dftest[3])
print("5. Critical Values : ",)
for key, val in dftest[4].items():
    print("\t",key, ": ",val)


# In[12]:


# p-value = 0.000247 hai, jo 0.05 se chhoti hai. Matlab, hum null hypothesis (H₀) ko reject kar sakte hain.

# Interpretation
# ✅ Null Hypothesis (H₀): Time series me unit root hai → Non-Stationary hai.
# ✅ Alternative Hypothesis (H₁): Time series me unit root nahi hai → Stationary hai.

# Kyuki p-value < 0.05, iska matlab series stationary hai 🚀.


# In[13]:


# it is not mandatory to find all.
# we can only check by using p value


# In[14]:


# // important: stationary means constant mean and variance 


# In[ ]:





# In[15]:


## 1. Persistence Model (Naive Forecast)


# The simplest time series method.
# assuems that the next value in the time seeries is equal to the last observed value.


# In[16]:


# random walk me hme ek age ka value pta hota hai.
# noise walk me hme kuch bhi pta ni hota.
# wer can't predict anything..


# In[17]:


# 2. AR

# A statistical model that predicts future values using past values with a weighted combinatioin of previous time steps.

# uses lagged observations as predictors.

# here we have given the window. like window 2, window 3.


# Example stock past 3 days.


# In[18]:


# # when to use which model


# persistance highly stable
# ar -> strong correlation of present data to the past.
# ex: t has strong correlattion to the t-1.


# persistance and ar both are correlatioin
# and ye acf and pcf se find krte hai.


# In[19]:


# acf depends on direct and indirect factors also.

# t ki values t pe bhi depend ahi.
# and previous lags pe bhi.
# yt-1, yt-2 is lag 2


# In[20]:


# pacf


# meansures correaltioin between time sereis and a specific lag after removing the influence of specific lags.

# means t jo hai wo yt-1 PE BHI DEPEND HO SKTA HAI

# YA PHIR YT-3 PE BHI. AND BICH KI VALUES PE NI.
# if is used to find ar model order.


# In[ ]:





# In[21]:


from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

pacf=plot_pacf(df['Temp'],lags=25)
acf=plot_acf(df['Temp'],lags=25)


# In[22]:


# Partial correlation measures the direct relationship between two variables while controlling 
# for the effect of other variables, whereas autocorrelation measures 
# the correlation of a variable with its past values over time.


# In[23]:


# fpacf p
# acf = q


# In[24]:


# pacf => ma
# acf => ar


# ## Split Dataset into Train and Test Testing: Last 7 days

# In[25]:


train=X[:len(X)-7]
test=X[len(X)-7:]


# In[26]:


model=AutoReg(train,lags=10)


# In[27]:


model_fit = model.fit()
print(model_fit.summary())


# In[28]:


# 0.05 < lag value 
# then significant


# In[ ]:





# In[29]:


print(len(train))


# In[30]:


print(len(test))


# ## Make Predictioned on Test Set and Compare

# In[31]:


pred = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)


# In[32]:


print(len(pred))


# In[33]:


print(len(test))


# In[34]:


import matplotlib.pyplot as plt

# Plot predictions vs actual values
plt.figure(figsize=(6,6))
plt.plot(pred, label='Prediction', linestyle='dashed')
plt.plot(test, label='Actual', linestyle='solid', color='red')
plt.legend()
plt.title('AutoRegression Model - Predicted vs Actual')
plt.show() 

# Print predicted values
print(pred)


# In[35]:


# //
print(pred)


# In[36]:


# //
print(test)


# ### Calcualte Error

# In[37]:


# // mycode

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Calculate errors
mae = mean_absolute_error(test, pred)
mse = mean_squared_error(test, pred)
rmse = np.sqrt(mse)

# Print error values
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


# In[38]:


test.shape


# In[39]:


pred.shape


# In[40]:


from sklearn.metrics import mean_squared_error


# In[42]:


rmse = np.sqrt(mean_squared_error(test, pred))


# In[ ]:





# In[ ]:





# In[ ]:




