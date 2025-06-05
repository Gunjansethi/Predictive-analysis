#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


# In[2]:


df= pd.read_csv("KOTAKBANK.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.isnull()


# In[10]:


#Parse dates and keep useful columns
df["Date"] = pd.to_datetime(df["Date"])
df = df[["Date", "Close", "Volume"]].sort_values("Date")


# In[11]:


# ⿣  Set index & re-sample to business-day freq (fills non-trading days)
ts = (df.set_index("Date")
      .asfreq("B")                      # "B" = business day
      .ffill())                          # forward-fill weekends/holidays


# In[12]:


# ⿤  Train/test split (last 1 year as hold-out)
train = ts.iloc[:-252]   # ≈252 trading days
test  = ts.iloc[-252:]

y_train, y_test = train["Close"], test["Close"]


# In[13]:


# ⿥  Exogenous regressor
exog_train = train["Volume"]
exog_test  = test["Volume"]



# In[14]:


## 2  Quick diagnostics (optional but exam-friendly)
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(6,3))
autocorrelation_plot(y_train); plt.title("ACF of Close Price")
plt.show()


# In[15]:


## 3  Model 1 – ARIMA (p,d,q)
# Basic parameters chosen by quick AIC scan; tweak if you like
arima_model = ARIMA(y_train, order=(5,1,0)).fit()
arima_pred  = arima_model.forecast(steps=len(test))
print("ARIMA MAE:", mean_absolute_error(y_test, arima_pred))


# In[ ]:


## 5  Model 3 – SARIMA (p,d,q)×(P,D,Q,s)
# Assume yearly seasonality of 252 trading days
sarima_model = SARIMAX(y_train,
                       order=(2,1,2),
                       seasonal_order=(1,0,1,252),
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit()
sarima_pred = sarima_model.forecast(len(test))
print("SARIMA MAE:", mean_absolute_error(y_test, sarima_pred))


# In[ ]:


## 7  Model 5 – Prophet
# Prophet needs two columns: ds (date) and y (value)
prophet_train = y_train.reset_index().rename(columns={"Date":"ds","Close":"y"})
prophet = Prophet(daily_seasonality=False, yearly_seasonality=True)
prophet.fit(prophet_train)

future = prophet.make_future_dataframe(periods=len(test), freq="B")
forecast = prophet.predict(future)

prophet_pred = forecast.set_index("ds")["yhat"].iloc[-len(test):]
print("Prophet MAE:", mean_absolute_error(y_test, prophet_pred))


# In[ ]:


## 8  One-look plot
plt.figure(figsize=(8,4))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, sarimax_pred, label="SARIMAX Forecast")
plt.legend(); plt.title("Actual vs Forecast")
plt.show()


# In[ ]:





# In[ ]:




