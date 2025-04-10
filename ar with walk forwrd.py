#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


# In[2]:


#load the dataset (assuming daily min temprature .csv with date and temp columns )
df=pd.read_csv('daily-min-temperatures.xls',parse_dates=['Date'],index_col='Date')
data=df['Temp'].values


# In[3]:


train_size = int(len(data)*0.8)
train, test = data[:train_size], data[train_size:]


# In[4]:


#walk forward validation 
history=train.tolist()#convert train set to a lost for dyanmic update 
predictions=[]
for t in test:
    #fit AR model 
    model= AutoReg(history,lags=7)#using last 7 days for autoregression 
    model_fit=model.fit()
    #predict the next value 
    y_pred=model_fit.predict(start=len(history),end=len(history))[0]
    predictions.append(y_pred)
    #updtae history wit actual observation 
    history.append(t)


# In[ ]:


#steps 
#convert the train dataset into history
#loop over the test dataset ,processing one observation at a time
#train AR model using the available history
#predict the  next value and store it
#uodate history with actual value 
#repeat for all test value


# In[7]:


#evaluate performance 
rmse=np.sqrt(mean_squared_error(test,predictions))
print(f'Walk-Forward Validation RMSE:{rmse:.4f}')



# In[8]:


#PLOT actual vs predicted value
plt.figure(figsize=(10,5))
plt.plot(test,label='Actual Temprature',marker='o')
plt.plot(predictions,label='Predicted Temprature',marker='x')
plt.xlabel('Days')
plt.ylabel('Temprature')
plt.title('AR Mode=Walk-Forward Validation')
plt.grid(True)
plt.show()


# # moving average model 

# In[9]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


# In[10]:


df=pd.read_csv('daily-min-temperatures.xls',parse_dates=['Date'])


# In[11]:


df['t']=df['Temp'].shift(1)


# In[12]:


df['Resid']=df['Temp'] -df['t']


# In[13]:


df.head()


# In[ ]:




