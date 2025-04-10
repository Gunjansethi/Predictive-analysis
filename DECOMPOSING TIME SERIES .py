#!/usr/bin/env python
# coding: utf-8

# In[3]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[4]:


import pandas as pd 
import numpy as np 


# In[5]:


miles_decomp_df=pd.read_csv('us-airlines-monthly-aircraft-miles-flown (1).csv',header=0,parse_dates=[0])


# In[6]:


miles_decomp_df.head()


# In[7]:


miles_decomp_df.index= miles_decomp_df['Month']


# In[8]:


result = seasonal_decompose(miles_decomp_df['MilesMM'], model='additive')


# In[9]:


result.plot()


# In[10]:


result2 = seasonal_decompose(miles_decomp_df['MilesMM'], model='multiplicative')


# In[11]:


result2.plot()


# In[12]:


#difference of additive and multiplicative 
#why we remove trend and seasonality 
#changing of csv to time series


# # DIFFERENCING

# In[13]:


miles_df=pd.read_csv('us-airlines-monthly-aircraft-miles-flown (1).csv',header=0,parse_dates=[0])


# In[14]:


miles_df.head()


# In[15]:


miles_df['lag1'] = miles_df['MilesMM'].shift(1)


# In[16]:


miles_df['MilesMM_diff_1'] = miles_df['MilesMM'].diff(periods=1)


# In[17]:


miles_df.head()


# In[18]:


miles_df.index = miles_df['Month']
result_a=seasonal_decompose(miles_df['MilesMM'],model='additive')
result.plot()


# In[20]:


miles_df.index = miles_df['Month']
result_b=seasonal_decompose(miles_df.iloc[1:,3],model='additive')
result_b.plot()


# In[21]:


miles_df['MilesMM'].plot()


# In[22]:


miles_df['MilesMM_diff_12'] = miles_df['MilesMM'].diff(periods=12)


# In[23]:


miles_df['MilesMM_diff_12'].plot()


# In[24]:


result_c = seasonal_decompose(miles_df.iloc[13:,4],model='additive')
result_c.plot()


# In[ ]:


#train-test split


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




