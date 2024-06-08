#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# In[2]:


tickers = ['^GSPC','^NSEI','^HSI','^STOXX','^BVSP']
start = datetime.datetime(2008,1,1)
end = datetime.datetime(2023,12,31)
price_data = yf.download(tickers, start=start, end=end)['Adj Close']
price_data = price_data.fillna(method='backfill')
price_data = price_data.fillna(method='ffill')
price_data.columns = ['BOVESPA','SP500','HSI','NIFTY50','STOXX']


# In[3]:


def convert_prices_to_returns(prices):
    returns = prices.pct_change(1)
    returns.iloc[0,:] = 0
    return returns

daily_returns = convert_prices_to_returns(price_data)


# In[4]:


daily_returns_sq = daily_returns**2
daily_returns_abs = daily_returns.abs()


# In[5]:


dr_rolling_mean_20 = daily_returns.rolling(window=20).mean()
dr_rolling_sq_mean_20 = daily_returns_sq.rolling(window=20).mean()

dr_rolling_mean_50 = daily_returns.rolling(window=50).mean()
dr_rolling_sq_mean_50 = daily_returns_sq.rolling(window=50).mean()

dr_rolling_mean_120 = daily_returns.rolling(window=120).mean()
dr_rolling_sq_mean_120 = daily_returns_sq.rolling(window=120).mean()


# In[6]:


dr_rolling_std_20 = daily_returns.rolling(window=20).std()
dr_rolling_std_50 = daily_returns.rolling(window=20).std()
dr_rolling_std_120 = daily_returns.rolling(window=20).std()


# In[7]:


SP_BV = daily_returns['SP500'] * daily_returns['BOVESPA']
SP_HS = daily_returns['SP500'] * daily_returns['HSI']
SP_NI = daily_returns['SP500'] * daily_returns['NIFTY50']
SP_ST = daily_returns['SP500'] * daily_returns['STOXX']
BV_HS = daily_returns['BOVESPA'] * daily_returns['HSI']
BV_NI = daily_returns['BOVESPA'] * daily_returns['NIFTY50']
BV_ST = daily_returns['BOVESPA'] * daily_returns['STOXX']
HS_NI = daily_returns['HSI'] * daily_returns['NIFTY50']
HS_ST = daily_returns['HSI'] * daily_returns['STOXX']
NI_ST = daily_returns['NIFTY50'] * daily_returns['STOXX']


# In[8]:


SP_BV_abs = (daily_returns['SP500'] * daily_returns['BOVESPA']).abs()
SP_HS_abs = (daily_returns['SP500'] * daily_returns['HSI']).abs()
SP_NI_abs = (daily_returns['SP500'] * daily_returns['NIFTY50']).abs()
SP_ST_abs = (daily_returns['SP500'] * daily_returns['STOXX']).abs()
BV_HS_abs = (daily_returns['BOVESPA'] * daily_returns['HSI']).abs()
BV_NI_abs = (daily_returns['BOVESPA'] * daily_returns['NIFTY50']).abs()
BV_ST_abs = (daily_returns['BOVESPA'] * daily_returns['STOXX']).abs()
HS_NI_abs = (daily_returns['HSI'] * daily_returns['NIFTY50']).abs()
HS_ST_abs = (daily_returns['HSI'] * daily_returns['STOXX']).abs()
NI_ST_abs = (daily_returns['NIFTY50'] * daily_returns['STOXX']).abs()


# In[9]:


SP_BV_rm_20 = (daily_returns['SP500'] * daily_returns['BOVESPA']).rolling(window=20).mean()
SP_HS_rm_20 = (daily_returns['SP500'] * daily_returns['HSI']).rolling(window=20).mean()
SP_NI_rm_20 = (daily_returns['SP500'] * daily_returns['NIFTY50']).rolling(window=20).mean()
SP_ST_rm_20  = (daily_returns['SP500'] * daily_returns['STOXX']).rolling(window=20).mean()
BV_HS_rm_20 = (daily_returns['BOVESPA'] * daily_returns['HSI']).rolling(window=20).mean()
BV_NI_rm_20 = (daily_returns['BOVESPA'] * daily_returns['NIFTY50']).rolling(window=20).mean()
BV_ST_rm_20 = (daily_returns['BOVESPA'] * daily_returns['STOXX']).rolling(window=20).mean()
HS_NI_rm_20 = (daily_returns['HSI'] * daily_returns['NIFTY50']).rolling(window=20).mean()
HS_ST_rm_20 = (daily_returns['HSI'] * daily_returns['STOXX']).rolling(window=20).mean()
NI_ST_rm_20 = (daily_returns['NIFTY50'] * daily_returns['STOXX']).rolling(window=20).mean()


# In[10]:


SP_BV_rm_50 = (daily_returns['SP500'] * daily_returns['BOVESPA']).rolling(window=50).mean()
SP_HS_rm_50 = (daily_returns['SP500'] * daily_returns['HSI']).rolling(window=50).mean()
SP_NI_rm_50 = (daily_returns['SP500'] * daily_returns['NIFTY50']).rolling(window=50).mean()
SP_ST_rm_50 = (daily_returns['SP500'] * daily_returns['STOXX']).rolling(window=50).mean()
BV_HS_rm_50 = (daily_returns['BOVESPA'] * daily_returns['HSI']).rolling(window=50).mean()
BV_NI_rm_50 = (daily_returns['BOVESPA'] * daily_returns['NIFTY50']).rolling(window=50).mean()
BV_ST_rm_50 = (daily_returns['BOVESPA'] * daily_returns['STOXX']).rolling(window=50).mean()
HS_NI_rm_50 = (daily_returns['HSI'] * daily_returns['NIFTY50']).rolling(window=50).mean()
HS_ST_rm_50 = (daily_returns['HSI'] * daily_returns['STOXX']).rolling(window=50).mean()
NI_ST_rm_50 = (daily_returns['NIFTY50'] * daily_returns['STOXX']).rolling(window=50).mean()


# In[11]:


SP_BV_rm_120 = (daily_returns['SP500'] * daily_returns['BOVESPA']).rolling(window=120).mean()
SP_HS_rm_120 = (daily_returns['SP500'] * daily_returns['HSI']).rolling(window=120).mean()
SP_NI_rm_120 = (daily_returns['SP500'] * daily_returns['NIFTY50']).rolling(window=120).mean()
SP_ST_rm_120 = (daily_returns['SP500'] * daily_returns['STOXX']).rolling(window=120).mean()
BV_HS_rm_120 = (daily_returns['BOVESPA'] * daily_returns['HSI']).rolling(window=120).mean()
BV_NI_rm_120 = (daily_returns['BOVESPA'] * daily_returns['NIFTY50']).rolling(window=120).mean()
BV_ST_rm_120 = (daily_returns['BOVESPA'] * daily_returns['STOXX']).rolling(window=120).mean()
HS_NI_rm_120 = (daily_returns['HSI'] * daily_returns['NIFTY50']).rolling(window=120).mean()
HS_ST_rm_120 = (daily_returns['HSI'] * daily_returns['STOXX']).rolling(window=120).mean()
NI_ST_rm_120 = (daily_returns['NIFTY50'] * daily_returns['STOXX']).rolling(window=120).mean()


# In[12]:


SP_BV_cor_20 = daily_returns['SP500'].rolling(window=20).corr(daily_returns['BOVESPA'])
SP_HS_cor_20 = daily_returns['SP500'].rolling(window=20).corr(daily_returns['HSI'])
SP_NI_cor_20 = daily_returns['SP500'].rolling(window=20).corr(daily_returns['NIFTY50'])
SP_ST_cor_20 = daily_returns['SP500'].rolling(window=20).corr(daily_returns['STOXX'])
BV_HS_cor_20 = daily_returns['BOVESPA'].rolling(window=20).corr(daily_returns['HSI'])
BV_NI_cor_20 = daily_returns['BOVESPA'].rolling(window=20).corr(daily_returns['NIFTY50'])
BV_ST_cor_20 = daily_returns['BOVESPA'].rolling(window=20).corr(daily_returns['STOXX'])
HS_NI_cor_20 = daily_returns['HSI'].rolling(window=20).corr(daily_returns['NIFTY50'])
HS_ST_cor_20 = daily_returns['HSI'].rolling(window=20).corr(daily_returns['STOXX'])
NI_ST_cor_20 = daily_returns['NIFTY50'].rolling(window=20).corr(daily_returns['STOXX'])


# In[13]:


SP_BV_cor_50 = daily_returns['SP500'].rolling(window=50).corr(daily_returns['BOVESPA'])
SP_HS_cor_50 = daily_returns['SP500'].rolling(window=50).corr(daily_returns['HSI'])
SP_NI_cor_50 = daily_returns['SP500'].rolling(window=50).corr(daily_returns['NIFTY50'])
SP_ST_cor_50 = daily_returns['SP500'].rolling(window=50).corr(daily_returns['STOXX'])
BV_HS_cor_50 = daily_returns['BOVESPA'].rolling(window=50).corr(daily_returns['HSI'])
BV_NI_cor_50 = daily_returns['BOVESPA'].rolling(window=50).corr(daily_returns['NIFTY50'])
BV_ST_cor_50 = daily_returns['BOVESPA'].rolling(window=50).corr(daily_returns['STOXX'])
HS_NI_cor_50 = daily_returns['HSI'].rolling(window=50).corr(daily_returns['NIFTY50'])
HS_ST_cor_50 = daily_returns['HSI'].rolling(window=50).corr(daily_returns['STOXX'])
NI_ST_cor_50 = daily_returns['NIFTY50'].rolling(window=50).corr(daily_returns['STOXX'])


# In[14]:


SP_BV_cor_120 = daily_returns['SP500'].rolling(window=120).corr(daily_returns['BOVESPA'])
SP_HS_cor_120 = daily_returns['SP500'].rolling(window=120).corr(daily_returns['HSI'])
SP_NI_cor_120 = daily_returns['SP500'].rolling(window=120).corr(daily_returns['NIFTY50'])
SP_ST_cor_120 = daily_returns['SP500'].rolling(window=120).corr(daily_returns['STOXX'])
BV_HS_cor_120 = daily_returns['BOVESPA'].rolling(window=120).corr(daily_returns['HSI'])
BV_NI_cor_120 = daily_returns['BOVESPA'].rolling(window=120).corr(daily_returns['NIFTY50'])
BV_ST_cor_120 = daily_returns['BOVESPA'].rolling(window=120).corr(daily_returns['STOXX'])
HS_NI_cor_120 = daily_returns['HSI'].rolling(window=120).corr(daily_returns['NIFTY50'])
HS_ST_cor_120 = daily_returns['HSI'].rolling(window=120).corr(daily_returns['STOXX'])
NI_ST_cor_120 = daily_returns['NIFTY50'].rolling(window=120).corr(daily_returns['STOXX'])


# In[15]:


#Target
SP_BV_y = daily_returns['SP500'].rolling(window=120).corr(daily_returns['BOVESPA']).shift(-119)
SP_HS_y = daily_returns['SP500'].rolling(window=120).corr(daily_returns['HSI']).shift(-119)
SP_NI_y = daily_returns['SP500'].rolling(window=120).corr(daily_returns['NIFTY50']).shift(-119)
SP_ST_y = daily_returns['SP500'].rolling(window=120).corr(daily_returns['STOXX']).shift(-119)
BV_HS_y = daily_returns['BOVESPA'].rolling(window=120).corr(daily_returns['HSI']).shift(-119)
BV_NI_y = daily_returns['BOVESPA'].rolling(window=120).corr(daily_returns['NIFTY50']).shift(-119)
BV_ST_y = daily_returns['BOVESPA'].rolling(window=120).corr(daily_returns['STOXX']).shift(-119)
HS_NI_y = daily_returns['HSI'].rolling(window=120).corr(daily_returns['NIFTY50']).shift(-119)
HS_ST_y = daily_returns['HSI'].rolling(window=120).corr(daily_returns['STOXX']).shift(-119)
NI_ST_y = daily_returns['NIFTY50'].rolling(window=120).corr(daily_returns['STOXX']).shift(-119)


# In[16]:


# SP_BV_y


# In[17]:


#Target
SP_y = daily_returns['SP500'].rolling(window=120).std().shift(-119)
ST_y = daily_returns['STOXX'].rolling(window=120).std().shift(-119)
BV_y = daily_returns['BOVESPA'].rolling(window=120).std().shift(-119)
NI_y = daily_returns['NIFTY50'].rolling(window=120).std().shift(-119)
HS_y = daily_returns['HSI'].rolling(window=120).std().shift(-119)


# In[18]:


#Creation of correlation dataframe
columns = ['Date','A_B','A','B',
           'Ret_A_0','Ret_A_1','Ret_A_2','Ret_A_3','Ret_A_4','Ret_A_5','Ret_A_6',
           'Ret_A_7','Ret_A_8','Ret_A_9','Ret_A_10','Ret_A_11','Ret_A_12','Ret_A_13',
           'Ret_A_14','Ret_A_15','Ret_A_16','Ret_A_17','Ret_A_18','Ret_A_19',
           'Ret_B_0','Ret_B_1','Ret_B_2','Ret_B_3','Ret_B_4','Ret_B_5','Ret_B_6',
           'Ret_B_7','Ret_B_8','Ret_B_9','Ret_B_10','Ret_B_11','Ret_B_12','Ret_B_13',
           'Ret_B_14','Ret_B_15','Ret_B_16','Ret_B_17','Ret_B_18','Ret_B_19',
           'Sq_Ret_A_0','Sq_Ret_A_1','Sq_Ret_A_2','Sq_Ret_A_3','Sq_Ret_A_4','Sq_Ret_A_5','Sq_Ret_A_6',
           'Sq_Ret_A_7','Sq_Ret_A_8','Sq_Ret_A_9','Sq_Ret_A_10','Sq_Ret_A_11','Sq_Ret_A_12','Sq_Ret_A_13',
           'Sq_Ret_A_14','Sq_Ret_A_15','Sq_Ret_A_16','Sq_Ret_A_17','Sq_Ret_A_18','Sq_Ret_A_19',
           'Sq_Ret_B_0','Sq_Ret_B_1','Sq_Ret_B_2','Sq_Ret_B_3','Sq_Ret_B_4','Sq_Ret_B_5','Sq_Ret_B_6',
           'Sq_Ret_B_7','Sq_Ret_B_8','Sq_Ret_B_9','Sq_Ret_B_10','Sq_Ret_B_11','Sq_Ret_B_12','Sq_Ret_B_13',
           'Sq_Ret_B_14','Sq_Ret_B_15','Sq_Ret_B_16','Sq_Ret_B_17','Sq_Ret_B_18','Sq_Ret_B_19',
           'Abs_Ret_A_0','Abs_Ret_A_1','Abs_Ret_A_2','Abs_Ret_A_3','Abs_Ret_A_4','Abs_Ret_A_5','Abs_Ret_A_6',
           'Abs_Ret_A_7','Abs_Ret_A_8','Abs_Ret_A_9','Abs_Ret_A_10','Abs_Ret_A_11','Abs_Ret_A_12','Abs_Ret_A_13',
           'Abs_Ret_A_14','Abs_Ret_A_15','Abs_Ret_A_16','Abs_Ret_A_17','Abs_Ret_A_18','Abs_Ret_A_19',
           'Abs_Ret_B_0','Abs_Ret_B_1','Abs_Ret_B_2','Abs_Ret_B_3','Abs_Ret_B_4','Abs_Ret_B_5','Abs_Ret_B_6',
           'Abs_Ret_B_7','Abs_Ret_B_8','Abs_Ret_B_9','Abs_Ret_B_10','Abs_Ret_B_11','Abs_Ret_B_12','Abs_Ret_B_13',
           'Abs_Ret_B_14','Abs_Ret_B_15','Abs_Ret_B_16','Abs_Ret_B_17','Abs_Ret_B_18','Abs_Ret_B_19',
           'Ret_AxRet_B_0','Ret_AxRet_B_1','Ret_AxRet_B_2','Ret_AxRet_B_3','Ret_AxRet_B_4','Ret_AxRet_B_5','Ret_AxRet_B_6',
           'Ret_AxRet_B_7','Ret_AxRet_B_8','Ret_AxRet_B_9','Ret_AxRet_B_10','Ret_AxRet_B_11','Ret_AxRet_B_12','Ret_AxRet_B_13',
           'Ret_AxRet_B_14','Ret_AxRet_B_15','Ret_AxRet_B_16','Ret_AxRet_B_17','Ret_AxRet_B_18','Ret_AxRet_B_19',
           'Abs_Ret_AxRet_B_0','Abs_Ret_AxRet_B_1','Abs_Ret_AxRet_B_2','Abs_Ret_AxRet_B_3','Abs_Ret_AxRet_B_4',
           'Abs_Ret_AxRet_B_5','Abs_Ret_AxRet_B_6','Abs_Ret_AxRet_B_7','Abs_Ret_AxRet_B_8','Abs_Ret_AxRet_B_9',
           'Abs_Ret_AxRet_B_10','Abs_Ret_AxRet_B_11','Abs_Ret_AxRet_B_12','Abs_Ret_AxRet_B_13','Abs_Ret_AxRet_B_14',
           'Abs_Ret_AxRet_B_15','Abs_Ret_AxRet_B_16','Abs_Ret_AxRet_B_17','Abs_Ret_AxRet_B_18','Abs_Ret_AxRet_B_19',
           'A_rm_20','B_rm_20','A_rm_50','B_rm_50','A_rm_120','B_rm_120',
           'A_rm_sq_20','B_rm_sq_20','A_rm_sq_50','B_rm_sq_50','A_rm_sq_120','B_rm_sq_120',
           'A_B_rm_20','A_B_rm_50','A_B_rm_120','A_B_cor_20','A_B_cor_50','A_B_cor_120',
           'Target']
df = pd.DataFrame(columns=columns)


# In[19]:


for i in range(120,daily_returns.shape[0]-20):
# for i in range(120,125):
    #1
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'SP500_STOXX',
        'A' : 'SP500',
        'B' : 'STOXX',
        
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_1' : daily_returns['SP500'].iloc[i-1],
        'Ret_A_2' : daily_returns['SP500'].iloc[i-2],
        'Ret_A_3' : daily_returns['SP500'].iloc[i-3],
        'Ret_A_4' : daily_returns['SP500'].iloc[i-4],
        'Ret_A_5' : daily_returns['SP500'].iloc[i-5],
        'Ret_A_6' : daily_returns['SP500'].iloc[i-6],
        'Ret_A_7' : daily_returns['SP500'].iloc[i-7],
        'Ret_A_8' : daily_returns['SP500'].iloc[i-8],
        'Ret_A_9' : daily_returns['SP500'].iloc[i-9],
        'Ret_A_10' : daily_returns['SP500'].iloc[i-10],
        'Ret_A_11' : daily_returns['SP500'].iloc[i-11],
        'Ret_A_12' : daily_returns['SP500'].iloc[i-12],
        'Ret_A_13' : daily_returns['SP500'].iloc[i-13],
        'Ret_A_14' : daily_returns['SP500'].iloc[i-14],
        'Ret_A_15' : daily_returns['SP500'].iloc[i-15],
        'Ret_A_16' : daily_returns['SP500'].iloc[i-16],
        'Ret_A_17' : daily_returns['SP500'].iloc[i-17],
        'Ret_A_18' : daily_returns['SP500'].iloc[i-18],
        'Ret_A_19' : daily_returns['SP500'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['STOXX'].iloc[i-0],
        'Ret_B_1' : daily_returns['STOXX'].iloc[i-1],
        'Ret_B_2' : daily_returns['STOXX'].iloc[i-2],
        'Ret_B_3' : daily_returns['STOXX'].iloc[i-3],
        'Ret_B_4' : daily_returns['STOXX'].iloc[i-4],
        'Ret_B_5' : daily_returns['STOXX'].iloc[i-5],
        'Ret_B_6' : daily_returns['STOXX'].iloc[i-6],
        'Ret_B_7' : daily_returns['STOXX'].iloc[i-7],
        'Ret_B_8' : daily_returns['STOXX'].iloc[i-8],
        'Ret_B_9' : daily_returns['STOXX'].iloc[i-9],
        'Ret_B_10' : daily_returns['STOXX'].iloc[i-10],
        'Ret_B_11' : daily_returns['STOXX'].iloc[i-11],
        'Ret_B_12' : daily_returns['STOXX'].iloc[i-12],
        'Ret_B_13' : daily_returns['STOXX'].iloc[i-13],
        'Ret_B_14' : daily_returns['STOXX'].iloc[i-14],
        'Ret_B_15' : daily_returns['STOXX'].iloc[i-15],
        'Ret_B_16' : daily_returns['STOXX'].iloc[i-16],
        'Ret_B_17' : daily_returns['STOXX'].iloc[i-17],
        'Ret_B_18' : daily_returns['STOXX'].iloc[i-18],
        'Ret_B_19' : daily_returns['STOXX'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['SP500'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['SP500'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['SP500'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['SP500'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['SP500'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['SP500'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['SP500'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['SP500'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['SP500'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['SP500'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['SP500'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['SP500'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['SP500'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['SP500'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['SP500'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['SP500'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['SP500'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['SP500'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['SP500'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['SP500'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['STOXX'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['STOXX'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['STOXX'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['STOXX'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['STOXX'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['STOXX'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['STOXX'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['STOXX'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['STOXX'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['STOXX'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['STOXX'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['STOXX'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['STOXX'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['STOXX'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['STOXX'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['STOXX'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['STOXX'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['STOXX'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['STOXX'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['STOXX'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['SP500'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['SP500'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['SP500'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['SP500'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['SP500'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['SP500'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['SP500'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['SP500'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['SP500'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['SP500'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['SP500'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['SP500'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['SP500'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['SP500'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['SP500'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['SP500'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['SP500'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['SP500'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['SP500'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['SP500'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['STOXX'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['STOXX'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['STOXX'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['STOXX'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['STOXX'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['STOXX'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['STOXX'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['STOXX'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['STOXX'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['STOXX'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['STOXX'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['STOXX'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['STOXX'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['STOXX'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['STOXX'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['STOXX'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['STOXX'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['STOXX'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['STOXX'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['STOXX'].iloc[i-19],

        'Ret_AxRet_B_0' : SP_ST.iloc[i-0],
        'Ret_AxRet_B_1' : SP_ST.iloc[i-1],
        'Ret_AxRet_B_2' : SP_ST.iloc[i-2],
        'Ret_AxRet_B_3' : SP_ST.iloc[i-3],
        'Ret_AxRet_B_4' : SP_ST.iloc[i-4],
        'Ret_AxRet_B_5' : SP_ST.iloc[i-5],
        'Ret_AxRet_B_6' : SP_ST.iloc[i-6],
        'Ret_AxRet_B_7' : SP_ST.iloc[i-7],
        'Ret_AxRet_B_8' : SP_ST.iloc[i-8],
        'Ret_AxRet_B_9' : SP_ST.iloc[i-9],
        'Ret_AxRet_B_10' : SP_ST.iloc[i-10],
        'Ret_AxRet_B_11' : SP_ST.iloc[i-11],
        'Ret_AxRet_B_12' : SP_ST.iloc[i-12],
        'Ret_AxRet_B_13' : SP_ST.iloc[i-13],
        'Ret_AxRet_B_14' : SP_ST.iloc[i-14],
        'Ret_AxRet_B_15' : SP_ST.iloc[i-15],
        'Ret_AxRet_B_16' : SP_ST.iloc[i-16],
        'Ret_AxRet_B_17' : SP_ST.iloc[i-17],
        'Ret_AxRet_B_18' : SP_ST.iloc[i-18],
        'Ret_AxRet_B_19' : SP_ST.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : SP_ST.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : SP_ST.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : SP_ST.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : SP_ST.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : SP_ST.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : SP_ST.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : SP_ST.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : SP_ST.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : SP_ST.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : SP_ST.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : SP_ST.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : SP_ST.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : SP_ST.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : SP_ST.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : SP_ST.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : SP_ST.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : SP_ST.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : SP_ST.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : SP_ST.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : SP_ST.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['SP500'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['STOXX'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['SP500'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['STOXX'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['SP500'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['STOXX'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['SP500'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['STOXX'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['SP500'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['STOXX'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['SP500'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['STOXX'].iloc[i],
        
        'A_B_rm_20' : SP_ST_rm_20.iloc[i],
        'A_B_rm_50' : SP_ST_rm_50.iloc[i],
        'A_B_rm_120' : SP_ST_rm_120.iloc[i],
        
        'A_B_cor_20' : SP_ST_cor_20.iloc[i],
        'A_B_cor_50' : SP_ST_cor_50.iloc[i],
        'A_B_cor_120' : SP_ST_cor_120.iloc[i],
        
        
        'Target' : SP_ST_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
    
    ######################
     #2
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'SP500_HSI',
        'A' : 'SP500',
        'B' : 'HSI',
        
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_1' : daily_returns['SP500'].iloc[i-1],
        'Ret_A_2' : daily_returns['SP500'].iloc[i-2],
        'Ret_A_3' : daily_returns['SP500'].iloc[i-3],
        'Ret_A_4' : daily_returns['SP500'].iloc[i-4],
        'Ret_A_5' : daily_returns['SP500'].iloc[i-5],
        'Ret_A_6' : daily_returns['SP500'].iloc[i-6],
        'Ret_A_7' : daily_returns['SP500'].iloc[i-7],
        'Ret_A_8' : daily_returns['SP500'].iloc[i-8],
        'Ret_A_9' : daily_returns['SP500'].iloc[i-9],
        'Ret_A_10' : daily_returns['SP500'].iloc[i-10],
        'Ret_A_11' : daily_returns['SP500'].iloc[i-11],
        'Ret_A_12' : daily_returns['SP500'].iloc[i-12],
        'Ret_A_13' : daily_returns['SP500'].iloc[i-13],
        'Ret_A_14' : daily_returns['SP500'].iloc[i-14],
        'Ret_A_15' : daily_returns['SP500'].iloc[i-15],
        'Ret_A_16' : daily_returns['SP500'].iloc[i-16],
        'Ret_A_17' : daily_returns['SP500'].iloc[i-17],
        'Ret_A_18' : daily_returns['SP500'].iloc[i-18],
        'Ret_A_19' : daily_returns['SP500'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['HSI'].iloc[i-0],
        'Ret_B_1' : daily_returns['HSI'].iloc[i-1],
        'Ret_B_2' : daily_returns['HSI'].iloc[i-2],
        'Ret_B_3' : daily_returns['HSI'].iloc[i-3],
        'Ret_B_4' : daily_returns['HSI'].iloc[i-4],
        'Ret_B_5' : daily_returns['HSI'].iloc[i-5],
        'Ret_B_6' : daily_returns['HSI'].iloc[i-6],
        'Ret_B_7' : daily_returns['HSI'].iloc[i-7],
        'Ret_B_8' : daily_returns['HSI'].iloc[i-8],
        'Ret_B_9' : daily_returns['HSI'].iloc[i-9],
        'Ret_B_10' : daily_returns['HSI'].iloc[i-10],
        'Ret_B_11' : daily_returns['HSI'].iloc[i-11],
        'Ret_B_12' : daily_returns['HSI'].iloc[i-12],
        'Ret_B_13' : daily_returns['HSI'].iloc[i-13],
        'Ret_B_14' : daily_returns['HSI'].iloc[i-14],
        'Ret_B_15' : daily_returns['HSI'].iloc[i-15],
        'Ret_B_16' : daily_returns['HSI'].iloc[i-16],
        'Ret_B_17' : daily_returns['HSI'].iloc[i-17],
        'Ret_B_18' : daily_returns['HSI'].iloc[i-18],
        'Ret_B_19' : daily_returns['HSI'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['SP500'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['SP500'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['SP500'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['SP500'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['SP500'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['SP500'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['SP500'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['SP500'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['SP500'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['SP500'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['SP500'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['SP500'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['SP500'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['SP500'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['SP500'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['SP500'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['SP500'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['SP500'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['SP500'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['SP500'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['HSI'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['HSI'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['HSI'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['HSI'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['HSI'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['HSI'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['HSI'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['HSI'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['HSI'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['HSI'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['HSI'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['HSI'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['HSI'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['HSI'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['HSI'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['HSI'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['HSI'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['HSI'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['HSI'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['HSI'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['SP500'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['SP500'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['SP500'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['SP500'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['SP500'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['SP500'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['SP500'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['SP500'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['SP500'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['SP500'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['SP500'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['SP500'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['SP500'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['SP500'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['SP500'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['SP500'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['SP500'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['SP500'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['SP500'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['SP500'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['HSI'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['HSI'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['HSI'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['HSI'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['HSI'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['HSI'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['HSI'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['HSI'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['HSI'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['HSI'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['HSI'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['HSI'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['HSI'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['HSI'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['HSI'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['HSI'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['HSI'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['HSI'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['HSI'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['HSI'].iloc[i-19],

        'Ret_AxRet_B_0' : SP_HS.iloc[i-0],
        'Ret_AxRet_B_1' : SP_HS.iloc[i-1],
        'Ret_AxRet_B_2' : SP_HS.iloc[i-2],
        'Ret_AxRet_B_3' : SP_HS.iloc[i-3],
        'Ret_AxRet_B_4' : SP_HS.iloc[i-4],
        'Ret_AxRet_B_5' : SP_HS.iloc[i-5],
        'Ret_AxRet_B_6' : SP_HS.iloc[i-6],
        'Ret_AxRet_B_7' : SP_HS.iloc[i-7],
        'Ret_AxRet_B_8' : SP_HS.iloc[i-8],
        'Ret_AxRet_B_9' : SP_HS.iloc[i-9],
        'Ret_AxRet_B_10' : SP_HS.iloc[i-10],
        'Ret_AxRet_B_11' : SP_HS.iloc[i-11],
        'Ret_AxRet_B_12' : SP_HS.iloc[i-12],
        'Ret_AxRet_B_13' : SP_HS.iloc[i-13],
        'Ret_AxRet_B_14' : SP_HS.iloc[i-14],
        'Ret_AxRet_B_15' : SP_HS.iloc[i-15],
        'Ret_AxRet_B_16' : SP_HS.iloc[i-16],
        'Ret_AxRet_B_17' : SP_HS.iloc[i-17],
        'Ret_AxRet_B_18' : SP_HS.iloc[i-18],
        'Ret_AxRet_B_19' : SP_HS.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : SP_HS.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : SP_HS.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : SP_HS.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : SP_HS.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : SP_HS.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : SP_HS.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : SP_HS.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : SP_HS.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : SP_HS.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : SP_HS.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : SP_HS.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : SP_HS.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : SP_HS.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : SP_HS.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : SP_HS.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : SP_HS.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : SP_HS.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : SP_HS.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : SP_HS.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : SP_HS.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['SP500'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['HSI'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['SP500'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['HSI'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['SP500'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['HSI'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['SP500'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['HSI'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['SP500'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['HSI'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['SP500'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['HSI'].iloc[i],
        
        'A_B_rm_20' : SP_HS_rm_20.iloc[i],
        'A_B_rm_50' : SP_HS_rm_50.iloc[i],
        'A_B_rm_120' : SP_HS_rm_120.iloc[i],
        
        'A_B_cor_20' : SP_HS_cor_20.iloc[i],
        'A_B_cor_50' : SP_HS_cor_50.iloc[i],
        'A_B_cor_120' : SP_HS_cor_120.iloc[i],
        
        
        'Target' : SP_HS_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
    #3
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'SP500_NIFTY50',
        'A' : 'SP500',
        'B' : 'NIFTY50',
        
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_1' : daily_returns['SP500'].iloc[i-1],
        'Ret_A_2' : daily_returns['SP500'].iloc[i-2],
        'Ret_A_3' : daily_returns['SP500'].iloc[i-3],
        'Ret_A_4' : daily_returns['SP500'].iloc[i-4],
        'Ret_A_5' : daily_returns['SP500'].iloc[i-5],
        'Ret_A_6' : daily_returns['SP500'].iloc[i-6],
        'Ret_A_7' : daily_returns['SP500'].iloc[i-7],
        'Ret_A_8' : daily_returns['SP500'].iloc[i-8],
        'Ret_A_9' : daily_returns['SP500'].iloc[i-9],
        'Ret_A_10' : daily_returns['SP500'].iloc[i-10],
        'Ret_A_11' : daily_returns['SP500'].iloc[i-11],
        'Ret_A_12' : daily_returns['SP500'].iloc[i-12],
        'Ret_A_13' : daily_returns['SP500'].iloc[i-13],
        'Ret_A_14' : daily_returns['SP500'].iloc[i-14],
        'Ret_A_15' : daily_returns['SP500'].iloc[i-15],
        'Ret_A_16' : daily_returns['SP500'].iloc[i-16],
        'Ret_A_17' : daily_returns['SP500'].iloc[i-17],
        'Ret_A_18' : daily_returns['SP500'].iloc[i-18],
        'Ret_A_19' : daily_returns['SP500'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['NIFTY50'].iloc[i-0],
        'Ret_B_1' : daily_returns['NIFTY50'].iloc[i-1],
        'Ret_B_2' : daily_returns['NIFTY50'].iloc[i-2],
        'Ret_B_3' : daily_returns['NIFTY50'].iloc[i-3],
        'Ret_B_4' : daily_returns['NIFTY50'].iloc[i-4],
        'Ret_B_5' : daily_returns['NIFTY50'].iloc[i-5],
        'Ret_B_6' : daily_returns['NIFTY50'].iloc[i-6],
        'Ret_B_7' : daily_returns['NIFTY50'].iloc[i-7],
        'Ret_B_8' : daily_returns['NIFTY50'].iloc[i-8],
        'Ret_B_9' : daily_returns['NIFTY50'].iloc[i-9],
        'Ret_B_10' : daily_returns['NIFTY50'].iloc[i-10],
        'Ret_B_11' : daily_returns['NIFTY50'].iloc[i-11],
        'Ret_B_12' : daily_returns['NIFTY50'].iloc[i-12],
        'Ret_B_13' : daily_returns['NIFTY50'].iloc[i-13],
        'Ret_B_14' : daily_returns['NIFTY50'].iloc[i-14],
        'Ret_B_15' : daily_returns['NIFTY50'].iloc[i-15],
        'Ret_B_16' : daily_returns['NIFTY50'].iloc[i-16],
        'Ret_B_17' : daily_returns['NIFTY50'].iloc[i-17],
        'Ret_B_18' : daily_returns['NIFTY50'].iloc[i-18],
        'Ret_B_19' : daily_returns['NIFTY50'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['SP500'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['SP500'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['SP500'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['SP500'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['SP500'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['SP500'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['SP500'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['SP500'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['SP500'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['SP500'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['SP500'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['SP500'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['SP500'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['SP500'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['SP500'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['SP500'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['SP500'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['SP500'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['SP500'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['SP500'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['NIFTY50'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['NIFTY50'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['NIFTY50'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['NIFTY50'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['NIFTY50'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['NIFTY50'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['NIFTY50'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['NIFTY50'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['NIFTY50'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['NIFTY50'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['NIFTY50'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['NIFTY50'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['NIFTY50'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['NIFTY50'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['NIFTY50'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['NIFTY50'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['NIFTY50'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['NIFTY50'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['NIFTY50'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['NIFTY50'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['SP500'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['SP500'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['SP500'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['SP500'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['SP500'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['SP500'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['SP500'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['SP500'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['SP500'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['SP500'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['SP500'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['SP500'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['SP500'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['SP500'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['SP500'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['SP500'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['SP500'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['SP500'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['SP500'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['SP500'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['NIFTY50'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['NIFTY50'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['NIFTY50'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['NIFTY50'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['NIFTY50'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['NIFTY50'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['NIFTY50'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['NIFTY50'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['NIFTY50'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['NIFTY50'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['NIFTY50'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['NIFTY50'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['NIFTY50'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['NIFTY50'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['NIFTY50'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['NIFTY50'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['NIFTY50'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['NIFTY50'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['NIFTY50'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['NIFTY50'].iloc[i-19],

        'Ret_AxRet_B_0' : SP_NI.iloc[i-0],
        'Ret_AxRet_B_1' : SP_NI.iloc[i-1],
        'Ret_AxRet_B_2' : SP_NI.iloc[i-2],
        'Ret_AxRet_B_3' : SP_NI.iloc[i-3],
        'Ret_AxRet_B_4' : SP_NI.iloc[i-4],
        'Ret_AxRet_B_5' : SP_NI.iloc[i-5],
        'Ret_AxRet_B_6' : SP_NI.iloc[i-6],
        'Ret_AxRet_B_7' : SP_NI.iloc[i-7],
        'Ret_AxRet_B_8' : SP_NI.iloc[i-8],
        'Ret_AxRet_B_9' : SP_NI.iloc[i-9],
        'Ret_AxRet_B_10' : SP_NI.iloc[i-10],
        'Ret_AxRet_B_11' : SP_NI.iloc[i-11],
        'Ret_AxRet_B_12' : SP_NI.iloc[i-12],
        'Ret_AxRet_B_13' : SP_NI.iloc[i-13],
        'Ret_AxRet_B_14' : SP_NI.iloc[i-14],
        'Ret_AxRet_B_15' : SP_NI.iloc[i-15],
        'Ret_AxRet_B_16' : SP_NI.iloc[i-16],
        'Ret_AxRet_B_17' : SP_NI.iloc[i-17],
        'Ret_AxRet_B_18' : SP_NI.iloc[i-18],
        'Ret_AxRet_B_19' : SP_NI.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : SP_NI.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : SP_NI.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : SP_NI.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : SP_NI.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : SP_NI.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : SP_NI.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : SP_NI.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : SP_NI.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : SP_NI.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : SP_NI.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : SP_NI.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : SP_NI.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : SP_NI.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : SP_NI.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : SP_NI.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : SP_NI.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : SP_NI.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : SP_NI.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : SP_NI.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : SP_NI.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['SP500'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['NIFTY50'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['SP500'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['NIFTY50'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['SP500'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['NIFTY50'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['SP500'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['NIFTY50'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['SP500'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['NIFTY50'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['SP500'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['NIFTY50'].iloc[i],
        
        'A_B_rm_20' : SP_NI_rm_20.iloc[i],
        'A_B_rm_50' : SP_NI_rm_50.iloc[i],
        'A_B_rm_120' : SP_NI_rm_120.iloc[i],
        
        'A_B_cor_20' : SP_NI_cor_20.iloc[i],
        'A_B_cor_50' : SP_NI_cor_50.iloc[i],
        'A_B_cor_120' : SP_NI_cor_120.iloc[i],
        
        
        'Target' : SP_NI_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
     #4
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'SP500_BOVESPA',
        'A' : 'SP500',
        'B' : 'BOVESPA',
        
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_1' : daily_returns['SP500'].iloc[i-1],
        'Ret_A_2' : daily_returns['SP500'].iloc[i-2],
        'Ret_A_3' : daily_returns['SP500'].iloc[i-3],
        'Ret_A_4' : daily_returns['SP500'].iloc[i-4],
        'Ret_A_5' : daily_returns['SP500'].iloc[i-5],
        'Ret_A_6' : daily_returns['SP500'].iloc[i-6],
        'Ret_A_7' : daily_returns['SP500'].iloc[i-7],
        'Ret_A_8' : daily_returns['SP500'].iloc[i-8],
        'Ret_A_9' : daily_returns['SP500'].iloc[i-9],
        'Ret_A_10' : daily_returns['SP500'].iloc[i-10],
        'Ret_A_11' : daily_returns['SP500'].iloc[i-11],
        'Ret_A_12' : daily_returns['SP500'].iloc[i-12],
        'Ret_A_13' : daily_returns['SP500'].iloc[i-13],
        'Ret_A_14' : daily_returns['SP500'].iloc[i-14],
        'Ret_A_15' : daily_returns['SP500'].iloc[i-15],
        'Ret_A_16' : daily_returns['SP500'].iloc[i-16],
        'Ret_A_17' : daily_returns['SP500'].iloc[i-17],
        'Ret_A_18' : daily_returns['SP500'].iloc[i-18],
        'Ret_A_19' : daily_returns['SP500'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_B_1' : daily_returns['BOVESPA'].iloc[i-1],
        'Ret_B_2' : daily_returns['BOVESPA'].iloc[i-2],
        'Ret_B_3' : daily_returns['BOVESPA'].iloc[i-3],
        'Ret_B_4' : daily_returns['BOVESPA'].iloc[i-4],
        'Ret_B_5' : daily_returns['BOVESPA'].iloc[i-5],
        'Ret_B_6' : daily_returns['BOVESPA'].iloc[i-6],
        'Ret_B_7' : daily_returns['BOVESPA'].iloc[i-7],
        'Ret_B_8' : daily_returns['BOVESPA'].iloc[i-8],
        'Ret_B_9' : daily_returns['BOVESPA'].iloc[i-9],
        'Ret_B_10' : daily_returns['BOVESPA'].iloc[i-10],
        'Ret_B_11' : daily_returns['BOVESPA'].iloc[i-11],
        'Ret_B_12' : daily_returns['BOVESPA'].iloc[i-12],
        'Ret_B_13' : daily_returns['BOVESPA'].iloc[i-13],
        'Ret_B_14' : daily_returns['BOVESPA'].iloc[i-14],
        'Ret_B_15' : daily_returns['BOVESPA'].iloc[i-15],
        'Ret_B_16' : daily_returns['BOVESPA'].iloc[i-16],
        'Ret_B_17' : daily_returns['BOVESPA'].iloc[i-17],
        'Ret_B_18' : daily_returns['BOVESPA'].iloc[i-18],
        'Ret_B_19' : daily_returns['BOVESPA'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['SP500'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['SP500'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['SP500'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['SP500'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['SP500'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['SP500'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['SP500'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['SP500'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['SP500'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['SP500'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['SP500'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['SP500'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['SP500'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['SP500'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['SP500'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['SP500'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['SP500'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['SP500'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['SP500'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['SP500'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['BOVESPA'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['BOVESPA'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['BOVESPA'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['BOVESPA'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['BOVESPA'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['BOVESPA'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['BOVESPA'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['BOVESPA'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['BOVESPA'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['BOVESPA'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['BOVESPA'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['BOVESPA'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['BOVESPA'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['BOVESPA'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['BOVESPA'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['BOVESPA'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['BOVESPA'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['BOVESPA'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['BOVESPA'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['BOVESPA'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['SP500'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['SP500'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['SP500'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['SP500'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['SP500'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['SP500'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['SP500'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['SP500'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['SP500'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['SP500'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['SP500'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['SP500'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['SP500'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['SP500'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['SP500'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['SP500'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['SP500'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['SP500'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['SP500'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['SP500'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['BOVESPA'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['BOVESPA'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['BOVESPA'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['BOVESPA'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['BOVESPA'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['BOVESPA'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['BOVESPA'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['BOVESPA'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['BOVESPA'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['BOVESPA'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['BOVESPA'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['BOVESPA'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['BOVESPA'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['BOVESPA'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['BOVESPA'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['BOVESPA'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['BOVESPA'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['BOVESPA'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['BOVESPA'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['BOVESPA'].iloc[i-19],

        'Ret_AxRet_B_0' : SP_BV.iloc[i-0],
        'Ret_AxRet_B_1' : SP_BV.iloc[i-1],
        'Ret_AxRet_B_2' : SP_BV.iloc[i-2],
        'Ret_AxRet_B_3' : SP_BV.iloc[i-3],
        'Ret_AxRet_B_4' : SP_BV.iloc[i-4],
        'Ret_AxRet_B_5' : SP_BV.iloc[i-5],
        'Ret_AxRet_B_6' : SP_BV.iloc[i-6],
        'Ret_AxRet_B_7' : SP_BV.iloc[i-7],
        'Ret_AxRet_B_8' : SP_BV.iloc[i-8],
        'Ret_AxRet_B_9' : SP_BV.iloc[i-9],
        'Ret_AxRet_B_10' : SP_BV.iloc[i-10],
        'Ret_AxRet_B_11' : SP_BV.iloc[i-11],
        'Ret_AxRet_B_12' : SP_BV.iloc[i-12],
        'Ret_AxRet_B_13' : SP_BV.iloc[i-13],
        'Ret_AxRet_B_14' : SP_BV.iloc[i-14],
        'Ret_AxRet_B_15' : SP_BV.iloc[i-15],
        'Ret_AxRet_B_16' : SP_BV.iloc[i-16],
        'Ret_AxRet_B_17' : SP_BV.iloc[i-17],
        'Ret_AxRet_B_18' : SP_BV.iloc[i-18],
        'Ret_AxRet_B_19' : SP_BV.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : SP_BV.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : SP_BV.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : SP_BV.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : SP_BV.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : SP_BV.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : SP_BV.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : SP_BV.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : SP_BV.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : SP_BV.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : SP_BV.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : SP_BV.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : SP_BV.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : SP_BV.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : SP_BV.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : SP_BV.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : SP_BV.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : SP_BV.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : SP_BV.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : SP_BV.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : SP_BV.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['SP500'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['BOVESPA'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['SP500'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['BOVESPA'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['SP500'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['BOVESPA'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['SP500'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['BOVESPA'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['SP500'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['BOVESPA'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['SP500'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['BOVESPA'].iloc[i],
        
        'A_B_rm_20' : SP_BV_rm_20.iloc[i],
        'A_B_rm_50' : SP_BV_rm_50.iloc[i],
        'A_B_rm_120' : SP_BV_rm_120.iloc[i],
        
        'A_B_cor_20' : SP_BV_cor_20.iloc[i],
        'A_B_cor_50' : SP_BV_cor_50.iloc[i],
        'A_B_cor_120' : SP_BV_cor_120.iloc[i],
        
        
        'Target' : SP_BV_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
     #5
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'BOVESPA_HSI',
        'A' : 'BOVESPA',
        'B' : 'HSI',
        
        'Ret_A_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_A_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_A_1' : daily_returns['BOVESPA'].iloc[i-1],
        'Ret_A_2' : daily_returns['BOVESPA'].iloc[i-2],
        'Ret_A_3' : daily_returns['BOVESPA'].iloc[i-3],
        'Ret_A_4' : daily_returns['BOVESPA'].iloc[i-4],
        'Ret_A_5' : daily_returns['BOVESPA'].iloc[i-5],
        'Ret_A_6' : daily_returns['BOVESPA'].iloc[i-6],
        'Ret_A_7' : daily_returns['BOVESPA'].iloc[i-7],
        'Ret_A_8' : daily_returns['BOVESPA'].iloc[i-8],
        'Ret_A_9' : daily_returns['BOVESPA'].iloc[i-9],
        'Ret_A_10' : daily_returns['BOVESPA'].iloc[i-10],
        'Ret_A_11' : daily_returns['BOVESPA'].iloc[i-11],
        'Ret_A_12' : daily_returns['BOVESPA'].iloc[i-12],
        'Ret_A_13' : daily_returns['BOVESPA'].iloc[i-13],
        'Ret_A_14' : daily_returns['BOVESPA'].iloc[i-14],
        'Ret_A_15' : daily_returns['BOVESPA'].iloc[i-15],
        'Ret_A_16' : daily_returns['BOVESPA'].iloc[i-16],
        'Ret_A_17' : daily_returns['BOVESPA'].iloc[i-17],
        'Ret_A_18' : daily_returns['BOVESPA'].iloc[i-18],
        'Ret_A_19' : daily_returns['BOVESPA'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['HSI'].iloc[i-0],
        'Ret_B_1' : daily_returns['HSI'].iloc[i-1],
        'Ret_B_2' : daily_returns['HSI'].iloc[i-2],
        'Ret_B_3' : daily_returns['HSI'].iloc[i-3],
        'Ret_B_4' : daily_returns['HSI'].iloc[i-4],
        'Ret_B_5' : daily_returns['HSI'].iloc[i-5],
        'Ret_B_6' : daily_returns['HSI'].iloc[i-6],
        'Ret_B_7' : daily_returns['HSI'].iloc[i-7],
        'Ret_B_8' : daily_returns['HSI'].iloc[i-8],
        'Ret_B_9' : daily_returns['HSI'].iloc[i-9],
        'Ret_B_10' : daily_returns['HSI'].iloc[i-10],
        'Ret_B_11' : daily_returns['HSI'].iloc[i-11],
        'Ret_B_12' : daily_returns['HSI'].iloc[i-12],
        'Ret_B_13' : daily_returns['HSI'].iloc[i-13],
        'Ret_B_14' : daily_returns['HSI'].iloc[i-14],
        'Ret_B_15' : daily_returns['HSI'].iloc[i-15],
        'Ret_B_16' : daily_returns['HSI'].iloc[i-16],
        'Ret_B_17' : daily_returns['HSI'].iloc[i-17],
        'Ret_B_18' : daily_returns['HSI'].iloc[i-18],
        'Ret_B_19' : daily_returns['HSI'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['BOVESPA'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['BOVESPA'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['BOVESPA'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['BOVESPA'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['BOVESPA'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['BOVESPA'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['BOVESPA'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['BOVESPA'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['BOVESPA'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['BOVESPA'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['BOVESPA'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['BOVESPA'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['BOVESPA'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['BOVESPA'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['BOVESPA'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['BOVESPA'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['BOVESPA'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['BOVESPA'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['BOVESPA'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['BOVESPA'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['HSI'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['HSI'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['HSI'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['HSI'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['HSI'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['HSI'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['HSI'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['HSI'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['HSI'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['HSI'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['HSI'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['HSI'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['HSI'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['HSI'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['HSI'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['HSI'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['HSI'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['HSI'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['HSI'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['HSI'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['BOVESPA'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['BOVESPA'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['BOVESPA'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['BOVESPA'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['BOVESPA'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['BOVESPA'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['BOVESPA'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['BOVESPA'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['BOVESPA'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['BOVESPA'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['BOVESPA'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['BOVESPA'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['BOVESPA'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['BOVESPA'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['BOVESPA'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['BOVESPA'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['BOVESPA'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['BOVESPA'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['BOVESPA'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['BOVESPA'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['HSI'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['HSI'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['HSI'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['HSI'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['HSI'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['HSI'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['HSI'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['HSI'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['HSI'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['HSI'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['HSI'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['HSI'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['HSI'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['HSI'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['HSI'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['HSI'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['HSI'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['HSI'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['HSI'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['HSI'].iloc[i-19],

        'Ret_AxRet_B_0' : BV_HS.iloc[i-0],
        'Ret_AxRet_B_1' : BV_HS.iloc[i-1],
        'Ret_AxRet_B_2' : BV_HS.iloc[i-2],
        'Ret_AxRet_B_3' : BV_HS.iloc[i-3],
        'Ret_AxRet_B_4' : BV_HS.iloc[i-4],
        'Ret_AxRet_B_5' : BV_HS.iloc[i-5],
        'Ret_AxRet_B_6' : BV_HS.iloc[i-6],
        'Ret_AxRet_B_7' : BV_HS.iloc[i-7],
        'Ret_AxRet_B_8' : BV_HS.iloc[i-8],
        'Ret_AxRet_B_9' : BV_HS.iloc[i-9],
        'Ret_AxRet_B_10' : BV_HS.iloc[i-10],
        'Ret_AxRet_B_11' : BV_HS.iloc[i-11],
        'Ret_AxRet_B_12' : BV_HS.iloc[i-12],
        'Ret_AxRet_B_13' : BV_HS.iloc[i-13],
        'Ret_AxRet_B_14' : BV_HS.iloc[i-14],
        'Ret_AxRet_B_15' : BV_HS.iloc[i-15],
        'Ret_AxRet_B_16' : BV_HS.iloc[i-16],
        'Ret_AxRet_B_17' : BV_HS.iloc[i-17],
        'Ret_AxRet_B_18' : BV_HS.iloc[i-18],
        'Ret_AxRet_B_19' : BV_HS.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : BV_HS.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : BV_HS.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : BV_HS.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : BV_HS.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : BV_HS.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : BV_HS.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : BV_HS.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : BV_HS.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : BV_HS.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : BV_HS.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : BV_HS.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : BV_HS.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : BV_HS.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : BV_HS.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : BV_HS.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : BV_HS.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : BV_HS.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : BV_HS.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : BV_HS.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : BV_HS.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['BOVESPA'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['HSI'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['BOVESPA'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['HSI'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['BOVESPA'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['HSI'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['BOVESPA'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['HSI'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['BOVESPA'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['HSI'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['BOVESPA'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['HSI'].iloc[i],
        
        'A_B_rm_20' : BV_HS_rm_20.iloc[i],
        'A_B_rm_50' : BV_HS_rm_50.iloc[i],
        'A_B_rm_120' : BV_HS_rm_120.iloc[i],
        
        'A_B_cor_20' : BV_HS_cor_20.iloc[i],
        'A_B_cor_50' : BV_HS_cor_50.iloc[i],
        'A_B_cor_120' : BV_HS_cor_120.iloc[i],
        
        
        'Target' : BV_HS_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
    #6
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'BOVESPA_NIFTY50',
        'A' : 'BOVESPA',
        'B' : 'NIFTY50',
        
        'Ret_A_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_A_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_A_1' : daily_returns['BOVESPA'].iloc[i-1],
        'Ret_A_2' : daily_returns['BOVESPA'].iloc[i-2],
        'Ret_A_3' : daily_returns['BOVESPA'].iloc[i-3],
        'Ret_A_4' : daily_returns['BOVESPA'].iloc[i-4],
        'Ret_A_5' : daily_returns['BOVESPA'].iloc[i-5],
        'Ret_A_6' : daily_returns['BOVESPA'].iloc[i-6],
        'Ret_A_7' : daily_returns['BOVESPA'].iloc[i-7],
        'Ret_A_8' : daily_returns['BOVESPA'].iloc[i-8],
        'Ret_A_9' : daily_returns['BOVESPA'].iloc[i-9],
        'Ret_A_10' : daily_returns['BOVESPA'].iloc[i-10],
        'Ret_A_11' : daily_returns['BOVESPA'].iloc[i-11],
        'Ret_A_12' : daily_returns['BOVESPA'].iloc[i-12],
        'Ret_A_13' : daily_returns['BOVESPA'].iloc[i-13],
        'Ret_A_14' : daily_returns['BOVESPA'].iloc[i-14],
        'Ret_A_15' : daily_returns['BOVESPA'].iloc[i-15],
        'Ret_A_16' : daily_returns['BOVESPA'].iloc[i-16],
        'Ret_A_17' : daily_returns['BOVESPA'].iloc[i-17],
        'Ret_A_18' : daily_returns['BOVESPA'].iloc[i-18],
        'Ret_A_19' : daily_returns['BOVESPA'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['NIFTY50'].iloc[i-0],
        'Ret_B_1' : daily_returns['NIFTY50'].iloc[i-1],
        'Ret_B_2' : daily_returns['NIFTY50'].iloc[i-2],
        'Ret_B_3' : daily_returns['NIFTY50'].iloc[i-3],
        'Ret_B_4' : daily_returns['NIFTY50'].iloc[i-4],
        'Ret_B_5' : daily_returns['NIFTY50'].iloc[i-5],
        'Ret_B_6' : daily_returns['NIFTY50'].iloc[i-6],
        'Ret_B_7' : daily_returns['NIFTY50'].iloc[i-7],
        'Ret_B_8' : daily_returns['NIFTY50'].iloc[i-8],
        'Ret_B_9' : daily_returns['NIFTY50'].iloc[i-9],
        'Ret_B_10' : daily_returns['NIFTY50'].iloc[i-10],
        'Ret_B_11' : daily_returns['NIFTY50'].iloc[i-11],
        'Ret_B_12' : daily_returns['NIFTY50'].iloc[i-12],
        'Ret_B_13' : daily_returns['NIFTY50'].iloc[i-13],
        'Ret_B_14' : daily_returns['NIFTY50'].iloc[i-14],
        'Ret_B_15' : daily_returns['NIFTY50'].iloc[i-15],
        'Ret_B_16' : daily_returns['NIFTY50'].iloc[i-16],
        'Ret_B_17' : daily_returns['NIFTY50'].iloc[i-17],
        'Ret_B_18' : daily_returns['NIFTY50'].iloc[i-18],
        'Ret_B_19' : daily_returns['NIFTY50'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['BOVESPA'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['BOVESPA'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['BOVESPA'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['BOVESPA'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['BOVESPA'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['BOVESPA'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['BOVESPA'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['BOVESPA'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['BOVESPA'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['BOVESPA'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['BOVESPA'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['BOVESPA'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['BOVESPA'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['BOVESPA'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['BOVESPA'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['BOVESPA'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['BOVESPA'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['BOVESPA'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['BOVESPA'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['BOVESPA'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['NIFTY50'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['NIFTY50'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['NIFTY50'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['NIFTY50'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['NIFTY50'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['NIFTY50'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['NIFTY50'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['NIFTY50'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['NIFTY50'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['NIFTY50'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['NIFTY50'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['NIFTY50'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['NIFTY50'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['NIFTY50'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['NIFTY50'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['NIFTY50'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['NIFTY50'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['NIFTY50'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['NIFTY50'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['NIFTY50'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['BOVESPA'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['BOVESPA'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['BOVESPA'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['BOVESPA'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['BOVESPA'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['BOVESPA'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['BOVESPA'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['BOVESPA'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['BOVESPA'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['BOVESPA'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['BOVESPA'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['BOVESPA'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['BOVESPA'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['BOVESPA'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['BOVESPA'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['BOVESPA'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['BOVESPA'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['BOVESPA'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['BOVESPA'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['BOVESPA'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['NIFTY50'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['NIFTY50'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['NIFTY50'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['NIFTY50'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['NIFTY50'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['NIFTY50'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['NIFTY50'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['NIFTY50'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['NIFTY50'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['NIFTY50'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['NIFTY50'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['NIFTY50'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['NIFTY50'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['NIFTY50'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['NIFTY50'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['NIFTY50'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['NIFTY50'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['NIFTY50'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['NIFTY50'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['NIFTY50'].iloc[i-19],

        'Ret_AxRet_B_0' : BV_NI.iloc[i-0],
        'Ret_AxRet_B_1' : BV_NI.iloc[i-1],
        'Ret_AxRet_B_2' : BV_NI.iloc[i-2],
        'Ret_AxRet_B_3' : BV_NI.iloc[i-3],
        'Ret_AxRet_B_4' : BV_NI.iloc[i-4],
        'Ret_AxRet_B_5' : BV_NI.iloc[i-5],
        'Ret_AxRet_B_6' : BV_NI.iloc[i-6],
        'Ret_AxRet_B_7' : BV_NI.iloc[i-7],
        'Ret_AxRet_B_8' : BV_NI.iloc[i-8],
        'Ret_AxRet_B_9' : BV_NI.iloc[i-9],
        'Ret_AxRet_B_10' : BV_NI.iloc[i-10],
        'Ret_AxRet_B_11' : BV_NI.iloc[i-11],
        'Ret_AxRet_B_12' : BV_NI.iloc[i-12],
        'Ret_AxRet_B_13' : BV_NI.iloc[i-13],
        'Ret_AxRet_B_14' : BV_NI.iloc[i-14],
        'Ret_AxRet_B_15' : BV_NI.iloc[i-15],
        'Ret_AxRet_B_16' : BV_NI.iloc[i-16],
        'Ret_AxRet_B_17' : BV_NI.iloc[i-17],
        'Ret_AxRet_B_18' : BV_NI.iloc[i-18],
        'Ret_AxRet_B_19' : BV_NI.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : BV_NI.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : BV_NI.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : BV_NI.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : BV_NI.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : BV_NI.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : BV_NI.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : BV_NI.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : BV_NI.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : BV_NI.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : BV_NI.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : BV_NI.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : BV_NI.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : BV_NI.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : BV_NI.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : BV_NI.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : BV_NI.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : BV_NI.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : BV_NI.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : BV_NI.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : BV_NI.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['BOVESPA'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['NIFTY50'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['BOVESPA'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['NIFTY50'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['BOVESPA'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['NIFTY50'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['BOVESPA'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['NIFTY50'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['BOVESPA'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['NIFTY50'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['BOVESPA'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['NIFTY50'].iloc[i],
        
        'A_B_rm_20' : BV_NI_rm_20.iloc[i],
        'A_B_rm_50' : BV_NI_rm_50.iloc[i],
        'A_B_rm_120' : BV_NI_rm_120.iloc[i],
        
        'A_B_cor_20' : BV_NI_cor_20.iloc[i],
        'A_B_cor_50' : BV_NI_cor_50.iloc[i],
        'A_B_cor_120' : BV_NI_cor_120.iloc[i],
        
        
        'Target' : BV_NI_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
     #7
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'BOVESPA_STOXX',
        'A' : 'BOVESPA',
        'B' : 'STOXX',
        
        'Ret_A_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_A_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_A_1' : daily_returns['BOVESPA'].iloc[i-1],
        'Ret_A_2' : daily_returns['BOVESPA'].iloc[i-2],
        'Ret_A_3' : daily_returns['BOVESPA'].iloc[i-3],
        'Ret_A_4' : daily_returns['BOVESPA'].iloc[i-4],
        'Ret_A_5' : daily_returns['BOVESPA'].iloc[i-5],
        'Ret_A_6' : daily_returns['BOVESPA'].iloc[i-6],
        'Ret_A_7' : daily_returns['BOVESPA'].iloc[i-7],
        'Ret_A_8' : daily_returns['BOVESPA'].iloc[i-8],
        'Ret_A_9' : daily_returns['BOVESPA'].iloc[i-9],
        'Ret_A_10' : daily_returns['BOVESPA'].iloc[i-10],
        'Ret_A_11' : daily_returns['BOVESPA'].iloc[i-11],
        'Ret_A_12' : daily_returns['BOVESPA'].iloc[i-12],
        'Ret_A_13' : daily_returns['BOVESPA'].iloc[i-13],
        'Ret_A_14' : daily_returns['BOVESPA'].iloc[i-14],
        'Ret_A_15' : daily_returns['BOVESPA'].iloc[i-15],
        'Ret_A_16' : daily_returns['BOVESPA'].iloc[i-16],
        'Ret_A_17' : daily_returns['BOVESPA'].iloc[i-17],
        'Ret_A_18' : daily_returns['BOVESPA'].iloc[i-18],
        'Ret_A_19' : daily_returns['BOVESPA'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['STOXX'].iloc[i-0],
        'Ret_B_1' : daily_returns['STOXX'].iloc[i-1],
        'Ret_B_2' : daily_returns['STOXX'].iloc[i-2],
        'Ret_B_3' : daily_returns['STOXX'].iloc[i-3],
        'Ret_B_4' : daily_returns['STOXX'].iloc[i-4],
        'Ret_B_5' : daily_returns['STOXX'].iloc[i-5],
        'Ret_B_6' : daily_returns['STOXX'].iloc[i-6],
        'Ret_B_7' : daily_returns['STOXX'].iloc[i-7],
        'Ret_B_8' : daily_returns['STOXX'].iloc[i-8],
        'Ret_B_9' : daily_returns['STOXX'].iloc[i-9],
        'Ret_B_10' : daily_returns['STOXX'].iloc[i-10],
        'Ret_B_11' : daily_returns['STOXX'].iloc[i-11],
        'Ret_B_12' : daily_returns['STOXX'].iloc[i-12],
        'Ret_B_13' : daily_returns['STOXX'].iloc[i-13],
        'Ret_B_14' : daily_returns['STOXX'].iloc[i-14],
        'Ret_B_15' : daily_returns['STOXX'].iloc[i-15],
        'Ret_B_16' : daily_returns['STOXX'].iloc[i-16],
        'Ret_B_17' : daily_returns['STOXX'].iloc[i-17],
        'Ret_B_18' : daily_returns['STOXX'].iloc[i-18],
        'Ret_B_19' : daily_returns['STOXX'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['BOVESPA'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['BOVESPA'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['BOVESPA'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['BOVESPA'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['BOVESPA'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['BOVESPA'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['BOVESPA'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['BOVESPA'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['BOVESPA'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['BOVESPA'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['BOVESPA'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['BOVESPA'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['BOVESPA'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['BOVESPA'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['BOVESPA'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['BOVESPA'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['BOVESPA'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['BOVESPA'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['BOVESPA'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['BOVESPA'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['STOXX'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['STOXX'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['STOXX'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['STOXX'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['STOXX'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['STOXX'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['STOXX'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['STOXX'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['STOXX'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['STOXX'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['STOXX'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['STOXX'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['STOXX'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['STOXX'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['STOXX'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['STOXX'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['STOXX'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['STOXX'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['STOXX'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['STOXX'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['BOVESPA'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['BOVESPA'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['BOVESPA'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['BOVESPA'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['BOVESPA'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['BOVESPA'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['BOVESPA'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['BOVESPA'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['BOVESPA'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['BOVESPA'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['BOVESPA'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['BOVESPA'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['BOVESPA'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['BOVESPA'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['BOVESPA'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['BOVESPA'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['BOVESPA'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['BOVESPA'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['BOVESPA'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['BOVESPA'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['STOXX'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['STOXX'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['STOXX'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['STOXX'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['STOXX'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['STOXX'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['STOXX'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['STOXX'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['STOXX'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['STOXX'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['STOXX'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['STOXX'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['STOXX'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['STOXX'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['STOXX'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['STOXX'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['STOXX'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['STOXX'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['STOXX'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['STOXX'].iloc[i-19],

        'Ret_AxRet_B_0' : BV_ST.iloc[i-0],
        'Ret_AxRet_B_1' : BV_ST.iloc[i-1],
        'Ret_AxRet_B_2' : BV_ST.iloc[i-2],
        'Ret_AxRet_B_3' : BV_ST.iloc[i-3],
        'Ret_AxRet_B_4' : BV_ST.iloc[i-4],
        'Ret_AxRet_B_5' : BV_ST.iloc[i-5],
        'Ret_AxRet_B_6' : BV_ST.iloc[i-6],
        'Ret_AxRet_B_7' : BV_ST.iloc[i-7],
        'Ret_AxRet_B_8' : BV_ST.iloc[i-8],
        'Ret_AxRet_B_9' : BV_ST.iloc[i-9],
        'Ret_AxRet_B_10' : BV_ST.iloc[i-10],
        'Ret_AxRet_B_11' : BV_ST.iloc[i-11],
        'Ret_AxRet_B_12' : BV_ST.iloc[i-12],
        'Ret_AxRet_B_13' : BV_ST.iloc[i-13],
        'Ret_AxRet_B_14' : BV_ST.iloc[i-14],
        'Ret_AxRet_B_15' : BV_ST.iloc[i-15],
        'Ret_AxRet_B_16' : BV_ST.iloc[i-16],
        'Ret_AxRet_B_17' : BV_ST.iloc[i-17],
        'Ret_AxRet_B_18' : BV_ST.iloc[i-18],
        'Ret_AxRet_B_19' : BV_ST.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : BV_ST.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : BV_ST.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : BV_ST.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : BV_ST.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : BV_ST.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : BV_ST.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : BV_ST.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : BV_ST.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : BV_ST.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : BV_ST.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : BV_ST.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : BV_ST.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : BV_ST.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : BV_ST.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : BV_ST.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : BV_ST.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : BV_ST.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : BV_ST.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : BV_ST.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : BV_ST.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['BOVESPA'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['STOXX'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['BOVESPA'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['STOXX'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['BOVESPA'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['STOXX'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['BOVESPA'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['STOXX'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['BOVESPA'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['STOXX'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['BOVESPA'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['STOXX'].iloc[i],
        
        'A_B_rm_20' : BV_ST_rm_20.iloc[i],
        'A_B_rm_50' : BV_ST_rm_50.iloc[i],
        'A_B_rm_120' : BV_ST_rm_120.iloc[i],
        
        'A_B_cor_20' : BV_ST_cor_20.iloc[i],
        'A_B_cor_50' : BV_ST_cor_50.iloc[i],
        'A_B_cor_120' : BV_ST_cor_120.iloc[i],
        
        
        'Target' : BV_ST_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
     #8
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'HSI_NIFTY50',
        'A' : 'HSI',
        'B' : 'NIFTY50',
        
        'Ret_A_0' : daily_returns['HSI'].iloc[i-0],
        'Ret_A_0' : daily_returns['HSI'].iloc[i-0],
        'Ret_A_1' : daily_returns['HSI'].iloc[i-1],
        'Ret_A_2' : daily_returns['HSI'].iloc[i-2],
        'Ret_A_3' : daily_returns['HSI'].iloc[i-3],
        'Ret_A_4' : daily_returns['HSI'].iloc[i-4],
        'Ret_A_5' : daily_returns['HSI'].iloc[i-5],
        'Ret_A_6' : daily_returns['HSI'].iloc[i-6],
        'Ret_A_7' : daily_returns['HSI'].iloc[i-7],
        'Ret_A_8' : daily_returns['HSI'].iloc[i-8],
        'Ret_A_9' : daily_returns['HSI'].iloc[i-9],
        'Ret_A_10' : daily_returns['HSI'].iloc[i-10],
        'Ret_A_11' : daily_returns['HSI'].iloc[i-11],
        'Ret_A_12' : daily_returns['HSI'].iloc[i-12],
        'Ret_A_13' : daily_returns['HSI'].iloc[i-13],
        'Ret_A_14' : daily_returns['HSI'].iloc[i-14],
        'Ret_A_15' : daily_returns['HSI'].iloc[i-15],
        'Ret_A_16' : daily_returns['HSI'].iloc[i-16],
        'Ret_A_17' : daily_returns['HSI'].iloc[i-17],
        'Ret_A_18' : daily_returns['HSI'].iloc[i-18],
        'Ret_A_19' : daily_returns['HSI'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['NIFTY50'].iloc[i-0],
        'Ret_B_1' : daily_returns['NIFTY50'].iloc[i-1],
        'Ret_B_2' : daily_returns['NIFTY50'].iloc[i-2],
        'Ret_B_3' : daily_returns['NIFTY50'].iloc[i-3],
        'Ret_B_4' : daily_returns['NIFTY50'].iloc[i-4],
        'Ret_B_5' : daily_returns['NIFTY50'].iloc[i-5],
        'Ret_B_6' : daily_returns['NIFTY50'].iloc[i-6],
        'Ret_B_7' : daily_returns['NIFTY50'].iloc[i-7],
        'Ret_B_8' : daily_returns['NIFTY50'].iloc[i-8],
        'Ret_B_9' : daily_returns['NIFTY50'].iloc[i-9],
        'Ret_B_10' : daily_returns['NIFTY50'].iloc[i-10],
        'Ret_B_11' : daily_returns['NIFTY50'].iloc[i-11],
        'Ret_B_12' : daily_returns['NIFTY50'].iloc[i-12],
        'Ret_B_13' : daily_returns['NIFTY50'].iloc[i-13],
        'Ret_B_14' : daily_returns['NIFTY50'].iloc[i-14],
        'Ret_B_15' : daily_returns['NIFTY50'].iloc[i-15],
        'Ret_B_16' : daily_returns['NIFTY50'].iloc[i-16],
        'Ret_B_17' : daily_returns['NIFTY50'].iloc[i-17],
        'Ret_B_18' : daily_returns['NIFTY50'].iloc[i-18],
        'Ret_B_19' : daily_returns['NIFTY50'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['HSI'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['HSI'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['HSI'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['HSI'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['HSI'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['HSI'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['HSI'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['HSI'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['HSI'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['HSI'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['HSI'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['HSI'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['HSI'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['HSI'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['HSI'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['HSI'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['HSI'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['HSI'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['HSI'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['HSI'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['NIFTY50'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['NIFTY50'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['NIFTY50'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['NIFTY50'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['NIFTY50'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['NIFTY50'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['NIFTY50'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['NIFTY50'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['NIFTY50'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['NIFTY50'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['NIFTY50'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['NIFTY50'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['NIFTY50'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['NIFTY50'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['NIFTY50'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['NIFTY50'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['NIFTY50'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['NIFTY50'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['NIFTY50'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['NIFTY50'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['HSI'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['HSI'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['HSI'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['HSI'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['HSI'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['HSI'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['HSI'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['HSI'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['HSI'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['HSI'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['HSI'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['HSI'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['HSI'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['HSI'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['HSI'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['HSI'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['HSI'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['HSI'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['HSI'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['HSI'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['NIFTY50'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['NIFTY50'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['NIFTY50'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['NIFTY50'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['NIFTY50'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['NIFTY50'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['NIFTY50'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['NIFTY50'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['NIFTY50'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['NIFTY50'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['NIFTY50'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['NIFTY50'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['NIFTY50'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['NIFTY50'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['NIFTY50'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['NIFTY50'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['NIFTY50'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['NIFTY50'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['NIFTY50'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['NIFTY50'].iloc[i-19],

        'Ret_AxRet_B_0' : HS_NI.iloc[i-0],
        'Ret_AxRet_B_1' : HS_NI.iloc[i-1],
        'Ret_AxRet_B_2' : HS_NI.iloc[i-2],
        'Ret_AxRet_B_3' : HS_NI.iloc[i-3],
        'Ret_AxRet_B_4' : HS_NI.iloc[i-4],
        'Ret_AxRet_B_5' : HS_NI.iloc[i-5],
        'Ret_AxRet_B_6' : HS_NI.iloc[i-6],
        'Ret_AxRet_B_7' : HS_NI.iloc[i-7],
        'Ret_AxRet_B_8' : HS_NI.iloc[i-8],
        'Ret_AxRet_B_9' : HS_NI.iloc[i-9],
        'Ret_AxRet_B_10' : HS_NI.iloc[i-10],
        'Ret_AxRet_B_11' : HS_NI.iloc[i-11],
        'Ret_AxRet_B_12' : HS_NI.iloc[i-12],
        'Ret_AxRet_B_13' : HS_NI.iloc[i-13],
        'Ret_AxRet_B_14' : HS_NI.iloc[i-14],
        'Ret_AxRet_B_15' : HS_NI.iloc[i-15],
        'Ret_AxRet_B_16' : HS_NI.iloc[i-16],
        'Ret_AxRet_B_17' : HS_NI.iloc[i-17],
        'Ret_AxRet_B_18' : HS_NI.iloc[i-18],
        'Ret_AxRet_B_19' : HS_NI.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : HS_NI.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : HS_NI.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : HS_NI.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : HS_NI.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : HS_NI.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : HS_NI.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : HS_NI.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : HS_NI.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : HS_NI.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : HS_NI.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : HS_NI.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : HS_NI.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : HS_NI.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : HS_NI.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : HS_NI.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : HS_NI.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : HS_NI.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : HS_NI.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : HS_NI.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : HS_NI.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['HSI'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['NIFTY50'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['HSI'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['NIFTY50'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['HSI'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['NIFTY50'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['HSI'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['NIFTY50'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['HSI'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['NIFTY50'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['HSI'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['NIFTY50'].iloc[i],
        
        'A_B_rm_20' : HS_NI_rm_20.iloc[i],
        'A_B_rm_50' : HS_NI_rm_50.iloc[i],
        'A_B_rm_120' : HS_NI_rm_120.iloc[i],
        
        'A_B_cor_20' : HS_NI_cor_20.iloc[i],
        'A_B_cor_50' : HS_NI_cor_50.iloc[i],
        'A_B_cor_120' : HS_NI_cor_120.iloc[i],
        
        
        'Target' : HS_NI_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
     #9
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'HSI_STOXX',
        'A' : 'HSI',
        'B' : 'STOXX',
        
        'Ret_A_0' : daily_returns['HSI'].iloc[i-0],
        'Ret_A_0' : daily_returns['HSI'].iloc[i-0],
        'Ret_A_1' : daily_returns['HSI'].iloc[i-1],
        'Ret_A_2' : daily_returns['HSI'].iloc[i-2],
        'Ret_A_3' : daily_returns['HSI'].iloc[i-3],
        'Ret_A_4' : daily_returns['HSI'].iloc[i-4],
        'Ret_A_5' : daily_returns['HSI'].iloc[i-5],
        'Ret_A_6' : daily_returns['HSI'].iloc[i-6],
        'Ret_A_7' : daily_returns['HSI'].iloc[i-7],
        'Ret_A_8' : daily_returns['HSI'].iloc[i-8],
        'Ret_A_9' : daily_returns['HSI'].iloc[i-9],
        'Ret_A_10' : daily_returns['HSI'].iloc[i-10],
        'Ret_A_11' : daily_returns['HSI'].iloc[i-11],
        'Ret_A_12' : daily_returns['HSI'].iloc[i-12],
        'Ret_A_13' : daily_returns['HSI'].iloc[i-13],
        'Ret_A_14' : daily_returns['HSI'].iloc[i-14],
        'Ret_A_15' : daily_returns['HSI'].iloc[i-15],
        'Ret_A_16' : daily_returns['HSI'].iloc[i-16],
        'Ret_A_17' : daily_returns['HSI'].iloc[i-17],
        'Ret_A_18' : daily_returns['HSI'].iloc[i-18],
        'Ret_A_19' : daily_returns['HSI'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['STOXX'].iloc[i-0],
        'Ret_B_1' : daily_returns['STOXX'].iloc[i-1],
        'Ret_B_2' : daily_returns['STOXX'].iloc[i-2],
        'Ret_B_3' : daily_returns['STOXX'].iloc[i-3],
        'Ret_B_4' : daily_returns['STOXX'].iloc[i-4],
        'Ret_B_5' : daily_returns['STOXX'].iloc[i-5],
        'Ret_B_6' : daily_returns['STOXX'].iloc[i-6],
        'Ret_B_7' : daily_returns['STOXX'].iloc[i-7],
        'Ret_B_8' : daily_returns['STOXX'].iloc[i-8],
        'Ret_B_9' : daily_returns['STOXX'].iloc[i-9],
        'Ret_B_10' : daily_returns['STOXX'].iloc[i-10],
        'Ret_B_11' : daily_returns['STOXX'].iloc[i-11],
        'Ret_B_12' : daily_returns['STOXX'].iloc[i-12],
        'Ret_B_13' : daily_returns['STOXX'].iloc[i-13],
        'Ret_B_14' : daily_returns['STOXX'].iloc[i-14],
        'Ret_B_15' : daily_returns['STOXX'].iloc[i-15],
        'Ret_B_16' : daily_returns['STOXX'].iloc[i-16],
        'Ret_B_17' : daily_returns['STOXX'].iloc[i-17],
        'Ret_B_18' : daily_returns['STOXX'].iloc[i-18],
        'Ret_B_19' : daily_returns['STOXX'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['HSI'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['HSI'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['HSI'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['HSI'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['HSI'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['HSI'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['HSI'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['HSI'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['HSI'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['HSI'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['HSI'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['HSI'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['HSI'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['HSI'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['HSI'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['HSI'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['HSI'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['HSI'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['HSI'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['HSI'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['STOXX'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['STOXX'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['STOXX'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['STOXX'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['STOXX'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['STOXX'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['STOXX'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['STOXX'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['STOXX'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['STOXX'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['STOXX'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['STOXX'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['STOXX'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['STOXX'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['STOXX'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['STOXX'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['STOXX'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['STOXX'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['STOXX'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['STOXX'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['HSI'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['HSI'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['HSI'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['HSI'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['HSI'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['HSI'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['HSI'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['HSI'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['HSI'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['HSI'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['HSI'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['HSI'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['HSI'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['HSI'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['HSI'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['HSI'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['HSI'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['HSI'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['HSI'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['HSI'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['STOXX'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['STOXX'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['STOXX'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['STOXX'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['STOXX'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['STOXX'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['STOXX'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['STOXX'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['STOXX'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['STOXX'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['STOXX'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['STOXX'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['STOXX'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['STOXX'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['STOXX'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['STOXX'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['STOXX'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['STOXX'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['STOXX'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['STOXX'].iloc[i-19],

        'Ret_AxRet_B_0' : HS_ST.iloc[i-0],
        'Ret_AxRet_B_1' : HS_ST.iloc[i-1],
        'Ret_AxRet_B_2' : HS_ST.iloc[i-2],
        'Ret_AxRet_B_3' : HS_ST.iloc[i-3],
        'Ret_AxRet_B_4' : HS_ST.iloc[i-4],
        'Ret_AxRet_B_5' : HS_ST.iloc[i-5],
        'Ret_AxRet_B_6' : HS_ST.iloc[i-6],
        'Ret_AxRet_B_7' : HS_ST.iloc[i-7],
        'Ret_AxRet_B_8' : HS_ST.iloc[i-8],
        'Ret_AxRet_B_9' : HS_ST.iloc[i-9],
        'Ret_AxRet_B_10' : HS_ST.iloc[i-10],
        'Ret_AxRet_B_11' : HS_ST.iloc[i-11],
        'Ret_AxRet_B_12' : HS_ST.iloc[i-12],
        'Ret_AxRet_B_13' : HS_ST.iloc[i-13],
        'Ret_AxRet_B_14' : HS_ST.iloc[i-14],
        'Ret_AxRet_B_15' : HS_ST.iloc[i-15],
        'Ret_AxRet_B_16' : HS_ST.iloc[i-16],
        'Ret_AxRet_B_17' : HS_ST.iloc[i-17],
        'Ret_AxRet_B_18' : HS_ST.iloc[i-18],
        'Ret_AxRet_B_19' : HS_ST.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : HS_ST.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : HS_ST.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : HS_ST.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : HS_ST.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : HS_ST.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : HS_ST.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : HS_ST.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : HS_ST.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : HS_ST.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : HS_ST.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : HS_ST.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : HS_ST.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : HS_ST.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : HS_ST.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : HS_ST.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : HS_ST.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : HS_ST.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : HS_ST.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : HS_ST.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : HS_ST.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['HSI'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['STOXX'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['HSI'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['STOXX'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['HSI'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['STOXX'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['HSI'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['STOXX'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['HSI'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['STOXX'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['HSI'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['STOXX'].iloc[i],
        
        'A_B_rm_20' : HS_ST_rm_20.iloc[i],
        'A_B_rm_50' : HS_ST_rm_50.iloc[i],
        'A_B_rm_120' : HS_ST_rm_120.iloc[i],
        
        'A_B_cor_20' : HS_ST_cor_20.iloc[i],
        'A_B_cor_50' : HS_ST_cor_50.iloc[i],
        'A_B_cor_120' : HS_ST_cor_120.iloc[i],
        
        
        'Target' : HS_ST_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
     #10
    df_dict = {
        'Date': daily_returns.index[i],
        'A_B' : 'NIFTY50_STOXX',
        'A' : 'NIFTY50',
        'B' : 'STOXX',
        
        'Ret_A_0' : daily_returns['NIFTY50'].iloc[i-0],
        'Ret_A_0' : daily_returns['NIFTY50'].iloc[i-0],
        'Ret_A_1' : daily_returns['NIFTY50'].iloc[i-1],
        'Ret_A_2' : daily_returns['NIFTY50'].iloc[i-2],
        'Ret_A_3' : daily_returns['NIFTY50'].iloc[i-3],
        'Ret_A_4' : daily_returns['NIFTY50'].iloc[i-4],
        'Ret_A_5' : daily_returns['NIFTY50'].iloc[i-5],
        'Ret_A_6' : daily_returns['NIFTY50'].iloc[i-6],
        'Ret_A_7' : daily_returns['NIFTY50'].iloc[i-7],
        'Ret_A_8' : daily_returns['NIFTY50'].iloc[i-8],
        'Ret_A_9' : daily_returns['NIFTY50'].iloc[i-9],
        'Ret_A_10' : daily_returns['NIFTY50'].iloc[i-10],
        'Ret_A_11' : daily_returns['NIFTY50'].iloc[i-11],
        'Ret_A_12' : daily_returns['NIFTY50'].iloc[i-12],
        'Ret_A_13' : daily_returns['NIFTY50'].iloc[i-13],
        'Ret_A_14' : daily_returns['NIFTY50'].iloc[i-14],
        'Ret_A_15' : daily_returns['NIFTY50'].iloc[i-15],
        'Ret_A_16' : daily_returns['NIFTY50'].iloc[i-16],
        'Ret_A_17' : daily_returns['NIFTY50'].iloc[i-17],
        'Ret_A_18' : daily_returns['NIFTY50'].iloc[i-18],
        'Ret_A_19' : daily_returns['NIFTY50'].iloc[i-19],
        
        'Ret_B_0' : daily_returns['STOXX'].iloc[i-0],
        'Ret_B_1' : daily_returns['STOXX'].iloc[i-1],
        'Ret_B_2' : daily_returns['STOXX'].iloc[i-2],
        'Ret_B_3' : daily_returns['STOXX'].iloc[i-3],
        'Ret_B_4' : daily_returns['STOXX'].iloc[i-4],
        'Ret_B_5' : daily_returns['STOXX'].iloc[i-5],
        'Ret_B_6' : daily_returns['STOXX'].iloc[i-6],
        'Ret_B_7' : daily_returns['STOXX'].iloc[i-7],
        'Ret_B_8' : daily_returns['STOXX'].iloc[i-8],
        'Ret_B_9' : daily_returns['STOXX'].iloc[i-9],
        'Ret_B_10' : daily_returns['STOXX'].iloc[i-10],
        'Ret_B_11' : daily_returns['STOXX'].iloc[i-11],
        'Ret_B_12' : daily_returns['STOXX'].iloc[i-12],
        'Ret_B_13' : daily_returns['STOXX'].iloc[i-13],
        'Ret_B_14' : daily_returns['STOXX'].iloc[i-14],
        'Ret_B_15' : daily_returns['STOXX'].iloc[i-15],
        'Ret_B_16' : daily_returns['STOXX'].iloc[i-16],
        'Ret_B_17' : daily_returns['STOXX'].iloc[i-17],
        'Ret_B_18' : daily_returns['STOXX'].iloc[i-18],
        'Ret_B_19' : daily_returns['STOXX'].iloc[i-19],
        
        'Sq_Ret_A_0' : daily_returns_sq['NIFTY50'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['NIFTY50'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['NIFTY50'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['NIFTY50'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['NIFTY50'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['NIFTY50'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['NIFTY50'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['NIFTY50'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['NIFTY50'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['NIFTY50'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['NIFTY50'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['NIFTY50'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['NIFTY50'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['NIFTY50'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['NIFTY50'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['NIFTY50'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['NIFTY50'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['NIFTY50'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['NIFTY50'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['NIFTY50'].iloc[i-19],
        
        'Sq_Ret_B_0' : daily_returns_sq['STOXX'].iloc[i-0],
        'Sq_Ret_B_1' : daily_returns_sq['STOXX'].iloc[i-1],
        'Sq_Ret_B_2' : daily_returns_sq['STOXX'].iloc[i-2],
        'Sq_Ret_B_3' : daily_returns_sq['STOXX'].iloc[i-3],
        'Sq_Ret_B_4' : daily_returns_sq['STOXX'].iloc[i-4],
        'Sq_Ret_B_5' : daily_returns_sq['STOXX'].iloc[i-5],
        'Sq_Ret_B_6' : daily_returns_sq['STOXX'].iloc[i-6],
        'Sq_Ret_B_7' : daily_returns_sq['STOXX'].iloc[i-7],
        'Sq_Ret_B_8' : daily_returns_sq['STOXX'].iloc[i-8],
        'Sq_Ret_B_9' : daily_returns_sq['STOXX'].iloc[i-9],
        'Sq_Ret_B_10' : daily_returns_sq['STOXX'].iloc[i-10],
        'Sq_Ret_B_11' : daily_returns_sq['STOXX'].iloc[i-11],
        'Sq_Ret_B_12' : daily_returns_sq['STOXX'].iloc[i-12],
        'Sq_Ret_B_13' : daily_returns_sq['STOXX'].iloc[i-13],
        'Sq_Ret_B_14' : daily_returns_sq['STOXX'].iloc[i-14],
        'Sq_Ret_B_15' : daily_returns_sq['STOXX'].iloc[i-15],
        'Sq_Ret_B_16' : daily_returns_sq['STOXX'].iloc[i-16],
        'Sq_Ret_B_17' : daily_returns_sq['STOXX'].iloc[i-17],
        'Sq_Ret_B_18' : daily_returns_sq['STOXX'].iloc[i-18],
        'Sq_Ret_B_19' : daily_returns_sq['STOXX'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['NIFTY50'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['NIFTY50'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['NIFTY50'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['NIFTY50'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['NIFTY50'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['NIFTY50'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['NIFTY50'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['NIFTY50'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['NIFTY50'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['NIFTY50'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['NIFTY50'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['NIFTY50'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['NIFTY50'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['NIFTY50'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['NIFTY50'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['NIFTY50'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['NIFTY50'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['NIFTY50'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['NIFTY50'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['NIFTY50'].iloc[i-19],
        
        'Abs_Ret_B_0' : daily_returns_abs['STOXX'].iloc[i-0],
        'Abs_Ret_B_1' : daily_returns_abs['STOXX'].iloc[i-1],
        'Abs_Ret_B_2' : daily_returns_abs['STOXX'].iloc[i-2],
        'Abs_Ret_B_3' : daily_returns_abs['STOXX'].iloc[i-3],
        'Abs_Ret_B_4' : daily_returns_abs['STOXX'].iloc[i-4],
        'Abs_Ret_B_5' : daily_returns_abs['STOXX'].iloc[i-5],
        'Abs_Ret_B_6' : daily_returns_abs['STOXX'].iloc[i-6],
        'Abs_Ret_B_7' : daily_returns_abs['STOXX'].iloc[i-7],
        'Abs_Ret_B_8' : daily_returns_abs['STOXX'].iloc[i-8],
        'Abs_Ret_B_9' : daily_returns_abs['STOXX'].iloc[i-9],
        'Abs_Ret_B_10' : daily_returns_abs['STOXX'].iloc[i-10],
        'Abs_Ret_B_11' : daily_returns_abs['STOXX'].iloc[i-11],
        'Abs_Ret_B_12' : daily_returns_abs['STOXX'].iloc[i-12],
        'Abs_Ret_B_13' : daily_returns_abs['STOXX'].iloc[i-13],
        'Abs_Ret_B_14' : daily_returns_abs['STOXX'].iloc[i-14],
        'Abs_Ret_B_15' : daily_returns_abs['STOXX'].iloc[i-15],
        'Abs_Ret_B_16' : daily_returns_abs['STOXX'].iloc[i-16],
        'Abs_Ret_B_17' : daily_returns_abs['STOXX'].iloc[i-17],
        'Abs_Ret_B_18' : daily_returns_abs['STOXX'].iloc[i-18],
        'Abs_Ret_B_19' : daily_returns_abs['STOXX'].iloc[i-19],

        'Ret_AxRet_B_0' : NI_ST.iloc[i-0],
        'Ret_AxRet_B_1' : NI_ST.iloc[i-1],
        'Ret_AxRet_B_2' : NI_ST.iloc[i-2],
        'Ret_AxRet_B_3' : NI_ST.iloc[i-3],
        'Ret_AxRet_B_4' : NI_ST.iloc[i-4],
        'Ret_AxRet_B_5' : NI_ST.iloc[i-5],
        'Ret_AxRet_B_6' : NI_ST.iloc[i-6],
        'Ret_AxRet_B_7' : NI_ST.iloc[i-7],
        'Ret_AxRet_B_8' : NI_ST.iloc[i-8],
        'Ret_AxRet_B_9' : NI_ST.iloc[i-9],
        'Ret_AxRet_B_10' : NI_ST.iloc[i-10],
        'Ret_AxRet_B_11' : NI_ST.iloc[i-11],
        'Ret_AxRet_B_12' : NI_ST.iloc[i-12],
        'Ret_AxRet_B_13' : NI_ST.iloc[i-13],
        'Ret_AxRet_B_14' : NI_ST.iloc[i-14],
        'Ret_AxRet_B_15' : NI_ST.iloc[i-15],
        'Ret_AxRet_B_16' : NI_ST.iloc[i-16],
        'Ret_AxRet_B_17' : NI_ST.iloc[i-17],
        'Ret_AxRet_B_18' : NI_ST.iloc[i-18],
        'Ret_AxRet_B_19' : NI_ST.iloc[i-19],
        
        'Abs_Ret_AxRet_B_0' : NI_ST.iloc[i-0],
        'Abs_Ret_AxRet_B_1' : NI_ST.iloc[i-1],
        'Abs_Ret_AxRet_B_2' : NI_ST.iloc[i-2],
        'Abs_Ret_AxRet_B_3' : NI_ST.iloc[i-3],
        'Abs_Ret_AxRet_B_4' : NI_ST.iloc[i-4],
        'Abs_Ret_AxRet_B_5' : NI_ST.iloc[i-5],
        'Abs_Ret_AxRet_B_6' : NI_ST.iloc[i-6],
        'Abs_Ret_AxRet_B_7' : NI_ST.iloc[i-7],
        'Abs_Ret_AxRet_B_8' : NI_ST.iloc[i-8],
        'Abs_Ret_AxRet_B_9' : NI_ST.iloc[i-9],
        'Abs_Ret_AxRet_B_10' : NI_ST.iloc[i-10],
        'Abs_Ret_AxRet_B_11' : NI_ST.iloc[i-11],
        'Abs_Ret_AxRet_B_12' : NI_ST.iloc[i-12],
        'Abs_Ret_AxRet_B_13' : NI_ST.iloc[i-13],
        'Abs_Ret_AxRet_B_14' : NI_ST.iloc[i-14],
        'Abs_Ret_AxRet_B_15' : NI_ST.iloc[i-15],
        'Abs_Ret_AxRet_B_16' : NI_ST.iloc[i-16],
        'Abs_Ret_AxRet_B_17' : NI_ST.iloc[i-17],
        'Abs_Ret_AxRet_B_18' : NI_ST.iloc[i-18],
        'Abs_Ret_AxRet_B_19' : NI_ST.iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['NIFTY50'].iloc[i],
        'B_rm_20' : dr_rolling_mean_20['STOXX'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['NIFTY50'].iloc[i],
        'B_rm_50' : dr_rolling_mean_50['STOXX'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['NIFTY50'].iloc[i],
        'B_rm_120' : dr_rolling_mean_120['STOXX'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['NIFTY50'].iloc[i],
        'B_rm_sq_20' : dr_rolling_sq_mean_20['STOXX'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['NIFTY50'].iloc[i],
        'B_rm_sq_50' : dr_rolling_sq_mean_50['STOXX'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['NIFTY50'].iloc[i],
        'B_rm_sq_120'  : dr_rolling_sq_mean_120['STOXX'].iloc[i],
        
        'A_B_rm_20' : NI_ST_rm_20.iloc[i],
        'A_B_rm_50' : NI_ST_rm_50.iloc[i],
        'A_B_rm_120' : NI_ST_rm_120.iloc[i],
        
        'A_B_cor_20' : NI_ST_cor_20.iloc[i],
        'A_B_cor_50' : NI_ST_cor_50.iloc[i],
        'A_B_cor_120' : NI_ST_cor_120.iloc[i],
        
        
        'Target' : NI_ST_y.iloc[i]           
    }
    df = pd.concat([df, pd.DataFrame(df_dict, index=[0])], ignore_index=True)


# In[20]:


#Creation of st dev dataframe
columns = ['Date','A',
           'Ret_A_0','Ret_A_1','Ret_A_2','Ret_A_3','Ret_A_4','Ret_A_5','Ret_A_6',
           'Ret_A_7','Ret_A_8','Ret_A_9','Ret_A_10','Ret_A_11','Ret_A_12','Ret_A_13',
           'Ret_A_14','Ret_A_15','Ret_A_16','Ret_A_17','Ret_A_18','Ret_A_19',
           'Sq_Ret_A_0','Sq_Ret_A_1','Sq_Ret_A_2','Sq_Ret_A_3','Sq_Ret_A_4','Sq_Ret_A_5','Sq_Ret_A_6',
           'Sq_Ret_A_7','Sq_Ret_A_8','Sq_Ret_A_9','Sq_Ret_A_10','Sq_Ret_A_11','Sq_Ret_A_12','Sq_Ret_A_13',
           'Sq_Ret_A_14','Sq_Ret_A_15','Sq_Ret_A_16','Sq_Ret_A_17','Sq_Ret_A_18','Sq_Ret_A_19',
           'Abs_Ret_A_0','Abs_Ret_A_1','Abs_Ret_A_2','Abs_Ret_A_3','Abs_Ret_A_4','Abs_Ret_A_5','Abs_Ret_A_6',
           'Abs_Ret_A_7','Abs_Ret_A_8','Abs_Ret_A_9','Abs_Ret_A_10','Abs_Ret_A_11','Abs_Ret_A_12','Abs_Ret_A_13',
           'Abs_Ret_A_14','Abs_Ret_A_15','Abs_Ret_A_16','Abs_Ret_A_17','Abs_Ret_A_18','Abs_Ret_A_19',
           'A_rm_20','A_rm_50','A_rm_120',
           'A_rm_sq_20','A_rm_sq_50','A_rm_sq_120',
           'A_std_20','A_std_50','A_std_120',
           'Target']
df_std = pd.DataFrame(columns=columns)


# In[21]:


for i in range(120,daily_returns.shape[0]-20):
    print(i,'/',daily_returns.shape[0]-20)
# for i in range(120,125):
    #1
    df_dict = {
        'Date': daily_returns.index[i],
        'A' : 'SP500',
        
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_0' : daily_returns['SP500'].iloc[i-0],
        'Ret_A_1' : daily_returns['SP500'].iloc[i-1],
        'Ret_A_2' : daily_returns['SP500'].iloc[i-2],
        'Ret_A_3' : daily_returns['SP500'].iloc[i-3],
        'Ret_A_4' : daily_returns['SP500'].iloc[i-4],
        'Ret_A_5' : daily_returns['SP500'].iloc[i-5],
        'Ret_A_6' : daily_returns['SP500'].iloc[i-6],
        'Ret_A_7' : daily_returns['SP500'].iloc[i-7],
        'Ret_A_8' : daily_returns['SP500'].iloc[i-8],
        'Ret_A_9' : daily_returns['SP500'].iloc[i-9],
        'Ret_A_10' : daily_returns['SP500'].iloc[i-10],
        'Ret_A_11' : daily_returns['SP500'].iloc[i-11],
        'Ret_A_12' : daily_returns['SP500'].iloc[i-12],
        'Ret_A_13' : daily_returns['SP500'].iloc[i-13],
        'Ret_A_14' : daily_returns['SP500'].iloc[i-14],
        'Ret_A_15' : daily_returns['SP500'].iloc[i-15],
        'Ret_A_16' : daily_returns['SP500'].iloc[i-16],
        'Ret_A_17' : daily_returns['SP500'].iloc[i-17],
        'Ret_A_18' : daily_returns['SP500'].iloc[i-18],
        'Ret_A_19' : daily_returns['SP500'].iloc[i-19],
        

        'Sq_Ret_A_0' : daily_returns_sq['SP500'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['SP500'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['SP500'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['SP500'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['SP500'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['SP500'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['SP500'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['SP500'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['SP500'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['SP500'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['SP500'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['SP500'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['SP500'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['SP500'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['SP500'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['SP500'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['SP500'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['SP500'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['SP500'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['SP500'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['SP500'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['SP500'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['SP500'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['SP500'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['SP500'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['SP500'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['SP500'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['SP500'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['SP500'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['SP500'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['SP500'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['SP500'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['SP500'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['SP500'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['SP500'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['SP500'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['SP500'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['SP500'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['SP500'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['SP500'].iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['SP500'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['SP500'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['SP500'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['SP500'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['SP500'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['SP500'].iloc[i],
        
        'A_std_20' : dr_rolling_std_20['SP500'].iloc[i],
        'A_std_50' : dr_rolling_std_50['SP500'].iloc[i],
        'A_std_120' : dr_rolling_std_120['SP500'].iloc[i],
        
        
        'Target' : SP_y.iloc[i]           
    }
    df_std = pd.concat([df_std, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
        #2
    df_dict = {
        'Date': daily_returns.index[i],
        'A' : 'STOXX',
        
        'Ret_A_0' : daily_returns['STOXX'].iloc[i-0],
        'Ret_A_0' : daily_returns['STOXX'].iloc[i-0],
        'Ret_A_1' : daily_returns['STOXX'].iloc[i-1],
        'Ret_A_2' : daily_returns['STOXX'].iloc[i-2],
        'Ret_A_3' : daily_returns['STOXX'].iloc[i-3],
        'Ret_A_4' : daily_returns['STOXX'].iloc[i-4],
        'Ret_A_5' : daily_returns['STOXX'].iloc[i-5],
        'Ret_A_6' : daily_returns['STOXX'].iloc[i-6],
        'Ret_A_7' : daily_returns['STOXX'].iloc[i-7],
        'Ret_A_8' : daily_returns['STOXX'].iloc[i-8],
        'Ret_A_9' : daily_returns['STOXX'].iloc[i-9],
        'Ret_A_10' : daily_returns['STOXX'].iloc[i-10],
        'Ret_A_11' : daily_returns['STOXX'].iloc[i-11],
        'Ret_A_12' : daily_returns['STOXX'].iloc[i-12],
        'Ret_A_13' : daily_returns['STOXX'].iloc[i-13],
        'Ret_A_14' : daily_returns['STOXX'].iloc[i-14],
        'Ret_A_15' : daily_returns['STOXX'].iloc[i-15],
        'Ret_A_16' : daily_returns['STOXX'].iloc[i-16],
        'Ret_A_17' : daily_returns['STOXX'].iloc[i-17],
        'Ret_A_18' : daily_returns['STOXX'].iloc[i-18],
        'Ret_A_19' : daily_returns['STOXX'].iloc[i-19],
        

        'Sq_Ret_A_0' : daily_returns_sq['STOXX'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['STOXX'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['STOXX'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['STOXX'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['STOXX'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['STOXX'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['STOXX'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['STOXX'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['STOXX'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['STOXX'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['STOXX'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['STOXX'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['STOXX'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['STOXX'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['STOXX'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['STOXX'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['STOXX'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['STOXX'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['STOXX'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['STOXX'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['STOXX'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['STOXX'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['STOXX'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['STOXX'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['STOXX'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['STOXX'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['STOXX'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['STOXX'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['STOXX'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['STOXX'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['STOXX'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['STOXX'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['STOXX'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['STOXX'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['STOXX'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['STOXX'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['STOXX'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['STOXX'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['STOXX'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['STOXX'].iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['STOXX'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['STOXX'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['STOXX'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['STOXX'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['STOXX'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['STOXX'].iloc[i],
        
        'A_std_20' : dr_rolling_std_20['STOXX'].iloc[i],
        'A_std_50' : dr_rolling_std_50['STOXX'].iloc[i],
        'A_std_120' : dr_rolling_std_120['STOXX'].iloc[i],
        
        
        'Target' : ST_y.iloc[i]           
    }
    df_std = pd.concat([df_std, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
        #3
    df_dict = {
        'Date': daily_returns.index[i],
        'A' : 'BOVESPA',
        
        'Ret_A_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_A_0' : daily_returns['BOVESPA'].iloc[i-0],
        'Ret_A_1' : daily_returns['BOVESPA'].iloc[i-1],
        'Ret_A_2' : daily_returns['BOVESPA'].iloc[i-2],
        'Ret_A_3' : daily_returns['BOVESPA'].iloc[i-3],
        'Ret_A_4' : daily_returns['BOVESPA'].iloc[i-4],
        'Ret_A_5' : daily_returns['BOVESPA'].iloc[i-5],
        'Ret_A_6' : daily_returns['BOVESPA'].iloc[i-6],
        'Ret_A_7' : daily_returns['BOVESPA'].iloc[i-7],
        'Ret_A_8' : daily_returns['BOVESPA'].iloc[i-8],
        'Ret_A_9' : daily_returns['BOVESPA'].iloc[i-9],
        'Ret_A_10' : daily_returns['BOVESPA'].iloc[i-10],
        'Ret_A_11' : daily_returns['BOVESPA'].iloc[i-11],
        'Ret_A_12' : daily_returns['BOVESPA'].iloc[i-12],
        'Ret_A_13' : daily_returns['BOVESPA'].iloc[i-13],
        'Ret_A_14' : daily_returns['BOVESPA'].iloc[i-14],
        'Ret_A_15' : daily_returns['BOVESPA'].iloc[i-15],
        'Ret_A_16' : daily_returns['BOVESPA'].iloc[i-16],
        'Ret_A_17' : daily_returns['BOVESPA'].iloc[i-17],
        'Ret_A_18' : daily_returns['BOVESPA'].iloc[i-18],
        'Ret_A_19' : daily_returns['BOVESPA'].iloc[i-19],
        

        'Sq_Ret_A_0' : daily_returns_sq['BOVESPA'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['BOVESPA'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['BOVESPA'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['BOVESPA'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['BOVESPA'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['BOVESPA'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['BOVESPA'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['BOVESPA'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['BOVESPA'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['BOVESPA'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['BOVESPA'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['BOVESPA'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['BOVESPA'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['BOVESPA'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['BOVESPA'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['BOVESPA'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['BOVESPA'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['BOVESPA'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['BOVESPA'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['BOVESPA'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['BOVESPA'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['BOVESPA'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['BOVESPA'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['BOVESPA'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['BOVESPA'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['BOVESPA'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['BOVESPA'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['BOVESPA'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['BOVESPA'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['BOVESPA'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['BOVESPA'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['BOVESPA'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['BOVESPA'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['BOVESPA'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['BOVESPA'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['BOVESPA'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['BOVESPA'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['BOVESPA'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['BOVESPA'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['BOVESPA'].iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['BOVESPA'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['BOVESPA'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['BOVESPA'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['BOVESPA'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['BOVESPA'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['BOVESPA'].iloc[i],
        
        'A_std_20' : dr_rolling_std_20['BOVESPA'].iloc[i],
        'A_std_50' : dr_rolling_std_50['BOVESPA'].iloc[i],
        'A_std_120' : dr_rolling_std_120['BOVESPA'].iloc[i],
        
        
        'Target' : BV_y.iloc[i]           
    }
    df_std = pd.concat([df_std, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
        #4
    df_dict = {
        'Date': daily_returns.index[i],
        'A' : 'NIFTY50',
        
        'Ret_A_0' : daily_returns['NIFTY50'].iloc[i-0],
        'Ret_A_0' : daily_returns['NIFTY50'].iloc[i-0],
        'Ret_A_1' : daily_returns['NIFTY50'].iloc[i-1],
        'Ret_A_2' : daily_returns['NIFTY50'].iloc[i-2],
        'Ret_A_3' : daily_returns['NIFTY50'].iloc[i-3],
        'Ret_A_4' : daily_returns['NIFTY50'].iloc[i-4],
        'Ret_A_5' : daily_returns['NIFTY50'].iloc[i-5],
        'Ret_A_6' : daily_returns['NIFTY50'].iloc[i-6],
        'Ret_A_7' : daily_returns['NIFTY50'].iloc[i-7],
        'Ret_A_8' : daily_returns['NIFTY50'].iloc[i-8],
        'Ret_A_9' : daily_returns['NIFTY50'].iloc[i-9],
        'Ret_A_10' : daily_returns['NIFTY50'].iloc[i-10],
        'Ret_A_11' : daily_returns['NIFTY50'].iloc[i-11],
        'Ret_A_12' : daily_returns['NIFTY50'].iloc[i-12],
        'Ret_A_13' : daily_returns['NIFTY50'].iloc[i-13],
        'Ret_A_14' : daily_returns['NIFTY50'].iloc[i-14],
        'Ret_A_15' : daily_returns['NIFTY50'].iloc[i-15],
        'Ret_A_16' : daily_returns['NIFTY50'].iloc[i-16],
        'Ret_A_17' : daily_returns['NIFTY50'].iloc[i-17],
        'Ret_A_18' : daily_returns['NIFTY50'].iloc[i-18],
        'Ret_A_19' : daily_returns['NIFTY50'].iloc[i-19],
        

        'Sq_Ret_A_0' : daily_returns_sq['NIFTY50'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['NIFTY50'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['NIFTY50'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['NIFTY50'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['NIFTY50'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['NIFTY50'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['NIFTY50'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['NIFTY50'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['NIFTY50'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['NIFTY50'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['NIFTY50'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['NIFTY50'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['NIFTY50'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['NIFTY50'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['NIFTY50'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['NIFTY50'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['NIFTY50'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['NIFTY50'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['NIFTY50'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['NIFTY50'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['NIFTY50'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['NIFTY50'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['NIFTY50'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['NIFTY50'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['NIFTY50'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['NIFTY50'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['NIFTY50'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['NIFTY50'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['NIFTY50'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['NIFTY50'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['NIFTY50'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['NIFTY50'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['NIFTY50'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['NIFTY50'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['NIFTY50'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['NIFTY50'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['NIFTY50'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['NIFTY50'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['NIFTY50'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['NIFTY50'].iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['NIFTY50'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['NIFTY50'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['NIFTY50'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['NIFTY50'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['NIFTY50'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['NIFTY50'].iloc[i],
        
        'A_std_20' : dr_rolling_std_20['NIFTY50'].iloc[i],
        'A_std_50' : dr_rolling_std_50['NIFTY50'].iloc[i],
        'A_std_120' : dr_rolling_std_120['NIFTY50'].iloc[i],
        
        
        'Target' : NI_y.iloc[i]           
    }
    df_std = pd.concat([df_std, pd.DataFrame(df_dict, index=[0])], ignore_index=True)
    
        #5
    df_dict = {
        'Date': daily_returns.index[i],
        'A' : 'HSI',
        
        'Ret_A_0' : daily_returns['HSI'].iloc[i-0],
        'Ret_A_0' : daily_returns['HSI'].iloc[i-0],
        'Ret_A_1' : daily_returns['HSI'].iloc[i-1],
        'Ret_A_2' : daily_returns['HSI'].iloc[i-2],
        'Ret_A_3' : daily_returns['HSI'].iloc[i-3],
        'Ret_A_4' : daily_returns['HSI'].iloc[i-4],
        'Ret_A_5' : daily_returns['HSI'].iloc[i-5],
        'Ret_A_6' : daily_returns['HSI'].iloc[i-6],
        'Ret_A_7' : daily_returns['HSI'].iloc[i-7],
        'Ret_A_8' : daily_returns['HSI'].iloc[i-8],
        'Ret_A_9' : daily_returns['HSI'].iloc[i-9],
        'Ret_A_10' : daily_returns['HSI'].iloc[i-10],
        'Ret_A_11' : daily_returns['HSI'].iloc[i-11],
        'Ret_A_12' : daily_returns['HSI'].iloc[i-12],
        'Ret_A_13' : daily_returns['HSI'].iloc[i-13],
        'Ret_A_14' : daily_returns['HSI'].iloc[i-14],
        'Ret_A_15' : daily_returns['HSI'].iloc[i-15],
        'Ret_A_16' : daily_returns['HSI'].iloc[i-16],
        'Ret_A_17' : daily_returns['HSI'].iloc[i-17],
        'Ret_A_18' : daily_returns['HSI'].iloc[i-18],
        'Ret_A_19' : daily_returns['HSI'].iloc[i-19],
        

        'Sq_Ret_A_0' : daily_returns_sq['HSI'].iloc[i-0],
        'Sq_Ret_A_1' : daily_returns_sq['HSI'].iloc[i-1],
        'Sq_Ret_A_2' : daily_returns_sq['HSI'].iloc[i-2],
        'Sq_Ret_A_3' : daily_returns_sq['HSI'].iloc[i-3],
        'Sq_Ret_A_4' : daily_returns_sq['HSI'].iloc[i-4],
        'Sq_Ret_A_5' : daily_returns_sq['HSI'].iloc[i-5],
        'Sq_Ret_A_6' : daily_returns_sq['HSI'].iloc[i-6],
        'Sq_Ret_A_7' : daily_returns_sq['HSI'].iloc[i-7],
        'Sq_Ret_A_8' : daily_returns_sq['HSI'].iloc[i-8],
        'Sq_Ret_A_9' : daily_returns_sq['HSI'].iloc[i-9],
        'Sq_Ret_A_10' : daily_returns_sq['HSI'].iloc[i-10],
        'Sq_Ret_A_11' : daily_returns_sq['HSI'].iloc[i-11],
        'Sq_Ret_A_12' : daily_returns_sq['HSI'].iloc[i-12],
        'Sq_Ret_A_13' : daily_returns_sq['HSI'].iloc[i-13],
        'Sq_Ret_A_14' : daily_returns_sq['HSI'].iloc[i-14],
        'Sq_Ret_A_15' : daily_returns_sq['HSI'].iloc[i-15],
        'Sq_Ret_A_16' : daily_returns_sq['HSI'].iloc[i-16],
        'Sq_Ret_A_17' : daily_returns_sq['HSI'].iloc[i-17],
        'Sq_Ret_A_18' : daily_returns_sq['HSI'].iloc[i-18],
        'Sq_Ret_A_19' : daily_returns_sq['HSI'].iloc[i-19],
        
        'Abs_Ret_A_0' : daily_returns_abs['HSI'].iloc[i-0],
        'Abs_Ret_A_1' : daily_returns_abs['HSI'].iloc[i-1],
        'Abs_Ret_A_2' : daily_returns_abs['HSI'].iloc[i-2],
        'Abs_Ret_A_3' : daily_returns_abs['HSI'].iloc[i-3],
        'Abs_Ret_A_4' : daily_returns_abs['HSI'].iloc[i-4],
        'Abs_Ret_A_5' : daily_returns_abs['HSI'].iloc[i-5],
        'Abs_Ret_A_6' : daily_returns_abs['HSI'].iloc[i-6],
        'Abs_Ret_A_7' : daily_returns_abs['HSI'].iloc[i-7],
        'Abs_Ret_A_8' : daily_returns_abs['HSI'].iloc[i-8],
        'Abs_Ret_A_9' : daily_returns_abs['HSI'].iloc[i-9],
        'Abs_Ret_A_10' : daily_returns_abs['HSI'].iloc[i-10],
        'Abs_Ret_A_11' : daily_returns_abs['HSI'].iloc[i-11],
        'Abs_Ret_A_12' : daily_returns_abs['HSI'].iloc[i-12],
        'Abs_Ret_A_13' : daily_returns_abs['HSI'].iloc[i-13],
        'Abs_Ret_A_14' : daily_returns_abs['HSI'].iloc[i-14],
        'Abs_Ret_A_15' : daily_returns_abs['HSI'].iloc[i-15],
        'Abs_Ret_A_16' : daily_returns_abs['HSI'].iloc[i-16],
        'Abs_Ret_A_17' : daily_returns_abs['HSI'].iloc[i-17],
        'Abs_Ret_A_18' : daily_returns_abs['HSI'].iloc[i-18],
        'Abs_Ret_A_19' : daily_returns_abs['HSI'].iloc[i-19],

        'A_rm_20' : dr_rolling_mean_20['HSI'].iloc[i],
        'A_rm_50' : dr_rolling_mean_50['HSI'].iloc[i],
        'A_rm_120' : dr_rolling_mean_120['HSI'].iloc[i],
        
        'A_rm_sq_20' : dr_rolling_sq_mean_20['HSI'].iloc[i],
        'A_rm_sq_50' : dr_rolling_sq_mean_50['HSI'].iloc[i],
        'A_rm_sq_120' : dr_rolling_sq_mean_120['HSI'].iloc[i],
        
        'A_std_20' : dr_rolling_std_20['HSI'].iloc[i],
        'A_std_50' : dr_rolling_std_50['HSI'].iloc[i],
        'A_std_120' : dr_rolling_std_120['HSI'].iloc[i],
        
        
        'Target' : HS_y.iloc[i]           
    }
    df_std = pd.concat([df_std, pd.DataFrame(df_dict, index=[0])], ignore_index=True)


# In[22]:


# df.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/Correlation_ML_df.csv')
# df_std.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/Std_ML_df.csv')


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import keras


# In[24]:


X = df_std.drop(columns = ['Date', 'A', 'Target'])
index_df = df_std[['Date', 'A']]
y = df_std['Target']              


# In[25]:


X_train = X.loc[:13614]
X_test = X.loc[13614 + 1:]

y_train = y[:13615]
y_test = y[13614 + 1:]


# In[26]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)

X_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)


# In[27]:


def resnet_block(inputs, num_layers = 32, dropout_rate = 0.1):
    x = tf.keras.layers.Dense(num_layers)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.Add()([x,inputs])
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.Add()([x,inputs])
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    return x

def attention_block(inputs, num_layers = 32, dropout_rate = 0.1):
    
    x = tf.keras.layers.Dense(num_layers)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers, activation='softmax')(x)
    
    return x
    
def resnet_attention(inputs,num_layers = 32, dropout_rate = 0.1):    
    
    res = resnet_block(inputs,num_layers,dropout_rate)
    attn = attention_block(inputs,num_layers,dropout_rate)
    res = tf.keras.layers.Multiply()([res,attn])
    
    res = tf.keras.layers.Dense(16)(res)
    res = tf.keras.layers.LeakyReLU()(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res) 
    
    res = tf.keras.layers.Dense(4)(res)
    res = tf.keras.layers.LeakyReLU()(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res) 
    
    res = tf.keras.layers.Dense(1, activation='relu')(res)
    
    model = tf.keras.Model(inputs = inputs,outputs=res)
    
    return model


# In[28]:


print(X_train_tf.shape)
print(X_test_tf.shape)
print(y_train_tf.shape)
print(y_test_tf.shape)


# In[29]:


def r_squared(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred)) 
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) 
#     print(SS_res, ",", SS_tot)
    return tf.round((1 - SS_res/SS_tot))


# In[30]:


inputs = tf.keras.Input(shape=(69,))
model = resnet_attention(inputs,69)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=500,decay_rate=0.995)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[r_squared])


# In[31]:


history = model.fit(X_train_tf, y_train_tf, epochs=100,batch_size=512, validation_split=0.2, verbose=2)


# In[32]:


test_loss, test_acc = model.evaluate(X_test_tf, y_test_tf)


# In[33]:


pred = model.predict(X_test_tf)


# In[34]:


plt.plot(index_df['Date'].iloc[13614 + 1:], y_test, label='Actual')
plt.plot(index_df['Date'].iloc[13614 + 1:], pred, label='Predicted')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Two Line Charts')
plt.legend()
plt.show()


# In[35]:


st_dev_pred = pd.DataFrame({'Date' : np.array(index_df['Date'].iloc[13614 + 1:]),
                            'A' : np.array(index_df['A'].iloc[13614 + 1:]),
                            'St_Dev' : np.array(pred).reshape(-1)})
st_dev_pred = st_dev_pred.pivot_table(index=st_dev_pred['Date'], columns='A', values='St_Dev', aggfunc='first')


# In[37]:


st_dev_pred.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/ML Std Stocks.csv')


# In[52]:


# index_df['Date'].iloc[27229]


# In[187]:


#Correlation model
X = df.drop(columns = ['Date', 'A_B', 'A', 'B', 'Target'])
index_df = df[['Date', 'A_B', 'A', 'B']]
y = df['Target']  

X_train = X.loc[:27229]
X_test = X.loc[27229 + 1:]

y_train = y[:27230]
y_test = y[27229 + 1:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)

X_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)


# In[188]:


print(X_train_tf.shape)
print(X_test_tf.shape)
print(y_train_tf.shape)
print(y_test_tf.shape)


# In[189]:


def resnet_block_cor(inputs, num_layers = 32, dropout_rate = 0.1):
    x = tf.keras.layers.Dense(num_layers)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.Add()([x,inputs])
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.Add()([x,inputs])
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    return x

def attention_block_cor(inputs, num_layers = 32, dropout_rate = 0.1):
    
    x = tf.keras.layers.Dense(num_layers)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dense(num_layers, activation='softmax')(x)
    
    return x
    
def resnet_attention_cor(inputs,num_layers = 32, dropout_rate = 0.1):    
    
    res = resnet_block_cor(inputs,num_layers,dropout_rate)
    attn = attention_block_cor(inputs,num_layers,dropout_rate)
    res = tf.keras.layers.Multiply()([res,attn])
    
    res = tf.keras.layers.Dense(64)(res)
    res = tf.keras.layers.LeakyReLU()(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res)
    
    res = tf.keras.layers.Dense(16)(res)
    res = tf.keras.layers.LeakyReLU()(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res) 
    
    res = tf.keras.layers.Dense(4)(res)
    res = tf.keras.layers.LeakyReLU()(res)
    res = tf.keras.layers.Dropout(dropout_rate)(res) 
    
    res = tf.keras.layers.Dense(1, activation='tanh')(res)
    
    model = tf.keras.Model(inputs = inputs,outputs=res)
    
    return model


# In[190]:


inputs = tf.keras.Input(shape=(178,))
model = resnet_attention_cor(inputs,178,0.2)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=500,decay_rate=0.995)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[r_squared])


# In[191]:


history = model.fit(X_train_tf, y_train_tf, epochs=100,batch_size=512, validation_split=0.2, verbose=2)


# In[192]:


test_loss, test_acc = model.evaluate(X_test_tf, y_test_tf)


# In[193]:


pred = model.predict(X_test_tf)


# In[194]:


plt.plot(index_df['Date'].iloc[27229 + 1:], y_test, label='Actual')
plt.plot(index_df['Date'].iloc[27229 + 1:], pred, label='Predicted')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Two Line Charts')
plt.legend()
plt.show()


# In[195]:


cor_pred = pd.DataFrame({'Date' : np.array(index_df['Date'].iloc[27229 + 1:]),
                            'A' : np.array(index_df['A'].iloc[27229 + 1:]),
                         'B' : np.array(index_df['B'].iloc[27229 + 1:]),
                         'A_B' : np.array(index_df['A_B'].iloc[27229 + 1:]),
                          'Cor' : np.array(pred).reshape(-1)})
cor_pred = cor_pred[['Date','A_B','Cor']].pivot_table(index=cor_pred['Date'], columns='A_B', values='Cor', aggfunc='first')


# In[196]:


cor_pred


# In[197]:


##############Calculating portfolios#############################
returns = daily_returns.iloc[2843:4118]


# In[198]:


st_dev_pred = daily_returns.rolling(window=600).std()
st_dev_pred = st_dev_pred.iloc[2843:4118]
st_dev_pred


# In[246]:


ML_w = pd.DataFrame(np.zeros_like(st_dev_pred))

for i in range(st_dev_pred.shape[0]):
    print(i,"/",st_dev_pred.shape[0])
    cov_mat_i = np.array([[st_dev_pred['BOVESPA'].iloc[i]**2,
           st_dev_pred['BOVESPA'].iloc[i]*st_dev_pred['SP500'].iloc[i]*cor_pred['SP500_BOVESPA'].iloc[i],
           st_dev_pred['BOVESPA'].iloc[i]*st_dev_pred['HSI'].iloc[i]*cor_pred['BOVESPA_HSI'].iloc[i],
           st_dev_pred['BOVESPA'].iloc[i]*st_dev_pred['NIFTY50'].iloc[i]*cor_pred['BOVESPA_NIFTY50'].iloc[i],
           st_dev_pred['BOVESPA'].iloc[i]*st_dev_pred['STOXX'].iloc[i]*cor_pred['BOVESPA_STOXX'].iloc[i]],
          [st_dev_pred['BOVESPA'].iloc[i]*st_dev_pred['SP500'].iloc[i]*cor_pred['SP500_BOVESPA'].iloc[i],
           st_dev_pred['SP500'].iloc[i]**2,
           st_dev_pred['SP500'].iloc[i]*st_dev_pred['HSI'].iloc[i]*cor_pred['SP500_HSI'].iloc[i],
           st_dev_pred['SP500'].iloc[i]*st_dev_pred['NIFTY50'].iloc[i]*cor_pred['SP500_NIFTY50'].iloc[i],
           st_dev_pred['SP500'].iloc[i]*st_dev_pred['STOXX'].iloc[i]*cor_pred['SP500_STOXX'].iloc[i]],
          [st_dev_pred['BOVESPA'].iloc[i]*st_dev_pred['HSI'].iloc[i]*cor_pred['BOVESPA_HSI'].iloc[i],
           st_dev_pred['SP500'].iloc[i]*st_dev_pred['HSI'].iloc[i]*cor_pred['SP500_HSI'].iloc[i],
           st_dev_pred['HSI'].iloc[i]**2,
           st_dev_pred['NIFTY50'].iloc[i]*st_dev_pred['HSI'].iloc[i]*cor_pred['HSI_NIFTY50'].iloc[i],
           st_dev_pred['STOXX'].iloc[i]*st_dev_pred['HSI'].iloc[i]*cor_pred['HSI_STOXX'].iloc[i]],
          [st_dev_pred['BOVESPA'].iloc[i]*st_dev_pred['NIFTY50'].iloc[i]*cor_pred['BOVESPA_NIFTY50'].iloc[i],
           st_dev_pred['SP500'].iloc[i]*st_dev_pred['NIFTY50'].iloc[i]*cor_pred['SP500_NIFTY50'].iloc[i],
           st_dev_pred['NIFTY50'].iloc[i]*st_dev_pred['HSI'].iloc[i]*cor_pred['HSI_NIFTY50'].iloc[i],
           st_dev_pred['NIFTY50'].iloc[i]**2,
           st_dev_pred['NIFTY50'].iloc[i]*st_dev_pred['STOXX'].iloc[i]*cor_pred['NIFTY50_STOXX'].iloc[i]],
          [st_dev_pred['BOVESPA'].iloc[i]*st_dev_pred['STOXX'].iloc[i]*cor_pred['BOVESPA_STOXX'].iloc[i],
           st_dev_pred['SP500'].iloc[i]*st_dev_pred['STOXX'].iloc[i]*cor_pred['SP500_STOXX'].iloc[i],
           st_dev_pred['STOXX'].iloc[i]*st_dev_pred['HSI'].iloc[i]*cor_pred['HSI_STOXX'].iloc[i],
           st_dev_pred['NIFTY50'].iloc[i]*st_dev_pred['STOXX'].iloc[i]*cor_pred['NIFTY50_STOXX'].iloc[i],
           st_dev_pred['STOXX'].iloc[i]**2]])
    ML_w.iloc[i] = np.dot(np.linalg.inv(cov_mat_i),np.ones(5))/np.dot(np.dot(np.transpose(np.ones(5)),np.linalg.inv(cov_mat_i)),np.ones(5))


# In[247]:


minvol_returns_ML = [np.dot(ML_w.iloc[i].T,returns.iloc[i].values) for i in range(returns.shape[0])]
# minvol_returns_ML


# In[248]:


def calc_max_drawdown(return_series):
    comp_ret = pd.Series((return_series+1).cumprod())
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()

def strategy_analytics(returns_alloc, rf = (1.04**(1/252)-1)):
    returns_alloc = pd.DataFrame({'returns':returns_alloc})
#     print(returns_alloc)
    mean_return = returns_alloc['returns'].mean()*250
    cum_return = ((1+returns_alloc['returns']).prod())**(1/5)-1
#     print(cum_return)
    st_dev = returns_alloc['returns'].std()*(250**0.5)
    sharpe_ratio = (cum_return - rf)/st_dev
    sortino_ratio = (cum_return - rf)/(returns_alloc['returns'].loc[returns_alloc['returns']<0].std()*(250**0.5))
    max_drawd = - calc_max_drawdown(np.array(returns_alloc['returns']))
    calmar_ratio = cum_return/ max_drawd
    success_rate = sum(returns_alloc['returns']>0)/len(returns_alloc['returns'])
    average_up = returns_alloc['returns'].loc[returns_alloc['returns']>0].mean()
    average_down = returns_alloc['returns'].loc[returns_alloc['returns']<0].mean()
    return cum_return,st_dev,sharpe_ratio,sortino_ratio,max_drawd,calmar_ratio,success_rate,average_up,average_down


# In[250]:


portfolio_df = strategy_analytics(np.array(minvol_returns_ML))
# portfolio_df.index = ['Annual Return','Annual St Dev','Sharpe','Sortino','Max Drawdown','Calmar','Success Rate','Avg Up','Avg Dn']
portfolio_df
np.std(minvol_returns_ML)*252**0.5


# In[251]:


Min_Vol_Returns = pd.DataFrame({'Returns' : minvol_returns_ML}, index = cor_pred.index)
Min_Vol_Returns.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/MinVol ML Returns_Stocks.csv')


# In[252]:


cor_pred.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/ML Correlations.csv')


# In[253]:


def daily_monthly_rollup(daily_returns):
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    return monthly_returns

monthly_minvol_returns = daily_monthly_rollup(Min_Vol_Returns['Returns'])


# In[254]:


monthly_minvol_returns.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/MinVol ML Monthly Returns_Stocks.csv')


# In[255]:


ML_w.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/ML weights.csv')


# In[256]:


monthly_stocks_returns = daily_monthly_rollup(daily_returns)
monthly_stocks_returns.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/Monthly Stocks Returns.csv')


# In[ ]:




