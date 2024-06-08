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


Pearson_std = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/Pearson_std.csv')
Pearson_std = Pearson_std.set_index('Date',drop= True).drop(Pearson_std.columns[0],axis=1)

Pearson_corr = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/Pearson_corr.csv')
Pearson_corr = Pearson_corr.set_index('Date',drop= True).drop(Pearson_corr.columns[0],axis=1)

DCC_std = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/DCC_std.csv')
DCC_std = DCC_std.set_index('Date',drop= True).drop(DCC_std.columns[0],axis=1)

DCC_corr = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/DCC_corr.csv')
DCC_corr = DCC_corr.set_index('Date',drop= True).drop(DCC_corr.columns[0],axis=1)

cGARCH_std = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/cGARCH_std.csv')
cGARCH_std = cGARCH_std.set_index('Date',drop= True).drop(cGARCH_std.columns[0],axis=1)

cGARCH_corr = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/cGARCH_corr.csv')
cGARCH_corr = cGARCH_corr.set_index('Date',drop= True).drop(cGARCH_corr.columns[0],axis=1)

ADCC_std = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/ADCC_std.csv')
ADCC_std = ADCC_std.set_index('Date',drop= True).drop(ADCC_std.columns[0],axis=1)

ADCC_corr = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/ADCC_corr.csv')
ADCC_corr = ADCC_corr.set_index('Date',drop= True).drop(ADCC_corr.columns[0],axis=1)

MSDCC_std = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/MSDCC_std.csv')
MSDCC_std = MSDCC_std.set_index('Date',drop= True)

MSDCC_corr = pd.read_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/MSDCC_corr.csv')
MSDCC_corr = MSDCC_corr.set_index('Date',drop= True)


# In[3]:


DCC_std = DCC_std.loc[Pearson_std.index]
DCC_corr = DCC_corr.loc[Pearson_std.index]
cGARCH_std = cGARCH_std.loc[Pearson_std.index]
cGARCH_corr = cGARCH_corr.loc[Pearson_std.index]
ADCC_std = ADCC_std.loc[Pearson_std.index]
ADCC_corr = ADCC_corr.loc[Pearson_std.index]
MSDCC_std = MSDCC_std.loc[Pearson_std.index]
MSDCC_corr = MSDCC_corr.loc[Pearson_std.index]
# print(Pearson_std.shape)
# print(Pearson_corr.shape)
# print(DCC_std.shape)
# print(DCC_corr.shape)
# print(cGARCH_std.shape)
# print(cGARCH_corr.shape)
# print(ADCC_std.shape)
# print(ADCC_corr.shape)
# print(MSDCC_std.shape)
# print(MSDCC_corr.shape)


# In[4]:


def min_var_wts(corr_row,st_dev_vec):

    corr_mat_i = np.array([[corr_row[j*10+i] for i in range(10)] for j in range(10)])
    std_i = np.array(st_dev_vec)
    cov_mat_i = np.dot(np.dot(np.diag(std_i),corr_mat_i),np.diag(std_i))
#     print(cov_mat_i)
    w_min = np.dot(np.linalg.inv(cov_mat_i),np.ones(10))/np.dot(np.dot(np.transpose(np.ones(10)),np.linalg.inv(cov_mat_i)),np.ones(10))
    return w_min


# In[5]:


DCC_w = pd.DataFrame(np.zeros_like(DCC_std))
DCC_w.columns = DCC_std.columns
DCC_w.index = DCC_std.index

Pearson_w = pd.DataFrame(np.zeros_like(DCC_std))
Pearson_w.columns = DCC_std.columns
Pearson_w.index = DCC_std.index

cGARCH_w = pd.DataFrame(np.zeros_like(DCC_std))
cGARCH_w.columns = DCC_std.columns
cGARCH_w.index = DCC_std.index

ADCC_w = pd.DataFrame(np.zeros_like(DCC_std))
ADCC_w.columns = DCC_std.columns
ADCC_w.index = DCC_std.index

MSDCC_w = pd.DataFrame(np.zeros_like(DCC_std))
MSDCC_w.columns = DCC_std.columns
MSDCC_w.index = DCC_std.index


# In[6]:


for i in range(DCC_w.shape[0]):
    print(i,"/",DCC_w.shape[0])
    Pearson_w.iloc[i] = min_var_wts(Pearson_corr.iloc[i],Pearson_std.iloc[i])
    DCC_w.iloc[i] = min_var_wts(DCC_corr.iloc[i],DCC_std.iloc[i])
    cGARCH_w.iloc[i] = min_var_wts(cGARCH_corr.iloc[i],cGARCH_std.iloc[i])
    ADCC_w.iloc[i] = min_var_wts(ADCC_corr.iloc[i],ADCC_std.iloc[i])
    MSDCC_w.iloc[i] = min_var_wts(MSDCC_corr.iloc[i],MSDCC_std.iloc[i])


# In[7]:


tickers = ['SPY','IJR','EEM','IEF','LQD','EMB','DX-Y.NYB','GLD','VNQ','^SPGSCI']
start = datetime.datetime(2008,6,24)
end = datetime.datetime(2024,1,3)
price_data = yf.download(tickers, start=start, end=end)['Adj Close']
price_data = price_data.rename(columns={'^SPGSCI': 'SPGSCI'})
price_data = price_data.rename(columns={'DX-Y.NYB': 'DXY'})

def convert_prices_to_returns(prices):
    returns = prices.pct_change(1)
    returns.iloc[0,:] = 0
    return returns

daily_returns = convert_prices_to_returns(price_data)
last_row = daily_returns.iloc[-1]
daily_returns = daily_returns.loc[Pearson_std.index]
daily_returns.loc[datetime.datetime(2024,1,2)] = last_row
daily_returns = daily_returns.drop(datetime.datetime(2008,6,24))


# In[8]:


np.where(np.isnan([np.dot(Pearson_w.iloc[i].T, daily_returns.iloc[i].values) for i in range(daily_returns.shape[0])]))


# In[9]:


minvol_returns = pd.DataFrame(np.zeros((daily_returns.shape[0], 5)), columns=['Pearson','DCC','cGARCH','ADCC','MSDCC'],
                             index = daily_returns.index)
minvol_returns['Pearson'] = [np.dot(Pearson_w.iloc[i].T, 
                             daily_returns.iloc[i].values) for i in range(daily_returns.shape[0])]
minvol_returns['DCC'] = [np.dot(DCC_w.iloc[i].T, 
                         daily_returns.iloc[i].values) for i in range(daily_returns.shape[0])]
minvol_returns['cGARCH'] = [np.dot(cGARCH_w.iloc[i].T, 
                            daily_returns.iloc[i].values) for i in range(daily_returns.shape[0])]
minvol_returns['ADCC'] = [np.dot(ADCC_w.iloc[i].T, 
                          daily_returns.iloc[i].values) for i in range(daily_returns.shape[0])] 
minvol_returns['MSDCC'] = [np.dot(MSDCC_w.iloc[i].T, 
                           daily_returns.iloc[i].values) for i in range(daily_returns.shape[0])]


# In[10]:


minvol_returns.std()*300**0.5


# In[12]:


mean_returns_year = minvol_returns.groupby(minvol_returns.index.year).agg({'Pearson': lambda x: (1+x).prod()-1,  
                                                                        'DCC': lambda x: (1+x).prod()-1,
                                                                        'cGARCH':lambda x: (1+x).prod()-1,
                                                                        'ADCC':lambda x: (1+x).prod()-1,
                                                                        'MSDCC':lambda x: (1+x).prod()-1
                                                                          })
mean_returns_year


# In[20]:


std_returns_year = minvol_returns.groupby(minvol_returns.index.year).agg({'Pearson': lambda x: x.std()*252**0.5,  
                                                                        'DCC': lambda x: x.std()*252**0.5,                                                                        'cGARCH':lambda x: x.std()*252**0.5,
                                                                        'ADCC':lambda x: x.std()*252**0.5,
                                                                        'MSDCC':lambda x: x.std()*252**0.5
                                                                          })
std_returns_year


# In[114]:


minvol_returns.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/MinVol Returns.csv')


# In[52]:


def calc_max_drawdown(return_series):
    comp_ret = pd.Series((return_series+1).cumprod())
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()

def strategy_analytics(returns_alloc, rf = (1.04**(1/12)-1)):
    returns_alloc = pd.DataFrame({'returns':returns_alloc})
#     print(returns_alloc)
    mean_return = returns_alloc['returns'].mean()*12
    cum_return = ((1+returns_alloc['returns']).prod())**(1/15)-1
#     print(cum_return)
    st_dev = returns_alloc['returns'].std()*(12**0.5)
    sharpe_ratio = (cum_return - rf)/st_dev
    sortino_ratio = (cum_return - rf)/(returns_alloc['returns'].loc[returns_alloc['returns']<0].std()*(12**0.5))
    max_drawd = - calc_max_drawdown(np.array(returns_alloc['returns']))
    calmar_ratio = cum_return/ max_drawd
    success_rate = sum(returns_alloc['returns']>0)/len(returns_alloc['returns'])
    average_up = returns_alloc['returns'].loc[returns_alloc['returns']>0].mean()
    average_down = returns_alloc['returns'].loc[returns_alloc['returns']<0].mean()
    return cum_return,st_dev,sharpe_ratio,sortino_ratio,max_drawd,calmar_ratio,success_rate,average_up,average_down


# In[53]:


portfolio_df = minvol_returns.apply(strategy_analytics, axis = 0)
portfolio_df.index = ['Annual Return','Annual St Dev','Sharpe','Sortino','Max Drawdown','Calmar','Success Rate','Avg Up','Avg Dn']
portfolio_df


# In[54]:


def convert_prices_to_returns(prices):
    returns = prices.pct_change(1)
    returns.iloc[0,:] = 0
    return returns

def daily_monthly_rollup(daily_returns):
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    return monthly_returns

minvol_nav = (1+minvol_returns).cumprod()
dr = convert_prices_to_returns(minvol_nav)
minvol_returns_monthly = daily_monthly_rollup(dr).iloc[:-1]


# In[55]:


portfolio_df = minvol_returns_monthly.apply(strategy_analytics, axis = 0)
portfolio_df.index = ['Annual Return','Annual St Dev','Sharpe','Sortino','Max Drawdown','Calmar','Success Rate','Avg Up','Avg Dn']
portfolio_df


# In[44]:





# In[ ]:




