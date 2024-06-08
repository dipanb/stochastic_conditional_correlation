#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# In[2]:


tickers = ['SPY','IJR','EEM','IEF','LQD','EMB','DX-Y.NYB','GLD','VNQ','^SPGSCI']
start = datetime.datetime(2008,1,1)
end = datetime.datetime(2023,12,31)
price_data = yf.download(tickers, start=start, end=end)['Adj Close']
price_data = price_data.rename(columns={'^SPGSCI': 'SPGSCI'})



# In[3]:


def convert_prices_to_returns(prices):
    returns = prices.pct_change(1)
    returns.iloc[0,:] = 0
    return returns

def daily_weekly_rollup(daily_returns):
    weekly_returns = (1 + daily_returns).resample('W').prod() - 1
    return weekly_returns

daily_returns = convert_prices_to_returns(price_data)
weekly_returns = daily_weekly_rollup(daily_returns)


# In[4]:


def garch_loss(theta, tr, p=1, q=1):
    r = np.array(tr[::-1])
    
    w, alpha, gamma, beta = theta[0], theta[1:1 + p], theta[1 + p:1 + p + p], theta[1 + p + p:]
#     print(w, alpha, gamma, beta)
    r = np.array(r)
    T = len(r) - 1
    s_int = np.std(r)
    L = max(p, q)
    s = [s_int for i in range(0, L)]
    for t in range(L, T + 1):
        r_temp = r[T - t + 1:T - t + 1 + q]  # [rt-1,...,rt-q]
        s_temp = s[0:p]  # [st-1,...st-p]
        var = np.array(s_temp) ** 2
        r_squared = np.array(r_temp) ** 2
        gjr = r_squared*(np.array(r_temp)<0)
        st = np.sqrt(np.abs(np.dot(np.array(beta), var) + np.dot(np.array(alpha), r_squared)+ np.dot(np.array(gamma), gjr) + w))
        s = [st] + s
        
    s = np.array(s)
    loss = 0.0
    for i in range(len(r)):
        loss += np.log(s[i] ** 2) + (r[i]/s[i])**2
    return loss

def garch_fit(r, p=1, q=1, max_itr=3, g_type = "Normal", early_stopping=True):
    
    theta0 = [0.1] + [0.15 for i in range(p)] + [0.1 for i in range(p)] + [0.1 for i in range(q)]
    def ub(x):
        return 1. - x[1] - 0.5*x[2] - x[3]
    def lb1(x):
        return x[1] + x[2]
    def lb2(x):
        return x[0]
    def lb3(x):
        return x[1]
    def lb4(x):
        return x[3]
    def lb5(x):
        return x[2]
    
    constraints_normal = [{'type':'ineq', 'fun':ub},{'type':'ineq', 'fun':lb1},
                        {'type':'ineq', 'fun':lb2},{'type':'ineq', 'fun':lb3},
                        {'type':'ineq', 'fun':lb4}, {'type':'eq', 'fun':lb5}]
    
    constraints_gjr = [{'type':'ineq', 'fun':ub},{'type':'ineq', 'fun':lb1},
                        {'type':'ineq', 'fun':lb2},{'type':'ineq', 'fun':lb3},
                        {'type':'ineq', 'fun':lb4}]
    
    tr_losses = []
    j = 0
    count = 0
    while j < max_itr:
        j += 1
        if g_type == 'gjr':
            res = minimize(garch_loss, theta0, args =r, method='trust-constr', 
                           options={'disp': True}, constraints= constraints_gjr)
        else:
            res = minimize(garch_loss, theta0, args =r, method='trust-constr', 
                           options={'disp': True}, constraints= constraints_normal)
        theta = res.x

        tr_loss = garch_loss(theta, r)
        tr_losses.append(tr_loss)
        print("Iteration: %d. Training loss: %.3E." % (j, tr_loss))

        if early_stopping is True:
            if j > 10:
                if abs(tr_losses[-1] - tr_losses[-2]) / tr_losses[-2] < 0.0001:
                    count += 1
                    if count >= 2:
                        print("Early Stopping...")
                        return theta, tr_losses
    return theta, tr_losses


# In[5]:


SPY_GARCH_params = garch_fit(daily_returns['SPY'], max_itr = 1, g_type = 'normal')[0]
SPGSCI_GARCH_params = garch_fit(daily_returns['SPGSCI'], max_itr = 1, g_type = 'normal')[0]

IJR_GARCH_params = garch_fit(daily_returns['IJR'], max_itr = 1, g_type = 'normal')[0]
EEM_GARCH_params = garch_fit(daily_returns['EEM'], max_itr = 1, g_type = 'normal')[0]
IEF_GARCH_params = garch_fit(daily_returns['IEF'], max_itr = 1, g_type = 'normal')[0]
LQD_GARCH_params = garch_fit(daily_returns['LQD'], max_itr = 1, g_type = 'normal')[0]
EMB_GARCH_params = garch_fit(daily_returns['EMB'], max_itr = 1, g_type = 'normal')[0]
DXY_GARCH_params = garch_fit(daily_returns['DX-Y.NYB'], max_itr = 1, g_type = 'normal')[0]
GLD_GARCH_params = garch_fit(daily_returns['GLD'], max_itr = 1, g_type = 'normal')[0]
VNQ_GARCH_params = garch_fit(daily_returns['VNQ'], max_itr = 1, g_type = 'normal')[0]


# In[6]:


def garch_process(r, theta, p=1, q=1):  # test data: [rT,...r0]
    r = np.array(r[::-1])
    
    w, alpha, gamma, beta = theta[0], theta[1:1 + p], theta[1 + p:1 + p + p], theta[1 + p + p:]
    T = len(r) - 1
    s_int = np.std(r)
    L = max(p, q)
    s = [s_int for i in range(0, L)]
    for t in range(L, T + 1):
        r_temp = r[T - t + 1:T - t + 1 + q]  # [rt-1,...,rt-q]
        s_temp = s[0:p]  # [st-1,...st-p]
        var = np.array(s_temp) ** 2
        r_squared = np.array(r_temp) ** 2
        gjr = r_squared*(np.array(r_temp)<0)
        st = np.sqrt(np.abs(np.dot(np.array(beta), var) + np.dot(np.array(alpha), r_squared)+ np.dot(np.array(gamma), gjr) + w))
        s = [st] + s
    return np.array(s)


# In[7]:


SPY_s = garch_process(daily_returns['SPY'], SPY_GARCH_params)
SPGSCI_s = garch_process(daily_returns['SPGSCI'], SPGSCI_GARCH_params)

IJR_s = garch_process(daily_returns['IJR'], IJR_GARCH_params)
EEM_s = garch_process(daily_returns['EEM'], EEM_GARCH_params)
IEF_s = garch_process(daily_returns['IEF'], IEF_GARCH_params)
LQD_s = garch_process(daily_returns['LQD'], LQD_GARCH_params)
EMB_s = garch_process(daily_returns['EMB'], EMB_GARCH_params)
DXY_s = garch_process(daily_returns['DX-Y.NYB'], DXY_GARCH_params)
GLD_s = garch_process(daily_returns['GLD'], GLD_GARCH_params)
VNQ_s = garch_process(daily_returns['VNQ'], VNQ_GARCH_params)

SPY_ep = (daily_returns['SPY'])[::-1]/SPY_s
SPGSCI_ep = (daily_returns['SPGSCI'])[::-1]/SPGSCI_s

IJR_ep = (daily_returns['IJR'])[::-1]/SPY_s
EEM_ep = (daily_returns['EEM'])[::-1]/EEM_s
IEF_ep = (daily_returns['IEF'])[::-1]/IEF_s
LQD_ep = (daily_returns['LQD'])[::-1]/LQD_s
EMB_ep = (daily_returns['EMB'])[::-1]/EMB_s
DXY_ep = (daily_returns['DX-Y.NYB'])[::-1]/DXY_s
GLD_ep = (daily_returns['GLD'])[::-1]/GLD_s
VNQ_ep = (daily_returns['VNQ'])[::-1]/VNQ_s

epsilon = np.array([SPY_ep,SPGSCI_ep,IJR_ep,EEM_ep,IEF_ep,LQD_ep,EMB_ep,DXY_ep,GLD_ep,VNQ_ep])


# In[21]:


ep_df = pd.DataFrame(epsilon.T, columns=['SPY','IJR','EEM','IEF','LQD','EMB','DXY','GLD','VNQ','SPGSCI'],
                    index = daily_returns[::-1].index)

s = np.array([SPY_s,SPGSCI_s,IJR_s,EEM_s,IEF_s,LQD_s,EMB_s,DXY_s,GLD_s,VNQ_s])
s_df = pd.DataFrame(s.T, columns=['SPY','IJR','EEM','IEF','LQD','EMB','DXY','GLD','VNQ','SPGSCI'],
                   index = daily_returns[::-1].index)
ep_df.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/epsilon_GARCH.csv')
s_df.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/stdev_GARCH.csv')


# In[56]:


def DCC_loss(ab, r):
    T = r.shape[1]
    n = r.shape[0]
    sum_Q = np.zeros([n,n])
    for i in range(T):
        sum_Q += np.outer(r[:,i],r[:,i])
    
    Q_int = sum_Q/T
    Q_list = [Q_int]
    T = r.shape[1] - 1
    a = ab[0]
    b = ab[1]
    for i in range(T):
        et_1 = r[:,T-i]
        Qt_1 = Q_list[0]
        Qt = (1.0-a-b)*Q_int + a*np.outer(et_1,et_1) + b*Qt_1
        Q_list = [Qt] + Q_list
    
    Q = Q_list
    R_list = []
    n = Q[0].shape[0]
    for i in Q:
        temp = 1.0/np.sqrt(np.abs(i))
        temp = temp * np.eye(n)
        R = np.dot(np.dot(temp,i),temp)
        R_list = R_list + [R]
    
    R = R_list
    loss = 0.0
    for i in range(len(R)):
        Ri = R[i]
        Ri_ = np.linalg.inv(Ri)
        ei = r[:,i]
        loss += np.log(np.linalg.det(Ri)) + np.dot(np.dot(ei,Ri_),ei)
        # print('training loss %f' % loss)
    return loss

def DCC_fit(r, max_itr = 1):
    ab0 = np.array([0.3, 0.6])
    
    def ub(x):
        return 1. - x[0] - x[1]
    def lb1(x):
        return x[0]
    def lb2(x):
        return x[1]
    constraints = [{'type':'ineq', 'fun':ub},{'type':'ineq', 'fun':lb1},{'type':'ineq', 'fun':lb2}]
    
    # Optimize using scipy and save theta
    tr_losses = []
    j = 0
    count = 0
    while j < max_itr:
        j += 1
        
        res = minimize(DCC_loss, ab0, args = r, method='SLSQP',
                        options={'disp': True},constraints=constraints)
        ab = res.x
        tr_loss = DCC_loss(ab, r)
        tr_losses.append(tr_loss)
        print("Iteration: %d. Training loss: %.3E." % (j, tr_loss))

    return ab, tr_losses


# In[57]:


ab = DCC_fit(epsilon)[0]
ab


# In[58]:


def DCC_process(ab, r):
    T = r.shape[1]
    n = r.shape[0]
    sum_Q = np.zeros([n,n])
    for i in range(T):
        sum_Q += np.outer(r[:,i],r[:,i])
    
    Q_int = sum_Q/T
    Q_list = [Q_int]
    
    T = r.shape[1] - 1
    a = ab[0]
    b = ab[1]
    for i in range(T):
        et_1 = r[:,T-i]
        Qt_1 = Q_list[0]
        Qt = (1.0-a-b)*Q_int + a*np.outer(et_1,et_1) + b*Qt_1
        Q_list = [Qt] + Q_list
    
    Q = Q_list
    R_list = []
    n = Q[0].shape[0]
    for i in Q:
        temp = 1.0/np.sqrt(np.abs(i))
        temp = temp * np.eye(n)
        R = np.dot(np.dot(temp,i),temp)
        R_list = R_list + [R]
        
    return R_list


# In[96]:


ab = [0.02,0.9]
DCC_matr = DCC_process(ab, epsilon)
DCC_matr = np.array(DCC_matr)
DCC_matr = DCC_matr.reshape(DCC_matr.shape[0], -1)


# In[97]:


cols = ['SPY','IJR','EEM','IEF','LQD','EMB','DXY','GLD','VNQ','SPGSCI']
cor_col = []
for i in cols:
    for j in cols:
        cor_col.append(i + " & " + j)

DCC_corr = pd.DataFrame(DCC_matr, columns = cor_col, index = daily_returns[::-1].index)


# In[98]:


plt.figure(figsize=(10, 6))
plt.plot(DCC_corr.index, DCC_corr['SPY & EEM'], label='SP500 & EM Equity')
plt.plot(DCC_corr.index, DCC_corr['SPY & LQD'], label='SP500 & IG Bonds')
plt.plot(DCC_corr.index, DCC_corr['SPY & SPGSCI'], label='SP500 & Commodity')
plt.plot(DCC_corr.index, DCC_corr['SPY & GLD'], label='SP500 & Gold')
plt.plot(DCC_corr.index, DCC_corr['SPY & VNQ'], label='SP500 & Real Estate')


plt.title('DCC Correlations')
plt.xlabel('Date')
plt.ylabel('6 month rolling correlation')
plt.ylim(-2, 2)
plt.legend()


# In[95]:


DCC_corr.to_csv('C:/Users/dipan/OneDrive/Desktop/LBS Courses/Business Project/DCC_correlations.csv')


# In[ ]:




