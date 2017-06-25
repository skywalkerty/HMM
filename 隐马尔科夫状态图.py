# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:26:21 2017

@author: Administrator
"""
from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import scipy.stats as sts
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import seaborn as sns
sns.set_style('white')
'''
from WindPy import w 
w.stop()
w.start()
data_test=w.wsd("RB.SHF", "open,high,low,close,volume,pct_chg", "2000-02-14", "2016-03-01", "")
'''
#定义上穿函数
def CrossOver(price1,price2):
    Con=[False]
    Precon=False
    conter=0
    for i in range(1,len(price1)):
        if (price1[i]>price2[i]):
            conter=1
            Con1=price1[i-1]==price2[i-1]
            while (Con1==True and i-conter>0):
                conter=conter+1
                Con1=(price1[i-conter]==price2[i-conter])
            Precon=(price1[i-conter]<price2[i-conter])
            Con.append(Precon)
            conter=0
        else:
            Con.append(False)
            conter=0
    return Con
#定义下穿函数
def CrossUnder(price1,price2):
    Con=[False]
    Precon=False
    conter=0
    for i in range(1,len(price1)):
        if (price1[i]<price2[i]):
            conter=1
            Con1=price1[i-1]==price2[i-1]
            while (Con1==True and i-conter>0):
                conter=conter+1
                Con1=(price1[i-conter]==price2[i-conter])
            Precon=(price1[i-conter]>price2[i-conter])
            Con.append(Precon)
            conter=0
        else:
            Con.append(False)
            conter=0
    return Con
#定义MACD函数
def EMA_MACO(data,d):
    test=pd.Series(index=range(len(data)))
    test=pd.ewma(data, span=d)
    return test
def MACD(data,FastLength,SlowLength,MACDLength):
    data['Diff']=''
    data['Diff']=EMA_MACO(data['open'],FastLength)-EMA_MACO(data['open'],SlowLength)
    data['DEA']=''
    data['DEA']=EMA_MACO(data['Diff'],MACDLength)
    data['MACD']=''
    data['MACD']=data['Diff']-data['DEA']
    return data
#导入数据，生成因子
data=pd.read_csv('rb888_2015.csv',parse_dates=True,index_col='time')
data.reset_index(inplace=True)
data['log_return']=np.log(data['open']/data['open'].shift(1))
data['log_return']=data['log_return'].fillna(0)
data['log_return_5']=np.log(data['open']/data['open'].shift(5))
data['log_return_5']=data['log_return_5'].fillna(0)
for h,k in [(5,10),(5,15),(5,20),(10,15),(10,20),(15,20),(15,30)]:
    data['fast_line']=''
    data['slow_line']=''
    data['fast_line']=pd.rolling_mean(data['open'],h)
    data['slow_line']=pd.rolling_mean(data['open'],k)
    data['fast_line']=data['fast_line'].fillna(value=pd.expanding_mean(data['open']))
    data['slow_line']=data['slow_line'].fillna(value=pd.expanding_mean(data['open']))
    data['dist_%s_%s'%(k,h)]=data['fast_line']-data['slow_line']
for i in range(5,31,5):
    data['MA_%s'%i]=pd.rolling_mean(data['open'],i)
    data['MA_%s'%i]=data['MA_%s'%i].fillna(0)-data['open']
data=MACD(data,12,26,9)
for h in range(10,26,5):
    data['fast_line']=''
    data['slow_line']=''
    data['fast_line']=pd.rolling_max(data['high'].shift(1),h)
    data['slow_line']=pd.rolling_min(data['low'].shift(1),h)
    data['fast_line']=data['fast_line'].fillna(value=pd.expanding_max(data['high']))
    data['slow_line']=data['slow_line'].fillna(value=pd.expanding_min(data['low']))
    data['dist_high_%s'%h]=data['high']-data['fast_line']
    data['dist_low_%s'%h]=data['low']-data['slow_line']
#引入隐马尔科夫模型
factor_list=['close','volume','dist_10_5','dist_15_5','dist_20_5','dist_15_10','dist_20_10','dist_20_15','dist_30_15','log_return','log_return_5','MACD','dist_high_10','dist_high_15','dist_high_20','dist_high_25','dist_low_10','dist_low_15','dist_low_20','dist_low_25','MA_5','MA_10','MA_15','MA_20','MA_25','MA_30']
for i in factor_list:
    X = np.column_stack([data[i]])
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000,random_state=0).fit(X)
    hidden_states = model.predict(X)
    plt.figure(figsize=(15, 8))  
    for k in range(model.n_components):
        idx = (hidden_states==k)
        plt.plot_date(data['time'][idx],data['close'][idx],'.',label='%dth hidden state'%k,lw=1)
        plt.legend()
        plt.grid(1)
    plt.savefig('C:/Users/Public/Documents/Python Scripts/隐马尔科夫状态刻画图集开盘价（2015）/%s.png'%(i))