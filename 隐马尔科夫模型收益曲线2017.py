# -*- coding: utf-8 -*-
"""
@author: ty
"""
from hmmlearn.hmm import GaussianHMM
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import scipy.stats as sts
import matplotlib.dates as dates
import seaborn as sns
from datetime import datetime
start=datetime.now()
PosSizeL=1
PosSizeS=1
data1=pd.read_csv('rb888_2015.csv',parse_dates=True,index_col='time')
data1.reset_index(inplace=True)
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
for h,k in [(5,20),(15,20),(5,10),(5,15),(10,15)]:
    data1['fast_line']=''
    data1['slow_line']=''
    data1['fast_line']=pd.rolling_mean(data1['close'],h)
    data1['slow_line']=pd.rolling_mean(data1['close'],k)
    data1['fast_line']=data1['fast_line'].fillna(value=pd.expanding_mean(data1['close']))
    data1['slow_line']=data1['slow_line'].fillna(value=pd.expanding_mean(data1['close']))
    data1['dist_%s_%s'%(k,h)]=data1['fast_line']-data1['slow_line']
for h in range(10,26,5):
    data1['fast_line']=''
    data1['slow_line']=''
    data1['fast_line']=pd.rolling_max(data1['high'].shift(1),h)
    data1['slow_line']=pd.rolling_min(data1['low'].shift(1),h)
    data1['fast_line']=data1['fast_line'].fillna(value=pd.expanding_max(data1['high']))
    data1['slow_line']=data1['slow_line'].fillna(value=pd.expanding_min(data1['low']))
    data1['dist_high_%s'%h]=data1['high']-data1['fast_line']
    data1['dist_low_%s'%h]=data1['low']-data1['slow_line']
data1=MACD(data1,12,26,9)
data2=pd.read_csv('rb888_2017.csv',parse_dates=True,index_col='time')
data2.reset_index(inplace=True)
data2['log_return']=np.log(data2['close']/data2['close'].shift(1))
data2['log_return']=data2['log_return'].fillna(0)
for h,k in [(5,20),(15,20),(5,10),(5,15),(10,15)]:
    data2['fast_line']=''
    data2['slow_line']=''
    data2['fast_line']=pd.rolling_mean(data2['close'],h)
    data2['slow_line']=pd.rolling_mean(data2['close'],k)
    data2['fast_line']=data2['fast_line'].fillna(value=pd.expanding_mean(data2['close']))
    data2['slow_line']=data2['slow_line'].fillna(value=pd.expanding_mean(data2['close']))
    data2['dist_%s_%s'%(k,h)]=data2['fast_line']-data2['slow_line']
for h in range(10,26,5):
    data2['fast_line']=''
    data2['slow_line']=''
    data2['fast_line']=pd.rolling_max(data2['high'].shift(1),h)
    data2['slow_line']=pd.rolling_min(data2['low'].shift(1),h)
    data2['fast_line']=data2['fast_line'].fillna(value=pd.expanding_max(data2['high']))
    data2['slow_line']=data2['slow_line'].fillna(value=pd.expanding_min(data2['low']))
    data2['dist_high_%s'%h]=data2['high']-data2['fast_line']
    data2['dist_low_%s'%h]=data2['low']-data2['slow_line']
data2=MACD(data2,12,26,9)
factor_list=['dist_low_20','MACD','dist_20_15','dist_10_5','dist_15_5','dist_15_10','dist_20_5']
hidden_states=[]
forward=100
count=0
result=pd.DataFrame(columns=['因子','交易次数','累积净利','最大回撤','收益风险比','胜率'],index=range(100))
for i in factor_list:
    X = np.column_stack([data1[i]])
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000,random_state=0).fit(X)
    Y = np.column_stack([data2[i]])
    hidden_states=model.predict(Y)
    if i=='dist_low_20':
        hidden_states=np.array(hidden_states)
        signal=np.where(hidden_states==1,-1,np.where(hidden_states==2,1,0))
    elif i=='MACD':
        hidden_states=np.array(hidden_states)
        signal=np.where(hidden_states==1,-1,np.where(hidden_states==2,1,0))
    elif i=='dist_20_15':
        hidden_states=np.array(hidden_states)
        signal=np.where(hidden_states==1,-1,np.where(hidden_states==2,1,0))
    elif i=='dist_10_5':
        hidden_states=np.array(hidden_states)
        signal=np.where(hidden_states==1,-1,np.where(hidden_states==2,1,0))
    elif i=='dist_15_5':
        hidden_states=np.array(hidden_states)
        signal=np.where(hidden_states==1,-1,np.where(hidden_states==2,1,0))
    elif i=='dist_20_5':
        hidden_states=np.array(hidden_states)
        signal=np.where(hidden_states==1,-1,np.where(hidden_states==2,1,0))
    else:
        hidden_states=np.array(hidden_states)
        signal=np.where(hidden_states==2,-1,hidden_states)        
    signal=np.append(0,signal[:-1])
    data2['signal']=signal
    buy=[]
    sell=[]
    Type=[]
    Hand=[]
    SetTime=[]
    OpenPosition=[]
    CoverTime=[]
    NetProfit=[]
    CoverPosition=[]
    StaticRights=[]
    Rights=0
    BarRights=Rights
    DynamicRights=[]
    ProfitShares=0
    position=0
    for index in data2.index:
        if data2['signal'][index]==1 and position==0:
            ep=data2['open'][index]
            buy.append(-ep*PosSizeL*10)
            position=1
            Type.append('多头')
            Hand.append(PosSizeL)
            SetTime.append(data2['time'][index])
            OpenPosition.append(ep)
        if data2['signal'][index]==0 and position==1:
            ep=data2['open'][index]
            buy.append(ep*PosSizeL*10)
            position=0
            CoverTime.append(data2['time'][index])
            NetProfit.append(ep*10-OpenPosition[-1]*10)            
            CoverPosition.append(ep)   
            StaticRights.append(sum(NetProfit))
        if data2['signal'][index]==-1 and position==1:
            ep=data2['open'][index]
            CoverTime.append(data2['time'][index])
            NetProfit.append(ep*10-OpenPosition[-1]*10)            
            CoverPosition.append(ep)   
            StaticRights.append(sum(NetProfit)) 
            buy.append(ep*PosSizeL*10)
            sell.append(ep*PosSizeS)
            position=-1
            Type.append('空头')
            Hand.append(PosSizeS)
            SetTime.append(data2['time'][index])
            OpenPosition.append(ep) 
        if data2['signal'][index]==-1 and position==0:
            ep=data2['open'][index]
            sell.append(ep*PosSizeS)
            position=-1
            Type.append('空头')
            Hand.append(PosSizeS)
            SetTime.append(data2['time'][index])
            OpenPosition.append(ep) 
        if data2['signal'][index]==0 and position==-1:
           ep=data2['open'][index]
           sell.append(-ep*PosSizeS)
           position=0
           CoverTime.append(data2['time'][index])
           NetProfit.append((-ep+OpenPosition[-1])*10)            
           CoverPosition.append(ep)   
           StaticRights.append(sum(NetProfit))   
        if data2['signal'][index]==1 and position==-1:
           ep=data2['open'][index]
           sell.append(-ep*PosSizeS)
           buy.append(-ep*PosSizeL*10)
           position=0
           CoverTime.append(data2['time'][index])
           NetProfit.append((-ep+OpenPosition[-1])*10)            
           CoverPosition.append(ep)   
           StaticRights.append(sum(NetProfit)) 
           position=1
           Type.append('多头')
           Hand.append(PosSizeL)
           SetTime.append(data2['time'][index])
           OpenPosition.append(ep)
        if position==1:
            BarRights=Rights+sum(buy)+data2.close[index]*10+sum(sell)*10
            DynamicRights.append(BarRights)
        if position==0:
            DynamicRights.append(BarRights)
        if position==-1:
            BarRights=Rights+(sum(sell)-data2.close[index])*10+sum(buy)
            DynamicRights.append(BarRights)
    if position==1:
        buy.append(data2.close[index]*10)
        CoverTime.append(data2['time'][index])
        CoverPosition.append(data2.close[index])
        NetProfit.append((data2.close[index]-OpenPosition[-1])*10)    
        StaticRights.append(sum(NetProfit))    
    if position==-1:
        sell.append(-data2.close[index])
        CoverTime.append(data2['time'][index])
        CoverPosition.append(data2.close[index])
        NetProfit.append((-data2.close[index]+OpenPosition[-1])*10)    
        StaticRights.append(sum(NetProfit))
    trade_info=pd.DataFrame(index=range(1,len(OpenPosition)+1))
    trade_info['建仓时间']=SetTime
    trade_info['建仓价格']=OpenPosition
    trade_info['平仓时间']=CoverTime
    trade_info['平仓价格']=CoverPosition
    trade_info['数量']=PosSizeL
    trade_info['净利']=NetProfit
    trade_info['累计净利']=StaticRights
    trade_info['收益率']=trade_info['净利']/trade_info['建仓价格']
    trade_info['累积收益率']=trade_info['收益率'].cumsum()
    trade_info.to_csv('%s因子交易记录2017.csv'%i,index=False)
    capital_change=pd.DataFrame(index=data2['time'])
    capital_change['动态权益']=DynamicRights
    def max_drawdown(date_line,capital_line):
        df=pd.DataFrame({'date':date_line,'capital':capital_line})
        df.sort('date',inplace=True)
        df.reset_index(drop=True,inplace=True)
        df['max2here']=pd.expanding_max(df['capital'])
        df['dd2here']=df['max2here']-df['capital']
        temp=df.sort('dd2here',ascending=False).iloc[0][['date','dd2here']]
        max_dd=temp['dd2here']
        end_date=temp['date']
        df=df[df['date']<=end_date]
        start_date=df.sort('capital',ascending=False).iloc[0]['date']
        return max_dd#'最大回撤为：%f,开始日期：%s,结束日期：%s'%(max_dd,start_date, end_date)
    date_line=capital_change.index
    capital_line=capital_change['动态权益']
    #trade_info['累计净利'].plot()
    #capital_change['动态权益'].plot()
    for index in range(len(NetProfit)):
       if NetProfit[index]>0:
        ProfitShares=ProfitShares+1
    WinRate=ProfitShares/len(NetProfit)
    plt.figure(figsize=(15, 8))
    plt.plot(trade_info['平仓时间'],trade_info['累计净利'])
    plt.savefig('C:/Users/Public/Documents/Python Scripts/隐马尔科夫模型收益曲线2017/%s.png'%(i))
    end=datetime.now()
    period=(end-start).seconds
    profit=sum(NetProfit)
    day=(data2['time'][data2.index[-1]]-data2['time'][0]).days
    maxdrawdown=max_drawdown(date_line,capital_line)
    annual_return=trade_info['累计净利'].iloc[-1]/day*356
    ratio_of_return_and_risk=annual_return/maxdrawdown
    result['因子'][count]=i
    result['交易次数'][count]=len(SetTime)
    result['累积净利'][count]=profit
    result['最大回撤'][count]=maxdrawdown
    result['收益风险比'][count]=ratio_of_return_and_risk
    result['胜率'][count]=WinRate    
    count=count+1
result.to_csv('隐马尔科夫模型绩效回测表2017.csv',index=False)
    
