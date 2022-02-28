import pandas as pd
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
"""
Step 1
Download data from tushare
"""
# Tushare.Pro Token
pro = ts.pro_api('00d803b166f55fc30c178d74c158985136010d6bd19271b182059eef')
# Reserch time period start at 2016
start_date = '20160101'
end_date = '20220224'
# 农业银行
stock_code_H = ['01288.HK']
stock_code_A = ['601288.SH']

# download data from Tushare.pro
df_H = pro.hk_daily(ts_code=stock_code_H[0], start_date=start_date, end_date=end_date)
df_A = pro.daily(ts_code=stock_code_A[0], start_date=start_date, end_date=end_date)

"""
Step 2
Calculate discount rate
"""
# Features 
feature_list = ['trade_date','open','close']

# Innner connect two DataFrame, x for A and y for HK
df_AH = pd.merge(df_A, df_H, how='inner')

# calculate discount rate: DR
df_AH = pd.merge(df_A[feature_list], df_H[feature_list], how='inner',left_on='trade_date', right_on='trade_date')
plt.plot(df_AH['close_x'].dropna(),linewidth=1,label='A')
plt.plot(df_AH['close_y'].dropna(),linewidth=1,label='HK')
plt.legend()
plt.savefig('data/股价对比时序图.png',dpi=600, bbox_inches='tight')
plt.show()

# df_AH['DR'] = (df_AH['close_y'] - df_AH['close_x'])/df_AH['close_x']
df_AH['DR'] =(df_AH['close_y'])/df_AH['close_x']


"""
Step 3
Analyze discount rate, including mean, std, variance
"""
rolling_window_size = 180
df_AH['DR_mean'] = df_AH['DR'].rolling(180).mean()
df_AH['DR_std'] = df_AH['DR'].rolling(180).std()
df_AH['DR_ub'] = df_AH['DR_mean']+df_AH['DR_std'] 
df_AH['DR_lb'] = df_AH['DR_mean']-df_AH['DR_std'] 
# plt.plot(df_AH[['DR','DR_mean']], color=['blue' ,'orange'], label=['DiscountRate' ,'Mean'])
plt.plot(df_AH.dropna()['DR'], color='steelblue', label='DiscountRate',linewidth=0.8)
plt.plot(df_AH.dropna()['DR_mean'], color='orange', label='Mean')
plt.plot(df_AH.dropna()['DR_ub'], color='orange',linestyle='--', label='ub')
plt.plot(df_AH.dropna()['DR_lb'], color='orange',linestyle='--', label='lb')
plt.legend()
plt.title('Discount Rate = HK.close/ A.close')
plt.savefig('data/折价率时序图.png',dpi=600, bbox_inches='tight')
plt.show()

"""
Step 4
Design trading rules
"""
def trading_rule(df):
    # extract data
    df = df.dropna(how='any')
    close = df['close_y'].values
    open = df['close_x'].values
    ub = df['DR_ub'].values
    lb = df['DR_lb'].values
    DR = df['DR'].values
    n = len(df)
    # initialize state
    value = np.zeros(n)
    buyPoint = np.zeros(n)
    sellPoint = np.zeros(n)
    psntValue = 100
    psntVolm = psntValue/open[0]
    isAllIn = True
    isSell = False
    isBuy = False
    for t in range(n):
        # cal present value
        if isAllIn:
            psntValue = psntVolm * open[t]
        value[t] = psntValue
        # no operation
        if not (isBuy or isSell):
            if DR[t] < lb[t] and not isAllIn:
                isBuy = True
                # all in
            elif DR[t] > ub[t] and isAllIn:
                # all out
                isSell = True
        # operation
        elif isBuy:
            buyPoint[t] = 1
            isBuy = False
            psntVolm = psntValue/open[t]
            isAllIn = True
        else:# sell
            sellPoint[t] = 1
            isSell = False
            isAllIn = False
    return value, psntValue, buyPoint, sellPoint
"""
Step 5
Evaluatioin
"""
value, psntValue, buyPoint, sellPoint = trading_rule(df_AH)

plt.plot(value, label='Value',linewidth=1)
plt.scatter(np.arange(len(value))[buyPoint == 1],value[buyPoint == 1],color='red',label='Buy',s=20)
plt.scatter(np.arange(len(value))[sellPoint == 1],value[sellPoint == 1],color='green',label='Sell',s=20)
plt.legend()
plt.title('Final Value: {:.2f}'.format(psntValue))
plt.savefig('data/资产价值时序图.png',dpi=600, bbox_inches='tight')
plt.show()

