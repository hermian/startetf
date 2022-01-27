# %% [markdown]
# # 고정비율 투자법 그림 3-10
# 
# 1. 마음에 드는 우량주 선택(예: 삼성전자)
# 2. 선정한 종목 매수(단, 투자 자금의 10%만 매수, 나머지 90%는 단기 국고채 ETF에 투자)
# 3. 매달 마지막 거래일에 주식:국고채 비중 = 1:9 로 리밸런싱
# 4. 
# 그림 3-15 코스피지수 변동성 조절 포트폴리오
# 

# %%
from settings import *

# %%
sec = fdr.DataReader("005930")[['Close']]

# %%
sec

# %%
sec['1997-12']

# %%
sec['2021-12'].tail()

# %%
s = '1997-12-27'
e = '2021-12-30'
data = pd.DataFrame()
data['SEC'] = sec.copy()
data = data[s:e]
# %%
dd = pd.DataFrame()
dd.index = pd.date_range(s, e, freq='D')
denominator = len(pd.date_range(s, e, freq='D'))/len(pd.date_range(s, e, freq='Y'))
dd['현금'] = pow(1.03, 1/denominator)
dd['현금'] = dd['현금'].shift(1, fill_value=1.0)
dd['현금'] = dd['현금'].cumprod()
#%%
data = pd.merge(data, dd, left_index=True, right_index=True, how='left')

# %%
data

# %%
def strategy(name, data, stock_w, cash_w):
    s = bt.Strategy(name, 
            [
                bt.algos.RunMonthly(run_on_end_of_period=True),
                bt.algos.SelectAll(),
                bt.algos.WeighSpecified(SEC=stock_w, 현금=cash_w),
                # bt.algos.PrintTempData(),
                bt.algos.Rebalance(),
                # bt.algos.PrintInfo('{now} {name} {_price} {temp} \n{_universe}')
            ])

    return bt.Backtest(s, data, initial_capital=100000000.0)


# %%
t1 = strategy("t1",   data, 0.1, 0.9) 

# %%
r1 = bt.run(t1)

# %%
r1.display()

# %%
bt_sec = long_only_ew(data, ['SEC'], "SEC")

# %%
r_all = bt.run(bt_sec, t1)

# %%
r_all.display()

# %%
plot_assets(r_all, s, e, "t1")

# %%
