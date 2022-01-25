# %% [markdown]
# # 시스템 손절매의 실례: 수익곡선 모멘텀 그림 2-38
# - 투자 기간 :1986년 7월~2017년 6월
# - 투자자산 : 코스피지수 월봉
# - 투자 전략 : 12개월 평균 모멘텀 스코어 전략(연 3% 수익 가정 현금 혼합 전략) + 6개월 평균 모멘텀 스코어 시스템 손절매 전략
#     - 12개월 평균 모멘텀 스코어 전략(현금 혼합)으로 전략이 마무리 되는 것이 아니라, 이 수익곡선 자체의 모멘텀 스코어를 매달 평가하여 기본 전략의 투자 비중을 매달 조절하는 전략입니다. 
#     - 예를 들어 이번 달의 수익곡선 모멘텀 스코어가 0.75이었다면, 투자 비중은 평균 모멘텀 스코어 전략 75%, 현금 25%가 됩니다. 이중 75%만 전략에 투자합니다.
# - 결과
#     - 코스피 : 7.2%/-73.1%
#     - 평균 모멘텀 스코어 현금 혼합 전략 : 6.4%/-19.1%
#     - 수익곡선 모멘텀 전략 : 6.4%/-10.2%
#     - 일종의 **시장 적응 전략(Market Adaptive Strategy)**
# 

# %%
from settings import *

# %%
# 직접생성
kospi = fdr.DataReader("KS11")['Close']
# df = pd.DataFrame()
# df['KOSPI'] = kospi.copy()
# df = df['1985-1':'2017-6'].resample('MS').first() # 실제 말 마지막날이 공휴일이어도 마지막날 거래 한것과 같이 된다.
# df['현금'] = pow(1.03, 1/12)
# df['현금'] = df['현금'].shift(1, fill_value=1.0)
# df['현금'] = df['현금'].cumprod()
# df

# %%
kospi.resample('M').last()['1985-1':]

# %%
df = pd.read_csv('kospi_m.csv', index_col=0, parse_dates=True)
#-------------------------------------------
df['현금'] = pow(1.03, 1/12)
df['현금'] = df['현금'].shift(1, fill_value=1.0)
df['현금'] = df['현금'].cumprod()
#-------------------------------------------
df.head()

# %%
df.rebase(1).plot(figsize=(12,8));

# %%
data = df[:'2017-06'].copy()

# %%
bt_ew = long_only_ew(data, ['KOSPI', '현금'], "ew")

# %%
def 평균모멘텀(데이터):
    초기값 = 0
    for i in range(1, 13):
        초기값 = 데이터 / 데이터.shift(i) + 초기값
    return 초기값 / 12

def 모멘텀순위(데이터, 순위):
    x = 평균모멘텀(데이터)
    y = x.iloc[ : , 0: len(x.columns)].rank(1, ascending=0)
    y[y <= 순위] = 1
    y[y > 순위] = 0
    return y

def 평균모멘텀스코어(데이터):
    a = 평균모멘텀(데이터).copy()
    초기값 = 0
    for i in range(1, 13):
        초기값 = np.where(데이터 / 데이터.shift(i) > 1, 1, 0) + 초기값
    a[a > -1] = 초기값/12
    return a

# %%
score = 평균모멘텀스코어(data)

# %%
# KOSPI만 평균모멘텀스코어 비중 만큼 투자하고 나머지는 현금
weights0 = pd.DataFrame()
weights0['KOSPI'] = score['KOSPI']
weights0['현금']  = 1.0 - score['KOSPI']
weights0.head(20) # 1986-1 이후 유효

# %%
# 현금도 하나의 자산으로 취급. 이론 현금은 항상 모멘텀스코어가 1이다.
# 따라서 주식 비중이 0.5를 넘을 수 없다.
weights = score.copy()
weights = weights.div(weights.sum(axis=1), axis=0) # !!
weights.head(20)

# %%
weights.plot.area(figsize=(12,4));

# %%
weights0.plot.area(figsize=(12,4));

# %%
def momentumscore(data, weights, name="MomScore", start='1986-1-03'):
    st = bt.Strategy(
        name,
        [
            bt.algos.RunMonthly(run_on_last_date=True),
            bt.algos.RunAfterDate(start), #1년 뒤부터 시작 : 시작전 12개월 데이터 필요
            bt.algos.SelectAll(),
            bt.algos.WeighTarget(weights),
#             bt.algos.PrintInfo("{now} {temp}"), #{_price} {_universe} 
            bt.algos.Rebalance()
        ]
    )
    return bt.Backtest(st, data, initial_capital=100000000.0)

# %%
# shift(1)을 하면 전달 기준으로 모멘텀 계산
bt_ms0 = momentumscore(data, weights0.dropna(), name='모멘텀 포트폴리오(KOSPI)')#.shift(1))
bt_ms = momentumscore(data, weights.dropna(), name='모멘텀 포트폴리오(현금혼합)')#.shift(1))


# %%
bt_kospi = long_only_ew(data, ['KOSPI'], 'KOSPI')
bt_현금 = long_only_ew(data, ['현금'], '현금')

# %%
r_ms = bt.run(bt_ms)#, bt_kospi, bt_ew)

# %%
r_ms.set_date_range('1986-1-4')
r_ms.display()

# %%
r_ms.plot(figsize=(12,8));

# %%
r_all = bt.run(bt_ms, bt_ms0, bt_kospi, bt_ew)

# %%
r_all.set_date_range('1986-1-4')
r_all.display()

# %%
r_all.plot(figsize=(12,8));

# %%
r_현금 = bt.run(bt_현금)
r_kospi = bt.run(bt_kospi)
r_ms0 = bt.run(bt_ms0)

# %% [markdown]
# # 그림 2.26

# %%
ax1 = r_ms.plot(color='g', figsize=(12,8));
r_kospi.plot(ax=ax1, ls='--', color='b', figsize=(12, 8));
r_현금.plot(ax=ax1, ls='--', color='r', figsize=(12, 8));
#----
# r_ms0.plot(ax=ax1, ls='--', color='gray', figsize=(12, 8)); # add hosung
# XXX 아래 데이터 코스피 값 이상함...2012~ 2016횡보장이 1200대로 나

# %%
# qs.reports.full(r_all.prices['모멘텀 포트폴리오(현금혼합)'].to_returns().dropna()['1986':],
#                 r_all.prices['KOSPI'].to_returns().dropna()['1986':])


# %% [markdown]
# # 일일데이터

# %%
kospi = fdr.DataReader("KS11")['Close']
data_daily = pd.DataFrame()
data_daily['KOSPI'] = kospi.copy()
data_daily = data_daily['1985-1':'2020-9']
# %%
dd = pd.DataFrame()
# dd.index = pd.date_range('1985', '2020-9-30', freq='D')
dd.index = pd.date_range('1985', '2020-12-31', freq='D')
denominator = len(pd.date_range('1985', '2020-12-31', freq='D'))/len(pd.date_range('1985', '2020-12-31', freq='Y'))
dd['현금'] = pow(1.03, 1/denominator)
dd['현금'] = dd['현금'].shift(1, fill_value=1.0)
dd['현금'] = dd['현금'].cumprod()
#%%
data_daily = pd.merge(data_daily, dd, left_index=True, right_index=True, how='left')

# %%
data_daily

# %%
# 월말에 계산하고 다음월 첫 거래일 종가로 거래 한다고 전재
# 12개월 평균모멘텀스코어 계산

class WeighAMSWithCash(bt.Algo):
    def __init__(self, lags, months, cash):
        super(WeighAMSWithCash, self).__init__()
        self.lags = lags
        self.months = months
        self.cash = cash

    def 평균모멘텀스코어(self, 데이터, 개월):
        초기값 = 0
        for i in range(1, 개월+1):
            초기값 += np.where(데이터 / 데이터.shift(i) > 1, 1, 0)
        return 초기값[-1]/개월

    def AMS(self, prc):
        ''' x : Series (DataFrame의 컬럼)
            x[-1] : 기준일. x의 현재값
            (오늘날짜/과거날짜 - 1) > 0 보다 크면 1, 아니면 0
            => 오늘날짜/과거날짜 > 1 => 오늘날짜 > 과거날짜  => x[-1] > x
        '''
        average_momentum_score = pd.Series(dtype='float64')
        # print(f"{list(np.where(x[-1]>x, 1, 0)[:-1])}, {len(np.where(x[-1]>x, 1, 0)[:-1])}")
        for c in prc.columns:
            average_momentum_score[c] = np.mean(np.where(prc[c][-1]>prc[c], 1, 0)[:-1])# 당일 날짜 비교는 제외해준다 [:-1]

        return average_momentum_score

    def __call__(self, target):
        selected = target.temp['selected'].copy()

        t0 = (target.now - pd.DateOffset(months=self.lags)).strftime("%Y-%m")
        start = (target.now - pd.DateOffset(months=self.lags) - pd.DateOffset(months=self.months)).strftime("%Y-%m")

        # t0 = (target.now).strftime("%Y-%m")
        # start = (target.now -  pd.DateOffset(years=1)).strftime("%Y-%m")

        # print(selected, t0)
#         print(f"\nprc : {target.now} {t0} ~ {start}")
        prc = target.universe[selected].loc[start:t0].resample('M').last()
        if (len(prc) < self.months+1):
            return False

        # weights = pd.Series(self.평균모멘텀스코어(prc, self.months), index=selected)
        weights = pd.Series(self.AMS(prc), index=selected)
        weights = weights/weights.sum()
        
        # print(self.평균모멘텀스코어(prc, self.months), prc)
        target.temp['weights'] = weights
#         print(f"{target.now} ", end =" ")
#         for i, v in weights.items():
#             print(f"{i}:{v:.3f}", end=" ")
#         print("")
        return True

# %%
#%%
def momentum_mixedcash(name, data, lags=0, months=12, cash='현금'):
    st = bt.Strategy(name,
        [
            bt.algos.RunMonthly(run_on_first_date=True,
                                run_on_end_of_period=False,
                                run_on_last_date=False),
            # bt.algos.PrintInfo('{name} : {now}'),
            bt.algos.SelectThese(['KOSPI', '현금']),
            WeighAMSWithCash(lags=lags, months=months, cash=cash),
            # bt.algos.PrintTempData(),
            bt.algos.Rebalance()
        ]
    )
    test = bt.Backtest(st, data, initial_capital=100000000)
    return test

# %%
bt_cash = momentum_mixedcash("모멘텀(현금혼합)", data)
r_cash = bt.run(bt_cash, bt_ms) #add hosung

# %%
# 처음과 동일함을 검증(로직)
r_cash.set_date_range('1986-1-4')
r_cash.display()

# %%
bt_daily = momentum_mixedcash("모멘텀(현금혼합,일간)", data_daily)

# %%
r_daily = bt.run(bt_daily)

# %%
r_daily.set_date_range('1986-1-4', '2017-06-01')
r_daily.display()

# %%
r_daily.prices['1986-1-4':'2017-6-1'].to_drawdown_series().describe() #add hosung

# %%
# add hosung 
bt_kospi1 = long_only_ew(data_daily, ['KOSPI'], 'KOSPI')
bt_현금1 = long_only_ew(data_daily, ['현금'], '현금')

r_kospi1 = bt.run(bt_kospi1)
r_현금1 = bt.run(bt_현금1)

r_kospi1.set_date_range('1986-1-4', '2017-06-01')
r_현금1.set_date_range('1986-1-4', '2017-06-01')

# %%
ax1 = r_daily.plot(color='g', figsize=(12,8));
r_kospi1.plot(ax=ax1, ls='--', color='b', figsize=(12, 8));
r_현금1.plot(ax=ax1, ls='--', color='r', figsize=(12, 8));

# %%

