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
kospi


# %% [markdown]
# ## 월별 비중 데이터프레임 (WeighTarget)
# 
# 월말리밸런싱을 가정하기 위해 월말로 된 데이터를 구성한다.

# %%

df = pd.DataFrame()
df['KOSPI'] = kospi.copy()
month_last = df.groupby(df.index.to_period('M')).apply(lambda x: x.index.max())
month_df = df.loc[month_last].copy()
# df = df['1985-1':'2017-6'].resample('MS').first() # 실제 말 마지막날이 공휴일이어도 마지막날 거래 한것과 같이 된다.
month_df['현금'] = pow(1.03, 1/12)
month_df['현금'] = month_df['현금'].shift(1, fill_value=1.0)
month_df['현금'] = month_df['현금'].cumprod()
month_df

# %%
# df = pd.read_csv('kospi_m.csv', index_col=0, parse_dates=True)
# #-------------------------------------------
# df['현금'] = pow(1.03, 1/12)
# df['현금'] = df['현금'].shift(1, fill_value=1.0)
# df['현금'] = df['현금'].cumprod()
# #-------------------------------------------
# df.head()

# %%
month_df.rebase(1).plot(figsize=(12,8));

# %%
data = month_df['1985-01':'2017-06'].copy()
data

# %%
s = '1985-1-30'
e = '2017-6-30'

# %%
bt_ew = long_only_ew(data, ['KOSPI', '현금'], "ew")

# %%
def 평균모멘텀(데이터, 개월=12):
    초기값 = 0
    for i in range(1, 개월+1):
        초기값 = 데이터 / 데이터.shift(i) + 초기값
    return 초기값 / 개월

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

def 평균모멘텀스코어6(데이터):
    a = 평균모멘텀(데이터, 6).copy() # bug 수익곡선은 6개월만 있으면 되는 12개월을 잡아먹음
    초기값 = 0
    for i in range(1, 7):
        초기값 = np.where(데이터 / 데이터.shift(i) > 1, 1, 0) + 초기값
    a[a > -1] = 초기값/6
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
            bt.algos.RunMonthly(run_on_end_of_period=True,run_on_last_date=True),
            bt.algos.RunAfterDate(start), #1년 뒤부터 시작 : 시작전 12개월 데이터 필요
            bt.algos.SelectAll(),
            bt.algos.WeighTarget(weights),
            bt.algos.PrintInfo("{now} {_price} {temp} "), #{_price} {_universe} 
            bt.algos.Rebalance()
        ]
    )
    return bt.Backtest(st, data, initial_capital=100000000.0)

# %%
# shift(1)을 하면 전달 기준으로 모멘텀 계산
bt_ms0 = momentumscore(data, weights0, name='모멘텀 포트폴리오(KOSPI)')#.shift(1))
bt_ms = momentumscore(data, weights, name='모멘텀 포트폴리오(현금혼합)')#.shift(1))


# %%
bt_kospi = long_only_ew(data, ['KOSPI'], 'KOSPI')
bt_현금 = long_only_ew(data, ['현금'], '현금')

# %%
r_ms = bt.run(bt_ms)

# %%
start평균모멘텀스코어 = '1986-1-31' # 최초 매수일

# %%
start수익곡선모멘텀 = '1986-7-31'

# %%
r_ms.set_date_range(start수익곡선모멘텀)
r_ms.display()

# %%
r_ms.plot(figsize=(12,8));

# %%
r_all = bt.run(bt_ms, bt_ms0, bt_kospi, bt_ew)

# %%
r_all.set_date_range(start수익곡선모멘텀)
r_all.display()

# %%
r_all.plot(figsize=(12,8));

# %%
r_현금 = bt.run(bt_현금)
r_kospi = bt.run(bt_kospi)
r_ms0 = bt.run(bt_ms0)

# %% [markdown]
# ### 그림 2.26

# %%
r_ms.set_date_range(start수익곡선모멘텀)
r_kospi.set_date_range(start수익곡선모멘텀)
r_현금.set_date_range(start수익곡선모멘텀)
ax1 = r_ms.plot(color='g', figsize=(12,8));
r_kospi.plot(ax=ax1, ls='--', color='b', figsize=(12, 8));
r_현금.plot(ax=ax1, ls='--', color='r', figsize=(12, 8));
#----
# r_ms0.plot(ax=ax1, ls='--', color='gray', figsize=(12, 8));

# %% [markdown]
# ### 수익곡선모멘텀(평균모멘텀스코어 6개월)

# %%
r_ms.set_date_range(start평균모멘텀스코어)
r_ms.prices

# %%
data1 = pd.DataFrame()
data1['평균모멘텀스코어현금혼합전략'] = r_ms.prices.rebase(1).copy() # rebase()로 현금과 맞춰준다. 디버깅을 쉽게하기 위해
data1['현금'] = pow(1.03, 1/12)
data1['현금'] = data1['현금'].shift(1, fill_value=1.0)
data1['현금'] = data1['현금'].cumprod()
data1

# %%
score1 = 평균모멘텀스코어6(data1)
score1.head(10)

# %%
weights1 = score1.copy()

# weights1 = weights1.div(weights1.sum(axis=1), axis=0) # !!

weights1['평균모멘텀스코어현금혼합전략'] = score1['평균모멘텀스코어현금혼합전략']
weights1['현금']  = 1.0 - score1['평균모멘텀스코어현금혼합전략']

weights1.dropna().head(10)

# %%
weights1.plot.area(figsize=(12,4));

# %%
bt_ms1 = momentumscore(data1, weights1, name='수익곡선모멘텀전략')#.shift(1))

# %%
r_ms1 = bt.run(bt_ms1)

# %%
r_ms1.set_date_range(start수익곡선모멘텀)
r_ms1.display()

# %% [markdown]
# ### 그림 2.38

# %%
r_ms.set_date_range(start수익곡선모멘텀)
r_kospi.set_date_range(start수익곡선모멘텀)
r_ms1.set_date_range(start수익곡선모멘텀)
ax1 = r_ms.plot(color='r', figsize=(12,8));
r_kospi.plot(ax=ax1, ls='--', color='g', figsize=(12, 8));
r_ms1.plot(ax=ax1, ls='-', color='b', figsize=(12, 8));
#----
# r_ms0.plot(ax=ax1, ls='--', color='gray', figsize=(12, 8));

# %%
# qs.reports.full(r_all.prices['모멘텀 포트폴리오(현금혼합)'].to_returns().dropna()['1986':],
#                 r_all.prices['KOSPI'].to_returns().dropna()['1986':])


# %% [markdown]
# ## 일일데이터 비중 클래스 구현 (일간->월간 변경 후 모멘텀스코어 구하는 법)

# %%
s, e, start평균모멘텀스코어, start수익곡선모멘텀

# %%
# kospi = fdr.DataReader("KS11")['Close']
data_daily = pd.DataFrame()
data_daily['KOSPI'] = kospi.copy()
data_daily = data_daily[s:e]
# %%
dd = pd.DataFrame()
# dd.index = pd.date_range('1985', '2020-9-30', freq='D')
dd.index = pd.date_range(s, e, freq='D')
denominator = len(pd.date_range(s, e, freq='D'))/len(pd.date_range(s, e, freq='Y'))
dd['현금'] = pow(1.03, 1/denominator)
dd['현금'] = dd['현금'].shift(1, fill_value=1.0)
dd['현금'] = dd['현금'].cumprod()
#%%
data_daily = pd.merge(data_daily, dd, left_index=True, right_index=True, how='left')

# %%
data_daily

# %% [markdown]
# ### 평균모멘텀스코어 WeighAMSWithCash

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
        print(f"\nprc : {target.now} {t0} ~ {start}")
        prc = target.universe[selected].loc[start:t0].resample('M').last()
        if (len(prc) < self.months+1):
            return False

        # weights = pd.Series(self.평균모멘텀스코어(prc, self.months), index=selected)
        weights = pd.Series(self.AMS(prc), index=selected)
        weights = weights/weights.sum()
        
        # print(self.평균모멘텀스코어(prc, self.months), prc)
        target.temp['weights'] = weights
        print(f"{target.now} ", end =" ")
        for i, v in weights.items():
            print(f"{i}:{v:.3f}", end=" ")
        print("")
        return True

# %%
#%%
def momentum_mixedcash(name, data, lags=0, months=12, cash='현금'):
    st = bt.Strategy(name,
        [
            bt.algos.RunMonthly(run_on_first_date=True,
                                run_on_end_of_period=True, #월말
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
bt_cash = momentum_mixedcash("모멘텀(현금혼합)", data, lags=0)
r_cash = bt.run(bt_cash, bt_ms)

# %%
# 비중데이터프레임의 결과와 동일함을 검증(로직)
r_cash.set_date_range(start수익곡선모멘텀)
r_cash.display()

# %%
bt_daily = momentum_mixedcash("모멘텀(현금혼합,일간)", data_daily, lags=0)

# %%
r_daily = bt.run(bt_daily)

# %%
s,e,start평균모멘텀스코어,start수익곡선모멘텀

# %%
r_daily.set_date_range(start수익곡선모멘텀)
r_daily.display()

# %%
r_daily.prices[start수익곡선모멘텀:].resample('M').last().to_drawdown_series().describe()

# %%
bt_kospi1 = long_only_ew(data_daily, ['KOSPI'], 'KOSPI')
bt_현금1 = long_only_ew(data_daily, ['현금'], '현금')

r_kospi1 = bt.run(bt_kospi1)
r_현금1 = bt.run(bt_현금1)

r_kospi1.set_date_range(start수익곡선모멘텀, e)
r_현금1.set_date_range(start수익곡선모멘텀, e)

# %%
ax1 = r_daily.plot(color='g', figsize=(12,8));
r_kospi1.plot(ax=ax1, ls='--', color='b', figsize=(12, 8));
r_현금1.plot(ax=ax1, ls='--', color='r', figsize=(12, 8));

# %%
r_daily.set_date_range(start평균모멘텀스코어)
r_daily.prices

# %% [markdown]
# ### 수익곡선모멘텀 WeighAMS

# %%
# 월말에 계산하고 다음월 첫 거래일 종가로 거래 한다고 전재
# 12개월 평균모멘텀스코어 계산

class WeighAMS(bt.Algo):
    def __init__(self, lags, months, cash):
        super(WeighAMS, self).__init__()
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
        selected.remove(self.cash)
        prc = target.universe[selected].loc[start:t0].resample('M').last()
        if (len(prc) < self.months+1):
            return False
        print(f"\nprc : {target.now} {t0} ~ {start}\n{prc}")

        # weights = pd.Series(self.평균모멘텀스코어(prc, self.months), index=selected)
        weights = pd.Series(self.AMS(prc), index=selected)
        weights[self.cash] = 1 - weights.sum()
        # print(self.평균모멘텀스코어(prc, self.months), prc)
        target.temp['weights'] = weights
        print(f"{target.now} ", end =" ")
        for i, v in weights.items():
            print(f"{i}:{v:.3f}", end=" ")
        print("")
        return True

# %%
s6 = bt.Strategy('수익곡선모멘텀6',
    [
        bt.algos.RunMonthly(run_on_first_date=True,
                            run_on_end_of_period=True, # 월말
                            run_on_last_date=False),
        # bt.algos.RunAfterDate('1986-7-31'),
        # bt.algos.PrintInfo('{name} : {now}'),
        bt.algos.SelectAll(),
        WeighAMS(lags=0, months = 6, cash='현금'), # lags=0이면 위와 같다.
        # bt.algos.PrintTempData(),
        bt.algos.Rebalance()
    ]
)

# %%
r_daily.prices

# %%
data_daily1 = pd.DataFrame()
data_daily1['수익곡선모멘텀6'] = r_daily.prices
# %%
dd = pd.DataFrame()
dd.index = pd.date_range(start평균모멘텀스코어, e, freq='D')
denominator = len(pd.date_range(start평균모멘텀스코어, e, freq='D'))/len(pd.date_range(start평균모멘텀스코어, e, freq='Y'))
dd['현금'] = pow(1.03, 1/denominator)
dd['현금'] = dd['현금'].shift(1, fill_value=1.0)
dd['현금'] = dd['현금'].cumprod()
#%%
data_daily1 = pd.merge(data_daily1, dd, left_index=True, right_index=True, how='left')

# %%
data_daily1

# %%
bt_s6 = bt.Backtest(s6, data_daily1, initial_capital=100000000)
r_6 = bt.run(bt_s6)

# %%
r_6.set_date_range(start수익곡선모멘텀)
r_6.display()

# %% [markdown]
# ### 그림 2.38 (일간)

# %%
r_daily.set_date_range(start수익곡선모멘텀)
r_kospi1.set_date_range(start수익곡선모멘텀)
r_6.set_date_range(start수익곡선모멘텀)
ax1 = r_daily.plot(color='r', figsize=(12,8));
r_kospi1.plot(ax=ax1, ls='--', color='g', figsize=(12, 8));
r_6.plot(ax=ax1, ls='-', color='b', figsize=(12, 8));
#----
# r_ms0.plot(ax=ax1, ls='--', color='gray', figsize=(12, 8));

# %% [markdown]
# # bt 적 방법
# 월말 리밸런싱(run_end_of_period=True) 계산은 하루전(lag=pd.DateOffset(days=1))

# %% [markdown]
# ### 평균모멘텀스코어 class WeighAverageMomentumScore(bt.Algo):
# 

# %%
class WeighAverageMomentumScore(bt.Algo):
    def __init__(self, months=12, lag=pd.DateOffset(days=0), cash='현금', cash_weigh=0.0, 현금자산제외=True):
        super(WeighAverageMomentumScore, self).__init__()
        self.lookback = months
        self.lag = lag
        self.cash = cash
        self.cash_weigh = cash_weigh
        self.현금자산제외 = 현금자산제외

    def __call__(self, target):
        selected = target.temp['selected'].copy()

        # print("===", selected)
        if self.현금자산제외: # 현금자산빼고 모멘텀 비중나누고 나머지는 현금으로 , 아니면 현금도 하나의 자산으로
            selected.remove(self.cash)

        t0 = target.now - self.lag

        if target.universe[selected].index[0] > (t0 - pd.DateOffset(months=self.lookback)): # !!!
            return False

        momentums_score = 0
        for lookback in range(1, self.lookback+1):
            start = t0 - pd.DateOffset(months=lookback)
            prc = target.universe[selected].loc[start:t0]
            momentum_score = np.where(prc.calc_total_return() > 0, 1, 0)
            momentums_score += momentum_score

        average_momentum_score = momentums_score / self.lookback
        print(average_momentum_score)

        average_momentum_score *= (1 - self.cash_weigh)
        
        # XXX systrader79/backtesting/ebook/dynamic/10.mxied_korea_us.py 수정필요
        if self.현금자산제외: ## XXX 이 로직이 맞는지 모르겠음
            weights = average_momentum_score/len(selected) 
            weights = pd.Series(weights, index=selected)
            weights[self.cash] = 1-weights.sum()
        else:
            weights = average_momentum_score/average_momentum_score.sum()
            weights = pd.Series(weights, index=selected)


        target.temp['weights'] = weights

        print(f"{target.now} ", end =" ")
        for i, v in weights.items():
            print(f"{i}:{v:.3f}", end=" ")
        print("")
        return True

# %%
#%%
def average_momentum_score_mixed_cash(name, data, months=12, lag=pd.DateOffset(days=0), cash='현금', 현금자산제외=False):
    st = bt.Strategy(name,
        [
            bt.algos.RunMonthly(run_on_first_date=True,
                                run_on_end_of_period=True, #월말
                                run_on_last_date=False),
            # bt.algos.PrintInfo('{name} : {now}'),
            bt.algos.SelectAll(),
            WeighAverageMomentumScore(months=months, lag=lag, cash=cash, 현금자산제외=현금자산제외),
            # bt.algos.PrintTempData(),
            bt.algos.Rebalance()
        ]
    )
    test = bt.Backtest(st, data, initial_capital=100000000)
    return test

# %%
data_daily

# %%
bt_ams_mixed_cash = average_momentum_score_mixed_cash("평균모멘텀스코어(현금혼합)", data_daily, 12, pd.DateOffset(days=1), '현금')
r_ams_mixed_cash = bt.run(bt_ams_mixed_cash)

# %%
r_ams_mixed_cash.set_date_range(start수익곡선모멘텀)
r_ams_mixed_cash.display()

# %%
bt_kospi1 = long_only_ew(data_daily, ['KOSPI'], 'KOSPI')
bt_현금1 = long_only_ew(data_daily, ['현금'], '현금')

r_kospi1 = bt.run(bt_kospi1)
r_현금1 = bt.run(bt_현금1)

r_kospi1.set_date_range(start평균모멘텀스코어, e)
r_현금1.set_date_range(start평균모멘텀스코어, e)

# %%
ax1 = r_ams_mixed_cash.plot(color='g', figsize=(12,8));
r_kospi1.plot(ax=ax1, ls='--', color='b', figsize=(12, 8));
r_현금1.plot(ax=ax1, ls='--', color='r', figsize=(12, 8));

# %% [markdown]
# ### 수익곡선모멘텀

# %%
r_ams_mixed_cash.set_date_range(start평균모멘텀스코어)
r_ams_mixed_cash.prices

# %%
data_daily2 = pd.DataFrame()
data_daily2['수익곡선모멘텀6'] = r_ams_mixed_cash.prices
# %%
dd = pd.DataFrame()
dd.index = pd.date_range(start평균모멘텀스코어, e, freq='D')
denominator = len(pd.date_range(start평균모멘텀스코어, e, freq='D'))/len(pd.date_range(start평균모멘텀스코어, e, freq='Y'))
dd['현금'] = pow(1.03, 1/denominator)
dd['현금'] = dd['현금'].shift(1, fill_value=1.0)
dd['현금'] = dd['현금'].cumprod()
#%%
data_daily2 = pd.merge(data_daily2, dd, left_index=True, right_index=True, how='left')

# %%
data_daily2

# %%
bt_returns_curve_mom = average_momentum_score_mixed_cash("평균모멘텀스코어(현금혼합)+수익곡선모멘텀", 
                                data_daily2, 6, pd.DateOffset(days=1), '현금', 현금자산제외=True)
r_returns_curve_mom = bt.run(bt_returns_curve_mom)

# %%
r_returns_curve_mom.set_date_range(start수익곡선모멘텀)
r_returns_curve_mom.display()

# %%
r_returns_curve_mom.get_security_weights().plot.area(figsize=(12,4));

# %%
r_returns_curve_mom.prices

# %%
plot_assets(r_returns_curve_mom, start수익곡선모멘텀, e, "평균모멘텀스코어(현금혼합)+수익곡선모멘텀")

# %% [markdown]
# ### 그림 2.38 (일간, bt적 알고)

# %%
r_ams_mixed_cash.set_date_range(start수익곡선모멘텀)
r_kospi1.set_date_range(start수익곡선모멘텀)
r_returns_curve_mom.set_date_range(start수익곡선모멘텀)
ax1 = r_ams_mixed_cash.plot(color='r', figsize=(12,8));
r_kospi1.plot(ax=ax1, ls='--', color='g', figsize=(12, 8));
r_returns_curve_mom.plot(ax=ax1, ls='-', color='b', figsize=(12, 8));
#----
# r_ms0.plot(ax=ax1, ls='--', color='gray', figsize=(12, 8));

# %% [markdown]
# # 비교
# 
# - 월별 비중 데이터프레임
#   - CAGR                 6.43%/Max Drawdown         -19.12%
#   - CAGR                 6.44%/Max Drawdown         -10.19% (수익곡선모멘텀6)
#   
# - 일일데이터 비중 클래스
#   - CAGR                 6.41%/Max Drawdown         -21.13%
#   - CAGR                 6.39%/ Max Drawdown         -13.74% (수익곡선모멘텀6)
# 
# - bt적 방법
#   - CAGR                 6.38%/Max Drawdown         -21.28%
#   - CAGR                 6.30%/Max Drawdown         -12.88% (수익곡선모멘텀6)

# %%
r_daily.set_date_range(start수익곡선모멘텀)
r_6.set_date_range(start수익곡선모멘텀)
r_ams_mixed_cash.set_date_range(start수익곡선모멘텀)
r_returns_curve_mom.set_date_range(start수익곡선모멘텀)
#--
ax1 = r_daily.plot(figsize=(12,8));
# r_6.plot(ax=ax1,figsize=(12,8));
#--
r_ams_mixed_cash.plot(ax=ax1,figsize=(12,8));
# r_returns_curve_mom.plot(ax=ax1,figsize=(12,8));

# %%
r_daily.set_date_range(start수익곡선모멘텀)
r_6.set_date_range(start수익곡선모멘텀)
r_ams_mixed_cash.set_date_range(start수익곡선모멘텀)
r_returns_curve_mom.set_date_range(start수익곡선모멘텀)
#--
# ax1 = r_daily.plot(figsize=(12,8));
ax1 = r_6.plot(figsize=(12,8));
#--
# r_ams_mixed_cash.plot(ax=ax1,figsize=(12,8));
r_returns_curve_mom.plot(ax=ax1,figsize=(12,8));

# %%
r_daily.set_date_range(start수익곡선모멘텀)
r_6.set_date_range(start수익곡선모멘텀)
r_ams_mixed_cash.set_date_range(start수익곡선모멘텀)
r_returns_curve_mom.set_date_range(start수익곡선모멘텀)
#--
ax1 = r_daily.plot(figsize=(12,8));
r_6.plot(ax=ax1,figsize=(12,8));
#--
r_ams_mixed_cash.plot(ax=ax1,figsize=(12,8));
r_returns_curve_mom.plot(ax=ax1,figsize=(12,8));

# %%
