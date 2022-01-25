# %% [markdown]
# # 주식채권현금 평균모멘텀스코어 분산투자 미국 그림 2-35
# - 투자기간 : 2003년 8월~2017년 6월
# - 투자 대상 : 미국 S&P500지수SPY, 미국 10년 만기 국채, 20년 만기 국채TLT, 현금SHY
# - 매수 규칙 : 주식 : 채권 : 현금 = S&P500지수 최근 12개월 평균 모멘텀 스코어 : 10년 (20년) 만기 국고채 지수 최근 12 개월 평균 모멘텀 스코어: 1
# - 매도 규칙 : 매달 말 위의 투자 비중을 새로 계산하여 주식 :채권 : 현금 투자 비중을 조절하여 반복
# - SPY 8.6%/-50.8%
# - SPY+IEF+SHY : 5.5%/-4.3%
# - SPY+TLT+SHY : 6.2%/-8.5%
# 

# %%
from settings import *

# %%
tickers = ['SPY', 'SHY', 'IEF', 'TLT']
read_df = yf.download(tickers)['Adj Close']
read_df = read_df[tickers]
read_df.info()

# %%
for c in read_df.columns:
    print(c, read_df[c].first_valid_index())

# %%
data = read_df.dropna().copy()
data.info()

# %% [markdown]
# # 일일데이터

# %%
# 월말에 계산하고 다음월 첫 거래일 종가로 거래 한다고 전재
# 12개월 평균모멘텀스코어 계산

class WeighAMS(bt.Algo):
    def __init__(self, lags, months):
        super(WeighAMS, self).__init__()
        self.lags = lags
        self.months = months

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

        print(f"\nprc : {target.now} {t0} ~ {start}")
        prc = target.universe[selected].loc[start:t0].resample('M').last()
        if (len(prc) < self.months+1):
            return False

        weights = pd.Series(self.AMS(prc), index=selected)
        weights = weights/weights.sum()

        print(self.평균모멘텀스코어(prc, self.months), prc)
        target.temp['weights'] = weights
        print(f"{target.now} ", end =" ")
        for i, v in weights.items():
            print(f"{i}:{v:.3f}", end=" ")
        print("")
        return True

# %%
def average_momentum_score_strategy(name, data, tickers, lags=1, months=12):
    st = bt.Strategy(name,
        [
            bt.algos.RunMonthly(run_on_first_date=True,
                                run_on_end_of_period=False, #월초
                                run_on_last_date=False),
            # bt.algos.PrintInfo('{name} : {now}'),
            bt.algos.SelectThese(tickers),
            WeighAMS(lags, months), # lags=0이면 위와 같다.
            # bt.algos.PrintTempData(),
            bt.algos.Rebalance()
        ]
    )
    return bt.Backtest(st, data, initial_capital=100000000)

# %%
bt_spy = long_only_ew(data, ['SPY'], 'SPY')
bt_10 = average_momentum_score_strategy('SPY+IEF+SHY', data, ['SPY', 'SHY', 'IEF'])
bt_20 = average_momentum_score_strategy('SPY+TLT+SHY', data, ['SPY', 'SHY', 'TLT'])

r10 = bt.run(bt_10)

# %%
r20 = bt.run(bt_20)
r_spy = bt.run(bt_spy)
r = bt.run(bt_10, bt_20, bt_spy)

# %%
r.set_date_range('2003-08-01')
r.display()

# %%
r.set_date_range('2003-08-01', '2017-6')
r.display()

# %%
r.prices['2003-08-01':'2017-6'].resample('MS').first().to_drawdown_series().describe()

# %%
r.prices['2003-08-01':'2017-6'].resample('M').last().to_drawdown_series().describe()

# %%
ax1 = r_spy.prices['2003-8-1':'2017-6'].rebase(1).plot(ls='--', figsize=(12, 8));
r10.prices['2003-8-1':'2017-6'].rebase(1).plot(ax=ax1, lw=2, figsize=(12, 8));
r20.prices['2003-8-1':'2017-6'].rebase(1).plot(ax=ax1, lw=2, figsize=(12, 8));

# %%
r_all = bt.run(bt_10, bt_20, bt_spy)
plot_assets(r_all, '2003-8', '2021', 'SPY+IEF+SHY')

# %%
r_all.set_date_range('2003-8-1')
r_all.display()
