# %% [markdown]
# # 주식채권현금 평균모멘텀스코어 분산투자 한국 그림 2-33
# - 투자 기간 : 2002년 1월~2017월 6일
# - 투자 대상 : 코스피200지수, 10년 만기 국고채 지수, 20년 만기 국고채 지수, 현금(3년 만기 국고채 지수)
# - 매수 규칙 : 주식:채권:현금=코스피200지수 최근 12개월 평균 모멘텀 스코어 10년(20년) 만기 국고채 지수 최근 12개월 평균 모멘텀 스코어 : 1(현금 모멘텀)
# - 매도 규칙 : 매달 말 위의 투자 비중을 새로 계산하여 주식:채권:현금 투자 비중을 조절하여 반복
# - 동일 비중 
#   - 20년 국채 9.1%/-15.1%
# 

# %%
from settings import *

# %%
tickers = ['kodex200', 'kbond3y', 'kbond10y', 'kbond20y']
read_df = get_data()
read_df = read_df[tickers]
read_df.info()

# %%
data = read_df['2001':].copy()
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
bt_kospi = long_only_ew(data, ['kodex200'], 'kodex200')
bt_10 = average_momentum_score_strategy('10년국채모멘텀', data, ['kodex200', 'kbond3y', 'kbond10y'])
bt_20 = average_momentum_score_strategy('20년국채모멘텀', data, ['kodex200', 'kbond3y', 'kbond20y'])

r10 = bt.run(bt_10)


# %%

r20 = bt.run(bt_20)
r_kospi = bt.run(bt_kospi)
r = bt.run(bt_10, bt_20, bt_kospi)

# %%
r.set_date_range('2002-02-01')
r.display()

# %%
r.set_date_range('2002-02-01', '2017-6')
r.display()

# %%
r.prices['2002-02-01':'2017-6'].to_drawdown_series().describe()

# %%
ax1 = r_kospi.prices['2002-2-1':'2017-6'].rebase(1).plot(ls='--', figsize=(12, 8));
r10.prices['2002-2-1':'2017-6'].rebase(1).plot(ax=ax1, lw=2, figsize=(12, 8));
r20.prices['2002-2-1':'2017-6'].rebase(1).plot(ax=ax1, lw=2, figsize=(12, 8));

# %%
bt_invvol10 = bt.Backtest(변동성역가중, data[['kodex200', 'kbond10y']], name='10년국채(변동성역가중)', initial_capital=100000000)
bt_invvol20 = bt.Backtest(변동성역가중, data[['kodex200', 'kbond20y']], name='20년국채(변동성역가중)', initial_capital=100000000)

# %%
r_2_30 = bt.run(bt_daily_10, bt_invvol10, bt_daily_20, bt_invvol20)

# %%
r_2_30.set_date_range(start, '2017-06')
r_2_30.display()

# %%
#그림 2.30
r_kospi.set_date_range(start, '2017-06')
ax1 = r_2_30.plot(figsize=(12,8));
r_kospi.plot(ax=ax1, ls='--', figsize=(12,8));

# %% [markdown]
# 책의 그림과 많이 다른데....성능이 책 만큼 나오지 않았다.

# %%
