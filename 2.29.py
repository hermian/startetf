# %% [markdown]
# # 평균모멘텀스코어 채권 혼합 그림 2-29
# - 10년 국채 10.58%/-13.5%, 20년 국채 15.48%/-14.3%
# - 투자 기간 : 2002년 1월 ~2017년 6월
# - 투자 대상 : 코스피200 지수, 10년 (또는 20년) 만기 국고채 지수
# - 매수 규칙 : 주식 :채권 = 코스피200 최근 12개월 평균 모멘텀 스코어:10년(또는 20년) 만기 국고채 지수 최근 12개월 평균 모멘텀 스코어 
# - 매도 규칙 : 매달 말 위의 투자 비중을 새로 계산하여 주식 :채권 투자 비중을 조절하여 반복
# - 변동성 역가중 전략과 비교 그림 2-30
#   - 10년 국채 8.3%/-11.9%
#   - 20년 국채 13.2%/-19.1%
# 
# % 책과 내가 가진 data에서 차이가 많이 난다.
# 

# %%
from settings import *

# %%
tickers = ['kodex200', 'kodex200x2', 'kbond10y', 'kbond20y']
read_df = get_data()
read_df = read_df[tickers]
read_df.info()

# %%
data = read_df['2001':].copy()
data.info()

# %%
data = data.resample('MS').first()

# %%
start = '2002-1-31'

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
score_주식_채권10년 = 평균모멘텀스코어(data[['kodex200', 'kbond10y']])
score_주식_채권20년 = 평균모멘텀스코어(data[['kodex200', 'kbond20y']])

score_주식2x_채권10년 = 평균모멘텀스코어(data[['kodex200x2', 'kbond10y']])
score_주식2x_채권20년 = 평균모멘텀스코어(data[['kodex200x2', 'kbond20y']])

# %%
weights_주식_채권10년 = score_주식_채권10년.div(score_주식_채권10년.sum(axis=1), axis=0)
weights_주식_채권20년 = score_주식_채권20년.div(score_주식_채권20년.sum(axis=1), axis=0)

weights_주식2x_채권10년 = score_주식2x_채권10년.div(score_주식2x_채권10년.sum(axis=1), axis=0)
weights_주식2x_채권20년 = score_주식2x_채권20년.div(score_주식2x_채권20년.sum(axis=1), axis=0)

# %%
weights_주식_채권10년.plot.area(figsize=(12,4));

# %%
weights_주식2x_채권20년.plot.area(figsize=(12,4));

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

bt_ms주식채권10년 = momentumscore(data[['kodex200', 'kbond10y']], weights_주식_채권10년, '10년국채모멘텀')
bt_ms주식채권20년 = momentumscore(data[['kodex200', 'kbond20y']], weights_주식_채권20년, '20년국채모멘텀')

bt_ms주식2x채권10년 = momentumscore(data[['kodex200x2', 'kbond10y']], weights_주식2x_채권10년, '주식2x_10년국채모멘텀')
bt_ms주식2x채권20년 = momentumscore(data[['kodex200x2', 'kbond20y']], weights_주식2x_채권20년, '주식2x_20년국채모멘텀')

# %%
bt_kospi = long_only_ew(data, ['kodex200'], 'kodex200')
bt_채권10년 = long_only_ew(data, ['kbond10y'], '채권10년')
bt_채권20년 = long_only_ew(data, ['kbond20y'], '채권20년')

# %%
r0 = bt.run(bt_ms주식채권10년, bt_ms주식2x채권10년, bt_ms주식채권20년, bt_ms주식2x채권20년, bt_kospi, bt_채권10년, bt_채권20년)

# %%
r0.plot(figsize=(12,8));

# %%
r0.set_date_range(start, '2017-6')
r0.display()

# %%
r0.get_security_weights('10년국채모멘텀').plot.area(figsize=(12,4))

# %%
r_kospi = bt.run(bt_kospi)
r_10 = bt.run(bt_ms주식채권10년)
r_20 = bt.run(bt_ms주식채권20년)

# %% [markdown]
# # 그림 2.29

# %%
end='2017-6'
ax1 = r_20.prices[start:end].rebase(1).plot(color='r', figsize=(12,8));
r_kospi.prices[start:end].rebase(1).plot(ax=ax1, ls='--', color='gray', figsize=(12, 8));
r_10.prices[start:end].rebase(1).plot(ax=ax1, ls='-', color='b', figsize=(12, 8));

# %%
# qs.reports.full(r_all.prices['모멘텀 포트폴리오(현금혼합)'].to_returns().dropna()['1986':],
#                 r_all.prices['KOSPI'].to_returns().dropna()['1986':])


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

        # print(self.평균모멘텀스코어(prc, self.months), prc)
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
bt_daily_12 = average_momentum_score_strategy('10년국채모멘텀1', data, ['kodex200', 'kbond10y'])
r_daily_12 = bt.run(bt_daily_12)
r_00 = bt.run(bt_daily_12, bt_ms주식채권10년)

# %%
# 처음과 동일함을 검증(로직)
r_00.set_date_range(start, '2017-6')
r_00.display()

# %%
data = read_df['2001':].copy()
data

# %%
bt_daily_10 = average_momentum_score_strategy('10년국채모멘텀(일간)', data, ['kodex200', 'kbond10y'])
bt_daily_20 = average_momentum_score_strategy('20년국채모멘텀(일간)', data, ['kodex200', 'kbond20y'])

bt_daily_2x_10 = average_momentum_score_strategy('10년국채모멘텀(일간,x2)', data, ['kodex200x2', 'kbond10y'])
bt_daily_2x_20 = average_momentum_score_strategy('20년국채모멘텀(일간,x2)', data, ['kodex200x2', 'kbond20y'])

# %%
r_daily = bt.run(bt_daily_10, bt_daily_2x_10, bt_daily_20, bt_daily_2x_20)

# %%
r_daily.set_date_range(start, '2017-06')
r_daily.display()

# %%
r_daily.prices[start:'2017-6'].to_drawdown_series().describe()

# %%
변동성역가중 = bt.Strategy('변동성역가중',
    [
        bt.algos.RunAfterDate(start),
        bt.algos.RunMonthly(run_on_end_of_period=True), #매월말
        bt.algos.PrintDate(),
        bt.algos.SelectAll(),
        bt.algos.WeighInvVol(lookback=pd.DateOffset(years=1), lag=pd.DateOffset(days=1)),
        bt.algos.PrintTempData(),
        bt.algos.Rebalance()

    ])

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

# %% [markdown]
# 
