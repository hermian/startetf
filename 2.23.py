# %%
from settings import *

# %%
kospi = fdr.DataReader("KS11")[['Close']] # dataframe [['Close]]
kospi.columns = ['KOSPI']
kospi.info() # 1981-05-01 to 2022-01-19

# %%
data = kospi['1986-1':'2017-6'].copy()
data_m = data.resample('MS').first() # 월초
data_m.info()

# %%
data_m

# %%
start = '1986-1'
end = '2017-6'

# %%
def momentum(data, trend, name='1개월모멘텀'):
    s = bt.Strategy(
        name,
        [
            bt.algos.RunMonthly(run_on_first_date=True, run_on_end_of_period=False, run_on_last_date=True),
            bt.algos.SelectWhere(trend), # !!!
            bt.algos.WeighEqually(),
            bt.algos.PrintInfo("{now} {temp}"),
            bt.algos.Rebalance()
        ])
    test = bt.Backtest(s, data, initial_capital=100000000)

    return test

# %%
bt_kospi = long_only_ew(data_m, ["KOSPI"], "KOSPI", initial_capital=100000000)

# %% [markdown]
# ## 비중 데이터프레임을 별도로 만들어서 일반화 함 (SelectWhere(weights))

# %%
# 절대 모멘텀은 이동평균이 아니라 단순히 과거 시점 대비 현재를 비교한다.
# 1개월 모멘텀 (data는 월간 데이터)
prev = data_m.shift(1)
trend = prev.copy()
trend[data_m > prev] = True
trend[data_m <= prev] = False
trend[prev.isnull()] = False
trend

# %%
bt_1m = momentum(data_m, trend, "1개월모멘텀")
rm1 = bt.run(bt_1m)

# %%
rm_kospi = bt.run(bt_kospi)
rm1_assets = bt.run(bt_kospi, bt_1m)

# %%
rm1_assets.prices

# %%
rm1_assets.set_date_range('1986-2-1', end)
rm1_assets.display()

# %%
ax1 = rm_kospi.plot(ls='--', figsize=(12, 8));
rm1.plot(ax=ax1, figsize=(12,8));

# %%
rm1.get_security_weights().plot.area(figsize=(12,3));

# %%
rm1.get_transactions().head(10)

# %%
# 거래 발생
rm1.backtest_list[0].positions.diff(1).head(10)

# %%
# outlays 유가 증권 매입(매각)에 의해 총 지출된 금액(거래한 날만 나옴, 거래 없는 날 0)
# 19862-1 162.45*615574.0
rm1.backtest_list[0].strategy.outlays.head(10)

# %% [markdown]
# ## 1-12개월별 절대 모멘텀 백테스팅

# %%
# Print Off
def momentum(data, trend, name='1개월모멘텀'):
    s = bt.Strategy(
        name,
        [
            bt.algos.RunMonthly(run_on_first_date=True, run_on_end_of_period=False, run_on_last_date=True), #월초
            bt.algos.SelectWhere(trend), # !!!
            bt.algos.WeighEqually(),
#             bt.algos.PrintInfo("{now} {temp}"),
            bt.algos.Rebalance()
        ])
    test = bt.Backtest(s, data, initial_capital=100000000)

    return test

# %%
#%% 1~12개월에 대한 백테스팅
# https://blog.naver.com/hermian71/222577014997
test_m_mom = dict()
for i in range(1, 13):
    prev = data_m.shift(i)
    trend = prev.copy()
    trend[data_m > prev] = True
    trend[data_m <= prev] = False
    trend[prev.isnull()] = False
    test_m_mom[i] = momentum(data_m, trend, name=str(i)+"개월모멘텀")

# %%
rm_1_12 = bt.run(*test_m_mom.values())

# %%
ax1 = rm_1_12.plot(figsize=(12, 8));
rm_kospi.plot(ax=ax1, lw=2, ls='-', color='gray', alpha=0.5, figsize=(12, 8));

# %% [markdown]
# plot_df()함수 사용(utils)
# 
# - legend 칼라와 순서를 맞춘다.

# %%
df = bt.merge(rm_1_12.prices, rm_kospi.prices)
plot_df(df, logy=False)

# %%
# legend의 순서가 그래프의 마지막값 순서
rm_1_12.prices.tail(1).T.sort_values(by='2017-06-01', ascending=False)

# %%
rm_1_12.get_security_weights(11).plot.area(figsize=(12,4));

# %%
rm369 = bt.run(test_m_mom[1], test_m_mom[3], test_m_mom[6],
               test_m_mom[9], test_m_mom[12], bt_kospi)

# %%
rm369.plot(figsize=(12,8));

# %%
r = rm369.prices.rebase()
ax = r.iloc[:, 0:-1].plot(legend=False);# kospi제외
r.iloc[:,-1].plot(ax=ax, color='gray', lw=1.5, ls=':', label='KOSPI', figsize=(12,8));
# leg = ax.legend()
# print(dir(list(ax.get_lines())[0]))

# for l in list(ax.get_lines()):
#     print(l.get_label(), l.get_color())

for i, line in enumerate(list(ax.get_lines())):
    ax.text(r.index[-1], r.iloc[-1,i], line.get_label(), size=12, color=line.get_color(), ha='center');

# style = dict(size=12, color='black', ha='center')
# ax.text(r.index[-1], r.iloc[-1,0], '1개월모멘텀 ', **style)
# ax.text(r.index[-1], r.iloc[-1,1], '3개월모멘텀 ', **style)
# ax.text(r.index[-1], r.iloc[-1,2], '6개월모멘텀 ', **style)
# ax.text(r.index[-1], r.iloc[-1,3], '9개월모멘텀 ', **style)
# ax.text(r.index[-1], r.iloc[-1,4], '12개월모멘텀 ', **style)
# ax.text(r.index[-1], r.iloc[-1,5], 'KOSPI ', **style)
plt.show();

# %%
# %% seaborn으로 그리면
import seaborn as sns
plt.figure(figsize=(12,8))
sns.lineplot(data=r);

# %% [markdown]
# ## bt의 Algo 클래스를 작성하여 동일 전략을 백테스팅 해보자.
# 
# 아래 클래스는 일간데이터만 처리 가능하다.
# 위의 월간 데이터와 같게 만들지는 않는다.
# 일반적인 bt 스타일의 백테스팅을 진행한다. => 결과는 알아서 비교해 볼것

# %%
class Signal(bt.Algo):
    """

    Mostly copied from StatTotalReturn

    Sets temp['Signal'] with total returns over a given period.

    Sets the 'Signal' based on the total return of each
    over a given lookback period.

    Args:
        * lookback (DateOffset): lookback period.
        * lag (DateOffset): Lag interval. Total return is calculated in
            the inteval [now - lookback - lag, now - lag]

    Sets:
        * stat

    Requires:
        * selected

    """


    def __init__(self, lookback=pd.DateOffset(months=3), lag=pd.DateOffset(days=0)):
        super(Signal, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        # print(f"\n====== {target.now}, {self.lookback}")
        selected = target.temp['selected']
        t0 = target.now - self.lag

        if target.universe[selected].index[0] > (t0 - self.lookback): # !!!
            return False

        prc = target.universe[selected].loc[t0 - self.lookback:t0]
        # print(target.now, t0 , t0 - self.lookback, '\n')#, prc)

        trend = prc.iloc[-1]/prc.iloc[0] - 1
        signal = trend > 0.

        target.temp['Signal'] = signal.astype(float)

        return True



class WeighFromSignal(bt.Algo):

    """
    Sets temp['weights'] from the signal.
    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(self):
        super(WeighFromSignal, self).__init__()

    def __call__(self, target):
        selected = target.temp['selected']
        if target.temp['Signal'] is None:
            raise(Exception('No Signal!'))

        target.temp['weights'] = target.temp['Signal']
        return True

# %%
def abs_momentum_month(data, tickers = ['KOSPI'], n=1, lag=0, name="1개월모멘텀"):
    if n == 12: # months=12로 하면 이상하게 months=12, years=1이되어 24개월의 모멘텀을 구한다.
        offset = pd.DateOffset(years=1)
    else:
        offset = pd.DateOffset(months=n)

    s = bt.Strategy(
        name,
        [
            bt.algos.RunMonthly(run_on_first_date=True, run_on_end_of_period=False, run_on_last_date=True),
            bt.algos.SelectThese(tickers),
            Signal(lookback=offset, lag=pd.DateOffset(days=lag)),
            WeighFromSignal(),
            bt.algos.PrintInfo("{now} {temp}"),
            bt.algos.Rebalance()
        ]
    )

    t = bt.Backtest(s, data)
    return t

# %%
data = kospi[start:end].copy() # 일간 데이터
data.info()

# %%
data

# %%
bt_c_1 = abs_momentum_month(data, ['KOSPI'], 1, lag=0, name='1개월')
r_c_1 = bt.run(bt_c_1)

# %%
r_c_1.get_security_weights().plot.area(figsize=(12, 4))

# %%
rm1.get_security_weights().plot.area(figsize=(12,4))

# %%
r_c_1.set_date_range('1986-3')
r_c_1.display()

# %%
r_c_1.prices.resample('M').last().to_drawdown_series().min()
r_c_1.prices.resample('MS').first().to_drawdown_series().min()

# %%
r_diff = bt.run(bt_1m)
r_diff.set_date_range('1986-3')
r_diff.display()

# %%
ax1 = r_c_1.prices.to_drawdown_series().plot.area(figsize=(12,5));
rm1.prices.to_drawdown_series().plot.area(ax=ax1, figsize=(12,5));

# %%
# PrintInfo 제거 버전
def abs_momentum_month(data, tickers = ['KOSPI'], n=1, lag=1, name="1개월모멘텀"):
    if n == 12: # months=12로 하면 이상하게 months=12, years=1이되어 24개월의 모멘텀을 구한다.
        offset = pd.DateOffset(years=1)
    else:
        offset = pd.DateOffset(months=n)

    s = bt.Strategy(
        name,
        [
            bt.algos.RunMonthly(run_on_first_date=True, run_on_end_of_period=False, run_on_last_date=True),
            bt.algos.SelectThese(tickers),
            Signal(lookback=offset, lag=pd.DateOffset(days=lag)),
            WeighFromSignal(),
#             bt.algos.PrintInfo("{now} {temp}"),
            bt.algos.Rebalance()
        ]
    )

    t = bt.Backtest(s, data)
    return t

test_mom = dict()
for i in range(1, 13):
    test_mom[i] = abs_momentum_month(data, ['KOSPI'], i, lag=1, name=str(i)+"개월모멘텀")


# %%
r1_12 = bt.run(*test_mom.values())

# %%
bt_kospi = long_only_ew(data, ["KOSPI"], "KOSPI", initial_capital=100000000)
r_kospi = bt.run(bt_kospi)

# %%
r1_12.prices.tail(1).T.sort_values(by='2017-06-30', ascending=False)

# %%
r1_12.stats

# %%
r1_12.prices.resample('M').last().to_drawdown_series().describe()

# %%
ax1 = r1_12.plot(figsize=(12, 8));
r_kospi.plot(ax=ax1, lw=2, ls='-', color='gray', alpha=0.5, figsize=(12, 8));

# %%
df = bt.merge(r1_12.prices, r_kospi.prices)
plot_df(df, logy=False)

# %%
r1_12.plot(freq='M',figsize=(12,8));

# %%
