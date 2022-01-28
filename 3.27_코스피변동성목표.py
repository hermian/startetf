# %% [markdown]
# - 특정 기간 동안 투자 종목의 단위 투자 기간(일간, 주간, 월간)의 수익률을 각각 계산
# - 구해진 수익률의 표준편차 계산
# - 최종 투자 비중 = 제한하기를 원하는 손실 한계(변동성 목표) / 수익률 표준편차
# - 나머지 투자 비중 = 현금 보유
# - 목표 1% 4.3%/-10.3%
# - 목표 2% 5.5%/-24.1%
# - 목표 3% 6.3%/-37.6%
# - 목표 4% 7.1%/-49.2%
# - 목표 5% 7.3%/-59.8%
# - 코스피 8.9%/-73.1%

# %%
from settings import * 

# %%
kospi = fdr.DataReader('KS11')[['Close']]

# %%
kospi

# %%
s = '1985-1-30'
e = '2017-6-30'
start = '1986-1-30'

# %%
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

# %%
# - 특정 기간 동안 투자 종목의 단위 투자 기간(일간, 주간, 월간)의 수익률을 각각 계산
# - 구해진 수익률의 표준편차 계산
# - 최종 투자 비중 = 제한하기를 원하는 손실 한계(변동성 목표) / 수익률 표준편차
# - 나머지 투자 비중 = 현금 보유

# XXX 필요하면 weights를 구해서 WeighTarget Algo로 해도 된다.
class WeighTargetVol(bt.Algo):
    """ 한개의 자산에 대해 즉 포트폴리오의 경우 그 결과에 대해서 TargetVol을 제어한다.

        자산 1개와 현금 1개로 구성된 prices DataFrame을 사용한다.
    """
    def __init__(self, targetvol=0.01, months=6, lag=pd.DateOffset(days=0), cash_name='현금'):
        super(WeighTargetVol, self).__init__()
        self.targetvol = targetvol
        self.lookback = months
        self.lag = lag
        self.cash_name = cash_name

    def __call__(self, target):
        selected = target.temp['selected'].copy()
        
        t0 = target.now - self.lag
        selected.remove(self.cash_name)

        start = t0 - pd.DateOffset(months=self.lookback)
        prc = target.universe.loc[start:t0, selected]

        # 월별 수익률의 변동성
        mret = prc.resample('M').last().pct_change().dropna()
        std = mret.std()
        print(std.values[0], mret)
    

        # H14 : std()
        # N$1 : targetvol
        # -------------------------------------------
        # =IF(H14>N$1, 
        # targetvol보다 변동성이 큰 경우
        #     N$1/H14 * B15/B14           <-- 주식 비중 TargetVol/RealVol
        #     +(1-N$1/H14)*1.03^(1/12),   <-- 현금 비중 (1 - 주식비중)
        # targetvol보다 변동성이 작은 경우
        #     B15/B14 투자비중 1인 경우
        #     )
        #     *K14 이전 수익률
        if std.values[0] > self.targetvol:
            weights = pd.Series(self.targetvol/std, index=selected)
        else:
            weights = pd.Series(1.0, index=selected)
        weights[self.cash_name] = 1.0 - weights.sum()

        target.temp['weights'] = weights
        # print("@@@", target.temp)

        return True

# %%
def strategy_targetvol(name, data, targetvol=0.01, months=12, lag=pd.DateOffset(days=0), cash_name='현금'):
    s = bt.Strategy(name, 
            [
                bt.algos.RunMonthly(run_on_end_of_period=True),
                bt.algos.RunAfterDate(start),
                bt.algos.SelectAll(),
                # bt.algos.PrintTempData(),
                #-------------------------------------------
                WeighTargetVol(targetvol, months, lag, cash_name),
                #-------------------------------------------
                bt.algos.PrintInfo("{now} {temp}"),
                bt.algos.Rebalance()
            ]
    )
    return bt.Backtest(s, data_daily, initial_capital=100000000.0)

# %%
bt목표1 = strategy_targetvol("목표1%", data_daily, targetvol=0.01)
r = bt.run(bt목표1)

# %%
r.set_date_range(start)
r.display()

# %%
r.plot(freq='M', figsize=(12, 8));

# %%
bt_kospi = long_only_ew(data_daily, ['KOSPI'], 'KOSPI')
r_kospi = bt.run(bt_kospi)

# %%
r_all = bt.run(bt_kospi, bt목표1)
r_all.set_date_range(start)
r_all.display()

# %%
plot_assets(r_all, start, e, "목표1%")

# %%
bt목표1 = strategy_targetvol("목표1%", data_daily, targetvol=0.01)
bt목표2 = strategy_targetvol("목표2%", data_daily, targetvol=0.02)
bt목표3 = strategy_targetvol("목표3%", data_daily, targetvol=0.03)
bt목표4 = strategy_targetvol("목표4%", data_daily, targetvol=0.04)
bt목표5 = strategy_targetvol("목표5%", data_daily, targetvol=0.05)
r_all = bt.run(bt목표1,bt목표2,bt목표3,bt목표4,bt목표5,bt_kospi)

# %%
r_all.set_date_range(start)
r_all.display()

# %%
r_all.prices.resample('M').last().to_drawdown_series().describe()

# %%
r_all.plot(freq='M', figsize=(12, 8));

# %% [markdown]
# ## 변동성 목표 전략 + 모멘텀 전략

# %%
# - 특정 기간 동안 투자 종목의 단위 투자 기간(일간, 주간, 월간)의 수익률을 각각 계산
# - 구해진 수익률의 표준편차 계산
# - 최종 투자 비중 = 제한하기를 원하는 손실 한계(변동성 목표) / 수익률 표준편차
# - 나머지 투자 비중 = 현금 보유

# XXX 필요하면 weights를 구해서 WeighTarget Algo로 해도 된다.
class WeighTargetVolWithAMS(bt.Algo):
    """ 한개의 자산에 대해 즉 포트폴리오의 경우 그 결과에 대해서 TargetVol을 제어한다.

        자산 1개와 현금 1개로 구성된 prices DataFrame을 사용한다.
    """
    def __init__(self, targetvol=0.01, months=12, lag=pd.DateOffset(days=0), cash_name='현금'):
        super(WeighTargetVolWithAMS, self).__init__()
        self.targetvol = targetvol
        self.lookback = months
        self.lag = lag
        self.cash_name = cash_name

    def __call__(self, target):
        selected = target.temp['selected'].copy()
        
        t0 = target.now - self.lag
        selected.remove(self.cash_name)

        start = t0 - pd.DateOffset(months=self.lookback)
        prc = target.universe.loc[start:t0, selected]

        if target.universe[selected].index[0] > (t0 - pd.DateOffset(months=self.lookback)): # !!!
            return False

        momentums_score = 0
        for lookback in range(1, self.lookback+1):
            start = t0 - pd.DateOffset(months=lookback)
            prc = target.universe[selected].loc[start:t0]
            momentum_score = np.where(prc.calc_total_return() > 0, 1, 0)
            momentums_score += momentum_score

        average_momentum_score = momentums_score / self.lookback

        mret = prc.resample('M').last().pct_change().dropna()
        std = mret.std()
        print(std.values[0], mret)
    

        # H14 : std()
        # N$1 : targetvol
        # -------------------------------------------
        # =IF(H14>N$1, 
        # targetvol보다 변동성이 큰 경우
        #     N$1/H14 * B15/B14           <-- 주식 비중 TargetVol/RealVol
        #     +(1-N$1/H14)*1.03^(1/12),   <-- 현금 비중 (1 - 주식비중)
        # targetvol보다 변동성이 작은 경우
        #     B15/B14 투자비중 1인 경우
        #     )
        #     *K14 이전 수익률
        if std.values[0] > self.targetvol:
            print("==================", self.targetvol/std)
            weights = pd.Series(self.targetvol/std * average_momentum_score, index=selected)
        else:
            weights = pd.Series(1.0, index=selected)
        weights[self.cash_name] = 1.0 - weights.sum()

        target.temp['weights'] = weights

        return True

# %%
def strategy_targetvol_with_ams(name, data, targetvol=0.01, months=12, lag=pd.DateOffset(days=0), cash_name='현금'):
    s = bt.Strategy(name, 
            [
                bt.algos.RunMonthly(run_on_end_of_period=True),
                bt.algos.RunAfterDate(start),
                bt.algos.SelectAll(),
                # bt.algos.PrintTempData(),
                #-------------------------------------------
                WeighTargetVolWithAMS(targetvol, months, lag, cash_name),
                #-------------------------------------------
                bt.algos.PrintInfo("{now} {temp}"),
                bt.algos.Rebalance()
            ]
    )
    return bt.Backtest(s, data_daily, initial_capital=100000000.0)

# %%
bt목표1모멘텀 = strategy_targetvol_with_ams("목표1%모멘텀", data_daily, targetvol=0.01)
r11 = bt.run(bt목표1모멘텀)

# %%
r11.set_date_range(start)
r11.display()

# %%
ax1 = r.plot(figsize=(12,8))
r11.plot(ax=ax1, figsize=(12,8))

# %%
bt목표2모멘텀 = strategy_targetvol_with_ams("목표2%모멘텀", data_daily, targetvol=0.02)
r12 = bt.run(bt목표2모멘텀)

# %%
r목표1 = bt.run(bt목표1)
r목표2 = bt.run(bt목표2)
r목표1모멘텀 = bt.run(bt목표1모멘텀)
r목표2모멘텀 = bt.run(bt목표2모멘텀)
r목표1.set_date_range(start)
r목표2.set_date_range(start)
r목표1모멘텀.set_date_range(start)
r목표2모멘텀.set_date_range(start)

ax1 = r목표1.plot(ls='--', figsize=(12, 8))
r목표1모멘텀.plot(ax=ax1, ls='-', figsize=(12, 8))
r목표2.plot(ax=ax1, ls='--', figsize=(12, 8))
r목표2모멘텀.plot(ax=ax1, ls='-', figsize=(12, 8))

# %%
r_all = bt.run(bt목표1, bt목표1모멘텀, bt목표2, bt목표2모멘텀, bt_kospi)
r_all.set_date_range(start)
r_all.display()

# %%



