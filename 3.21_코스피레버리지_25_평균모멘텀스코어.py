# XXX 책과 좀 상이한 결과
# %%
from settings import *

# %%
tickers  = ['kodex200', 'kodex200x2']
read_df = get_data()
data = read_df[tickers].copy()
data = data.dropna()
data.info()

# %%
s = '2000-1-4'
e = '2021-12-30'

# %%
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
def strategy(name, data, weights):
    # print(*kwargs)
    s = bt.Strategy(name, 
            [
                bt.algos.RunMonthly(run_on_end_of_period=True),
                bt.algos.SelectAll(),
                bt.algos.WeighSpecified(**weights),
                # bt.algos.PrintTempData(),
                bt.algos.Rebalance(),
                # bt.algos.PrintInfo('{now} {name} {_price} {temp} \n{_universe}')
            ])

    return bt.Backtest(s, data, initial_capital=100000000.0)


# %%
bt코스피20현금80 = strategy("코스피20%+현금80%",   data[['kodex200', '현금']], {'kodex200': 0.2, '현금':0.8}) 
bt코스피레버리지10현금90 = strategy("코스피레버리지10%+현금90%",   data[['kodex200x2', '현금']], {'kodex200x2': 0.1, '현금':0.9}) 

# %%
r_all = bt.run(bt코스피20현금80, bt코스피레버리지10현금90)

# %%
r_all.plot()

# %%
r_all.display()

# %%
plot_assets(r_all, '2000', '2021', '코스피레버리지10%+현금90%')

# %%
class WeighFixedRateAverageMomentumScore(bt.Algo):
    def __init__(self, months=12, lag=pd.DateOffset(days=0), fixed_rate=0.25, cash='현금', cash_weigh=0.0, 현금자산제외=True):
        super(WeighFixedRateAverageMomentumScore, self).__init__()
        self.lookback = months
        self.lag = lag
        self.fixed_rate = fixed_rate
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
            weights *= self.fixed_rate #!!!! 추가
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
def average_momentum_score_fixed_rate(name, data, months=12, lag=pd.DateOffset(days=0), 
                                        fixed_rate=0.25, cash='현금', 현금자산제외=False):
    st = bt.Strategy(name,
        [
            bt.algos.RunMonthly(run_on_first_date=True,
                                run_on_end_of_period=True, #월말
                                run_on_last_date=False),
            # bt.algos.PrintInfo('{name} : {now}'),
            bt.algos.SelectAll(),
            WeighFixedRateAverageMomentumScore(months=months, lag=lag, 
                                        fixed_rate=fixed_rate,cash=cash, 현금자산제외=현금자산제외),
            # bt.algos.PrintTempData(),
            bt.algos.Rebalance()
        ]
    )
    test = bt.Backtest(st, data, initial_capital=100000000)
    return test

# %%
bt레버리지모멘텀 = average_momentum_score_fixed_rate("레버리지고정비율+모멘텀", data[['kodex200x2', '현금']], 
                                            12, pd.DateOffset(days=1), 
                                            0.25, '현금', 현금자산제외=True)
r레버리지모멘텀 = bt.run(bt레버리지모멘텀)

# %%
bt모멘텀 = average_momentum_score_fixed_rate("고정비율+모멘텀", data[['kodex200', '현금']], 
                                            12, pd.DateOffset(days=1), 
                                            0.25, '현금', 현금자산제외=True)
r모멘텀 = bt.run(bt모멘텀)

# %%
bt고정비율 = strategy("고정비율",   data[['kodex200', '현금']], {'kodex200': 0.25, '현금':0.75}) 
bt레버리지고정비율 = strategy("레버리지고정비율",   data[['kodex200x2', '현금']], {'kodex200x2': 0.25, '현금':0.75}) 
r고정비율 = bt.run(bt고정비율)
r레버리지고정비율 = bt.run(bt레버리지고정비율)

# %%
r_all = bt.run(bt고정비율, bt모멘텀, bt레버리지고정비율, bt레버리지모멘텀)

# %%
r_all.set_date_range('2001-01-31')
r_all.plot(figsize=(12,8));

# %%
r_all.display()

# %%



