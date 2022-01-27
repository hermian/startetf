# %% [markdown]
# # 고정비율+평균모멘텀스코어 그림 3-17
# - 주식에 투자할 고정비율을 정함 : 25%
# - 매월 말 코스피지수의 12개월 평균 모멘텀 스코어를 구한다.
# - 미리 정해놓은 고정비율(25%)와 평균 모멘텀 스코어를 곱한 비율이 주식투자비율, 나머지가 현금비율
# - 매월 말 이 비율을 계산하여 리밸런싱한다
#   - 코스피 고정 투자 비율 25%, 코스피 평균 모멘텀 스코어 0.5 인 경우
# - 주식투자비율 = 25% x 0.5 = 12.5%,
# - 현금 투자비율 = 100% - 12.5% = 87.5%
# - 현금비율 추가하면 (1:1로 섞기 때문에 나누기 2를 해준다.)
#     - 주식투자비율 : 25%(고정비율) x 0.5(평균모멘텀스코어) x 현금비율/2  
#     - 현금투자비율 : 1 - 주식투자비율
# 

# %%
from settings import *

# %%
kospi = fdr.DataReader('KS11')[['Close']]

# %%
kospi

# %%
kospi['1981-5'].tail()

# %%
kospi['2021-12'].tail()

# %%
s = '1981-5-29'
e = '2021-12-30'

# %%
data = pd.DataFrame()
data['KOSPI'] = kospi.copy()
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
data

# %%
bt_fr = average_momentum_score_fixed_rate("고정비율+모멘텀", data, 12, pd.DateOffset(days=1), 
                                            0.25, '현금', 현금자산제외=True)
r_fr = bt.run(bt_fr)

# %%
r_fr.set_date_range('1982-05-31')
r_fr.display()

# %%
def strategy(name, data, stock_w, cash_w):
    s = bt.Strategy(name, 
            [
                bt.algos.RunMonthly(run_on_end_of_period=True),
                bt.algos.SelectAll(),
                bt.algos.WeighSpecified(KOSPI=stock_w, 현금=cash_w),
                # bt.algos.PrintTempData(),
                bt.algos.Rebalance(),
                # bt.algos.PrintInfo('{now} {name} {_price} {temp} \n{_universe}')
            ])

    return bt.Backtest(s, data, initial_capital=100000000.0)


# %%
t고정비율 = strategy("고정비율",   data, 0.25, 0.75) 

# %%
r고정비율 = bt.run(t고정비율)

# %%
r_all = bt.run(bt_fr, t고정비율)

# %%
r_all.set_date_range('1986-1-4', '2017-6')
r_all.display()

# %%
r_all.plot(figsize=(12,8))

# %%
plot_assets(r_all, '1986-01-04', '2017-06-01', '고정비율+모멘텀')

# %%
