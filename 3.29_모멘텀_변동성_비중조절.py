# TODO 클래서 작성중
# %%
from settings import *

# %%
df = get_data()

# %%
df.columns

# %%
tickers = ['kodex200', 'us500_UH', 'kbond10y', 'usbond10y_UH', 'kbond3y']

# %%
df = df[tickers].copy()
df.info()

# %%
df = df.dropna()
df.info()

# %%
s = '2002-7-30'
e = '2017-6-30'
start = '2003-7-31'

data = df[s:e].copy()
data.info()

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
# - 특정 기간 동안 투자 종목의 단위 투자 기간(일간, 주간, 월간)의 수익률을 각각 계산
# - 구해진 수익률의 표준편차 계산
# - 최종 투자 비중 = 제한하기를 원하는 손실 한계(변동성 목표) / 수익률 표준편차
# - 나머지 투자 비중 = 현금 보유

# XXX 필요하면 weights를 구해서 WeighTarget Algo로 해도 된다.
class WeighMomentumVolRatio(bt.Algo):
    """ 한개의 자산에 대해 즉 포트폴리오의 경우 그 결과에 대해서 TargetVol을 제어한다.

        자산 1개와 현금 1개로 구성된 prices DataFrame을 사용한다.
    """
    def __init__(self, months=12, lag=pd.DateOffset(days=0), cash_name='현금'):
        super(WeighMomentumVolRatio, self).__init__()
        self.lookback = months
        self.lag = lag
        self.cash_name = cash_name

    def avearge_momentum(self, t0, prices):
        momentums = 0
        for m in range(1, self.lookback+1):
            start = t0 - pd.DateOffset(months=m)
            prc = prices.loc[start:t0]
            momentum = prc.iloc[-1] / prc.iloc[0]
            momentums += momentum
        
        return momentums/self.lookback

    def average_momentum_score(self, t0, prices):
        momentums_score = 0
        for m in range(1, self.lookback+1):
            start = t0 - pd.DateOffset(months=m)
            prc = prices.loc[start:t0]
            momentum_score = np.where(prc.calc_total_return() > 0, 1, 0)
            momentums_score += momentum_score

        return momentums_score / self.lookback        


    def __call__(self, target):
        selected = target.temp['selected'].copy()
        
        selected.remove(self.cash_name)

        t0 = target.now - self.lag
        start = t0 - pd.DateOffset(months=self.lookback)
        prc = target.universe.loc[start:t0, selected]

        if target.universe[selected].index[0] > (t0 - pd.DateOffset(months=self.lookback)): # !!!
            return False

        momentum = self.avearge_momentum(t0, prc)
        score = self.average_momentum_score(t0, prc)
        mret = prc.resample('M').last().pct_change().dropna()
        std = mret.std()
        # print(std.values[0], mret)
    
        # print(momentum, "@@@", ((momentum > 1)/len(momentum)).sum(),(momentum > 1)/len(momentum))
        w = (momentum > 1)/len(momentum)
        cash_w = 1- w.sum()
        print(w, cash_w)
        
        
        # if std.values[0] > self.targetvol:
        #     print("==================", self.targetvol/std)
        #     weights = pd.Series(self.targetvol/std * score, index=selected)
        # else:
        #     weights = pd.Series(1.0, index=selected)
        # weights[self.cash_name] = 1.0 - weights.sum()

        # target.temp['weights'] = weights

        return True

# %%
def strategy_momentum_vol(name, start, data, months=12, lag=pd.DateOffset(days=0), cash_name='현금'):
    s = bt.Strategy(name, 
            [
                bt.algos.RunMonthly(run_on_end_of_period=True),
                bt.algos.RunAfterDate(start),
                bt.algos.SelectAll(),
                # bt.algos.PrintTempData(),
                #-------------------------------------------
                WeighMomentumVolRatio(months, lag, cash_name),
                #-------------------------------------------
                bt.algos.PrintInfo("{now} {temp}"),
                bt.algos.Rebalance()
            ]
    )
    return bt.Backtest(s, data, initial_capital=100000000.0)

# %%
bt모멘텀변동성 = strategy_momentum_vol("모멘텀변동성", start, data, 
                                     12, pd.DateOffset(days=0), 'kbond3y')

# %%
r모멘텀변동성 = bt.run(bt모멘텀변동성)

# %%



