from pathlib import Path
import numpy as np
import pandas as pd
import bt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

def get_data() -> pd.DataFrame:
    return pd.read_csv(f'{get_root()}/data/asset_db.csv', index_col=0, parse_dates=True)

def get_root() -> Path:
    return Path(__file__).parent.parent

def plot_correlations(df, figsize=(11,9)):
    corr = df.to_returns().dropna().corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='coolwarm')

def plot_df(df, figsize=(12,9), title="", legend_loc="best", legend_ordered=True, logy=True):
    columns = df.iloc[-1].sort_values(ascending=False).index #마지막값 높은 순서로
    # print(columns)
    df = df[columns] # 데이터프레임 열 순서를 변경
    ax = df.plot(figsize=(12,8), title=title, logy=logy)
    leg = ax.legend(loc=legend_loc)
    if legend_ordered:
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color()) # 범례의 글씨 색깔을 범례와 동일하게

def plot_assets(backtest_result, start, end, strategy_name, **kwargs):
    res1 = backtest_result
    start = start
    end = end
    plt.rcParams["figure.figsize"] = [16, 12]
    plt.subplots_adjust(hspace=0)

    color_dict = kwargs.pop('color_dict', None)

    # 첫번째 칸에 그림을 그린다.
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    # 두개를 한 칸에 그리기 위해 ax=ax1으로 axis공유
    if color_dict:
        color = [color_dict.get(x, "#333333") for x in res1.prices.columns]
        ax2 = res1.prices[start:end].rebase(1).plot(ax=ax1, lw=1, color=color, logy=True, **kwargs) # 모든 데이터 r_all
    else:
        ax2 = res1.prices[start:end].rebase(1).plot(ax=ax1, lw=1, logy=True, **kwargs) # 모든 데이터 r_all
    for line in ax2.get_lines():
        if line.get_label() == strategy_name:
            line.set_linewidth(3)
    plt.legend(loc="upper left");
    plt.title(strategy_name, fontsize=20)
    if color_dict:
        color = [color_dict.get(x, "#333333") for x in res1.get_security_weights(strategy_name).columns]
        res1.get_security_weights(strategy_name)[start:end].plot.area(alpha=0.2, ax=ax1, color=color, secondary_y=True, **kwargs)
    else:
        res1.get_security_weights(strategy_name)[start:end].plot.area(alpha=0.2, ax=ax1, secondary_y=True, **kwargs)


    # 두번째 칸에 그림을 그린다.
    # drawdown을 그림다. 두개를 하나에 그리기 위해 ax=ax2로 axis를 공유
    ax2 = plt.subplot2grid((3,1), (2,0))
    if color_dict:
        color = [color_dict.get(x, "#333333") for x in res1.prices.columns]
        res1.prices[start:end].to_drawdown_series().plot.area(stacked=False, color=color, legend=True, ax=ax2, **kwargs)
    else:
        res1.prices[start:end].to_drawdown_series().plot.area(stacked=False,legend=True, ax=ax2, **kwargs)
    res1.prices.loc[start:end,strategy_name].to_drawdown_series().plot(legend=False, color='black', alpha=1, lw=1, ls='-', ax=ax2)

def 투자진입시점별CAGRMDD(backtest, title, miny=None, maxy=None):
    """

    Args:
        end : 백테스트기간 종료보다 1년전 날짜 설정, cagr이 연률화이기 때문에 1년 미만은 과대한 수치를 보여준다.
    """
    r = bt.run(backtest)
    cagrs = {}
    mdds = {}
    #for m in pd.date_range(start_date, '2021-10', freq='M'):
    # 전수를 하려면 아래, 월별로 하려면 위의 루프를 실행한다.
    for m in r.prices.index:
        # print(m)
        try:
            cagrs[m] = r.prices[m:].calc_cagr().values[0]
            mdds[m] = r.prices[m:].calc_max_drawdown().values[0]
        except:
            print(m)

    cagr_df = pd.DataFrame([cagrs]).T*100
    mdd_df = pd.DataFrame([mdds]).T*100

    tdf = bt.merge(cagr_df, mdd_df)
    tdf.columns = ['cagr', 'mdd']

    end = r.prices.index[-1] - pd.DateOffset(years=1)
    print(end)

    ####### plot
    # tdf[:end].plot(figsize=(12,6)) # area의 경우 cagr negative문제 있음
    df = tdf[:end]
    fig, ax = plt.subplots(figsize=(12,6))
    # split dataframe df into negative only and positive only values
    df_neg, df_pos = df.clip(upper=0), df.clip(lower=0)
    # stacked area plot of positive values
    df_pos.plot.area(ax=ax, stacked=True, linewidth=0.)
    # reset the color cycle
    ax.set_prop_cycle(None)
    # stacked area plot of negative values, prepend column names with '_' such that they don't appear in the legend
    df_neg.rename(columns=lambda x: '_' + x).plot.area(ax=ax, stacked=True, linewidth=0.)
    # rescale the y axis
    if miny:
        ax.set_ylim([miny, maxy])
    else:
        ax.set_ylim([df_neg.sum(axis=1).min(), df_pos.sum(axis=1).max()])
        print(df_neg.sum(axis=1).min(), df_pos.sum(axis=1).max())
    plt.title(title, size=20)

    return tdf

# 2022.1.3 by hosung ffn의 to_ulcer_index() 계산이 틀린 듯
def ulcer_index(prices):
    dd = prices.to_drawdown_series()
    # return np.divide(np.sqrt(np.sum(np.power(dd, 2))), dd.count())
    return np.sqrt(np.divide(np.sum(np.power(dd, 2)),dd.count()))

def upi(prices, rf=0.0, nperiods=None):
    # if type(rf) is float and rf != 0 and nperiods is None:
    if isinstance(rf, float) and rf != 0 and nperiods is None:
        raise Exception("nperiods must be set if rf != 0 and rf is not a price series")

    er = prices.to_returns().to_excess_returns(rf, nperiods=nperiods)
    # print(er.mean())

    return np.divide(er.mean(), ulcer_index(prices))
