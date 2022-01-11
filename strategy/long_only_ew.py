# simple backtest to test long-only allocation
import bt

def long_only_ew(data, tickers, name, initial_capital=100000000.0):
    ''' tickers : must be list
    '''
    s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.PrintDate(),
                           bt.algos.SelectThese(tickers),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
    return bt.Backtest(s, data, initial_capital=initial_capital)

def long_only_ew_run_after(data, tickers, name, run_after, initial_capital=100000000.0):
    ''' tickers : must be list
    '''
    s = bt.Strategy(name, [run_after,
                           bt.algos.RunOnce(),
                           bt.algos.PrintDate(),
                           bt.algos.SelectThese(tickers),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
    return bt.Backtest(s, data, initial_capital=initial_capital)