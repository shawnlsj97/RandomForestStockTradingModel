from StrategyLearner import StrategyLearner
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import marketsimcode as msc

def generate_df_orders(df_trades):
    df_orders = pd.DataFrame(index=df_trades.index, columns=["Symbol", "Order", "Shares"])
    df_orders["Symbol"] = df_trades.columns[0]
    df_orders["Order"] = np.where(df_trades > 0, "BUY", np.where(df_trades < 0, "SELL", np.nan))
    df_orders["Shares"] = np.abs(df_trades)
    return df_orders

def generate_df_benchmark_orders(df_trades):
    df_benchmark_orders = pd.DataFrame(index=df_trades.index, columns=["Symbol", "Order", "Shares"])
    df_benchmark_orders["Symbol"] = df_trades.columns[0]
    df_benchmark_orders["Order"] = np.nan
    df_benchmark_orders.loc[df_benchmark_orders.index[0], "Order"] = "BUY"
    df_benchmark_orders["Shares"] = 0
    df_benchmark_orders.loc[df_benchmark_orders.index[0], "Shares"] = 1000
    return df_benchmark_orders

def generate_port_charts(title, fileName, strategy_portvals_1, strategy_portvals_2, strategy_portvals_3):
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(strategy_portvals_1, label="Impact = 0.0005", color="red")
    plt.plot(strategy_portvals_2, label="Impact = 0.005", color="purple")
    plt.plot(strategy_portvals_3, label="Impact = 0.05", color="blue")

    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title(title)
    plt.legend()
    plt.savefig("./images/" + fileName + ".png")

def generate_num_trades(impacts, num_trades, strategy_portvals_1, strategy_portvals_2, strategy_portvals_3):
    cum_ret_1, avg_daily_ret_1, std_daily_ret_1, sharpe_ratio_1 = msc.get_portfolio_stats(
        strategy_portvals_1)
    cum_ret_2, avg_daily_ret_2, std_daily_ret_2, sharpe_ratio_2 = msc.get_portfolio_stats(
        strategy_portvals_2)
    cum_ret_3, avg_daily_ret_3, std_daily_ret_3, sharpe_ratio_3 = msc.get_portfolio_stats(
        strategy_portvals_3)

    results = {
        "Market Impacts": impacts,
        "Number of Trades": num_trades,
        "Cumulative Return": [cum_ret_1, cum_ret_2, cum_ret_3],
        "Average Daily Return": [avg_daily_ret_1, avg_daily_ret_2, avg_daily_ret_3],
        "Standard Deviation of Daily Return": [std_daily_ret_1, std_daily_ret_2, std_daily_ret_3],
        "Sharpe Ratio": [sharpe_ratio_1, sharpe_ratio_2, sharpe_ratio_3]
    }
    results_df = pd.DataFrame(results).round(6)
    results_df.to_csv("Market Impact - In-Sample for JPM" + ".txt", sep="\t", index=False, float_format="%.6f")


def plot_experiment_2(title, fileName):
    impact1 = 0.0005
    impact2 = 0.005
    impact3 = 0.05
    commission = 0
    start_val = 100000

    # ====================== StrategyLearner1 ======================
    strategyLearner1 = StrategyLearner(verbose = False, impact=impact1, commission=commission)
    strategyLearner1.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = start_val)
    df_trades_strategyLearner1 = strategyLearner1.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = start_val)
    df_orders_strategy1 = generate_df_orders(df_trades_strategyLearner1)
    strategy_portvals_1 = msc.compute_portvals(df_orders_strategy1, start_val=start_val,
                                                     commission=commission, impact=impact1)
    strategy_portvals_1 = strategy_portvals_1[strategy_portvals_1.columns[0]]
    strategy_portvals_1 = strategy_portvals_1 / strategy_portvals_1[0]  # normalize
    num_trades_1 = (df_trades_strategyLearner1.iloc[:,0] != 0).sum()

    # ====================== StrategyLearner2 ======================
    strategyLearner2 = StrategyLearner(verbose = False, impact=impact2, commission=commission)
    strategyLearner2.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = start_val)
    df_trades_strategyLearner2 = strategyLearner2.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = start_val)
    df_orders_strategy2 = generate_df_orders(df_trades_strategyLearner2)
    strategy_portvals_2 = msc.compute_portvals(df_orders_strategy2, start_val=start_val,
                                                     commission=commission, impact=impact2)
    strategy_portvals_2 = strategy_portvals_2[strategy_portvals_2.columns[0]]
    strategy_portvals_2 = strategy_portvals_2 / strategy_portvals_2[0]  # normalize
    num_trades_2 = (df_trades_strategyLearner2.iloc[:, 0] != 0).sum()

    # ====================== StrategyLearner3 ======================
    strategyLearner3 = StrategyLearner(verbose = False, impact=impact3, commission=commission)
    strategyLearner3.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = start_val)
    df_trades_strategyLearner3 = strategyLearner3.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = start_val)
    df_orders_strategy3 = generate_df_orders(df_trades_strategyLearner3)
    strategy_portvals_3 = msc.compute_portvals(df_orders_strategy3, start_val=start_val,
                                                     commission=commission, impact=impact3)
    strategy_portvals_3 = strategy_portvals_3[strategy_portvals_3.columns[0]]
    strategy_portvals_3 = strategy_portvals_3 / strategy_portvals_3[0]  # normalize
    num_trades_3 = (df_trades_strategyLearner3.iloc[:, 0] != 0).sum()

    generate_port_charts(title, fileName, strategy_portvals_1, strategy_portvals_2, strategy_portvals_3)
    generate_num_trades([impact1, impact2, impact3], [num_trades_1, num_trades_2, num_trades_3], strategy_portvals_1, strategy_portvals_2, strategy_portvals_3)



if __name__ == "__main__":
    pass
