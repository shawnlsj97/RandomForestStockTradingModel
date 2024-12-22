from ManualStrategy import ManualStrategy
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

def generate_port_charts(title, fileName, manual_portvals, strategy_portvals, benchmark_portvals):
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(manual_portvals, label="Manual Strategy (Red)", color="red")
    plt.plot(benchmark_portvals, label="Benchmark (Purple)", color="purple")
    plt.plot(strategy_portvals, label="Strategy Learner (Blue)", color="blue")

    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title(title)
    plt.legend()
    plt.savefig("./images/" + fileName + ".png")

def plot_experiment_1(sd, ed, title, fileName):
    impact = 0.005
    commission = 9.95
    start_val = 100000
    # ====================== MANUAL STRATEGY ======================
    manualStrategy = ManualStrategy(False, impact=impact, commission=commission)
    df_trades_manual = manualStrategy.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = start_val)
    df_orders_manual = generate_df_orders(df_trades_manual)
    manual_portvals = msc.compute_portvals(df_orders_manual, start_val=start_val, commission=commission, impact=impact)
    manual_portvals = manual_portvals[manual_portvals.columns[0]]
    manual_portvals = manual_portvals / manual_portvals[0]  # normalize

    # ====================== StrategyLearner ======================
    strategyLearner = StrategyLearner(verbose = False, impact=impact, commission=commission)
    strategyLearner.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = start_val)
    df_trades_strategyLearner = strategyLearner.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = start_val)
    df_orders_strategy = generate_df_orders(df_trades_strategyLearner)
    strategy_portvals = msc.compute_portvals(df_orders_strategy, start_val=start_val, commission=commission, impact=impact)
    strategy_portvals = strategy_portvals[strategy_portvals.columns[0]]
    strategy_portvals = strategy_portvals / strategy_portvals[0]  # normalize

    # ====================== BENCHMARK ======================
    df_benchmark_orders = generate_df_benchmark_orders(df_trades_manual)
    benchmark_portvals = msc.compute_portvals(df_benchmark_orders, start_val=start_val, commission=commission, impact=impact)
    benchmark_portvals = benchmark_portvals[benchmark_portvals.columns[0]]
    benchmark_portvals = benchmark_portvals / benchmark_portvals[0]

    generate_port_charts(title, fileName, manual_portvals, strategy_portvals, benchmark_portvals)



if __name__ == "__main__":
    plot_experiment_1(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31))
