import numpy as np
import pandas as pd
from util import get_data


def get_portfolio_value(prices, allocs, start_val=1):
    normed = prices / prices.iloc[0]
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    return port_val


def get_portfolio_stats(port_val, daily_risk_free_ret=0.0, annual_sample_freq=252):
    daily_ret = (port_val / port_val.shift(1)) - 1
    daily_ret.iloc[0] = 0  # set first day return to 0
    daily_ret = daily_ret[1:]  # exclude first day return (0) from avg and std calculations
    cum_ret = (port_val[-1] / port_val[0]) - 1
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    # Handle cases where std_daily_ret is 0 or NaN
    if std_daily_ret == 0 or np.isnan(std_daily_ret):
        sharpe_ratio = 0  # Define Sharpe Ratio as 0 in these cases
    else:
        sharpe_ratio_adjustment_factor = np.sqrt(annual_sample_freq)
        sharpe_ratio = sharpe_ratio_adjustment_factor * (avg_daily_ret - daily_risk_free_ret) / std_daily_ret
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


# Solution based on lecture found at: https://www.youtube.com/watch?v=1ysZptg2Ypk&ab_channel=TuckerBalch
# Construct a series of intermediate dataframes to compute daily portfolio value from prices and orders
def construct_df_prices(symbols, dates):
    df_prices = get_data(symbols, dates).drop(columns=["SPY"])  # Get price data for trading days only
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    df_prices["CASH"] = 1.0  # Add cash column with same value of 1.0 for entire column
    return df_prices


def construct_df_trades(df_prices, orders, commission, impact):
    df_trades = pd.DataFrame(data=0, index=df_prices.index, columns=df_prices.columns)
    # Iterate through orders and encode multiple trades on same day into 1 row in df_trades
    for date, order in orders.iterrows():
        symbol = order["Symbol"]
        order_type = order["Order"]
        shares = order["Shares"]
        # Account for slippage and commission when updating CASH column
        if order_type == "BUY":
            df_trades.loc[date, symbol] += shares
            df_trades.loc[date, "CASH"] -= (df_prices.loc[date, symbol] * shares) * (1 + impact) + commission
        elif order_type == "SELL":
            df_trades.loc[date, symbol] -= shares
            df_trades.loc[date, "CASH"] += (df_prices.loc[date, symbol] * shares) * (1 - impact) - commission
    return df_trades


def construct_df_holdings(df_trades, start_val):
    df_holdings = df_trades.cumsum()
    df_holdings["CASH"] += start_val
    return df_holdings


def construct_df_portvals(df_value):
    df_portvals = pd.DataFrame(df_value.sum(axis=1), columns=["Portfolio Value"])
    return df_portvals


def compute_portvals(
        orders,
        start_val=100000,
        commission=0,
        impact=0,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe indexed by date, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    symbols = list(set(orders["Symbol"]))  # Get symbols from orders

    # Get date range from orders
    start_date = orders.index.min()
    end_date = orders.index.max()
    dates = pd.date_range(start_date, end_date)

    # Construct df_prices
    df_prices = construct_df_prices(symbols, dates)

    # Construct df_trades
    df_trades = construct_df_trades(df_prices, orders, commission, impact)

    # Construct df_holdings
    df_holdings = construct_df_holdings(df_trades, start_val)

    # Construct df_value
    df_value = df_holdings * df_prices

    # Construct df_portvals
    df_portvals = construct_df_portvals(df_value)

    return df_portvals