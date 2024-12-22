import datetime as dt
import pandas as pd
from util import get_data
import matplotlib.pyplot as plt


def plot_bollinger_bands(df, bbp, sma, upper_band, lower_band, window, num_std):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex='all')
    stock_name = df.columns[0]

    # Upper plot: Stock price with Bollinger Bands
    ax[0].plot(df, label=f'{stock_name} Price')
    ax[0].plot(sma, label=f'SMA ({window})', linestyle='--')
    ax[0].plot(upper_band, label=f'Upper Band ({window}, {num_std})', linestyle='--', color='r')
    ax[0].plot(lower_band, label=f'Lower Band ({window}, {num_std})', linestyle='--', color='g')
    ax[0].fill_between(df.index, lower_band.squeeze(), upper_band.squeeze(), color='gray', alpha=0.2)
    ax[0].set_title(f'Bollinger Bands for {stock_name}')
    ax[0].set_ylabel('Price')
    ax[0].legend()

    # Lower plot: Bollinger Band Percent (BBP)
    ax[1].plot(bbp, label='BBP')
    ax[1].set_title('Bollinger Band Percent (BBP)')
    ax[1].set_ylabel('BBP')
    ax[1].axhline(0, color='g', linestyle='--')
    ax[1].axhline(1, color='r', linestyle='--')
    ax[1].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig("./images/bbp.png")
    plt.close(fig)

def bollinger_bands(df, window=20, num_std=2):
    """
    Overbought: BBP > 1: , Oversold: BBP < 0
    :return: BBP
    """
    sma = df.rolling(window=window, min_periods=window).mean()
    std = df.rolling(window=window, min_periods=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    bbp = (df - lower_band) / (upper_band - lower_band)

    # plot_bollinger_bands(df, bbp, sma, upper_band, lower_band, window, num_std)

    return bbp

def plot_rsi(df, rsi, window):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    stock_name = df.columns[0]

    # Upper plot: Stock price
    ax[0].plot(df, label=f'{stock_name} Price')
    ax[0].set_title(f'Stock Price for {stock_name}')
    ax[0].set_ylabel('Price')
    ax[0].legend()

    # Lower plot: RSI
    ax[1].plot(rsi, label=f'RSI ({window})', color='orange')
    ax[1].axhline(30, linestyle='--', color='r', label='Oversold (30)')
    ax[1].axhline(70, linestyle='--', color='g', label='Overbought (70)')
    ax[1].set_title('Relative Strength Index (RSI)')
    ax[1].set_ylabel('RSI')
    ax[1].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig("./images/rsi.png")
    plt.close(fig)

def rsi(df, window=14):
    """
    Overbought: RSI > 70, Oversold: RSI < 70
    """
    # Compute change in price
    delta = df.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # plot_rsi(df, rsi, window)

    return rsi

def plot_macd(df, macd_histogram, macd_line, signal_line):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    stock_name = df.columns[0]

    # Upper plot: Stock price
    ax[0].plot(df, label=f'{stock_name} Price')
    ax[0].set_title(f'Stock Price for {stock_name}')
    ax[0].set_ylabel('Price')
    ax[0].legend()

    # Lower plot: MACD
    ax[1].plot(macd_line, label='MACD Line', color='blue')
    ax[1].plot(signal_line, label='Signal Line', color='orange')
    ax[1].plot(macd_histogram, label='MACD Histogram', color='red')
    ax[1].axhline(0, linestyle='--', color='black')
    ax[1].set_title('MACD Indicator')
    ax[1].set_ylabel('MACD')
    ax[1].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig("./images/macd.png")
    plt.close(fig)


def macd(df, short_window=12, long_window=26, signal_window=9):
    """
    MACD Line = EMA(12) - EMA(26)
    Signal Line = EMA(9) of MACD Line
    MACD Histogram = MACD Line - Signal Line
    Usage:
    -Bullish: MACD Histogram > 0 (MACD line cross above signal line)
    -Bearish: MACD Histogram < 0 (MACD line cross below signal line)
    """
    short_ema = df.ewm(span=short_window).mean()
    long_ema = df.ewm(span=long_window).mean()

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window).mean()

    macd_histogram = macd_line - signal_line

    # plot_macd(df, macd_histogram, macd_line, signal_line)

    return macd_histogram

def plot_stochastic_oscillator(df, k, d):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    stock_name = df.columns[0]

    # Upper plot: Stock price
    ax[0].plot(df, label=f'{stock_name} Price')
    ax[0].set_title(f'Stock Price for {stock_name}')
    ax[0].set_ylabel('Price')
    ax[0].legend()

    # Lower plot: Stochastic Oscillator
    ax[1].plot(k, label='%K', color='blue')
    ax[1].plot(d, label='%D', color='orange')
    ax[1].axhline(20, linestyle='--', color='r', label='Oversold (20)')
    ax[1].axhline(80, linestyle='--', color='g', label='Overbought (80)')
    ax[1].set_title('Stochastic Oscillator')
    ax[1].set_ylabel('%K, %D')
    ax[1].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig("./images/stoch.png")
    plt.close(fig)

def stochastic_oscillator(df_prices, symbol, dates, k_window=14, d_window=3):
    """
    Compares closing price to price range over specified window.
`   %K: Measures closing price relative to range of prices over specified lookback period
    %D: Signal line for stoch oscillator. Calculated as a moving average of %K over specified period

    Bullish: %D cross above 20
    Bearish: %D cross below 80
    """
    df_close = get_data([symbol], dates, colname="Close").drop(columns=["SPY"])
    df_adjustment = df_prices / df_close
    df_high = get_data([symbol], dates, colname="High").drop(columns=["SPY"]) * df_adjustment
    df_low = get_data([symbol], dates, colname="Low").drop(columns=["SPY"]) * df_adjustment

    low_min = df_low.rolling(window=k_window).min()
    high_max = df_high.rolling(window=k_window).max()

    k = 100 * (df_prices - low_min) / (high_max - low_min)
    d = k.rolling(window=d_window).mean()

    # plot_stochastic_oscillator(df_prices, k, d)

    return d

def plot_momentum(df, mom, period):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex='all')
    stock_name = df.columns[0]

    # Upper plot: Stock price
    ax[0].plot(df, label=f'{stock_name} Price')
    ax[0].set_title(f'Stock Price for {stock_name}')
    ax[0].set_ylabel('Price')
    ax[0].legend()

    # Lower plot: Momentum
    ax[1].plot(mom, label=f'Momentum ({period})', color='purple')
    ax[1].axhline(0, linestyle='--', color='black')
    ax[1].set_title('Momentum')
    ax[1].set_ylabel('Momentum')
    ax[1].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig("./images/mom.png")
    plt.close(fig)

def momentum(df, period=14):
    """
    Buy signal: momentum cross above 0
    Sell signal: momentum cross below 0
    """
    mom = df.diff(periods=period)
    # plot_momentum(df, mom, period)

    return mom

def run(symbol, sd, ed):
    # Retrieve prices
    dates = pd.date_range(sd, ed)
    df_prices = get_data([symbol], dates).drop(columns=["SPY"])  # Get price data for trading days only
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)

    # Call all signals and plot charts
    bollinger_bands(df_prices)
    rsi(df_prices)
    macd(df_prices)
    stochastic_oscillator(df_prices, symbol, dates)
    momentum(df_prices)

if __name__ == "__main__":
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    run(symbol="JPM", sd=sd, ed=ed)