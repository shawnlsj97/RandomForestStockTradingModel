import datetime as dt
import numpy as np
import pandas as pd
import util as ut
import indicators as id
import matplotlib.pyplot as plt
import marketsimcode as msc


class ManualStrategy(object):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    A manual learner that can learn (essentially human coded rules) a trading policy using the same indicators
    used in StrategyLearner.

    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    """

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

        # this method should create a QLearner, and train it for trading

    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=100000,
    ):
        # add evidence is used here for consistency with the Strategy Learner. In actuality, it is likely that
        # this function does nothing as the "rules" will have been coded by the developer of the class. If that is the case,
        # this method might simply consist of a "pass" statement.
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on  		  	   		 	   		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        pass

    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=100000,
    ):
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		 	   		  		  		    	 		 		   		 		  

        :param symbol: The stock symbol that you trained on
        :type symbol: str  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
        """

        # here we build a fake set of trades  		  	   		 	   		  		  		    	 		 		   		 		  
        # your code should return the same sort of data  		  	   		 	   		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)
        df_prices = ut.get_data([symbol], dates).drop(columns=["SPY"])
        df_prices.fillna(method="ffill", inplace=True)
        df_prices.fillna(method="bfill", inplace=True)
        df_trades = pd.DataFrame(data=0.0, index=df_prices.index, columns=[symbol])

        extended_dates = pd.date_range(sd - pd.Timedelta(days=30), ed)  # Load 30 extra days so that indicators will not start off with nan values due to training window
        extended_prices = ut.get_data([symbol], extended_dates).drop(columns=["SPY"])

        # Shift indicators by one day to use yesterday's indicator data for today’s trades
        bbp = id.bollinger_bands(extended_prices).shift(1).loc[df_prices.index].iloc[:, 0]
        rsi = id.rsi(extended_prices).shift(1).loc[df_prices.index].iloc[:, 0]
        stoch_d = id.stochastic_oscillator(extended_prices, symbol, extended_dates).shift(1).loc[df_prices.index].iloc[:, 0]

        position = 0
        for i in range(len(df_prices)):
            if position == 0: # Out of market
                if bbp[i] < 0.2 and rsi[i] < 35 and stoch_d[i] < 20: # Enter long
                    df_trades[symbol][i] = 1000
                    position = 1000
                elif bbp[i] > 0.8 and rsi[i] > 65 and stoch_d[i] > 80: # Enter short
                    df_trades[symbol][i] = -1000
                    position = -1000
            elif position == 1000: # Currently long
                if bbp[i] > 0.9 or rsi[i] > 80 or stoch_d[i] > 90: # Flip short
                    df_trades[symbol][i] = -2000
                    position = -1000
                elif bbp[i] > 0.8 or rsi[i] > 65 or stoch_d[i] > 80: # Close long
                    df_trades[symbol][i] = -1000
                    position = 0
            elif position == -1000: # Currently short
                if bbp[i] < 0.1 or rsi[i] < 20 or stoch_d[i] < 10: # Flip long
                    df_trades[symbol][i] = 2000
                    position = 1000
                elif bbp[i] < 0.2 or rsi[i] < 35 or stoch_d[i] < 20:  # Close short
                    df_trades[symbol][i] = 1000
                    position = 0

        if self.verbose:
            print(type(df_trades))  # it better be a DataFrame!
        if self.verbose:
            print(df_trades)
        if self.verbose:
            print(df_prices)
        return df_trades

    def generate_port_stats(self, manual_portvals, benchmark_portvals, title):
        cum_ret_manual, avg_daily_ret_manual, std_daily_ret_manual, sharpe_ratio_manual = msc.get_portfolio_stats(manual_portvals)
        cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark, sharpe_ratio_benchmark = msc.get_portfolio_stats(
            benchmark_portvals)

        results = {
            "Metric": ["Cumulative Return", "Standard Deviation", "Mean Daily Return"],
            "Manual Strategy": [cum_ret_manual, std_daily_ret_manual, avg_daily_ret_manual],
            "Benchmark": [cum_ret_benchmark, std_daily_ret_benchmark, avg_daily_ret_benchmark],
        }
        results_df = pd.DataFrame(results).round(6)  # Round to 6 decimal places
        results_df.to_csv(title + ".txt", sep="\t", index=False, float_format="%.6f")

    def generate_port_charts(self, manual_portvals, benchmark_portvals, df_trades, title, fileName):
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(manual_portvals, label="Manual Strategy (Red)", color="red")
        plt.plot(benchmark_portvals, label="Benchmark (Purple)", color="purple")

        df_holdings = df_trades.cumsum()
        # Add markers for entry points
        for idx, trade in df_trades.iloc[:,0].iteritems():
            holding = df_holdings.loc[idx].iloc[0]
            if (trade == 1000 and holding == 1000) or (trade == 2000):
                plt.axvline(x=idx, color="blue")
            elif (trade == -1000 and holding == -1000) or (trade == -2000):
                plt.axvline(x=idx, color="black")

        plt.plot([], [], label='Long Entry', color='blue', linestyle='-')
        plt.plot([], [], label='Short Entry', color='black', linestyle='-')

        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.title(title)
        plt.legend()
        plt.savefig("./images/" + fileName + ".png")

    def generate_df_orders(self, df_trades):
        df_orders = pd.DataFrame(index=df_trades.index, columns=["Symbol", "Order", "Shares"])
        df_orders["Symbol"] = df_trades.columns[0]
        df_orders["Order"] = np.where(df_trades > 0, "BUY", np.where(df_trades < 0, "SELL", np.nan))
        df_orders["Shares"] = np.abs(df_trades)
        return df_orders

    def generate_df_benchmark_orders(self, df_trades):
        df_benchmark_orders = pd.DataFrame(index=df_trades.index, columns=["Symbol", "Order", "Shares"])
        df_benchmark_orders["Symbol"] = df_trades.columns[0]
        df_benchmark_orders["Order"] = np.nan
        df_benchmark_orders.loc[df_benchmark_orders.index[0], "Order"] = "BUY"
        df_benchmark_orders["Shares"] = 0
        df_benchmark_orders.loc[df_benchmark_orders.index[0], "Shares"] = 1000
        return df_benchmark_orders

    def run(self):
        # ---------- In Sample ---------- #
        sd_in_sample = dt.datetime(2008, 1, 1)
        ed_in_sample = dt.datetime(2009, 12, 31)
        df_trades_manual_in_sample = self.testPolicy("JPM", sd_in_sample, ed_in_sample, 100000)

        # Create df_orders from df_trades that works with marketsim
        df_orders_manual_in_sample = self.generate_df_orders(df_trades_manual_in_sample)

        manual_portvals_in_sample = msc.compute_portvals(df_orders_manual_in_sample, start_val=100000, commission=self.commission, impact = self.impact)
        manual_portvals_in_sample = manual_portvals_in_sample[manual_portvals_in_sample.columns[0]]
        manual_portvals_in_sample = manual_portvals_in_sample / manual_portvals_in_sample[0]  # normalize

        df_benchmark_orders_in_sample = self.generate_df_benchmark_orders(df_trades_manual_in_sample)

        benchmark_portvals_in_sample = msc.compute_portvals(df_benchmark_orders_in_sample, start_val=100000, commission=self.commission, impact = self.impact)
        benchmark_portvals_in_sample = benchmark_portvals_in_sample[benchmark_portvals_in_sample.columns[0]]
        benchmark_portvals_in_sample = benchmark_portvals_in_sample / benchmark_portvals_in_sample[0]

        self.generate_port_charts(manual_portvals_in_sample, benchmark_portvals_in_sample, df_trades_manual_in_sample, "Manual Strategy - In-Sample for JPM", "manual_strategy_in_sample")
        self.generate_port_stats(manual_portvals_in_sample, benchmark_portvals_in_sample, "Performance - In-Sample for JPM")

        # ---------- Out of Sample ---------- #
        sd_out_of_sample = dt.datetime(2010, 1, 1)
        ed_out_of_sample = dt.datetime(2011, 12, 31)
        df_trades_manual_out_of_sample = self.testPolicy("JPM", sd_out_of_sample, ed_out_of_sample, 100000)
        df_orders_manual_out_of_sample = self.generate_df_orders(df_trades_manual_out_of_sample)

        manual_portvals_out_of_sample = msc.compute_portvals(df_orders_manual_out_of_sample, start_val=100000, commission=self.commission,
                                               impact=self.impact)
        manual_portvals_out_of_sample = manual_portvals_out_of_sample[manual_portvals_out_of_sample.columns[0]]
        manual_portvals_out_of_sample = manual_portvals_out_of_sample / manual_portvals_out_of_sample[0]  # normalize

        df_benchmark_orders_out_of_sample = self.generate_df_benchmark_orders(df_trades_manual_out_of_sample)

        benchmark_portvals_out_of_sample = msc.compute_portvals(df_benchmark_orders_out_of_sample, start_val=100000, commission=self.commission,
                                                  impact=self.impact)
        benchmark_portvals_out_of_sample = benchmark_portvals_out_of_sample[benchmark_portvals_out_of_sample.columns[0]]
        benchmark_portvals_out_of_sample = benchmark_portvals_out_of_sample / benchmark_portvals_out_of_sample[0]

        self.generate_port_charts(manual_portvals_out_of_sample, benchmark_portvals_out_of_sample, df_trades_manual_out_of_sample,
                                  "Manual Strategy - Out-of-Sample for JPM", "manual_strategy_out_of_sample")
        self.generate_port_stats(manual_portvals_out_of_sample, benchmark_portvals_out_of_sample,
                                 "Performance - Out-of-Sample for JPM")


if __name__ == "__main__":
    print("One does not simply think up a strategy")  		  	   		 	   		  		  		    	 		 		   		 		  
