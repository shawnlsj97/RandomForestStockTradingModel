import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import indicators as id
import RandomTreeLearner as rt
import BootstrapAggregationLearner as bl
import marketsimcode as msc


class StrategyLearner(object):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
        self.learner = bl.BootstrapAggregationLearner(learner = rt.RandomTreeLearner, kwargs={"leaf_size":5}, bags=20)

    def generate_df_indicators(self, prices, sd, ed, symbol):
        syms = [symbol]
        extended_dates = pd.date_range(sd - pd.Timedelta(days=30),
                                       ed)  # Load 30 extra days so that indicators will not start off with nan values due to training window
        extended_prices = ut.get_data(syms, extended_dates)[syms]
        bbp = id.bollinger_bands(extended_prices).rename(columns={symbol:'BBP'})
        rsi = id.rsi(extended_prices).rename(columns={symbol:'RSI'})
        stoch_d = id.stochastic_oscillator(extended_prices, symbol, extended_dates).rename(columns={symbol:'STOCH_D'})

        indicators = pd.concat([bbp, rsi, stoch_d], axis=1).loc[prices.index]  # slice away extra days

        return indicators

    def generate_df_orders(self, df_trades):
        df_orders = pd.DataFrame(index=df_trades.index, columns=["Symbol", "Order", "Shares"])
        df_orders["Symbol"] = df_trades.columns[0]
        df_orders["Order"] = np.where(df_trades > 0, "BUY", np.where(df_trades < 0, "SELL", np.nan))
        df_orders["Shares"] = np.abs(df_trades)
        return df_orders
  		  	   		 	   		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		 	   		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		 	   		  		  		    	 		 		   		 		  
        self,  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		 	   		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
        sv=100000,
    ):  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
        # add your code to do learning here
        syms = [symbol]  		  	   		 	   		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		 	   		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		 	   		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols
        N = 5
        if self.verbose:  		  	   		 	   		  		  		    	 		 		   		 		  
            print(prices)

        df_indicators = self.generate_df_indicators(prices, sd, ed, symbol)

        # Construct trainX
        df_indicators = df_indicators[:-N]
        trainX = df_indicators.values

        # Construct trainY
        trainY = []
        for i in range(len(prices)-N):
            N_day_returns = (prices.loc[prices.index[i+N], symbol] - prices.loc[prices.index[i], symbol]) / prices.loc[prices.index[i], symbol]
            if N_day_returns > (0.032 + self.impact):
                trainY.append(1) # BUY
            elif N_day_returns < (-0.032 - self.impact):
                trainY.append(-1) # SELL
            else:
                trainY.append(0) # NOTHING
        trainY = np.array(trainY)

        self.learner.add_evidence(trainX, trainY)

    # this method should use the existing policy and test it against new data  		  	   		 	   		  		  		    	 		 		   		 		  
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

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        if self.verbose:
            print(prices)

        df_indicators = self.generate_df_indicators(prices, sd, ed, symbol)
        testX = df_indicators.values

        testY = self.learner.query(testX)

        trades = pd.DataFrame(data=0.0, index=prices.index, columns=[symbol])
        holdings = 0
        for i in range(len(prices)):
            trade = 0
            if testY[i] == 0:  # NOTHING
                continue
                # if holdings == 1000:
                #     trade = -1000  # LONG to CASH
                # elif holdings == -1000:
                #     trade = 1000  # SHORT to CASH
            elif testY[i] > 0.5:  # LONG
                if holdings == -1000:
                    trade = 2000  # SHORT to LONG
                elif holdings == 0:
                    trade = 1000  # CASH to LONG
            elif testY[i] < -0.5:  # SHORT
                if holdings == 1000:
                    trade = -2000  # LONG to SHORT
                elif holdings == 0:
                    trade = -1000  # CASH to SHORT

            holdings += trade
            trades.loc[prices.index[i], symbol] = trade

        df_orders = self.generate_df_orders(trades)
        df_portvals = msc.compute_portvals(df_orders, start_val=sv, commission=self.commission, impact=self.impact)
        port_val = df_portvals.iloc[-1].item()

        if self.verbose:  		  	   		 	   		  		  		    	 		 		   		 		  
            print(type(trades))  # it better be a DataFrame!
            print(trades)
            print(prices_all)
            print("Strategy Learner Port Val: " + str(port_val))
        return trades
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		  	   		 	   		  		  		    	 		 		   		 		  
