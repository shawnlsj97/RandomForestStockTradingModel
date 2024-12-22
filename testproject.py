from ManualStrategy import ManualStrategy
import datetime as dt
import numpy as np
import experiment1 as exp1
import experiment2 as exp2


def gtid():
    return 903997563

def main():
    np.random.seed(gtid())  # do this only once
    manualStrategy = ManualStrategy(False, impact=0.005, commission=9.95)
    manualStrategy.run()

    exp1.plot_experiment_1(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), title="In-Sample for JPM", fileName="rf_vs_manual_in_sample")
    exp1.plot_experiment_1(sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), title="Out-of-Sample for JPM", fileName="rf_vs_manual_out_of_sample")

    exp2.plot_experiment_2(title="Strategy Learner with Different Impacts - In-Sample for JPM", fileName="rf_market_impact_in_sample")


if __name__ == "__main__":
    main()
