import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import pandas as pd
import datetime as dt
from matplotlib import style

def portfolio():
    portfolio_composition = [('VMC', 0.25), ('EMR', 0.25), ('CSX', 0.25), ('UNP', 0.25)]
    returns = pd.DataFrame({})
    for t in portfolio_composition:
        name = t[0]
        ticker = yfinance.Ticker(name)
        data = ticker.history(interval="1d",
                              start="1980-01-01", end="2010-12-31")

        data['return_%s' % name] = data['Close'].pct_change()
        returns = returns.join(data[['return_%s' % name]],
                               how="outer").dropna()
    return returns


if __name__ == '__main__':
    data = portfolio()
    print(data)

    VMC = web.get_data_yahoo("VMC",
                                 start="1980-01-01",
                                 end="2010-12-31")

    VMC['Close'].plot()
    plt.xlabel("Date")
    plt.ylabel("Adjusted")
    plt.title("VMC1 Price data")
    plt.show()


    VMC_daily_returns = VMC['Close'].pct_change()
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.plot(VMC_daily_returns)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Percent")
    ax1.set_title("VMC daily returns data")
    plt.show()

    data['return_VMC'].plot()
    plt.xlabel("Date")
    plt.ylabel("Adjusted")
    plt.title("VMC Price data")
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    VMC_daily_returns.plot.hist(bins=60)
    ax1.set_xlabel("Daily returns %")
    ax1.set_ylabel("Percent")
    ax1.set_title("VMC daily returns data")
    ax1.text(-0.35, 200, "Extreme Low\nreturns")
    ax1.text(0.25, 200, "Extreme High\nreturns")
    plt.show()
    plt.hist(data["return_VMC"], bins=100)
    plt.title('VMC')
    plt.show()

    plt.hist(data["return_EMR"], bins=100)
    plt.title('EMR')
    plt.show()

    plt.hist(data["return_CSX"], bins=100)
    plt.title('CSX')
    plt.show()

    plt.hist(data["return_UNP"], bins=100)
    plt.title('UNP')
    plt.show()

    print(data["return_VMC"].describe())
    print(data["return_EMR"].describe())
    print(data["return_CSX"].describe())
    print(data["return_UNP"].describe())

    data["return_INTC"].corr(data["return_INTC"])
    trans = data["return_INTC"].T
    print(data["return_INTC"].corr(trans))

    # num_of_simulations = 100
    # time_slots = 10 #days
    # simulation_df = pd.DataFrame()
    # for i in range(num_of_simulations):
    #     pass

