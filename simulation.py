import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def portfolio():
    portfolio_composition = [('INTC', 0.25), ('IBM', 0.25), ('DIS', 0.25), ('MCD', 0.25)]
    returns = pd.DataFrame({})
    for t in portfolio_composition:
        name = t[0]
        ticker = yfinance.Ticker(name)
        data = ticker.history(interval="1d",
                              start="1980-01-01", end="2010-12-31")

        data['return_%s' % name] = data['Close'].pct_change(1) * 100
        returns = returns.join(data[['return_%s' % name]],
                               how="outer").dropna()
    return returns


if __name__ == '__main__':
    data = portfolio()
    # print(data)

    # plt.hist(data["return_INTC"], bins=100)
    # plt.title('INTC')
    # plt.show()
    #
    # plt.hist(data["return_IBM"], bins=100)
    # plt.title('IBM')
    # plt.show()
    #
    # plt.hist(data["return_DIS"], bins=100)
    # plt.title('DIS')
    # plt.show()
    #
    # plt.hist(data["return_MCD"], bins=100)
    # plt.title('MCD')
    # plt.show()

    # print(data["return_INTC"].describe())
    # print(data["return_IBM"].describe())
    # print(data["return_DIS"].describe())
    # print(data["return_MCD"].describe())

    # data["return_INTC"].corr(data["return_INTC"])
    trans = data["return_INTC"].T
    print(data["return_INTC"].corr(trans))
