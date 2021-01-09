import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feather
from os import path
from pandas_datareader import data as pdr
import json
import datetime as dt
from scipy.stats import norm

DATA_FILE = 'files/data.csv'
RETURNS_FILE = 'files/returns.csv'


stock_names = {
    'INTC': 'Intel',
    'IBM': 'IBM',
    'DIS': 'Disney',
    'MCD': 'McDonald\'s'
}

def stocks_graph(returns, location):
    plt.figure(figsize=(14, 7))
    for i in returns.columns.values:
        plt.plot(returns.index, returns[i], lw=2, alpha=0.8, label=i)
    plt.legend(loc=location, fontsize=14)
    plt.ylabel('daily returns')
    plt.show()


def portfolio():
    portfolio_composition = [('INTC', 0.25), ('IBM', 0.25), ('DIS', 0.25), ('MCD', 0.25)]
    tickers = [stock[0] for stock in portfolio_composition]
    data = pdr.get_data_yahoo(tickers, start="1980-03-17", end=dt.date.today())['Close']
    old_data = data.copy(deep=True)
    returns = data.pct_change()
    # stocks_graph(data, 'upper left')
    return old_data, returns


def plot_histograms(column_name, column_data):
    plt.hist(column_data, bins=100)
    plt.title(column_name)
    plt.show()


def load_data():
    if not (path.exists(DATA_FILE) and path.exists(RETURNS_FILE)):
        loaded_data, loaded_returns = portfolio()
        loaded_data.to_csv(DATA_FILE)
        loaded_returns.to_csv(RETURNS_FILE)
    else:
        loaded_data = pd.read_csv(DATA_FILE, index_col=0)
        loaded_returns = pd.read_csv(RETURNS_FILE, index_col=0)

    return loaded_data, loaded_returns


def q_1(data):
    stocks_array = []
    # Plot histogram for each Stock and print the description of it
    for (columnName, columnData) in data.iteritems():
        columnName = stock_names[str(columnName).replace('return_', '')]
        plot_histograms(columnName, columnData)
        stock_dict = {
            'stock': columnName,
            'mean': columnData.describe()['mean'],
            'std': columnData.describe()['std']
        }
        stocks_array.append(stock_dict)

    return stocks_array



def q_3(sent_data):
    # Generate Random Simulations
    mean_returns = sent_data.mean()
    cov_matrix = sent_data.cov()
    # Number of portfolios to simulate
    num_portfolios = 10000
    # Risk free rate (used for Sharpe ratio below)
    # anchored on treasury bond rates
    risk_free_rate = 0.018
    display_simulated_portfolios(sent_data, mean_returns, cov_matrix, num_portfolios, risk_free_rate)


# Define function to calculate returns, volatility
def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    # Given the avg returns, weights of equities calc. the portfolio return
    returns = np.sum(mean_returns * weights) * 252
    # Standard deviation of portfolio (using dot product against covariance, weights)
    # 252 trading days
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    # Initialize array of shape 3 x N to store our results,
    # where N is the number of portfolios we're going to simulate
    results = np.zeros((3,num_portfolios))
    # Array to store the weights of each equity
    weight_array = []
    for i in range(num_portfolios):
        # Randomly assign floats to our 4 equities
        weights = np.random.random(len(mean_returns))
        # Convert the randomized floats to percentages (summing to 100)
        weights /= np.sum(weights)
        # Add to our portfolio weight array
        weight_array.append(weights)
        # Pull the standard deviation, returns from our function above using
        # the weights, mean returns generated in this function
        portfolio_std_dev, portfolio_return = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
        # Store output
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        # Sharpe ratio
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weight_array


def display_simulated_portfolios(sent_data, mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    # pull results, weights from random portfolios
    results, weights = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    # pull the max portfolio Sharpe ratio (3rd element in results array from
    # generate_random_portfolios function)
    max_sharpe_idx = np.argmax(results[2])

    # pull the associated standard deviation, annualized return w/ the max Sharpe ratio
    stdev_portfolio, returns_portfolio = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

    # pull the allocation associated with max Sharpe ratio
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=sent_data.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    print("-" * 100)
    print("Portfolio at maximum Sharpe Ratio\n")
    print("--Returns, volatility--\n")
    print("Annualized Return:", round(returns_portfolio, 2))
    print("Annualized Volatility:", round(stdev_portfolio, 2))

    print("\n")
    print("--Allocation at max Sharpe ratio--\n")
    print(max_sharpe_allocation)
    print("-" * 100)

    plt.figure(figsize=(16, 9))
    # x = volatility, y = annualized return, color mapping = sharpe ratio
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='winter', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    # Mark the portfolio w/ max Sharpe ratio
    plt.scatter(stdev_portfolio, returns_portfolio, marker='x', color='r', s=150, label='Max Sharpe ratio')
    plt.title('Simulated portfolios illustrating efficient frontier')
    plt.xlabel('annualized volatility')
    plt.ylabel('annualized returns')
    plt.legend(labelspacing=1.2)
    plt.show()


def generate_market_linked_returns(sent_data, returns):
    # פקדון מובנה
    # data['linked_market'] = generate_market_linked(data)
    values_of_linked_returns = []
    try:
        stock_36_dict = dict()
        for stock in sent_data.columns:
            stock_36_precentage = sent_data[stock][0] * 1.36
            stock_36_dict[stock] = stock_36_precentage

        stock_linked = dict()
        for column in sent_data.columns:
            max_precent_under_36 = ((sent_data.loc[sent_data[column] < stock_36_dict[column], column].max()) - 1) / (sent_data[column][0]) * 100
            if len(sent_data.loc[sent_data[column] >= stock_36_dict[column]]) > 0:
                stock_linked[column] = 0.02
            else:
                stock_linked[column] = max_precent_under_36

        total_returns = (sum([stock_linked[stock] for stock in stock_linked]))/sent_data.shape[0]
        returns['market_linked'] = np.ones(sent_data.shape[0]) * total_returns
    except Exception as e:
        print(e)

if __name__ == '__main__':

    data, returns = load_data()
    shape = returns.shape[0]
    returns['Bank'] = np.zeros(shape)

    generate_market_linked_returns(data, returns)
    # print(data)
    #################################################
    #################### Q1 #########################
    #################################################
    # all_stocks_dict = q_1(data)
    # print(json.dumps(all_stocks_dict, indent=4))

    #################################################
    #################### Q3 #########################
    #################################################
    # stocks_graph(data, 'lower center')
    q_3(returns)


    # data["return_INTC"].corr(data["return_INTC"])
    # trans = data["return_INTC"].T
    # print(data["return_INTC"].corr(trans))
