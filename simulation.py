import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from pandas_datareader import data as pdr
import json
import datetime as dt
import random
import time
from scipy.stats import norm
import seaborn as sns
import scipy.stats

DATA_FILE = 'files/data.csv'
RETURNS_FILE = 'files/returns.csv'


def mean_confidence_interval(array_data, confidence=0.90):
    a = 1.0 * np.array(array_data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


stock_names = {
    'VMC': 'volcan',
    'EMR': 'Emerson Electric ',
    'CSX': 'CSX',
    'UNP': 'Union Pacific'
}


def stocks_graph(returns, location):
    plt.figure(figsize=(14, 7))
    for i in returns.columns.values:
        plt.plot(returns.index, returns[i], lw=2, alpha=0.8, label=i)
    plt.legend(loc=location, fontsize=14)
    plt.ylabel('daily returns')
    plt.show()


def portfolio():
    portfolio_composition = [('VMC', 0.25), ('EMR', 0.25), ('CSX', 0.25), ('UNP', 0.25)]
    tickers = [stock[0] for stock in portfolio_composition]
    data = pdr.get_data_yahoo(tickers, start="1980-03-17", end=dt.date.today())['Adj Close']
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


def q_1(data_to_work: pd.DataFrame):
    stocks_array = []
    # Plot histogram for each Stock and print the description of it
    for (columnName, columnData) in data_to_work.iteritems():
        auto_curr = pd.Series(data_to_work[columnName].values)
        columnName = stock_names[str(columnName).replace('return_', '')]
        plot_histograms(columnName, columnData)
        stock_dict = {
            'stock': columnName,
            'mean': columnData.describe()['mean'],
            'std': columnData.describe()['std'],
            'auto_corr': auto_curr.autocorr()
        }
        stocks_array.append(stock_dict)

    print()
    return stocks_array


def plot_heatmap_cov(data_to_work):
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(data_to_work.cov(),
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f')
    plt.tight_layout()
    plt.show()




def str_time_prop(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def random_date(start, end, prop):
    return str_time_prop(start, end, '%Y-%m-%d', prop)


def generate_random_slot(data_dropNA):
    flag = True
    res = None
    while flag:
        try:
            daterand = random_date("1980-03-17", "2020-03-17", random.random())
            # print(daterand)
            res = data_dropNA[daterand:].head(10)
            if res is not None:
                # print(data_dropNA[daterand:].head(10))
                flag = False
        except:
            continue
    return res


def generate_880_dataframe(returns_dropNA):
    df = pd.DataFrame()
    for i in range(88):
        df = df.append(generate_random_slot(returns_dropNA))

    return df


def q_2(returns_dropNA, Q='2a'):
    '''
    Q is one of ['2a', '2b']
    :param returns_dropNA:
    :param Q:
    :return:
    '''
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title('VMC')
    axs[0, 1].set_title('EMR')
    axs[1, 0].set_title('CSX')
    axs[1, 1].set_title('UNP')
    last_value_precent = []
    num_of_simulations = 100
    for i in range(num_of_simulations):
        print(i)
        hugeDF = generate_880_dataframe(returns_dropNA)
        stocks = [[1], [1], [1], [1]]  # VMC,EMR, CSX, UNP
        score_before = 1
        for index, row in hugeDF.iterrows():
            stocks[0].append(row['VMC'] * stocks[0][-1])
            stocks[1].append(row['EMR'] * stocks[1][-1])
            stocks[2].append(row['CSX'] * stocks[2][-1])
            stocks[3].append(row['UNP'] * stocks[3][-1])

        if Q == '2b':
            vmc_stock_value = 1.02 if len([stock for stock in stocks[0] if stock >= 1.36]) > 0 else stocks[0][-1]
            emr_stock_value = 1.02 if len([stock for stock in stocks[1] if stock >= 1.36]) > 0 else stocks[1][-1]
            csx_stock_value = 1.02 if len([stock for stock in stocks[2] if stock >= 1.36]) > 0 else stocks[2][-1]
            unp_stock_value = 1.02 if len([stock for stock in stocks[3] if stock >= 1.36]) > 0 else stocks[3][-1]
            score_after = (vmc_stock_value + emr_stock_value + csx_stock_value + unp_stock_value) / 4 - 1
            score_after = 0 if score_after <= 0 else score_after
        else:
            score_after = (stocks[0][-1] + stocks[1][-1] + stocks[2][-1] + stocks[3][-1]) / 4 - 1


        #
        stock_scores = {
            'before': score_before,
            'after': score_after
        }

        last_value_precent.append(stock_scores)
        # plt.plot(score_after)
        axs[0, 0].plot(stocks[0])
        axs[0, 1].plot(stocks[1])
        axs[1, 0].plot(stocks[2])
        axs[1, 1].plot(stocks[3])
    plt.show()


    print('0% change:',
          len([value['after'] for value in last_value_precent if -0.005 <= value['after'] <= 0.005]) / num_of_simulations)
    print('2% change:', len([value['after'] for value in last_value_precent if
                             0.015 <= value['after'] < 0.025]) / num_of_simulations)
    print('(2%-20%] change:', len([value['after'] for value in last_value_precent if
                                   0.02 < value['after'] <= 0.2]) / num_of_simulations)
    print('(20%-36%) change:', len([value['after'] for value in last_value_precent if
                                    0.2 < value['after'] < 0.36]) / num_of_simulations)
    m, h_m, h_p = mean_confidence_interval([value['after'] for value in last_value_precent])
    print('(', h_m, '<=', m, '<=', h_p, ')')

    #print("######### END ###########")


if __name__ == '__main__':
    data, returns = load_data()
    shape = returns.shape[0]

    #################################################
    #################### Q1 #########################
    #################################################
    # all_stocks_dict = q_1(returns * 100)
    # plot_heatmap_cov(returns * 100)
    returns_dropNA = returns.dropna() + 1
    # print(json.dumps(all_stocks_dict, indent=4))
    # print(returns_dropNA)

    #################################################
    #################### Q2 #########################
    #################################################
    for i in range(1):
        print("######## 2a", str(i), "############")
        q_2(returns_dropNA)
        print("######## 2b", str(i), "############")
        # q_2(returns_dropNA, Q='2b')
