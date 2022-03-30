import os
import tqdm
import glob
import platform
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def linear_regression(return_data, ff_data):
    Y = return_data['Wkly-Rtns']-ff_data['RF']
    Y = Y.to_numpy()
    X = np.vstack((ff_data['Mkt-RF'].to_numpy(), ff_data['SMB'].to_numpy(), ff_data['HML'].to_numpy()))

    Y_ = np.array([0])
    X_ = np.array([0, 0, 0]).transpose()

    for i in range(Y.size):
        # Except data contains NaN
        if not np.isnan(X[:, i]).any() and not np.isnan(Y[i]).any():
            Y_ = np.vstack((Y_, Y[i]))
            X_ = np.vstack((X_, X[:, i]))

    # EVRG & LIN have no data available!
    if Y_.size == 1:
        return {'b_Mkt-RF': np.nan, 'b_SMB': np.nan, 'b_HML': np.nan, 'a': np.nan, 'Y_avg': np.nan}

    model = LinearRegression()
    model.fit(X_[1:], Y_[1:])
    [[mkt_rf, smb, hml]] = model.coef_
    [intercept] = model.intercept_

    Y_avg = np.mean(Y_[1:])

    return {'b_Mkt-RF': mkt_rf, 'b_SMB': smb, 'b_HML': hml, 'a': intercept, 'Y_avg': Y_avg}


def save_csv(pd_frame, save_path, save_name):
    # assert isinstance(pd_frame, pd.core.series.Series) \
           # or isinstance(pd_frame, pd.core.frame.DataFrame)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = '{}/{}.csv'.format(save_path, save_name)
    if not os.path.exists(save_path):
        if platform.system() == 'Windows':
            with open(save_path, 'a+') as f:
                f.close()
        else:
            os.mknod(save_path)
    pd_frame.to_csv(save_path)


def calc_weekly_returns(stock_path, save=False, start_date='20150102', end_date='20171231'):
    stock_name = os.path.split(stock_path)[-1].split('.')[0]
    stock_data = pd.read_csv(stock_path, index_col=0)

    # Convert index type to datetime
    stock_data.index = pd.to_datetime(stock_data.index)

    # Define date range
    index_range = pd.date_range(start_date, end_date)
    stock_in_range = pd.DataFrame(stock_data, index=index_range).ffill()

    # Sample weekly returns
    stock_weekly_returns = stock_in_range['close'].resample('W-FRI').ffill().pct_change()
    stock_weekly_returns = pd.DataFrame(stock_weekly_returns)
    stock_weekly_returns.columns=['Wkly-Rtns']

    if save:
        save_csv(stock_weekly_returns,
                 'weekly_returns_{}_to_{}/'.format(start_date, end_date), stock_name)

    return stock_weekly_returns

# use name.csv
def get_names_with_number():
    stock_path_list = glob.glob('stock_dfs/*.csv')
    names = pd.DataFrame(columns=['name'])
    for stock_path in tqdm.tqdm(stock_path_list):
        name = os.path.split(stock_path)[-1].split('.')[0]
        names = names.append({'name': name}, ignore_index=True)
    save_csv(names, '.', 'names')


def calc_all_weekly_returns(start_date='20150102', end_date='20171231', save=False):
    names = pd.read_csv('names.csv', index_col=0)
    stock_path_list = glob.glob('stock_dfs/*.csv')
    returns_list = []
    for stock_path in tqdm.tqdm(stock_path_list):
        returns = calc_weekly_returns(stock_path, save=save, start_date=start_date, end_date=end_date)
        returns_list.append(returns)

    return returns_list, names


def calc_factor_loadings(returns, start_date='20150102', end_date='20171231'):
    if isinstance(returns, str):
        stock_name = os.path.split(returns)[-1].split('.')[0]
        return_data = pd.read_csv(returns, index_col=0)
        return_data.index = pd.to_datetime(return_data.index)
    else:
        return_data = returns

    ff_data = pd.read_csv('FF_factors_wkly_202201.csv', index_col=0)

    # Convert by changing the format
    ff_data.index = pd.to_datetime(ff_data.index, format='%Y%m%d')

    # Define date range
    index_range = pd.date_range(start_date, end_date)
    ff_in_range = pd.DataFrame(ff_data, index=index_range).resample('W-FRI').ffill()

    regr_dict = linear_regression(return_data, ff_in_range)
    return regr_dict


def calc_all_factor_loadings(start_date='20150102', end_date='20171231'):
    data = pd.DataFrame(columns=['a', 'b_Mkt-RF', 'b_SMB', 'b_HML'])
    Y_data = pd.DataFrame(columns=['Y'])

    stock_path_list = glob.glob('stock_dfs/*.csv')
    for stock_path in tqdm.tqdm(stock_path_list):
        weekly_returns = calc_weekly_returns(stock_path, save=False,
                                             start_date=start_date, end_date=end_date)
        regression_dict = calc_factor_loadings(weekly_returns,
                                               start_date=start_date, end_date=end_date)
        data = data.append({'a': regression_dict['a'],
                            'b_Mkt-RF': regression_dict['b_Mkt-RF'],
                            'b_SMB': regression_dict['b_SMB'],
                            'b_HML': regression_dict['b_HML']}, ignore_index=True)
        Y_data = Y_data.append({'Y': regression_dict['Y_avg']}, ignore_index=True)

    save_csv(data, 'factor_loadings', 'factor_loadings_{}_to_{}'.format(start_date, end_date))
    save_csv(Y_data, 'Y_avg', 'Y_avg_{}_to_{}'.format(start_date, end_date))
    return data, Y_data


def significant(start_date='20150102', end_date='20171231'):
    factor_path = 'factor_loadings/factor_loadings_{}_to_{}.csv'.format(start_date, end_date)
    factor_data = pd.read_csv(factor_path, index_col=0)

    factor_data = factor_data.dropna()
    size = factor_data.shape[0]

    Y_path = 'Y_avg/Y_avg_{}_to_{}.csv'.format(start_date, end_date)
    Y_data = pd.read_csv(Y_path, index_col=0)
    Y_data = Y_data.dropna()

    y = Y_data.to_numpy().reshape(-1)
    a = factor_data['a'].to_numpy()
    b_mkt_rf = factor_data['b_Mkt-RF'].to_numpy()
    b_smb = factor_data['b_SMB'].to_numpy()
    b_hml = factor_data['b_HML'].to_numpy()

    tt_mkt = ttest_ind(y, b_mkt_rf)[0]
    tt_smb = ttest_ind(y, b_smb)[0]
    tt_hml = ttest_ind(y, b_hml)[0]

    factor_names = ['b_Mkt-RF', 'b_SMB', 'b_HML']
    factor_values = [tt_mkt, tt_smb, tt_hml]
    significant_idx = np.argmax([np.abs(i) for i in factor_values])

    return factor_values, factor_names, significant_idx


def calc_average_returns(source_date = ['20150102', '20171231'], target_date = ['20171231', '20180107']):
    start_date_source, end_date_source = source_date
    stock_names = pd.read_csv('names.csv', index_col=0)

    factor_path = 'factor_loadings/factor_loadings_{}_to_{}.csv'.format(start_date_source, end_date_source)
    factor_data = pd.read_csv(factor_path, index_col=0)
    factor_data = pd.merge(stock_names, factor_data, left_index=True, right_index=True)
    factor_data = factor_data.dropna()

    values, names, idx = significant(start_date=start_date_source, end_date=end_date_source)
    sig = names[idx]
    sorted_factor_data = factor_data.sort_values(sig, ascending=True, inplace=False)

    top = sorted_factor_data[:75]
    top_names = top['name']
    bottom = sorted_factor_data[-75:]
    bottom_names = bottom['name']

    # since the algorithm before uses FRIDAY to resample,
    # the target date should be 2 days earlier
    target_date_start = pd.to_datetime(target_date[0]) + pd.Timedelta(days=-2)
    target_date_end = pd.to_datetime(target_date[1]) + pd.Timedelta(days=-2)

    top_returns = []
    for top_name in top_names:
        stock_path = 'stock_dfs/{}.csv'.format(top_name)
        curr_return = calc_weekly_returns(
            stock_path, start_date=target_date_start, end_date=target_date_end)
        curr_return = curr_return.dropna()
        top_returns.append(curr_return['Wkly-Rtns'][0])

    bottom_returns = []
    for bottom_name in bottom_names:
        stock_path = 'stock_dfs/{}.csv'.format(bottom_name)
        curr_return = calc_weekly_returns(
            stock_path, start_date=target_date_start, end_date=target_date_end)
        curr_return = curr_return.dropna()
        bottom_returns.append(curr_return['Wkly-Rtns'][0])

    top_avg = np.mean(np.array(top_returns))
    bottom_avg = np.mean(np.array(bottom_returns))

    return top_avg, bottom_avg


def date_to_str(date):
    return ''.join(str(date).split(' ')[0].split('-'))


def run_task1():
    returns_list, names = calc_all_weekly_returns()
    for idx, returns in enumerate(returns_list):
        name = names['name'][idx]
        print('Weekly returns of {} is (head only): \n{}'.format(name, returns.head()))


def run_task2():
    res = calc_all_factor_loadings()
    print(res)


def run_task3():
    factor_values, factor_names, significant_idx = significant()
    print('T-statistics of Mkt, SMB, HML is:\n\t{}'.format(factor_values))
    print('The most significant factor is: {}'.format
          (factor_names[significant_idx].split('_')[-1]))
    print('Relationship: {}'.format
          ('Positive' if factor_values[significant_idx]<0 else 'Negative'))


def run_task4():
    target_date = ['20171231', '20180107']
    top_avg, bottom_avg = calc_average_returns()
    print('\nAverage return of top 20% of the stocks\n\t'
          'from {} to {} is: {}'.format(target_date[0], target_date[1], top_avg))
    print('Average return of bottom 20% of the stocks\n\t'
          'from {} to {} is: {}'.format(target_date[0], target_date[1], bottom_avg))

def run_task5a():
    source_date = ['20150109', '20180107']
    target_date = ['20180107', '20180115']
    calc_all_factor_loadings(
        start_date=source_date[0], end_date=source_date[1])
    top_avg, bottom_avg = calc_average_returns(
        source_date=source_date, target_date=target_date)
    print('\nAverage return of top 20% of the stocks\n\t'
          'from {} to {} is: {}'.format(target_date[0], target_date[1], top_avg))
    print('Average return of bottom 20% of the stocks\n\t'
          'from {} to {} is: {}'.format(target_date[0], target_date[1], bottom_avg))

def run_task5b():
    initial_start_time = pd.to_datetime('20150109')
    initial_end_time = pd.to_datetime('20180107')

    initial_target_start = pd.to_datetime('20180107')
    initial_target_end = pd.to_datetime('20180115')

    target_start_list = []
    target_end_list = []

    top_avg_list = []
    bottom_avg_list = []

    for i in range(1, 21):
        time_delta = pd.Timedelta(days=i*7)
        source_date = [date_to_str(initial_start_time+time_delta),
                       date_to_str(initial_end_time+time_delta)]
        target_date = [date_to_str(initial_target_start+time_delta),
                       date_to_str(initial_target_end+time_delta)]

        target_start_list.append(target_date[0])
        target_end_list.append(target_date[1])

        print('\n({}/20) Regression range: {} to {}'
              .format(i, source_date[0], source_date[1]))
        calc_all_factor_loadings(
            start_date=source_date[0], end_date=source_date[1])
        top_avg, bottom_avg = calc_average_returns(
            source_date=source_date, target_date=target_date)

        top_avg_list.append(top_avg)
        bottom_avg_list.append(bottom_avg)

        print('\nAverage return of top 20% of the stocks\n\t'
              'from {} to {} is: {}'.format(target_date[0], target_date[1], top_avg))
        print('Average return of bottom 20% of the stocks\n\t'
              'from {} to {} is: {}'.format(target_date[0], target_date[1], bottom_avg))

    data = pd.DataFrame()
    data['start_time'] = target_start_list
    data['end_time'] = target_end_list
    data['top_average_returns'] = top_avg_list
    data['bottom_average_returns'] = bottom_avg_list
    save_csv(data, '.', 'average_returns')

    return data

def run_task5c():
    ar = pd.read_csv('average_returns.csv')
    top_returns = ar['top_average_returns']
    bottom_returns = ar['bottom_average_returns']

    cumulative_top = (top_returns + 1).cumprod()
    cumulative_bottom = (bottom_returns + 1).cumprod()

    cumulative_top.plot()
    cumulative_bottom.plot()
    plt.legend(['Top', 'Bottom'])
    plt.show()

def run_task6():
    sort_date_start = '20171231'
    sort_date_end = '20180107'
    # According to the algorithm before is resampled with W=FRI, the time should minus 2
    real_date_start = date_to_str(pd.to_datetime(sort_date_start) + pd.Timedelta(days=-2))
    real_date_end = date_to_str(pd.to_datetime(sort_date_end) + pd.Timedelta(days=-2))

    returns_list_next, names_next = calc_all_weekly_returns(real_date_start, real_date_end)
    returns_list_next = [returns_list_next[j]['Wkly-Rtns'][1] for j in range(len(returns_list_next))]
    names_next['returns'] = returns_list_next
    names_next = names_next.dropna()
    data_next = names_next.sort_values('returns', ascending=True, inplace=False)
    data_next = data_next.dropna()

    data = pd.DataFrame(columns=['sort_date', 'bottom_returns', 'top_returns'])

    for i in range(1, 21):
        sort_date = date_to_str(pd.to_datetime(sort_date_start) + pd.Timedelta(days=(i-1)*7))
        print('\n({}/20)\tsort_date: {}'.format(i, sort_date))
        real_date_start_next = date_to_str(pd.to_datetime(real_date_start) + pd.Timedelta(days=i*7))
        real_date_end_next = date_to_str(pd.to_datetime(real_date_end) + pd.Timedelta(days=i*7))

        data_curr = data_next
        bottom_curr = data_curr['returns'][-75:]
        bottom_mean = np.mean(bottom_curr.to_numpy())

        returns_list_next, names_next = calc_all_weekly_returns(real_date_start_next, real_date_end_next)
        returns_list_next = [returns_list_next[j]['Wkly-Rtns'][1] for j in range(len(returns_list_next))]
        names_next['returns'] = returns_list_next
        names_next = names_next.dropna()
        data_next = names_next.sort_values('returns', ascending=True, inplace=False)
        data_next = data_next.dropna()
        top_next = data_next['returns'][:75]
        top_mean = np.mean(top_next.to_numpy())

        data = data.append({'sort_date': sort_date, 'bottom_returns': bottom_mean, 'top_returns': top_mean}, ignore_index=True)

    save_csv(data, '.', 'momentum')
    print(data)

def run_task6_plot():
    data = pd.read_csv('momentum.csv', index_col=0)
    bottom_returns = data['bottom_returns'].dropna()
    top_returns = data['top_returns'].dropna()

    cumulative_top = (top_returns + 1).cumprod()
    cumulative_bottom = (bottom_returns + 1).cumprod()

    cumulative_top.plot()
    cumulative_bottom.plot()
    plt.legend(['Top', 'Bottom'])
    plt.show()

def run_all():
    run_task1()
    run_task2()
    run_task3()
    run_task4()
    run_task5a()
    run_task5b()
    run_task5c()
    run_task6()
    run_task6_plot()

if __name__ == '__main__':
    run_all()
    # run_task6_plot()

    # significant()
    # stock_path = 'stock_dfs/F.csv'
    # weekly_returns = calc_weekly_returns(stock_path, start_date='20171229', end_date='20180105', save=False)
    # print(weekly_returns)
    # weekly_returns.plot()
    # plt.show()

    # print(calc_factor_loadings(weekly_returns))


    # calc_all_weekly_returns(save=False)

    # returns_path = 'weekly_returns/EBAY.csv'
    # calc_excess_returns(returns_path)

