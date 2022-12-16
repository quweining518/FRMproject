import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils.preprocess import *
from utils.funcs import *
import utils.VaRModel as Model
from system import *
import os
import warnings
warnings.filterwarnings("ignore")


class BacktestCreator(object):
    def __init__(self, bt_name, hist_data, var_data):
        func = lambda x: datetime.utcfromtimestamp(x.tolist() / 1e9).date()
        self.index = [func(x) for x in var_data.index.values]
        self.bt_name = bt_name
        self.hist_data = hist_data.filter(items = var_data.index.to_list(), axis = 0).log_rtn
        self.var_data = var_data.param_VaR
        curr_time = datetime.now()
        curr_time = curr_time.strftime('%Y%m%d_%H%M')
        self.path = os.path.join('./backtest', bt_name + "_" + curr_time)
        os.makedirs(os.path.join('./backtest', bt_name + "_" + curr_time))
        len(self.hist_data)

    def get_risk_params(self, sys_params, data_params):
        risk_params = {}
        risk_config = sys_params['risk_config']
        self.horizon = risk_config['horizon']
        # VaR Stats
        risk_params['Horizon (days)'] = self.horizon
        risk_params['Start Date'] = datetime.strptime(risk_config["start"], "%Y-%m-%d") #start date of VaR calculation
        risk_params['Dnd Date'] = datetime.strptime(risk_config["end"], "%Y-%m-%d") #end date of VaR calculation
        risk_params['VaR Percentile'] = risk_config["var_percentile"]
        risk_params['ES Percentile'] = risk_config["es_percentile"]

        # Calibration Stats
        risk_params['Calibration Window'] = sys_params["calib_window"]
        risk_params['Lambda (Exponentially Weighted'] = sys_params["calib_lambda"]
        risk_params['Data Weighting'] = 'Unweighted' if sys_params["calib_weighting"] == "unweighting" else 'Exponentially Weighted'
        self.long_port = {}
        self.long_port['tickers'] = data_params['stock_config']['long_tickers']
        risk_params['Portfolio Long Components'] = self.long_port['tickers']
        self.long_port['weight method'] = data_params['stock_config']['long_weight']
        risk_params['Portfolio Long Weighting Method'] = self.long_port['weight method']
        self.long_port['weights'] = data_params['stock_config']['long_custom_weight']
        risk_params['Portfolio Long Weights'] = self.long_port['weights']
        self.short_port = {}
        self.short_port['tickers'] = data_params['stock_config']['short_tickers']
        risk_params['Portfolio Short Components'] = self.short_port['tickers']
        self.short_port['weight method'] = data_params['stock_config']['short_weight']
        risk_params['Portfolio Short Weighting Method'] = self.short_port['weight method']
        self.short_port['weights'] = data_params['stock_config']['short_custom_weight']
        risk_params['Portfolio Short Weights'] = self.short_port['weights']
        return risk_params

    def get_port_losses(self):
        pass

    def get_exceptions(self):
        hist_data = self.hist_data
        var_data = self.var_data
        count = []
        for i in range(len(var_data)):
            temp_hist = hist_data[i:i + 252].values.flatten()
            temp_var = var_data[i:i + 252].values.flatten()
            count_i = len([1 for i, j in zip(temp_hist, temp_var) if -i > j])
            count.append(count_i)
        return count

    def plot_exceptions(self, count):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.index, count, lw=1, label="# of exceptions")
        ax.set_ylabel("# of counts")
        ax2 = ax.twinx()
        ax2.plot(self.index, self.hist_data.values.flatten(), lw=1, label= str(self.horizon) + ' day change in portfolio value', color='green')
        ax2.plot(self.index, self.var_data.values.flatten(), lw=1, label='VaR', color='black')
        ax2.set_ylabel("Value")
        title = self.bt_name
        ax.set_title(title)
        ax.legend(loc = 2)
        ax2.legend(loc = 1)
        plt.savefig(os.path.join(self.path, 'exceptions.png'))




"""
portfolio_type choices:
1 - Stocks only + Long only
2 - Stocks only + Short only
3 - Stocks only + Long short
4 - Stocks & Options + Long only
5 - Stocks & Options + Long short

"""
# portfolio_type = 1
# use_history = False
#
# total_position = 10000  # int (unit: dollar), default: 10000
# tickers = ['BA', 'NOC']
# weight = 'equal'  # 'equal' or 'custom' (default: 'equal' and custom input will be ignored)
# custom_weight = [0.4, 0.6] # long: positive float sum to 1; short: negative float sum to -1
#
# horizon = 5  # int (unit: day), default: 5 (5-day VaR/ES)
# percentile = 0.99  # float (range (0,1)): percentile of
#

if __name__ == '__main__':
    # Setup portfolio and import data (.data/data_config.py and upload data files if using historical data)
    data_params = load_config("data")
    check_data_config(data_params)
    # a = yf.download(['BA', 'NOC', 'PFE', 'AAPL'], start='2002-01-01', end='2022-11-30')
    if data_params["portfolio_type"] in [1,2,3]:
        stock_use, pf_use = setup(data_params, data_params["portfolio_type"])[:2]
    else:
        stock_use, pf_use, option_use = setup(data_params, data_params["portfolio_type"])


    # Initial model object
    sys_params = load_config("params")
    check_param_config(sys_params)
    system = Model.varmodel(sys_params)


    pf_type = data_params["portfolio_type"]
    print("Portfolio type is ", pf_type)
    long_nstock = len(data_params['stock_config']['long_tickers'])
    print("Number of stocks: ", long_nstock)
    short_nstock = len(data_params['stock_config']['short_tickers'])
    print("Number of stocks: ", short_nstock)
    num_stock = long_nstock + short_nstock

    if sys_params['param_model'] or sys_params['mc_model']:
        system.param_calibration(num_stock, stock_use, pf_use)
        calibrated = True
        print("Calibration is done.")
    if sys_params['param_model'] and calibrated:
        param_result = system.cal_param_var(pf_type, data_params)
    if sys_params['mc_model'] and calibrated:
        mc_result = system.cal_mc_var(pf_type, data_params)
    if sys_params['hist_model']:
        hist_result = system.cal_hist_var(pf_type, data_params, pf_use['log_rtn'])

    backtest = BacktestCreator('test', pf_use, param_result)
    backtest.get_risk_params(sys_params, data_params)
    count = backtest.get_exceptions()
    backtest.plot_exceptions(count)