import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils.preprocess import *
from utils.funcs import *
import utils.VaRModel as Model
import warnings
warnings.filterwarnings("ignore")

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
def setup(params):
    tk_all = [params['stock_config']['tickers'], params['option_config']['tickers']]
    startstr = params['datastart']
    print(startstr)
    endstr = params['dataend']
    print(endstr)

    # load data
    data = load_data(tk_all, startstr, endstr, params['use_history'])
    stocks = data[0]
    options = data[1:]
    stock_use, pf_use = stock_handle(stocks, params)
    # options_use = option_handle(options, params)
    return stock_use, pf_use, option_use




if __name__ == '__main__':
    # Setup portfolio and import data (.data/data_config.py and upload data files if using historical data)
    data_params = load_config("data")
    check_data_config(data_params)

    if data_params["portfolio_type"] in [1,2,3]:
        stock_use, pf_use = setup(data_params)[:2]
    else:
        stock_use, pf_use, option_use = setup(data_params)

    # Initial model object
    sys_params = load_config("params")
    check_param_config(sys_params)
    system = Model.varmodel(sys_params)

    num_stock = len(data_params['stock_config']['tickers'])
    if sys_params['param_model'] or sys_params['mc_model']:
        system.param_calibration(num_stock, stock_use, pf_use)
        calibrated = True
    if sys_params['param_model']:
        param_result = system.cal_param_var(data_params)
    # if params['mc_model']:
    #     mc_var, mc_es = use_montecarlo()
    # if params['hist_model']:
    #     hist_var, hist_es = use_historical()