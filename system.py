import pandas as pd
import numpy as np
from utils.preprocess import *
import utils.VaRModel as Model
import warnings
warnings.filterwarnings("ignore")

"""
portfolio_type = 1
use_history = False
total_position = 10000  # int (unit: dollar), default: 10000
tickers = ['BA', 'NOC']
weight = 'equal'  # 'equal' or 'custom' (default: 'equal' and custom input will be ignored)
custom_weight = [0.4, 0.6] # long: positive float sum to 1; short: negative float sum to -1
horizon = 5  # int (unit: day), default: 5 (5-day VaR/ES)
percentile = 0.99  # float (range (0,1)): percentile of
"""


def setup(params, portfolio_type):
    long_nstock = len(params['stock_config']['long_tickers'])
    short_nstock = len(params['stock_config']['short_tickers'])

    # 1: Long only portfolio
    if portfolio_type == 1:
        tk_all = [params['stock_config']['long_tickers'], []]
        weights = [1 / len(tk_all[0])] * len(tk_all[0]) if params["stock_config"]["long_weight"] == "equal" else \
            params["stock_config"]["long_custom_weight"]
        num_stock = long_nstock
    # 2: Short only portfolio
    elif portfolio_type == 2:
        tk_all = [params['stock_config']['short_tickers'], []]
        weights = [1 / len(tk_all[0])] * len(tk_all[0]) if params["stock_config"]["short_weight"] == "equal" else \
            params["stock_config"]["short_custom_weight"]
        num_stock = short_nstock
    # 3: Long only stock + ATM put option for VaR reduction
    else:
        tk_all = [params['stock_config']['long_tickers'], params['option_config']['tickers']]
        weights = [1 / len(tk_all[0])] * len(tk_all[0]) if params["stock_config"]["long_weight"] == "equal" else \
            params["stock_config"]["long_custom_weight"]
        num_stock = long_nstock

    startstr = params['datastart']
    endstr = params['dataend']

    # load data
    stocks = load_data(tk_all, startstr, endstr, params['use_history'])[0]
    stock_use, pf_use = stock_handle(stocks, weights)
    # option_use = option_handle(options, optype = params["option_config"]["option_type"])
    if portfolio_type == 3:
        imvol = pd.read_csv("./data/impliedvol.csv", index_col = 0, parse_dates=True).iloc[:,0]
    return num_stock, stock_use, pf_use, imvol



if __name__ == '__main__':
    # Setup portfolio and import data (.data/data_config.py and upload data files if using historical data)
    data_params = load_config("data")
    check_data_config(data_params)
    pf_type = data_params["portfolio_type"]  # get portfolio type
    print("Portfolio type is ", pf_type)
    if pf_type in [1,2]:
        num_stock, stock_use, pf_use = setup(data_params, pf_type)[:3]
    else:
        num_stock, stock_use, pf_use, im_vol = setup(data_params, pf_type)

    # Initial model object
    sys_params = load_config("params")
    check_param_config(sys_params)
    system = Model.varmodel(data_params, sys_params) # initialize the system

    if sys_params['param_model'] or sys_params['mc_model']:
        system.param_calibration(num_stock, stock_use, pf_use)
        calibrated = True
        print("Calibration is done.")
    if sys_params['param_model'] and calibrated:
        param_result = system.cal_param_var(pf_type, data_params)
    if sys_params['mc_model'] and calibrated:
        mc_result = system.cal_mc_var(pf_type, stock_use, im_vol/100, data_params)
    if sys_params['hist_model']:
        hist_result = system.cal_hist_var(pf_type, pf_use['log_rtn'])
