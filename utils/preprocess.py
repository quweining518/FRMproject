from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json
import yfinance as yf # https://github.com/ranaroussi/yfinance

def load_file():
    path = r'./data/'
    df_stock = None
    df_option = None
    for f in os.listdir(path):
        if f.split('.')[0] == 'data_stock':
            if f.split('.')[1] == 'csv':
                df_stock = pd.read_csv(path+f, index_col = 0, parse_dates = True)
            elif f.split('.')[1] == 'xlsx':
                df_stock = pd.read_excel(path+f, index_col = 0, parse_dates = True)
            elif f.split('.')[1] == 'txt':
                df_stock = pd.read_table(path+f, index_col = 0, parse_dates = True)
            else:
                raise TypeError("Acceptable types of file are 'csv', 'xlsx' and 'txt' only.")

            if df_stock.index[0] > df_stock.index[-1]:
                df_stock = df_stock[::-1] # reverse datetime to be chronological order

        elif f.split('.')[0] == 'data_option':
            if f.split('.')[1] == 'csv':
                df_option = pd.read_csv(path+f, index_col = 0, parse_dates = True)
            elif f.split('.')[1] == 'xlsx':
                df_option = pd.read_excel(path+f, index_col = 0, parse_dates = True)
            elif f.split('.')[1] == 'txt':
                df_option = pd.read_table(path+f, index_col = 0, parse_dates = True)
            else:
                raise TypeError("Acceptable types of file are 'csv', 'xlsx' and 'txt' only.")

    his_dict = dict()
    his_dict['stock'] = df_stock
    his_dict['option'] = dict()
    his_dict['option']['call'] = df_option
    his_dict['option']['put'] = df_option
    return his_dict

def load_config(config):
    if config == "data":
        name = r"./data/data_config.json"
    else:
        name = r"./data/param_config.json"
    with open(name) as config_file:
        config = json.load(config_file)
    # print(config)
    return config

def check_data_config(config):
    if not isinstance(config, dict):
        raise TypeError("Configuration file should be a dict.")
    if config["use_history"] not in [0,1]:
        raise ValueError("Illegal input type or value.")
    if not isinstance(config["datastart"], str):
        raise TypeError("'datastart' field should be a string of format 'yyyy-mm-dd'.")
    else:
        try:
            datetime.strptime(config["datastart"], "%Y-%m-%d")
        except:
            raise ValueError("Illegal input format or value.")
    if not isinstance(config["dataend"], str):
        raise TypeError("'dataend' field should be a string of format 'yyyy-mm-dd'.")
    else:
        try:
            datetime.strptime(config["dataend"], "%Y-%m-%d")
        except:
            raise ValueError("Illegal input format or value.")
    if config["portfolio_type"] not in [1,2,3,4]:
        raise ValueError("Illegal input type or value.")
    if not isinstance(config["total_position"], int) and not isinstance(config["total_position"], float):
        raise TypeError("Illegal input type.")
    if config["total_position"] <= 0:
        raise ValueError("Total position should be positive")
    if not isinstance(config["option_weight"], int) and not isinstance(config["option_weight"], float):
        raise TypeError("Illegal input type.")
    if (config["option_weight"] < 0) or (config["option_weight"] > 1):
        raise ValueError("Option weight should be a float/int within range [0,1].")
    if len(config["stock_config"]["long_tickers"]) <= 0:
        raise ValueError("Empty stock list.")
    if config["stock_config"]["long_weight"] not in ['equal', 'custom']:
        raise ValueError("Acceptable field value are 'equal' and 'custom' only.")
    if len(config["stock_config"]["short_tickers"]) <= 0:
        raise ValueError("Empty stock list.")
    if config["stock_config"]["short_weight"] not in ['equal', 'custom']:
        raise ValueError("Acceptable field value are 'equal' and 'custom' only.")

    pos_weight = config["stock_config"]["long_custom_weight"]
    neg_weight = config["stock_config"]["short_custom_weight"]
    if len(pos_weight) > 0 and sum(pos_weight) != 1.0:
        raise ValueError("Sum of long portfolio weight should be 1.")
    if len(neg_weight) > 0 and sum(neg_weight) != 1.0:
        raise ValueError("Sum of short portfolio weight should be 1.")
    if len(config["option_config"]["tickers"]) > 0:
        if config["option_config"]["option_type"] not in ["call", "put"]:
            raise ValueError("Acceptable field value are 'call' and 'put' only.")
        if not isinstance(config["option_config"]["moneyness"], float):
            raise TypeError("Illegal input type.")
    return

def check_param_config(config):
    if not isinstance(config, dict):
        raise TypeError("Configuration file should be a dict.")
    if not isinstance(config["risk_config"]["horizon"], int) or config["risk_config"]["horizon"]<= 0:
        raise ValueError("Illegal input type or value.")
    if not isinstance(config["risk_config"]["start"], str):
        raise TypeError("'start' field should be a string of format 'yyyy-mm-dd'.")
    else:
        try:
            datetime.strptime(config["risk_config"]["start"], "%Y-%m-%d")
        except:
            raise ValueError("Illegal input format or value.")
    if not isinstance(config["risk_config"]["end"], str):
        raise TypeError("'end' field should be a string of format 'yyyy-mm-dd'.")
    else:
        try:
            datetime.strptime(config["risk_config"]["end"], "%Y-%m-%d")
        except:
            raise ValueError("Illegal input format or value.")
    if (config["risk_config"]["var_percentile"] <= 0) or (config["risk_config"]["var_percentile"] >= 1):
        raise ValueError("Percentile should be a float within range (0,1).")
    if (config["risk_config"]["es_percentile"] <= 0) or (config["risk_config"]["es_percentile"] >= 1):
        raise ValueError("Percentile should be a float within range (0,1).")
    if not isinstance(config["calib_window"], int) or config["calib_window"] <= 0:
        raise ValueError("Illegal input format or value.")
    if not isinstance(config["calib_lambda"], float) or config["calib_lambda"] <= 0 or config["calib_lambda"] >= 1:
        raise ValueError("Illegal input format or value.")
    if not isinstance(config["tradedays"], int) or config["tradedays"] <= 0 or config["tradedays"] > 366:
        raise ValueError("Illegal input format or value.")
    if config["calib_weighting"] not in ['unweighting', 'exponential']:
        raise ValueError("Acceptable field value are 'unweighting' and 'exponential' only.")
    if config["param_config"]["assumption"] not in ["gbm", "normal"]:
        raise ValueError("Illegal input format or value.")
    if not isinstance(config["hist_window"], int) or config["hist_window"] <= 0:
        raise ValueError("Illegal input format or value.")
    if not isinstance(config["mc_config"]["n_paths"], int) or config["mc_config"]["n_paths"] <= 0:
        raise ValueError("Illegal input format or value.")
    if config["mc_config"]["assumption"] not in ["gbm", "normal"]:
        raise ValueError("Illegal input format or value.")
    if config["param_model"] not in [0,1]:
        raise ValueError("Illegal input type or value.")
    if config["hist_model"] not in [0,1]:
        raise ValueError("Illegal input type or value.")
    if config["mc_model"] not in [0,1]:
        raise ValueError("Illegal input type or value.")
    if config["plot_figure"] not in [0,1]:
        raise ValueError("Illegal input type or value.")
    if config["save_output"] not in [0,1]:
        raise ValueError("Illegal input type or value.")
    return


def load_data(tickers=None, startdate=None, enddate=None, use_history = False):

    if use_history:
        his_dict = load_file()
        df_stock = his_dict['stock']
        df_call = his_dict['option']['call']
        df_put = his_dict['option']['put']

    else:
        tk_stock = tickers[0]
        tk_option = tickers[1]

        if len(tk_stock) > 0:
            print('Fetching stocks: %s ' % tk_stock)
            all_stock = yf.download(tk_stock, start = startdate, end = enddate)
            df_stock = pd.DataFrame(all_stock.iloc[:,:len(tk_stock)].values,
                                    index = all_stock.index, columns= tk_stock)
            print(df_stock.head())
        else:
            df_stock = None

        if len(tk_option) > 0:
            df_call = pd.DataFrame()
            df_put = pd.DataFrame()
            for tk in tk_option:
                print('Fetching option: %s' % tk)
                yftk = yf.Ticker(tk)
                exps = yftk.options  # expiration dates
                try:
                    for e in exps:
                        opt = yftk.option_chain(e)
                        calls, puts = opt.calls, opt.puts
                        calls['expirationDate'] = e
                        calls['Symbol'] = tk
                        puts['expirationDate'] = e
                        puts['Symbol'] = tk
                        df_call = df_call.append(calls, ignore_index=True)
                        df_put = df_put.append(puts, ignore_index= True)
                except:
                    pass
        else:
            df_call = None
            df_put = None

    return [df_stock, df_call, df_put]

def stock_handle(df_price, weights):
    log_return = lambda x: np.log(x/x.shift(1))
    log_return_sq = lambda ret: np.power(ret, 2)

    df_ret = df_price.apply(log_return)
    df_ret_sq = df_ret.apply(log_return_sq)
    weights = np.array(weights)
    if np.sum(weights) == 1 or np.sum(weights) == 0:
        pf_ret = df_ret.apply(lambda x: np.sum(x * weights), axis=1)
        pf_ret_sq = pf_ret.apply(log_return_sq)
    else:
        raise ValueError("Weight scheme not acceptable.")
    df_handle = pd.concat([df_price, df_ret, df_ret_sq], axis = 1).dropna()
    print(df_handle.head())
    pf_handle = pd.concat([pf_ret, pf_ret_sq], axis = 1).dropna()
    pf_handle.columns = ['log_rtn', 'log_rtn_sq']
    print(pf_handle.head())
    return df_handle, pf_handle

def bs_price(S0, K, T, vol, r, optype='put'):
    if optype == 'put':
        sign = -1
    else:
        sign = 1
    F = S0 * np.exp(r * T)
    v = vol * np.sqrt(T)
    d1 = np.log(F / K) / v + 0.5 * v
    d2 = d1 - v

    price = sign * (F * norm.cdf(sign * d1) - K * norm.cdf(sign * d2)) * np.exp(-r * T)
    return price