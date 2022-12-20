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
    def __init__(self, bt_name, var_data, var_method):
        self.bt_name = bt_name
        self.var_method = var_method
        self.var_data = pd.DataFrame(var_data.iloc[:, 0])
        curr_time = datetime.now()
        curr_time1 = curr_time.strftime('%Y/%m/%d %H:%M')
        print("Backtested at", curr_time1)
        curr_time2 = curr_time.strftime('%Y%m%d_%H%M')
        self.path = os.path.join('./backtest', bt_name + "_" + var_method + "_" + curr_time2)
        os.makedirs(os.path.join('./backtest', bt_name + "_" + var_method + "_" + curr_time2))

    def get_risk_params(self, sys_params, data_params):
        risk_params = {}
        risk_config = sys_params['risk_config']
        self.horizon = risk_config['horizon']
        # VaR Stats
        risk_params['Horizon (days)'] = self.horizon
        risk_params['Start Date'] = datetime.strptime(risk_config["start"], "%Y-%m-%d")  # start date of VaR calculation
        self.sdate = risk_params['Start Date']
        risk_params['End Date'] = datetime.strptime(risk_config["end"], "%Y-%m-%d")  # end date of VaR calculation
        self.edate = risk_params['End Date']
        risk_params['VaR Percentile'] = risk_config["var_percentile"]
        self.p_var = risk_params['VaR Percentile']
        risk_params['ES Percentile'] = risk_config["es_percentile"]
        self.p_es = risk_params['ES Percentile']

        # Calibration Stats
        risk_params['Calibration Window'] = sys_params["calib_window"]
        risk_params['Lambda (Exponentially Weighted'] = sys_params["calib_lambda"]
        risk_params['Data Weighting'] = 'Unweighted' if sys_params[
                                                            "calib_weighting"] == "unweighting" else 'Exponentially Weighted'
        self.p_type = data_params['portfolio_type']
        risk_params['Portfolio Type'] = self.p_type
        self.position = data_params['total_position']
        risk_params['total_position'] = data_params['total_position']
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
        horizon = self.horizon
        startdate = self.sdate - timedelta(days=400)
        enddate = self.edate
        total_position = self.position
        portfolio_type = self.p_type
        # 1: Long only portfolio
        if portfolio_type == 1:
            tickers = self.long_port['tickers']
            n = len(tickers)
            w_method = self.long_port['weight method']
            if w_method == 'equal':
                weights = [(1 / n) for i in range(n)]
                S0 = [total_position * (1 / n) for i in range(n)]
            else:
                weights = self.long_port['weights']
                S0 = [total_position * weights[i] for i in range(n)]
        # 2: Short only portfolio
        elif portfolio_type == 2:
            tickers = self.short_port['tickers']
            n = len(tickers)
            w_method = self.short_port['weight method']
            if w_method == 'equal':
                weights = [total_position * (1 / n) for i in range(n)]
                S0 = [total_position * (1 / n) for i in range(n)]
            else:
                weights = self.long_port['weights']
                S0 = [total_position * weights[i] for i in range(n)]

        # 3: Long-short portfolio
        # (hedged portfolio: can only take one long and one short stock with equal weight, i.e: 10000 for long, 10000 for short)

        df = pd.DataFrame(yf.download(tickers, start=startdate, end=enddate)['Adj Close'])

        if n > 1:
            n_shares = [S0[i] / df[tickers[i]][0] for i in range(n)]
            # different methods to calculate returns. Normalized by the start date -> use df*n_shares
            # treat the portfolio as the stock. Normalized by the total position -> use df*weights
            df['port level'] = (df * n_shares).sum(1)
            if portfolio_type == 1:
                df['losses'] = -total_position * ((df['port level'] / df['port level'].shift(horizon)) - 1)
            elif portfolio_type == 2:
                df['losses'] = total_position * ((df['port level'] / df['port level'].shift(horizon)) - 1)

        elif n == 1:
            if portfolio_type == 1:
                df['losses'] = -total_position * ((df.values.flatten() / df.shift(horizon).values.flatten()) - 1)
            elif portfolio_type == 2:
                df['losses'] = total_position * ((df.values.flatten() / df.shift(horizon).values.flatten()) - 1)

        df_losses = df['losses']
        self.hist_data = df_losses

    def return_data(self):
        return self.hist_data, self.var_data

    def get_exceptions(self):
        hist_data = self.hist_data
        var_data = self.var_data
        count = []
        for i in range(len(var_data), 0, -1):
            temp_hist = hist_data[-i:-i + 252].values.flatten()
            temp_var = var_data[-i:-i + 252].values.flatten()
            count_i = len([1 for i, j in zip(temp_hist, temp_var) if i > j])
            count.append(count_i)
        df = pd.DataFrame(index=var_data.index, data=count[-len(var_data):])
        return df

    def plot_exceptions(self, count):
        hist_data = self.hist_data
        var_data = self.var_data

        self.hist_data = hist_data.filter(items=hist_data.index.intersection(var_data.index), axis=0)
        self.var_data = var_data.filter(items=hist_data.index, axis=0)
        func = lambda x: datetime.utcfromtimestamp(x.tolist() / 1e9).date()
        self.index = [func(x) for x in self.var_data.index.values]
        count = count.filter(items = hist_data.index, axis = 0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.index, count.values.flatten(), lw=1, label="# of exceptions per year")
        ax.set_ylabel("# of counts")
        ax2 = ax.twinx()
        ax2.plot(self.index, self.hist_data.values.flatten(), lw=1,
                 label=str(self.horizon) + ' day change in portfolio value', color='green')
        ax2.plot(self.index, self.var_data.values.flatten(), lw=1, label='VaR', color='black')
        ax2.set_ylabel("Value")
        title = self.bt_name + ':' + self.var_method
        ax.set_title(title)
        ax.legend(loc=2)
        ax2.legend(loc=1)
        plt.savefig(os.path.join(self.path, 'exceptions.png'))

    def get_tl_test(self, count):
        count = count.values.flatten()
        res = dict()
        n = len(count)
        res['Backtest Window (yrs)'] = np.round(len(count) / 252, 2)
        res['% in Green Zone (0-4 exceptions)'] = np.round(sum([1 for i in count if i <= 4]) / n, 2) * 100
        res['% in Amber Zone (5-9 exceptions)'] = np.round(sum([1 for i in count if i <= 9 if i >= 5]) / n, 2) * 100
        res['% in Rd Zone (10 or more exceptions)'] = np.round(sum([1 for i in count if i >= 10]) / n, 2) * 100
        return res

    def binomial(self, count):
        count_vals = count.values.flatten()
        N = 252
        p = 1 - risk_params['VaR Percentile']
        Z_bin = count_vals - N * p
        Z_bin = np.abs(Z_bin / np.sqrt(N * p * (1 - p)))
        plt.figure()
        plt.plot(count.index, Z_bin, label="statistics")
        plt.axhline(y=2.326, color='r', linestyle='--', lw=0.5, label="99% CL")
        plt.axhline(y=1.645, color='b', linestyle='--', lw=0.5, label="95% CL")
        plt.title('Binomial Test Result')
        plt.legend()
        plt.savefig(os.path.join(self.path, 'binomial_test.png'))

    def POF(self, count):
        count_vals = count.values.flatten()
        N = 252
        p = 1 - risk_params['VaR Percentile']
        num1 = [(p ** x) * ((1 - p) ** (252 - x)) for x in count_vals]
        denom1 = [((1 - (x / 252)) ** (252 - x)) * ((x / 252) ** x) for x in count_vals]
        LR_pof = [-2 * np.log(x / y) for x, y in zip(num1, denom1)]
        plt.figure()
        plt.plot(count.index, LR_pof, label='statistics')
        plt.axhline(y=6.635, color='r', linestyle='--', lw=0.5, label="99% CL")
        plt.axhline(y=1.645, color='b', linestyle='--', lw=0.5, label="95% CL")
        plt.title("Kupiec's POF test")
        plt.legend()
        plt.savefig(os.path.join(self.path, 'POF_test.png'))

    def TUFF(self, count):
        tuff = []
        for i in range(len(var_data), 0, -1):
            temp_hist = hist_data[-i:-i + 252].values.flatten()
            temp_var = var_data[-i:-i + 252].values.flatten()
            compare = [1 if i > j else 0 for i, j in zip(temp_hist, temp_var)]
            try:
                idx = [i for i, x in enumerate(compare) if x][0]
            except:
                idx = 252
            tuff.append(idx)
        N = 252
        p = 1 - risk_params['VaR Percentile']
        num2 = [((1 - p) ** (v - 1)) * p for v in tuff]
        denom2 = [(1 / v) * ((1 - (1 / v)) ** (v - 1)) if v != 0 else np.nan for v in tuff]
        LR_tuff = [-2 * np.log(x / y) for x, y in zip(num2, denom2)]

        reject99 = sum([1 for x in LR_tuff if x > 6.635]) / len(LR_tuff)
        reject95 = sum([1 for x in LR_tuff if x > 1.645]) / len(LR_tuff)

        return reject99, reject95

if __name__ == '__main__':
    # Setup portfolio and import data (.data/data_config.py and upload data files if using historical data)
    data_params = load_config("data")
    check_data_config(data_params)
    pf_type = data_params["portfolio_type"]  # get portfolio type
    print("Portfolio type is ", pf_type)
    if pf_type in [1,2,3]:
        num_stock, stock_use, pf_use = setup(data_params, pf_type)[:3]
    else:
        num_stock, stock_use, pf_use, option_use = setup(data_params, pf_type)

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
        mc_result = system.cal_mc_var(pf_type, data_params)
    if sys_params['hist_model']:
        hist_result = system.cal_hist_var(pf_type, pf_use['log_rtn'])

    ####################################################################################################################
    """ Start of Backtest section """
    test_name = 'long_10_5d_5_win'
    if sys_params["param_config"]["assumption"] == "gbm":
        """run test 1,2,3"""
        backtest = BacktestCreator(test_name, param_result, 'paramgbm')
        risk_params = backtest.get_risk_params(sys_params, data_params)
        backtest.get_port_losses()  # must go before get_exceptions for indices alignment
        hist_data, var_data = backtest.return_data()
        count = backtest.get_exceptions()
        backtest.plot_exceptions(count)
        tl_test = backtest.get_tl_test(count)
        backtest.binomial(count)
        backtest.POF(count)
        reject99, reject95 = backtest.TUFF(count)
        print('paramgbm:')
        print(reject95, reject99)
        print(tl_test)

        backtest = BacktestCreator(test_name, hist_result, 'hist')
        risk_params = backtest.get_risk_params(sys_params, data_params)
        backtest.get_port_losses()  # must go before get_exceptions for indices alignment
        hist_data, var_data = backtest.return_data()
        count = backtest.get_exceptions()
        backtest.plot_exceptions(count)
        tl_test = backtest.get_tl_test(count)
        backtest.binomial(count)
        backtest.POF(count)
        reject99, reject95 = backtest.TUFF(count)
        print('historical:')
        print(reject95, reject99)
        print(tl_test)

        backtest = BacktestCreator(test_name, mc_result, 'mcgbm')
        risk_params = backtest.get_risk_params(sys_params, data_params)
        backtest.get_port_losses()  # must go before get_exceptions for indices alignment
        hist_data, var_data = backtest.return_data()
        count = backtest.get_exceptions()
        backtest.plot_exceptions(count)
        tl_test = backtest.get_tl_test(count)
        backtest.binomial(count)
        backtest.POF(count)
        reject99, reject95 = backtest.TUFF(count)
        print('mcgbm:')
        print(reject95, reject99)
        print(tl_test)

    else:
        """run parametric normal assumption test only"""
        backtest = BacktestCreator(test_name, param_result, 'paramnormal')
        risk_params = backtest.get_risk_params(sys_params, data_params)
        backtest.get_port_losses()  # must go before get_exceptions for indices alignment
        hist_data, var_data = backtest.return_data()
        count = backtest.get_exceptions()
        backtest.plot_exceptions(count)
        tl_test = backtest.get_tl_test(count)
        backtest.binomial(count)
        backtest.POF(count)
        reject99, reject95 = backtest.TUFF(count)
        print('paramnormal:')
        print(reject95, reject99)
        print(tl_test)
