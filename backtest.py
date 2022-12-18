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
        curr_time1 = curr_time.strftime('%Y/%m/%d %H:%M:%s')
        print("Backtested at", curr_time1)
        curr_time2 = curr_time.strftime('%Y%m%d_%H%M')
        self.path = os.path.join('./backtest', bt_name + "_" + curr_time2)
        os.makedirs(os.path.join('./backtest', bt_name + "_" + curr_time2))

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
        startdate = self.sdate - timedelta(days=3 * horizon)
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

        df = yf.download(tickers, start=startdate, end=enddate)['Adj Close']
        n_shares = [S0[i] / df[tickers[i]][0] for i in range(n)]

        if n > 1:
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
        self.hist_data = df_losses.filter(items=df_losses.index.intersection(self.var_data.index), axis=0)
        self.var_data = self.var_data.filter(items=self.hist_data.index, axis=0)
        func = lambda x: datetime.utcfromtimestamp(x.tolist() / 1e9).date()
        self.index = [func(x) for x in self.var_data.index.values]

    def return_data(self):
        return self.hist_data, self.var_data

    def get_exceptions(self):
        hist_data = self.hist_data
        var_data = self.var_data
        count = []
        for i in range(len(var_data)):
            temp_hist = hist_data[i:i + 252].values.flatten()
            temp_var = var_data[i:i + 252].values.flatten()
            count_i = len([1 for i, j in zip(temp_hist, temp_var) if i > j])
            count.append(count_i)
        return hist_data.index, count

    def plot_exceptions(self, count):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.index, count, lw=1, label="# of exceptions per year")
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
        res = dict()
        n = len(count)
        res['Backtest Window (yrs)'] = np.round(len(count) / 252, 2)
        res['% in Green Zone (0-4 exceptions)'] = np.round(sum([1 for i in count if i <= 4]) / n, 2) * 100
        res['% in Amber Zone (5-9 exceptions)'] = np.round(sum([1 for i in count if i <= 9 if i >= 5]) / n, 2) * 100
        res['% in Rd Zone (10 or more exceptions)'] = np.round(sum([1 for i in count if i >= 10]) / n, 2) * 100
        return res

    def get_accuracy_test(self, count):
        n = len(count)
        x = sum([1 for i in count if i <= 2.5])
        N = len(count)
        p = 1 - risk_params['VaR Percentile']
        Z_bin = x - N * p
        Z_bin = Z_bin / np.sqrt(N * p * (1 - p))
        yrs = N / 252
        x_py = x / yrs
        num1 = (p ** x_py) * ((1 - p) ** (252 - x_py))
        denom1 = ((1 - (x_py / 252)) ** (252 - x_py)) * ((x_py / 252) ** x_py)
        LR_pof = -2 * np.log(num1 / denom1)
        lst = []
        for i in range(n - 251):
            temp = count[i:i + 252]
            v = [i for i, x in enumerate(temp) if x][0]
            lst.append(v)
        v = np.average(lst)
        num2 = ((1 - p) ** (v - 1)) * p
        denom2 = (1 / v) * ((1 - (1 / v)) ** (v - 1))
        LR_tuff = -2 * np.log(num2 / denom2)

        res = dict()
        res['Binomial Test Stats'] = Z_bin
        res["Kupiec's POF Test Stats"] = LR_pof
        res["Kupiec's TUFF Test Stats"] = LR_tuff
        return res

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
    sys_params['plot_fiture'] = 0
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
        hist_result = system.cal_hist_var(pf_type, data_params, pf_use['log_rtn'])

    backtest = BacktestCreator('test', mc_result, 'Parametric VaR')
    risk_params = backtest.get_risk_params(sys_params, data_params)
    backtest.get_port_losses() #must go before get_exceptions for indices alignment
    hist_data, var_data = backtest.return_data()
    index, count = backtest.get_exceptions()
    backtest.plot_exceptions(count)
    res1 = backtest.get_tl_test(count)
    res2 = backtest.get_accuracy_test(count)