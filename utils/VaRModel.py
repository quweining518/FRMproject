from utils.funcs import *
from datetime import datetime
# from datetime import timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

class varmodel(object):
    version = '1.0'

    def __init__(self, params):
        risk_config = params['risk_config']
        self.horizon = risk_config['horizon']/risk_config['tradedays']
        self.start = datetime.strptime(risk_config["start"])
        self.end = datetime.strptime(risk_config["end"])
        self.pvar = risk_config["var_percentile"]
        self.pes = risk_config["es_percentile"]
        self.calib_win = params["calib_window"]
        self.calib_lambda = params["calib_lambda"]
        self.calib_weight = 1 if params["calib_weighting"] == "unweighting" else 2
        self.params = params
        self.plot_figure = params["plot_figure"]
        self.save_output = params["save_output"]
        self.dt = 1/risk_config['tradedays']

        # result initialization
        self.calib_params = dict()
        self.param_result = pd.DataFrame()
        self.hist_result = dict()
        self.mc_result = dict()

    def param_calibration(self, N, stock_handle, pf_handle):

        tickers = list(stock_handle.columns)[:N]
        self.tickers = tickers
        self.stock_handle = stock_handle
        length = self.calib_win
        lambd = self.calib_lambda
        dt = self.dt
        res_all = dict.fromkeys(['drift', 'volatility'], [])
        for i in range(N):
            print(tickers[i])
            if self.calib_weight == 1:
                df_drift, df_vol = drift_vol(stock_handle.iloc[:,N+i], stock_handle.iloc[:,2*N+i],
                                             dt, length, 0, type='window')
            else:
                df_drift, df_vol = drift_vol(stock_handle.iloc[:,N+i], stock_handle.iloc[:,2*N+i],
                                             dt, 0, lambd, type='equiv')
            res_all['drift'].append(df_drift)
            res_all['volatility'].append(df_vol)
        if self.calib_weight == 1:
            df_drift, df_vol = drift_vol(pf_handle['log_rtn'], pf_handle['log_rtn_sq'],
                                         dt, length, 0, type='window')
        elif self.calib_weight == 2:
            df_drift, df_vol = drift_vol(pf_handle['log_rtn'], pf_handle['log_rtn_sq'],
                                         dt, 0, lambd, type='equiv')
        res_all['drift'].append(pd.DataFrame(df_drift, columns=['portfolio']))
        res_all['volatility'].append(pd.DataFrame(df_vol, columns=['portfolio']))

        res_all['drift'] = pd.concat(res_all['drift'], axis=1)
        res_all['volatility'] = pd.concat(res_all['volatility'], axis = 1)
        self.calib_params = res_all


    def cal_param_var(self, data_params):

        V_0 = data_params["total_position"]
        assumption = self.params["param_config"]["assumption"]
        startdate = self.start

        res_all = pd.DataFrame(columns=['param_VaR', 'param_ES'])
        if assumption == "gbm":
            res_all["param_VaR"] = param_var(V_0, self.horizon, self.pvar,
                                       self.calib_params["drift"].loc[startdate, "portfolio"],
                                       self.calib_params["volatility"].loc[startdate, "portfolio"])
            res_all["param_ES"] = param_es(V_0, self.horizon, self.pes,
                                       self.calib_params["drift"].loc[startdate, "portfolio"],
                                       self.calib_params["volatility"].loc[startdate, "portfolio"])
        else:
            N = len(self.tickers)
            P_0 = self.stock_handle.loc[startdate, :].iloc[:N].values
            weight = [1/N] * N if data_params["stock_config"]["weight"] == "equal" else data_params["stock_config"]["custom_weight"]
            V_each = V_0 * np.array(weight)
            Q_0 = V_each / P_0

            z1 = 2.326
            z2 = 1.96

        self.param_result = res_all

        if self.plot_figure:
            # plot VaR
            plot_output(res_all, "Parametric VaR and ES", 'parametric_all')

        if self.save_output:
            with pd.ExcelWriter("../output/result_parametric.xlsx", date_format="YYYY-MM-DD") as writer:
                res_all.to_excel(writer)

        return res_all