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
        self.horizon = risk_config['horizon']/params['tradedays']
        self.start = datetime.strptime(risk_config["start"], "%Y-%m-%d")
        self.end = datetime.strptime(risk_config["end"], "%Y-%m-%d")
        self.pvar = risk_config["var_percentile"]
        self.pes = risk_config["es_percentile"]
        self.calib_win = params["calib_window"]
        self.calib_lambda = params["calib_lambda"]
        self.calib_weight = 1 if params["calib_weighting"] == "unweighting" else 2
        self.params = params
        self.plot_figure = params["plot_figure"]
        self.save_output = params["save_output"]
        self.dt = 1/params['tradedays']

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
        all_drift = pd.DataFrame(columns = tickers)
        all_volatility = pd.DataFrame(columns = tickers)
        for i in range(N):
            print(tickers[i])
            if self.calib_weight == 1:
                df_drift, df_vol = drift_vol(stock_handle.iloc[:,N+i], stock_handle.iloc[:,2*N+i],
                                             dt, length, 0, type='window')
            else:
                df_drift, df_vol = drift_vol(stock_handle.iloc[:,N+i], stock_handle.iloc[:,2*N+i],
                                             dt, 0, lambd, type='equiv')
            all_drift[tickers[i]] = df_drift
            all_volatility[tickers[i]] = df_vol

        if self.calib_weight == 1:
            df_drift, df_vol = drift_vol(pf_handle['log_rtn'], pf_handle['log_rtn_sq'],
                                         dt, length, 0, type='window')
        elif self.calib_weight == 2:
            df_drift, df_vol = drift_vol(pf_handle['log_rtn'], pf_handle['log_rtn_sq'],
                                         dt, 0, lambd, type='equiv')
        all_drift["portfolio"] = df_drift
        all_volatility["portfolio"] = df_vol
        self.calib_drift = all_drift
        self.calib_vol = all_volatility


    def cal_param_var(self, data_params):
        V_0 = data_params["total_position"]
        assumption = self.params["param_config"]["assumption"]
        startdate = self.start
        enddate = self.end

        res_all = pd.DataFrame(columns=['param_VaR', 'param_ES'])
        if assumption == "gbm":
            res_all["param_VaR"] = param_var(V_0, self.horizon, self.pvar,
                                       self.calib_drift.loc[startdate:enddate, "portfolio"],
                                       self.calib_vol.loc[startdate:enddate, "portfolio"])
            res_all["param_ES"] = param_es(V_0, self.horizon, self.pes,
                                       self.calib_drift.loc[startdate:enddate, "portfolio"],
                                       self.calib_vol.loc[startdate:enddate, "portfolio"])
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
            with pd.ExcelWriter(r"./output/result_parametric.xlsx", date_format="YYYY-MM-DD") as writer:
                res_all.to_excel(writer)

        return res_all

    def cal_hist_var(self, data_params, pf_log_rtn):
        V_0 = data_params["total_position"]
        startdate = self.start
        enddate = self.end

        res_all = pd.DataFrame(columns=['hist_VaR', 'hist_ES'])
        res_all["hist_VaR"] = historical_var(pf_log_rtn, V_0, self.pvar, self.dt, self.calib_win)
        res_all["hist_ES"] = historical_es(pf_log_rtn, V_0, self.pes, self.dt, self.calib_win)
        self.hist_result = res_all.loc[startdate:enddate,:]

        if self.plot_figure:
            # plot VaR
            plot_output(res_all.loc[startdate:enddate,:], "Historical VaR and ES", 'historical_all')
        if self.save_output:
            with pd.ExcelWriter(r"./output/result_historical.xlsx", date_format="YYYY-MM-DD") as writer:
                res_all.loc[startdate:enddate,:].to_excel(writer)

        return res_all
