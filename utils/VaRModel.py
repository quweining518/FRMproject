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
        length, lambd, dt = self.calib_win, self.calib_lambda, self.dt
        all_drift = pd.DataFrame(columns = tickers)
        all_volatility = pd.DataFrame(columns = tickers)
        for i in range(N):
            if self.calib_weight == 1:
                df_drift, df_vol = drift_vol(stock_handle.iloc[:,N+i], stock_handle.iloc[:,2*N+i],
                                             dt, length, 0, type='window')
            else:
                df_drift, df_vol = drift_vol(stock_handle.iloc[:,N+i], stock_handle.iloc[:,2*N+i],
                                             dt, 0, lambd, type='equiv')
            all_drift[tickers[i]], all_volatility[tickers[i]] = df_drift, df_vol
        if self.calib_weight == 1:
            df_drift, df_vol = drift_vol(pf_handle['log_rtn'], pf_handle['log_rtn_sq'],
                                         dt, length, 0, type='window')
        elif self.calib_weight == 2:
            df_drift, df_vol = drift_vol(pf_handle['log_rtn'], pf_handle['log_rtn_sq'],
                                         dt, 0, lambd, type='equiv')
        all_drift["portfolio"], all_volatility["portfolio"] = df_drift, df_vol

        cov_all = []
        corr_all = []
        if self.calib_weight == 1:
            for i in range(N):
                for j in range(i+1,N):
                    cov = covariance(stock_handle.iloc[:,N+i], stock_handle.iloc[:,N+j], self.dt,
                                     length, 0, type = "window")
                    cov_all.append(cov)
                    corr = correlation(cov, all_volatility.iloc[:,i], all_volatility.iloc[:,j], self.dt)
                    corr_all.append(corr)
        elif self.calib_weight == 2:
            for i in range(N):
                for j in range(i+1,N):
                    cov = covariance(stock_handle.iloc[:,N+i], stock_handle.iloc[:,N+j], self.dt,
                                     0, lambd, type = "equiv")
                    cov_all.append(cov)
                    corr = correlation(cov, all_volatility.iloc[:, i], all_volatility.iloc[:, j], self.dt)
                    corr_all.append(corr)
        covs = pd.concat(cov_all, axis=1)
        covs.columns = [x for x in range(N*(N-1)//2)]
        corrs = pd.concat(corr_all, axis=1)
        corrs.columns = [x for x in range(N*(N-1)//2)]

        self.calib_drift, self.calib_vol = all_drift, all_volatility
        self.calib_cov = covs
        self.calib_corr = corrs


    def cal_param_var(self, pf_type, data_params):
        V_0 = data_params["total_position"]
        assumption = self.params["param_config"]["assumption"]
        startdate, enddate = self.start, self.end
        res_all = pd.DataFrame(columns=['param_VaR', 'param_ES'])
        if assumption == "gbm":
            if pf_type == 1:
                res_all["param_VaR"] = param_var(V_0, self.horizon, self.pvar,
                                           self.calib_drift.loc[startdate:enddate, "portfolio"],
                                           self.calib_vol.loc[startdate:enddate, "portfolio"])
                res_all["param_ES"] = param_es(V_0, self.horizon, self.pes,
                                           self.calib_drift.loc[startdate:enddate, "portfolio"],
                                           self.calib_vol.loc[startdate:enddate, "portfolio"])
            elif pf_type == 2:
                res_all["param_VaR"] = var_short(V_0, self.horizon, self.pvar,
                                                 self.calib_drift.loc[startdate:enddate, "portfolio"],
                                                 self.calib_vol.loc[startdate:enddate, "portfolio"])
                res_all["param_ES"] = es_short(V_0, self.horizon, self.pes,
                                               self.calib_drift.loc[startdate:enddate, "portfolio"],
                                               self.calib_vol.loc[startdate:enddate, "portfolio"])
            elif pf_type == 3:
                pass

        else:
            z_var = norm.ppf(self.pvar)
            z_es = norm.ppf(self.pes)

            if pf_type == 1:
                tickers = data_params["stock_config"]["long_tickers"]
                N = len(tickers)
                P_0 = self.stock_handle.loc[startdate, :].iloc[:N].values
                weight = [1/N] * N if data_params["stock_config"]["long_weight"] == "equal" else data_params[
                                    "stock_config"]["long_custom_weight"]
                V_each = V_0 * np.array(weight)

                cum_evt = 0
                cum_evt2 = 0
                s = 0
                for i in range(N):
                    mu_i = self.calib_drift.loc[startdate:enddate, tickers[i]]
                    vol_i = self.calib_vol.loc[startdate:enddate, tickers[i]]
                    cum_evt += V_each[i] * np.exp(mu_i * self.horizon)
                    cum_evt2 += V_each[i]**2 * np.exp((2*mu_i+vol_i**2) * self.horizon)
                    for j in range(i+1, N):
                        corr_ij = self.calib_corr[s]
                        mu_j = self.calib_drift.loc[startdate:enddate, tickers[j]]
                        vol_j = self.calib_vol.loc[startdate:enddate, tickers[j]]
                        cum_evt2 += 2*V_each[i]*V_each[j]*np.exp((mu_i + mu_j + corr_ij*vol_j*vol_i) * self.horizon)
                        s += 1

                var_vt = cum_evt2 - cum_evt**2
                res_all["param_VaR"] = V_0 - (cum_evt - z_var * np.sqrt(var_vt))
                res_all["param_ES"] = V_0 - (cum_evt - np.sqrt(var_vt)/(1-self.pes) * np.exp(-z_es**2/2)/np.sqrt(2*np.pi))

        self.param_result = res_all

        if self.plot_figure:
            plot_output(res_all, "Parametric VaR and ES", 'parametric_all')
        if self.save_output:
            with pd.ExcelWriter(r"./output/result_parametric.xlsx", date_format="YYYY-MM-DD") as writer:
                res_all.to_excel(writer)
        return res_all

    def cal_hist_var(self, pf_type, data_params, pf_log_rtn):
        V_0 = data_params["total_position"]
        startdate, enddate = self.start, self.end
        res_all = pd.DataFrame(columns=['hist_VaR', 'hist_ES'])
        res_all["hist_VaR"] = historical_var(pf_log_rtn, V_0, self.pvar, self.dt, self.calib_win)
        res_all["hist_ES"] = historical_es(pf_log_rtn, V_0, self.pes, self.dt, self.calib_win)
        self.hist_result = res_all.loc[startdate:enddate,:]

        if self.plot_figure:
            plot_output(res_all.loc[startdate:enddate,:], "Historical VaR and ES", 'historical_all')
        if self.save_output:
            with pd.ExcelWriter(r"./output/result_historical.xlsx", date_format="YYYY-MM-DD") as writer:
                res_all.loc[startdate:enddate,:].to_excel(writer)
        return res_all

    def cal_mc_var(self, pf_type, data_params):
        V_0 = data_params["total_position"]
        startdate, enddate = self.start, self.end
        res_all = pd.DataFrame(columns = ['mc_VaR', 'mc_ES'])
        assumption = self.params["param_config"]["assumption"]
        n_paths = self.params["mc_config"]["n_paths"]
        if assumption == "gbm":
            pf_var = np.zeros_like(self.calib_drift.loc[startdate:enddate, "portfolio"])
            pf_es = np.zeros_like(self.calib_drift.loc[startdate:enddate, "portfolio"])
            print(len(pf_var))
            print(int(self.horizon/self.dt))
            dateindex = list(self.calib_drift.loc[startdate:enddate,:].index)
            for i in range(len(pf_var)):
                pf_var[i] = mc_var_es(self.dt, int(self.horizon/self.dt), n_paths, V_0,
                                      self.calib_vol.loc[dateindex[i], "portfolio"],
                                      self.calib_drift.loc[dateindex[i], "portfolio"], self.pvar, stat='var')[1]
                pf_es[i] = mc_var_es(self.dt, int(self.horizon/self.dt), n_paths, V_0,
                                      self.calib_vol.loc[dateindex[i], "portfolio"],
                                      self.calib_drift.loc[dateindex[i], "portfolio"], self.pes, stat='es')[1]
            res_all["mc_VaR"] = pf_var
            res_all["mc_ES"] = pf_es
        else:
            N = len(self.tickers)

        self.param_result = res_all

        if self.plot_figure:
            plot_output(res_all, "Monte Carlo VaR and ES", 'mc_all')
        if self.save_output:
            with pd.ExcelWriter(r"./output/result_montecarlo.xlsx", date_format="YYYY-MM-DD") as writer:
                res_all.to_excel(writer)
        return res_all


