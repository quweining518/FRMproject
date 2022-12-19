import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

def param_var(x, T, p, mu, vol):
    var = x - x*np.exp(vol*np.sqrt(T)*norm.ppf(1-p) + (mu-vol**2/2)*T)
    return var
def param_es(x, T, p, mu, vol):
    k = x - param_var(x, T, p, mu, vol)
    d1 = (np.log(x/k) + (mu + vol**2/2)*T)/(vol*np.sqrt(T))
    es = x - 1/(1-p)*np.exp(mu*T)*x*(1-norm.cdf(d1))
    return es
def es_short(x, T, p, mu, vol):
    return (- param_es(x, T, 0, mu, vol) + p*param_es(x, T, 1-p, mu, vol))/(1-p)

def var_short(x, T, p, mu, vol):
    return - param_var(x, T, 1-p, mu, vol)

def historical_var(df, notional, p, dt, window):
    df1 = df.rolling(window*int(1/dt)).apply(lambda x: np.percentile(x, 100*(1-p)))
    return - notional * df1

def historical_es(df, notional, p, dt, window):
    def cal_es(x):
        threshold = np.percentile(x, 100*(1-p))
        tail = x[x <= threshold]
        es = np.mean(tail)
        return es
    df1 = df.rolling(window*int(1/dt)).apply(lambda x: cal_es(x))
    return - notional * df1

def covariance(rtn1, rtn2, dt, window, lambd, max_win=5000, type='window'):
    """calculate covariance between two stocks"""
    if type == 'window':
        mean1 = rtn1.rolling(min(max_win, int(1/dt) * window)).mean()
        mean2 = rtn2.rolling(min(max_win, int(1/dt) * window)).mean()
        rtn12 = rtn1 * rtn2
        cov = rtn12.rolling(min(max_win, int(1/dt) * window)).mean() - mean1 * mean2
    else:
        expo = np.array([np.power(lambd, i) for i in range(max_win)])
        mean1 = rtn1.rolling(min_periods=1, window=max_win).apply(lambda x: rolling_weights(x, expo))
        mean2 = rtn2.rolling(min_periods=1, window=max_win).apply(lambda x: rolling_weights(x, expo))
        norm_rtn1 = rtn1 - mean1
        norm_rtn2 = rtn2 - mean2
        norm_rtn12 = norm_rtn1 * norm_rtn2
        cov = norm_rtn12.rolling(min_periods=1, window=max_win).apply(lambda x: rolling_weights(x, expo))
    return cov

def correlation(cov, vol1, vol2, dt):
    """calculate correlation between two stocks"""
    corr = cov / (vol1 * vol2 * dt)
    return corr

def mc_var_es(dt, T, n_paths, S0, vol, drift, p, longboolean = True, stat='var'):
    """
    :param dt: the time step used (e.g: 1/252)
    :param T: (integer) the length of time steps to simulate
    :param n_paths: (integer) simulation paths
    :param S0: the original price or portfolio value
    :param vol: calibrated volatility for gbm
    :param drift: calibrated drift for gbm
    :param p: the percentile of threshold; the portfolio loss is less than threhold 100*p% of time
    :param longboolean: long - True; short - False
    :param stat: "var" or "es"
    :return:
    """
    paths = np.full((T, n_paths), np.nan, dtype=np.float)
    paths[0] = S0
    for i in range(T - 1):
        dW = np.sqrt(dt) * np.random.randn(n_paths)
        paths[i + 1] = paths[i] * np.exp((drift - 1 / 2 * vol ** 2) * dt + vol * dW)

    pl = S0 - paths[-1] # pnl = V0 - VT
    # For long portfolio, find the 100*p% percentile paths (or mean above for ES).
    # For short portfolio, find the 100*(1-p)% paths (or below for ES).
    if stat == 'var':
        pp = p if longboolean else 1-p
        result = np.abs(np.percentile(pl, 100 * pp))
    else:
        if longboolean:
            threshold = np.percentile(pl, 100 * p)
            tail = pl[pl >= threshold]
            result = np.mean(tail)
        else:
            threshold = np.percentile(pl, 100 * (1-p))
            tail = pl[pl <= threshold]
            result = np.abs(np.mean(tail))
    return pl, result


def rolling_weights(x, expo):
    expo = expo[:len(x)][::-1]  # reverse weight
    weights = expo / np.sum(expo)
    return np.sum(weights * x)

def drift_vol(log_rtn, log_rtn_sq, dt, window, lambd, max_win=5000, type='window'):
    """
    df: [price, log_rtn, log_rtn_sq] as chronological order
    window: int (year)
    lambda: float [0,1]
    type: window | equiv
    """
    if type == 'window':
        samp_mean = log_rtn.rolling(min(max_win, int(1/dt) * window)).mean()
        samp_std = np.sqrt(log_rtn_sq.rolling(min(max_win, int(1/dt) * window)).mean() - samp_mean ** 2)
    else:
        expo = np.array([np.power(lambd, i) for i in range(max_win)])
        samp_mean = log_rtn.rolling(min_periods=1, window=max_win).apply(
            lambda x: rolling_weights(x, expo))
        samp_std = np.sqrt(log_rtn_sq.rolling(min_periods=1, window=max_win).apply(
            lambda x: rolling_weights(x, expo)) - samp_mean ** 2)

    volatility = samp_std / np.sqrt(dt)
    drift = samp_mean / dt + volatility ** 2 / 2
    return drift, volatility

def plot_output(res, title, filename, figsize = (10,6)):
    plt.figure(figsize=figsize)
    labels = list(res.columns)
    for col in range(res.shape[1]):
        plt.plot(res.iloc[:,col], label = labels[col])
    plt.title(title)
    plt.xlabel("range of time (t)")
    plt.ylabel("VaR/ES ($)")
    plt.legend()
    plt.grid()
    plt.savefig(r"./output/figure/%s.png" % filename, format = 'png')
    plt.show()
