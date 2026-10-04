"""Value-at-Risk and Expected Shortfall models.

All functions take a 1-D array of returns (or P&L) and return the risk number as a
POSITIVE loss, i.e. VaR_99 = 0.021 means "99% of the time we lose less than 2.1%".

Rolling forecasts are strictly out-of-sample: the forecast for day t only uses
observations up to t-1, which is what a VaR backtest requires.
"""
import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns, alpha=0.99):
    r = np.asarray(returns, dtype=float)
    return -np.quantile(r, 1 - alpha)


def historical_es(returns, alpha=0.99):
    r = np.asarray(returns, dtype=float)
    q = np.quantile(r, 1 - alpha)
    return -r[r <= q].mean()


def parametric_var(returns, alpha=0.99):
    """Variance-covariance (delta-normal) VaR."""
    r = np.asarray(returns, dtype=float)
    return -(r.mean() + stats.norm.ppf(1 - alpha) * r.std(ddof=1))


def parametric_es(returns, alpha=0.99):
    r = np.asarray(returns, dtype=float)
    z = stats.norm.ppf(1 - alpha)
    return -(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / (1 - alpha))


def cornish_fisher_var(returns, alpha=0.99):
    """Modified VaR: adjusts the normal quantile for skew and excess kurtosis."""
    r = np.asarray(returns, dtype=float)
    z = stats.norm.ppf(1 - alpha)
    s = stats.skew(r)
    k = stats.kurtosis(r)  # excess kurtosis
    z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36)
    return -(r.mean() + z_cf * r.std(ddof=1))


def ewma_volatility(returns, lam=0.94):
    """RiskMetrics EWMA volatility. sigma[t] is the forecast for day t made at t-1."""
    r = np.asarray(returns, dtype=float)
    var = np.empty_like(r)
    var[0] = r[: min(30, len(r))].var()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(var)


def filtered_historical_var(returns, alpha=0.99, lam=0.94, window=250):
    """Filtered HS (Barone-Adesi): rescale past standardised residuals by today's EWMA vol."""
    r = np.asarray(returns, dtype=float)
    sigma = ewma_volatility(r, lam)
    z = r / sigma
    return -np.quantile(z[-window:], 1 - alpha) * sigma[-1]


METHODS = {
    "historical": historical_var,
    "parametric": parametric_var,
    "cornish_fisher": cornish_fisher_var,
}


def rolling_var(returns, method="historical", alpha=0.99, window=250, lam=0.94):
    """Out-of-sample rolling VaR forecasts. Entry t uses returns[t-window : t]."""
    r = pd.Series(returns, dtype=float).reset_index(drop=True)
    out = pd.Series(np.nan, index=r.index)

    if method == "ewma":
        sigma = ewma_volatility(r.values, lam)
        z = stats.norm.ppf(1 - alpha)
        out.iloc[window:] = -z * sigma[window:]
        return out

    if method == "fhs":
        sigma = ewma_volatility(r.values, lam)
        std_resid = r.values / sigma
        for t in range(window, len(r)):
            out.iloc[t] = -np.quantile(std_resid[t - window:t], 1 - alpha) * sigma[t]
        return out

    fn = METHODS[method]
    for t in range(window, len(r)):
        out.iloc[t] = fn(r.values[t - window:t], alpha)
    return out


def rolling_es(returns, alpha=0.975, window=250):
    r = pd.Series(returns, dtype=float).reset_index(drop=True)
    out = pd.Series(np.nan, index=r.index)
    for t in range(window, len(r)):
        out.iloc[t] = historical_es(r.values[t - window:t], alpha)
    return out
