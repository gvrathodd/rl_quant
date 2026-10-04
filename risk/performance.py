"""Strategy performance metrics."""
import numpy as np
import pandas as pd


def sharpe(returns, periods=252, rf=0.0):
    r = np.asarray(returns, dtype=float) - rf / periods
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else np.sqrt(periods) * r.mean() / sd


def sortino(returns, periods=252, rf=0.0):
    r = np.asarray(returns, dtype=float) - rf / periods
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return 0.0 if downside == 0 else np.sqrt(periods) * r.mean() / downside


def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def calmar(returns, periods=252):
    r = np.asarray(returns, dtype=float)
    equity = np.cumprod(1 + r)
    ann_ret = equity[-1] ** (periods / len(r)) - 1
    mdd = abs(max_drawdown(equity))
    return 0.0 if mdd == 0 else ann_ret / mdd


def summary(returns, positions=None, periods=252):
    r = np.asarray(returns, dtype=float)
    equity = np.cumprod(1 + r)
    out = {
        "total_return": equity[-1] - 1,
        "ann_return": equity[-1] ** (periods / len(r)) - 1,
        "ann_vol": r.std(ddof=1) * np.sqrt(periods),
        "sharpe": sharpe(r, periods),
        "sortino": sortino(r, periods),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(r, periods),
        "hit_rate": float((r[r != 0] > 0).mean()) if (r != 0).any() else 0.0,
    }
    if positions is not None:
        p = np.asarray(positions, dtype=float)
        out["turnover"] = float(np.abs(np.diff(p)).sum() / len(p))
        out["time_in_market"] = float((p != 0).mean())
    return pd.Series(out)
