"""VaR backtesting: exception counting, Kupiec POF, Christoffersen independence,
conditional coverage, and the Basel traffic-light zones."""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


def exceptions(returns, var_forecast):
    """1 where the realised loss exceeded the VaR forecast (VaR given as a positive loss)."""
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var_forecast, dtype=float)
    mask = ~np.isnan(v)
    return (r[mask] < -v[mask]).astype(int)


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits, alpha=0.99):
    """Proportion-of-failures likelihood ratio test (unconditional coverage).
    H0: exception rate == 1 - alpha. LR ~ chi2(1)."""
    hits = np.asarray(hits)
    n, x = len(hits), int(hits.sum())
    p = 1 - alpha
    p_hat = x / n
    ll_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1 - p_hat) + _xlogy(x, p_hat)
    lr = -2 * (ll_null - ll_alt)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_independence(hits):
    """Tests whether exceptions cluster (first-order Markov). LR ~ chi2(1)."""
    h = np.asarray(hits)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = (_xlogy(n00, 1 - pi0) + _xlogy(n01, pi0)
              + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1))
    lr = -2 * (ll_null - ll_alt)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def basel_traffic_light(n_exceptions, n_obs=250, alpha=0.99):
    """Basel zones scaled to the sample size via the binomial CDF
    (green < 95%, yellow < 99.99%, red otherwise; equals 0-4 / 5-9 / 10+ at n=250)."""
    cdf = stats.binom.cdf(n_exceptions, n_obs, 1 - alpha)
    if cdf < 0.95:
        return "GREEN"
    if cdf < 0.9999:
        return "YELLOW"
    return "RED"


@dataclass
class BacktestResult:
    model: str
    n_obs: int
    exceptions: int
    expected: float
    exception_rate: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    cc_lr: float
    cc_p: float
    traffic_light: str

    @property
    def passed(self):
        return self.kupiec_p > 0.05 and self.christoffersen_p > 0.05


def backtest_var(returns, var_forecast, alpha=0.99, model="VaR"):
    hits = exceptions(returns, var_forecast)
    n, x = len(hits), int(hits.sum())
    pof_lr, pof_p = kupiec_pof(hits, alpha)
    ind_lr, ind_p = christoffersen_independence(hits)
    cc_lr = pof_lr + ind_lr
    return BacktestResult(
        model=model,
        n_obs=n,
        exceptions=x,
        expected=n * (1 - alpha),
        exception_rate=x / n,
        kupiec_lr=pof_lr,
        kupiec_p=pof_p,
        christoffersen_lr=ind_lr,
        christoffersen_p=ind_p,
        cc_lr=cc_lr,
        cc_p=1 - stats.chi2.cdf(cc_lr, 2),
        traffic_light=basel_traffic_light(x, n, alpha),
    )


def results_table(results):
    df = pd.DataFrame([asdict(r) | {"passed": r.passed} for r in results])
    return df.set_index("model")
