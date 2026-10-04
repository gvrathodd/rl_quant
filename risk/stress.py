"""Stress testing of a strategy's daily P&L.

Two kinds of scenario:
  * shock scenarios   - instantaneous price gaps applied to the position held
  * vol-regime stress - re-run the return series with volatility scaled up,
                        to see how VaR and drawdown respond
"""
import numpy as np
import pandas as pd

from .performance import max_drawdown
from .var_models import historical_var, historical_es

# Single-day moves, loosely calibrated to well-known equity sell-offs / rallies.
SHOCK_SCENARIOS = {
    "Black Monday 1987":      -0.205,
    "Lehman (15-Sep-2008)":   -0.047,
    "Flash Crash 2010":       -0.090,
    "COVID crash (16-Mar-20)": -0.120,
    "Short squeeze +10%":     +0.100,
    "Generic -5% gap":        -0.050,
}


def shock_pnl(position, shocks=SHOCK_SCENARIOS, notional=1.0):
    """P&L of the current position under each instantaneous shock.
    position: -1 short, 0 flat, +1 long (or any fractional exposure)."""
    return pd.Series({name: position * move * notional for name, move in shocks.items()},
                     name="pnl")


def worst_case_shock(positions, shocks=SHOCK_SCENARIOS):
    """Worst loss across all scenarios for the largest long and short exposure taken."""
    positions = np.asarray(positions, dtype=float)
    rows = []
    for name, move in shocks.items():
        rows.append({
            "scenario": name,
            "move": move,
            "pnl_max_long": positions.max() * move,
            "pnl_max_short": positions.min() * move,
        })
    df = pd.DataFrame(rows).set_index("scenario")
    df["worst"] = df[["pnl_max_long", "pnl_max_short"]].min(axis=1)
    return df.sort_values("worst")


def vol_stress(returns, multipliers=(1.0, 1.5, 2.0, 3.0), alpha=0.99):
    """Scale demeaned returns by k and recompute risk metrics."""
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    rows = []
    for k in multipliers:
        rk = mu + k * (r - mu)
        rows.append({
            "vol_multiplier": k,
            f"VaR_{int(alpha*100)}": historical_var(rk, alpha),
            "ES_97.5": historical_es(rk, 0.975),
            "max_drawdown": max_drawdown(np.cumprod(1 + rk)),
        })
    return pd.DataFrame(rows).set_index("vol_multiplier")
