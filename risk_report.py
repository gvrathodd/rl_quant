"""Model-risk report for a strategy's P&L: VaR/ES, backtests, stress tests, performance.

Usage
-----
    python risk_report.py --demo                                   # synthetic data, runs anywhere
    python risk_report.py --returns data/results/ppo_test_returns.csv --window 500

The returns CSV needs a `net_return` column (and optionally `position`), e.g. the
output of sub1/rl_model.py.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risk import backtesting as bt
from risk import performance, stress, var_models

VAR_MODELS = {
    "Historical":     "historical",
    "Parametric":     "parametric",
    "Cornish-Fisher": "cornish_fisher",
    "EWMA (lambda=0.94)":  "ewma",
    "Filtered HS":    "fhs",
}


def synthetic_strategy(n=2500, seed=7):
    """Regime-switching market with GARCH(1,1) vol and Student-t shocks, traded by a
    simple trend rule. Gives fat tails + vol clustering, which is what breaks naive VaR."""
    rng = np.random.default_rng(seed)
    regimes = np.zeros(n, dtype=int)
    trans = np.array([[0.98, 0.02], [0.05, 0.95]])  # calm <-> stressed
    for t in range(1, n):
        regimes[t] = rng.choice(2, p=trans[regimes[t - 1]])

    omega, a, b = 2e-6, 0.08, 0.90
    var = np.full(n, omega / (1 - a - b))
    r = np.zeros(n)
    drift = np.where(regimes == 0, 4e-4, -8e-4)
    shocks = rng.standard_t(df=4, size=n) / np.sqrt(2)  # unit variance
    eps = np.zeros(n)  # GARCH innovation, kept separate so the regime scale can't make it explode
    for t in range(1, n):
        var[t] = omega + a * eps[t - 1] ** 2 + b * var[t - 1]
        eps[t] = np.sqrt(var[t]) * shocks[t]
        scale = 1.0 if regimes[t] == 0 else 1.8
        r[t] = drift[t] + scale * eps[t]

    prices = 100 * np.cumprod(1 + r)
    fast = pd.Series(prices).ewm(span=10).mean()
    slow = pd.Series(prices).ewm(span=50).mean()
    position = np.sign(fast - slow).shift(1).fillna(0).values  # decide at t-1
    fee = 3e-4 * np.abs(np.diff(position, prepend=0))
    strat = position * r - fee
    return pd.DataFrame({"net_return": strat, "position": position, "market_return": r})


def plot_backtest(returns, forecasts, hits_model, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(range(len(returns)), returns, width=1.0, color="#9aa5b1", label="P&L (return)")
    colors = ["#d64545", "#2f80ed", "#f2994a", "#27ae60", "#8e44ad"]
    for (name, f), c in zip(forecasts.items(), colors):
        ax.plot(-f.values, lw=1.1, color=c, label=f"-VaR {name}")
    f = forecasts[hits_model].values
    idx = np.where(returns < -f)[0]
    ax.scatter(idx, returns[idx], color="black", s=14, zorder=5,
               label=f"Exceptions ({hits_model})")
    ax.set_title("99% VaR backtest: out-of-sample forecasts vs realised P&L")
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    ax.set_xlabel("Period")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity(returns, path):
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(equity, color="#2f80ed")
    a1.set_title("Equity curve")
    a2.fill_between(range(len(dd)), dd, 0, color="#d64545", alpha=0.5)
    a2.set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt(df, floatfmt="{:.4f}"):
    return df.to_markdown(floatfmt=".4f") if hasattr(df, "to_markdown") else str(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns", help="CSV with a net_return column")
    ap.add_argument("--demo", action="store_true", help="use synthetic data")
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--window", type=int, default=250)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    if args.demo or not args.returns:
        df = synthetic_strategy()
        source = "synthetic regime-switching GARCH-t market, EMA(10/50) trend strategy"
    else:
        df = pd.read_csv(args.returns)
        source = args.returns

    os.makedirs(args.out, exist_ok=True)
    r = df["net_return"].values
    pos = df["position"].values if "position" in df else None
    a, w = args.alpha, args.window

    # 1. VaR forecasts + backtests
    forecasts, results = {}, []
    for name, method in VAR_MODELS.items():
        f = var_models.rolling_var(r, method=method, alpha=a, window=w)
        forecasts[name] = f.iloc[w:].reset_index(drop=True)
        results.append(bt.backtest_var(r[w:], forecasts[name].values, a, model=name))
    table = bt.results_table(results)

    es = var_models.rolling_es(r, alpha=0.975, window=w).iloc[w:]
    realised_tail = r[w:][r[w:] < -forecasts["Historical"].values]

    # 2. Stress
    shock = stress.worst_case_shock(pos) if pos is not None else None
    vol = stress.vol_stress(r, alpha=a)

    # 3. Performance
    perf = performance.summary(r, pos)

    oos = r[w:]
    plot_backtest(oos, forecasts, "Historical", os.path.join(args.out, "var_backtest.png"))
    plot_equity(r, os.path.join(args.out, "equity_drawdown.png"))

    best = table[table["passed"]].sort_values("cc_p", ascending=False)
    verdict = (f"**{best.index[0]}** passes both Kupiec and Christoffersen at 5% "
               f"and is the recommended model." if len(best)
               else "**No model passes both tests** - the VaR model needs recalibration.")

    lines = [
        "# Strategy Risk & Model Performance Report",
        "",
        f"- Data: {source}",
        f"- Observations: {len(r)} (backtest window: {len(oos)} out-of-sample, {w}-period estimation window)",
        f"- Confidence: VaR {a:.0%}, ES 97.5%",
        "",
        "## 1. VaR backtest",
        "",
        "Exceptions = periods where the realised loss exceeded the previous period's VaR forecast.",
        "Kupiec POF tests the exception *rate*; Christoffersen tests whether exceptions *cluster*;",
        "Basel traffic-light zones are scaled to the sample size.",
        "",
        table[["exceptions", "expected", "exception_rate", "kupiec_p",
               "christoffersen_p", "cc_p", "traffic_light", "passed"]].to_markdown(floatfmt=".4f"),
        "",
        verdict,
        "",
        "![VaR backtest](var_backtest.png)",
        "",
        "## 2. Expected Shortfall (97.5%)",
        "",
        f"- Mean ES forecast: {es.mean():.4%}",
        f"- Mean realised loss on Historical-VaR exception days: "
        f"{(-realised_tail.mean() if len(realised_tail) else float('nan')):.4%}",
        "",
        "## 3. Stress testing",
        "",
        "### Volatility stress (returns re-scaled around the mean)",
        "",
        vol.to_markdown(floatfmt=".4f"),
        "",
    ]
    if shock is not None:
        lines += ["### Instantaneous shock scenarios (worst exposure held)", "",
                  shock.to_markdown(floatfmt=".4f"), ""]
    lines += [
        "## 4. Strategy performance",
        "",
        perf.to_frame("value").to_markdown(floatfmt=".4f"),
        "",
        "![Equity and drawdown](equity_drawdown.png)",
        "",
    ]
    path = os.path.join(args.out, "risk_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(table[["exceptions", "expected", "kupiec_p", "christoffersen_p", "traffic_light", "passed"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nReport written to {path}")


if __name__ == "__main__":
    main()
