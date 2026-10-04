# rl_quant: regime-aware RL trading with a model-risk framework

An intraday trading pipeline (regime labels → XGBoost regime probabilities → PPO agent),
plus an independent **risk layer** that measures, backtests and stress-tests the strategy
the way a bank's risk methodology team would validate a model.

```
price data ──► label.py ──► base_models_4fold.py / train_model.py ──► rl_model.py ──► risk_report.py
               regimes      XGBoost P(regime | features)              PPO agent       VaR / ES / backtest / stress
```

## Quick start

```bash
pip install -r requirements.txt
python risk_report.py --demo        # runs anywhere; writes reports/risk_report.md
python -m pytest -q tests           # unit tests for the risk library
```

The demo uses a synthetic **regime-switching GARCH(1,1) market with Student-t shocks**
(fat tails + volatility clustering), so it runs without the proprietary tick data.

## 1. Trading pipeline (`sub1/`)

| File | What it does |
|---|---|
| `label.py` | Labels each tick with a 5-class forward-return regime (bull / weak bull / sideways / weak bear / bear) for 40 horizons, h = 50…2000 ticks. |
| `base_models_4fold.py` | XGBoost per horizon with 4-fold **chronological** CV (out-of-fold probabilities for days 1–200, so the RL agent never trains on in-sample predictions). |
| `train_model.py` | Refits on days 1–200 and predicts days 201–228 (true hold-out). |
| `rl_model.py` | Gymnasium environment + PPO agent (short / flat / long). The reward includes transaction costs, minimum-hold and anti-churn penalties. Features are z-scored on an **expanding window** to avoid look-ahead bias. Writes out-of-sample per-step returns for the risk report. |

## 2. Risk framework (`risk/`)

| Module | Contents |
|---|---|
| `var_models.py` | Historical, Parametric (delta-normal), Cornish-Fisher (modified), EWMA/RiskMetrics, Filtered Historical Simulation VaR; Historical & Parametric ES; strictly out-of-sample rolling forecasts |
| `backtesting.py` | Exception counting, **Kupiec POF** (unconditional coverage), **Christoffersen** independence and conditional-coverage tests, **Basel traffic-light** zones (binomial, sample-size scaled) |
| `stress.py` | Historical shock scenarios (1987, Lehman, Flash Crash, COVID) on the worst exposure held; volatility-regime stress (1×–3× vol) |
| `performance.py` | Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover, time in market |

`risk_report.py` combines these into a model-monitoring report: [`reports/risk_report.md`](reports/risk_report.md).

### Demo result (99% VaR, 2,250 out-of-sample days)

| Model | Exceptions (exp. 22.5) | Kupiec p | Christoffersen p | Basel zone |
|---|---|---|---|---|
| Historical | 28 | 0.26 | 0.40 | GREEN ✅ |
| Parametric | 40 | 0.001 | 0.23 | YELLOW ❌ |
| Cornish-Fisher | 27 | 0.36 | 0.003 | GREEN ❌ (clustered) |
| EWMA (normal) | 48 | <0.001 | 0.98 | RED ❌ |
| Filtered HS | 37 | 0.005 | 0.15 | YELLOW ❌ |

![VaR backtest](reports/var_backtest.png)

What the results show:
- **Normal-based models fail on fat tails.** Parametric and EWMA-normal VaR get the volatility right but use a Gaussian quantile, so they have about 2× the expected number of exceptions.
- **Correcting the quantile doesn't fix the timing.** Cornish-Fisher gets the exception *count* right, but its exceptions cluster after volatility spikes because it reacts slowly. Only the Christoffersen test catches this, which is why a count-based traffic light alone isn't enough.
- **No single test is sufficient.** A model should pass both coverage and independence before it is approved.

## Running on real data

```bash
cd sub1
python label.py && python base_models_4fold.py && python train_model.py
python rl_model.py                          # trains PPO, writes ../data/results/ppo_test_returns.csv
cd ..
python risk_report.py --returns data/results/ppo_test_returns.csv --window 500
```
