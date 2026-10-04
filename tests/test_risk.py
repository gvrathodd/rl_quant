import numpy as np
import pytest
from scipy import stats

from risk import backtesting as bt
from risk import performance, stress, var_models


@pytest.fixture
def normal_returns():
    return np.random.default_rng(0).normal(0, 0.01, 100_000)


def test_parametric_var_matches_closed_form(normal_returns):
    expected = -stats.norm.ppf(0.01) * 0.01
    assert var_models.parametric_var(normal_returns, 0.99) == pytest.approx(expected, rel=0.02)


def test_historical_var_and_es_on_normal(normal_returns):
    var = var_models.historical_var(normal_returns, 0.99)
    es = var_models.historical_es(normal_returns, 0.99)
    assert var == pytest.approx(0.02326, rel=0.03)
    assert es > var
    assert es == pytest.approx(var_models.parametric_es(normal_returns, 0.99), rel=0.03)


def test_cornish_fisher_equals_normal_without_skew_or_kurtosis(normal_returns):
    assert var_models.cornish_fisher_var(normal_returns) == pytest.approx(
        var_models.parametric_var(normal_returns), rel=0.02)


def test_rolling_var_is_out_of_sample():
    r = np.random.default_rng(1).normal(0, 0.01, 400)
    shocked = r.copy()
    shocked[300] = -0.5  # a huge loss must not change its own forecast, only later ones
    f = var_models.rolling_var(r, "historical", window=250)
    g = var_models.rolling_var(shocked, "historical", window=250)
    assert np.isnan(f.iloc[249])
    assert f.iloc[300] == g.iloc[300]
    assert g.iloc[301] > f.iloc[301]


def test_kupiec_accepts_correct_rate_and_rejects_wrong_rate():
    hits = np.zeros(1000, dtype=int)
    hits[::100] = 1  # exactly 1%
    assert bt.kupiec_pof(hits, 0.99)[1] > 0.5
    hits[::20] = 1   # 5%
    assert bt.kupiec_pof(hits, 0.99)[1] < 0.01


def test_christoffersen_detects_clustering():
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1
    clustered = np.zeros(1000, dtype=int)
    clustered[500:510] = 1
    assert bt.christoffersen_independence(spread)[1] > 0.05
    assert bt.christoffersen_independence(clustered)[1] < 0.01


@pytest.mark.parametrize("n_exc,zone", [(4, "GREEN"), (5, "YELLOW"), (9, "YELLOW"), (10, "RED")])
def test_basel_traffic_light_matches_regulatory_table(n_exc, zone):
    assert bt.basel_traffic_light(n_exc, 250, 0.99) == zone


def test_max_drawdown():
    assert performance.max_drawdown([1, 2, 1, 3]) == pytest.approx(-0.5)


def test_vol_stress_increases_var(normal_returns):
    table = stress.vol_stress(normal_returns[:2000])
    assert table.iloc[:, 0].is_monotonic_increasing
