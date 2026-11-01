import numpy as np
import pytest

from quant_backtester.options import (
    bs_greeks, bs_price, crr_price, implied_volatility, mc_asian_price, mc_price,
)

ARGS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


# ---------- Black-Scholes ----------
def test_reference_values():
    assert bs_price(**ARGS, option="call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(**ARGS, option="put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S,K,T,r,sigma,q", [
    (100, 90, 0.5, 0.03, 0.25, 0.0), (80, 100, 2.0, 0.01, 0.4, 0.02), (120, 100, 0.25, 0.05, 0.15, 0.01),
])
def test_put_call_parity(S, K, T, r, sigma, q):
    c, p = (bs_price(S, K, T, r, sigma, o, q) for o in ("call", "put"))
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_expiry_is_intrinsic_value():
    assert bs_price(110, 100, 0, 0.05, 0.2, "call") == pytest.approx(10.0)
    assert bs_price(110, 100, 0, 0.05, 0.2, "put") == pytest.approx(0.0)


def test_price_increases_with_volatility():
    assert bs_price(**{**ARGS, "sigma": 0.4}) > bs_price(**ARGS)


@pytest.mark.parametrize("option", ["call", "put"])
def test_greeks_match_finite_differences(option):
    g = bs_greeks(**ARGS, option=option, q=0.01)

    def price(**over):
        return bs_price(**{**ARGS, **over}, option=option, q=0.01)

    h = 1e-4
    assert g["delta"] == pytest.approx((price(S=100 + h) - price(S=100 - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((price(S=100 + 0.1) - 2 * price() + price(S=100 - 0.1)) / 0.01, rel=1e-3)
    assert g["vega"] == pytest.approx((price(sigma=0.2 + h) - price(sigma=0.2 - h)) / (2 * h), rel=1e-5)
    assert g["theta"] == pytest.approx(-(price(T=1 + h) - price(T=1 - h)) / (2 * h), rel=1e-4)
    assert g["rho"] == pytest.approx((price(r=0.05 + h) - price(r=0.05 - h)) / (2 * h), rel=1e-5)


def test_delta_bounds():
    assert 0 < bs_greeks(**ARGS, option="call")["delta"] < 1
    assert -1 < bs_greeks(**ARGS, option="put")["delta"] < 0


def test_implied_volatility_round_trip():
    market = bs_price(**{**ARGS, "sigma": 0.3}, option="put")
    assert implied_volatility(market, 100, 100, 1.0, 0.05, "put") == pytest.approx(0.3, abs=1e-8)


def test_implied_volatility_rejects_arbitrage_price():
    with pytest.raises(ValueError):
        implied_volatility(0.5, 100, 90, 1.0, 0.05, "call")  # sous la valeur intrinseque actualisee


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bs_price(**ARGS, option="straddle")
    with pytest.raises(ValueError):
        bs_price(-1, 100, 1, 0.05, 0.2)


def test_vectorized_over_spot():
    spots = np.array([80.0, 100.0, 120.0])
    assert bs_price(spots, 100, 1, 0.05, 0.2).shape == (3,)


# ---------- arbre binomial ----------
@pytest.mark.parametrize("option", ["call", "put"])
def test_binomial_converges_to_black_scholes(option):
    assert crr_price(**ARGS, option=option, steps=2000) == pytest.approx(bs_price(**ARGS, option=option), abs=0.01)


def test_american_put_worth_more_than_european():
    amer = crr_price(**ARGS, option="put", steps=500, american=True)
    euro = crr_price(**ARGS, option="put", steps=500)
    assert amer > euro


def test_american_call_without_dividends_equals_european():
    amer = crr_price(**ARGS, option="call", steps=300, american=True)
    euro = crr_price(**ARGS, option="call", steps=300)
    assert amer == pytest.approx(euro, abs=1e-9)


def test_american_put_dominates_intrinsic_deep_in_the_money():
    price = crr_price(50, 100, 1.0, 0.05, 0.2, "put", steps=300, american=True)
    assert price >= 50 - 1e-9


# ---------- Monte Carlo ----------
@pytest.mark.parametrize("option", ["call", "put"])
def test_monte_carlo_within_four_standard_errors(option):
    price, se = mc_price(**ARGS, option=option, n_paths=200_000, seed=1)
    assert abs(price - bs_price(**ARGS, option=option)) < 4 * se


def test_monte_carlo_is_reproducible_with_seed():
    assert mc_price(**ARGS, seed=7) == mc_price(**ARGS, seed=7)


def test_asian_call_cheaper_than_european_call():
    asian, se = mc_asian_price(**ARGS, n_paths=20_000, n_steps=50, seed=3)
    assert asian + 4 * se < bs_price(**ARGS, option="call")
