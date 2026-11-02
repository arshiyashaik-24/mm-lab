"""Validation of the market-making models against theory and the original paper."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mmlab.model import (
    DriftModel, GLFTSolution, Params, as_quotes, certainty_equivalent,
    certainty_equivalent_se, simulate,
)

P = Params()


# ---- Exact solution -------------------------------------------------------
def test_glft_terminal_quotes_equal_myopic_half_spread():
    g = GLFTSolution(P, 10)
    bid, ask = g.quote_table(0.0)
    c = np.log1p(P.gamma / P.k) / P.gamma
    assert np.allclose(bid[:-1], c) and np.allclose(ask[1:], c)


def test_glft_matches_direct_integration_of_the_hjb():
    """Integrate the nonlinear HJB for theta(t, q) directly and compare quotes.

    0 = d theta/dt - gamma sigma^2 q^2 / 2
        + sum over sides A/(k+gamma) exp(-k delta*),
    delta*_bid = theta(q) - theta(q+1) + c, delta*_ask = theta(q) - theta(q-1) + c.
    """
    g = GLFTSolution(P, 8)
    qs = g.qs
    c = g.c
    A, k, gam, sig = P.A, P.k, P.gamma, P.sigma

    def rhs(tau, th):  # d theta / d tau, with tau = T - t
        db = np.full_like(th, np.inf)
        da = np.full_like(th, np.inf)
        db[:-1] = th[:-1] - th[1:] + c
        da[1:] = th[1:] - th[:-1] + c
        return -0.5 * gam * sig**2 * qs**2 + A / (k + gam) * (np.exp(-k * db) + np.exp(-k * da))

    sol = solve_ivp(rhs, (0, 1.0), np.zeros(len(qs)), rtol=1e-10, atol=1e-12)
    th = sol.y[:, -1]
    bid_direct = th[:-1] - th[1:] + c
    bid_exact, _ = g.quote_table(1.0)
    assert np.max(np.abs(bid_direct - bid_exact[:-1])) < 1e-6


def test_glft_quotes_are_symmetric_in_inventory():
    g = GLFTSolution(P, 12)
    bid, ask = g.quote_table(0.7)
    # delta_bid(q) = delta_ask(-q)
    assert np.allclose(bid[:-1], ask[1:][::-1], atol=1e-10)


def test_long_inventory_lowers_both_quotes():
    g = GLFTSolution(P, 12)
    for quotes in (g.quotes(np.array([0, 3]), 0.8), as_quotes(P, np.array([0, 3]), 0.8)):
        db, da = quotes
        assert db[1] > db[0] and da[1] < da[0]  # wider bid, tighter ask when long


# ---- Simulation -----------------------------------------------------------
def test_fill_rate_matches_intensity():
    # Constant quotes at distance delta: fills per unit time -> A exp(-k delta).
    p = Params(gamma=1e-9)  # near risk-neutral: AS spread is constant in time
    r = simulate(p, "symmetric", n_paths=4000, seed=3)
    delta = 0.5 * r["mean_spread"][0]
    expected = 2 * p.A * np.exp(-p.k * delta) * p.T
    assert r["n_fills"].mean() == pytest.approx(expected, rel=0.02)


def test_reproduces_avellaneda_stoikov_table():
    # Avellaneda & Stoikov (2008), gamma = 0.1, 1000 paths: inventory strategy
    # profit 62.94 (std 5.89), final q std 2.80; symmetric 67.21 (13.43), q std 8.66.
    a = simulate(P, "as", n_paths=4000, seed=7)
    s = simulate(P, "symmetric", n_paths=4000, seed=7)
    assert a["pnl"].mean() == pytest.approx(62.94, rel=0.06)
    assert a["pnl"].std() == pytest.approx(5.89, rel=0.15)
    assert a["q"].std() == pytest.approx(2.80, rel=0.10)
    assert s["pnl"].mean() == pytest.approx(67.21, rel=0.03)
    assert s["pnl"].std() == pytest.approx(13.43, rel=0.06)
    assert s["q"].std() == pytest.approx(8.66, rel=0.06)


def test_exact_solution_is_optimal_in_monte_carlo():
    g = GLFTSolution(P, 25)
    ce = {}
    for name, st in (("glft", g), ("as", "as"), ("sym", "symmetric")):
        pnl = simulate(P, st, n_paths=4000, seed=11)["pnl"]
        ce[name] = (certainty_equivalent(pnl, P.gamma), certainty_equivalent_se(pnl, P.gamma))
    assert ce["glft"][0] > ce["as"][0] > ce["sym"][0]
    assert ce["glft"][0] - ce["as"][0] > 3 * np.hypot(ce["glft"][1], ce["as"][1])


def test_inventory_limit_is_respected():
    g = GLFTSolution(P, 4)
    r = simulate(P, g, n_paths=1000, seed=5)
    assert r["max_abs_q"].max() <= 4


# ---- Drift and filtering --------------------------------------------------
def test_kalman_filter_error_matches_its_own_variance():
    d = DriftModel()
    p = Params()
    rng = np.random.default_rng(0)
    n = 20000
    mu, mu_hat, P_ = 0.0, 0.0, d.stationary_var
    errs = []
    for i in range(n):
        mu = mu - d.kappa * mu * p.dt + d.nu * np.sqrt(p.dt) * rng.standard_normal()
        z = mu + np.sqrt(d.obs_var / p.dt) * rng.standard_normal()
        a = 1 - d.kappa * p.dt
        mp, Pp = a * mu_hat, a * a * P_ + d.nu**2 * p.dt
        K = Pp / (Pp + d.obs_var / p.dt)
        mu_hat, P_ = mp + K * (z - mp), (1 - K) * Pp
        if i > 1000:
            errs.append(mu - mu_hat)
    assert np.var(errs) == pytest.approx(P_, rel=0.1)
    assert P_ < 0.6 * d.stationary_var  # the signal is informative


def test_signal_value_ordering():
    d = DriftModel()
    ce = {}
    for est in ("none", "kalman", "oracle"):
        pnl = simulate(P, "as", n_paths=3000, seed=21, drift=d, estimator=est)["pnl"]
        ce[est] = certainty_equivalent(pnl, P.gamma)
    assert ce["oracle"] > ce["kalman"] > ce["none"]
