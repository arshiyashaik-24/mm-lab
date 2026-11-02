"""Market making under inventory risk.

Model (Avellaneda & Stoikov, Quant. Finance 8(3), 2008):
    mid price      dS = mu_t dt + sigma dW
    order arrivals a quote at distance delta from the mid is filled with
                   intensity lambda(delta) = A exp(-k delta)
    objective      maximise E[-exp(-gamma (X_T + q_T S_T))]  (CARA utility)

Strategies:
    * ``as_quotes``: the Avellaneda-Stoikov asymptotic approximation
      (reservation price and spread);
    * ``symmetric_quotes``: the same spread centred on the mid (benchmark);
    * ``GLFTSolution``: the exact optimum with an inventory limit, from
      Gueant, Lehalle & Fernandez-Tapia (Math. Financ. Econ. 7, 2013), where
      the HJB equation reduces to a linear ODE system dw/dt = M w solved by
      a matrix exponential.

Everything is vectorised over simulation paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Params:
    """Defaults are the parameters of Avellaneda & Stoikov (2008), section 4."""

    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    A: float = 140.0
    k: float = 1.5
    gamma: float = 0.1

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))


# --------------------------------------------------------------------------
# Quote rules: each returns (delta_bid, delta_ask), distances below/above mid.
# --------------------------------------------------------------------------
def as_spread(p: Params, tau):
    """Total AS spread gamma sigma^2 tau + (2/gamma) ln(1 + gamma/k)."""
    return p.gamma * p.sigma**2 * tau + 2.0 / p.gamma * np.log1p(p.gamma / p.k)


def as_quotes(p: Params, q, tau, drift_term=0.0):
    """Avellaneda-Stoikov: quote symmetrically around the reservation price
    r = s + drift_term - q gamma sigma^2 tau (drift_term is the expected price
    change over the remaining horizon, zero without a signal)."""
    shift = drift_term - q * p.gamma * p.sigma**2 * tau  # r - s
    half = 0.5 * as_spread(p, tau)
    # bid = r - half -> delta_bid = s - bid = half - shift; ask = r + half.
    return half - shift, half + shift


def symmetric_quotes(p: Params, q, tau, drift_term=0.0):
    half = 0.5 * as_spread(p, tau)
    return np.full_like(np.asarray(q, float), half), np.full_like(np.asarray(q, float), half)


@dataclass
class GLFTSolution:
    """Exact optimal quotes with inventory constrained to [-q_max, q_max].

    Substituting theta(t, q) = (1/k) ln w(t, q) into the HJB equation gives
        dw_q/dt = alpha q^2 w_q - eta (w_{q+1} + w_{q-1}),   w(T, q) = 1,
    with alpha = k gamma sigma^2 / 2 and eta = A (1 + gamma/k)^-(1 + k/gamma).
    So w(t) = exp(-M (T - t)) 1 with M symmetric tridiagonal, solved here by
    eigendecomposition. Optimal distances:
        delta_bid(t, q) = (1/k) ln(w_q / w_{q+1}) + (1/gamma) ln(1 + gamma/k)
        delta_ask(t, q) = (1/k) ln(w_q / w_{q-1}) + (1/gamma) ln(1 + gamma/k)
    """

    p: Params
    q_max: int = 25
    qs: np.ndarray = field(init=False)

    def __post_init__(self):
        p = self.p
        self.qs = np.arange(-self.q_max, self.q_max + 1)
        alpha = p.k * p.gamma * p.sigma**2 / 2.0
        self.eta = p.A * (1.0 + p.gamma / p.k) ** (-(1.0 + p.k / p.gamma))
        n = len(self.qs)
        M = np.diag(alpha * self.qs.astype(float) ** 2)
        M -= self.eta * (np.eye(n, k=1) + np.eye(n, k=-1))
        self.M = M
        self.eigval, self.eigvec = np.linalg.eigh(M)
        self.c = np.log1p(p.gamma / p.k) / p.gamma

    def w(self, tau):
        """w(T - tau) for scalar tau >= 0, normalised (only ratios matter)."""
        coeff = self.eigvec.T @ np.ones(len(self.qs))
        lam = self.eigval - self.eigval.min()  # rescale to avoid overflow
        return self.eigvec @ (np.exp(-lam * tau) * coeff)

    def quote_table(self, tau):
        """(delta_bid, delta_ask) for every inventory level at time-to-go tau.
        NaN where the inventory limit forbids quoting that side."""
        # At extreme inventories w underflows (those states are practically
        # unreachable); a positive floor keeps the log finite there.
        w = np.maximum(self.w(tau), 1e-300)
        lw = np.log(w)
        bid = np.full(len(w), np.nan)
        ask = np.full(len(w), np.nan)
        bid[:-1] = (lw[:-1] - lw[1:]) / self.p.k + self.c
        ask[1:] = (lw[1:] - lw[:-1]) / self.p.k + self.c
        return bid, ask

    def quotes(self, q, tau):
        bid, ask = self.quote_table(tau)
        idx = np.clip(np.asarray(q, int) + self.q_max, 0, len(self.qs) - 1)
        return bid[idx], ask[idx]


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@dataclass
class DriftModel:
    """Optional short-term predictability: an Ornstein-Uhlenbeck drift
        d mu = -kappa mu dt + nu dB,
    which also tilts order flow (more aggressive buyers when mu > 0):
        lambda_ask = A e^{-k delta_a} (1 + beta mu),  lambda_bid = A e^{-k delta_b} (1 - beta mu).
    The market maker observes z = mu + noise with variance obs_var / dt per step.
    """

    kappa: float = 10.0          # signal half-life ln2/kappa ~ 0.07 T
    nu: float = 6.0 * np.sqrt(20.0)  # stationary drift std 6 (3x sigma per unit time)
    beta: float = 0.1            # order-flow tilt: informed buyers when mu > 0
    obs_var: float = 1.0         # signal noise (filter error ~ half the prior variance)

    @property
    def stationary_var(self):
        return self.nu**2 / (2 * self.kappa)

    def expected_move(self, mu, tau):
        """E[S_{t+tau} - S_t | mu_t] = mu (1 - e^{-kappa tau}) / kappa."""
        return mu * (1.0 - np.exp(-self.kappa * tau)) / self.kappa


def simulate(p: Params, strategy, n_paths=1000, seed=0, drift: DriftModel | None = None,
             estimator: str = "none"):
    """Simulate a market maker.

    Args:
        strategy: "as", "symmetric", or a GLFTSolution.
        drift: None for the driftless AS model, or a DriftModel.
        estimator: how the quotes use the drift signal: "none" (ignore it),
            "kalman" (filtered estimate from the noisy observations), or
            "oracle" (true mu, an upper bound on the value of the signal).
    Returns a dict of per-path arrays: pnl, q (final inventory), n_fills,
    mean_spread, plus 'cash' and 'max_abs_q'.
    """
    rng = np.random.default_rng(seed)
    dt, n = p.dt, p.n_steps
    s = np.full(n_paths, p.s0)
    x = np.zeros(n_paths)
    q = np.zeros(n_paths, dtype=int)
    fills = np.zeros(n_paths, dtype=int)
    spread_sum = np.zeros(n_paths)
    max_abs_q = np.zeros(n_paths, dtype=int)

    mu = np.zeros(n_paths)
    if drift is not None:
        mu = rng.normal(0.0, np.sqrt(drift.stationary_var), n_paths)
        mu_hat = np.zeros(n_paths)
        P = np.full(n_paths, drift.stationary_var)  # Kalman variance

    for i in range(n):
        tau = p.T - i * dt
        drift_term = 0.0
        if drift is not None and estimator != "none":
            m = mu if estimator == "oracle" else mu_hat
            drift_term = drift.expected_move(m, tau)
        if isinstance(strategy, GLFTSolution):
            db, da = strategy.quotes(q, tau)
            if drift is not None:
                # Expected rise shifts both quotes up: bid closer, ask further.
                db, da = db - drift_term, da + drift_term
        elif strategy == "as":
            db, da = as_quotes(p, q, tau, drift_term)
        elif strategy == "symmetric":
            db, da = symmetric_quotes(p, q, tau, drift_term)
        else:
            raise ValueError(strategy)

        lam_b = p.A * np.exp(-p.k * np.nan_to_num(db, nan=np.inf))
        lam_a = p.A * np.exp(-p.k * np.nan_to_num(da, nan=np.inf))
        if drift is not None:
            lam_a = lam_a * np.clip(1 + drift.beta * mu, 0, None)
            lam_b = lam_b * np.clip(1 - drift.beta * mu, 0, None)
        hit_b = rng.random(n_paths) < np.minimum(lam_b * dt, 1.0)
        hit_a = rng.random(n_paths) < np.minimum(lam_a * dt, 1.0)

        bid = s - np.nan_to_num(db, nan=0.0)
        ask = s + np.nan_to_num(da, nan=0.0)
        x += np.where(hit_a, ask, 0.0) - np.where(hit_b, bid, 0.0)
        q += hit_b.astype(int) - hit_a.astype(int)
        fills += hit_a.astype(int) + hit_b.astype(int)
        spread_sum += np.nan_to_num(da + db, nan=0.0)
        max_abs_q = np.maximum(max_abs_q, np.abs(q))

        # Price and drift move.
        s = s + mu * dt + p.sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        if drift is not None:
            mu_new = mu - drift.kappa * mu * dt + drift.nu * np.sqrt(dt) * rng.standard_normal(n_paths)
            # Noisy observation of the new drift, then a Kalman update.
            z = mu_new + np.sqrt(drift.obs_var / dt) * rng.standard_normal(n_paths)
            a = 1 - drift.kappa * dt
            mu_pred = a * mu_hat
            P_pred = a * a * P + drift.nu**2 * dt
            R = drift.obs_var / dt
            K = P_pred / (P_pred + R)
            mu_hat = mu_pred + K * (z - mu_pred)
            P = (1 - K) * P_pred
            mu = mu_new

    return {"pnl": x + q * s, "q": q, "n_fills": fills, "mean_spread": spread_sum / n,
            "cash": x, "max_abs_q": max_abs_q}


def certainty_equivalent(pnl, gamma):
    """CE = -(1/gamma) ln E[exp(-gamma pnl)], computed stably."""
    z = -gamma * np.asarray(pnl)
    m = z.max()
    return -(m + np.log(np.mean(np.exp(z - m)))) / gamma


def certainty_equivalent_se(pnl, gamma, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl)
    boots = [certainty_equivalent(rng.choice(pnl, len(pnl)), gamma) for _ in range(n_boot)]
    return float(np.std(boots))
