"""Estimating the fill-intensity curve lambda(delta) = A exp(-k delta) from data.

A market maker never sees A and k; it sees how often quotes at different
distances get filled. With quotes held at distance delta_i for a time tau,
the number of fills is Poisson with mean A exp(-k delta_i) tau. This module
fits (A, k) by maximum likelihood and gives Wald confidence intervals from
the Fisher information, which a test checks for correct coverage.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def simulate_fill_counts(A, k, deltas, tau, rng):
    return rng.poisson(A * np.exp(-k * np.asarray(deltas)) * tau)


def neg_log_likelihood(theta, deltas, counts, tau):
    """theta = (log A, k). Poisson log-likelihood without the constant term."""
    log_a, k = theta
    log_lam = log_a - k * deltas + np.log(tau)
    return -(counts * log_lam - np.exp(log_lam)).sum()


def fit_intensity(deltas, counts, tau, conf=0.95):
    """MLE of (A, k) with Wald confidence intervals.

    Parameterised as (log A, k) so the likelihood is concave. The Fisher
    information is sum_i lambda_i tau [[1, -d_i], [-d_i, d_i^2]].
    Returns dict with A, k, their CIs, and the covariance of (log A, k).
    """
    deltas = np.asarray(deltas, float)
    counts = np.asarray(counts, float)
    # Starting point from a log-linear least-squares fit.
    pos = counts > 0
    slope, intercept = np.polyfit(deltas[pos], np.log(counts[pos] / tau), 1)
    res = minimize(neg_log_likelihood, x0=[intercept, -slope], args=(deltas, counts, tau),
                   method="Newton-CG", jac=_grad, hess=_hess)
    log_a, k = res.x
    lam_tau = np.exp(log_a - k * deltas) * tau
    info = np.array([[lam_tau.sum(), -(lam_tau * deltas).sum()],
                     [-(lam_tau * deltas).sum(), (lam_tau * deltas**2).sum()]])
    cov = np.linalg.inv(info)
    z = norm.ppf(0.5 + conf / 2)
    se_log_a, se_k = np.sqrt(np.diag(cov))
    return {
        "A": float(np.exp(log_a)), "k": float(k),
        "A_ci": (float(np.exp(log_a - z * se_log_a)), float(np.exp(log_a + z * se_log_a))),
        "k_ci": (float(k - z * se_k), float(k + z * se_k)),
        "cov": cov, "converged": bool(res.success),
    }


def _grad(theta, deltas, counts, tau):
    log_a, k = theta
    lam_tau = np.exp(log_a - k * deltas) * tau
    r = counts - lam_tau
    return -np.array([r.sum(), -(r * deltas).sum()])


def _hess(theta, deltas, counts, tau):
    log_a, k = theta
    lam_tau = np.exp(log_a - k * deltas) * tau
    return np.array([[lam_tau.sum(), -(lam_tau * deltas).sum()],
                     [-(lam_tau * deltas).sum(), (lam_tau * deltas**2).sum()]])
