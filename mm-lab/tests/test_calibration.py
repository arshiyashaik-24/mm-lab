"""Maximum-likelihood calibration: accuracy and honest confidence intervals."""
import numpy as np
import pytest

from mmlab.calibration import fit_intensity, simulate_fill_counts

A, K = 140.0, 1.5
DELTAS = np.linspace(0.2, 2.0, 10)
TAU = 0.5


def test_mle_recovers_parameters_with_lots_of_data():
    rng = np.random.default_rng(0)
    counts = simulate_fill_counts(A, K, DELTAS, 200.0, rng)
    fit = fit_intensity(DELTAS, counts, 200.0)
    assert fit["converged"]
    assert fit["A"] == pytest.approx(A, rel=0.02)
    assert fit["k"] == pytest.approx(K, rel=0.02)


def test_confidence_intervals_have_nominal_coverage():
    # Over many independent experiments, 95% intervals should contain the
    # true value about 95% of the time (binomial sd ~ 1% for 500 trials).
    rng = np.random.default_rng(1)
    hit_a = hit_k = 0
    n = 500
    for _ in range(n):
        fit = fit_intensity(DELTAS, simulate_fill_counts(A, K, DELTAS, TAU, rng), TAU)
        hit_a += fit["A_ci"][0] <= A <= fit["A_ci"][1]
        hit_k += fit["k_ci"][0] <= K <= fit["k_ci"][1]
    assert 0.92 <= hit_a / n <= 0.98
    assert 0.92 <= hit_k / n <= 0.98


def test_interval_width_shrinks_like_one_over_root_n():
    rng = np.random.default_rng(2)
    w = []
    for tau in (1.0, 16.0):
        fit = fit_intensity(DELTAS, simulate_fill_counts(A, K, DELTAS, tau, rng), tau)
        w.append(fit["k_ci"][1] - fit["k_ci"][0])
    assert w[0] / w[1] == pytest.approx(4.0, rel=0.15)
