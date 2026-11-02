"""Generate the figures and a numeric summary.  python examples/make_figures.py"""
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmlab.calibration import fit_intensity, simulate_fill_counts
from mmlab.model import (
    DriftModel, GLFTSolution, Params, as_quotes, certainty_equivalent, certainty_equivalent_se, simulate,
)

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
})
P = Params()
N = 4000
summary = []


def fig_strategies():
    g = GLFTSolution(P, 25)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)

    q = np.arange(-8, 9)
    gb, ga = g.quotes(q, 1.0)
    ab, aa = as_quotes(P, q, 1.0)
    ax = axes[0]
    ax.plot(q, gb, color=BLUE, label="bid, exact (GLFT)")
    ax.plot(q, ga, color=ORANGE, label="ask, exact (GLFT)")
    ax.plot(q, ab, color=BLUE, ls="--", lw=1.4, label="bid, AS approximation")
    ax.plot(q, aa, color=ORANGE, ls="--", lw=1.4, label="ask, AS approximation")
    ax.set(title="(a) Optimal quote distance vs inventory, t = 0", xlabel="Inventory q", ylabel="Distance from mid")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    runs = {}
    for name, st, col in (("Exact (GLFT)", g, BLUE), ("Avellaneda-Stoikov", "as", AQUA), ("Symmetric", "symmetric", ORANGE)):
        r = simulate(P, st, N, seed=7)
        runs[name] = r
        ax.hist(r["pnl"], bins=60, range=(20, 110), histtype="step", lw=1.8, color=col, label=name, density=True)
        ce = certainty_equivalent(r["pnl"], P.gamma)
        se = certainty_equivalent_se(r["pnl"], P.gamma)
        summary.append(f"{name:20s} profit {r['pnl'].mean():6.2f} ± {r['pnl'].std():5.2f}  "
                       f"final q std {r['q'].std():5.2f}  CE {ce:6.2f} ± {se:.2f}")
    ax.set(title="(b) Profit distribution, 4000 paths, γ = 0.1", xlabel="Profit", ylabel="Density")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for name, maker, col in (("Exact (GLFT)", lambda p: GLFTSolution(p, 25), BLUE), ("Avellaneda-Stoikov", lambda p: "as", AQUA)):
        pts = []
        for gam in (0.01, 0.03, 0.1, 0.3, 1.0):
            p = replace(P, gamma=gam)
            r = simulate(p, maker(p), 2000, seed=9)
            pts.append((r["pnl"].std(), r["pnl"].mean()))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], "o-", color=col, ms=4, label=name)
        for (x, y), gam in zip(pts, (0.01, 0.03, 0.1, 0.3, 1.0)):
            if name.startswith("Exact"):
                ax.annotate(f"γ={gam:g}", (x, y), textcoords="offset points", xytext=(5, -10), fontsize=7, color=MUTED)
    s = runs["Symmetric"]["pnl"]
    ax.plot(s.std(), s.mean(), "s", color=ORANGE, ms=6, label="Symmetric, γ = 0.1")
    ax.set(title="(c) Risk-return trade-off as risk aversion varies", xlabel="Profit std", ylabel="Mean profit")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "01_strategies.png", dpi=160)
    plt.close(fig)


def fig_calibration():
    A, k = P.A, P.k
    deltas = np.linspace(0.2, 2.0, 10)
    tau = 0.5
    rng = np.random.default_rng(0)
    counts = simulate_fill_counts(A, k, deltas, tau, rng)
    fit = fit_intensity(deltas, counts, tau)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), constrained_layout=True)
    ax = axes[0]
    dd = np.linspace(0.1, 2.1, 200)
    draws = rng.multivariate_normal([np.log(fit["A"]), fit["k"]], fit["cov"], 2000)
    curves = np.exp(draws[:, :1] - draws[:, 1:] * dd)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    ax.fill_between(dd, lo, hi, color=BLUE, alpha=0.18, lw=0, label="95% band (fit)")
    ax.semilogy(dd, fit["A"] * np.exp(-fit["k"] * dd), color=BLUE, label=f"fit: A = {fit['A']:.0f}, k = {fit['k']:.2f}")
    ax.semilogy(dd, A * np.exp(-k * dd), color=MUTED, ls="--", lw=1.3, label="truth: A = 140, k = 1.5")
    ax.semilogy(deltas, np.maximum(counts, 0.5) / tau, "o", color=ORANGE, ms=5, label="observed fill rate")
    ax.set(title="(a) Fill intensity from 5 time units of quoting", xlabel="Quote distance δ", ylabel="Fills per unit time")
    ax.legend(frameon=False, fontsize=8)

    levels = np.array([0.5, 0.68, 0.8, 0.9, 0.95, 0.99])
    reps = 400
    hits = np.zeros((len(levels), 2))
    rng = np.random.default_rng(1)
    for _ in range(reps):
        c = simulate_fill_counts(A, k, deltas, tau, rng)
        for j, lev in enumerate(levels):
            f = fit_intensity(deltas, c, tau, conf=lev)
            hits[j, 0] += f["A_ci"][0] <= A <= f["A_ci"][1]
            hits[j, 1] += f["k_ci"][0] <= k <= f["k_ci"][1]
    cov = hits / reps
    ax = axes[1]
    ax.plot([0.45, 1], [0.45, 1], color=MUTED, lw=1.2, ls="--", label="perfect calibration")
    ax.plot(levels, cov[:, 0], "o-", color=BLUE, ms=4, label="A")
    ax.plot(levels, cov[:, 1], "o-", color=ORANGE, ms=4, label="k")
    ax.set(title=f"(b) Interval coverage over {reps} repeated experiments", xlabel="Nominal confidence level", ylabel="Observed coverage")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "02_calibration.png", dpi=160)
    plt.close(fig)
    summary.append(f"Calibration: A = {fit['A']:.1f} [{fit['A_ci'][0]:.1f}, {fit['A_ci'][1]:.1f}], "
                   f"k = {fit['k']:.3f} [{fit['k_ci'][0]:.3f}, {fit['k_ci'][1]:.3f}]; "
                   f"95% coverage A {cov[4, 0]:.3f}, k {cov[4, 1]:.3f}")


def fig_signal():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    base = DriftModel()
    noises = np.array([0.05, 0.2, 0.5, 1, 2, 5, 20])
    res = {e: [] for e in ("none", "kalman", "oracle")}
    for ov in noises:
        d = replace(base, obs_var=ov)
        for e in res:
            res[e].append(certainty_equivalent(simulate(P, "as", 3000, seed=21, drift=d, estimator=e)["pnl"], P.gamma))
    no_flow = certainty_equivalent(simulate(P, "as", 3000, seed=21)["pnl"], P.gamma)
    ax = axes[0]
    for e, col, lab in (("oracle", MUTED, "knows the true drift"), ("kalman", BLUE, "Kalman-filtered signal"), ("none", ORANGE, "ignores the signal")):
        ax.semilogx(noises, res[e], marker="o" if e == "kalman" else None, color=col, ms=4, label=lab,
                    ls="--" if e != "kalman" else "-")
    ax.axhline(no_flow, color=AQUA, lw=1.2, ls=":", label="no informed flow")
    ax.set(title="(a) Value of the signal vs its noise", xlabel="Signal noise variance", ylabel="Certainty equivalent")
    ax.legend(frameon=False, fontsize=8, loc="center right")

    betas = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.15])
    cost = {e: [] for e in ("none", "kalman")}
    for b in betas:
        d = replace(base, beta=b)
        for e in cost:
            cost[e].append(certainty_equivalent(simulate(P, "as", 3000, seed=23, drift=d, estimator=e)["pnl"], P.gamma))
    ax = axes[1]
    ax.plot(betas, cost["none"], "o-", color=ORANGE, ms=4, label="ignores the signal")
    ax.plot(betas, cost["kalman"], "o-", color=BLUE, ms=4, label="Kalman-filtered signal")
    ax.set(title="(b) Adverse selection: informed order flow", xlabel="Order-flow tilt β", ylabel="Certainty equivalent")
    ax.legend(frameon=False, fontsize=8)

    rng = np.random.default_rng(4)
    d = base
    n = P.n_steps
    mu, mh, Pv = np.zeros(n), np.zeros(n), d.stationary_var
    m, est = rng.normal(0, np.sqrt(d.stationary_var)), 0.0
    for i in range(n):
        m = m - d.kappa * m * P.dt + d.nu * np.sqrt(P.dt) * rng.standard_normal()
        z = m + np.sqrt(d.obs_var / P.dt) * rng.standard_normal()
        a = 1 - d.kappa * P.dt
        mp, Pp = a * est, a * a * Pv + d.nu**2 * P.dt
        K = Pp / (Pp + d.obs_var / P.dt)
        est, Pv = mp + K * (z - mp), (1 - K) * Pp
        mu[i], mh[i] = m, est
    t = np.arange(n) * P.dt
    ax = axes[2]
    ax.fill_between(t, mh - 2 * np.sqrt(Pv), mh + 2 * np.sqrt(Pv), color=BLUE, alpha=0.15, lw=0, label="±2σ filter band")
    ax.plot(t, mu, color=MUTED, lw=1.4, label="true drift μ")
    ax.plot(t, mh, color=BLUE, lw=1.6, label="Kalman estimate")
    ax.set(title="(c) Tracking the hidden drift (one path)", xlabel="Time", ylabel="Drift")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "03_signal_and_adverse_selection.png", dpi=160)
    plt.close(fig)
    j = int(np.where(noises == 1)[0][0])
    summary.append(f"No informed flow CE {no_flow:.2f}; at beta=0.1, noise 1: ignore {res['none'][j]:.2f}, "
                   f"Kalman {res['kalman'][j]:.2f}, oracle {res['oracle'][j]:.2f}")
    summary.append("Signal-noise sweep (Kalman CE): " + ", ".join(f"{ov:g}:{v:.1f}" for ov, v in zip(noises, res["kalman"])))
    summary.append("beta sweep ignore/Kalman: " + ", ".join(f"{b:g}:{a:.1f}/{c:.1f}" for b, a, c in zip(betas, cost["none"], cost["kalman"])))


if __name__ == "__main__":
    fig_strategies()
    fig_calibration()
    fig_signal()
    print("\n".join(summary))
