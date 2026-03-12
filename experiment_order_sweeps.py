"""
Higher-order-only Monte Carlo sweeps on the Gaussian target.

This script generates two kinds of figures for K=3 and K=4 separately:
1. gamma sweeps at a fixed h,
2. h sweeps at a fixed gamma.

The evaluation is again Monte Carlo based: we simulate particles, compare them
against a Monte Carlo reference cloud from the target Gaussian, and estimate W2
by optimal matching.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment import (
    empirical_w2,
    higher_order_batch_step,
    make_initial_position,
    make_lambdas,
    make_quadratic_grad,
    sample_target_gaussian,
)
from higher_order_langevin_fast import HigherOrderLangevin


# =============================================================================
# Sweep settings.
# =============================================================================

DIMENSION = 2
KAPPA = 20.0
INIT_SHIFT = 5.0
N_PARTICLES = 64
N_REPEATS = 3
OUTPUT_DIR = Path("./gaussian_mc_order_sweeps")

# Gamma sweeps: one plot for K=3 and one plot for K=4.
GAMMA_SWEEPS = {
    3: {
        "h": 0.5,
        "n_steps": 120,
        "gammas": [5, 8.0, 10.0],
    },
    4: {
        "h": 0.5,
        "n_steps": 120,
        "gammas": [2, 5, 10.0],
    },
}

# h sweeps: keep gamma fixed and vary h.
# We do not use h = 0.5 for K=3 here because, with one common gamma, it is too
# easy to enter an unstable regime.  The purpose of this plot is to compare
# stable step sizes, not to show divergence.
H_SWEEPS = {
    3: {
        "gamma": 3.0,
        "h_values": [0.2, 0.1, 0.05],
        "n_steps": [200, 200, 200],
    },
    4: {
        "gamma": 2.5,
        "h_values": [0.2, 0.1, 0.05],
        "n_steps": [200, 200, 200],
    },
}


# =============================================================================
# Monte Carlo runner for one higher-order configuration.
# =============================================================================


def run_ho_curve(
    K: int,
    lambdas: np.ndarray,
    h: float,
    gamma: float,
    n_steps: int,
    n_particles: int,
    n_repeats: int,
    x0: np.ndarray,
) -> np.ndarray:
    """Average W2 curve for one (K, h, gamma) configuration."""
    d = len(lambdas)
    H = np.diag(lambdas)
    grad = make_quadratic_grad(H)

    curves = []
    for rep in range(n_repeats):
        seed = 10_000 * rep
        ref = sample_target_gaussian(n_particles, lambdas, np.random.default_rng(seed + 1000))

        sampler = HigherOrderLangevin(
            K=K,
            d=d,
            h=h,
            gamma=gamma,
            grad_U_fn=grad,
            nu_star=K - 1,
            rng=np.random.default_rng(seed + 1),
        )

        state = np.zeros((n_particles, K * d), dtype=float)
        state[:, :d] = x0
        curve = np.empty(n_steps, dtype=float)
        for t in range(n_steps):
            state = higher_order_batch_step(sampler, state, grad)
            x = state.reshape(n_particles, K, d)[:, 0, :]
            curve[t] = empirical_w2(x, ref)
        curves.append(curve)

    return np.mean(curves, axis=0)


# =============================================================================
# Plotting helpers.
# =============================================================================


def plot_gamma_sweep(K: int, sweep: dict, lambdas: np.ndarray, x0: np.ndarray) -> Path:
    """Plot several gamma values for a fixed order K and a fixed h."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for gamma in sweep["gammas"]:
        curve = run_ho_curve(
            K=K,
            lambdas=lambdas,
            h=sweep["h"],
            gamma=float(gamma),
            n_steps=sweep["n_steps"],
            n_particles=N_PARTICLES,
            n_repeats=N_REPEATS,
            x0=x0,
        )
        ax.plot(np.arange(1, len(curve) + 1), curve, label=rf"$\gamma={gamma}$")

    ax.set_title(rf"HO-LMC $(K={K})$: gamma sweep at $h={sweep['h']}$")
    ax.set_xlabel("number of iterations")
    ax.set_ylabel(r"estimated $W_2$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / f"k{K}_gamma_sweep_mc.png"
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path



def plot_h_sweep(K: int, sweep: dict, lambdas: np.ndarray, x0: np.ndarray) -> Path:
    """Plot several h values for a fixed order K and a fixed gamma."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for h, n_steps in zip(sweep["h_values"], sweep["n_steps"]):
        curve = run_ho_curve(
            K=K,
            lambdas=lambdas,
            h=float(h),
            gamma=float(sweep["gamma"]),
            n_steps=int(n_steps),
            n_particles=N_PARTICLES,
            n_repeats=N_REPEATS,
            x0=x0,
        )
        ax.plot(np.arange(1, len(curve) + 1), curve, label=rf"$h={h}$")

    ax.set_title(rf"HO-LMC $(K={K})$: h sweep at $\gamma={sweep['gamma']}$")
    ax.set_xlabel("number of iterations")
    ax.set_ylabel(r"estimated $W_2$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / f"k{K}_h_sweep_mc.png"
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return path


# =============================================================================
# Main entry point.
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lambdas = make_lambdas(DIMENSION, KAPPA)
    x0 = make_initial_position(DIMENSION, INIT_SHIFT)

    generated = []
    for K in [3, 4]:
        generated.append(plot_gamma_sweep(K, GAMMA_SWEEPS[K], lambdas, x0))
        generated.append(plot_h_sweep(K, H_SWEEPS[K], lambdas, x0))

    meta_path = OUTPUT_DIR / "order_sweeps_meta.txt"
    lines = [
        "Monte Carlo higher-order sweeps",
        "=" * 40,
        f"dimension = {DIMENSION}",
        f"kappa = {KAPPA}",
        f"initial shift = {INIT_SHIFT}",
        f"n_particles = {N_PARTICLES}",
        f"n_repeats = {N_REPEATS}",
        "",
        f"generated files: {', '.join(str(p.name) for p in generated)}",
    ]
    meta_path.write_text("\n".join(lines))

    for path in generated:
        print(f"Saved figure to {path}")
    print(f"Saved meta to {meta_path}")


if __name__ == "__main__":
    main()
