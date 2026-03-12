"""
Clean Monte Carlo Gaussian experiment for the paper plots.

This script compares OD-LMC, UD-LMC, HO-LMC(K=3), and HO-LMC(K=4) on a
diagonal Gaussian target.  The evaluation is sample-based: for each repeat we
simulate particles, draw a Monte Carlo reference sample from the target, and
estimate W2 by optimal matching between the two point clouds.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from baseline import OverdampedLMC, UnderdampedLangevinExp
from higher_order_langevin_fast import HigherOrderLangevin

# Main experiment settings.
DIMENSION = 10
KAPPA = 3.0
INIT_SHIFT = 5
N_PARTICLES = 64
N_REPEATS = 3
PANELS = [
    {
        "name": "h0.5",
        "title": r"$h = 0.5$",
        "h": 0.5,
        "n_steps": 40,
        "gammas": {"OD-LMC": 0.12, "UD-LMC": 7.0, "HO-LMC (K=3)": 5.5, "HO-LMC (K=4)": 3},
    },
    {
        "name": "h0.05",
        "title": r"$h = 0.05$",
        "h": 0.05,
        "n_steps": 200,
        "gammas": {"OD-LMC": 0.15, "UD-LMC": 5, "HO-LMC (K=3)": 4, "HO-LMC (K=4)": 3},
    },
]
OUTPUT_DIR = Path("./gaussian_mc_clean")
OUTPUT_NAME = f"gaussian_mc_compare_d{DIMENSION}_kappa{KAPPA}_h0.5-0.05"


def make_lambdas(d: int, kappa: float) -> np.ndarray:
    """Diagonal Hessian entries with condition number `kappa`."""
    if d == 1:
        return np.array([float(kappa)], dtype=float)
    return np.geomspace(1.0, float(kappa), num=d)


def make_initial_position(d: int, shift: float) -> np.ndarray:
    """Deterministic initial position used for all particles."""
    x0 = np.zeros(d, dtype=float)
    x0[0] = float(shift)
    return x0


def make_quadratic_grad(hessian: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Gradient of U(x)=0.5 x^T H x, vectorized over batches."""
    H = np.asarray(hessian, dtype=float)
    def grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return H @ x if x.ndim == 1 else x @ H.T
    return grad


def sample_target_gaussian(n: int, lambdas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reference Monte Carlo cloud from the target Gaussian."""
    cov = np.diag(1.0 / np.asarray(lambdas, dtype=float))
    return rng.multivariate_normal(np.zeros(len(lambdas)), cov, size=n)


def empirical_w2(x: np.ndarray, y: np.ndarray) -> float:
    """Exact W2 between two equally weighted empirical measures."""
    cost = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return math.sqrt(float(cost[row_ind, col_ind].mean()))


def higher_order_batch_step(sampler: HigherOrderLangevin, states: np.ndarray, grad_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Batch version of `sampler.step`, used only to speed up the Monte Carlo run."""
    batch_size = states.shape[0]
    K, d = sampler.K, sampler.d
    x_blocks = states.reshape(batch_size, K, d)
    if sampler.M_free == 0:
        return states.copy()

    z = sampler.rng.standard_normal((batch_size, sampler.noise_small_free_dim, d))
    w = np.einsum("ab,cbd->cad", sampler.noise_factor_small_free, z, optimize=True)
    w_free = w.reshape(batch_size, sampler.M_free, K, d)
    base = np.einsum("aij,bjd->baid", sampler.expA_small_free, x_blocks, optimize=True) + w_free

    grad0 = grad_fn(x_blocks[:, 0, :])
    const_drift = sampler.alpha_grad_const_free[None, :, :, None] * grad0[:, None, None, :]
    x_free = np.broadcast_to(x_blocks[:, None, :, :], (batch_size, sampler.M_free, K, d)).copy()
    for _ in range(sampler.nu_star):
        grads = grad_fn(x_free[:, :, 0, :].reshape(batch_size * sampler.M_free, d)).reshape(batch_size, sampler.M_free, d)
        var_drift = np.einsum("ajr,bjd->bard", sampler.alpha_grad_var_free, grads, optimize=True)
        x_free = base + sampler.h * (const_drift + var_drift)
    return x_free[:, -1, :, :].reshape(batch_size, K * d)


def run_one_repeat(lambdas: np.ndarray, h: float, n_steps: int, gammas: dict[str, float], n_particles: int, x0: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """One Monte Carlo repeat for all four methods."""
    d = len(lambdas)
    grad = make_quadratic_grad(np.diag(lambdas))
    ref = sample_target_gaussian(n_particles, lambdas, np.random.default_rng(seed + 1000))

    od = OverdampedLMC(d=d, h=h, grad_U_fn=grad, gamma=gammas["OD-LMC"], rng=np.random.default_rng(seed + 1))
    ud = UnderdampedLangevinExp(d=d, h=h, grad_U_fn=grad, gamma=gammas["UD-LMC"], rng=np.random.default_rng(seed + 2))
    ho3 = HigherOrderLangevin(K=3, d=d, h=h, gamma=gammas["HO-LMC (K=3)"], grad_U_fn=grad, nu_star=2, rng=np.random.default_rng(seed + 3))
    ho4 = HigherOrderLangevin(K=4, d=d, h=h, gamma=gammas["HO-LMC (K=4)"], grad_U_fn=grad, nu_star=3, rng=np.random.default_rng(seed + 4))

    x_od = np.repeat(x0[None, :], n_particles, axis=0)
    x_ud = x_od.copy(); v_ud = np.zeros_like(x_ud)
    s3 = np.zeros((n_particles, 3 * d), dtype=float); s3[:, :d] = x0
    s4 = np.zeros((n_particles, 4 * d), dtype=float); s4[:, :d] = x0

    curves = {name: np.empty(n_steps, dtype=float) for name in ["OD-LMC", "UD-LMC", "HO-LMC (K=3)", "HO-LMC (K=4)"]}
    for t in range(n_steps):
        x_od = od.step(x_od)
        x_ud, v_ud = ud.step(x_ud, v_ud)
        s3 = higher_order_batch_step(ho3, s3, grad)
        s4 = higher_order_batch_step(ho4, s4, grad)
        curves["OD-LMC"][t] = empirical_w2(x_od, ref)
        curves["UD-LMC"][t] = empirical_w2(x_ud, ref)
        curves["HO-LMC (K=3)"][t] = empirical_w2(s3.reshape(n_particles, 3, d)[:, 0, :], ref)
        curves["HO-LMC (K=4)"][t] = empirical_w2(s4.reshape(n_particles, 4, d)[:, 0, :], ref)
    return curves


def average_curves(lambdas: np.ndarray, panel: dict, n_particles: int, n_repeats: int, x0: np.ndarray) -> dict[str, np.ndarray]:
    """Average W2 curves over several Monte Carlo repeats."""
    store = {"OD-LMC": [], "UD-LMC": [], "HO-LMC (K=3)": [], "HO-LMC (K=4)": []}
    for rep in range(n_repeats):
        curves = run_one_repeat(lambdas, panel["h"], panel["n_steps"], panel["gammas"], n_particles, x0, seed=10_000 * rep)
        for method, curve in curves.items():
            store[method].append(curve)
    return {method: np.mean(curves, axis=0) for method, curves in store.items()}


def save_csv(csv_path: Path, panel_results: list[tuple[dict, dict[str, np.ndarray]]]) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "h", "step", "method", "w2"])
        for panel, curves in panel_results:
            for method, curve in curves.items():
                for step, value in enumerate(curve, start=1):
                    writer.writerow([panel["name"], panel["h"], step, method, float(value)])


def save_meta(meta_path: Path, lambdas: np.ndarray, x0: np.ndarray, panel_results: list[tuple[dict, dict[str, np.ndarray]]]) -> None:
    lines = [
        "Monte Carlo Gaussian comparison",
        "=" * 36,
        f"dimension = {len(lambdas)}",
        f"kappa = {float(lambdas[-1] / lambdas[0]):.6g}",
        f"lambdas = {np.array2string(lambdas, precision=6)}",
        f"initial position = {np.array2string(x0, precision=6)}",
        f"n_particles = {N_PARTICLES}",
        f"n_repeats = {N_REPEATS}",
        "",
    ]
    for panel, curves in panel_results:
        lines.append(f"panel: {panel['title']}")
        lines.append(f"  h = {panel['h']}")
        lines.append(f"  n_steps = {panel['n_steps']}")
        for method, gamma in panel["gammas"].items():
            lines.append(f"  gamma[{method}] = {gamma}")
        for method, curve in curves.items():
            lines.append(f"  final W2 [{method}] = {float(curve[-1]):.10g}")
        lines.append("")
    meta_path.write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lambdas = make_lambdas(DIMENSION, KAPPA)
    x0 = make_initial_position(DIMENSION, INIT_SHIFT)

    panel_results = []
    for panel in PANELS:
        curves = average_curves(lambdas, panel, N_PARTICLES, N_REPEATS, x0)
        panel_results.append((panel, curves))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.0 * len(PANELS), 4.5), squeeze=False)
    axes = axes[0]
    for ax, (panel, curves) in zip(axes, panel_results):
        for method, curve in curves.items():
            ax.plot(np.arange(1, len(curve) + 1), curve, label=method)
        ax.set_title(panel["title"])
        ax.set_xlabel("number of iterations")
        ax.set_ylabel(r"estimated $W_2$")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(rf"Gaussian target, $d={DIMENSION}$, $\kappa={KAPPA}$", y=1.02)
    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{OUTPUT_NAME}.png"
    csv_path = OUTPUT_DIR / f"{OUTPUT_NAME}.csv"
    meta_path = OUTPUT_DIR / f"{OUTPUT_NAME}_meta.txt"
    fig.savefig(png_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    save_csv(csv_path, panel_results)
    save_meta(meta_path, lambdas, x0, panel_results)
    print(f"Saved figure to {png_path}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved meta to {meta_path}")


if __name__ == "__main__":
    main()
