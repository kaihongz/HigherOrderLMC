"""
Clean simulated Bayesian logistic regression experiment.


Main design choices
-------------------
1. The comparison uses a *common step size h* across OD-LMC, UD-LMC,
   HO-LMC(K=3), and HO-LMC(K=4).
2. The error metric is the relative L2 error between the particle mean and the
   ground-truth coefficient vector beta_star.
3. The variability band is generated from *repeated synthetic data sets*.
   This is wider and more informative than a band built only from sampler noise
   on one fixed synthetic data set.

The script saves
----------------
- one multi-panel PNG,
- one CSV file with the plotted curves,
- one text file that records the settings used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime


from blr_utils import (
    MethodConfig,
    aggregate_curve_dicts,
    assert_common_step_size,
    average_curve_dicts,
    make_blr_gradient,
    make_simulated_blr_problem,
    method_labels,
    relative_l2_metric,
    run_method_once,
)


# =============================================================================
# USER CONFIG: edit this block when tuning locally.
# =============================================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path("./blr_clean/simulated_outputs") / timestamp

# Dimensions shown in the final figure.
DIMENSIONS = [50]

# Synthetic data generation.
N_TRAIN_MULTIPLIER = 10       # n_train = N_TRAIN_MULTIPLIER * d
KAPPA = 2.0                  # condition number of the Gaussian design
BETA_SCALE = 1.5              # norm of the true beta_star
ALPHA = 0.05                  # ridge strength in the Bayesian posterior

# Monte Carlo budget.
N_STEPS = 60                # number of step in Monte Carlo
N_PARTICLES = 64            # number of particles each round
N_DATASET_REPEATS = 10         # outer repeats: new synthetic data set each time
N_SAMPLER_REPEATS = 2         # inner repeats: new sampler noise on the same data set

# confidence band.
BAND_TYPE = "stderr"            # choose from: "std", "stderr"
SHOW_BAND = True
PLOT_MODE = "all"            # choose from: "all", "ho-only"

# Seeds.
DATASET_SEED_BASE = 32
SAMPLER_SEED_BASE = 42

# Methods.
# Keep the same h for every method in one experiment.
DEFAULT_METHODS = [
    MethodConfig(label="Overdamped-LMC", kind="od", h=0.01, gamma=0.05),
    MethodConfig(label="Underdamped-LMC", kind="ud", h=0.01, gamma=14.0),
    MethodConfig(label="HigherOrder-LMC (K=3)", kind="ho", h=0.01, gamma=8.0, K=3, nu_star=2),
    MethodConfig(label="HigherOrder-LMC (K=4)", kind="ho", h=0.01, gamma=20.0, K=4, nu_star=3),
]

# Optional dimension-specific overrides.  If one dimension is not listed here,
# the script falls back to DEFAULT_METHODS.
METHODS_BY_DIM: dict[int, list[MethodConfig]] = {
    10: DEFAULT_METHODS,
    50: DEFAULT_METHODS,
}

# =============================================================================
# End of USER CONFIG.
# =============================================================================


def run_one_dimension(d: int, methods: list[MethodConfig]) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Run the full simulated experiment for one dimension.

    Output structure
    ----------------
    result[method_label][metric_name] = (mean_curve, band_curve)
    """
    dim_start_time = time.time()
    print(f"\n[START] d={d}, n_train={N_TRAIN_MULTIPLIER * d}, "
        f"dataset_repeats={N_DATASET_REPEATS}, sampler_repeats={N_SAMPLER_REPEATS}, "
        f"steps={N_STEPS}, particles={N_PARTICLES}")
    n_train = N_TRAIN_MULTIPLIER * d
    method_results: dict[str, list[dict[str, np.ndarray]]] = {m.label: [] for m in methods}

    for data_rep in range(N_DATASET_REPEATS):
        data_start_time = time.time()
        print(f"[DATASET {data_rep + 1}/{N_DATASET_REPEATS}] "
            f"d={d}, n_train={n_train} ... generating synthetic data")
        problem_seed = DATASET_SEED_BASE + 10000 * d + data_rep
        X, y01, beta_star = make_simulated_blr_problem(
            seed=problem_seed,
            d=d,
            n_train=n_train,
            kappa=KAPPA,
            beta_scale=BETA_SCALE,
        )
        grad_fn = make_blr_gradient(X, y01, alpha=ALPHA, penalize_last_coordinate=True)

        for method_idx, method in enumerate(methods):
            method_start_time = time.time()
            print(f"  -> method={method.label} on dataset {data_rep + 1}/{N_DATASET_REPEATS}")
            inner_curves: list[dict[str, np.ndarray]] = []
            for sampler_rep in range(N_SAMPLER_REPEATS):
                seed = SAMPLER_SEED_BASE + 100000 * d + 1000 * data_rep + 17 * method_idx + sampler_rep
                curve = run_method_once(
                    method=method,
                    d=d,
                    grad_fn=grad_fn,
                    n_steps=N_STEPS,
                    n_particles=N_PARTICLES,
                    metric_fn=lambda beta, beta_star=beta_star: relative_l2_metric(beta, beta_star),
                    seed=seed,
                )
                inner_curves.append(curve)

            # Average sampler randomness inside one synthetic problem.
            # The outer band will then reflect problem-to-problem variability.
            method_results[method.label].append(average_curve_dicts(inner_curves))
            method_elapsed = time.time() - method_start_time
            print(f"  <- method={method.label} finished in {method_elapsed:.1f}s")
            data_elapsed = time.time() - data_start_time
            total_elapsed = time.time() - dim_start_time
            print(f"[DATASET {data_rep + 1}/{N_DATASET_REPEATS}] done in {data_elapsed:.1f}s "
                f"(elapsed total for d={d}: {total_elapsed:.1f}s)")
    return {
        label: aggregate_curve_dicts(curves, band_type=BAND_TYPE)
        for label, curves in method_results.items()
    }


def plot_results(
    results_by_dim: dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]],
    methods: list[MethodConfig],
    out_path: Path,
) -> None:
    """Plot one panel per dimension."""
    labels_to_show = method_labels(methods, PLOT_MODE)
    steps = np.arange(1, N_STEPS + 1)

    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(5.2 * len(DIMENSIONS), 4.2), squeeze=False)
    axes = axes[0]

    for ax, d in zip(axes, DIMENSIONS):
        panel = results_by_dim[d]
        for method in methods:
            if method.label not in labels_to_show or method.label not in panel:
                continue
            mean, band = panel[method.label]["rel_l2"]
            ax.plot(steps, mean, label=method.label)
            if SHOW_BAND:
                ax.fill_between(steps, mean - band, mean + band, alpha=0.18)
        ax.set_xlabel("steps")
        ax.set_ylabel("relative $L_2$ error")
        ax.set_title(f"Simulated BLR (d = {d})")

    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_csv(
    results_by_dim: dict[int, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]],
    methods: list[MethodConfig],
    out_path: Path,
) -> None:
    """Save one long-form CSV.

    Long-form is easier to filter and re-plot locally than a very wide CSV.
    """
    labels_to_show = method_labels(methods, PLOT_MODE)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "step", "method", "metric", "mean", "band"])
        for d in DIMENSIONS:
            panel = results_by_dim[d]
            for method in methods:
                if method.label not in labels_to_show or method.label not in panel:
                    continue
                mean, band = panel[method.label]["rel_l2"]
                for t in range(N_STEPS):
                    writer.writerow([d, t + 1, method.label, "rel_l2", float(mean[t]), float(band[t])])


def save_meta(methods_by_dim: dict[int, list[MethodConfig]], out_path: Path) -> None:
    """Save a plain-text summary of the experiment settings."""
    lines = [
        "Clean simulated BLR experiment",
        f"dimensions = {DIMENSIONS}",
        f"n_train = {N_TRAIN_MULTIPLIER} * d",
        f"kappa = {KAPPA}",
        f"beta_scale = {BETA_SCALE}",
        f"alpha = {ALPHA}",
        f"n_steps = {N_STEPS}",
        f"n_particles = {N_PARTICLES}",
        f"n_dataset_repeats = {N_DATASET_REPEATS}",
        f"n_sampler_repeats = {N_SAMPLER_REPEATS}",
        f"band_type = {BAND_TYPE}",
        f"show_band = {SHOW_BAND}",
        f"plot_mode = {PLOT_MODE}",
        "",
        "Band construction:",
        "- First average over sampler repeats inside one synthetic problem.",
        "- Then compute the band across independently generated synthetic problems.",
        "",
        "Method settings by dimension:",
    ]
    for d in DIMENSIONS:
        lines.append(f"d = {d}")
        for method in methods_by_dim[d]:
            lines.append(
                f"  {method.label}: kind={method.kind}, h={method.h}, gamma={method.gamma}, K={method.K}, nu_star={method.nu_star}"
            )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    methods_by_dim: dict[int, list[MethodConfig]] = {}
    for d in DIMENSIONS:
        methods = METHODS_BY_DIM.get(d, DEFAULT_METHODS)
        assert_common_step_size(methods)
        methods_by_dim[d] = methods

    # All dimensions should show the same set of labels when plotting.  To keep
    # the figure simple, we use the labels from the first panel.
    reference_methods = methods_by_dim[DIMENSIONS[0]]

    results_by_dim = {
        d: run_one_dimension(d, methods_by_dim[d])
        for d in DIMENSIONS
    }

    figure_name = "simulated_blr_clean"
    if PLOT_MODE == "ho-only":
        figure_name += "_ho_only"

    plot_results(results_by_dim, reference_methods, OUT_DIR / f"{figure_name}.png")
    # save_csv(results_by_dim, reference_methods, OUT_DIR / f"{figure_name}.csv")
    save_meta(methods_by_dim, OUT_DIR / f"{figure_name}_meta.txt")


if __name__ == "__main__":
    main()
