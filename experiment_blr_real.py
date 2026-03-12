"""
Real-data Bayesian logistic regression experiment.


Default data sets
-----------------
- Breast Cancer Wisconsin (Diagnostic)
- Ionosphere

The script saves
----------------
- one log-loss figure,
- one AUROC figure,
- one accuracy figure,
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


from blr_utils import (
    MethodConfig,
    aggregate_curve_dicts,
    assert_common_step_size,
    average_curve_dicts,
    load_binary_uci_dataset,
    make_blr_gradient,
    make_standardized_split,
    method_labels,
    predictive_metrics,
    run_method_once,
)


# =============================================================================
# USER CONFIG: edit this block when tuning locally.
# =============================================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = Path("./blr_clean/real_outputs")
CACHE_DIR = OUT_DIR / "dataset_cache"
OUT_DIR = Path("./blr_clean/real_outputs") / timestamp

# Data sets shown in the figure.
# DATASETS = ["breast_cancer", "ionosphere"]
DATASETS = ["breast_cancer"]

# Train/test evaluation design.
TEST_SIZE = 0.30
N_SPLITS = 10                 # outer repeats: different train/test splits
N_SAMPLER_REPEATS = 2        # inner repeats: different sampler randomness on the same split
ALPHA = 0.5                  # ridge strength; intercept is left unpenalized

# Monte Carlo budget.
N_STEPS = 30
N_PARTICLES = 64

# Variability band.
BAND_TYPE = "stderr"           # choose from: "std", "stderr"
SHOW_BAND = True
PLOT_MODE = "ho-only"           # choose from: "all", "ho-only"

# Seeds.
SPLIT_SEED_BASE = 500
SAMPLER_SEED_BASE = 1500

# Methods.
# Keep the same h for every method within one data set.
BREAST_CANCER_METHODS = [
    MethodConfig(label="OD-LMC", kind="od", h=0.5, gamma=0.05),
    MethodConfig(label="UD-LMC", kind="ud", h=0.5, gamma=20.0),
    MethodConfig(label="HO-LMC (K=3)", kind="ho", h=0.5, gamma=5.0, K=3, nu_star=2),
    MethodConfig(label="HO-LMC (K=4)", kind="ho", h=0.5, gamma=4.0, K=4, nu_star=3),
]

# These are only starting values for Ionosphere.  They are meant to be tuned
# locally in the same way as the Breast Cancer settings.
IONOSPHERE_METHODS = [
    MethodConfig(label="OD-LMC", kind="od", h=0.5, gamma=0.05),
    MethodConfig(label="UD-LMC", kind="ud", h=0.5, gamma=12.0),
    MethodConfig(label="HO-LMC (K=3)", kind="ho", h=0.5, gamma=5.0, K=3, nu_star=2),
    MethodConfig(label="HO-LMC (K=4)", kind="ho", h=0.5, gamma=4.0, K=4, nu_star=3),
]

METHODS_BY_DATASET: dict[str, list[MethodConfig]] = {
    "breast_cancer": BREAST_CANCER_METHODS,
    "ionosphere": IONOSPHERE_METHODS,
}

# =============================================================================
# End of USER CONFIG.
# =============================================================================


def run_one_dataset(dataset_name: str, methods: list[MethodConfig]) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Run the full experiment for one real data set.

    Output structure
    ----------------
    result[method_label][metric_name] = (mean_curve, band_curve)
    """
    X_full, y_full = load_binary_uci_dataset(dataset_name, cache_dir=CACHE_DIR)
    d_with_intercept = X_full.shape[1] + 1

    method_results: dict[str, list[dict[str, np.ndarray]]] = {m.label: [] for m in methods}

    for split_rep in range(N_SPLITS):
        split_seed = SPLIT_SEED_BASE + 1000 * split_rep
        X_train, y_train, X_test, y_test = make_standardized_split(
            X_full,
            y_full,
            split_seed=split_seed,
            test_size=TEST_SIZE,
            add_intercept=True,
        )
        grad_fn = make_blr_gradient(
            X_train,
            y_train,
            alpha=ALPHA,
            penalize_last_coordinate=False,
        )

        for method_idx, method in enumerate(methods):
            inner_curves: list[dict[str, np.ndarray]] = []
            for sampler_rep in range(N_SAMPLER_REPEATS):
                seed = SAMPLER_SEED_BASE + 100000 * d_with_intercept + 1000 * split_rep + 17 * method_idx + sampler_rep
                curve = run_method_once(
                    method=method,
                    d=d_with_intercept,
                    grad_fn=grad_fn,
                    n_steps=N_STEPS,
                    n_particles=N_PARTICLES,
                    metric_fn=lambda beta, X_test=X_test, y_test=y_test: predictive_metrics(beta, X_test, y_test),
                    seed=seed,
                )
                inner_curves.append(curve)

            # Average sampler randomness inside one train/test split.
            # The outer band will then reflect split-to-split variability.
            method_results[method.label].append(average_curve_dicts(inner_curves))

    return {
        label: aggregate_curve_dicts(curves, band_type=BAND_TYPE)
        for label, curves in method_results.items()
    }


def plot_metric(
    results_by_dataset: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]],
    methods: list[MethodConfig],
    *,
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Plot one panel per data set for one evaluation metric."""
    labels_to_show = method_labels(methods, PLOT_MODE)
    steps = np.arange(1, N_STEPS + 1)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.2 * len(DATASETS), 4.2), squeeze=False)
    axes = axes[0]

    for ax, dataset_name in zip(axes, DATASETS):
        panel = results_by_dataset[dataset_name]
        for method in methods:
            if method.label not in labels_to_show or method.label not in panel:
                continue
            mean, band = panel[method.label][metric]
            ax.plot(steps, mean, label=method.label)
            if SHOW_BAND:
                ax.fill_between(steps, mean - band, mean + band, alpha=0.18)
        ax.set_xlabel("steps")
        ax.set_ylabel(ylabel)
        ax.set_title(dataset_name.replace("_", " ").title())

    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_csv(
    results_by_dataset: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]],
    methods: list[MethodConfig],
    out_path: Path,
) -> None:
    """Save one long-form CSV with all metrics."""
    labels_to_show = method_labels(methods, PLOT_MODE)
    metrics = ["logloss", "auc", "acc"]

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "step", "method", "metric", "mean", "band"])
        for dataset_name in DATASETS:
            panel = results_by_dataset[dataset_name]
            for method in methods:
                if method.label not in labels_to_show or method.label not in panel:
                    continue
                for metric in metrics:
                    mean, band = panel[method.label][metric]
                    for t in range(N_STEPS):
                        writer.writerow([dataset_name, t + 1, method.label, metric, float(mean[t]), float(band[t])])


def save_meta(methods_by_dataset: dict[str, list[MethodConfig]], out_path: Path) -> None:
    """Save a plain-text summary of the experiment settings."""
    lines = [
        "Clean real-data BLR experiment",
        f"datasets = {DATASETS}",
        f"test_size = {TEST_SIZE}",
        f"alpha = {ALPHA}",
        f"n_steps = {N_STEPS}",
        f"n_particles = {N_PARTICLES}",
        f"n_splits = {N_SPLITS}",
        f"n_sampler_repeats = {N_SAMPLER_REPEATS}",
        f"band_type = {BAND_TYPE}",
        f"show_band = {SHOW_BAND}",
        f"plot_mode = {PLOT_MODE}",
        "",
        "Band construction:",
        "- First average over sampler repeats inside one train/test split.",
        "- Then compute the band across repeated stratified train/test splits.",
        "",
        "Method settings by data set:",
    ]
    for dataset_name in DATASETS:
        lines.append(dataset_name)
        for method in methods_by_dataset[dataset_name]:
            lines.append(
                f"  {method.label}: kind={method.kind}, h={method.h}, gamma={method.gamma}, K={method.K}, nu_star={method.nu_star}"
            )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    methods_by_dataset: dict[str, list[MethodConfig]] = {}
    for dataset_name in DATASETS:
        methods = METHODS_BY_DATASET[dataset_name]
        assert_common_step_size(methods)
        methods_by_dataset[dataset_name] = methods

    # For simplicity, keep the same method order in every panel.
    reference_methods = methods_by_dataset[DATASETS[0]]

    results_by_dataset = {
        dataset_name: run_one_dataset(dataset_name, methods_by_dataset[dataset_name])
        for dataset_name in DATASETS
    }

    prefix = "real_blr_clean"
    if PLOT_MODE == "ho-only":
        prefix += "_ho_only"

    plot_metric(
        results_by_dataset,
        reference_methods,
        metric="logloss",
        ylabel="test log-loss",
        out_path=OUT_DIR / f"{prefix}_logloss.png",
    )
    plot_metric(
        results_by_dataset,
        reference_methods,
        metric="auc",
        ylabel="test AUROC",
        out_path=OUT_DIR / f"{prefix}_auc.png",
    )
    plot_metric(
        results_by_dataset,
        reference_methods,
        metric="acc",
        ylabel="test accuracy",
        out_path=OUT_DIR / f"{prefix}_acc.png",
    )
    # save_csv(results_by_dataset, reference_methods, OUT_DIR / f"{prefix}.csv")
    save_meta(methods_by_dataset, OUT_DIR / f"{prefix}_meta.txt")


if __name__ == "__main__":
    main()
