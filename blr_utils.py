"""
Shared utilities for Bayesian logistic regression experiments.

The goal of this file is to keep the actual experiment scripts short and easy to
read.  Everything here is reusable numerical plumbing:

- synthetic-data generation for simulated BLR,
- UCI data loading for real-data BLR,
- BLR gradients,
- vectorized one-step simulation for OD / UD / higher-order samplers,
- metric computation,
- aggregation helpers for variability bands.

Nothing in this file is tied to one specific paper figure.  The two clean
experiment scripts simply import these helpers and define small configuration
blocks at the top.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import urlretrieve
import sys

import numpy as np
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Make the parent directory importable so this clean folder can reuse the
# sampler implementations stored in /mnt/data.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baseline import OverdampedLMC, UnderdampedLangevinExp
from higher_order_langevin_fast import HigherOrderLangevin

Array = np.ndarray


# -----------------------------------------------------------------------------
# Small configuration objects.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodConfig:
    """Configuration of one sampler.

    Parameters
    ----------
    label:
        Name shown in plots and CSV files.
    kind:
        One of {"od", "ud", "ho"}.
    h:
        Step size used by this method.
    gamma:
        Damping / friction parameter.
    K:
        Order of the higher-order method.  Only used when ``kind == 'ho'``.
    nu_star:
        Number of Picard iterations.  If ``None`` and ``kind == 'ho'``, the
        higher-order sampler default is used.
    """

    label: str
    kind: str
    h: float
    gamma: float
    K: int | None = None
    nu_star: int | None = None


# -----------------------------------------------------------------------------
# Basic validation.
# -----------------------------------------------------------------------------
def assert_common_step_size(methods: list[MethodConfig], tol: float = 1e-14) -> float:
    """Check that all methods use the same step size.

    The BLR figures requested later in the conversation should compare methods
    at a common step size, so we enforce that here.
    """
    if not methods:
        raise ValueError("Method list must not be empty.")

    h0 = float(methods[0].h)
    for method in methods[1:]:
        if abs(float(method.h) - h0) > tol:
            raise ValueError("All methods in this experiment must use the same step size h.")
    return h0


def method_labels(methods: list[MethodConfig], plot_mode: str) -> list[str]:
    """Return the method labels that should appear in the plot.

    ``plot_mode == 'all'`` keeps every method.
    ``plot_mode == 'ho-only'`` keeps only higher-order methods.
    """
    if plot_mode not in {"all", "ho-only"}:
        raise ValueError("plot_mode must be 'all' or 'ho-only'.")
    if plot_mode == "all":
        return [m.label for m in methods]
    return [m.label for m in methods if m.kind == "ho"]


# -----------------------------------------------------------------------------
# Simulated BLR data.
# -----------------------------------------------------------------------------
def make_simulated_blr_problem(
    *,
    seed: int,
    d: int,
    n_train: int,
    kappa: float,
    beta_scale: float,
) -> tuple[Array, Array, Array]:
    """Generate one synthetic BLR problem.

    The feature covariance is anisotropic with condition number ``kappa``.  The
    true coefficient vector is dense and has Euclidean norm ``beta_scale``.

    Returns
    -------
    X : ndarray, shape (n_train, d)
        Training features.
    y01 : ndarray, shape (n_train,)
        Bernoulli labels in {0, 1}.
    beta_star : ndarray, shape (d,)
        Ground-truth coefficient vector.
    """
    if d <= 0 or n_train <= 0:
        raise ValueError("d and n_train must be positive")
    if kappa < 1.0:
        raise ValueError("kappa must be at least 1")

    rng = np.random.default_rng(seed)

    # Random orthogonal basis plus geometric spectrum gives a clean anisotropic
    # design matrix with controlled condition number.
    G = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(G)
    eigvals = np.geomspace(1.0, kappa, d)
    Sigma = Q @ np.diag(eigvals) @ Q.T

    X = rng.multivariate_normal(np.zeros(d), Sigma, size=n_train)

    beta_raw = rng.normal(size=d)
    beta_star = beta_scale * beta_raw / max(np.linalg.norm(beta_raw), 1e-12)

    probs = expit(X @ beta_star)
    y01 = rng.binomial(1, probs, size=n_train)

    return X.astype(float), y01.astype(int), beta_star.astype(float)


# -----------------------------------------------------------------------------
# Real-data loaders.
# -----------------------------------------------------------------------------
IONOSPHERE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"


def _download_if_missing(path: Path, url: str) -> None:
    """Download one file only when it is not already cached locally."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    urlretrieve(url, path)


def load_binary_uci_dataset(name: str, *, cache_dir: str | Path | None = None) -> tuple[Array, Array]:
    """Load one binary classification data set.

    Supported names
    ---------------
    - ``'breast_cancer'``: Breast Cancer Wisconsin (Diagnostic), via scikit-learn.
    - ``'ionosphere'``: Ionosphere, downloaded from the UCI repository and
      cached locally.

    Returns
    -------
    X : ndarray, shape (n_samples, d)
        Raw features, before standardization.
    y : ndarray, shape (n_samples,)
        Labels in {0, 1}.
    """
    dataset = name.lower()

    if dataset == "breast_cancer":
        data = load_breast_cancer()
        X = data.data.astype(float)
        y = data.target.astype(int)
        return X, y

    if dataset == "ionosphere":
        cache_root = Path(cache_dir) if cache_dir is not None else Path.cwd()
        cache_path = cache_root / "ionosphere.data"
        _download_if_missing(cache_path, IONOSPHERE_URL)

        raw = np.genfromtxt(cache_path, delimiter=",", dtype=str)
        X = raw[:, :-1].astype(float)
        y = (raw[:, -1] == "g").astype(int)
        return X, y

    raise ValueError("Unsupported dataset. Use 'breast_cancer' or 'ionosphere'.")


# -----------------------------------------------------------------------------
# Standard train/test preprocessing.
# -----------------------------------------------------------------------------
def make_standardized_split(
    X: Array,
    y: Array,
    *,
    split_seed: int,
    test_size: float,
    add_intercept: bool,
) -> tuple[Array, Array, Array, Array]:
    """Create one stratified train/test split.

    Standardization uses only the training split.  This avoids test-set leakage.
    When ``add_intercept`` is True, an all-ones column is appended after
    standardization.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        test_size=test_size,
        stratify=y,
        random_state=split_seed,
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0.0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    if add_intercept:
        X_train = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
        X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)

    return X_train.astype(float), y_train.astype(int), X_test.astype(float), y_test.astype(int)


# -----------------------------------------------------------------------------
# BLR gradients.
# -----------------------------------------------------------------------------
def make_blr_gradient(
    X: Array,
    y01: Array,
    *,
    alpha: float,
    penalize_last_coordinate: bool,
) -> Callable[[Array], Array]:
    """Return the gradient of the BLR negative log posterior.

    The potential is

        U(beta) = alpha / 2 * ||beta||^2 + sum_i log(1 + exp(x_i^T beta)) - y_i x_i^T beta,

    except that the last coordinate can be left unpenalized when it represents
    an intercept.

    The returned function accepts shapes ``(d,)``, ``(N, d)``, and ``(N, M, d)``.
    """
    X = np.asarray(X, dtype=float)
    y01 = np.asarray(y01, dtype=float)
    d = X.shape[1]

    reg_mask = np.ones(d, dtype=float)
    if not penalize_last_coordinate:
        reg_mask[-1] = 0.0

    def grad(beta: Array) -> Array:
        beta_arr = np.asarray(beta, dtype=float)

        if beta_arr.ndim == 1:
            z = X @ beta_arr
            p = expit(z)
            return alpha * reg_mask * beta_arr + X.T @ (p - y01)

        if beta_arr.ndim == 2:
            z = beta_arr @ X.T
            p = expit(z)
            return alpha * (reg_mask[None, :] * beta_arr) + (p - y01[None, :]) @ X

        if beta_arr.ndim == 3:
            z = np.einsum("nmd,td->nmt", beta_arr, X, optimize=True)
            p = expit(z)
            return alpha * (reg_mask[None, None, :] * beta_arr) + np.einsum(
                "nmt,td->nmd", p - y01[None, None, :], X, optimize=True
            )

        raise ValueError(f"Unsupported beta shape {beta_arr.shape}")

    return grad


# -----------------------------------------------------------------------------
# Vectorized higher-order step.
# -----------------------------------------------------------------------------
def higher_order_step_batch(sampler: HigherOrderLangevin, states: Array) -> Array:
    """Advance many higher-order particles by one step.

    The sampler itself already exposes a single-state ``step`` method.  This
    helper uses the same precomputed coefficients to update a whole batch of
    particles in one call, which is much faster for experiments.
    """
    states_arr = np.asarray(states, dtype=float)
    if states_arr.ndim != 2 or states_arr.shape[1] != sampler.dim:
        raise ValueError(f"Expected states with shape (N, {sampler.dim}), got {states_arr.shape}")

    n_particles = states_arr.shape[0]
    X_blocks = states_arr.reshape(n_particles, sampler.K, sampler.d).copy()

    if sampler.nu_star == 0 or sampler.M_free == 0:
        return X_blocks.reshape(n_particles, sampler.dim).copy()

    # One joint Gaussian draw per particle for the free collocation nodes.
    z = sampler.rng.standard_normal((n_particles, sampler.noise_small_free_dim, sampler.d))
    W_free = np.einsum("ab,nbd->nad", sampler.noise_factor_small_free, z, optimize=True)
    W_free = W_free.reshape(n_particles, sampler.M_free, sampler.K, sampler.d)

    base_free = np.einsum("aij,njd->naid", sampler.expA_small_free, X_blocks, optimize=True) + W_free

    # The node c_1 = 0 is fixed at the current state during this step.
    grad0 = np.asarray(sampler.grad_U_fn(X_blocks[:, 0, :]), dtype=float)
    const_drift_free = sampler.alpha_grad_const_free[None, :, :, None] * grad0[:, None, None, :]

    X_free = np.broadcast_to(X_blocks[:, None, :, :], (n_particles, sampler.M_free, sampler.K, sampler.d)).copy()
    for _ in range(sampler.nu_star):
        grads_free = np.asarray(sampler.grad_U_fn(X_free[:, :, 0, :]), dtype=float)
        var_drift_free = np.einsum("ajr,njd->nard", sampler.alpha_grad_var_free, grads_free, optimize=True)
        X_free = base_free + sampler.h * (const_drift_free + var_drift_free)

    return X_free[:, -1].reshape(n_particles, sampler.dim).copy()


# -----------------------------------------------------------------------------
# One Monte Carlo run of one method.
# -----------------------------------------------------------------------------
def run_method_once(
    *,
    method: MethodConfig,
    d: int,
    grad_fn: Callable[[Array], Array],
    n_steps: int,
    n_particles: int,
    metric_fn: Callable[[Array], dict[str, float]],
    seed: int,
) -> dict[str, Array]:
    """Run one method once and return full metric curves.

    ``metric_fn`` receives the current particle cloud of ``beta`` values and
    returns a dictionary such as ``{"rel_l2": value}`` or
    ``{"logloss": ..., "auc": ..., "acc": ...}``.
    """
    rng = np.random.default_rng(seed)

    if method.kind == "od":
        sampler = OverdampedLMC(d=d, h=method.h, gamma=method.gamma, grad_U_fn=grad_fn, rng=rng)
        beta = np.zeros((n_particles, d), dtype=float)
        out: dict[str, Array] | None = None
        for t in range(n_steps):
            beta = sampler.step(beta)
            metrics = metric_fn(beta)
            if out is None:
                out = {name: np.empty(n_steps, dtype=float) for name in metrics}
            for name, value in metrics.items():
                out[name][t] = float(value)
        assert out is not None
        return out

    if method.kind == "ud":
        sampler = UnderdampedLangevinExp(d=d, h=method.h, gamma=method.gamma, grad_U_fn=grad_fn, rng=rng)
        beta = np.zeros((n_particles, d), dtype=float)
        velocity = np.zeros((n_particles, d), dtype=float)
        out = None
        for t in range(n_steps):
            beta, velocity = sampler.step(beta, velocity)
            metrics = metric_fn(beta)
            if out is None:
                out = {name: np.empty(n_steps, dtype=float) for name in metrics}
            for name, value in metrics.items():
                out[name][t] = float(value)
        assert out is not None
        return out

    if method.kind == "ho":
        if method.K is None:
            raise ValueError("Higher-order methods must specify K.")
        nu_star = method.nu_star if method.nu_star is not None else method.K - 1
        sampler = HigherOrderLangevin(
            K=method.K,
            d=d,
            h=method.h,
            gamma=method.gamma,
            grad_U_fn=grad_fn,
            nu_star=nu_star,
            rng=rng,
        )
        state = np.zeros((n_particles, sampler.dim), dtype=float)
        out = None
        for t in range(n_steps):
            state = higher_order_step_batch(sampler, state)
            beta = state.reshape(n_particles, method.K, d)[:, 0, :]
            metrics = metric_fn(beta)
            if out is None:
                out = {name: np.empty(n_steps, dtype=float) for name in metrics}
            for name, value in metrics.items():
                out[name][t] = float(value)
        assert out is not None
        return out

    raise ValueError("method.kind must be one of {'od', 'ud', 'ho'}")


# -----------------------------------------------------------------------------
# Metrics.
# -----------------------------------------------------------------------------
def relative_l2_metric(beta_particles: Array, beta_star: Array) -> dict[str, float]:
    """Relative L2 error between the particle mean and the true parameter."""
    beta_mean = np.asarray(beta_particles, dtype=float).mean(axis=0)
    err = np.linalg.norm(beta_mean - beta_star) / max(np.linalg.norm(beta_star), 1e-12)
    return {"rel_l2": float(err)}


def predictive_metrics(beta_particles: Array, X_eval: Array, y_eval: Array) -> dict[str, float]:
    """Posterior-predictive log-loss, AUROC, and accuracy from one particle cloud."""
    probs = expit(np.asarray(beta_particles, dtype=float) @ np.asarray(X_eval, dtype=float).T).mean(axis=0)
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return {
        "logloss": float(log_loss(y_eval, probs)),
        "auc": float(roc_auc_score(y_eval, probs)),
        "acc": float(accuracy_score(y_eval, probs >= 0.5)),
    }


# -----------------------------------------------------------------------------
# Aggregation for variability bands.
# -----------------------------------------------------------------------------
def average_curve_dicts(curves: list[dict[str, Array]]) -> dict[str, Array]:
    """Average several curve dictionaries with identical keys.

    This is used in a two-level design:

    - simulated BLR: average over sampler repeats inside one synthetic problem,
    - real-data BLR: average over sampler repeats inside one train/test split.
    """
    if not curves:
        raise ValueError("curves must not be empty")
    keys = curves[0].keys()
    return {key: np.mean(np.stack([curve[key] for curve in curves], axis=0), axis=0) for key in keys}


def aggregate_curve_dicts(
    curves: list[dict[str, Array]],
    *,
    band_type: str,
) -> dict[str, tuple[Array, Array]]:
    """Aggregate repeated curve dictionaries into mean + band.

    Parameters
    ----------
    curves:
        List of dictionaries that map metric names to stepwise curves.
    band_type:
        ``'std'`` produces a standard-deviation band.
        ``'stderr'`` produces a standard-error band.

    Notes
    -----
    The user explicitly asked for wider bands than before, so the clean BLR
    scripts use ``band_type='std'`` by default.  This is a variability band, not
    a formal confidence interval.
    """
    if not curves:
        raise ValueError("curves must not be empty")
    if band_type not in {"std", "stderr"}:
        raise ValueError("band_type must be 'std' or 'stderr'.")

    n = len(curves)
    metrics = curves[0].keys()
    out: dict[str, tuple[Array, Array]] = {}

    for metric in metrics:
        stack = np.stack([curve[metric] for curve in curves], axis=0)
        mean = stack.mean(axis=0)
        if n == 1:
            band = np.zeros_like(mean)
        else:
            band = stack.std(axis=0, ddof=1)
            if band_type == "stderr":
                band = band / np.sqrt(n)
        out[metric] = (mean, band)

    return out
