"""
Baseline Langevin samplers in class form for reuse and testing.

This file contains:
- OverdampedLMC: Euler-Maruyama discretization of overdamped Langevin,
- UnderdampedLangevinExp: exponential-integrator discretization of underdamped
  Langevin with frozen gradient over one step.

For the underdamped class, setting ``u = 1`` matches the K = 2 dynamics in the
paper:
    dX_1 = X_2 dt,
    dX_2 = -gamma X_2 dt - grad U(X_1) dt + sqrt(2 gamma) dB_t.
The optional parameter ``u`` keeps the slightly more general kinetic scaling
already present in the original baseline code.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


class OverdampedLMC:
    """
    Euler-Maruyama discretization of overdamped Langevin.

    The continuous-time target-preserving SDE is
        dX_t = -gamma * grad U(X_t) dt + sqrt(2 gamma) dB_t.
    """

    def __init__(
        self,
        d: int,
        h: float,
        grad_U_fn,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be positive")
        if h <= 0:
            raise ValueError("h must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.d = int(d)
        self.h = float(h)
        self.gamma = float(gamma)
        self.grad_U_fn = grad_U_fn
        self.rng = rng or np.random.default_rng()
        self.noise_scale = np.sqrt(2.0 * self.gamma * self.h)

    @property
    def grad_evals_per_step(self) -> int:
        return 1

    def reset_rng(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _as_state(self, x: Array) -> Array:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape == (self.d,):
            return x_arr
        if x_arr.ndim >= 2 and x_arr.shape[-1] == self.d:
            return x_arr
        raise ValueError(f"Expected x with last dimension {self.d}, got {x_arr.shape}")

    def step(self, x: Array) -> Array:
        x_arr = self._as_state(x)
        grad = np.asarray(self.grad_U_fn(x_arr), dtype=float)
        if grad.shape != x_arr.shape:
            raise ValueError(
                f"grad_U_fn returned shape {grad.shape}, expected {x_arr.shape}"
            )
        noise = self.rng.normal(size=x_arr.shape)
        return x_arr - self.gamma * self.h * grad + self.noise_scale * noise

    def sample(
        self,
        N_steps: int,
        burn_in: int = 0,
        x0: Array | None = None,
        *,
        seed: int | None = None,
    ) -> Array:
        if N_steps < 0:
            raise ValueError("N_steps must be nonnegative")
        if burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        if seed is not None:
            self.reset_rng(seed)

        x = np.zeros(self.d, dtype=float) if x0 is None else self._as_state(x0).copy()
        n_keep = max(0, N_steps - burn_in)
        out = np.empty((n_keep,) + x.shape, dtype=float)

        t = 0
        for n in range(N_steps):
            x = self.step(x)
            if n >= burn_in:
                out[t] = x
                t += 1
        return out


class UnderdampedLangevinExp:
    """
    Exponential-integrator scheme for underdamped Langevin.

    With u = 1, the continuous-time dynamics are
        dX_t = V_t dt,
        dV_t = -gamma V_t dt - grad U(X_t) dt + sqrt(2 gamma) dB_t.

    More generally, the implementation keeps the original scaling parameter u:
        dV_t = -gamma V_t dt - u grad U(X_t) dt + sqrt(2 gamma u) dB_t.
    """

    def __init__(
        self,
        d: int,
        h: float,
        grad_U_fn,
        gamma: float,
        u: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be positive")
        if h <= 0:
            raise ValueError("h must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if u <= 0:
            raise ValueError("u must be positive")

        self.d = int(d)
        self.h = float(h)
        self.gamma = float(gamma)
        self.u = float(u)
        self.grad_U_fn = grad_U_fn
        self.rng = rng or np.random.default_rng()

        # Stable scalar coefficients.
        gh = self.gamma * self.h
        self.exp1 = np.exp(-gh)
        self.exp2 = np.exp(-2.0 * gh)
        self.c = -np.expm1(-gh) / self.gamma                # (1 - exp(-gh)) / gamma
        c2 = -np.expm1(-2.0 * gh) / self.gamma              # (1 - exp(-2gh)) / gamma

        var_v = self.u * (-np.expm1(-2.0 * gh))
        cov_xv = self.u * self.gamma * self.c * self.c
        var_x = (2.0 * self.u / self.gamma) * (self.h - 2.0 * self.c + 0.5 * c2)

        Sigma = np.array([[var_x, cov_xv], [cov_xv, var_v]], dtype=float)
        self.Sigma = 0.5 * (Sigma + Sigma.T)
        self.noise_factor = self._factor_psd(self.Sigma)

    @property
    def grad_evals_per_step(self) -> int:
        return 1

    def reset_rng(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def _as_state(self, x: Array, *, name: str) -> Array:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape == (self.d,):
            return x_arr
        if x_arr.ndim >= 2 and x_arr.shape[-1] == self.d:
            return x_arr
        raise ValueError(f"Expected {name} with last dimension {self.d}, got {x_arr.shape}")

    @staticmethod
    def _factor_psd(S: Array) -> Array:
        evals, evecs = np.linalg.eigh(0.5 * (S + S.T))
        evals = np.clip(evals, 0.0, None)
        return evecs * np.sqrt(evals)[None, :]

    def step(self, x: Array, v: Array) -> tuple[Array, Array]:
        x_arr = self._as_state(x, name="x")
        v_arr = self._as_state(v, name="v")
        if x_arr.shape != v_arr.shape:
            raise ValueError(f"x and v must have the same shape, got {x_arr.shape} and {v_arr.shape}")

        grad = np.asarray(self.grad_U_fn(x_arr), dtype=float)
        if grad.shape != x_arr.shape:
            raise ValueError(
                f"grad_U_fn returned shape {grad.shape}, expected {x_arr.shape}"
            )

        mu_v = self.exp1 * v_arr - self.c * self.u * grad
        mu_x = x_arr + self.c * v_arr - (self.u / self.gamma) * (self.h - self.c) * grad

        z = self.rng.normal(size=x_arr.shape + (2,))
        noise = z @ self.noise_factor.T
        x_new = mu_x + noise[..., 0]
        v_new = mu_v + noise[..., 1]
        return x_new, v_new

    def sample(
        self,
        N_steps: int,
        burn_in: int = 0,
        x0: Array | None = None,
        v0: Array | None = None,
        *,
        seed: int | None = None,
        return_velocity: bool = False,
    ) -> Array | tuple[Array, Array]:
        if N_steps < 0:
            raise ValueError("N_steps must be nonnegative")
        if burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        if seed is not None:
            self.reset_rng(seed)

        x = np.zeros(self.d, dtype=float) if x0 is None else self._as_state(x0, name="x0").copy()
        v = np.zeros(self.d, dtype=float) if v0 is None else self._as_state(v0, name="v0").copy()
        if x.shape != v.shape:
            raise ValueError(f"x0 and v0 must have the same shape, got {x.shape} and {v.shape}")

        n_keep = max(0, N_steps - burn_in)
        out_x = np.empty((n_keep,) + x.shape, dtype=float)
        out_v = np.empty((n_keep,) + v.shape, dtype=float) if return_velocity else None

        t = 0
        for n in range(N_steps):
            x, v = self.step(x, v)
            if n >= burn_in:
                out_x[t] = x
                if out_v is not None:
                    out_v[t] = v
                t += 1

        if out_v is not None:
            return out_x, out_v
        return out_x


__all__ = ["OverdampedLMC", "UnderdampedLangevinExp"]
