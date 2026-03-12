"""
High-order Langevin sampler (Picard-Lagrange), optimized implementation.

Key changes versus the original prototype:
- Uses the Kronecker/block structure A = A_small ⊗ I_d and D = D_small ⊗ I_d.
- Caches gradients within each Picard sweep.
- Reuses step-invariant base terms outside the Picard loop.
- Samples Gaussian noise via a pre-factorized small covariance matrix.
- Exploits the fact that g(X) is nonzero only on the second block.
- Computes alpha and Sigma_C exactly with augmented/block matrix exponentials
  instead of Gauss-Legendre quadrature.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.linalg import expm


Array = np.ndarray


class HigherOrderLangevin:
    """
    Picard-Lagrange Langevin Monte Carlo (K >= 3) as a reusable sampler.

    Notes
    -----
    Internally the implementation stores all dynamics in "small" K x K form,
    exploiting the Kronecker structure with I_d. This avoids constructing dense
    (K*d) x (K*d) matrices.

    The legacy arguments ``n_quad_alpha`` and ``n_quad_noise`` are retained in
    the constructor for backward compatibility, but are no longer used because
    alpha and Sigma_C are now computed exactly.
    """

    def __init__(
        self,
        K: int,
        d: int,
        h: float,
        gamma: float,
        grad_U_fn: Callable[[Array], Array],
        *,
        nu_star: int | None = None,
        n_quad_alpha: int = 32,
        n_quad_noise: int = 32,
        rng: np.random.Generator | None = None,
    ) -> None:
        del n_quad_alpha, n_quad_noise

        if K < 3:
            raise ValueError("Higher-order scheme requires K >= 3")
        if d <= 0:
            raise ValueError("d must be positive")
        if h <= 0:
            raise ValueError("h must be positive")

        self.K = int(K)
        self.d = int(d)
        self.h = float(h)
        self.gamma = float(gamma)
        self.grad_U_fn = grad_U_fn
        self.nu_star = (self.K - 1) if nu_star is None else int(nu_star)
        self.M = self.K - 1
        self.nodes = np.linspace(0.0, 1.0, self.M)
        self.dim = self.K * self.d
        self.rng = rng or np.random.default_rng()

        # Small K x K matrices.
        self.D_small, self.Q_small = self.build_D_Q_small(self.K, self.gamma)
        self.J_small = self.build_J_small(self.K)
        self.A_small = self.build_A_small(self.D_small, self.Q_small, self.J_small)

        # Exact precomputations.
        self.expA_small, self.alpha_grad = self.precompute_alpha_and_expA_exact(
            self.A_small, self.h, self.nodes
        )
        self.Sigma_C_small = self.precompute_noise_covariance_exact(
            self.A_small, self.D_small, self.h, self.nodes, self.expA_small
        )

        # Convenience slices that exploit c_1 = 0.
        self.M_free = self.M - 1
        self.expA_small_free = self.expA_small[1:]  # shape: (M_free, K, K)
        self.alpha_grad_free = self.alpha_grad[1:]  # shape: (M_free, M, K)
        self.alpha_grad_const_free = self.alpha_grad_free[:, 0, :]  # j = 0 term
        self.alpha_grad_var_free = self.alpha_grad_free[:, 1:, :]   # j = 1..M-1

        # Reduced covariance excludes the deterministic first node (c_1 = 0).
        if self.M_free > 0:
            start = self.K
            self.Sigma_C_small_free = self.Sigma_C_small[start:, start:]
            self.noise_factor_small_free = self._factor_psd(self.Sigma_C_small_free)
            self.noise_small_free_dim = self.M_free * self.K
        else:
            self.Sigma_C_small_free = np.zeros((0, 0), dtype=float)
            self.noise_factor_small_free = np.zeros((0, 0), dtype=float)
            self.noise_small_free_dim = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def grad_evals_per_step(self) -> int:
        """
        Gradient evaluations performed by one call to ``step`` in the current
        implementation.

        We evaluate:
        - grad U at the fixed c_1 = 0 node once per step, and
        - grad U at each of the M-1 free nodes once per Picard sweep.
        """
        return 1 + self.M_free * self.nu_star

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the internal RNG. Use to get reproducible trajectories."""
        self.rng = np.random.default_rng(seed)

    def step(self, X: Array) -> Array:
        """
        Single PL-LMC step.

        Parameters
        ----------
        X:
            Current state, either flattened with shape (K*d,) or blocked with
            shape (K, d).

        Returns
        -------
        np.ndarray
            Next state, flattened with shape (K*d,).
        """
        X_blocks = self._as_blocks(X)

        # Joint Gaussian noise at the free collocation nodes only.
        W_free = self._sample_joint_noise_free_blocks()  # (M_free, K, d)

        # These terms do not depend on the Picard iterate and should be formed
        # once per step.
        base_free = np.einsum("aij,jd->aid", self.expA_small_free, X_blocks) + W_free

        if self.M_free == 0:
            return X_blocks.reshape(-1).copy()

        # The c_1 = 0 node stays fixed at the current state throughout the step.
        grad0 = self._eval_grad(X_blocks[0])
        const_drift_free = self.alpha_grad_const_free[:, :, None] * grad0[None, None, :]

        X_free = np.broadcast_to(X_blocks, (self.M_free, self.K, self.d)).copy()

        for _ in range(self.nu_star):
            grads_free = np.empty((self.M_free, self.d), dtype=float)
            for j in range(self.M_free):
                grads_free[j] = self._eval_grad(X_free[j, 0])

            var_drift_free = np.einsum(
                "ajr,jd->ard", self.alpha_grad_var_free, grads_free, optimize=True
            )
            X_free = base_free + self.h * (const_drift_free + var_drift_free)

        return X_free[-1].reshape(-1).copy()

    def sample(
        self,
        N_steps: int,
        burn_in: int = 0,
        X0: Array | None = None,
        *,
        seed: int | None = None,
        return_full_state: bool = False,
    ) -> Array:
        """
        Run the sampler and return either x_1 blocks or full flattened states.
        """
        if seed is not None:
            self.reset_rng(seed)

        X = np.zeros(self.dim, dtype=float) if X0 is None else self._as_blocks(X0).reshape(-1)
        keep: list[Array] = []
        for n in range(N_steps):
            X = self.step(X)
            if n >= burn_in:
                if return_full_state:
                    keep.append(X.copy())
                else:
                    keep.append(X.reshape(self.K, self.d)[0].copy())
        return np.array(keep)

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------
    def _as_blocks(self, X: Array) -> Array:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.shape == (self.K, self.d):
            return X_arr.copy()
        if X_arr.shape == (self.dim,):
            return X_arr.reshape(self.K, self.d).copy()
        raise ValueError(
            f"Expected X with shape {(self.dim,)} or {(self.K, self.d)}, got {X_arr.shape}"
        )

    def _eval_grad(self, x1: Array) -> Array:
        grad = np.asarray(self.grad_U_fn(np.asarray(x1, dtype=float)), dtype=float)
        return grad.reshape(self.d)

    def _sample_joint_noise_free_blocks(self) -> Array:
        if self.noise_small_free_dim == 0:
            return np.zeros((0, self.K, self.d), dtype=float)
        z = self.rng.standard_normal((self.noise_small_free_dim, self.d))
        w = self.noise_factor_small_free @ z
        return w.reshape(self.M_free, self.K, self.d)

    @staticmethod
    def _factor_psd(S: Array) -> Array:
        if S.size == 0:
            return np.zeros_like(S)
        evals, evecs = np.linalg.eigh(0.5 * (S + S.T))
        evals = np.clip(evals, 0.0, None)
        return evecs * np.sqrt(evals)[None, :]

    # ------------------------------------------------------------------
    # Original full-matrix helpers kept for familiarity / compatibility.
    # ------------------------------------------------------------------
    @staticmethod
    def build_D_Q(K: int, d: int, gamma: float) -> tuple[Array, Array]:
        D_small, Q_small = HigherOrderLangevin.build_D_Q_small(K, gamma)
        eye_d = np.eye(d, dtype=float)
        return np.kron(D_small, eye_d), np.kron(Q_small, eye_d)

    @staticmethod
    def build_J(K: int, d: int) -> Array:
        return np.kron(HigherOrderLangevin.build_J_small(K), np.eye(d, dtype=float))

    @classmethod
    def build_A(cls, D: Array, Q: Array, K: int, d: int) -> Array:
        return -(D + Q) @ cls.build_J(K, d)

    # ------------------------------------------------------------------
    # Small-matrix structure.
    # ------------------------------------------------------------------
    @staticmethod
    def build_D_Q_small(K: int, gamma: float) -> tuple[Array, Array]:
        T = np.zeros((K, K), dtype=float)
        if K >= 2:
            T[0, 1] = -1.0
            T[1, 0] = 1.0
        for i in range(1, K - 1):
            T[i, i + 1] = -gamma
            T[i + 1, i] = gamma

        Q_small = T
        diag_entries = np.concatenate([np.zeros(K - 1), np.array([gamma], dtype=float)])
        D_small = np.diag(diag_entries)
        return D_small, Q_small

    @staticmethod
    def build_J_small(K: int) -> Array:
        diag_entries = np.concatenate([np.array([0.0]), np.ones(K - 1)])
        return np.diag(diag_entries)

    @staticmethod
    def build_A_small(D_small: Array, Q_small: Array, J_small: Array) -> Array:
        return -(D_small + Q_small) @ J_small

    @staticmethod
    def lagrange_basis_values(j: int, s: Array, nodes: Array) -> Array:
        cj = nodes[j]
        vals = np.ones_like(s, dtype=float)
        for m, cm in enumerate(nodes):
            if m == j:
                continue
            vals *= (s - cm) / (cj - cm)
        return vals

    @staticmethod
    def lagrange_basis_coefficients(nodes: Array) -> Array:
        """
        Return coefficients of the Lagrange basis polynomials on ``nodes``.

        Output shape is (M, M), where row j contains coefficients of ell_j(s)
        in increasing monomial order:
            ell_j(s) = sum_{m=0}^{M-1} coeffs[j, m] * s**m.
        """
        M = len(nodes)
        V = np.vander(nodes, N=M, increasing=True)
        invV = np.linalg.inv(V)
        return invV.T

    # ------------------------------------------------------------------
    # Exact precomputations.
    # ------------------------------------------------------------------
    @classmethod
    def precompute_alpha_and_expA_exact(
        cls,
        A_small: Array,
        h: float,
        nodes: Array,
    ) -> tuple[Array, Array]:
        """
        Exact precomputation of exp(c_k h A) and the drift coefficients needed
        in the update, exploiting that g(X) only populates the second block.

        Returns
        -------
        expA_small:
            Array of shape (M, K, K) with exp(c_k h A_small).
        alpha_grad:
            Array of shape (M, M, K) where
                alpha_grad[k, j, :] = -alpha_j(c_k, h) @ e_2,
            i.e. the exact coefficient vector multiplying grad U at node j.
        """
        K = A_small.shape[0]
        M = len(nodes)
        coeffs = cls.lagrange_basis_coefficients(nodes)  # (M, M)
        factorials = np.array([math.factorial(m) for m in range(M)], dtype=float)
        e2 = np.zeros(K, dtype=float)
        e2[1] = 1.0

        expA_small = np.zeros((M, K, K), dtype=float)
        alpha_grad = np.zeros((M, M, K), dtype=float)

        # Augmented matrix for phi_1, ..., phi_M applied to e2.
        p = M
        for k, tau in enumerate(nodes):
            if tau == 0.0:
                expA_small[k] = np.eye(K, dtype=float)
                continue

            Z = tau * h * A_small
            aug = np.zeros((K + p, K + p), dtype=float)
            aug[:K, :K] = Z
            aug[:K, K] = e2
            for r in range(p - 1):
                aug[K + r, K + r + 1] = 1.0

            exp_aug = expm(aug)
            expA_small[k] = exp_aug[:K, :K]
            phi_e2 = exp_aug[:K, K : K + p]  # columns: phi_1(Z)e2, ..., phi_M(Z)e2

            scales = (tau ** np.arange(1, M + 1)) * factorials
            # beta_m(tau) = tau^(m+1) * m! * phi_{m+1}(tau h A) e2
            # alpha_j(tau, h) @ e2 = sum_m coeffs[j,m] * beta_m(tau)
            scaled_coeffs = coeffs * scales[None, :]
            alpha_cols = scaled_coeffs @ phi_e2.T  # (M, K)
            alpha_grad[k] = -alpha_cols

        return expA_small, alpha_grad

    @staticmethod
    def ou_covariance_exact(A_small: Array, D_small: Array, t: float) -> Array:
        """
        Exact covariance
            S(t) = 2 * integral_0^t exp(u A) D exp(u A^T) du
        via a Van Loan block exponential.
        """
        K = A_small.shape[0]
        if t == 0.0:
            return np.zeros((K, K), dtype=float)

        van_loan = np.zeros((2 * K, 2 * K), dtype=float)
        van_loan[:K, :K] = A_small
        van_loan[:K, K:] = 2.0 * D_small
        van_loan[K:, K:] = -A_small.T

        exp_vl = expm(t * van_loan)
        E = exp_vl[:K, :K]
        Y = exp_vl[:K, K:]
        S = Y @ E.T
        return 0.5 * (S + S.T)

    @classmethod
    def precompute_noise_covariance_exact(
        cls,
        A_small: Array,
        D_small: Array,
        h: float,
        nodes: Array,
        expA_small: Array | None = None,
    ) -> Array:
        """
        Exact joint covariance of [W(c_1), ..., W(c_M)] in small form.

        Because A = A_small ⊗ I_d and D = D_small ⊗ I_d, the full
        covariance is Sigma_C_small ⊗ I_d.
        """
        M = len(nodes)
        K = A_small.shape[0]

        if expA_small is None:
            expA_small = np.stack([expm((tau * h) * A_small) for tau in nodes], axis=0)

        # S_k = Cov[W(c_k)] in small K x K form.
        S_nodes = np.stack(
            [cls.ou_covariance_exact(A_small, D_small, tau * h) for tau in nodes],
            axis=0,
        )

        Sigma = np.zeros((M * K, M * K), dtype=float)
        for a in range(M):
            for b in range(a + 1):
                if a == b:
                    block = S_nodes[b]
                else:
                    diff = a - b
                    # Since nodes are equispaced on [0, 1], c_a - c_b is again a grid node.
                    E_diff = expA_small[diff]
                    block = E_diff @ S_nodes[b]
                ia = a * K
                ib = b * K
                Sigma[ia : ia + K, ib : ib + K] = block
                if a != b:
                    Sigma[ib : ib + K, ia : ia + K] = block.T

        return 0.5 * (Sigma + Sigma.T)


__all__ = ["HigherOrderLangevin"]
