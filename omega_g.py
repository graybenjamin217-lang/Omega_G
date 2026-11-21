"""
Variable-Exponent Feedback Operator (Omega_G) and Its Odd Extension (Omega_G_odd).

Implements:
- Scalar generators f(x) = x^(1/x - 1) on [0, inf) and smooth odd extension f_odd on R
- Exact first and second scalar derivatives
- Continuous functional calculus on Hermitian/symmetric matrices (eigendecomposition)
- Bounded spectral adapters for unbounded self-adjoint operators (Remark 6.2)
- Pure-NumPy holomorphic extension for non-normal matrices with positive real spectra
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Analytical & Full-Precision float64 Constants (Sections 3-5)
# ---------------------------------------------------------------------------
X_REPELLER: float = 0.5
REPELLER_MULTIPLIER: float = 1.0 + 2.0 * math.log(2.0)  # 2.3862943611198906
X_MAX: float = 2.8874748541959434                       # unique root of f(x) = 1/2 in (1, inf)
X_CRIT: float = 0.7258278400933647                      # unique root of f'(x) = 1 in (1/2, 1)
DELTA: float = 1.0 - X_CRIT                             # 0.2741721599066353

# Strictly below the exact IEEE-754 float64 underflow cutoff (~0.006677):
# avoids subnormal traps and 0*inf warnings with zero precision loss in float64.
FLOAT64_UNDERFLOW_CLAMP: float = 0.0066
NEAR_ONE_TOL: float = 1e-5


# ---------------------------------------------------------------------------
# Scalar Generators & Derivatives
# ---------------------------------------------------------------------------
def f_scalar(x: ArrayLike, *, check_domain: bool = True) -> NDArray[np.float64] | float:
    """
    Evaluate f(x) = x^(1/x - 1) for x > 0 with continuous extension f(0) = 0.

    Uses a 4th-order Taylor expansion for |x - 1| < 1e-5, log1p stabilization
    elsewhere, and an exact float64 underflow clamp for 0 <= x <= 0.0066.
    """
    is_scalar = np.ndim(x) == 0
    arr = np.asarray(x, dtype=np.float64)

    if check_domain and np.any(arr < 0.0):
        raise ValueError("f_scalar domain is [0, inf); received negative input.")

    out = np.zeros_like(arr, dtype=np.float64)
    h = arr - 1.0

    near_one = np.abs(h) < NEAR_ONE_TOL
    active = (arr > FLOAT64_UNDERFLOW_CLAMP) & (~near_one)

    if np.any(near_one):
        hn = h[near_one]
        out[near_one] = 1.0 - hn**2 + 1.5 * hn**3 - (4.0 / 3.0) * hn**4

    if np.any(active):
        xa = arr[active]
        ha = h[active]
        # ((1 - x) / x) * ln(x) = (-h / x) * log1p(h), avoiding cancellation near 1
        out[active] = np.exp((-ha / xa) * np.log1p(ha))

    return float(out) if is_scalar else out


def f_odd_scalar(x: ArrayLike) -> NDArray[np.float64] | float:
    """Evaluate the C^inf odd reflection f_odd(x) = sign(x) * f(|x|) on R."""
    is_scalar = np.ndim(x) == 0
    arr = np.asarray(x, dtype=np.float64)
    mag = f_scalar(np.abs(arr), check_domain=False)
    out = np.sign(arr) * mag
    return float(out) if is_scalar else out


def f_prime(x: ArrayLike) -> NDArray[np.float64] | float:
    """Exact first derivative f'(x) = f(x) * (1 - x - ln(x)) / x^2 on [0, inf)."""
    is_scalar = np.ndim(x) == 0
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("f_prime domain is [0, inf).")

    out = np.zeros_like(arr, dtype=np.float64)
    h = arr - 1.0
    near_one = np.abs(h) < NEAR_ONE_TOL
    active = (arr > FLOAT64_UNDERFLOW_CLAMP) & (~near_one)

    if np.any(near_one):
        hn = h[near_one]
        out[near_one] = -2.0 * hn + 4.5 * hn**2 - (16.0 / 3.0) * hn**3

    if np.any(active):
        xa = arr[active]
        ha = h[active]
        fx = np.exp((-ha / xa) * np.log1p(ha))
        out[active] = fx * (-ha - np.log1p(ha)) / (xa * xa)

    return float(out) if is_scalar else out


def f_double_prime(x: ArrayLike) -> NDArray[np.float64] | float:
    """
    Exact second derivative on [0, inf):
        f''(x) = f(x) * [ ((1 - x - ln x) / x^2)^2 + (x - 3 + 2 ln x) / x^3 ].
    """
    is_scalar = np.ndim(x) == 0
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("f_double_prime domain is [0, inf).")

    out = np.zeros_like(arr, dtype=np.float64)
    h = arr - 1.0
    near_one = np.abs(h) < NEAR_ONE_TOL
    active = (arr > FLOAT64_UNDERFLOW_CLAMP) & (~near_one)

    if np.any(near_one):
        hn = h[near_one]
        out[near_one] = -2.0 + 9.0 * hn - 16.0 * hn**2

    if np.any(active):
        xa = arr[active]
        ha = h[active]
        lx = np.log1p(ha)
        fx = np.exp((-ha / xa) * lx)
        u = (-ha - lx) / (xa * xa)
        u_prime = (xa - 3.0 + 2.0 * lx) / (xa**3)
        out[active] = fx * (u * u + u_prime)

    return float(out) if is_scalar else out


# Derived quadratic error bounds (computed directly so they never drift from X_CRIT)
K_DELTA: float = (1.0 - float(f_scalar(X_CRIT))) / (DELTA**2)   # ~1.516611
C_DELTA: float = 0.5 * abs(float(f_double_prime(X_CRIT)))       # ~2.812794


# ---------------------------------------------------------------------------
# Self-Adjoint C*-Algebra Functional Calculus (Hermitian Matrices)
# ---------------------------------------------------------------------------
def _assert_hermitian(A: NDArray[np.complexfloating | np.floating], atol: float = 1e-12) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {A.shape}.")
    if not np.allclose(A, A.conj().T, atol=atol, rtol=0.0):
        raise ValueError("Matrix is not Hermitian/symmetric within tolerance.")


def omega_g_matrix(
    A: NDArray[np.complexfloating | np.floating],
    iterations: int = 1,
    *,
    atol: float = 1e-12,
) -> NDArray[np.complexfloating | np.floating]:
    """
    Apply Omega_G^iterations to a positive semidefinite Hermitian matrix A via
    continuous functional calculus (exact eigendecomposition).
    """
    if iterations < 0:
        raise ValueError("iterations must be >= 0.")
    A_arr = np.asarray(A)
    _assert_hermitian(A_arr, atol=atol)
    if iterations == 0:
        return A_arr.copy()

    eigvals, eigvecs = np.linalg.eigh(A_arr)
    if np.min(eigvals) < -atol:
        raise ValueError(
            f"omega_g_matrix requires positive semidefinite input; min eigenvalue = {np.min(eigvals):.3e}."
        )
    vals = np.maximum(eigvals, 0.0)
    for _ in range(iterations):
        vals = np.asarray(f_scalar(vals, check_domain=False), dtype=np.float64)

    res = (eigvecs * vals) @ eigvecs.conj().T
    return res.real if np.isrealobj(A_arr) else res


def omega_g_odd_matrix(
    A: NDArray[np.complexfloating | np.floating],
    iterations: int = 1,
    *,
    atol: float = 1e-12,
) -> NDArray[np.complexfloating | np.floating]:
    """
    Apply (Omega_G^odd)^iterations to any Hermitian/symmetric matrix A via
    continuous functional calculus.
    """
    if iterations < 0:
        raise ValueError("iterations must be >= 0.")
    A_arr = np.asarray(A)
    _assert_hermitian(A_arr, atol=atol)
    if iterations == 0:
        return A_arr.copy()

    eigvals, eigvecs = np.linalg.eigh(A_arr)
    vals = eigvals.astype(np.float64, copy=True)
    for _ in range(iterations):
        vals = np.asarray(f_odd_scalar(vals), dtype=np.float64)

    res = (eigvecs * vals) @ eigvecs.conj().T
    return res.real if np.isrealobj(A_arr) else res


# ---------------------------------------------------------------------------
# Bounded Spectral Adapters (Remark 6.2)
# ---------------------------------------------------------------------------
def positive_resolvent(D: NDArray[np.floating | np.complexfloating], tau: float) -> NDArray:
    """Compute R_tau(D) = (I + tau * D^2)^(-1) for Hermitian D and tau > 0."""
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive.")
    D_arr = np.asarray(D)
    _assert_hermitian(D_arr)
    eigvals, eigvecs = np.linalg.eigh(D_arr)
    r_vals = 1.0 / (1.0 + tau * (eigvals**2))
    res = (eigvecs * r_vals) @ eigvecs.conj().T
    return res.real if np.isrealobj(D_arr) else res


def bounded_odd_transform(D: NDArray[np.floating | np.complexfloating], tau: float) -> NDArray:
    """Compute F_tau(D) = sqrt(tau) * D * (I + tau * D^2)^(-1/2) for Hermitian D, tau > 0."""
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive.")
    D_arr = np.asarray(D)
    _assert_hermitian(D_arr)
    eigvals, eigvecs = np.linalg.eigh(D_arr)
    f_vals = (math.sqrt(tau) * eigvals) / np.sqrt(1.0 + tau * (eigvals**2))
    res = (eigvecs * f_vals) @ eigvecs.conj().T
    return res.real if np.isrealobj(D_arr) else res


# ---------------------------------------------------------------------------
# Pure-NumPy Holomorphic Extension for Non-Normal Matrices (Edge Case 3)
# ---------------------------------------------------------------------------
def expm_numpy(Z: NDArray[np.floating | np.complexfloating]) -> NDArray:
    """Matrix exponential via scaling-and-squaring with a Taylor series."""
    Z_arr = np.asarray(Z, dtype=np.complex128 if np.iscomplexobj(Z) else np.float64)
    n = Z_arr.shape[0]
    norm = float(np.linalg.norm(Z_arr, ord=1))
    s = max(0, int(math.ceil(math.log2(norm / 0.5)))) if norm > 0.5 else 0
    A = Z_arr / (2.0**s)

    res = np.eye(n, dtype=A.dtype)
    term = np.eye(n, dtype=A.dtype)
    for k in range(1, 22):
        term = (term @ A) / k
        res = res + term
        if float(np.linalg.norm(term, ord=1)) < 1e-18:
            break

    for _ in range(s):
        res = res @ res
    return res


def _sqrtm_denman_beavers(A: NDArray, max_iter: int = 60, tol: float = 1e-14) -> NDArray:
    """Principal matrix square root via Denman-Beavers iteration."""
    n = A.shape[0]
    Y = A.astype(np.complex128 if np.iscomplexobj(A) else np.float64, copy=True)
    Z = np.eye(n, dtype=Y.dtype)
    for _ in range(max_iter):
        Y_next = 0.5 * (Y + np.linalg.inv(Z))
        Z_next = 0.5 * (Z + np.linalg.inv(Y))
        if float(np.linalg.norm(Y_next - Y, ord="fro")) <= tol * max(1.0, float(np.linalg.norm(Y, ord="fro"))):
            return Y_next
        Y, Z = Y_next, Z_next
    raise RuntimeError("Denman-Beavers square root did not converge.")


def logm_pos_spectrum(A: NDArray, max_sqrts: int = 20) -> NDArray:
    """
    Principal matrix logarithm via inverse scaling-and-squaring + Mercator series.
    Works on non-normal matrices (including Jordan blocks) whose eigenvalues have Re(z) > 0.
    """
    n = A.shape[0]
    dtype = np.complex128 if np.iscomplexobj(A) else np.float64
    B = np.asarray(A, dtype=dtype).copy()
    I_n = np.eye(n, dtype=dtype)

    k = 0
    while float(np.linalg.norm(B - I_n, ord="fro")) > 0.25:
        if k >= max_sqrts:
            raise RuntimeError("Inverse scaling-and-squaring exceeded max_sqrts.")
        B = _sqrtm_denman_beavers(B)
        k += 1

    E = B - I_n
    log_B = np.zeros_like(E)
    power = E.copy()
    for j in range(1, 50):
        term = ((-1.0) ** (j - 1)) * power / j
        log_B = log_B + term
        if float(np.linalg.norm(term, ord="fro")) < 1e-18:
            break
        power = power @ E

    return (2.0**k) * log_B


def omega_g_nonnormal(A: NDArray, iterations: int = 1) -> NDArray:
    """
    Evaluate Omega_G(A) = exp((A^(-1) - I) * log(A)) on an invertible (possibly
    non-normal / defective) matrix A whose eigenvalues lie strictly in Re(z) > 0.
    """
    if iterations < 0:
        raise ValueError("iterations must be >= 0.")
    M = np.asarray(A, dtype=np.complex128 if np.iscomplexobj(A) else np.float64).copy()
    n = M.shape[0]
    I_n = np.eye(n, dtype=M.dtype)
    for _ in range(iterations):
        log_M = logm_pos_spectrum(M)
        M_inv = np.linalg.inv(M)
        M = expm_numpy((M_inv - I_n) @ log_M)
    return M
