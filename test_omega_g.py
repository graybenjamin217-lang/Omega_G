"""
Strict, adversarial pytest suite for Omega_G and Omega_G_odd.

Verifies:
- Every Proposition, Theorem, and Remark in the revised paper
- All five documented edge cases:
    1. Upper-cutoff foldback above x_max (~2.8875)
    2. Repeller critical slowing down & boundary / continuous-spectrum breakdown
    3. Non-normal Jordan-block annihilation in ceil(log2(m)) steps & repeller amplification
    4. IEEE-754 float64 subnormal / underflow cliff (~0.00668) & near-unity precision
    5. Bandpass outlier-rejection behavior vs. monotonic gates
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import pytest

import omega_g as og


# ---------------------------------------------------------------------------
# Independent Root-Finder & Random Matrix Helpers
# ---------------------------------------------------------------------------
def bisect_root(fn, a: float, b: float, tol: float = 1e-14, max_iter: int = 200) -> float:
    fa, fb = fn(a), fn(b)
    assert fa * fb < 0.0, f"Root not bracketed on [{a}, {b}]: f(a)={fa}, f(b)={fb}"
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = fn(mid)
        if abs(fm) < tol or 0.5 * (b - a) < tol:
            return mid
        if fa * fm < 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def random_hermitian(n: int, rng: np.random.Generator, complex_mat: bool = False) -> np.ndarray:
    M = rng.standard_normal((n, n))
    if complex_mat:
        M = M + 1j * rng.standard_normal((n, n))
    return 0.5 * (M + M.conj().T)


# ===========================================================================
# PAPER VERIFICATION: PROPOSITIONS 1.1 - 7.1
# ===========================================================================
class TestProposition1_1Smoothness:
    def test_super_exponential_bound_near_zero(self):
        xs = np.geomspace(0.007, math.exp(-1.0), 400)
        vals = og.f_scalar(xs)
        upper = math.e * np.exp(-1.0 / xs)
        assert np.all(vals > 0.0)
        assert np.all(vals <= upper + 1e-16)

    def test_odd_flatness_at_origin(self):
        assert og.f_odd_scalar(0.0) == 0.0
        h = 1e-3
        d1 = (og.f_odd_scalar(h) - og.f_odd_scalar(-h)) / (2 * h)
        d2 = (og.f_odd_scalar(h) - 2 * og.f_odd_scalar(0.0) + og.f_odd_scalar(-h)) / (h**2)
        d3 = (
            og.f_odd_scalar(2 * h)
            - 2 * og.f_odd_scalar(h)
            + 2 * og.f_odd_scalar(-h)
            - og.f_odd_scalar(-2 * h)
        ) / (2 * h**3)
        assert d1 == 0.0 and d2 == 0.0 and d3 == 0.0

    def test_negative_domain_rejected_by_f_scalar(self):
        with pytest.raises(ValueError):
            og.f_scalar(-0.1)


class TestProposition2_2AlgebraicProperties:
    def test_jordan_decomposition_and_properties(self):
        rng = np.random.default_rng(101)
        A = random_hermitian(8, rng, complex_mat=True)
        eigvals, eigvecs = np.linalg.eigh(A)
        A_plus = (eigvecs * np.maximum(eigvals, 0.0)) @ eigvecs.conj().T
        A_minus = (eigvecs * np.maximum(-eigvals, 0.0)) @ eigvecs.conj().T

        assert np.allclose(A_plus - A_minus, A, atol=1e-13)
        assert np.allclose(A_plus @ A_minus, 0.0, atol=1e-13)

        om_odd = og.omega_g_odd_matrix(A)
        om_p = og.omega_g_matrix(A_plus)
        om_m = og.omega_g_matrix(A_minus)

        # (1) Decomposition and orthogonality
        assert np.allclose(om_odd, om_p - om_m, atol=1e-13)
        assert np.allclose(om_p @ om_m, 0.0, atol=1e-13)

        # (2) Oddness, commutativity, and norm <= 1
        assert np.allclose(og.omega_g_odd_matrix(-A), -om_odd, atol=1e-13)
        assert np.allclose(om_odd @ A, A @ om_odd, atol=1e-13)
        assert np.linalg.norm(om_odd, ord=2) <= 1.0 + 1e-14

        # (3) Spectral mapping for n = 3
        om3 = og.omega_g_odd_matrix(A, iterations=3)
        spec_mat = np.sort(np.linalg.eigvalsh(om3))
        spec_scalar = np.sort(og.f_odd_scalar(og.f_odd_scalar(og.f_odd_scalar(eigvals))))
        assert np.allclose(spec_mat, spec_scalar, atol=1e-13)


class TestTheorem3_1FixedPointsAndBasins:
    def test_five_fixed_points_and_derivatives(self):
        for fp in (-1.0, -0.5, 0.0, 0.5, 1.0):
            assert math.isclose(og.f_odd_scalar(fp), fp, abs_tol=1e-15)

        assert math.isclose(og.f_prime(1.0), 0.0, abs_tol=1e-15)
        assert math.isclose(og.f_double_prime(1.0), -2.0, abs_tol=1e-14)
        expected_mult = 1.0 + 2.0 * math.log(2.0)
        assert math.isclose(og.f_prime(0.5), expected_mult, rel_tol=1e-14)
        assert math.isclose(og.REPELLER_MULTIPLIER, expected_mult, rel_tol=1e-14)

    def test_independent_x_max_and_basins(self):
        x_max = bisect_root(lambda t: og.f_scalar(t) - 0.5, 2.0, 4.0)
        assert math.isclose(x_max, 2.8874748541959434, rel_tol=1e-12)
        assert math.isclose(x_max, og.X_MAX, rel_tol=1e-12)

        def iterate(val: float, steps: int = 80) -> float:
            for _ in range(steps):
                val = og.f_odd_scalar(val)
            return val

        # B(+1) = (0.5, x_max)
        for x0 in (0.501, 0.7258, 1.0, 2.0, 2.88):
            assert math.isclose(iterate(x0), 1.0, abs_tol=1e-14)

        # B(-1) = (-x_max, -0.5)
        for x0 in (-2.88, -2.0, -1.0, -0.501):
            assert math.isclose(iterate(x0), -1.0, abs_tol=1e-14)

        # B(0) = (-inf, -x_max) U (-0.5, 0.5) U (x_max, inf)
        for x0 in (-100.0, -2.90, -0.499, 0.0, 0.499, 2.90, 100.0):
            assert math.isclose(iterate(x0), 0.0, abs_tol=1e-14)


class TestTheorem4_1FrechetExpansion:
    def test_fourth_order_scalar_expansion_and_old_proxy_divergence(self):
        # Verify f(1+h) = 1 - h^2 + 3/2 h^3 - 4/3 h^4 + O(h^5)
        h1, h2 = 2e-2, 1e-2
        r1 = abs(og.f_scalar(1 + h1) - (1 - h1**2 + 1.5 * h1**3 - (4 / 3) * h1**4))
        r2 = abs(og.f_scalar(1 + h2) - (1 - h2**2 + 1.5 * h2**3 - (4 / 3) * h2**4))
        # Fifth-order residual scales by 2^5 = 32
        assert 29.0 < (r1 / r2) < 35.0

        # Confirm the old Dual-Dilation cubic proxy A^-1(A - H^2 + 1/2 H^3) fails at 4th order (-3/2 vs -4/3)
        def old_proxy(h: float) -> float:
            return (1.0 + h - h**2 + 0.5 * h**3) / (1.0 + h)

        h_p = 2e-3
        diff_proxy = (og.f_scalar(1 + h_p) - old_proxy(h_p)) / (h_p**4)
        # Exact leading difference is (-4/3) - (-3/2) = +1/6
        assert math.isclose(diff_proxy, 1.0 / 6.0, rel_tol=2e-2)

    def test_first_and_corrected_second_frechet_derivatives(self):
        rng = np.random.default_rng(42)
        H = random_hermitian(5, rng)
        K = random_hermitian(5, rng)
        assert np.linalg.norm(H @ K - K @ H, ord=2) > 0.5
        I = np.eye(5)

        # First Fréchet derivative at I is identically zero (central diff error ~ 1.5 * eps^2 * ||H||^3)
        H_unit = H / np.linalg.norm(H, ord=2)
        eps = 1e-5
        D1 = (og.omega_g_matrix(I + eps * H_unit) - og.omega_g_matrix(I - eps * H_unit)) / (2.0 * eps)
        assert np.linalg.norm(D1, ord=2) < 1e-9

        # Mixed second Fréchet derivative D^2 Omega_G(I)[H, K] = -(HK + KH)
        e = 1e-3
        D2_num = (
            og.omega_g_matrix(I + e * H + e * K)
            - og.omega_g_matrix(I + e * H - e * K)
            - og.omega_g_matrix(I - e * H + e * K)
            + og.omega_g_matrix(I - e * H - e * K)
        ) / (4.0 * e * e)
        expected_D2 = -(H @ K + K @ H)
        old_wrong_D2 = -0.5 * (H @ K + K @ H)

        assert np.linalg.norm(D2_num - expected_D2, ord=2) < 1e-5
        assert np.linalg.norm(D2_num - old_wrong_D2, ord=2) > 0.5

        # Odd reflection around -I: D^2 Omega_G^odd(-I)[H, K] = +(HK + KH)
        D2_neg = (
            og.omega_g_odd_matrix(-I + e * H + e * K)
            - og.omega_g_odd_matrix(-I + e * H - e * K)
            - og.omega_g_odd_matrix(-I - e * H + e * K)
            + og.omega_g_odd_matrix(-I - e * H - e * K)
        ) / (4.0 * e * e)
        assert np.linalg.norm(D2_neg - (H @ K + K @ H), ord=2) < 1e-5

    def test_frechet_super_attraction_at_origin(self):
        rng = np.random.default_rng(7)
        H0 = random_hermitian(6, rng)
        H0 = H0 / np.linalg.norm(H0, ord=2)
        for s in (0.3, 0.1, 0.05, 0.01):
            H = s * H0
            norm_out = np.linalg.norm(og.omega_g_odd_matrix(H), ord=2)
            assert math.isclose(norm_out, og.f_scalar(s), rel_tol=1e-12, abs_tol=1e-15)
            assert norm_out <= math.e * math.exp(-1.0 / s) + 1e-16


class TestTheorem5_1StabilityRadiusAndBounds:
    def test_independent_constants_and_quadratic_contraction(self):
        x_crit = bisect_root(lambda t: og.f_prime(t) - 1.0, 0.6, 0.9)
        delta = 1.0 - x_crit
        k_delta = (1.0 - og.f_scalar(x_crit)) / (delta**2)
        c_delta = 0.5 * abs(og.f_double_prime(x_crit))

        assert math.isclose(x_crit, og.X_CRIT, rel_tol=1e-12)
        assert math.isclose(delta, og.DELTA, rel_tol=1e-12)
        assert math.isclose(k_delta, og.K_DELTA, rel_tol=1e-12)
        assert math.isclose(c_delta, og.C_DELTA, rel_tol=1e-12)

        grid = np.linspace(1.0 - delta, 1.0 + delta, 5000)
        assert np.max(np.abs(og.f_double_prime(grid))) <= abs(og.f_double_prime(x_crit)) + 1e-10
        assert k_delta * delta < 0.42
        assert c_delta * delta < 0.78

        rng = np.random.default_rng(99)
        I = np.eye(6)
        for _ in range(10):
            H = random_hermitian(6, rng)
            H = (delta * rng.uniform(0.1, 1.0)) * (H / np.linalg.norm(H, ord=2))
            h_norm = np.linalg.norm(H, ord=2)

            err1 = np.linalg.norm(og.omega_g_matrix(I + H) - I, ord=2)
            assert err1 <= k_delta * (h_norm**2) + 1e-14
            assert err1 <= c_delta * (h_norm**2) + 1e-14

            for n in (1, 2, 3, 4):
                err_n = np.linalg.norm(og.omega_g_matrix(I + H, iterations=n) - I, ord=2)
                bound_n = (1.0 / k_delta) * ((k_delta * h_norm) ** (2**n))
                assert err_n <= bound_n + 1e-14


class TestTheorem6_1SpectralProjectionsAndAdapters:
    def test_norm_convergence_to_partial_symmetry(self):
        rng = np.random.default_rng(123)
        eigs = np.array([-4.0, -2.1, -0.7, -0.25, 0.0, 0.3, 0.85, 2.5, 3.5])
        Q, _ = np.linalg.qr(rng.standard_normal((len(eigs), len(eigs))))
        A = (Q * eigs) @ Q.T

        p_plus_eigs = ((eigs > 0.5) & (eigs < og.X_MAX)).astype(float)
        p_minus_eigs = ((eigs < -0.5) & (eigs > -og.X_MAX)).astype(float)
        P_plus = (Q * p_plus_eigs) @ Q.T
        P_minus = (Q * p_minus_eigs) @ Q.T

        assert np.allclose(P_plus @ P_minus, 0.0, atol=1e-14)
        A_iter = og.omega_g_odd_matrix(A, iterations=40)
        assert np.linalg.norm(A_iter - (P_plus - P_minus), ord=2) < 1e-13

    def test_remark_6_2_adapters(self):
        rng = np.random.default_rng(321)
        eigs = np.array([-5.0, -1.5, -0.4, 0.1, 0.6, 2.2, 6.0])
        Q, _ = np.linalg.qr(rng.standard_normal((len(eigs), len(eigs))))
        D = (Q * eigs) @ Q.T
        tau = 1.0

        # Positive resolvent cutoff: |lambda| < tau^(-1/2) = 1.0
        R = og.positive_resolvent(D, tau)
        R_lim = og.omega_g_matrix(R, iterations=40)
        expected_low = (Q * (np.abs(eigs) < 1.0 / math.sqrt(tau)).astype(float)) @ Q.T
        assert np.linalg.norm(R_lim - expected_low, ord=2) < 1e-13

        # Odd bounded transform cutoff: |lambda| > (3*tau)^(-1/2)
        cut = 1.0 / math.sqrt(3.0 * tau)
        F = og.bounded_odd_transform(D, tau)
        F_lim = og.omega_g_odd_matrix(F, iterations=40)
        expected_signed = (
            Q * ((eigs > cut).astype(float) - (eigs < -cut).astype(float))
        ) @ Q.T
        assert np.linalg.norm(F_lim - expected_signed, ord=2) < 1e-13


class TestProposition7_1SymmetricBCH:
    def test_fourth_order_commutator_entry(self):
        rng = np.random.default_rng(2026)
        X0 = random_hermitian(4, rng)
        Y0 = random_hermitian(4, rng)
        X0 /= np.linalg.norm(X0, ord=2)
        Y0 /= np.linalg.norm(Y0, ord=2)
        I = np.eye(4)

        def comm(A, B):
            return A @ B - B @ A

        residuals_no_comm = []
        residuals_full = []
        for eps in (0.01, 0.005):
            X, Y = eps * X0, eps * Y0
            S = X + Y
            C3 = -(1.0 / 24.0) * comm(X, comm(X, Y)) - (1.0 / 12.0) * comm(Y, comm(X, Y))

            A = og.expm_numpy(0.5 * X) @ og.expm_numpy(Y) @ og.expm_numpy(0.5 * X)
            exact = og.omega_g_matrix(A)

            approx_commuting = I - (S @ S) + 0.5 * (S @ S @ S) + (1.0 / 3.0) * (S @ S @ S @ S)
            approx_full = approx_commuting - (S @ C3 + C3 @ S)

            r_nc = float(np.linalg.norm(exact - approx_commuting, ord=2))
            r_fl = float(np.linalg.norm(exact - approx_full, ord=2))
            residuals_no_comm.append(r_nc)
            residuals_full.append(r_fl)

        # Without -(SC3 + C3S), residual scales as O(eps^4) -> ratio ~ 16
        ratio_no_comm = residuals_no_comm[0] / residuals_no_comm[1]
        assert 14.5 < ratio_no_comm < 17.5

        # With -(SC3 + C3S), residual scales as O(eps^5) -> ratio ~ 32
        ratio_full = residuals_full[0] / residuals_full[1]
        assert 28.0 < ratio_full < 36.0
        assert residuals_full[1] < 0.05 * residuals_no_comm[1]


# ===========================================================================
# ALL FIVE EDGE CASES
# ===========================================================================
class TestEdgeCase1FoldbackAboveXMax:
    def test_non_monotonic_foldback_and_rapid_spike_collapse(self):
        f3 = og.f_scalar(3.0)
        assert math.isclose(f3, 3.0 ** (-2.0 / 3.0), rel_tol=1e-14)
        assert f3 < 0.5

        val = 3.0
        for _ in range(20):
            val = og.f_scalar(val)
        assert val == 0.0

        step1 = og.f_scalar(100.0)
        assert math.isclose(step1, 100.0 ** (-0.99), rel_tol=1e-14)
        step2 = og.f_scalar(step1)
        assert step2 < 1e-180
        step3 = og.f_scalar(step2)
        assert step3 == 0.0


class TestEdgeCase2RepellerSlowdownAndBoundaryBreakdown:
    def test_critical_slowing_down_near_half(self):
        def steps_to_lock(x0: float, target: float = 1.0, tol: float = 1e-12) -> int:
            x = x0
            for n in range(1, 200):
                x = og.f_scalar(x)
                if abs(x - target) < tol:
                    return n
            raise AssertionError("Did not converge")

        fast_steps = steps_to_lock(0.9)
        slow_steps = steps_to_lock(0.5 + 1e-10)
        assert fast_steps <= 6
        assert 24 <= slow_steps <= 38

    def test_eigenvalue_pinned_at_repeller_fails_projection_identity(self):
        A = np.diag([0.2, 0.5, 0.9])
        A_lim = og.omega_g_matrix(A, iterations=50)
        assert np.allclose(np.diag(A_lim), [0.0, 0.5, 1.0], atol=1e-14)
        assert np.linalg.norm(A_lim @ A_lim - A_lim, ord=2) == pytest.approx(0.25, abs=1e-14)

    def test_discretized_continuous_spectrum_fails_uniform_norm_convergence(self):
        grid = np.linspace(0.0, 1.0, 4001)
        step_target = (grid > 0.5).astype(float)
        vals = grid.copy()
        for _ in range(15):
            vals = og.f_scalar(vals)
        assert float(np.max(np.abs(vals - step_target))) >= 0.45


class TestEdgeCase3JordanBlockAnnihilation:
    @pytest.mark.parametrize("m", [2, 3, 4, 5, 8])
    def test_exact_ceil_log2_annihilation_at_identity(self, m: int):
        N = np.eye(m, k=1)
        J = np.eye(m) + N
        expected_steps = math.ceil(math.log2(m))

        prev = og.omega_g_nonnormal(J, iterations=expected_steps - 1)
        assert np.linalg.norm(prev - np.eye(m), ord="fro") >= 0.99

        final = og.omega_g_nonnormal(J, iterations=expected_steps)
        assert np.linalg.norm(final - np.eye(m), ord="fro") < 1e-11

    def test_jordan_defect_amplification_at_repeller(self):
        eps = 1e-4
        J_rep = np.array([[0.5, eps], [0.0, 0.5]], dtype=np.float64)
        n_steps = 5
        out = og.omega_g_nonnormal(J_rep, iterations=n_steps)

        expected_offdiag = eps * (og.REPELLER_MULTIPLIER**n_steps)
        assert math.isclose(out[0, 0], 0.5, abs_tol=1e-13)
        assert math.isclose(out[1, 1], 0.5, abs_tol=1e-13)
        assert math.isclose(out[0, 1], expected_offdiag, rel_tol=1e-11)


class TestEdgeCase4Float64UnderflowAndPrecision:
    def test_subnormal_and_underflow_cliffs(self):
        tiny = np.finfo(np.float64).tiny

        def raw_log_exp(x: float) -> float:
            return math.exp(((1.0 - x) / x) * math.log(x))

        assert raw_log_exp(0.0070) >= tiny
        val_sub = raw_log_exp(0.0069)
        assert 0.0 < val_sub < tiny
        assert raw_log_exp(0.0066) == 0.0

        lo, hi = 0.0066, 0.0068
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if raw_log_exp(mid) == 0.0:
                lo = mid
            else:
                hi = mid
        assert 0.00665 < hi < 0.00670
        assert og.FLOAT64_UNDERFLOW_CLAMP < lo

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            extreme = np.array([0.0, 1e-300, 1e-12, 0.0066, 0.00665])
            out = og.f_scalar(extreme)
            assert np.all(out == 0.0)
            assert np.all(np.isfinite(out))
            assert len(caught) == 0

    def test_near_unity_precision(self):
        h = 1e-5
        expected_diff = h**2 - 1.5 * h**3 + (4.0 / 3.0) * h**4
        actual_diff = 1.0 - og.f_scalar(1.0 + h)
        assert abs(actual_diff - expected_diff) < 1e-14


class TestEdgeCase5BandpassOutlierRejection:
    def test_bandpass_beats_monotonic_gates_under_impulse_spikes(self):
        rng = np.random.default_rng(777)
        n = 5000
        truth = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.25, 0.50, 0.25])
        signal = truth + rng.normal(0.0, 0.12, size=n)

        zero_idx = np.flatnonzero(truth == 0.0)
        spike_idx = rng.choice(zero_idx, size=int(0.16 * len(zero_idx)), replace=False)
        signal[spike_idx] += rng.choice([-1.0, 1.0], size=len(spike_idx)) * rng.uniform(
            5.0, 15.0, size=len(spike_idx)
        )

        y_omega = signal.copy()
        for _ in range(10):
            y_omega = og.f_odd_scalar(y_omega)

        y_hard = np.where(np.abs(signal) > 0.5, np.sign(signal), 0.0)
        y_tanh = np.tanh(2.5 * signal)
        y_band_hard = np.where(
            (np.abs(signal) > 0.5) & (np.abs(signal) < og.X_MAX), np.sign(signal), 0.0
        )

        mse_omega = float(np.mean((y_omega - truth) ** 2))
        mse_hard = float(np.mean((y_hard - truth) ** 2))
        mse_tanh = float(np.mean((y_tanh - truth) ** 2))
        mse_band_hard = float(np.mean((y_band_hard - truth) ** 2))

        assert mse_omega < 0.25 * mse_hard
        assert mse_omega < 0.25 * mse_tanh
        assert abs(mse_omega - mse_band_hard) < 0.005
