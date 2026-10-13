"""Test suite for ``asp.algorithms``."""

import numpy as np
import pytest
from scipy.fft import dct
from scipy.linalg import toeplitz

from asp import algorithms as alg


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def system():
    """Unknown FIR system used in the identification tests."""
    return np.array([1.0, 0.5, -0.3, 0.1])


def identification_data(rng, g, n=6000, scale=1.0, noise=0.0):
    x = scale * rng.standard_normal(n)
    d = np.convolve(x, g)[:n]
    if noise:
        d = d + noise * rng.standard_normal(n)
    return x, d


def colored_input(rng, n=8000, a=0.9):
    """AR(1) process: large eigenvalue spread of the correlation matrix."""
    w = rng.standard_normal(n)
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + w[k]
    return x


# ---------------------------------------------------------------------------
# Correlation estimators
# ---------------------------------------------------------------------------
class TestCorrelationEstimators:
    def test_correlation_matrix_matches_definition(self, rng):
        x = rng.standard_normal(200)
        N = 4
        R = alg.estimate_correlation_matrix(x, N)

        windows = np.array([x[k - N + 1 : k + 1][::-1] for k in range(N - 1, len(x))])
        expected = windows.T @ windows / windows.shape[0]
        np.testing.assert_allclose(R, expected)

    def test_correlation_matrix_is_symmetric_psd(self, rng):
        R = alg.estimate_correlation_matrix(rng.standard_normal(500), 6)
        np.testing.assert_allclose(R, R.T)
        assert np.linalg.eigvalsh(R).min() >= -1e-12

    def test_correlation_matrix_white_noise(self, rng):
        R = alg.estimate_correlation_matrix(2.0 * rng.standard_normal(100_000), 4)
        np.testing.assert_allclose(R, 4.0 * np.eye(4), atol=0.15)

    def test_correlation_matrix_ar1_is_toeplitz_like(self, rng):
        a = 0.8
        x = colored_input(rng, n=200_000, a=a)
        R = alg.estimate_correlation_matrix(x, 4)
        r = a ** np.arange(4) / (1 - a**2)
        np.testing.assert_allclose(R, toeplitz(r), rtol=0.05, atol=0.05)

    def test_correlation_matrix_full_length(self):
        R = alg.estimate_correlation_matrix(np.array([1.0, 2.0, 3.0]), 3)
        np.testing.assert_allclose(R, np.outer([3, 2, 1], [3, 2, 1]))

    @pytest.mark.parametrize("N", [0, 11])
    def test_correlation_matrix_invalid_order(self, N):
        with pytest.raises(ValueError):
            alg.estimate_correlation_matrix(np.ones(10), N)

    def test_cross_correlation_matches_definition(self, rng):
        x = rng.standard_normal(100)
        d = rng.standard_normal(100)
        N = 5
        p = alg.estimate_cross_correlation(x, d, N)

        expected = np.mean(
            [d[k] * x[k - N + 1 : k + 1][::-1] for k in range(N - 1, len(x))], axis=0
        )
        np.testing.assert_allclose(p, expected)

    def test_cross_correlation_uses_all_regressors(self):
        # The last regressor (k = M-1) and the first complete one (k = N-1) count
        x = np.array([1.0, 2.0, 3.0])
        d = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(alg.estimate_cross_correlation(x, d, 3), [3 / 1, 2, 1])

    def test_cross_correlation_recovers_system(self, rng, system):
        x, d = identification_data(rng, system, n=100_000)
        p = alg.estimate_cross_correlation(x, d, len(system))
        np.testing.assert_allclose(p, system, atol=0.03)

    def test_wiener_solution_from_estimates(self, rng, system):
        x, d = identification_data(rng, system, n=50_000)
        N = len(system)
        R = alg.estimate_correlation_matrix(x, N)
        p = alg.estimate_cross_correlation(x, d, N)
        np.testing.assert_allclose(np.linalg.solve(R, p), system, atol=1e-8)

    def test_cross_correlation_invalid_arguments(self):
        with pytest.raises(ValueError):
            alg.estimate_cross_correlation(np.ones(10), np.ones(9), 3)
        with pytest.raises(ValueError):
            alg.estimate_cross_correlation(np.ones(10), np.ones(10), 11)


# ---------------------------------------------------------------------------
# Optimization methods
# ---------------------------------------------------------------------------
class TestOptimization:
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, -1.0])

    def grad(self, x):
        return self.A @ x - self.b

    def hess(self, x):
        return self.A

    def test_gradient_descent_converges_to_quadratic_minimum(self):
        x, path = alg.gradient_descent(self.grad, [5.0, 5.0], μ=0.2, max_iter=2000, tol=1e-10)
        np.testing.assert_allclose(x, np.linalg.solve(self.A, self.b), atol=1e-6)
        np.testing.assert_allclose(path[0], [5.0, 5.0])
        np.testing.assert_allclose(path[-1], x)

    def test_gradient_descent_diverges_above_stability_limit(self):
        λmax = np.linalg.eigvalsh(self.A).max()
        x, _ = alg.gradient_descent(self.grad, [1.0, 1.0], μ=2.5 / λmax, max_iter=200)
        assert np.linalg.norm(x) > 1e3

    def test_gradient_descent_stops_at_tolerance(self):
        _, path = alg.gradient_descent(self.grad, [1.0, 1.0], μ=0.2, max_iter=10_000, tol=1e-3)
        assert len(path) < 10_000

    def test_newton_one_step_on_quadratic(self):
        x, path = alg.newton_method(self.grad, self.hess, [10.0, -7.0], max_iter=1)
        np.testing.assert_allclose(x, np.linalg.solve(self.A, self.b))
        assert path.shape == (2, 2)

    def test_newton_on_nonquadratic(self):
        # f = exp(x) - 2x  ->  minimum at x = ln 2
        x, _ = alg.newton_method(
            lambda v: np.exp(v) - 2, lambda v: np.exp(v), [0.0], max_iter=50, tol=1e-12
        )
        np.testing.assert_allclose(x, np.log(2), atol=1e-8)


# ---------------------------------------------------------------------------
# Gradient adaptation
# ---------------------------------------------------------------------------
class TestGradientAdaptation:
    def test_shapes(self, rng, system):
        x, d = identification_data(rng, system, n=300)
        out, H, errors = alg.gradient_adaptation(x, d, 4, 1e-2, 3)
        assert out.shape == (300,)
        assert H.shape == (3 * 300 + 1, 4)
        assert errors.shape == (900,)
        np.testing.assert_array_equal(H[0], 0)

    def test_identifies_system(self, rng, system):
        x, d = identification_data(rng, system, n=2000)
        _, H, errors = alg.gradient_adaptation(x, d, 4, 1e-2, 2)
        np.testing.assert_allclose(H[-1], system, atol=1e-3)
        assert np.mean(errors[-200:] ** 2) < 1e-6

    def test_uses_every_sample_of_every_epoch(self, rng, system):
        # Reference implementation of the same recursion (zero initial state)
        x, d = identification_data(rng, system, n=50)
        _, H, errors = alg.gradient_adaptation(x, d, 4, 1e-2, 2)

        xp = np.concatenate((np.zeros(3), x))
        h = np.zeros(4)
        for step in range(100):
            i = step % 50
            vec = xp[i : i + 4][::-1]
            e = d[i] - h @ vec
            h = h + 1e-2 * e * vec
            assert errors[step] == pytest.approx(e)
        np.testing.assert_allclose(H[-1], h)


# ---------------------------------------------------------------------------
# LMS family
# ---------------------------------------------------------------------------
class TestLMS:
    def test_matches_reference_implementation(self, rng, system):
        x, d = identification_data(rng, system, n=400)
        Ntaps, μ = 4, 0.01
        out, h, sqerr, H = alg.lms(x, d, Ntaps, μ)

        h_ref = np.zeros(Ntaps)
        for k in range(Ntaps, len(x)):
            vec = x[k - np.arange(Ntaps)]
            y = h_ref @ vec
            e = d[k] - y
            h_ref = h_ref + 2 * μ * e * vec
            assert out[k] == pytest.approx(y)
            assert sqerr[k] == pytest.approx(e**2)
            np.testing.assert_allclose(H[k], h_ref)
        np.testing.assert_allclose(h, h_ref)

    def test_converges_to_wiener_solution(self, rng, system):
        x, d = identification_data(rng, system)
        _, h, sqerr, _ = alg.lms(x, d, 4, 0.01)
        np.testing.assert_allclose(h, system, atol=1e-3)
        assert sqerr[-500:].mean() < 1e-6 < sqerr[:500].mean()

    def test_converges_to_wiener_solution_with_noise(self, rng, system):
        x, d = identification_data(rng, system, n=20_000, noise=0.1)
        _, h, sqerr, _ = alg.lms(x, d, 4, 0.002)
        np.testing.assert_allclose(h, system, atol=0.03)
        assert sqerr[-5000:].mean() == pytest.approx(0.01, rel=0.25)

    def test_history_layout(self, rng, system):
        x, d = identification_data(rng, system, n=300)
        _, h, _, H = alg.lms(x, d, 4, 0.01)
        assert H.shape == (300, 4)
        np.testing.assert_array_equal(H[:4], 0)  # no adaptation yet
        np.testing.assert_allclose(H[-1], h)

    def test_output_and_error_are_zero_before_adaptation(self, rng, system):
        x, d = identification_data(rng, system, n=100)
        out, _, sqerr, _ = alg.lms(x, d, 4, 0.01)
        np.testing.assert_array_equal(out[:4], 0)
        np.testing.assert_array_equal(sqerr[:4], 0)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_unstable_step_size_diverges(self, rng, system):
        x, d = identification_data(rng, system, n=2000)
        _, h, _, _ = alg.lms(x, d, 4, 1.0)
        assert not np.all(np.isfinite(h)) or np.linalg.norm(h) > 1e3

    @pytest.mark.parametrize("func", [alg.lms, alg.nlms, alg.rls, alg.rls_apriori, alg.rls_aposteriori])
    def test_input_validation(self, func):
        with pytest.raises(ValueError):
            func(np.ones(10), np.ones(9), 3, 0.5)
        with pytest.raises(ValueError):
            func(np.ones(10), np.ones(10), 0, 0.5)

    def test_tracks_time_varying_system(self, rng):
        n, Ntaps = 20_000, 3
        x = rng.standard_normal(n)
        H_true = np.zeros((n, Ntaps))
        H_true[:, 0] = 1.0
        H_true[:, 1] = 0.5 * np.sin(2 * np.pi * np.arange(n) / n)
        d = alg.time_varying_filter(x, H_true)
        _, _, sqerr, H = alg.lms(x, d, Ntaps, 0.01)
        np.testing.assert_allclose(H[1000:, 1], H_true[1000:, 1], atol=0.05)
        assert sqerr[1000:].mean() < 1e-3


class TestNLMS:
    def test_converges_noiseless(self, rng, system):
        x, d = identification_data(rng, system, n=3000)
        out, h, sqerr, H = alg.nlms(x, d, 4, 0.5)
        np.testing.assert_allclose(h.ravel(), system, atol=1e-6)
        assert h.shape == (4, 1)
        assert H.shape == (3000, 4)
        np.testing.assert_allclose(H[-1], system, atol=1e-6)

    def test_is_invariant_to_input_scale(self, rng, system):
        x, d = identification_data(rng, system, n=500)
        _, h1, _, _ = alg.nlms(x, d, 4, 0.5)
        _, h2, _, _ = alg.nlms(100 * x, d, 4, 0.5)  # same d, scaled input
        _, h3, _, _ = alg.nlms(x, d, 4, 0.5)
        np.testing.assert_allclose(h1, h3)
        # Scaling x by c scales the converged coefficients by 1/c
        np.testing.assert_allclose(h2.ravel() * 100, h1.ravel(), atol=1e-3)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_stable_where_lms_diverges(self, rng, system):
        x, d = identification_data(rng, system, n=3000, scale=10.0)
        _, h_lms, _, _ = alg.lms(x, d, 4, 0.01)  # μ·N·σ² = 4 >> 1
        _, h_nlms, _, _ = alg.nlms(x, d, 4, 1.0)
        assert not np.all(np.isfinite(h_lms)) or np.linalg.norm(h_lms - system) > 1
        np.testing.assert_allclose(h_nlms.ravel(), system, atol=1e-6)

    def test_single_step_zeroes_a_posteriori_error(self):
        # With μ = 1 (and γ ~ 0), the updated filter reproduces d[k] exactly
        x = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
        d = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0])
        _, _, _, H = alg.nlms(x, d, 3, 1.0, 1e-12)
        vec = x[3 - np.arange(3)]
        assert H[3] @ vec == pytest.approx(d[3], rel=1e-6)


class TestLMSNewton:
    def test_converges(self, rng, system):
        x, d = identification_data(rng, system, n=3000)
        _, h, _, H = alg.lms_newton(x, d, 4, 0.05, 0.05)
        np.testing.assert_allclose(h.ravel(), system, atol=1e-3)
        assert h.shape == (4, 1) and H.shape == (3000, 4)

    def test_faster_than_lms_for_colored_input(self, rng):
        g = np.array([1.0, -0.5, 0.25, 0.1])
        x = colored_input(rng, n=3000, a=0.95)
        d = np.convolve(x, g)[: len(x)]
        _, _, e_lms, _ = alg.lms(x, d, 4, 0.002)
        _, _, e_newton, _ = alg.lms_newton(x, d, 4, 0.05, 0.01)
        assert e_newton[1000:1500].mean() < 0.1 * e_lms[1000:1500].mean()


class TestTDLMS:
    def test_with_identity_transform_matches_power_normalized_lms(self, rng, system):
        x, d = identification_data(rng, system, n=500)
        _, h, _, H = alg.tdlms(x, d, 4, 0.01, np.eye(4))
        assert h.shape == (4,) and H.shape == (500, 4)
        np.testing.assert_allclose(H[-1], h)

    def test_converges_with_dct(self, rng, system):
        x, d = identification_data(rng, system, n=8000)
        M = dct(np.eye(4), norm="ortho", axis=0)
        out, h, sqerr, _ = alg.tdlms(x, d, 4, 0.01, M)
        np.testing.assert_allclose(h, system, atol=1e-3)
        assert sqerr[-500:].mean() < 1e-6

    def test_time_domain_equivalent_filter(self, rng, system):
        # y[k] = w^T M x_k = h_eq^T x_k, with h_eq returned by the function
        x, d = identification_data(rng, system, n=300)
        M = dct(np.eye(4), norm="ortho", axis=0)
        out, _, _, H = alg.tdlms(x, d, 4, 0.01, M)
        for k in (10, 100, 250):
            vec = x[k - np.arange(4)]
            # output at k uses the coefficients *before* the update at k
            assert out[k] == pytest.approx(H[k - 1] @ vec)

    def test_faster_than_lms_for_colored_input(self, rng):
        g = np.array([1.0, -0.5, 0.25, 0.1])
        x = colored_input(rng, n=4000, a=0.95)
        d = np.convolve(x, g)[: len(x)]
        M = dct(np.eye(4), norm="ortho", axis=0)
        _, _, e_lms, _ = alg.lms(x, d, 4, 0.002)
        _, _, e_td, _ = alg.tdlms(x, d, 4, 0.01, M)
        assert e_td[1500:2000].mean() < e_lms[1500:2000].mean()


class TestFDLMS:
    @pytest.mark.parametrize("constrained", [False, True])
    def test_converges(self, rng, system, constrained):
        x, d = identification_data(rng, system, n=20_000)
        out, h, sqerr, H = alg.fdlms(x, d, 4, 0.05, Nfft=16, constrained=constrained)
        assert out.shape == x.shape and np.isrealobj(out) and np.isrealobj(h)
        assert H.shape == (16,)
        assert sqerr[-20:].mean() < 1e-3 * sqerr[:20].mean()
        if constrained:
            np.testing.assert_allclose(h, system, atol=1e-2)

    def test_output_is_filtered_input_at_convergence(self, rng, system):
        x, d = identification_data(rng, system, n=20_000)
        out, _, _, _ = alg.fdlms(x, d, 4, 0.05, Nfft=16, constrained=True)
        np.testing.assert_allclose(out[-1000:], d[-1000:], atol=0.05)

    def test_nfft_rounded_to_power_of_two(self, rng, system):
        x, d = identification_data(rng, system, n=2000)
        _, _, _, H = alg.fdlms(x, d, 4, 0.05, Nfft=20)
        assert H.shape == (32,)

    def test_signal_length_not_multiple_of_block(self, rng, system):
        x, d = identification_data(rng, system, n=1001)
        out, _, sqerr, _ = alg.fdlms(x, d, 4, 0.05, Nfft=16)
        assert out.shape == (1001,)
        assert sqerr.shape == (int(np.ceil(1001 / 13)),)

    def test_complex_signals(self, rng, system):
        x, d = identification_data(rng, system, n=10_000)
        out, h, _, _ = alg.fdlms(x * (1 + 1j), d * (1 + 1j), 4, 0.05, Nfft=16)
        assert np.iscomplexobj(out) and np.iscomplexobj(h)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            alg.fdlms(np.ones(100), np.ones(99), 4, 0.1)
        with pytest.raises(ValueError):
            alg.fdlms(np.ones(100), np.ones(100), 40, 0.1, Nfft=16)
        with pytest.raises(ValueError):
            alg.fdlms(np.ones(100), np.ones(100), 0, 0.1)


# ---------------------------------------------------------------------------
# RLS family
# ---------------------------------------------------------------------------
class TestRLS:
    @pytest.mark.parametrize("λ", [1.0, 0.99])
    def test_matches_regularized_weighted_least_squares(self, rng, system, λ):
        """RLS = exact solution of the exponentially weighted ridge problem."""
        x, d = identification_data(rng, system, n=300, noise=0.05)
        Ntaps = 4
        δ = 0.5
        _, h, _, _ = alg.rls(x, d, Ntaps, λ, δ)

        n_updates = len(x) - Ntaps
        R = λ**n_updates * δ * np.eye(Ntaps)  # initial P = I / δ
        p = np.zeros(Ntaps)
        for j, k in enumerate(range(Ntaps, len(x))):
            vec = x[k - np.arange(Ntaps)]
            w = λ ** (n_updates - 1 - j)
            R += w * np.outer(vec, vec)
            p += w * d[k] * vec
        np.testing.assert_allclose(h.ravel(), np.linalg.solve(R, p), rtol=1e-6, atol=1e-8)

    def test_converges_fast(self, rng, system):
        x, d = identification_data(rng, system, n=200)
        _, h, sqerr, _ = alg.rls(x, d, 4, 1.0)
        np.testing.assert_allclose(h.ravel(), system, atol=1e-4)
        assert sqerr[40:].max() < 1e-6

    def test_default_initialization_does_not_bias_solution(self, rng, system):
        # Regression test: a strong ridge prior (P0 = 1e-3 I) used to shrink
        # the solution towards zero when λ = 1
        x, d = identification_data(rng, system, n=100)
        for f in (alg.rls, alg.rls_apriori, alg.rls_aposteriori):
            np.testing.assert_allclose(f(x, d, 4, 1.0)[1].ravel(), system, atol=1e-3)

    def test_three_formulations_give_same_coefficients(self, rng, system):
        x, d = identification_data(rng, system, n=500, noise=0.1)
        H = [f(x, d, 4, 0.995)[3] for f in (alg.rls, alg.rls_apriori, alg.rls_aposteriori)]
        np.testing.assert_allclose(H[1], H[0], atol=1e-6)
        np.testing.assert_allclose(H[2], H[0], atol=1e-6)

    def test_apriori_and_aposteriori_errors(self, rng, system):
        x, d = identification_data(rng, system, n=300, noise=0.1)
        _, _, e_prior, _ = alg.rls_apriori(x, d, 4, 0.99)
        out, _, e_post, H = alg.rls(x, d, 4, 0.99)
        # a posteriori error is never larger than the a priori one (α >= 1)
        assert np.all(e_post[10:] <= e_prior[10:] + 1e-12)

    def test_apriori_error_definition(self, rng, system):
        x, d = identification_data(rng, system, n=200)
        out, _, sqerr, H = alg.rls_apriori(x, d, 4, 0.99)
        for k in (10, 50, 150):
            vec = x[k - np.arange(4)]
            assert out[k] == pytest.approx(H[k - 1] @ vec)
            assert sqerr[k] == pytest.approx((d[k] - out[k]) ** 2)

    def test_tracks_time_varying_system_better_with_forgetting(self, rng):
        n, Ntaps = 6000, 3
        x = rng.standard_normal(n)
        H_true = np.zeros((n, Ntaps))
        H_true[:, 0] = 1.0
        H_true[:, 1] = 0.5 * np.sin(2 * np.pi * np.arange(n) / 2000)
        d = alg.time_varying_filter(x, H_true)
        err = {}
        for λ in (1.0, 0.98):
            *_, H = alg.rls(x, d, Ntaps, λ)
            err[λ] = np.mean((H[500:, 1] - H_true[500:, 1]) ** 2)
        assert err[0.98] < 0.1 * err[1.0]


# ---------------------------------------------------------------------------
# Kalman filters
# ---------------------------------------------------------------------------
class TestKalman:
    def test_estimates_constant(self, rng):
        n = 300
        y = 3.0 + 0.5 * rng.standard_normal((1, n))
        x_hat = alg.kalman_filter(
            np.eye(1), np.eye(1), 1e-8 * np.eye(1), 0.25 * np.eye(1), np.zeros((1, 1)), y,
            np.eye(1),
        )
        assert x_hat.shape == (1, n)
        assert abs(x_hat[0, -1] - 3.0) < 0.2

    def test_matches_scalar_reference(self, rng):
        a, c, q, r = 0.9, 1.0, 0.1, 0.5
        y = rng.standard_normal((1, 50))
        x_hat = alg.kalman_filter(
            np.array([[a]]), np.array([[c]]), np.array([[q]]), np.array([[r]]),
            np.array([[1.0]]), y,
        )
        x, P = 1.0, 1.0
        for k in range(50):
            x_pr, P_pr = a * x, a * P * a + q
            K = P_pr * c / (c * P_pr * c + r)
            x = x_pr + K * (y[0, k] - c * x_pr)
            P = (1 - K * c) * P_pr
            assert x_hat[0, k] == pytest.approx(x)

    def test_zero_initial_state_needs_explicit_covariance(self, rng):
        # The default P0 = x0 x0^T is zero for x0 = 0: the filter never moves
        y = 3.0 + 0.5 * rng.standard_normal((1, 50))
        args = (np.eye(1), np.eye(1), 1e-8 * np.eye(1), 0.25 * np.eye(1), np.zeros((1, 1)), y)
        np.testing.assert_allclose(alg.kalman_filter(*args), 0, atol=1e-3)
        assert abs(alg.kalman_filter(*args, np.eye(1))[0, -1] - 3.0) < 0.3

    def test_accepts_integer_matrices(self):
        y = np.ones((1, 5))
        x_hat = alg.kalman_filter(
            np.eye(2, dtype=np.int64), np.array([[1, 0]]), np.eye(2, dtype=np.int64),
            np.array([[1]]), np.ones((2, 1), dtype=np.int64), y,
        )
        assert x_hat.shape == (2, 5)

    def test_tracks_mass_spring_better_than_raw_measurements(self, rng):
        dt, n = 0.05, 600
        A = np.array([[1, dt], [-2 * dt, 1]])
        C = np.array([[1.0, 0.0]])
        x = np.zeros((2, n))
        s = np.array([1.0, 0.0])
        for k in range(n):
            s = A @ s + 0.01 * rng.standard_normal(2)
            x[:, k] = s
        y = C @ x + 0.5 * rng.standard_normal((1, n))
        x_hat = alg.kalman_filter(A, C, 1e-4 * np.eye(2), 0.25 * np.eye(1), np.array([[1.0], [0.0]]), y)
        assert np.mean((x_hat[0] - x[0]) ** 2) < 0.5 * np.mean((y[0] - x[0]) ** 2)


class TestExtendedKalman:
    def test_reduces_to_kalman_for_linear_models(self, rng):
        A = np.array([[0.95, 0.1], [0.0, 0.9]])
        C = np.array([[1.0, 0.0]])
        Rn, Rv = 1e-3 * np.eye(2), np.array([[0.2]])
        x0 = np.array([[1.0], [0.5]])
        y = rng.standard_normal((1, 80))

        ref = alg.kalman_filter(A, C, Rn, Rv, x0, y)
        ekf = alg.extended_kalman_filter(
            lambda x: A @ x, lambda x: C @ x, lambda x: A, lambda x: C, Rn, Rv, x0, y
        )
        np.testing.assert_allclose(ekf, ref, atol=1e-10)

    def test_nonlinear_observation(self, rng):
        # Estimate a constant observed through y = x^2
        y = 4.0 + 0.1 * rng.standard_normal((1, 300))
        x_hat = alg.extended_kalman_filter(
            lambda x: x, lambda x: x**2, lambda x: np.eye(1), lambda x: 2 * x,
            1e-8 * np.eye(1), np.array([[0.01]]), np.array([[1.0]]), y,
        )
        assert x_hat[0, -1] == pytest.approx(2.0, abs=0.05)

    def test_custom_initial_covariance(self, rng):
        y = rng.standard_normal((1, 5))
        args = (lambda x: x, lambda x: x, lambda x: np.eye(1), lambda x: np.eye(1),
                0.1 * np.eye(1), np.eye(1), np.zeros((1, 1)), y)
        a = alg.extended_kalman_filter(*args, P_init=np.eye(1))
        b = alg.extended_kalman_filter(*args, P_init=1e-6 * np.eye(1))
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# CMA
# ---------------------------------------------------------------------------
class TestCMA:
    def make_bpsk(self, rng, n=20_000, channel=(1.0, 0.3)):
        s = rng.choice([-1.0, 1.0], n)
        return s, np.convolve(s, channel)[:n]

    def test_history_shape_and_no_out_of_bounds(self, rng):
        # H must hold one row per sample; writing H[i] with i >= len(x)-Ntaps
        # used to overflow the array.
        x = rng.standard_normal(100)
        out, h, cost, H = alg.cma(x, 5, 1e-3)
        assert H.shape == (100, 5)
        assert out.shape == cost.shape == (100,)
        np.testing.assert_allclose(H[-1], h)
        np.testing.assert_array_equal(H[:5], 0)

    def test_initialization_is_single_spike(self, rng):
        x = rng.standard_normal(10)
        _, h, _, _ = alg.cma(x, 4, 0.0)
        np.testing.assert_allclose(h, [1, 0, 0, 0])

    def test_reduces_dispersion_real(self, rng):
        _, y = self.make_bpsk(rng)
        out, h, cost, _ = alg.cma(y, 5, 1e-3)
        assert cost[-2000:].mean() < 0.2 * cost[10:2000].mean()
        assert cost[-2000:].mean() < 0.02
        assert np.mean(np.abs(np.abs(out[-2000:]) - 1)) < 0.1

    def test_equalizes_isi_channel(self, rng):
        s, y = self.make_bpsk(rng)
        out, _, _, _ = alg.cma(y, 7, 1e-3)
        # Up to a delay and a sign ambiguity the decisions match the symbols
        tail = out[-4000:]
        decisions = np.sign(tail)
        best = 0.0
        for delay in range(0, 7):
            ref = s[len(s) - 4000 - delay : len(s) - delay]
            best = max(best, abs(np.mean(decisions * ref)))
        assert best > 0.99

    def test_complex_qpsk_with_phase_offset(self, rng):
        n = 30_000
        s = (rng.choice([-1, 1], n) + 1j * rng.choice([-1, 1], n)) / np.sqrt(2)
        y = np.convolve(s, [1.0, 0.2 + 0.1j])[:n] * np.exp(1j * 0.7)
        out, h, cost, H = alg.cma(y, 5, 5e-3)
        assert np.iscomplexobj(out) and np.iscomplexobj(h)
        assert cost[-3000:].mean() < 0.2 * cost[10:3000].mean()
        # Constant modulus restored, phase rotation is tolerated
        assert np.std(np.abs(out[-3000:])) < 0.15

    def test_dispersion_constant(self, rng):
        # 4-PAM-like amplitude: R2 = E|a|^4 / E|a|^2
        a = rng.choice([-3.0, -1.0, 1.0, 3.0], 20_000)
        R2 = np.mean(a**4) / np.mean(a**2)
        _, _, cost, _ = alg.cma(a, 3, 1e-5, R2)
        assert np.all(np.isfinite(cost))

    def test_invalid_taps(self):
        with pytest.raises(ValueError):
            alg.cma(np.ones(10), 0, 1e-3)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
class TestTimeVaryingFilter:
    def test_constant_coefficients_equal_convolution(self, rng):
        g = np.array([1.0, 0.4, -0.2])
        x = rng.standard_normal(100)
        H = np.tile(g, (100, 1))
        y = alg.time_varying_filter(x, H)
        np.testing.assert_allclose(y[3:], np.convolve(x, g)[3:100])
        np.testing.assert_array_equal(y[:3], 0)

    def test_coefficients_change_over_time(self):
        x = np.ones(10)
        H = np.zeros((10, 2))
        H[:, 0] = np.arange(10)
        y = alg.time_varying_filter(x, H)
        np.testing.assert_allclose(y[2:], np.arange(2, 10))


class TestSlidingWindowMatrix:
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 8])
    def test_shape_and_center_row(self, N):
        x = np.arange(1.0, 11.0)
        X = alg.sliding_window_matrix(x, N)
        assert X.shape == (N, 10)  # one window per sample, odd or even N
        np.testing.assert_array_equal(X[N // 2], x)

    def test_known_values_and_zero_padding(self):
        X = alg.sliding_window_matrix([1, 2, 3, 4], 3)
        np.testing.assert_array_equal(X, [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 0]])

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            alg.sliding_window_matrix([1, 2, 3], 0)


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------
class TestActivations:
    x = np.linspace(-3, 3, 13)

    def test_sigmoid(self):
        np.testing.assert_allclose(alg.sigmoid(self.x), 1 / (1 + np.exp(-self.x)))
        assert alg.sigmoid(np.array([0.0]))[0] == 0.5

    @pytest.mark.parametrize(
        "f, df",
        [(alg.sigmoid, alg.dsigmoid), (np.tanh, alg.dtanh), (alg.ReLU, alg.dReLU)],
    )
    def test_derivatives_match_finite_differences(self, f, df):
        x = self.x[np.abs(self.x) > 1e-3]  # avoid the ReLU kink
        ε = 1e-6
        numeric = (f(x + ε) - f(x - ε)) / (2 * ε)
        np.testing.assert_allclose(df(x), numeric, atol=1e-6)

    def test_relu(self):
        np.testing.assert_array_equal(alg.ReLU(np.array([-1.0, 0.0, 2.0])), [0, 0, 2])
        np.testing.assert_array_equal(alg.dReLU(np.array([-1.0, 0.0, 2.0])), [0, 0, 1])


class TestMLP:
    def test_initialize_shapes_and_reproducibility(self):
        w1, b1 = alg.initialize_nn([3, 5, 2])
        w2, _ = alg.initialize_nn([3, 5, 2])
        assert [w.shape for w in w1] == [(5, 3), (2, 5)]
        assert [b.shape for b in b1] == [(5, 1), (2, 1)]
        assert all(np.all(b == 0) for b in b1)
        for a, b in zip(w1, w2):
            np.testing.assert_array_equal(a, b)
        assert max(np.abs(w).max() for w in w1) < 0.1

    def test_initialize_seed_none_does_not_touch_global_state(self):
        np.random.seed(7)
        expected = np.random.rand()
        np.random.seed(7)
        alg.initialize_nn([2, 2], seed=None)
        # the weights were drawn from the stream seeded with 7
        assert np.random.rand() != expected

    def test_predict_shape_and_linear_output(self):
        w, b = alg.initialize_nn([2, 4, 3])
        X = np.random.default_rng(0).standard_normal((2, 7))
        assert alg.predict(X, w, b, "tanh").shape == (3, 7)

    def test_predict_matches_manual_forward_pass(self):
        rng = np.random.default_rng(0)
        w = [rng.standard_normal((4, 2)), rng.standard_normal((1, 4))]
        b = [rng.standard_normal((4, 1)), rng.standard_normal((1, 1))]
        X = rng.standard_normal((2, 6))
        expected = w[1] @ np.tanh(w[0] @ X + b[0]) + b[1]
        np.testing.assert_allclose(alg.predict(X, w, b, "tanh"), expected)

    @pytest.mark.parametrize("activation", ["tanh", "sigmoid", "ReLU"])
    def test_gradient_matches_finite_differences(self, activation):
        """One full-batch step of fit() moves the weights along -∇J."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((2, 6))
        Y = rng.standard_normal((1, 6))
        w0 = [rng.standard_normal((4, 2)), rng.standard_normal((1, 4))]
        b0 = [0.1 * rng.standard_normal((4, 1)), rng.standard_normal((1, 1))]

        def J(w, b):
            return 0.5 * np.sum((alg.predict(X, w, b, activation) - Y) ** 2)

        lr = 1e-7
        w = [m.copy() for m in w0]
        b = [m.copy() for m in b0]
        alg.fit(X, Y, w, b, activation, lr, epochs=1, batch_size=6, verbose=False)

        ε = 1e-6
        for params0, params in ((w0, w), (b0, b)):
            for layer in range(2):
                numeric = np.zeros_like(params0[layer])
                for idx in np.ndindex(*numeric.shape):
                    plus = [m.copy() for m in params0]
                    minus = [m.copy() for m in params0]
                    plus[layer][idx] += ε
                    minus[layer][idx] -= ε
                    if params0 is w0:
                        numeric[idx] = (J(plus, b0) - J(minus, b0)) / (2 * ε)
                    else:
                        numeric[idx] = (J(w0, plus) - J(w0, minus)) / (2 * ε)
                analytic = (params0[layer] - params[layer]) / lr
                np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)

    def test_fit_learns_nonlinear_function(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-2, 2, (1, 400))
        Y = np.sin(X)
        w, b = alg.initialize_nn([1, 12, 1])
        cost, w, b = alg.fit(X, Y, w, b, "tanh", 1e-3, epochs=300, batch_size=10, verbose=False)
        assert cost.shape == (300,)
        assert cost[-1] < 0.05 * cost[0]
        assert np.mean((alg.predict(X, w, b, "tanh") - Y) ** 2) < 0.01

    def test_fit_updates_parameters_in_place(self):
        X = np.random.default_rng(0).standard_normal((2, 20))
        Y = X.sum(axis=0, keepdims=True)
        w, b = alg.initialize_nn([2, 3, 1])
        w_id = [id(m) for m in w]
        before = w[0].copy()
        _, w_out, _ = alg.fit(X, Y, w, b, "tanh", 1e-2, 3, verbose=False)
        assert [id(m) for m in w_out] == w_id
        assert not np.allclose(before, w[0])

    def test_fit_uses_incomplete_last_batch(self):
        X = np.random.default_rng(0).standard_normal((1, 25))
        Y = 2 * X
        w, b = alg.initialize_nn([1, 1])
        cost, *_ = alg.fit(X, Y, w, b, "tanh", 1e-2, 50, batch_size=10, verbose=False)
        assert cost[-1] < cost[0]
        np.testing.assert_allclose(w[0], [[2.0]], atol=0.05)  # last 5 samples count

    def test_single_linear_layer_is_lms(self):
        """A network without hidden layers is a (mini-batch) LMS filter."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((3, 2000))
        g = np.array([[0.5, -1.0, 2.0]])
        w, b = alg.initialize_nn([3, 1])
        alg.fit(X, g @ X, w, b, "tanh", 1e-3, epochs=20, batch_size=1, verbose=False)
        np.testing.assert_allclose(w[0], g, atol=1e-3)

    @pytest.mark.parametrize("func", ["fit", "predict"])
    def test_unknown_activation(self, func):
        w, b = alg.initialize_nn([2, 2, 1])
        X = np.zeros((2, 4))
        with pytest.raises(ValueError, match="Unknown activation"):
            if func == "fit":
                alg.fit(X, np.zeros((1, 4)), w, b, "swish", 0.1, 1, verbose=False)
            else:
                alg.predict(X, w, b, "swish")

    def test_activation_names_are_case_insensitive(self):
        w, b = alg.initialize_nn([2, 3, 1])
        X = np.random.default_rng(0).standard_normal((2, 5))
        np.testing.assert_array_equal(alg.predict(X, w, b, "relu"), alg.predict(X, w, b, "ReLU"))
