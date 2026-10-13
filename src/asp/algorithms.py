r"""
Adaptive signal processing algorithms.

This module collects the algorithms studied in the course, from the classical
optimization methods that motivate adaptive filtering to the Wiener filter,
the LMS family, RLS, Kalman filters, blind equalization (CMA) and a small
multi-layer perceptron. Every function is accompanied by the theory it is
based on, in the *Notes* section of its docstring.

Overview
--------
==================================  ===============================================
Topic                               Functions
==================================  ===============================================
Statistical estimation              :func:`estimate_correlation_matrix`,
                                    :func:`estimate_cross_correlation`
Objective-function optimization     :func:`gradient_descent`, :func:`newton_method`
Gradient adaptation (FIR)           :func:`gradient_adaptation`
LMS family                          :func:`lms`, :func:`nlms`, :func:`lms_newton`,
                                    :func:`tdlms`, :func:`fdlms`
RLS family                          :func:`rls`, :func:`rls_apriori`,
                                    :func:`rls_aposteriori`
Kalman filtering                    :func:`kalman_filter`,
                                    :func:`extended_kalman_filter`
Blind adaptation                    :func:`cma`
Utilities                           :func:`time_varying_filter`,
                                    :func:`sliding_window_matrix`
Neural networks (MLP)               :func:`initialize_nn`, :func:`fit`,
                                    :func:`predict`, activation functions
==================================  ===============================================

Conventions
-----------

* ``x`` is the filter input, ``d`` the desired (reference) signal, ``Ntaps``
  the number of filter coefficients and the regressor vector at time ``k`` is
  :math:`\mathbf{x}_k = [x[k], x[k-1], \dots, x[k-N+1]]^T`.
* The adaptive filter output is :math:`y[k] = \mathbf{h}_k^T \mathbf{x}_k` and
  the error is :math:`e[k] = d[k] - y[k]`.
* Adaptive filters start adapting at sample ``k = Ntaps`` (the first
  ``Ntaps`` samples are used to fill the delay line); output, error and
  coefficient histories are zero before that instant.
* The coefficient history ``H`` returned by the adaptive filters has shape
  ``(len(x), Ntaps)`` and ``H[k]`` holds the coefficients **after** the update
  performed at sample ``k``, so that ``H`` is directly comparable with a
  time-varying system whose coefficients are stored with the same layout (see
  :func:`time_varying_filter`).
* The algorithms operate on real-valued signals, except :func:`cma`, which
  also accepts complex baseband signals.
"""

import numpy as np
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

__all__ = [
    "estimate_correlation_matrix",
    "estimate_cross_correlation",
    "gradient_descent",
    "newton_method",
    "gradient_adaptation",
    "lms",
    "nlms",
    "lms_newton",
    "tdlms",
    "fdlms",
    "rls",
    "rls_apriori",
    "rls_aposteriori",
    "kalman_filter",
    "extended_kalman_filter",
    "cma",
    "time_varying_filter",
    "sliding_window_matrix",
    "sigmoid",
    "dsigmoid",
    "ReLU",
    "dReLU",
    "dtanh",
    "initialize_nn",
    "fit",
    "predict",
]


# ---------------------------------------------------------------------------
# Statistical estimation
# ---------------------------------------------------------------------------
def estimate_correlation_matrix(x, N):
    r"""
    Estimate the correlation matrix :math:`\mathbf{R}` of order ``N`` from data.

    Parameters
    ----------
    x : array-like
        Sequence of samples :math:`x[0], \dots, x[M-1]` (real-valued).
    N : int
        Order of the correlation matrix. Must satisfy ``1 <= N <= M``.

    Returns
    -------
    R : numpy.ndarray
        Estimated correlation matrix, shape ``(N, N)``, symmetric and positive
        semi-definite.

    Raises
    ------
    ValueError
        If ``N`` is smaller than 1 or larger than the length of ``x``.

    Notes
    -----
    For a wide-sense stationary process the correlation matrix of the
    regressor :math:`\mathbf{x}_k = [x[k], \dots, x[k-N+1]]^T` is the Toeplitz
    matrix

    .. math::

        \mathbf{R} = E\left[\mathbf{x}_k \mathbf{x}_k^T\right], \qquad
        [\mathbf{R}]_{ij} = r_x[i-j] = E\left[x[k-i]\,x[k-j]\right].

    It is estimated by the sample average over all :math:`M-N+1` complete
    regressors that can be formed from the data,

    .. math::

        \hat{\mathbf{R}} = \frac{1}{M-N+1}\sum_{k=N-1}^{M-1}
        \mathbf{x}_k \mathbf{x}_k^T ,

    which is an unbiased estimator of :math:`\mathbf{R}` because each term has
    expected value :math:`\mathbf{R}`. :math:`\mathbf{R}` is the matrix that
    appears in the Wiener-Hopf equations (see :func:`lms`).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(50_000)
    >>> R = estimate_correlation_matrix(x, 3)
    >>> bool(np.allclose(R, np.eye(3), atol=0.03))
    True
    """
    x = np.asarray(x, dtype=float)
    M = len(x)
    if N < 1 or N > M:
        raise ValueError(
            "Order N should be between 1 and the length of the sequence M."
        )

    # Row i holds x[k-i] for k = N-1, ..., M-1
    X = sliding_window_view(x, N)[:, ::-1].T

    return (X @ X.T) / (M - N + 1)


def estimate_cross_correlation(x, d, N):
    r"""
    Estimate the cross-correlation vector :math:`\mathbf{p}` between the
    regressor of ``x`` and the desired signal ``d``.

    Parameters
    ----------
    x : array-like
        Input sequence of length ``M`` (real-valued).
    d : array-like
        Desired sequence, also of length ``M``.
    N : int
        Length of the regressor vector. Must satisfy ``1 <= N <= M``.

    Returns
    -------
    p : numpy.ndarray
        Estimated cross-correlation vector, shape ``(N,)``;
        ``p[i]`` estimates :math:`E[d[k]\,x[k-i]]`.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` have different lengths or ``N`` is out of range.

    Notes
    -----
    The vector :math:`\mathbf{p}` of the Wiener-Hopf equations is

    .. math::

        \mathbf{p} = E\left[d[k]\,\mathbf{x}_k\right],

    and it is estimated by averaging over all :math:`M-N+1` complete
    regressors,

    .. math::

        \hat{\mathbf{p}} = \frac{1}{M-N+1}\sum_{k=N-1}^{M-1} d[k]\,\mathbf{x}_k .

    Together with :func:`estimate_correlation_matrix` it gives the
    sample-based Wiener solution :math:`\mathbf{h}_o = \hat{\mathbf{R}}^{-1}
    \hat{\mathbf{p}}`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(20_000)
    >>> d = np.convolve(x, [1.0, 0.5])[: len(x)]
    >>> estimate_cross_correlation(x, d, 2).round(1)
    array([1. , 0.5])
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    M = len(x)
    if len(d) != M:
        raise ValueError("The sequences x and d must have the same length.")
    if N < 1 or N > M:
        raise ValueError(
            "N should be between 1 and the length of the sequences."
        )

    X = sliding_window_view(x, N)[:, ::-1]  # row k: [x[k], ..., x[k-N+1]]

    return (X * d[N - 1 :, None]).mean(axis=0)


# ---------------------------------------------------------------------------
# Optimization of objective functions
# ---------------------------------------------------------------------------
def gradient_descent(grad, x0, μ, max_iter=1000, tol=1e-6):
    r"""
    Minimize a differentiable function with the gradient descent method.

    Parameters
    ----------
    grad : callable
        Gradient of the objective function. Takes an array of shape ``(n,)``
        and returns an array of the same shape.
    x0 : array-like
        Initial guess, shape ``(n,)``.
    μ : float
        Step size.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.
    tol : float, optional
        The iterations stop when :math:`\|\Delta\mathbf{x}\|_1 <` ``tol``.

    Returns
    -------
    x : numpy.ndarray
        Last iterate, shape ``(n,)``.
    path : numpy.ndarray
        All iterates, including ``x0``, shape ``(n_iter + 1, n)``.

    Notes
    -----
    Gradient descent (steepest descent) updates the estimate in the direction
    opposite to the gradient of the objective function :math:`J(\mathbf{x})`,

    .. math::

        \mathbf{x}_{k+1} = \mathbf{x}_k - \mu\,\nabla J(\mathbf{x}_k).

    For a quadratic cost :math:`J = \tfrac12\mathbf{x}^T\mathbf{A}\mathbf{x}
    - \mathbf{b}^T\mathbf{x}` the method converges to
    :math:`\mathbf{A}^{-1}\mathbf{b}` if and only if
    :math:`0 < \mu < 2/\lambda_{\max}(\mathbf{A})`, and the convergence speed
    is governed by the eigenvalue spread of :math:`\mathbf{A}`. This is the
    deterministic counterpart of the LMS algorithm, obtained when the exact
    gradient is replaced by a noisy instantaneous estimate.

    Examples
    --------
    >>> import numpy as np
    >>> x, _ = gradient_descent(lambda v: 2 * v, [3.0, -2.0], μ=0.25)
    >>> bool(np.allclose(x, 0, atol=1e-4))
    True
    """
    xk = np.array(x0, dtype=float)
    path = [xk.copy()]
    for _ in range(max_iter):
        Δx = -μ * np.asarray(grad(xk), dtype=float)
        xk = xk + Δx
        path.append(xk.copy())
        if np.sum(np.abs(Δx)) < tol:
            break

    return xk, np.array(path)


def newton_method(grad, hess, x0, μ=1.0, max_iter=100, tol=1e-6):
    r"""
    Minimize a twice-differentiable function with Newton's method.

    Parameters
    ----------
    grad : callable
        Gradient of the objective function, ``(n,) -> (n,)``.
    hess : callable
        Hessian of the objective function, ``(n,) -> (n, n)``.
    x0 : array-like
        Initial guess, shape ``(n,)``.
    μ : float, optional
        Step size (damping factor). ``μ = 1`` is the pure Newton method.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.
    tol : float, optional
        The iterations stop when :math:`\|\Delta\mathbf{x}\|_1 <` ``tol``.

    Returns
    -------
    x : numpy.ndarray
        Last iterate, shape ``(n,)``.
    path : numpy.ndarray
        All iterates, including ``x0``, shape ``(n_iter + 1, n)``.

    Notes
    -----
    Newton's method uses the second-order Taylor expansion of
    :math:`J(\mathbf{x})` around the current point,

    .. math::

        \mathbf{x}_{k+1} = \mathbf{x}_k - \mu\,
        \mathbf{H}^{-1}(\mathbf{x}_k)\,\nabla J(\mathbf{x}_k),

    where :math:`\mathbf{H}` is the Hessian. For a quadratic cost a single
    step with :math:`\mu=1` reaches the minimum regardless of the eigenvalue
    spread, which is the motivation for the LMS-Newton (:func:`lms_newton`)
    and RLS (:func:`rls`) algorithms: they approximate :math:`\mathbf{H}^{-1}`
    (equal to :math:`\mathbf{R}^{-1}` for the mean-square error cost) from
    the data.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 1.0]])
    >>> b = np.array([1.0, 2.0])
    >>> x, path = newton_method(lambda v: A @ v - b, lambda v: A, [0.0, 0.0])
    >>> bool(np.allclose(x, np.linalg.solve(A, b)))
    True
    """
    xk = np.array(x0, dtype=float)
    path = [xk.copy()]
    for _ in range(max_iter):
        Δx = -μ * np.linalg.solve(np.atleast_2d(hess(xk)), np.asarray(grad(xk), dtype=float))
        xk = xk + Δx
        path.append(xk.copy())
        if np.sum(np.abs(Δx)) < tol:
            break

    return xk, np.array(path)


@njit
def gradient_adaptation(x, y, Ntaps=5, μ=1e-3, epochs=2):
    r"""
    Adapt an FIR filter by repeated stochastic-gradient passes over a
    recorded signal.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    y : numpy.ndarray
        Reference signal, shape ``(N,)``.
    Ntaps : int, optional
        Number of filter coefficients. Default is 5.
    μ : float, optional
        Step size. Default is ``1e-3``.
    epochs : int, optional
        Number of passes over the whole signal. Default is 2.

    Returns
    -------
    out : numpy.ndarray
        Filter output during the last epoch, shape ``(N,)``.
    H : numpy.ndarray
        History of the coefficients, shape ``(epochs * N + 1, Ntaps)``;
        ``H[0]`` is the initial (all-zero) filter.
    errors : numpy.ndarray
        Error :math:`e = y - \hat{y}` at every step, shape ``(epochs * N,)``.

    Notes
    -----
    The input is zero-padded with ``Ntaps - 1`` samples so that every sample of
    ``y`` is used. At each step the coefficients follow the instantaneous
    gradient of the squared error,

    .. math::

        e[k] = y[k] - \mathbf{h}_k^T\mathbf{x}_k, \qquad
        \mathbf{h}_{k+1} = \mathbf{h}_k + \mu\, e[k]\,\mathbf{x}_k .

    This is the LMS algorithm (:func:`lms`) in the parametrization used in
    the introductory notebook, i.e. with the factor 2 absorbed in
    :math:`\mu`, and with the option of replaying the same data several times.
    """
    h = np.zeros(Ntaps)
    N = len(y)

    errors = np.zeros(epochs * N)
    out = np.zeros(N)
    H = np.zeros((epochs * N + 1, Ntaps))

    xp = np.concatenate((np.zeros(Ntaps - 1), x))

    for epoch in range(epochs):
        for i in range(N):
            x_vec = xp[i : i + Ntaps][::-1]  # [x[i], x[i-1], ..., x[i-Ntaps+1]]

            y_hat = np.sum(h * x_vec)
            error = y[i] - y_hat
            h += μ * error * x_vec

            H[i + epoch * N + 1, :] = h
            errors[i + epoch * N] = error
            out[i] = y_hat

    return out, H, errors


# ---------------------------------------------------------------------------
# LMS family
# ---------------------------------------------------------------------------
@njit
def _check_inputs(x, d, Ntaps):
    if Ntaps < 1:
        raise ValueError("Ntaps must be a positive integer.")
    if len(x) != len(d):
        raise ValueError("x and d must have the same length.")


@njit
def lms(x, d, Ntaps, μ):
    r"""
    Least Mean Squares (LMS) adaptive filter.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    d : numpy.ndarray
        Desired signal, shape ``(N,)``.
    Ntaps : int
        Number of filter coefficients.
    μ : float
        Step size.

    Returns
    -------
    out : numpy.ndarray
        Filter output :math:`y[k]`, shape ``(N,)``.
    h : numpy.ndarray
        Final coefficients, shape ``(Ntaps,)``.
    squaredError : numpy.ndarray
        Squared error :math:`e^2[k]`, shape ``(N,)``.
    H : numpy.ndarray
        Coefficient history, shape ``(N, Ntaps)`` (see the module notes).

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length or ``Ntaps < 1``.

    Notes
    -----
    **Wiener filter.** The Wiener filter minimizes the mean-square error
    :math:`J(\mathbf{h}) = E[e^2[k]]`, which is quadratic in :math:`\mathbf{h}`:

    .. math::

        J(\mathbf{h}) = \sigma_d^2 - 2\mathbf{h}^T\mathbf{p}
        + \mathbf{h}^T\mathbf{R}\mathbf{h}, \qquad
        \nabla J = -2\mathbf{p} + 2\mathbf{R}\mathbf{h}.

    Setting the gradient to zero gives the Wiener-Hopf equations
    :math:`\mathbf{R}\,\mathbf{h}_o = \mathbf{p}`.

    **Steepest descent.** Iterating
    :math:`\mathbf{h}_{k+1}=\mathbf{h}_k-\mu\nabla J` avoids the matrix
    inversion but still requires :math:`\mathbf{R}` and :math:`\mathbf{p}`.

    **LMS.** Widrow and Hoff replaced the expectations by their instantaneous
    values, :math:`\hat{\nabla}J = -2e[k]\,\mathbf{x}_k`, which gives

    .. math::

        y[k] = \mathbf{h}_k^T\mathbf{x}_k, \quad
        e[k] = d[k] - y[k], \quad
        \mathbf{h}_{k+1} = \mathbf{h}_k + 2\mu\, e[k]\,\mathbf{x}_k .

    The algorithm needs :math:`O(N)` operations per sample. On average it
    converges to :math:`\mathbf{h}_o` if
    :math:`0<\mu<1/\lambda_{\max}(\mathbf{R})` and, since
    :math:`\lambda_{\max}\le\operatorname{tr}\mathbf{R}=N\sigma_x^2`, a
    conservative practical bound is :math:`\mu < 1/(N\sigma_x^2)`. The mode
    associated with eigenvalue :math:`\lambda_i` decays as
    :math:`(1-2\mu\lambda_i)^k`, so a large eigenvalue spread
    :math:`\lambda_{\max}/\lambda_{\min}` slows convergence (the motivation
    for :func:`nlms`, :func:`lms_newton` and :func:`rls`). A larger
    :math:`\mu` converges faster but yields a larger steady-state excess MSE.

    References
    ----------
    * B. Widrow and M. E. Hoff, "Adaptive switching circuits", IRE
      WESCON Convention Record, 1960.
    * S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014, ch. 5-6.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> x = rng.standard_normal(5000)
    >>> g = np.array([1.0, 0.5, -0.3])
    >>> d = np.convolve(x, g)[: len(x)]
    >>> out, h, sqerr, H = lms(x, d, 3, 0.01)
    >>> bool(np.allclose(h, g, atol=1e-3))
    True
    """
    _check_inputs(x, d, Ntaps)

    h = np.zeros(Ntaps, dtype=np.float64)
    H = np.zeros((len(x), Ntaps), dtype=np.float64)
    ind = np.arange(0, Ntaps)

    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)

    for i in range(Ntaps, len(x)):
        x_vec = x[i - ind]

        # Filter output and error
        y = np.sum(x_vec * h)
        error = d[i] - y

        # LMS update rule
        h += 2 * μ * error * x_vec

        squaredError[i] = error**2
        out[i] = y
        H[i, :] = h

    return out, h, squaredError, H


@njit
def nlms(x, d, Ntaps, μ, γ=1e-6):
    r"""
    Normalized Least Mean Squares (NLMS) adaptive filter.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    d : numpy.ndarray
        Desired signal, shape ``(N,)``.
    Ntaps : int
        Number of filter coefficients.
    μ : float
        Step size, normally in ``(0, 2)``.
    γ : float, optional
        Regularization constant that avoids division by zero. Default ``1e-6``.

    Returns
    -------
    out : numpy.ndarray
        Filter output, shape ``(N,)``.
    h : numpy.ndarray
        Final coefficients, shape ``(Ntaps, 1)``.
    squaredError : numpy.ndarray
        Squared error, shape ``(N,)``.
    H : numpy.ndarray
        Coefficient history, shape ``(N, Ntaps)``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length or ``Ntaps < 1``.

    Notes
    -----
    The LMS step size that guarantees stability depends on the input power,
    which is unknown and may vary. NLMS removes this dependence by
    normalizing the update by the instantaneous energy of the regressor,

    .. math::

        \mathbf{h}_{k+1} = \mathbf{h}_k +
        \frac{\mu}{\gamma + \mathbf{x}_k^T\mathbf{x}_k}\,e[k]\,\mathbf{x}_k .

    It can be derived as the solution of a constrained problem: choose the
    smallest change :math:`\|\mathbf{h}_{k+1}-\mathbf{h}_k\|^2` such that the
    *a posteriori* error is zero, :math:`d[k]=\mathbf{h}_{k+1}^T\mathbf{x}_k`
    (the case :math:`\mu=1`). For :math:`0<\mu<2` the algorithm is stable
    for any input power, with a convergence rate that is far less sensitive
    to the input scaling than LMS.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> x = 10 * rng.standard_normal(2000)
    >>> g = np.array([1.0, -0.5])
    >>> d = np.convolve(x, g)[: len(x)]
    >>> out, h, sqerr, H = nlms(x, d, 2, 0.5)
    >>> bool(np.allclose(h.ravel(), g, atol=1e-4))
    True
    """
    _check_inputs(x, d, Ntaps)

    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x), Ntaps), dtype=np.float64)
    ind = np.arange(0, Ntaps)

    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        y = np.sum(x_vec * h)
        error = d[i] - y

        # NLMS update rule
        h += μ * error * x_vec / (γ + x_vec.T @ x_vec)

        squaredError[i] = error**2
        out[i] = y
        H[i, :] = h.T

    return out, h, squaredError, H


@njit
def lms_newton(x, d, Ntaps, μ, α):
    r"""
    LMS-Newton adaptive filter.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    d : numpy.ndarray
        Desired signal, shape ``(N,)``.
    Ntaps : int
        Number of filter coefficients.
    μ : float
        Step size.
    α : float
        Smoothing factor of the correlation matrix estimate, ``0 < α < 1``
        (small values average over many samples).

    Returns
    -------
    out : numpy.ndarray
        Filter output, shape ``(N,)``.
    h : numpy.ndarray
        Final coefficients, shape ``(Ntaps, 1)``.
    squaredError : numpy.ndarray
        Squared error, shape ``(N,)``.
    H : numpy.ndarray
        Coefficient history, shape ``(N, Ntaps)``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length or ``Ntaps < 1``.

    Notes
    -----
    Newton's method (see :func:`newton_method`) applied to the MSE cost
    multiplies the gradient by :math:`\mathbf{R}^{-1}`, which makes the
    convergence independent of the eigenvalue spread. LMS-Newton keeps the
    stochastic gradient of LMS but replaces :math:`\mathbf{R}` by a recursive
    estimate

    .. math::

        \hat{\mathbf{R}}_k = (1-\alpha)\hat{\mathbf{R}}_{k-1}
        + \alpha\,\mathbf{x}_k\mathbf{x}_k^T ,

    whose inverse is propagated directly with the matrix inversion lemma
    (Sherman-Morrison formula), avoiding an :math:`O(N^3)` inversion:

    .. math::

        \hat{\mathbf{R}}_k^{-1} = \frac{1}{1-\alpha}\left[
        \hat{\mathbf{R}}_{k-1}^{-1} -
        \frac{\hat{\mathbf{R}}_{k-1}^{-1}\mathbf{x}_k\mathbf{x}_k^T
        \hat{\mathbf{R}}_{k-1}^{-1}}
        {\frac{1-\alpha}{\alpha} + \mathbf{x}_k^T\hat{\mathbf{R}}_{k-1}^{-1}
        \mathbf{x}_k}\right],

    .. math::

        \mathbf{h}_{k+1} = \mathbf{h}_k + \mu\,e[k]\,
        \hat{\mathbf{R}}_k^{-1}\mathbf{x}_k .

    The cost is :math:`O(N^2)` per sample. The inverse matrix is initialized
    as :math:`10^{-3}\mathbf{I}`.

    References
    ----------
    * P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
      Implementation*, 5th ed., Springer, 2020, ch. 7.
    """
    _check_inputs(x, d, Ntaps)

    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x), Ntaps), dtype=np.float64)
    R_inv = 1e-3 * np.eye(Ntaps, dtype=np.float64)

    ind = np.arange(0, Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        y = np.sum(x_vec * h)
        error = d[i] - y

        # Update the inverse of the correlation matrix estimate
        R_inv = (
            1
            / (1 - α)
            * (
                R_inv
                - (R_inv @ (x_vec @ x_vec.T) @ R_inv)
                / ((1 - α) / α + x_vec.T @ R_inv @ x_vec)
            )
        )

        # LMS-Newton update rule
        h += μ * error * R_inv @ x_vec

        squaredError[i] = error**2
        out[i] = y
        H[i, :] = h.T

    return out, h, squaredError, H


@njit
def tdlms(x, d, Ntaps, μ, M, γ=1e-3, α=0.1):
    r"""
    Transform-domain LMS (TD-LMS) adaptive filter.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    d : numpy.ndarray
        Desired signal, shape ``(N,)``.
    Ntaps : int
        Number of filter coefficients.
    μ : float
        Step size.
    M : numpy.ndarray
        Transform matrix, shape ``(Ntaps, Ntaps)``, normally orthogonal
        (e.g. the DCT matrix).
    γ : float, optional
        Constant that prevents large steps when the power of a transformed
        component is small. Default ``1e-3``.
    α : float, optional
        Smoothing factor of the power estimates, ``0 < α <= 1``. Default 0.1.

    Returns
    -------
    out : numpy.ndarray
        Filter output, shape ``(N,)``.
    h : numpy.ndarray
        Final coefficients **in the time domain** (``M.T @ h_transform``),
        shape ``(Ntaps,)``.
    squaredError : numpy.ndarray
        Squared error, shape ``(N,)``.
    H : numpy.ndarray
        Time-domain coefficient history, shape ``(N, Ntaps)``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length or ``Ntaps < 1``.

    Notes
    -----
    A strongly correlated input (large eigenvalue spread of
    :math:`\mathbf{R}`) slows down LMS. TD-LMS pre-processes the regressor
    with a transform :math:`\mathbf{M}` that approximately decorrelates it,
    :math:`\mathbf{s}_k=\mathbf{M}\mathbf{x}_k`, so that the correlation
    matrix of :math:`\mathbf{s}_k` is nearly diagonal, and then equalizes the
    power of the components with a per-component step size:

    .. math::

        y[k] = \mathbf{w}_k^T\mathbf{s}_k, \qquad
        p_i[k] = \alpha\,s_i^2[k] + (1-\alpha)\,p_i[k-1],

    .. math::

        \mathbf{w}_{k+1} = \mathbf{w}_k + \mu\,e[k]\,
        \operatorname{diag}\!\left(\frac{1}{p_i[k]+\gamma}\right)\mathbf{s}_k .

    This is the LMS-Newton algorithm with the inverse correlation matrix
    approximated by the inverse of its diagonal, an :math:`O(N)` operation
    (plus the cost of the transform). The equivalent time-domain filter is
    :math:`\mathbf{h}=\mathbf{M}^T\mathbf{w}` because
    :math:`y=\mathbf{w}^T\mathbf{M}\mathbf{x}`.
    """
    _check_inputs(x, d, Ntaps)

    w = np.zeros(Ntaps, dtype=np.float64)
    H = np.zeros((len(x), Ntaps), dtype=np.float64)
    ind = np.arange(0, Ntaps)
    Mc = np.ascontiguousarray(M).astype(np.float64)
    MT = np.ascontiguousarray(Mc.T)

    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    power = np.ones(Ntaps, dtype=np.float64)

    for i in range(Ntaps, len(x)):
        x_vec = x[i - ind].astype(np.float64)

        # Transform the regressor
        s_vec = Mc @ x_vec

        y = np.sum(s_vec * w)
        error = d[i] - y

        # Power estimate of each transformed component
        power = α * s_vec**2 + (1 - α) * power

        # TD-LMS update rule (power-normalized)
        w += μ * error * s_vec / (power + γ)

        squaredError[i] = error**2
        out[i] = y
        H[i, :] = MT @ w

    return out, MT @ w, squaredError, H


def fdlms(x, d, Ntaps, μ, Nfft=32, constrained=False):
    r"""
    Frequency-domain LMS (FD-LMS) with the overlap-and-save method.

    Parameters
    ----------
    x : array-like
        Input signal, shape ``(N,)``.
    d : array-like
        Desired signal, shape ``(N,)``.
    Ntaps : int
        Number of filter coefficients.
    μ : float
        Step size.
    Nfft : int, optional
        FFT size; rounded up to the next power of two and required to be at
        least ``Ntaps``. Each block processes ``Nfft - Ntaps + 1`` new samples.
        Default is 32.
    constrained : bool, optional
        If ``True``, the time-domain coefficients beyond ``Ntaps`` are forced
        to zero after every update (*constrained* FD-LMS). Default ``False``
        (unconstrained version).

    Returns
    -------
    out : numpy.ndarray
        Filter output, shape ``(N,)`` (real if ``x`` and ``d`` are real).
    h : numpy.ndarray
        Final time-domain coefficients, shape ``(Ntaps,)``.
    squaredError : numpy.ndarray
        Mean squared error of every block, shape ``(n_blocks,)``.
    H : numpy.ndarray
        Final frequency-domain coefficients, shape ``(Nfft,)``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length, ``Ntaps < 1`` or ``Nfft < Ntaps``.

    Notes
    -----
    Filtering and the correlation :math:`\sum_n e[n]\,x[n-m]` that forms the
    LMS gradient are both convolution-type operations, which become
    products in the frequency domain. Using blocks of ``Nfft`` input samples
    :math:`\mathbf{X}=\operatorname{FFT}(\mathbf{x}_{\text{block}})` and the
    frequency response :math:`\mathbf{H}` of the filter, the *overlap-and-save*
    method discards the first ``Ntaps - 1`` samples of the circular
    convolution (they are corrupted by wrap-around):

    .. math::

        \mathbf{y} = \operatorname{IFFT}(\mathbf{X}\odot\mathbf{H}),

    .. math::

        \mathbf{e} = [\mathbf{0}_{N_{taps}-1},\, \mathbf{d}-\mathbf{y}_{\text{valid}}],
        \qquad
        \mathbf{H} \leftarrow \mathbf{H} + \frac{\mu}{\sqrt{N_{FFT}}}\,
        \operatorname{FFT}(\mathbf{e})\odot\mathbf{X}^{*} .

    The coefficients are updated once per block, and the cost per sample drops
    from :math:`O(N)` to :math:`O(\log N)`, which is attractive for long
    filters. With the update above the effective time-domain step is
    :math:`\mu/\sqrt{N_{FFT}}`.

    References
    ----------
    * J. J. Shynk, "Frequency-domain and multirate adaptive filtering",
      IEEE Signal Processing Magazine, 9(1), 1992.
    """
    x = np.asarray(x)
    d = np.asarray(d)
    if Ntaps < 1:
        raise ValueError("Ntaps must be a positive integer.")
    if len(x) != len(d):
        raise ValueError("x and d must have the same length.")

    Nfft = int(2 ** np.ceil(np.log2(Nfft)))
    if Nfft < Ntaps:
        raise ValueError("Nfft must be greater than or equal to Ntaps.")

    isreal = np.isrealobj(x) and np.isrealobj(d)
    L = len(x)

    # Overlap-and-save parameters
    b = Nfft - Ntaps + 1  # new samples per block
    nblocks = int(np.ceil(L / b))

    # Zero-pad: Ntaps-1 samples for the first overlap, plus the last block
    x = np.concatenate(
        (np.zeros(Ntaps - 1), x.astype(np.complex128), np.zeros(nblocks * b - L))
    )
    d = np.concatenate((d.astype(np.complex128), np.zeros(nblocks * b - L)))

    y = np.zeros(b * nblocks, dtype=np.complex128)
    H = np.zeros(Nfft, dtype=np.complex128)
    error = np.zeros(Nfft, dtype=np.complex128)
    squaredError = np.zeros(nblocks)

    for i in range(nblocks):
        indSave = slice(i * b, (i + 1) * b)
        X = np.fft.fft(x[i * b : i * b + Nfft])

        # Filter in the frequency domain, discard the wrapped-around samples
        y[indSave] = np.fft.ifft(X * H)[Ntaps - 1 :]

        # Error is only defined on the valid samples; the zero-padding that
        # completes the last block is not part of the signal and is ignored
        n_valid = min(b, L - i * b)
        error[Ntaps - 1 :] = 0
        error[Ntaps - 1 : Ntaps - 1 + n_valid] = (
            d[i * b : i * b + n_valid] - y[i * b : i * b + n_valid]
        )

        # Frequency-domain LMS update
        H += μ * np.fft.fft(error) * X.conj() / np.sqrt(Nfft)

        if constrained:
            h_time = np.fft.ifft(H)
            h_time[Ntaps:] = 0
            H = np.fft.fft(h_time)

        squaredError[i] = np.mean(np.abs(error[Ntaps - 1 : Ntaps - 1 + n_valid]) ** 2)

    h = np.fft.ifft(H)[:Ntaps]
    y = y[:L]

    if isreal:
        y, h = y.real, h.real

    return y, h, squaredError, H


# ---------------------------------------------------------------------------
# RLS family
# ---------------------------------------------------------------------------
@njit
def rls(x, d, Ntaps, λ, δ=1e-2):
    r"""
    Recursive Least Squares (RLS) adaptive filter (conventional form).

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    d : numpy.ndarray
        Desired signal, shape ``(N,)``.
    Ntaps : int
        Number of filter coefficients.
    λ : float
        Forgetting factor, ``0 < λ <= 1``. Smaller values weight recent data
        more, allowing faster tracking of non-stationary systems.

    δ : float, optional
        Regularization of the initialization, :math:`\mathbf{P}_0=\delta^{-1}\mathbf{I}`
        (see Notes). Default ``1e-2``.

    Returns
    -------
    out : numpy.ndarray
        Filter output (computed with the *updated* coefficients), shape
        ``(N,)``.
    h : numpy.ndarray
        Final coefficients, shape ``(Ntaps, 1)``.
    squaredError : numpy.ndarray
        Squared a posteriori error, shape ``(N,)``.
    H : numpy.ndarray
        Coefficient history, shape ``(N, Ntaps)``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length or ``Ntaps < 1``.

    Notes
    -----
    RLS minimizes the exponentially weighted least-squares cost

    .. math::

        J_k(\mathbf{h}) = \sum_{i=1}^{k}\lambda^{k-i}\,
        \left(d[i]-\mathbf{h}^T\mathbf{x}_i\right)^2 ,

    whose solution satisfies the normal equations
    :math:`\hat{\mathbf{R}}_k\mathbf{h}_k=\hat{\mathbf{p}}_k` with

    .. math::

        \hat{\mathbf{R}}_k=\lambda\hat{\mathbf{R}}_{k-1}+\mathbf{x}_k
        \mathbf{x}_k^T, \qquad
        \hat{\mathbf{p}}_k=\lambda\hat{\mathbf{p}}_{k-1}+d[k]\mathbf{x}_k .

    Instead of solving the equations at each step, the inverse
    :math:`\mathbf{P}_k=\hat{\mathbf{R}}_k^{-1}` is updated recursively with
    the matrix inversion lemma,

    .. math::

        \mathbf{P}_k=\frac{1}{\lambda}\left[\mathbf{P}_{k-1}-
        \frac{\mathbf{P}_{k-1}\mathbf{x}_k\mathbf{x}_k^T\mathbf{P}_{k-1}}
        {\lambda+\mathbf{x}_k^T\mathbf{P}_{k-1}\mathbf{x}_k}\right],
        \qquad
        \mathbf{h}_k=\mathbf{P}_k\hat{\mathbf{p}}_k .

    The cost is :math:`O(N^2)` per sample. Convergence takes about :math:`2N`
    samples and is independent of the eigenvalue spread of the input; the
    price is the higher complexity and a sensitivity to numerical errors
    when :math:`\lambda<1`. The effective memory is :math:`1/(1-\lambda)`
    samples.

    The recursion is initialized with :math:`\mathbf{P}_0=\delta^{-1}
    \mathbf{I}` (so :math:`\hat{\mathbf{R}}_0=\delta\mathbf{I}`) and
    :math:`\mathbf{h}_0=\mathbf{0}`. This is a ridge regularization whose
    weight :math:`\delta\lambda^k` fades with time; it should be small
    compared with the input power, otherwise (and when :math:`\lambda=1`)
    it keeps shrinking the solution towards zero.

    References
    ----------
    * S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014, ch. 10.
    """
    _check_inputs(x, d, Ntaps)

    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x), Ntaps), dtype=np.float64)
    Rxx_inv = (1 / δ) * np.eye(Ntaps, dtype=np.float64)
    pxd = np.zeros((Ntaps, 1), dtype=np.float64)

    ind = np.arange(0, Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        # Update the inverse of the correlation matrix
        Rxx_inv = (
            1
            / λ
            * (
                Rxx_inv
                - (Rxx_inv @ (x_vec @ x_vec.T) @ Rxx_inv)
                / (λ + x_vec.T @ Rxx_inv @ x_vec)
            )
        )

        # Update the cross-correlation vector
        pxd = x_vec * d[i] + λ * pxd

        # Normal equations solution
        h = Rxx_inv @ pxd

        y = np.sum(x_vec * h)

        # A posteriori error
        error = d[i] - y

        squaredError[i] = error**2
        out[i] = y
        H[i, :] = h.T

    return out, h, squaredError, H


@njit
def rls_apriori(x, d, Ntaps, λ, δ=1e-2):
    r"""
    RLS adaptive filter, *a priori* error formulation (gain-vector form).

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    d : numpy.ndarray
        Desired signal, shape ``(N,)``.
    Ntaps : int
        Number of filter coefficients.
    λ : float
        Forgetting factor, ``0 < λ <= 1``.

    δ : float, optional
        Regularization of the initialization, :math:`\mathbf{P}_0=\delta^{-1}\mathbf{I}`
        (see Notes). Default ``1e-2``.

    Returns
    -------
    out : numpy.ndarray
        Filter output computed with the coefficients *before* the update,
        shape ``(N,)``.
    h : numpy.ndarray
        Final coefficients, shape ``(Ntaps, 1)``.
    squaredError : numpy.ndarray
        Squared a priori error, shape ``(N,)``.
    H : numpy.ndarray
        Coefficient history, shape ``(N, Ntaps)``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length or ``Ntaps < 1``.

    Notes
    -----
    Writing the RLS solution in terms of the *gain vector* (a Kalman-gain-like
    quantity) and of the a priori error
    :math:`\epsilon[k]=d[k]-\mathbf{h}_{k-1}^T\mathbf{x}_k`:

    .. math::

        \mathbf{g}_k=\frac{\lambda^{-1}\mathbf{P}_{k-1}\mathbf{x}_k}
        {1+\lambda^{-1}\mathbf{x}_k^T\mathbf{P}_{k-1}\mathbf{x}_k},
        \qquad
        \mathbf{h}_k=\mathbf{h}_{k-1}+\mathbf{g}_k\,\epsilon[k],

    .. math::

        \mathbf{P}_k=\lambda^{-1}\mathbf{P}_{k-1}-\lambda^{-1}\mathbf{g}_k
        \mathbf{x}_k^T\mathbf{P}_{k-1}.

    It is algebraically equivalent to :func:`rls` (the same coefficients up
    to the initialization of :math:`\mathbf{P}`), but it follows the
    prediction-correction structure of the Kalman filter: the correction is
    the gain times the prediction error.
    """
    _check_inputs(x, d, Ntaps)

    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x), Ntaps), dtype=np.float64)
    P = (1 / δ) * np.eye(Ntaps, dtype=np.float64)

    ind = np.arange(0, Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        # Gain vector
        g = (1 / λ) * P @ x_vec / (1 + (1 / λ) * x_vec.T @ P @ x_vec)

        y = np.sum(x_vec * h)

        # A priori error
        error = d[i] - y

        h += g * error

        # Update the inverse correlation matrix
        P = (1 / λ) * P - (1 / λ) * g @ x_vec.T @ P

        squaredError[i] = error**2
        out[i] = y
        H[i, :] = h.T

    return out, h, squaredError, H


@njit
def rls_aposteriori(x, d, Ntaps, λ, δ=1e-2):
    r"""
    RLS adaptive filter, *a posteriori* error formulation.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    d : numpy.ndarray
        Desired signal, shape ``(N,)``.
    Ntaps : int
        Number of filter coefficients.
    λ : float
        Forgetting factor, ``0 < λ <= 1``.

    δ : float, optional
        Regularization of the initialization, :math:`\mathbf{P}_0=\delta^{-1}\mathbf{I}`
        (see Notes). Default ``1e-2``.

    Returns
    -------
    out : numpy.ndarray
        Filter output computed with the coefficients *before* the update,
        shape ``(N,)``.
    h : numpy.ndarray
        Final coefficients, shape ``(Ntaps, 1)``.
    squaredError : numpy.ndarray
        Squared a priori error, shape ``(N,)``.
    H : numpy.ndarray
        Coefficient history, shape ``(N, Ntaps)``.

    Raises
    ------
    ValueError
        If ``x`` and ``d`` differ in length or ``Ntaps < 1``.

    Notes
    -----
    The a priori error :math:`\epsilon[k]` and the a posteriori error
    :math:`e[k]=d[k]-\mathbf{h}_k^T\mathbf{x}_k` are related by the
    *conversion factor*

    .. math::

        e[k]=\frac{\epsilon[k]}{\alpha_k}, \qquad
        \alpha_k = 1+\lambda^{-1}\mathbf{x}_k^T\mathbf{P}_{k-1}\mathbf{x}_k
        \ \ (\ge 1),

    so the update can be driven by the a posteriori error:

    .. math::

        \mathbf{g}_k=\lambda^{-1}\mathbf{P}_{k-1}\mathbf{x}_k, \quad
        \mathbf{h}_k=\mathbf{h}_{k-1}+\mathbf{g}_k\,\frac{\epsilon[k]}{\alpha_k},
        \quad
        \mathbf{P}_k=\lambda^{-1}\mathbf{P}_{k-1}-\lambda^{-1}
        \frac{\mathbf{g}_k}{\alpha_k}\mathbf{x}_k^T\mathbf{P}_{k-1}.

    It produces the same coefficients as :func:`rls_apriori`; only the
    intermediate quantities (and the way the gain is normalized) differ.
    """
    _check_inputs(x, d, Ntaps)

    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x), Ntaps), dtype=np.float64)
    P = (1 / δ) * np.eye(Ntaps, dtype=np.float64)

    ind = np.arange(0, Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        # Unnormalized gain vector
        g = (1 / λ) * P @ x_vec

        y = np.sum(x_vec * h)

        # A priori error
        error = d[i] - y

        # Conversion factor between a priori and a posteriori errors
        α = 1 + g.T @ x_vec

        h += g * (error / α)

        # Update the inverse correlation matrix
        P = (1 / λ) * P - (1 / λ) * (g / α) @ x_vec.T @ P

        squaredError[i] = error**2
        out[i] = y
        H[i, :] = h.T

    return out, h, squaredError, H


# ---------------------------------------------------------------------------
# Kalman filtering
# ---------------------------------------------------------------------------
@njit
def kalman_filter(A, C, Rn, Rv, x_init, y, P_init=None):
    r"""
    Kalman filter for a linear state-space model without external inputs.

    Parameters
    ----------
    A : numpy.ndarray
        State transition matrix, shape ``(N, N)``.
    C : numpy.ndarray
        Observation matrix, shape ``(L, N)``.
    Rn : numpy.ndarray
        Process noise covariance, shape ``(N, N)``.
    Rv : numpy.ndarray
        Measurement noise covariance, shape ``(L, L)``.
    x_init : numpy.ndarray
        Initial state estimate, shape ``(N, 1)``.
    y : numpy.ndarray
        Observations, shape ``(L, M)``; column ``k`` is the measurement at
        instant ``k``.
    P_init : numpy.ndarray, optional
        Initial error covariance, shape ``(N, N)``. By default
        :math:`\mathbf{x}_0\mathbf{x}_0^T` is used, which is **zero** when
        ``x_init`` is zero (the filter would then fully trust its initial
        state and ignore the measurements): in that case pass ``P_init``
        explicitly, e.g. a multiple of the identity.

    Returns
    -------
    x_hat : numpy.ndarray
        State estimates, shape ``(N, M)``; column ``k`` is the estimate after
        processing ``y[:, k]``.

    Notes
    -----
    The filter estimates the state of

    .. math::

        \mathbf{x}[k+1]=\mathbf{A}\mathbf{x}[k]+\mathbf{n}[k], \qquad
        \mathbf{y}[k]=\mathbf{C}\mathbf{x}[k]+\mathbf{v}[k],

    with :math:`\mathbf{n}\sim\mathcal{N}(\mathbf{0},\mathbf{R}_n)` and
    :math:`\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{R}_v)`. It is the
    recursive minimum mean-square error estimator of
    :math:`\mathbf{x}[k]` given :math:`\mathbf{y}[0],\dots,\mathbf{y}[k]`
    (optimal for Gaussian noise, the best *linear* estimator otherwise).
    Each iteration has two steps.

    *Prediction* (propagate the estimate and its covariance through the
    model):

    .. math::

        \hat{\mathbf{x}}^{-}[k]=\mathbf{A}\hat{\mathbf{x}}[k-1], \qquad
        \mathbf{P}^{-}[k]=\mathbf{A}\mathbf{P}[k-1]\mathbf{A}^T+\mathbf{R}_n .

    *Update* (correct with the innovation, weighted by the Kalman gain):

    .. math::

        \mathbf{K}[k]=\mathbf{P}^{-}\mathbf{C}^T
        \left(\mathbf{C}\mathbf{P}^{-}\mathbf{C}^T+\mathbf{R}_v\right)^{-1},

    .. math::

        \hat{\mathbf{x}}[k]=\hat{\mathbf{x}}^{-}+\mathbf{K}
        \left(\mathbf{y}[k]-\mathbf{C}\hat{\mathbf{x}}^{-}\right), \qquad
        \mathbf{P}[k]=(\mathbf{I}-\mathbf{K}\mathbf{C})\mathbf{P}^{-}.

    The gain is large when the measurement is reliable (:math:`\mathbf{R}_v`
    small) and small when the model is reliable (:math:`\mathbf{R}_n`
    small). RLS is a special case of the Kalman filter, with the filter
    coefficients as the state.

    References
    ----------
    * R. E. Kalman, "A new approach to linear filtering and prediction
      problems", Journal of Basic Engineering, 82(1), 1960.
    """
    # Make sure everything is floating point
    A = A.astype(np.float64)
    C = C.astype(np.float64)
    Rv = Rv.astype(np.float64)
    Rn = Rn.astype(np.float64)
    x = x_init.astype(np.float64)

    if P_init is None:
        Re_posterior = x * x.T
    else:
        Re_posterior = P_init.astype(np.float64)

    x_hat = np.zeros((x.shape[0], y.shape[1]), dtype=np.float64)
    I = np.eye(Re_posterior.shape[0], dtype=np.float64)

    for ind in range(y.shape[1]):
        y_ = y[:, ind : ind + 1]

        # Prediction step
        x_prior = A @ x
        Re_prior = A @ Re_posterior @ A.T + Rn

        # Kalman gain
        K = Re_prior @ C.T @ np.linalg.inv(C @ Re_prior @ C.T + Rv)

        # Update step
        x = x_prior + K @ (y_ - C @ x_prior)
        Re_posterior = (I - K @ C) @ Re_prior

        x_hat[:, ind] = x.flatten()

    return x_hat


def extended_kalman_filter(f, h, F_jac, H_jac, Rn, Rv, x_init, y, P_init=None):
    r"""
    Extended Kalman filter (EKF) for a nonlinear state-space model.

    Parameters
    ----------
    f : callable
        State transition function, ``f(x) -> x_next``, with ``x`` of shape
        ``(N, 1)``.
    h : callable
        Observation function, ``h(x) -> y``, returning shape ``(L, 1)``.
    F_jac : callable
        Jacobian of ``f`` evaluated at ``x``, shape ``(N, N)``.
    H_jac : callable
        Jacobian of ``h`` evaluated at ``x``, shape ``(L, N)``.
    Rn : numpy.ndarray
        Process noise covariance, shape ``(N, N)``.
    Rv : numpy.ndarray
        Measurement noise covariance, shape ``(L, L)``.
    x_init : numpy.ndarray
        Initial state estimate, shape ``(N, 1)``.
    y : numpy.ndarray
        Observations, shape ``(L, M)``.
    P_init : numpy.ndarray, optional
        Initial error covariance, shape ``(N, N)``. By default
        :math:`\mathbf{x}_0\mathbf{x}_0^T`, as in :func:`kalman_filter`.

    Returns
    -------
    x_hat : numpy.ndarray
        State estimates, shape ``(N, M)``.

    Notes
    -----
    For the nonlinear model

    .. math::

        \mathbf{x}[k+1]=f(\mathbf{x}[k])+\mathbf{n}[k], \qquad
        \mathbf{y}[k]=h(\mathbf{x}[k])+\mathbf{v}[k],

    the EKF applies the Kalman filter to the model linearized around the
    current estimate. With the Jacobians
    :math:`\mathbf{F}=\partial f/\partial\mathbf{x}` and
    :math:`\mathbf{H}=\partial h/\partial\mathbf{x}`:

    .. math::

        \hat{\mathbf{x}}^{-}=f(\hat{\mathbf{x}}[k-1]), \qquad
        \mathbf{P}^{-}=\mathbf{F}\mathbf{P}[k-1]\mathbf{F}^T+\mathbf{R}_n,

    .. math::

        \mathbf{K}=\mathbf{P}^{-}\mathbf{H}^T
        (\mathbf{H}\mathbf{P}^{-}\mathbf{H}^T+\mathbf{R}_v)^{-1}, \quad
        \hat{\mathbf{x}}=\hat{\mathbf{x}}^{-}+\mathbf{K}
        \left(\mathbf{y}[k]-h(\hat{\mathbf{x}}^{-})\right), \quad
        \mathbf{P}=(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}^{-}.

    The state is propagated through the *nonlinear* functions; the Jacobians
    are only used for the covariances. The EKF is not optimal and may diverge
    if the linearization is poor or the initialization is far from the true
    state. When ``f`` and ``h`` are linear it reduces to
    :func:`kalman_filter`.

    Examples
    --------
    Estimating a constant from its square:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = (2.0**2 + 0.1 * rng.standard_normal((1, 200)))
    >>> x_hat = extended_kalman_filter(
    ...     f=lambda x: x, h=lambda x: x**2,
    ...     F_jac=lambda x: np.eye(1), H_jac=lambda x: 2 * x,
    ...     Rn=1e-6 * np.eye(1), Rv=np.array([[0.01]]),
    ...     x_init=np.array([[1.0]]), y=y)
    >>> bool(abs(x_hat[0, -1] - 2.0) < 0.05)
    True
    """
    Rn = np.asarray(Rn, dtype=float)
    Rv = np.asarray(Rv, dtype=float)
    x = np.array(x_init, dtype=float)
    y = np.asarray(y, dtype=float)
    P = x @ x.T if P_init is None else np.array(P_init, dtype=float)

    I = np.eye(x.shape[0])
    x_hat = np.zeros((x.shape[0], y.shape[1]))

    for k in range(y.shape[1]):
        y_ = y[:, k : k + 1]

        # Prediction step
        F = np.atleast_2d(F_jac(x))
        x_prior = np.asarray(f(x), dtype=float).reshape(x.shape)
        P_prior = F @ P @ F.T + Rn

        # Update step
        Hk = np.atleast_2d(H_jac(x_prior))
        y_pred = np.asarray(h(x_prior), dtype=float).reshape(y_.shape)
        K = P_prior @ Hk.T @ np.linalg.inv(Hk @ P_prior @ Hk.T + Rv)

        x = x_prior + K @ (y_ - y_pred)
        P = (I - K @ Hk) @ P_prior

        x_hat[:, k] = x.ravel()

    return x_hat


# ---------------------------------------------------------------------------
# Blind adaptation
# ---------------------------------------------------------------------------
@njit
def cma(x, Ntaps, μ, R2=1.0):
    r"""
    Constant Modulus Algorithm (CMA) for blind equalization.

    Parameters
    ----------
    x : numpy.ndarray
        Received signal, shape ``(N,)``; real or complex floating point.
    Ntaps : int
        Number of equalizer coefficients.
    μ : float
        Step size.
    R2 : float, optional
        Dispersion constant :math:`R_2=E|a|^4/E|a|^2` of the transmitted
        constellation (1 for unit-modulus constellations such as PSK or
        2-PAM with unit amplitude). Default 1.

    Returns
    -------
    out : numpy.ndarray
        Equalizer output, shape ``(N,)``.
    h : numpy.ndarray
        Final coefficients, shape ``(Ntaps,)``.
    costFunction : numpy.ndarray
        CMA cost :math:`(|y|^2-R_2)^2` at each iteration, shape ``(N,)``.
    H : numpy.ndarray
        Coefficient history, shape ``(N, Ntaps)``.

    Notes
    -----
    Blind equalization recovers the transmitted symbols without a training
    sequence. The CMA exploits the fact that many modulation formats have
    (nearly) constant modulus, while the intersymbol interference introduced
    by the channel makes the modulus of the received samples fluctuate. It
    minimizes the *dispersion* cost

    .. math::

        J_{CMA}(\mathbf{h}) = E\left[\left(|y[k]|^2-R_2\right)^2\right],
        \qquad y[k]=\mathbf{h}^T\mathbf{x}_k ,

    by stochastic gradient descent:

    .. math::

        \mathbf{h}_{k+1}=\mathbf{h}_k-\mu\,\left(|y[k]|^2-R_2\right)y[k]\,
        \mathbf{x}_k^{*} .

    The "error" :math:`(|y|^2-R_2)\,y` replaces the desired-signal error of
    LMS, so no reference is needed. Because the cost depends only on the
    modulus, the CMA is insensitive to carrier phase offsets (the solution is
    defined up to a constant phase rotation, which must be corrected
    afterwards) and the cost is not convex, so the initialization matters:
    a single-spike initialization (first tap equal to one) is used. For
    suitable channels the CMA converges to the global minimum, which is close
    to the zero-forcing/MMSE equalizer, up to a delay and a phase rotation.

    References
    ----------
    * D. N. Godard, "Self-recovering equalization and carrier tracking in
      two-dimensional data communication systems", IEEE Trans. Commun.,
      28(11), 1980.
    * J. R. Treichler and B. G. Agee, "A new approach to multipath
      correction of constant modulus signals", IEEE Trans. Acoust., Speech,
      Signal Process., 31(2), 1983.
    """
    if Ntaps < 1:
        raise ValueError("Ntaps must be a positive integer.")

    h = np.zeros(Ntaps, dtype=x.dtype)
    h[0] = 1.0  # single-spike initialization

    H = np.zeros((len(x), Ntaps), dtype=x.dtype)
    ind = np.arange(0, Ntaps)

    costFunction = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=x.dtype)

    for i in range(Ntaps, len(x)):
        x_vec = x[i - ind]

        y = np.sum(x_vec * h)

        # CMA "error" term
        error = (np.abs(y) ** 2 - R2) * y

        # Stochastic gradient update of the dispersion cost
        h -= μ * error * np.conj(x_vec)

        costFunction[i] = (np.abs(y) ** 2 - R2) ** 2
        out[i] = y
        H[i, :] = h

    return out, h, costFunction, H


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
@njit
def time_varying_filter(x, H):
    r"""
    Filter a signal with an FIR filter whose coefficients change over time.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, shape ``(N,)``.
    H : numpy.ndarray
        Coefficients, shape ``(N, Ntaps)``; row ``k`` holds the coefficients
        used to compute the output at instant ``k``.

    Returns
    -------
    numpy.ndarray
        Filtered signal, shape ``(N,)``. The first ``Ntaps`` samples are zero.

    Notes
    -----
    Implements the time-varying convolution

    .. math::

        y[k]=\sum_{m=0}^{N_{taps}-1} h_m[k]\,x[k-m],

    which reduces to ordinary convolution when :math:`h_m[k]` does not depend
    on :math:`k`. It is used to create non-stationary systems for testing the
    tracking capability of adaptive filters.
    """
    Ntaps = H.shape[1]
    ind = np.arange(0, Ntaps)

    out = np.zeros(x.shape)

    for i in range(Ntaps, len(x)):
        x_vec = x[i - ind]
        out[i] = np.sum(x_vec * H[i, :])

    return out


def sliding_window_matrix(x, N):
    r"""
    Build the matrix whose columns are sliding windows of a signal.

    Parameters
    ----------
    x : array-like
        Input signal, shape ``(L,)``.
    N : int
        Window length.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(N, L)``; column ``k`` is the window of ``N``
        samples of ``x`` centered at sample ``k`` (``x`` is zero-padded at
        the borders).

    Notes
    -----
    Feeding a neural network with the columns of this matrix turns it into a
    nonlinear FIR filter (a *nonlinear equalizer* or system identifier): the
    network output is a function of ``N`` consecutive input samples.

    Examples
    --------
    >>> sliding_window_matrix([1, 2, 3, 4], 3)
    array([[0, 1, 2, 3],
           [1, 2, 3, 4],
           [2, 3, 4, 0]])
    """
    x = np.asarray(x)
    if N < 1:
        raise ValueError("N must be a positive integer.")
    left = N // 2
    xp = np.pad(x, (left, N - 1 - left))

    return sliding_window_view(xp, N).T


# ---------------------------------------------------------------------------
# Multi-layer perceptron
# ---------------------------------------------------------------------------
@njit
def sigmoid(x):
    r"""
    Logistic sigmoid activation, :math:`\sigma(x)=1/(1+e^{-x})`.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Element-wise sigmoid of ``x``, with values in ``(0, 1)``.
    """
    return 1 / (1 + np.exp(-x))


@njit
def dsigmoid(x):
    r"""
    Derivative of the sigmoid, :math:`\sigma'(x)=\sigma(x)(1-\sigma(x))`.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Element-wise derivative of the sigmoid.
    """
    return sigmoid(x) * (1 - sigmoid(x))


@njit
def ReLU(x):
    r"""
    Rectified linear unit, :math:`\max(0, x)`.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Element-wise ReLU of ``x``.
    """
    return x * (x > 0)


@njit
def dReLU(x):
    r"""
    Derivative of the ReLU (1 for ``x > 0`` and 0 otherwise).

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Element-wise derivative of the ReLU.
    """
    return 1.0 * (x > 0)


@njit
def dtanh(x):
    r"""
    Derivative of the hyperbolic tangent, :math:`1-\tanh^2(x)`.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
        Element-wise derivative of ``tanh``.
    """
    return 1 - np.tanh(x) ** 2


def _activation(name):
    """Return the pair ``(f, df)`` of the activation called ``name``."""
    table = {
        "tanh": (np.tanh, dtanh),
        "sigmoid": (sigmoid, dsigmoid),
        "relu": (ReLU, dReLU),
    }
    try:
        return table[str(name).lower()]
    except KeyError:
        raise ValueError(
            f"Unknown activation '{name}'. Options: 'tanh', 'sigmoid', 'ReLU'."
        ) from None


def initialize_nn(layers, seed=42):
    r"""
    Initialize the weights and biases of a multi-layer perceptron.

    Parameters
    ----------
    layers : list of int
        Number of neurons in each layer, input layer first. For example
        ``[3, 5, 2]`` is a network with 3 inputs, one hidden layer with 5
        neurons and 2 outputs.
    seed : int or None, optional
        Seed of NumPy's global random generator (``np.random.seed``), for
        reproducible results. Use ``None`` to leave the generator untouched.
        Default 42.

    Returns
    -------
    weights : list of numpy.ndarray
        ``weights[i]`` has shape ``(layers[i+1], layers[i])``.
    biases : list of numpy.ndarray
        ``biases[i]`` has shape ``(layers[i+1], 1)``.

    Notes
    -----
    Weights are drawn from :math:`\mathcal{N}(0, 0.01^2)`: small random
    values break the symmetry between neurons (identical neurons would
    receive identical gradients and learn the same feature) while keeping the
    activations in the linear region of the nonlinearity at the beginning of
    training. Biases start at zero.

    Examples
    --------
    >>> weights, biases = initialize_nn([3, 5, 2])
    >>> [w.shape for w in weights]
    [(5, 3), (2, 5)]
    """
    if seed is not None:
        np.random.seed(seed)

    weights = []
    biases = []

    for i in range(1, len(layers)):
        weights.append(np.random.randn(layers[i], layers[i - 1]) * 0.01)
        biases.append(np.zeros((layers[i], 1)))

    return weights, biases


def fit(
    X,
    Y,
    weights,
    biases,
    activation,
    learning_rate,
    epochs,
    batch_size=10,
    verbose=True,
):
    r"""
    Train a multi-layer perceptron with back-propagation and mini-batch
    gradient descent.

    Parameters
    ----------
    X : numpy.ndarray
        Inputs, shape ``(n_features, m)``; ``m`` is the number of examples.
    Y : numpy.ndarray
        Targets, shape ``(n_outputs, m)``.
    weights : list of numpy.ndarray
        Weight matrices (see :func:`initialize_nn`); **updated in place**.
    biases : list of numpy.ndarray
        Bias vectors (see :func:`initialize_nn`); **updated in place**.
    activation : {'tanh', 'sigmoid', 'ReLU'}
        Activation of the hidden layers. The output layer is linear.
    learning_rate : float
        Gradient descent step size.
    epochs : int
        Number of passes over the training set.
    batch_size : int, optional
        Examples per mini-batch (the last batch may be smaller). Default 10.
    verbose : bool, optional
        Show a progress bar. Default ``True``.

    Returns
    -------
    cost : numpy.ndarray
        Cost at each epoch, shape ``(epochs,)``: the average over the
        mini-batches of :math:`\tfrac12\,\mathrm{mean}\,(\hat{Y}-Y)^2`.
    weights : list of numpy.ndarray
        Updated weights.
    biases : list of numpy.ndarray
        Updated biases.

    Raises
    ------
    ValueError
        If ``activation`` is unknown.

    Notes
    -----
    A layer computes :math:`\mathbf{a}_l=\mathbf{W}_l\mathbf{z}_{l-1}+
    \mathbf{b}_l` and :math:`\mathbf{z}_l=\phi(\mathbf{a}_l)`, with
    :math:`\mathbf{z}_0` the input. The weights are adapted to minimize the
    squared error :math:`J=\tfrac12\sum\|\hat{\mathbf{y}}-\mathbf{y}\|^2` of a
    mini-batch. The gradient is computed layer by layer with the chain rule
    (**back-propagation**): starting with the output error
    :math:`\mathbf{e}_L=\hat{\mathbf{y}}-\mathbf{y}`,

    .. math::

        \boldsymbol{\delta}_l=\mathbf{e}_l\odot\phi'(\mathbf{a}_l), \quad
        \frac{\partial J}{\partial\mathbf{W}_l}=\boldsymbol{\delta}_l
        \mathbf{z}_{l-1}^T, \quad
        \frac{\partial J}{\partial\mathbf{b}_l}=\sum\boldsymbol{\delta}_l, \quad
        \mathbf{e}_{l-1}=\mathbf{W}_l^T\boldsymbol{\delta}_l,

    (with :math:`\phi'=1` at the linear output layer) and then
    :math:`\mathbf{W}_l\leftarrow\mathbf{W}_l-\eta\,\partial J/\partial
    \mathbf{W}_l`. Each weight is updated with the gradient of its own layer,
    computed before that layer is modified. The gradient is accumulated
    (summed) over the mini-batch. This is the nonlinear generalization of
    LMS: for a network with a single linear layer it is LMS with a
    mini-batch.

    References
    ----------
    * D. E. Rumelhart, G. E. Hinton and R. J. Williams, "Learning
      representations by back-propagating errors", Nature, 323, 1986.
    """
    phi, df = _activation(activation)
    m = X.shape[1]
    cost = np.zeros(epochs)
    num_batches = int(np.ceil(m / batch_size))

    iterator = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in iterator:
        for batch in range(num_batches):
            sl = slice(batch * batch_size, (batch + 1) * batch_size)
            X_, Y_ = X[:, sl], Y[:, sl]

            # Forward propagation
            Z = X_
            caches = {"Z0": X_}
            for i, (W, b) in enumerate(zip(weights, biases)):
                A = np.dot(W, Z) + b
                Z = A if i == len(weights) - 1 else phi(A)
                caches[f"A{i+1}"] = A
                caches[f"Z{i+1}"] = Z

            Y_hat = Z
            cost[epoch] += 0.5 * np.mean((Y_hat - Y_) ** 2)

            # Backward propagation
            E = Y_hat - Y_  # per-example output error
            for i in reversed(range(len(weights))):
                A = caches[f"A{i+1}"]
                Z_prev = caches[f"Z{i}"]

                dA = np.ones(A.shape) if i == len(weights) - 1 else df(A)

                delta = E * dA
                dW = np.dot(delta, Z_prev.T)
                db = np.sum(delta, axis=1, keepdims=True)

                # Back-propagate with the weights *before* they are updated
                E = np.dot(weights[i].T, delta)

                weights[i] -= learning_rate * dW
                biases[i] -= learning_rate * db

    cost /= num_batches

    return cost, weights, biases


def predict(X, weights, biases, activation):
    r"""
    Forward propagation through a trained multi-layer perceptron.

    Parameters
    ----------
    X : numpy.ndarray
        Inputs, shape ``(n_features, m)``.
    weights : list of numpy.ndarray
        Weight matrices.
    biases : list of numpy.ndarray
        Bias vectors.
    activation : {'tanh', 'sigmoid', 'ReLU'}
        Activation of the hidden layers. The output layer is linear.

    Returns
    -------
    numpy.ndarray
        Network output, shape ``(n_outputs, m)``.

    Raises
    ------
    ValueError
        If ``activation`` is unknown.

    Notes
    -----
    Computes :math:`\mathbf{z}_l=\phi(\mathbf{W}_l\mathbf{z}_{l-1}+
    \mathbf{b}_l)` layer by layer, with a linear output layer. By the
    universal approximation theorem, a network with one hidden layer of
    enough neurons can approximate any continuous function on a compact set,
    which is why MLPs can model the nonlinear systems and channels that
    linear adaptive filters cannot.
    """
    phi, _ = _activation(activation)

    Z = X
    for i, (W, b) in enumerate(zip(weights, biases)):
        A = np.dot(W, Z) + b
        Z = A if i == len(weights) - 1 else phi(A)

    return Z
