import numpy as np
from numba import njit
from tqdm import tqdm


def estimate_correlation_matrix(x, N):
    # sourcery skip: inline-immediately-returned-variable
    """
    Estimate the unbiased correlation matrix R of order N from a sequence of random values x.

    Parameters
    ----------
    x : array-like
        The sequence of random values (x0, x1, ..., xM-1) of length M.
    N : int
        The order of the correlation matrix. Must be less than or equal to M.

    Returns
    -------
    R : np.ndarray
        The estimated unbiased correlation matrix of shape (N, N).
        Each element R[i, j] represents the unbiased estimate of the correlation between x[k]
        and x[k - |i - j|] over the sequence.

    Raises
    ------
    ValueError
        If N is greater than the length of the sequence M.

    Notes
    -----
    The unbiased correlation matrix is estimated by normalizing the sum of products
    of values from the sequence x at different lags. This compensates for the
    decreasing number of available pairs as the lag increases.

    """
    M = len(x)
    if N > M:
        raise ValueError(
            "Order N should be smaller than or equal to the length of the sequence M."
        )

    # Create a matrix where each row is a shifted version of the original sequence
    X = np.array([x[i : M - N + i + 1] for i in range(N)])

    # Compute the unbiased correlation matrix
    R = (X @ X.T) / (M - np.arange(N)[:, None])

    return R


def estimate_cross_correlation(x, d, N):
    """
    Estimate the unbiased cross-correlation vector p[k] between x_vec[k] = [x[k], x[k-1], ..., x[k-N+1]]
    and the sample d[k] of the sequence d.

    Parameters
    ----------
    x : array-like
        The sequence of random values (x0, x1, ..., xM-1) of length M.
    d : array-like
        The sequence d of length M, representing the values to correlate with x_vec.
    N : int
        The length of the vector x_vec (order of cross-correlation). Must be less than or equal to M.

    Returns
    -------
    p : np.ndarray
        The unbiased cross-correlation vector of length N.
        Each element represents the estimated unbiased cross-correlation between the
        vector x_vec = [x[k], x[k-1], ..., x[k-N+1]] and d[k].

    Raises
    ------
    ValueError
        If the sequences `x` and `d` do not have the same length.
        If `N` is greater than the length of the sequences.

    Notes
    -----
    The cross-correlation is estimated by sliding a vector x_vec of length N over the sequence x
    and computing the product of x_vec and d[k]. The result is normalized to account for the
    decreasing number of terms that contribute to the estimate as k approaches the end of the sequence.

    """
    M = len(x)
    if len(d) != M:
        raise ValueError("The sequences x and d must have the same length.")
    if N > M:
        raise ValueError(
            "N should be smaller than or equal to the length of the sequences."
        )

    p = np.zeros(N)
    count = np.zeros(
        N
    )  # To keep track of the number of terms contributing to each element of p

    # Estimate the unbiased cross-correlation
    for k in range(N, M):  # Start from k = N to ensure we can form x_vec
        x_vec = x[
            k : k - N : -1
        ]  # Create the vector x_vec = [x[k], x[k-1], ..., x[k-N+1]]
        p += x_vec * d[k]
        count += 1  # Keep track of the number of terms contributing to each element

    # Normalize to make the estimation unbiased
    p /= count

    return p


@njit
def lms(x, d, Ntaps, μ):
    """
    Implements the Least Mean Squares (LMS) adaptive filter algorithm.

    Parameters
    ----------
    x : ndarray
        The input signal, a 1D array representing the signal to be filtered.
    d : ndarray
        The reference signal, a 1D array of the same length as `x`, representing the desired output.
    Ntaps : int
        The number of taps (coefficients) in the adaptive filter.
    μ : float
        The step size for the LMS algorithm, controlling the rate of convergence.

    Returns
    -------
    out : ndarray
        The output signal, a 1D array representing the signal produced by the filter at each iteration.
    h : ndarray
        The final filter coefficients, a 1D array of length `Ntaps` representing the filter after the last iteration.
    squaredError : ndarray
        The squared error between the reference signal and the output signal at each iteration, a 1D array of the same length as `x`.
    H : ndarray
        A 2D array where each row contains the filter coefficients at each iteration, with shape `(len(x) - Ntaps, Ntaps)`.

    Notes
    -----
    The LMS algorithm adjusts the filter coefficients to minimize the error between the filter output and the reference signal.
    The filter coefficients are updated at each iteration based on the current error and input signal.
    """
    # Initialize the equalizer filter coefficients
    h = np.zeros(Ntaps, dtype=np.float64)
    H = np.zeros((len(x) - Ntaps, Ntaps), dtype=np.float64)
    ind = np.arange(0, Ntaps)

    # Apply the LMS algorithm
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)

    # Iterate through each sample of the signal
    for i in range(Ntaps, len(x)):
        x_vec = x[i - ind]

        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)

        # Compute the error between the estimated signal and the reference signal
        error = d[i] - y

        # Update the filter coefficients using the LMS update rule
        h += 2 * μ * error * x_vec

        squaredError[i] = error**2
        out[i] = y
        H[i - Ntaps, :] = h

    return out, h, squaredError, H


@njit
def nlms(x, d, Ntaps, μ, γ=1e-6):
    """
    Applies the Normalized Least Mean Squares (NLMS) algorithm for adaptive filtering.

    The NLMS algorithm is a variant of the Least Mean Squares (LMS) algorithm that adjusts
    the step size based on the norm of the input signal to improve convergence stability.

    Parameters
    ----------
    x : ndarray
        Input signal of shape (n_samples,).
    d : ndarray
        Reference or desired signal of shape (n_samples,).
    Ntaps : int
        Number of filter taps (coefficients).
    μ : float
        Step size parameter for the NLMS update, controlling the adaptation rate.
    γ : float, optional
        Small regularization constant to avoid division by zero, by default 1e-6.

    Returns
    -------
    out : ndarray
        Output signal of shape (n_samples,), representing the filtered signal.
    h : ndarray
        Final filter coefficients of shape (Ntaps, 1).
    squaredError : ndarray
        Squared error at each iteration of shape (n_samples,).
    H : ndarray
        Evolution of filter coefficients across iterations, with shape (n_samples - Ntaps, Ntaps).

    """
    # Initialize the equalizer filter coefficients
    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x) - Ntaps, Ntaps), dtype=np.float64)
    ind = np.arange(0, Ntaps)

    # Apply the LMS algorithm
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    # Iterate through each sample of the signal
    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)

        # Compute the error between the estimated signal and the reference signal
        error = d[i] - y

        # Update the filter coefficients using the NLMS update rule
        h += μ * error * x_vec / (γ + x_vec.T @ x_vec)

        squaredError[i] = error**2
        out[i] = y
        H[i - Ntaps, :] = h.T

    return out, h, squaredError, H


@njit
def lms_newton(x, d, Ntaps, μ, α):
    """
    Applies the Least Mean Squares (LMS) Newton algorithm for adaptive filtering.

    The LMS-Newton algorithm adapts the filter coefficients by approximating the inverse
    correlation matrix of the input signal, resulting in improved convergence properties
    compared to standard LMS.

    Parameters
    ----------
    x : ndarray
        Input signal of shape (n_samples,).
    d : ndarray
        Reference or desired signal of shape (n_samples,).
    Ntaps : int
        Number of filter taps (coefficients).
    μ : float
        Step size parameter for the LMS update, controlling the adaptation rate.
    α : float
        Parameter for updating the inverse correlation matrix, balancing the influence
        of new and old information in the matrix.

    Returns
    -------
    out : ndarray
        Output signal of shape (n_samples,), representing the filtered signal.
    h : ndarray
        Final filter coefficients of shape (Ntaps, 1).
    squaredError : ndarray
        Squared error at each iteration, with shape (n_samples,).
    H : ndarray
        Evolution of filter coefficients across iterations, with shape (n_samples - Ntaps, Ntaps).
    """
    # Initialize the equalizer filter coefficients
    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x) - Ntaps, Ntaps), dtype=np.float64)
    R_inv = 1e-3 * np.eye(Ntaps, dtype=np.float64)

    # Apply the LMS-Newton algorithm
    ind = np.arange(0, Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    # Iterate through each sample of the signal
    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)

        # Compute the error between the estimated signal and the reference signal
        error = d[i] - y

        # Update inverse correlation matrix
        R_inv = (
            1
            / (1 - α)
            * (
                R_inv
                - (R_inv @ (x_vec @ x_vec.T) @ R_inv)
                / ((1 - α) / α + x_vec.T @ R_inv @ x_vec)
            )
        )

        # Update the filter coefficients using the LMS-Newton update rule
        h += μ * error * R_inv @ x_vec

        squaredError[i] = error**2
        out[i] = y
        H[i - Ntaps, :] = h.T

    return out, h, squaredError, H


@njit
def rls(x, d, Ntaps, λ):
    """
    Applies the Recursive Least Squares (RLS) algorithm for adaptive filtering.

    The RLS algorithm recursively computes the filter coefficients to minimize the
    sum of the squares of the errors, providing fast convergence for time-varying
    signals by adjusting the forgetting factor `λ`.

    Parameters
    ----------
    x : ndarray
        Input signal of shape (n_samples,).
    d : ndarray
        Reference or desired signal of shape (n_samples,).
    Ntaps : int
        Number of filter taps (coefficients).
    λ : float
        Forgetting factor for the RLS algorithm, where 0 < λ ≤ 1. Lower values give more
        weight to recent data, enhancing adaptation in non-stationary environments.

    Returns
    -------
    out : ndarray
        Output signal of shape (n_samples,), representing the filtered signal.
    h : ndarray
        Final filter coefficients of shape (Ntaps, 1).
    squaredError : ndarray
        Squared error at each iteration, with shape (n_samples,).
    H : ndarray
        Evolution of filter coefficients across iterations, with shape (n_samples - Ntaps, Ntaps).

    """
    # Initialize the equalizer filter coefficients
    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x) - Ntaps, Ntaps), dtype=np.float64)
    Rxx_inv = 1e-3 * np.eye(Ntaps, dtype=np.float64)
    pxd = np.zeros((Ntaps, 1), dtype=np.float64)

    # Apply the LMS-Newton algorithm
    ind = np.arange(0, Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    # Iterate through each sample of the signal
    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        # Update inverse correlation matrix
        Rxx_inv = (
            1
            / λ
            * (
                Rxx_inv
                - (Rxx_inv @ (x_vec @ x_vec.T) @ Rxx_inv)
                / (λ + x_vec.T @ Rxx_inv @ x_vec)
            )
        )

        # Update inverse correlation matrix
        pxd = x_vec * d[i] + λ * pxd

        # Update the filter coefficients using the RLS update rule
        h = Rxx_inv @ pxd

        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)

        # Compute the a posteriori error between the estimated signal and the reference signal
        error = d[i] - y

        squaredError[i] = error**2
        out[i] = y
        H[i - Ntaps, :] = h.T

    return out, h, squaredError, H


@njit
def rls_apriori(x, d, Ntaps, λ):
    """
    The Recursive Least Squares (RLS) algorithm with a priori error.

    Parameters
    ----------
    x : ndarray
        The input signal, a 1D array representing the signal to be filtered.
    d : ndarray
        The reference signal, a 1D array of the same length as `x`, representing the desired output.
    Ntaps : int
        The number of filter taps (coefficients).
    λ : float
        The forgetting factor for the RLS algorithm, where 0 < λ ≤ 1. Lower values give more
        weight to recent data, enhancing adaptation in non-stationary environments.

    Returns
    -------
    out : ndarray
        The output signal, a 1D array representing the filtered signal at each iteration.
    h : ndarray
        The final filter coefficients, a 1D array of length `Ntaps` representing the filter after the last iteration.
    squaredError : ndarray
        The squared error between the reference signal and the output signal at each iteration, a 1D array of the same length as `x`.
    H : ndarray
        A 2D array where each row contains the filter coefficients at each iteration, with shape `(len(x) - Ntaps, Ntaps)`.
    """
    # Initialize the equalizer filter coefficients
    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x) - Ntaps, Ntaps), dtype=np.float64)
    P = 1e-3 * np.eye(Ntaps, dtype=np.float64)

    # Apply the LMS-Newton algorithm
    ind = np.arange(0, Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    # Iterate through each sample of the signal
    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]
        # Calculate the Kalman gain
        g = (1 / λ) * P @ x_vec / (1 + (1 / λ) * x_vec.T @ P @ x_vec)

        # Generate the filter output
        y = np.sum(x_vec * h)

        # Compute the a priori error between the estimated signal and the reference signal
        error = d[i] - y

        # Update the filter coefficients using the RLS update rule with error a priori
        h += g * error

        # Update inverse correlation matrix
        P = (1 / λ) * P - (1 / λ) * g @ x_vec.T @ P

        squaredError[i] = error**2
        out[i] = y
        H[i - Ntaps, :] = h.T

    return out, h, squaredError, H


@njit
def rls_aposteriori(x, d, Ntaps, λ):
    """
    The Recursive Least Squares (RLS) algorithm with a posteriori error.

    Parameters
    ----------
    x : ndarray
        The input signal, a 1D array representing the signal to be filtered.
    d : ndarray
        The reference signal, a 1D array of the same length as `x`, representing the desired output.
    Ntaps : int
        The number of filter taps (coefficients).
    λ : float
        The forgetting factor for the RLS algorithm, where 0 < λ ≤ 1. Lower values give more
        weight to recent data, enhancing adaptation in non-stationary environments.

    Returns
    -------
    out : ndarray
        The output signal, a 1D array representing the filtered signal at each iteration.
    h : ndarray
        The final filter coefficients, a 1D array of length `Ntaps` representing the filter after the last iteration.
    squaredError : ndarray
        The squared error between the reference signal and the output signal at each iteration, a 1D array of the same length as `x`.
    H : ndarray
        A 2D array where each row contains the filter coefficients at each iteration, with shape `(len(x) - Ntaps, Ntaps)`.
    """
    # Initialize the equalizer filter coefficients
    h = np.zeros((Ntaps, 1), dtype=np.float64)
    H = np.zeros((len(x) - Ntaps, Ntaps), dtype=np.float64)
    P = 1e-3 * np.eye(Ntaps, dtype=np.float64)

    # Apply the LMS-Newton algorithm
    ind = np.arange(0, Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    x = x.reshape(-1, 1).astype(np.float64)

    # Iterate through each sample of the signal
    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i - ind, :]

        # Calculate the Kalman gain
        g = (1 / λ) * P @ x_vec

        # Generate the filter output
        y = np.sum(x_vec * h)

        # Compute the a priori error between the estimated signal and the reference signal
        error = d[i] - y

        # Calculate conversion factor
        α = 1 + g.T @ x_vec

        # Update the filter coefficients using the RLS update rule with error a posteriori
        h += g * (error / α)

        # Update inverse correlation matrix
        P = (1 / λ) * P - (1 / λ) * (g / α) @ x_vec.T @ P

        squaredError[i] = error**2
        out[i] = y
        H[i - Ntaps, :] = h.T

    return out, h, squaredError, H


@njit
def kalman_filter(A, C, Rn, Rv, x_init, y):
    """
    Applies the Kalman filter to a series of observations for a system without external inputs.

    The Kalman filter iteratively estimates the state of a linear dynamic system by minimizing
    the mean squared error. It operates in two steps: a prediction step and an update step.

    System model:
        x[k+1] = A * x[k] + n[k]
        y[k] = C * x[k] + v[k]

    where:
        n[k] ~ Normal(0, Rn)
        v[k] ~ Normal(0, Rv)

    Parameters
    ----------
    A : ndarray
        State transition matrix of shape (N, N).
    C : ndarray
        Observation matrix of shape (L, N).
    Rn : ndarray
        Process noise covariance matrix of shape (N, N).
    Rv : ndarray
        Measurement noise covariance matrix of shape (L, L).
    x_init : ndarray
        Initial state estimate of shape (N, 1).
    y : ndarray
        Array of observations (measurement vectors) with shape (L, M),
        where each column corresponds to an observation at a time instant.

    Returns
    -------
    x_hat : ndarray
        Array of state estimates with shape (N, M), containing the estimated state after
        each observation.

    Notes
    -----
    N is the dimension of the state vector; L is the dimension of the observation vector;
    M is the number of measurement samples. The Kalman gain `K` is computed at each time
    step to optimally balance the prediction with the measurement. `Re_posterior` is updated
    to refine the state covariance estimate, and an identity matrix.

    """
    # Initialize state and covariance
    x = x_init
    Re_posterior = x_init * x_init.T

    # Assume all float variables
    Rv = Rv.astype(np.float64)
    Rn = Rn.astype(np.float64)
    x = x.astype(np.float64)
    C = C.astype(np.float64)

    # Pre-allocate
    x_hat = np.zeros((x_init.shape[0], y.shape[1]), dtype=np.float64)
    I = np.eye(Re_posterior.shape[0], dtype=np.float64)

    for ind in range(y.shape[1]):
        y_ = y[:, ind : ind + 1]

        # Prediction step: E[x[k]|x[k-1]]
        x_prior = A @ x
        Re_prior = A @ Re_posterior @ A.T + Rn

        # Kalman gain
        K = Re_prior @ C.T @ np.linalg.inv(C @ Re_prior @ C.T + Rv)

        # Update step: E[x[k]|y[k]]
        x = x_prior + K @ (y_ - C @ x_prior)
        Re_posterior = (I - K @ C) @ Re_prior

        # Store estimates
        x_hat[:, ind] = x.flatten()

    return x_hat


@njit
def cma(x, Ntaps, mu):
    """
    The Constant Modulus Algorithm (CMA) for blind equalization.

    Parameters
    ----------
    x : ndarray
        Input signal, a 1D array representing the received signal to be filtered.
    Ntaps : int
        The number of filter taps.
    mu : float
        The CMA step size.

    Returns
    -------
    out : ndarray
        The output signal, a 1D array representing the filtered signal at each iteration.
    h : ndarray
        The final filter coefficients, a 1D array of length `Ntaps`.
    costFunction : ndarray
        The CMA cost function value at each iteration, a 1D array of the same length as `x`.
    H : ndarray
        A 2D array where each row contains the filter coefficients at each iteration, with shape `(len(x) - Ntaps, Ntaps)`.

    """
    # Initialize the equalizer filter coefficients
    h = np.zeros(Ntaps)
    h[0] = 1.0  # Initialize first tap to 1

    H = np.zeros((len(x) - Ntaps, Ntaps))
    ind = np.arange(0, Ntaps)

    # Apply the CMA algorithm
    costFunction = np.zeros(x.shape)
    out = np.zeros(x.shape)

    # Iterate through each sample of the signal
    for i in range(Ntaps, len(x)):
        x_vec = x[i - ind]

        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)

        # Compute the error between the estimated signal and the reference signal
        error = (np.abs(y) ** 2 - 1) * y

        # Update the filter coefficients using the CMA update rule
        h -= mu * error * x_vec

        costFunction[i] = (np.abs(y) ** 2 - 1) ** 2
        out[i] = y
        H[i, :] = h

    return out, h, costFunction, H


@njit
def time_varying_filter(x, H):
    """
    Implements the time-varying filter algorithm.

    Parameters
    ----------
    x : ndarray
        Input signal, a 1D array representing the signal to be filtered.
    H : ndarray
        Time-varying filter coefficients, a 2D array where each row represents the filter coefficients at a given time instant.

    Returns
    -------
    ndarray
        The filtered signal.

    """
    # Initialize the equalizer filter coefficients
    Ntaps = H.shape[1]
    ind = np.arange(0, Ntaps)

    # Apply the filtering algorithm
    out = np.zeros(x.shape)

    # Iterate through each sample of the signal
    for i in range(Ntaps, len(x)):
        x_vec = x[i - ind]
        # Generate the filter output
        out[i] = np.sum(x_vec * H[i, :])

    return out


@njit
def sigmoid(x):
    """
    Computes the sigmoid activation function.

    Parameters
    ----------
    x : ndarray
        Input array for which to compute the sigmoid function.

    Returns
    -------
    ndarray
        The sigmoid of the input array, computed element-wise.
    """
    return 1 / (1 + np.exp(-x))


@njit
def dsigmoid(x):
    """
    Computes the derivative of the sigmoid activation function.

    Parameters
    ----------
    x : ndarray
        Input array for which to compute the derivative of the sigmoid function.

    Returns
    -------
    ndarray
        The derivative of the sigmoid of the input array, computed element-wise.
    """
    return sigmoid(x) * (1 - sigmoid(x))


@njit
def ReLU(x):
    """
    Computes the Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    x : ndarray
        Input array for which to compute the ReLU function.

    Returns
    -------
    ndarray
        The ReLU of the input array, computed element-wise.
    """
    return x * (x > 0)


@njit
def dReLU(x):
    """
    Computes the derivative of the Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    x : ndarray
        Input array for which to compute the derivative of the ReLU function.

    Returns
    -------
    ndarray
        The derivative of the ReLU of the input array, computed element-wise.
    """
    return 1.0 * (x > 0)


@njit
def dtanh(x):
    """
    Computes the derivative of the hyperbolic tangent (tanh) activation function.

    Parameters
    ----------
    x : ndarray
        Input array for which to compute the derivative of the tanh function.

    Returns
    -------
    ndarray
        The derivative of the tanh of the input array, computed element-wise.
    """
    return 1 - np.tanh(x) ** 2


def initialize_nn(layers):
    """
    Initializes weight matrices and biases vectors for each layer in the network structure.

    Parameters
    ----------
    layers : list of int
        A list where each element represents the number of neurons in the corresponding layer of the network.
        For example, [3, 5, 2] represents a network with an input layer of 3 neurons, one hidden layer of 5 neurons,
        and an output layer of 2 neurons.

    Returns
    -------
    weights : list of np.ndarray
        A list of weight matrices for each layer, where each matrix is of shape (neurons_in_current_layer, neurons_in_previous_layer).
    biases : list of np.ndarray
        A list of bias vectors for each layer, where each vector is of shape (neurons_in_current_layer, 1).

    Notes
    -----
    The weights are initialized using a small random normal distribution (scaled by 0.01) to break symmetry and ensure that neurons learn different features.
    The biases are initialized to zero. The random seed is set to 42 for reproducibility of results across different runs. The function assumes that the input layer (first element in `layers`) does not require weights or biases, as it serves as the input to the network.
    """
    np.random.seed(42)  # For reproducibility
    weights = []
    biases = []

    # Loop through each layer (skip input layer at index 0)
    for i in range(1, len(layers)):
        weight_matrix = np.random.randn(layers[i], layers[i - 1]) * 0.01
        bias_vector = np.zeros((layers[i], 1))

        weights.append(weight_matrix)
        biases.append(bias_vector)

    return weights, biases


def fit(X, Y, weights, biases, activation, learning_rate, epochs, batch_size=10):
    """
    Trains the neural network by performing forward and backward propagation
    and updating weights and biases based on the error.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_features, m), where `m` is the number of examples.
    Y : np.ndarray
        True labels of shape (1, m).
    weights : list of np.ndarray
        List of weight matrices for each layer, where each matrix is of shape
        (neurons_in_current_layer, neurons_in_previous_layer).
    biases : list of np.ndarray
        List of bias vectors for each layer, where each vector is of shape
        (neurons_in_current_layer, 1).
    activation : str
        Activation function to use in each layer, options are 'tanh', 'sigmoid', or 'ReLU'.
    learning_rate : float
        Learning rate for gradient descent.
    epochs : int
        Number of times to iterate over the entire dataset.
    batch_size : int, optional
        Number of examples per batch for batch gradient descent. Default is 10.

    Returns
    -------
    cost : np.ndarray
        Array containing the cost (mean squared error) at each epoch.
    weights : list of np.ndarray
        Updated list of weight matrices after training.
    biases : list of np.ndarray
        Updated list of bias vectors after training.

    Notes
    -----
    This function performs forward propagation, computes the cost using mean squared error,
    performs backward propagation, and updates the weights and biases using gradient descent.
    Each epoch's cost is calculated by averaging the mean squared error across all batches.
    """
    cost = np.zeros(epochs)
    num_batches = int(X.shape[1] / batch_size)

    for epoch in tqdm(range(epochs)):
        for batch in range(num_batches):
            X_ = X[:, batch * batch_size : (batch + 1) * batch_size]
            Y_ = Y[:, batch * batch_size : (batch + 1) * batch_size]

            # Step 1: Forward Propagation
            Z = X_  # Start with input data
            caches = {"Z0": X_}  # Store activations for each layer
            for i, (W, b) in enumerate(zip(weights, biases)):
                A = np.dot(W, Z) + b  # Linear step

                if i == len(weights) - 1:
                    Z = A
                elif activation == "tanh":
                    Z = np.tanh(A)
                elif activation == "sigmoid":
                    Z = sigmoid(A)
                elif activation == "ReLU":
                    Z = ReLU(A)

                caches[f"A{i+1}"] = A
                caches[f"Z{i+1}"] = Z

            # Final output of the network
            Y_hat = Z

            cost[epoch] += 0.5 * np.mean((Y_hat - Y_) ** 2)  # Mean squared error cost

            # Step 2: Backward Propagation
            E = np.mean(Y_hat - Y_, axis=1, keepdims=True)  # Average batch error

            for i in reversed(range(len(weights))):
                A = caches[f"A{i+1}"]
                Z_prev = caches[f"Z{i}"]

                if i == len(weights) - 1:
                    activation_derivative = 1.0 * np.ones(A.shape, dtype=np.float64)
                elif activation == "tanh":
                    activation_derivative = dtanh(A)
                elif activation == "sigmoid":
                    activation_derivative = dsigmoid(A)
                elif activation == "ReLU":
                    activation_derivative = dReLU(A)

                delta = E * activation_derivative  # Error term
                dW = np.dot(delta, Z_prev.T)  # Gradient of cost w.r.t. W
                db = np.sum(delta, axis=1, keepdims=True)  # Gradient of cost w.r.t. b
                E = np.dot(weights[i].T, delta)  # Update E

                # Update weights and biases
                weights[i] -= learning_rate * dW
                biases[i] -= learning_rate * db

    cost /= num_batches

    return cost, weights, biases


def predict(X, weights, biases, activation):
    """
    Performs forward propagation (prediction) through the network.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_features, m), where `m` is the number of examples.
    weights : list of np.ndarray
        List of weight matrices for each layer, where each matrix is of shape
        (neurons_in_current_layer, neurons_in_previous_layer).
    biases : list of np.ndarray
        List of bias vectors for each layer, where each vector is of shape
        (neurons_in_current_layer, 1).
    activation : str
        Activation function to use in each layer, options are 'tanh', 'sigmoid', or 'ReLU'.

    Returns
    -------
    np.ndarray
        The predictions for each example in `X`, with the same shape as the final layer's output.

    Notes
    -----
    This function performs only forward propagation, applying the specified activation function
    in each layer to compute the final output, which is the network's prediction.
    """
    Z = X  # Start with input data
    for i, (W, b) in enumerate(zip(weights, biases)):
        A = np.dot(W, Z) + b  # Linear step

        if i == len(weights) - 1:
            Z = A
        elif activation == "tanh":
            Z = np.tanh(A)
        elif activation == "sigmoid":
            Z = sigmoid(A)
        elif activation == "ReLU":
            Z = ReLU(A)

    return Z
