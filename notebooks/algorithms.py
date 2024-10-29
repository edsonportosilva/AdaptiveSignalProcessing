import numpy as np
from numba import njit

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
        raise ValueError("Order N should be smaller than or equal to the length of the sequence M.")
    
    # Create a matrix where each row is a shifted version of the original sequence
    X = np.array([x[i:M-N+i+1] for i in range(N)])
    
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
        raise ValueError("N should be smaller than or equal to the length of the sequences.")

    p = np.zeros(N)
    count = np.zeros(N)  # To keep track of the number of terms contributing to each element of p
    
    # Estimate the unbiased cross-correlation
    for k in range(N, M):  # Start from k = N to ensure we can form x_vec
        x_vec = x[k:k-N:-1]  # Create the vector x_vec = [x[k], x[k-1], ..., x[k-N+1]]
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
    h = np.zeros(Ntaps) 
    H = np.zeros((len(x)-Ntaps, Ntaps))
    ind = np.arange(0,Ntaps)
   
    # Apply the LMS algorithm
    squaredError = np.zeros(x.shape)
    out  = np.zeros(x.shape)
        
    # Iterate through each sample of the signal
    for i in range(Ntaps, len(x)):
        x_vec = x[i-ind]

        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)

        # Compute the error between the estimated signal and the reference signal
        error = d[i] - y

        # Update the filter coefficients using the LMS update rule
        h += μ * error * x_vec 

        squaredError[i] = error**2
        out[i] = y
        H[i,:] = h

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
    h = np.zeros((Ntaps,1), dtype=np.float64) 
    H = np.zeros((len(x)-Ntaps, Ntaps), dtype=np.float64)
    ind = np.arange(0,Ntaps)
   
    # Apply the LMS algorithm
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)    
    x = x.reshape(-1,1).astype(np.float64)
    
    # Iterate through each sample of the signal
    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i-ind,:]    
              
        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)
    
        # Compute the error between the estimated signal and the reference signal
        error = d[i] - y             
                              
        # Update the filter coefficients using the NLMS update rule
        h += μ * error * x_vec/(γ + x_vec.T@x_vec)
        
        squaredError[i] = error**2
        out[i] = y
        H[i-Ntaps,:] = h.T

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
    h = np.zeros((Ntaps,1), dtype=np.float64) 
    H = np.zeros((len(x)-Ntaps, Ntaps), dtype=np.float64)
    R_inv = 1e-3*np.eye(Ntaps, dtype=np.float64)
       
    # Apply the LMS-Newton algorithm
    ind = np.arange(0,Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)    
    x = x.reshape(-1,1).astype(np.float64)
    
    # Iterate through each sample of the signal
    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i-ind,:]    
              
        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)
    
        # Compute the error between the estimated signal and the reference signal
        error = d[i] - y             
        
        # Update inverse correlation matrix      
        R_inv = 1/(1-α)*(R_inv - ( R_inv @ (x_vec@x_vec.T) @ R_inv)/( (1-α)/α + x_vec.T @ R_inv @ x_vec) )
                               
        # Update the filter coefficients using the LMS-Newton update rule
        h += μ * error * R_inv @ x_vec 
        
        squaredError[i] = error**2
        out[i] = y
        H[i-Ntaps,:] = h.T

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
    h = np.zeros((Ntaps,1), dtype=np.float64) 
    H = np.zeros((len(x)-Ntaps, Ntaps), dtype=np.float64)
    Rxx_inv = 1e-3*np.eye(Ntaps, dtype=np.float64)
    pxd = np.zeros((Ntaps,1), dtype=np.float64)
       
    # Apply the LMS-Newton algorithm
    ind = np.arange(0,Ntaps)
    squaredError = np.zeros(x.shape, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.float64)    
    x = x.reshape(-1,1).astype(np.float64)
    
    # Iterate through each sample of the signal
    for i in range(Ntaps, x.shape[0]):
        x_vec = x[i-ind,:]    

        # Update inverse correlation matrix      
        Rxx_inv = 1/λ*(Rxx_inv - ( Rxx_inv @ (x_vec@x_vec.T) @ Rxx_inv)/( λ + x_vec.T @ Rxx_inv @ x_vec) )

        # Update inverse correlation matrix      
        pxd = x_vec * d[i] + λ*pxd

        # Update the filter coefficients using the RLS update rule
        h = Rxx_inv @ pxd 
              
        # Generate the estimated signal using the equalizer filter
        y = np.sum(x_vec * h)
    
        # Compute the a posteriori error between the estimated signal and the reference signal
        error = d[i] - y    
        
        squaredError[i] = error**2
        out[i] = y
        H[i-Ntaps,:] = h.T

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
    Re_posterior = x_init*x_init.T
    
    # Assume all float variables
    Rv = Rv.astype(np.float64)
    Rn = Rn.astype(np.float64)
    x = x.astype(np.float64)
    C = C.astype(np.float64)       
    
    # Pre-allocate
    x_hat = np.zeros((x_init.shape[0], y.shape[1]), dtype=np.float64)
    I = np.eye(Re_posterior.shape[0], dtype=np.float64)
    
    for ind in range(y.shape[1]):
        y_ = y[:,ind:ind+1]
                
        # Prediction step: E[x[k]|x[k-1]]
        x_prior = A @ x   
        Re_prior = A @ Re_posterior @ A.T + Rn     
               
        # Kalman gain                        
        K = Re_prior @ C.T @ np.linalg.inv(C @ Re_prior @ C.T + Rv)
        
        # Update step: E[x[k]|y[k]]
        x = x_prior + K @ (y_ - C @ x_prior)
        Re_posterior = (I - K @ C) @ Re_prior
                
        # Store estimates
        x_hat[:,ind] = x.flatten()

    return x_hat
@njit
def time_varying_filter(x, H):
    """
    Implements the time-varying filter algorithm.

    """
    # Initialize the equalizer filter coefficients
    Ntaps = H.shape[1]
    ind = np.arange(0,Ntaps)
   
    # Apply the filtering algorithm   
    out  = np.zeros(x.shape)
        
    # Iterate through each sample of the signal
    for i in range(Ntaps, len(x)):
        x_vec = x[i-ind]
        # Generate the filter output
        out[i] = np.sum(x_vec * H[i,:])
      
    return out