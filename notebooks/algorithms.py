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
    The Normalized Least Mean Squares (LMS) Newton algorithm.

    Parameters:
        x (ndarray): The input signal.
        d (ndarray): The reference signal.
        Ntaps (int): The number of filter taps.
        μ (float): The LMS step size.
        γ (float): Parameter to avoid large step sizes when the norm of the input vector is small.

    Returns:
        tuple: A tuple containing:
            - ndarray: The output signal.
            - ndarray: The final filter coefficients.
            - ndarray: The squared error at each iteration.

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
    The Least Mean Squares (LMS) Newton algorithm.

    Parameters:
        x (ndarray): The input signal.
        d (ndarray): The reference signal.
        Ntaps (int): The number of filter taps.
        μ (float): The LMS step size.
        α (float): The correlation matrix update parameter.

    Returns:
        tuple: A tuple containing:
            - ndarray: The output signal.
            - ndarray: The final filter coefficients.
            - ndarray: The squared error at each iteration.

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