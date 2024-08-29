
import numpy as np
from scipy.signal import square
import matplotlib.pyplot as plt

def random_square_signal(num_samples, period, duty_cycle=0.5):
    """
    Generate a pseudo-random sequence of square pulses with a specific period, number of samples, and duty cycle.

    Parameters:
    - num_samples (int): Total number of samples.
    - period (int): The period of the square wave in samples.
    - duty_cycle (float): Duty cycle of the wave as a fraction (0 to 1).

    Returns:
    - np.ndarray: Array containing the pseudo-random square wave.
    """
    # Initialize the square wave array
    square_wave = np.zeros(num_samples)
    
    # Calculate the target number of high samples within a period
    high_samples = int(period * duty_cycle)
    low_samples = period - high_samples

    # Current position in the signal
    pos = 0

    while pos < num_samples:
        # Randomly generate the high duration within a reasonable range around the target
        high_duration = np.random.randint(1, high_samples * 2)
        
        # Randomly generate the low duration within a reasonable range around the target
        low_duration = np.random.randint(1, low_samples * 2)
        
        # Adjust high and low durations if they exceed the period or remaining samples
        total_duration = high_duration + low_duration
        if total_duration > period:
            scale = period / total_duration
            high_duration = int(high_duration * scale)
            low_duration = period - high_duration
        
        # Assign the high values
        end_pos = min(pos + high_duration, num_samples)
        square_wave[pos:end_pos] = 1
        pos = end_pos
        
        # Skip the low values
        pos += low_duration
    
    return square_wave

def gradientAdaptation(x, y, Ntaps=5, μ=1e-3, epochs=2):
    """
    Perform gradient adaptation to update filter coefficients of an FIR filter.

    Parameters
    ----------
    x : ndarray
        Input signal array to be processed.
    y : ndarray
        Reference signal array that the input signal is compared against.
    Ntaps : int, optional
        Number of taps (coefficients) for the filter, by default 5.
    μ : float, optional
        Step size (learning rate) for the gradient adaptation, by default 1e-3.
    epochs : int, optional
        Number of epochs (iterations) over the entire signal, by default 2.

    Returns
    -------
    out : ndarray
        Array containing the estimated signal after filtering.
    H : ndarray
        Array containing the history of filter coefficients over time. 
        Shape is `(epochs * len(y) + 1, Ntaps)`.
    errors : ndarray
        Array containing the error between the estimated signal and the reference signal 
        at each step. Length is `epochs * len(y)`.

    Notes
    -----
    This function applies a gradient-based adaptation algorithm to update the coefficients
    of an FIR filter. The filter attempts to minimize the error between the estimated
    signal (`y_hat`) and the reference signal (`y`) by adjusting its coefficients based 
    on the gradient of the error with respect to the coefficients.

    The function can iterate multiple times (`epochs`) over the entire input signal to 
    refine the filter coefficients.
    
    The filter coefficients are updated using the rule:

        `h[k] = h[k] + μ * error * x_vec[k]`
    
    where `h` are the filter coefficients, `μ` is the learning rate, `error` is the difference
    between the reference signal and the estimated signal, and `x_vec[k]` are the delayed 
    versions of the input signal.
    """
    # Initialize filter coefficients
    h = np.zeros(Ntaps)       
    L = Ntaps//2
    N = len(y)

    # Gradient adapation algorithm
    errors = np.zeros(epochs*N)
    out  = np.zeros(y.shape)
    ind = np.arange(0,Ntaps)
    H = np.zeros((epochs*len(y)+1, Ntaps))

    x = np.pad(x,(Ntaps-1,0))

    # Iterate through each sample of the signal
    for epoch in range(epochs):
        for i in range(0, len(x)-Ntaps):
            x_vec = x[i+ind][-1::-1] # x[i-k]

            # Generate the estimated signal using the equalizer filter
            y_hat = np.sum(h*x_vec)

            # Compute the error between the estimated signal and the reference signal
            error = y[i] - y_hat

            # Update the filter coefficients using the gradient update rule
            h += μ * error * x_vec  
            
            H[i + epoch*N + 1,:] = h
            errors[i + epoch*N] = error
            out[i] = y_hat       
    
    return out, H, errors

plt.ion()

if __name__ == "__main__":
    
    periods = 100   
    T = 16     # period of the square wave in samples
    Nsamples = periods*T
    n = np.arange(Nsamples)

    # Generate the square wave
    #x = 1 * (n % T < T // 2) - 1 * (n % T >= T // 2)

    x = random_square_signal(Nsamples, T, 0.5)

    L = 2*T
    g = np.exp(-0.8*n)
    g = g[0:L]
    g = g/g.sum()

    plt.figure()
    plt.plot(g,'-o')

    plt.figure()
    y = np.convolve(g, x, mode='full')[:-L+1]
    plt.plot(x,'-o')
    plt.plot(y,'-o')   
    
    out, h, errors = gradientAdaptation(x, y, Ntaps=10, μ=2e-2)
    
    plt.figure()
    plt.plot(h,'-o')
    plt.plot(g,'-x')

    plt.figure()
    #plt.plot(errors)
    plt.plot(out,'-o')
    plt.plot(y,'-o')

    plt.show()