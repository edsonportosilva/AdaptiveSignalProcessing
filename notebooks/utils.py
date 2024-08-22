import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import Math, Latex, display
import sympy as sp

def set_preferences():

    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)

    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (5,2.5)

    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['font.size'] = 10

    plt.rcParams['axes.linewidth'] =  0.5
    plt.rcParams['grid.linewidth'] =  0.5
    plt.rcParams['lines.linewidth'] =  0.5
    plt.rcParams['lines.markersize'] =  2
                    
    # Grid lines
    plt.rcParams['axes.grid'] =   False
    plt.rcParams['axes.axisbelow'] =  False
    plt.rcParams['grid.linestyle'] =  'dashed'
    plt.rcParams['grid.color'] =   'k'
    plt.rcParams['grid.alpha'] =   0.25
    plt.rcParams['grid.linewidth'] =   0.5

    # Legend
    plt.rcParams['legend.frameon'] =   False
    plt.rcParams['legend.framealpha'] =   0.25
    plt.rcParams['legend.fancybox'] =   False
    plt.rcParams['legend.numpoints'] =   1

    return None

def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res
    

def discreteConvolution(x, h, steps, D=1):
    """
    Perform discrete convolution between input signal x and impulse response h.

    Parameters
    ----------
    x : array-like
        Input signal.
    h : array-like
        Impulse response.
    steps : int
        Number of steps to perform the convolution.
    D : int, optional
        Delay parameter (default=1).

    Returns
    -------
    array
        Convolved signal.

    """
    x = np.flip(x)
    y = []
    for ind in range(1, steps + 1):
        y.append(np.dot(h, roll_zeropad(x, -(D - ind))))

    return np.array(y)
def discreteConvolution(x, h, steps, D=1):  
    x = np.flip(x)   
    y = []
    for ind in range(1,steps+1):
        y.append(np.dot(h, roll_zeropad(x, -(D-ind))))

    return np.array(y)


def genConvGIF(
    x,
    h,
    nStart,
    nEnd,
    figName,
    xlabel=[],
    ylabel=[],
    inter=20,
    writer=None,):
    """
    Create and save a discrete convolution plot animation as GIF

    :param x: x[n] signal
    :param h: h[n] signal 
    :param nInterval: array of time instants where the functions will be evaluated [nparray]
    :param nStart: time when animation starts [scalar]
    :param nEnd: time when animation stops [scalar]
    :param figName: figure file name w/ folder path [string]
    :param xlabel: xlabel [string]
    :param ylabel: ylabel [string]
    :param fram: number of frames [int]
    :param inter: time interval between frames [milliseconds]

    """ 
    nInterval = np.arange(nStart, nEnd)
    padRight_x = nEnd - len(x)
    padRight_h = nEnd - len(h)

    k = np.arange(nStart, nEnd)
    x = np.pad(x, (np.abs(nStart), padRight_x), "constant")
    h = np.pad(h, (np.abs(nStart), padRight_h), "constant")

    y = discreteConvolution(x, h, len(k), D=nEnd)

    ymax = np.max([x, h, y])
    ymin = np.min([x, h, y])

    figAnim = plt.figure()    
    ax = plt.axes(
        xlim=(nInterval.min(), nInterval.max()),
        ylim=(ymin - 0.1 * np.abs(ymax), ymax + 0.1 * np.abs(ymax)),
    )


    # Initial stem plots
    markerline1, stemlines1, baseline1 = ax.stem(k, h, 'k', basefmt=" ", label=ylabel[1], markerfmt='o')
    markerline2, stemlines2, baseline2 = ax.stem(k, np.zeros(h.shape), 'b', basefmt=" ", label=ylabel[0], markerfmt='s')
    markerline3, stemlines3, baseline3 = ax.stem(k, np.zeros(h.shape), 'r', basefmt=" ", label=ylabel[2], markerfmt='D')
    
    markerline1.set_markersize(4)
    stemlines1.set_linewidth(1)

    markerline2.set_markersize(4)
    stemlines2.set_linewidth(1)
    
    markerline3.set_markersize(4)
    stemlines3.set_linewidth(1)

    ax.grid()
    ax.legend(loc="upper right")

    if len(xlabel):
        ax.set_xlabel(xlabel)


    def init():
        # Initialize the animation with empty data
        markerline2.set_ydata(np.nan * np.ones(len(k)))
        markerline3.set_ydata(np.nan * np.ones(len(k)))

        # stemlines
        stemlines2.set_paths([np.array([[xx, 0], 
                                   [xx, yy]]) for (xx, yy) in zip(k, np.nan * np.ones(len(k)))])
        
        stemlines3.set_paths([np.array([[xx, 0], 
                                   [xx, yy]]) for (xx, yy) in zip(k, np.nan * np.ones(len(k)))])

        return markerline2, markerline3, stemlines2, stemlines3
   
    delays = nInterval

    totalFrames = len(delays)



    def animate(i): 
        
        # Update the data for each stem plot      
        y2_updated = roll_zeropad(np.flip(x), -(nEnd - i))
        y3_updated = y[:i]        
        
        markerline2.set_ydata(y2_updated)        
        markerline3.set_ydata(y3_updated)
        markerline3.set_xdata(k[:i])
        
         # stemlines
        stemlines2.set_paths([np.array([[xx, 0], 
                                   [xx, yy]]) for (xx, yy) in zip(k, y2_updated)])
        
        stemlines3.set_paths([np.array([[xx, 0], 
                                   [xx, yy]]) for (xx, yy) in zip(k[:i], y3_updated)])        
        
        if i > 0:
            plt.title(f"$y[{k[i-1]}] =\sum_k h[k]x[{k[i-1]}-k] = {y[i-1]:.2f}$")
        else:
            plt.title(f"$y[n] =\sum_k h[k]x[n-k]$")
        plt.tight_layout()

        return markerline2, markerline3, stemlines2, stemlines3

    anim = FuncAnimation(
        figAnim,
        animate,
        init_func=init,
        frames=totalFrames,
        interval=inter,
        blit=True,
    )

    if writer is None:
        anim.save(figName, dpi=200)
    else:
        anim.save(figName, dpi=200, writer=writer)

    plt.close()

    return None

def genTapsUpdateGIF(H, figName, period=1, reference=None, xlabel=[], ylabel=[], inter=20, writer=None):
    """
    Create and save a discrete convolution plot animation as GIF

    :param x: x[n] signal
    :param h: h[n] signal 
    :param nInterval: array of time instants where the functions will be evaluated [nparray]
    :param nStart: time when animation starts [scalar]
    :param nEnd: time when animation stops [scalar]
    :param figName: figure file name w/ folder path [string]
    :param xlabel: xlabel [string]
    :param ylabel: ylabel [string]
    :param fram: number of frames [int]
    :param inter: time interval between frames [milliseconds]

    """
    H = H[0::period, :]
    h = H[0,:]
    k = np.arange(0, len(h))

    if reference is None:
        ymax = np.max(H)
        k = np.arange(0, len(h))
    else:
        ymax = np.max(reference)
        k = np.arange(0, np.max([len(h), len(reference)]))

    ymin = 0
    figAnim = plt.figure()
    ax = plt.axes(
            xlim=(-1, k.max()+1),
            ylim=(ymin - 0.1 * np.abs(ymax), ymax + 0.1 * np.abs(ymax)),
        )
    if reference is not None:
        markerline_ref, stemlines_ref, baseline_ref = ax.stem(reference, 'r', basefmt=" ", label='reference: $g[k]$', markerfmt='s')    
        markerline_ref.set_markersize(4)
        stemlines_ref.set_linewidth(1)

    # Initial stem plot
    markerline, stemlines, baseline = ax.stem(0*h, 'k', basefmt=" ", label='adaptive filter: $h[k]$', markerfmt='o')    
    markerline.set_markersize(4)
    stemlines.set_linewidth(1)    
    plt.xlabel(xlabel)
    plt.grid()
    plt.legend(loc="upper right")
          
    def init():
        # Initialize the animation with empty data
        markerline.set_ydata(0 * np.ones(len(k)))
        
        # stemlines
        stemlines.set_paths([np.array([[xx, 0], 
                                   [xx, yy]]) for (xx, yy) in zip(k, 0 * np.ones(len(k)))])        
        

        return markerline, stemlines
      
    totalFrames = H.shape[0]

    def animate(i):        
        # Update the data of stem plot                    
        markerline.set_ydata(H[i,:])        
        
        # stemlines
        stemlines.set_paths([np.array([[xx, 0], 
                                   [xx, yy]]) for (xx, yy) in zip(k, H[i,:])])
        
        plt.title(f"iteration {i*period}: $h[k]$ ={str(np.round(H[i,:],3))}")
        plt.tight_layout()

        return markerline, stemlines

    anim = FuncAnimation(
        figAnim,
        animate,
        init_func=init,
        frames=totalFrames,
        interval=inter,
        blit=True,
    )

    if writer is None:
        anim.save(figName, dpi=200)
    else:
        anim.save(figName, dpi=200, writer=writer)

    plt.close()

    return None

def symdisp(expr, var=None, unit=None, numDig=None):
    """
    Display sympy expressions in Latex style.

    :param expr: The expression to be displayed in Latex style. It can be a string or a sympy expression.
    :param var: The sympy variable, function, or expression associated with the expression. (default: None)
    :param unit: The unit of the variable. (default: None)
    :param numDig: The number of digits to round the expression to. (default: None)
    """
    if unit is None:
        unit = " "
    if numDig is None:
        numDig = 8

    if var is None:
        display(Math(expr + "\mathrm{" + unit + "}"))
    else:
        display(Math(expr + sp.latex(round_expr(var, numDig)) + "\;" + "\mathrm{" + unit + "}"))

        
def round_expr(expr, numDig):
    """
    Rounds numerical values in sympy expressions

    :param expr: sympy symbolic expression
    :param numDig: number of rounding decimals

    :return: rounded expression
    """
    return expr.xreplace({n: round(n, numDig) for n in expr.atoms(sp.Number)})

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