import matplotlib.pyplot as plt
from matplotlib import rc

def set_preferences():

    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)

    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['figure.figsize'] = (2.5,2.5)

    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['font.size'] = 7

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
