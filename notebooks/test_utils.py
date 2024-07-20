import numpy as np
from utils import genConvGIF, set_preferences
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee'])

def test_genConvGIF():
    x = np.ones([5])/5
    h = np.exp(-0.5*np.arange(0,8))    
    nStart = -10
    nEnd = 25
    figName = "conv_animation.gif"
    xlabel = "n"
    ylabel = ["x[n-k]", "h[k]", "y[n]"]
    fram = 200
    inter = 800
    plotConv = True

    genConvGIF(x, h, nStart, nEnd, figName, xlabel, ylabel, fram, inter, plotConv)

        # Add assertions to check if the GIF file is created successfully or any other desired behavior

plt.ion()

if __name__ == "__main__":
    set_preferences()
    test_genConvGIF()