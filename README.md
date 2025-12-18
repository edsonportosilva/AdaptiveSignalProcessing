# Adaptive Signal Processing

This repository contains comprehensive course materials for the Adaptive Signal Processing course from the Graduate Program in Electrical Engineering at the Federal University of Campina Grande (UFCG), Brazil.

<img class="center" src="https://github.com/edsonportosilva/AdaptiveSignalProcessing/blob/main/notebooks/figures/capa.png" width="800">

## Course Content

The course covers fundamental and advanced topics in adaptive signal processing through interactive Jupyter notebooks with theory, Python implementations, and visualizations.

### Topics Covered

The course is organized in nine Jupyter notebooks:

1. **Introduction to Adaptive Signal Processing** - Applications of adaptive signal processing and review of basic concepts of digital signal processing (sampling theorem, discrete-time convolution, z-transform, DTFT, and DFT)

2. **Introduction to Adaptive Filtering** - General structure of adaptive filters, system identification, linear prediction, interference cancellation, and inverse modeling

3. **Review of Probability and Stochastic Processes** - Random variables, probability distributions, expected values, correlation functions, and stationarity

4. **The Wiener Filter and the LMS Algorithm** - Optimal linear filtering in the mean-square sense, Wiener-Hopf equations, gradient descent optimization, and the Least Mean Squares (LMS) algorithm

5. **Variants of the LMS Algorithm** - Normalized LMS (NLMS), LMS-Newton algorithm, and convergence analysis

6. **RLS Algorithms** - Recursive Least Squares algorithm, forgetting factor, and comparison with LMS methods

7. **Kalman Filters** - State-space representation, Kalman filtering for linear systems, prediction and update steps

8. **MLP Neural Networks** - Multi-layer perceptrons, backpropagation algorithm, and applications in adaptive signal processing

9. **Blind Adaptive Filtering** - Adaptive equalization without training sequences, constant modulus algorithm (CMA), and other blind adaptation techniques

## Repository Structure

```
AdaptiveSignalProcessing/
├── notebooks/              # Jupyter notebooks with course materials
│   ├── *.ipynb            # 9 interactive notebooks covering all topics
│   ├── algorithms.py      # Adaptive filtering algorithm implementations
│   ├── utils.py           # Utility functions for visualization and processing
│   └── figures/           # Images and diagrams used in notebooks
├── python scripts/        # Additional Python scripts
│   └── gradient_adaptation.py
├── slides pt-br/          # Course slides in Portuguese
└── README.md              # This file
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Basic knowledge of signal processing and linear algebra
- Familiarity with Python and NumPy

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/edsonportosilva/AdaptiveSignalProcessing.git
   cd AdaptiveSignalProcessing
   ```

2. **Install required packages:**
   
   The notebooks require the following Python packages:
   - NumPy
   - Matplotlib
   - SciPy
   - SymPy
   - Numba
   - SciencePlots (for IEEE-style plots)
   
   Install them using pip:
   ```bash
   pip install numpy matplotlib scipy sympy numba scienceplots
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   Navigate to the `notebooks/` directory and open any notebook to start learning.

### Using Google Colab

The notebooks are designed to work seamlessly with Google Colab. Simply:

1. Open any notebook in GitHub
2. Click the "Open in Colab" button (or manually open in Colab)
3. The notebook will automatically clone the repository and install dependencies

## How to Use This Course

### For Students

1. **Follow the Sequential Order**: The notebooks are numbered and build upon each other. Start with notebook 1 and progress sequentially.

2. **Run the Code**: Execute all code cells to see the algorithms in action. Experiment by modifying parameters to develop intuition.

3. **Study the Theory**: Each notebook contains mathematical derivations, explanations, and visualizations to help you understand the concepts.

4. **Explore the Utilities**: Check out `algorithms.py` and `utils.py` to see implementations of adaptive filtering algorithms and helper functions.

5. **Consult the Slides**: Portuguese speakers can refer to the PDF slides in `slides pt-br/` for additional explanations.

### For Instructors

- The notebooks can be used directly in lectures or assigned as homework
- Each notebook is self-contained with theory and practical examples
- The modular structure allows selecting specific topics as needed
- Python implementations provide a foundation for student projects

## Key Features

- **Interactive Learning**: Jupyter notebooks with executable code and visualizations
- **Complete Implementations**: Implementations of classic adaptive algorithms with Numba JIT compilation for computational efficiency
- **Visual Demonstrations**: Animated GIFs and plots showing algorithm behavior
- **Mathematical Rigor**: Detailed derivations with SymPy for symbolic mathematics
- **Bilingual Support**: Code and notebooks in English, lecture slides available in Portuguese

## Python Modules

### `algorithms.py`

This module provides optimized implementations of adaptive filtering algorithms using Numba's just-in-time compilation:

- **`estimate_correlation_matrix(x, N)`**: Estimates the unbiased autocorrelation matrix from a signal sequence
- **`estimate_cross_correlation(x, d, N)`**: Estimates the unbiased cross-correlation vector between input and desired signals
- **`lms(x, d, Ntaps, μ)`**: Least Mean Squares adaptive filter with step size μ
- **`nlms(x, d, Ntaps, μ, γ)`**: Normalized Least Mean Squares algorithm with step size μ and regularization parameter γ
- **`lms_newton(x, d, Ntaps, μ, α)`**: LMS-Newton algorithm with step size μ and inverse correlation matrix update parameter α
- **`rls(x, d, Ntaps, λ)`**: Recursive Least Squares algorithm with forgetting factor λ
- **`kalman_filter(A, C, Rn, Rv, x_init, y)`**: Kalman filter for state estimation in linear systems
- **`time_varying_filter(x, H)`**: Applies a time-varying filter with coefficient evolution matrix H

All adaptive filtering functions return the filtered output, final filter coefficients, squared error history, and the evolution of filter coefficients over time.

### `utils.py`

This module provides visualization and utility functions for the notebooks:

- **`set_preferences()`**: Configures matplotlib plotting parameters for consistent figure styling
- **`roll_zeropad(a, shift, axis)`**: Rolls array elements with zero-padding
- **`discreteConvolution(x, h, steps, D)`**: Computes discrete convolution with visualization support
- **`genConvGIF()`**: Generates animated GIF visualizations of the convolution process
- **`genTapsUpdateGIF(H, figName, ...)`**: Creates animated visualizations of adaptive filter coefficient evolution
- **`symdisp(expr, var, unit, numDig)`**: Displays symbolic mathematical expressions with proper formatting
- **`round_expr(expr, numDig)`**: Rounds symbolic expressions to specified precision
- **`random_square_signal(num_samples, period, duty_cycle)`**: Generates random square wave signals for testing

## License

This work is licensed under CC0 1.0 Universal (Public Domain). Feel free to use, modify, and distribute for educational and research purposes.

## Author

**Edson Porto Silva**  
Graduate Program in Electrical Engineering  
Federal University of Campina Grande (UFCG)

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## Acknowledgments

This material has been developed to support graduate students in understanding and implementing adaptive signal processing techniques.
