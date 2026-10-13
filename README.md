# Adaptive Signal Processing

This repository contains the course materials for the Adaptive Signal Processing course of the Graduate Program in Electrical Engineering at the Federal University of Campina Grande (UFCG), Brazil: interactive notebooks, slides, and an installable Python package (`asp`) with tested reference implementations of the algorithms, documented with their underlying theory.

<img class="center" src="https://github.com/edsonportosilva/AdaptiveSignalProcessing/blob/main/notebooks/figures/capa.png" width="800">

## Course Content

The course covers fundamental and advanced topics in adaptive signal processing through interactive Jupyter notebooks with theory, Python implementations, and visualizations.

### Topics Covered

The course is organized in nine Jupyter notebooks:

1. **Introduction to Adaptive Signal Processing** - Applications of adaptive signal processing and review of basic concepts of digital signal processing (sampling theorem, discrete-time convolution, z-transform, DTFT, and DFT)

2. **Introduction to Adaptive Filtering** - General structure of adaptive filters, system identification, linear prediction, interference cancellation, inverse modeling, and the gradient/Newton optimization methods

3. **Review of Probability and Stochastic Processes** - Random variables, probability distributions, expected values, correlation functions, and stationarity

4. **The Wiener Filter and the LMS Algorithm** - Optimal linear filtering in the mean-square sense, Wiener-Hopf equations, gradient descent optimization, and the Least Mean Squares (LMS) algorithm

5. **Variants of the LMS Algorithm** - Normalized LMS (NLMS), LMS-Newton, transform-domain LMS, frequency-domain LMS, and convergence analysis

6. **RLS Algorithms** - Recursive Least Squares algorithm, forgetting factor, and comparison with LMS methods

7. **Kalman Filters** - State-space representation, Kalman filtering for linear systems, prediction and update steps, and the extended Kalman filter

8. **MLP Neural Networks** - Multi-layer perceptrons, backpropagation algorithm, and applications in adaptive signal processing

9. **Blind Adaptive Filtering** - Adaptive equalization without training sequences, constant modulus algorithm (CMA), and other blind adaptation techniques

## Repository Structure

```
AdaptiveSignalProcessing/
├── src/
│   └── asp/   # Installable Python package
│       ├── algorithms.py             # Adaptive filtering algorithms
│       └── utils.py                  # Plotting / animation / display helpers
├── tests/                            # pytest test suite
├── docs/                             # Sphinx documentation (theory + API reference)
├── notebooks/                        # Jupyter notebooks, figures and GIFs
│   ├── *.ipynb                       # 9 interactive notebooks covering all topics
│   └── figures/                      # Images and diagrams used in notebooks
├── python scripts/                   # Additional stand-alone scripts
├── slides pt-br/                     # Course slides in Portuguese
├── pyproject.toml                    # Package metadata and dependencies
└── README.md
```

## Getting Started

### Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/edsonportosilva/AdaptiveSignalProcessing.git
cd AdaptiveSignalProcessing
pip install -e .
```

The core dependencies (NumPy, SciPy, Numba, tqdm, Matplotlib, SymPy and IPython) are installed automatically. Optional extras:

| Extra | Installs | Use |
|-------|----------|-----|
| `notebooks` | JupyterLab, SciencePlots | run the notebooks |
| `test` | pytest | run the test suite |
| `docs` | Sphinx, Furo | build the documentation |
| `dev` | all of `test` and `docs` | development |

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

The notebooks import the package as `asp`, so install it before opening them. Rendering the figures with LaTeX fonts (`utils.set_preferences`) additionally requires a working LaTeX installation.

### Quick example

```python
import numpy as np
from asp.algorithms import lms, rls

rng = np.random.default_rng(0)
x = rng.standard_normal(5000)                 # input signal
g = np.array([1.0, 0.5, -0.3, 0.1])           # unknown system
d = np.convolve(x, g)[:len(x)]                # desired signal

out, h, squared_error, H = lms(x, d, Ntaps=4, μ=0.01)
print(h.round(2))                             # [ 1.   0.5 -0.3  0.1]
```

### Using Google Colab

The notebooks can be opened in Google Colab from GitHub. Install the package in the first cell with `!pip install git+https://github.com/edsonportosilva/AdaptiveSignalProcessing.git`.

## Documentation

The `docs/` folder contains a Sphinx project with a chapter of theory for each family of algorithms (with derivations, figures generated from the package, pitfalls, and exercises with solutions), a map between notebooks and chapters, and the API reference built from the docstrings. To build it locally:

```bash
pip install -e ".[docs]"
cd docs
make html            # on Windows: make.bat html
```

and open `docs/_build/html/index.html`.

## Python Modules

### `asp.algorithms`

Numba-accelerated implementations of the algorithms studied in the course. Every function documents the theory it is based on in the *Notes* section of its docstring.

| Topic | Function | Description |
|-------|----------|-------------|
| Statistical estimation | `estimate_correlation_matrix(x, N)` | Sample correlation matrix **R** of order N |
| | `estimate_cross_correlation(x, d, N)` | Sample cross-correlation vector **p** |
| Optimization | `gradient_descent(grad, x0, μ, ...)` | Steepest descent for generic objective functions |
| | `newton_method(grad, hess, x0, μ, ...)` | Newton's method |
| | `gradient_adaptation(x, y, Ntaps, μ, epochs)` | Stochastic gradient adaptation of an FIR filter over several epochs |
| LMS family | `lms(x, d, Ntaps, μ)` | Least Mean Squares |
| | `nlms(x, d, Ntaps, μ, γ)` | Normalized LMS |
| | `lms_newton(x, d, Ntaps, μ, α)` | LMS-Newton (recursive inverse correlation matrix) |
| | `tdlms(x, d, Ntaps, μ, M, γ, α)` | Transform-domain LMS |
| | `fdlms(x, d, Ntaps, μ, Nfft, constrained)` | Frequency-domain LMS (overlap-and-save) |
| RLS family | `rls(x, d, Ntaps, λ, δ)` | Recursive Least Squares |
| | `rls_apriori(x, d, Ntaps, λ, δ)` | RLS, a priori error formulation |
| | `rls_aposteriori(x, d, Ntaps, λ, δ)` | RLS, a posteriori error formulation |
| Kalman filtering | `kalman_filter(A, C, Rn, Rv, x_init, y, P_init)` | Kalman filter for linear systems |
| | `extended_kalman_filter(f, h, F_jac, H_jac, Rn, Rv, x_init, y, P_init)` | Extended Kalman filter for nonlinear systems |
| Blind adaptation | `cma(x, Ntaps, μ, R2)` | Constant Modulus Algorithm (real and complex signals) |
| Utilities | `time_varying_filter(x, H)` | FIR filtering with time-varying coefficients |
| | `sliding_window_matrix(x, N)` | Sliding windows of a signal as columns of a matrix |
| Neural networks | `initialize_nn(layers, seed)` | Weights and biases of an MLP |
| | `fit(X, Y, weights, biases, activation, learning_rate, epochs, batch_size)` | Back-propagation with mini-batch gradient descent |
| | `predict(X, weights, biases, activation)` | Forward propagation |
| | `sigmoid`, `dsigmoid`, `ReLU`, `dReLU`, `dtanh` | Activation functions and derivatives |

The adaptive filters (`lms`, `nlms`, `lms_newton`, `tdlms`, `rls*`, `cma`) return the filter output, the final coefficients, the (squared) error history, and the evolution `H` of the coefficients, with shape `(len(x), Ntaps)`, where row `k` holds the coefficients after the update at sample `k`.

### `asp.utils`

Helpers for visualization and processing used in the notebooks:

- **`set_preferences()`**: Configures matplotlib plotting parameters for consistent figure styling
- **`roll_zeropad(a, shift, axis)`**: Rolls array elements with zero-padding
- **`discreteConvolution(x, h, steps, D)`**: Computes discrete convolution with visualization support
- **`genConvGIF(...)`**: Generates animated GIF visualizations of the convolution process
- **`genTapsUpdateGIF(H, figName, ...)`**: Creates animated visualizations of adaptive filter coefficient evolution
- **`symdisp(expr, var, unit, numDig)`**: Displays symbolic mathematical expressions with proper formatting
- **`round_expr(expr, numDig)`**: Rounds symbolic expressions to specified precision
- **`random_square_signal(num_samples, period, duty_cycle)`**: Generates random square wave signals for testing

## Testing

```bash
pip install -e ".[test]"
pytest
```

The suite (`tests/`) checks each algorithm against reference implementations, closed-form solutions (e.g. the exponentially weighted least-squares solution for RLS), finite-difference gradients (neural network), and convergence behavior; the docstring examples are run as doctests.

## License

This work is licensed under CC0 1.0 Universal (Public Domain). Feel free to use, modify, and distribute for educational and research purposes.

## Author

**Edson Porto Silva**  
Graduate Program in Electrical Engineering  
Federal University of Campina Grande (UFCG)

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
