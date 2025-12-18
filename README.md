# Adaptive Signal Processing

This repository contains comprehensive course materials for the Adaptive Signal Processing course from the Graduate Program in Electrical Engineering at the Federal University of Campina Grande (UFCG), Brazil.

<img class="center" src="https://github.com/edsonportosilva/AdaptiveSignalProcessing/blob/main/notebooks/figures/capa.png" width="800">

## 📚 Course Content

The course covers fundamental and advanced topics in adaptive signal processing through interactive Jupyter notebooks with theory, Python implementations, and visualizations:

### Topics Covered

1. **Introduction to Adaptive Signal Processing** - Fundamental concepts and applications
2. **Introduction to Adaptive Filtering** - Basic principles and filter structures
3. **Review of Probability and Stochastic Processes** - Mathematical foundations
4. **The Wiener Filter and the LMS Algorithm** - Optimal filtering and least mean squares
5. **Variants of the LMS Algorithm** - Normalized LMS, Sign LMS, and other adaptations
6. **RLS Algorithms** - Recursive Least Squares methods
7. **Kalman Filters** - State-space filtering and estimation
8. **MLP Neural Networks** - Multi-layer perceptrons for adaptive processing
9. **Blind Adaptive Filtering** - Adaptive methods without reference signals

## 📁 Repository Structure

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

## 🚀 Getting Started

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

## 💡 How to Use This Course

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

## 🔧 Key Features

- **Interactive Learning**: Jupyter notebooks with executable code and visualizations
- **Complete Implementations**: Production-ready implementations of classic adaptive algorithms (LMS, NLMS, RLS, Kalman, etc.)
- **Visual Demonstrations**: Animated GIFs and plots showing algorithm behavior
- **Mathematical Rigor**: Detailed derivations with SymPy for symbolic mathematics
- **Performance Optimized**: Uses Numba JIT compilation for computational efficiency
- **Bilingual Support**: Code comments in English, slides available in Portuguese

## 📖 Additional Resources

- **Python Utilities** (`notebooks/algorithms.py`): Contains implementations of:
  - LMS and variants (NLMS, Sign-Error LMS, Sign-Regressor LMS)
  - RLS algorithms
  - Correlation matrix estimation
  - And more adaptive filtering algorithms

- **Visualization Tools** (`notebooks/utils.py`): Functions for:
  - Plot styling and preferences
  - Animation generation
  - Mathematical expression rendering

## 📝 License

This work is licensed under CC0 1.0 Universal (Public Domain). Feel free to use, modify, and distribute for educational and research purposes.

## 👨‍🏫 Author

**Edson Porto Silva**  
Graduate Program in Electrical Engineering  
Federal University of Campina Grande (UFCG)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## ⭐ Acknowledgments

This material has been developed to support graduate students in understanding and implementing adaptive signal processing techniques. If you find this repository useful, please consider giving it a star!
