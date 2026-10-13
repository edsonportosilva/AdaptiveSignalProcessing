Installation
============

The package requires Python 3.9 or newer.

From a clone of the repository
------------------------------

.. code-block:: bash

   git clone https://github.com/edsonportosilva/AdaptiveSignalProcessing.git
   cd AdaptiveSignalProcessing
   pip install -e .

The ``-e`` (editable) flag lets you modify the source and use the changes
immediately. Optional dependency groups are available:

.. code-block:: bash

   pip install -e ".[notebooks]"   # JupyterLab + SciencePlots, to run the notebooks
   pip install -e ".[test]"        # pytest
   pip install -e ".[docs]"        # Sphinx + Furo theme, to build this site
   pip install -e ".[dev]"         # everything above except notebooks

Quick check
-----------

.. code-block:: python

   >>> import numpy as np
   >>> from asp.algorithms import lms
   >>> x = np.random.randn(2000)
   >>> d = np.convolve(x, [1.0, 0.5])[:2000]
   >>> out, h, sq_err, H = lms(x, d, 2, 0.01)
   >>> h.round(2)
   array([1. , 0.5])

.. note::

   The first call of each ``numba``-accelerated function triggers a
   just-in-time compilation that takes a few seconds; later calls are fast.

Running the tests
-----------------

.. code-block:: bash

   pytest

The suite checks every algorithm against reference implementations, exact
closed-form solutions (e.g. weighted least squares for RLS), finite-difference
gradients (for the neural network) and convergence properties.

Building this documentation
---------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html            # on Windows: make.bat html

and open ``docs/_build/html/index.html`` in a browser. The figures are
generated while the site is built.

Notebooks
---------

The notebooks live in the ``notebooks/`` directory and import the package as
``asp``. After installing it, launch Jupyter from the
repository root:

.. code-block:: bash

   jupyter lab notebooks/
