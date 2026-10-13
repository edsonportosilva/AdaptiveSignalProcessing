Companion notebooks
===================

Each chapter of the :doc:`theory <theory/index>` has a Jupyter notebook with
simulations, animations and exercises. They are in the ``notebooks/``
directory of the repository.

.. list-table::
   :header-rows: 1
   :widths: 5 35 35 25
   :class: algorithm-table

   * - #
     - Notebook
     - Theory chapter
     - Main functions
   * - 1
     - Introduction to adaptive signal processing
     - Review of DSP (sampling, convolution, z-transform, DFT)
     - ``utils.genConvGIF``
   * - 2
     - Introduction to adaptive filtering
     - :doc:`theory/optimization`
     - :func:`~asp.algorithms.gradient_descent`,
       :func:`~asp.algorithms.newton_method`,
       :func:`~asp.algorithms.gradient_adaptation`
   * - 3
     - Probability and stochastic processes
     - :doc:`theory/wiener` (statistical background)
     - —
   * - 4
     - The Wiener filter and the LMS algorithm
     - :doc:`theory/wiener`, :doc:`theory/lms`
     - :func:`~asp.algorithms.estimate_correlation_matrix`,
       :func:`~asp.algorithms.lms`
   * - 5
     - Variants of the LMS algorithm
     - :doc:`theory/lms_variants`
     - :func:`~asp.algorithms.nlms`,
       :func:`~asp.algorithms.lms_newton`,
       :func:`~asp.algorithms.tdlms`,
       :func:`~asp.algorithms.fdlms`
   * - 6
     - RLS algorithms
     - :doc:`theory/rls`
     - :func:`~asp.algorithms.rls`,
       :func:`~asp.algorithms.rls_apriori`,
       :func:`~asp.algorithms.rls_aposteriori`
   * - 7
     - Kalman filters
     - :doc:`theory/kalman`
     - :func:`~asp.algorithms.kalman_filter`,
       :func:`~asp.algorithms.extended_kalman_filter`
   * - 8
     - MLP neural networks
     - :doc:`theory/neural`
     - :func:`~asp.algorithms.initialize_nn`,
       :func:`~asp.algorithms.fit`,
       :func:`~asp.algorithms.predict`
   * - 9
     - Blind adaptive filtering
     - :doc:`theory/blind`
     - :func:`~asp.algorithms.cma`

.. note::

   Some notebooks keep their own, didactic copy of an algorithm (written
   cell by cell, as it is introduced in the text). The versions in
   :mod:`asp.algorithms` are the maintained, tested
   reference implementations and may differ in small details; see the
   *Notes* of each function.
