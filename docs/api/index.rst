API reference
=============

.. note::

   Conventions shared by all adaptive filters (regressor definition, output
   and error, layout of the coefficient history ``H``) are described in the
   module docstring below.

``asp.algorithms``
-----------------------------------------

.. automodule:: asp.algorithms
   :no-members:

.. rubric:: Statistical estimation

.. autofunction:: asp.algorithms.estimate_correlation_matrix
.. autofunction:: asp.algorithms.estimate_cross_correlation

.. rubric:: Optimization of objective functions

.. autofunction:: asp.algorithms.gradient_descent
.. autofunction:: asp.algorithms.newton_method
.. autofunction:: asp.algorithms.gradient_adaptation

.. rubric:: LMS family

.. autofunction:: asp.algorithms.lms
.. autofunction:: asp.algorithms.nlms
.. autofunction:: asp.algorithms.lms_newton
.. autofunction:: asp.algorithms.tdlms
.. autofunction:: asp.algorithms.fdlms

.. rubric:: RLS family

.. autofunction:: asp.algorithms.rls
.. autofunction:: asp.algorithms.rls_apriori
.. autofunction:: asp.algorithms.rls_aposteriori

.. rubric:: Kalman filters

.. autofunction:: asp.algorithms.kalman_filter
.. autofunction:: asp.algorithms.extended_kalman_filter

.. rubric:: Blind adaptation

.. autofunction:: asp.algorithms.cma

.. rubric:: Utilities

.. autofunction:: asp.algorithms.time_varying_filter
.. autofunction:: asp.algorithms.sliding_window_matrix

.. rubric:: Multi-layer perceptron

.. autofunction:: asp.algorithms.initialize_nn
.. autofunction:: asp.algorithms.fit
.. autofunction:: asp.algorithms.predict
.. autofunction:: asp.algorithms.sigmoid
.. autofunction:: asp.algorithms.dsigmoid
.. autofunction:: asp.algorithms.ReLU
.. autofunction:: asp.algorithms.dReLU
.. autofunction:: asp.algorithms.dtanh

``asp.utils``
------------------------------------

Helpers used by the notebooks: plotting preferences, animations of
convolution and coefficient adaptation, symbolic display, and test-signal
generators. They require ``matplotlib``, ``sympy`` and ``IPython``.

.. automodule:: asp.utils
   :members:
