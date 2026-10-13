Theory
======

The chapters follow the order of the course. Each one starts with its
learning objectives, develops the algorithm from first principles, shows a
figure generated with the code of this package, and ends with exercises
(click on *Solution* to check your answer).

.. toctree::
   :maxdepth: 1

   wiener
   optimization
   lms
   lms_variants
   rls
   kalman
   blind
   neural

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symbol
     - Meaning
   * - :math:`x[k]`, :math:`d[k]`, :math:`y[k]`
     - input, desired and output signals at instant :math:`k`
   * - :math:`e[k] = d[k]-y[k]`
     - error
   * - :math:`\mathbf{x}_k=[x[k],\dots,x[k-N+1]]^T`
     - regressor (contents of the delay line), :math:`N` = ``Ntaps``
   * - :math:`\mathbf{h}_k`
     - coefficient vector of the adaptive filter at instant :math:`k`
   * - :math:`\mathbf{h}_o`
     - optimal (Wiener) coefficient vector
   * - :math:`\mathbf{R}=\operatorname{E}[\mathbf{x}_k\mathbf{x}_k^T]`
     - correlation matrix of the regressor
   * - :math:`\mathbf{p}=\operatorname{E}[d[k]\mathbf{x}_k]`
     - cross-correlation vector
   * - :math:`\mu`
     - step size
   * - :math:`\lambda_i`
     - eigenvalues of :math:`\mathbf{R}`; :math:`\lambda_{\max}/\lambda_{\min}`
       is the *eigenvalue spread*
   * - :math:`\lambda`
     - forgetting factor of RLS (not to be confused with an eigenvalue)

All signals are real-valued unless stated otherwise, and vectors are columns.
