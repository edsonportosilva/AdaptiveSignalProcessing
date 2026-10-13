Optimization of objective functions
===================================

.. admonition:: Learning objectives
   :class: objectives

   * Explain why iterative optimization is preferred to solving the
     Wiener-Hopf equations directly.
   * Derive the gradient-descent and Newton updates, and the stability
     condition for gradient descent on a quadratic cost.
   * Relate the eigenvalue spread of :math:`\mathbf{R}` to the convergence
     speed.
   * Run both methods with
     :func:`~asp.algorithms.gradient_descent` and
     :func:`~asp.algorithms.newton_method`.

Why iterate?
------------

The Wiener solution :math:`\mathbf{h}_o=\mathbf{R}^{-1}\mathbf{p}` is one step
away, but it needs :math:`O(N^3)` operations and perfect knowledge of
:math:`\mathbf{R}` and :math:`\mathbf{p}`. Iterative methods start from a
guess and improve it, and — more importantly for us — they can *keep
improving* while the data arrive, which is what adaptation means.

Gradient descent (steepest descent)
-----------------------------------

The gradient of a function points in the direction of fastest increase.
Moving against it decreases the function:

.. math::
   :label: gd

   \mathbf{x}_{k+1}=\mathbf{x}_k-\mu\,\nabla J(\mathbf{x}_k).

For the MSE cost :math:`\nabla J=2(\mathbf{R}\mathbf{h}-\mathbf{p})`, so with
the error vector :math:`\mathbf{v}_k=\mathbf{h}_k-\mathbf{h}_o`,

.. math::

   \mathbf{v}_{k+1}=(\mathbf{I}-2\mu\mathbf{R})\,\mathbf{v}_k .

Decomposing along the eigenvectors of :math:`\mathbf{R}`, each mode evolves
independently,

.. math::

   \tilde v_i[k]=(1-2\mu\lambda_i)^k\,\tilde v_i[0].

.. admonition:: Key idea
   :class: keyidea

   *Stability:* every mode decays if :math:`|1-2\mu\lambda_i|<1`, i.e.

   .. math::

      0<\mu<\frac{1}{\lambda_{\max}} .

   *Speed:* mode :math:`i` has time constant
   :math:`\tau_i\approx 1/(2\mu\lambda_i)` samples. The step size is limited by
   the *largest* eigenvalue, but the convergence time is determined by the
   *smallest*. The overall time is therefore proportional to the eigenvalue
   spread :math:`\lambda_{\max}/\lambda_{\min}`.

Newton's method
---------------

Newton's method uses the curvature (Hessian :math:`\mathbf{H}`) of the cost
to rescale the gradient:

.. math::
   :label: newton

   \mathbf{x}_{k+1}=\mathbf{x}_k-\mu\,\mathbf{H}^{-1}\nabla J(\mathbf{x}_k).

For the MSE cost :math:`\mathbf{H}=2\mathbf{R}`; with :math:`\mu=1` a
single step lands exactly on :math:`\mathbf{h}_o`, whatever the eigenvalue
spread. The price is the need for :math:`\mathbf{R}^{-1}`. Chapters
:doc:`lms_variants` and :doc:`rls` show how to approximate it from the data.

.. plot::
   :caption: Gradient descent (red) and Newton's method (blue) on a
             quadratic function with eigenvalue spread 20. Gradient descent
             zig-zags along the narrow valley; Newton goes straight to the
             minimum in one step.

   A = np.array([[10.0, 0.0], [0.0, 0.5]])
   grad = lambda v: A @ v
   hess = lambda v: A

   _, p_gd = alg.gradient_descent(grad, [1.0, 2.0], μ=0.18, max_iter=60, tol=1e-12)
   _, p_nw = alg.newton_method(grad, hess, [1.0, 2.0], μ=1.0, max_iter=5, tol=1e-12)

   a, b = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-0.5, 2.3, 200))
   J = 0.5 * (A[0, 0] * a**2 + A[1, 1] * b**2)
   fig, ax = plt.subplots(figsize=(5.5, 3.6))
   ax.contour(a, b, J, 25, cmap='Greys')
   ax.plot(*p_gd.T, 'r.-', label='gradient descent (60 it.)')
   ax.plot(*p_nw.T, 'bo-', label='Newton (1 it.)')
   ax.legend(); ax.set(xlabel='$x_1$', ylabel='$x_2$')

Gradient adaptation of an FIR filter
------------------------------------

Inside an adaptive filter, the exact gradient is replaced by a quantity that
can be computed from the available samples. The simplest is the gradient of
the *instantaneous* squared error :math:`e^2[k]`, which is
:math:`-2e[k]\mathbf{x}_k`. This gives

.. math::

   \mathbf{h}_{k+1}=\mathbf{h}_k+\mu\,e[k]\,\mathbf{x}_k,

implemented (with several passes over a recorded signal) by
:func:`~asp.algorithms.gradient_adaptation`. The next
chapter studies this rule — the LMS algorithm — in detail.

Exercises
---------

.. admonition:: Exercise
   :class: exercise

   A white input of power :math:`\sigma_x^2=1` feeds a filter with
   :math:`N=10` coefficients. Give a safe upper bound for the step size of
   gradient descent without computing eigenvalues.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

For a white input :math:`\mathbf{R}=\sigma_x^2\mathbf{I}` and every
eigenvalue is 1, so :math:`\mu<1`. In general
:math:`\lambda_{\max}\le\operatorname{tr}\mathbf{R}=N\sigma_x^2=10`, hence
:math:`\mu<1/(N\sigma_x^2)=0.1` is a safe, conservative bound valid for any
input with that power.

.. raw:: html

   </details>

.. seealso::

   :func:`~asp.algorithms.gradient_descent`,
   :func:`~asp.algorithms.newton_method`,
   :func:`~asp.algorithms.gradient_adaptation`,
   notebook 2.
