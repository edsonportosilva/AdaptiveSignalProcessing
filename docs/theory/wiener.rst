The Wiener filter
=================

.. admonition:: Learning objectives
   :class: objectives

   After this chapter you should be able to

   * state the *adaptive filtering problem* and its canonical configurations;
   * derive the mean-square error (MSE) surface and show that it is a
     paraboloid;
   * obtain the Wiener-Hopf equations and the minimum MSE;
   * estimate :math:`\mathbf{R}` and :math:`\mathbf{p}` from data with
     :func:`~asp.algorithms.estimate_correlation_matrix`
     and
     :func:`~asp.algorithms.estimate_cross_correlation`.

Where adaptive filters are used
-------------------------------

All applications share one structure: a filter with input :math:`x[k]`
produces :math:`y[k]`, which is compared with a desired signal :math:`d[k]`.
What changes is *where* :math:`x` and :math:`d` come from:

* **System identification** — :math:`x` is the input of an unknown system and
  :math:`d` its output; the filter becomes a model of the system.
* **Inverse modeling / equalization** — :math:`x` is the output of a channel
  and :math:`d` a (delayed) version of its input; the filter inverts the
  channel.
* **Prediction** — :math:`x[k]=d[k-\Delta]`: the filter predicts the signal
  from its past.
* **Interference cancellation** — :math:`x` is a reference correlated with the
  noise that corrupts :math:`d`; the error :math:`e` is the cleaned signal.

The mean-square error surface
-----------------------------

Consider a transversal (FIR) filter with :math:`N` coefficients,

.. math::

   y[k] = \sum_{i=0}^{N-1} h_i\,x[k-i] = \mathbf{h}^T\mathbf{x}_k .

We look for the :math:`\mathbf{h}` that minimizes the **mean-square error**

.. math::

   J(\mathbf{h}) = \operatorname{E}\left[e^2[k]\right]
   = \operatorname{E}\left[\left(d[k]-\mathbf{h}^T\mathbf{x}_k\right)^2\right].

Expanding the square and using the linearity of the expectation,

.. math::
   :label: mse-surface

   J(\mathbf{h}) = \sigma_d^2 - 2\,\mathbf{h}^T\mathbf{p}
   + \mathbf{h}^T\mathbf{R}\,\mathbf{h},

with

.. math::

   \mathbf{R}=\operatorname{E}\left[\mathbf{x}_k\mathbf{x}_k^T\right],\qquad
   \mathbf{p}=\operatorname{E}\left[d[k]\,\mathbf{x}_k\right],\qquad
   \sigma_d^2=\operatorname{E}\left[d^2[k]\right].

.. admonition:: Key idea
   :class: keyidea

   :math:`J(\mathbf{h})` is a **quadratic** function of the coefficients.
   Because :math:`\mathbf{R}` is positive (semi-)definite, the surface is a
   paraboloid with a single minimum: there are no local minima to get trapped
   in. All the algorithms of the course try to find this minimum — they only
   differ in *how much they know* (:math:`\mathbf{R}`, :math:`\mathbf{p}`) and
   *how much they compute*.

The Wiener-Hopf equations
-------------------------

The gradient of :eq:`mse-surface` is
:math:`\nabla J = -2\mathbf{p}+2\mathbf{R}\mathbf{h}`. Setting it to zero:

.. math::
   :label: wiener-hopf

   \mathbf{R}\,\mathbf{h}_o=\mathbf{p}
   \quad\Longrightarrow\quad
   \mathbf{h}_o=\mathbf{R}^{-1}\mathbf{p}.

This is the **Wiener filter**. Substituting back,

.. math::

   J_{\min}=\sigma_d^2-\mathbf{p}^T\mathbf{R}^{-1}\mathbf{p}
   =\sigma_d^2-\mathbf{h}_o^T\mathbf{p}.

Another way to read :eq:`wiener-hopf` is the **orthogonality principle**: at
the optimum, the error is uncorrelated with every sample in the regressor,

.. math::

   \operatorname{E}\left[e_o[k]\,x[k-i]\right]=0,\quad i=0,\dots,N-1 .

Writing :math:`J` around the optimum with the shifted coordinates
:math:`\mathbf{v}=\mathbf{h}-\mathbf{h}_o` gives

.. math::

   J(\mathbf{h})=J_{\min}+\mathbf{v}^T\mathbf{R}\,\mathbf{v}
   =J_{\min}+\sum_{i=0}^{N-1}\lambda_i\,\tilde v_i^2 ,

where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{R}` and
:math:`\tilde{\mathbf{v}}=\mathbf{Q}^T\mathbf{v}` (with the eigenvectors in
the columns of :math:`\mathbf{Q}`). The contour lines of :math:`J` are
ellipses whose axes are the eigenvectors; the *eigenvalue spread*
:math:`\lambda_{\max}/\lambda_{\min}` measures how elongated they are — this
quantity controls the convergence speed of most algorithms.

.. plot::
   :caption: MSE surface contours for two inputs with the same power. A white
             input (left) has circular contours; a correlated input (right)
             has elongated ones. The red star is the Wiener solution.

   def contours(ax, R, p, title):
       h0, h1 = np.meshgrid(np.linspace(-1, 3, 200), np.linspace(-1, 3, 200))
       H = np.stack([h0.ravel(), h1.ravel()])
       J = 1 - 2 * p @ H + np.sum(H * (R @ H), axis=0)
       ax.contour(h0, h1, J.reshape(h0.shape), 20, cmap='viridis')
       ho = np.linalg.solve(R, p)
       ax.plot(*ho, 'r*', ms=10)
       ax.set(title=title, xlabel='$h_0$', ylabel='$h_1$', aspect='equal')

   fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
   contours(ax[0], np.array([[1, 0.0], [0.0, 1]]), np.array([1.0, 1.0]),
            r'white input: spread $=1$')
   contours(ax[1], np.array([[1, 0.9], [0.9, 1]]), np.array([1.0, 1.0]),
            r'correlated input: spread $=19$')
   fig.tight_layout()

Estimating :math:`\mathbf{R}` and :math:`\mathbf{p}` from data
--------------------------------------------------------------

In practice the statistics are unknown, but for ergodic signals the
expectations can be replaced by time averages over the :math:`M-N+1` complete
regressors available:

.. math::

   \hat{\mathbf{R}}=\frac{1}{M-N+1}\sum_{k=N-1}^{M-1}\mathbf{x}_k\mathbf{x}_k^T,
   \qquad
   \hat{\mathbf{p}}=\frac{1}{M-N+1}\sum_{k=N-1}^{M-1}d[k]\,\mathbf{x}_k .

.. plot::
   :caption: Wiener solution computed from data for an unknown 4-tap system
             observed in noise. The estimate is excellent because the system is
             FIR and has the same length as the filter.

   rng = np.random.default_rng(1)
   g = np.array([1.0, 0.5, -0.3, 0.1])
   x = rng.standard_normal(10000)
   d = np.convolve(x, g)[:len(x)] + 0.1 * rng.standard_normal(len(x))

   R = alg.estimate_correlation_matrix(x, 4)
   p = alg.estimate_cross_correlation(x, d, 4)
   h_o = np.linalg.solve(R, p)

   fig, ax = plt.subplots(figsize=(5, 2.8))
   ax.stem(np.arange(4) - 0.08, g, 'k', markerfmt='ko', basefmt=' ', label='true $g$')
   ax.stem(np.arange(4) + 0.08, h_o, 'r', markerfmt='rs', basefmt=' ', label='Wiener $h_o$')
   ax.legend(); ax.set(xlabel='$i$')

.. admonition:: Pitfall
   :class: pitfall

   Solving :eq:`wiener-hopf` requires :math:`O(N^3)` operations and the
   knowledge (or a good estimate) of the statistics *of the whole signal*. It
   gives a single, fixed filter. If the system changes with time, or you need
   a sample-by-sample answer at :math:`O(N)` cost, you need an **adaptive**
   algorithm — the subject of the next chapters.

Exercises
---------

.. admonition:: Exercise 1
   :class: exercise

   For a white input with variance :math:`\sigma_x^2` and a noiseless system
   :math:`d[k]=\sum_i g_i x[k-i]` of length :math:`N`, find :math:`\mathbf{R}`,
   :math:`\mathbf{p}` and :math:`\mathbf{h}_o`.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

:math:`\mathbf{R}=\sigma_x^2\mathbf{I}` and
:math:`\mathbf{p}=\sigma_x^2\mathbf{g}`, so
:math:`\mathbf{h}_o=\mathbf{R}^{-1}\mathbf{p}=\mathbf{g}`: the Wiener filter
identifies the system exactly and :math:`J_{\min}=0`.

.. raw:: html

   </details>

.. admonition:: Exercise 2
   :class: exercise

   A filter with a single coefficient has :math:`\sigma_d^2=4`,
   :math:`r=\operatorname{E}[x^2]=2` and :math:`p=2`. Compute :math:`h_o` and
   :math:`J_{\min}`.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

:math:`h_o=p/r=1` and :math:`J_{\min}=\sigma_d^2-p^2/r=4-2=2`.
(Note that the three numbers cannot be arbitrary: by the Cauchy-Schwarz
inequality :math:`p^2\le\sigma_d^2\,r`, otherwise :math:`J_{\min}` would be
negative.)

.. raw:: html

   </details>

.. seealso::

   :func:`~asp.algorithms.estimate_correlation_matrix`,
   :func:`~asp.algorithms.estimate_cross_correlation`,
   notebooks 3 and 4.
