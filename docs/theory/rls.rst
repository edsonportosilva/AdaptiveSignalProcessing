Recursive Least Squares (RLS)
=============================

.. admonition:: Learning objectives
   :class: objectives

   * Formulate the exponentially weighted least-squares problem and its
     normal equations.
   * Derive the RLS recursion with the matrix inversion lemma.
   * Interpret the forgetting factor :math:`\lambda`, and the a priori and a
     posteriori error formulations.
   * Compare RLS with LMS in convergence speed, tracking and cost.

Least squares versus mean-square
--------------------------------

LMS minimizes an *ensemble* average, :math:`\operatorname{E}[e^2]`, in a
stochastic-gradient way. **Least squares** (LS) minimizes instead the sum of
the squared errors over the *observed data*. With an exponential window that
discounts old samples:

.. math::
   :label: ls-cost

   J_k(\mathbf{h})=\sum_{i=0}^{k}\lambda^{k-i}\left(d[i]-\mathbf{h}^T\mathbf{x}_i\right)^2,
   \qquad 0<\lambda\le1 .

Differentiating and equating to zero gives the **normal equations**

.. math::

   \hat{\mathbf{R}}_k\,\mathbf{h}_k=\hat{\mathbf{p}}_k,\qquad
   \hat{\mathbf{R}}_k=\sum_{i=0}^{k}\lambda^{k-i}\mathbf{x}_i\mathbf{x}_i^T,\quad
   \hat{\mathbf{p}}_k=\sum_{i=0}^{k}\lambda^{k-i}d[i]\mathbf{x}_i ,

which are the Wiener-Hopf equations with time averages in place of
expectations. Solving them at every sample costs :math:`O(N^3)`; RLS finds
the same solution recursively.

The RLS recursion
-----------------

Both sums can be updated from one sample to the next:

.. math::

   \hat{\mathbf{R}}_k=\lambda\hat{\mathbf{R}}_{k-1}+\mathbf{x}_k\mathbf{x}_k^T,
   \qquad
   \hat{\mathbf{p}}_k=\lambda\hat{\mathbf{p}}_{k-1}+d[k]\mathbf{x}_k .

The only difficulty is the inverse :math:`\mathbf{P}_k=\hat{\mathbf{R}}_k^{-1}`.
The **matrix inversion lemma**,

.. math::

   (\mathbf{A}+\mathbf{u}\mathbf{v}^T)^{-1}=\mathbf{A}^{-1}-
   \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^T\mathbf{A}^{-1}}
   {1+\mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}},

applied to a rank-one update, lets us update :math:`\mathbf{P}_k` in
:math:`O(N^2)` operations without inverting anything:

.. math::
   :label: rls

   \begin{aligned}
   \mathbf{g}_k &= \frac{\lambda^{-1}\mathbf{P}_{k-1}\mathbf{x}_k}
                        {1+\lambda^{-1}\mathbf{x}_k^T\mathbf{P}_{k-1}\mathbf{x}_k}
                   &&\text{(gain vector)}\\
   \epsilon[k] &= d[k]-\mathbf{h}_{k-1}^T\mathbf{x}_k
                   &&\text{(a priori error)}\\
   \mathbf{h}_k &= \mathbf{h}_{k-1}+\mathbf{g}_k\,\epsilon[k]\\
   \mathbf{P}_k &= \lambda^{-1}\mathbf{P}_{k-1}-\lambda^{-1}\mathbf{g}_k\mathbf{x}_k^T\mathbf{P}_{k-1}
   \end{aligned}

This is the structure implemented by
:func:`~asp.algorithms.rls_apriori`. The same
recursion can be written in terms of the **a posteriori** error
:math:`e[k]=d[k]-\mathbf{h}_k^T\mathbf{x}_k`, which relates to the a priori
error through the *conversion factor*

.. math::

   e[k]=\frac{\epsilon[k]}{\alpha_k},\qquad
   \alpha_k=1+\lambda^{-1}\mathbf{x}_k^T\mathbf{P}_{k-1}\mathbf{x}_k\ge1
   \quad\text{(}\texttt{rls\_aposteriori}\text{)}.

:func:`~asp.algorithms.rls` implements the
"textbook" version that updates :math:`\mathbf{P}_k` and
:math:`\hat{\mathbf{p}}_k` and then computes
:math:`\mathbf{h}_k=\mathbf{P}_k\hat{\mathbf{p}}_k` directly. The three
variants produce the same coefficients; they differ in the intermediate
quantities that they expose.

.. admonition:: Key idea
   :class: keyidea

   RLS is **Newton's method with the exact (time-averaged) Hessian**: the
   gain vector :math:`\mathbf{g}_k=\mathbf{P}_k\mathbf{x}_k` is the regressor
   whitened by the inverse correlation matrix. As a consequence the
   convergence speed does **not** depend on the eigenvalue spread, and takes
   about :math:`2N` samples — typically an order of magnitude faster than
   LMS.

Initialization
--------------

The recursion starts with :math:`\mathbf{h}_{-1}=\mathbf{0}` and
:math:`\mathbf{P}_{-1}=\delta^{-1}\mathbf{I}`, i.e. a ridge-regularised
cost with weight :math:`\delta\lambda^{k}` on :math:`\|\mathbf{h}\|^2`.
Choose :math:`\delta` small with respect to the input power; a *large*
:math:`\delta` biases the solution towards zero for a long time when
:math:`\lambda=1`.

The forgetting factor
---------------------

The weights :math:`\lambda^{k-i}` give RLS an effective memory of
:math:`\approx 1/(1-\lambda)` samples.

* :math:`\lambda=1` (growing window) gives the lowest steady-state error but
  the filter cannot track changes.
* :math:`\lambda<1` allows tracking at the cost of a higher estimation
  variance (proportional to :math:`(1-\lambda)/(1+\lambda)`).

.. plot::
   :caption: RLS and LMS identifying a 4-tap system. RLS converges in a few
             tens of samples (:math:`\approx 2N`), LMS needs hundreds.

   rng = np.random.default_rng(8)
   g = np.array([1.0, 0.5, -0.3, 0.1])
   x = rng.standard_normal(1000)
   d = np.convolve(x, g)[:len(x)] + 0.05 * rng.standard_normal(len(x))

   *_, e_lms, _ = alg.lms(x, d, 4, 0.01)
   *_, e_rls, _ = alg.rls(x, d, 4, 0.99)
   sm = lambda e: np.convolve(e, np.ones(10) / 10, mode='same')

   fig, ax = plt.subplots(figsize=(6.4, 3))
   ax.semilogy(sm(e_lms) + 1e-12, label='LMS ($\\mu=0.01$)')
   ax.semilogy(sm(e_rls) + 1e-12, label='RLS ($\\lambda=0.99$)')
   ax.set(xlabel='$k$', ylabel='smoothed $e^2[k]$'); ax.legend()

.. plot::
   :caption: Tracking a coefficient that changes with time. With
             :math:`\lambda=1` RLS averages over the whole history and cannot
             follow; with :math:`\lambda=0.98` it tracks well.

   n, Ntaps = 6000, 3
   rng = np.random.default_rng(9)
   x = rng.standard_normal(n)
   G = np.zeros((n, Ntaps)); G[:, 0] = 1.0
   G[:, 1] = 0.5 * np.sin(2 * np.pi * np.arange(n) / 2000)
   d = alg.time_varying_filter(x, G)

   fig, ax = plt.subplots(figsize=(6.4, 3))
   ax.plot(G[:, 1], 'k--', label='true')
   for λ in (1.0, 0.98):
       *_, H = alg.rls(x, d, Ntaps, λ)
       ax.plot(H[:, 1], lw=0.8, label=fr'RLS $\lambda={λ}$')
   ax.set(xlabel='$k$', title='coefficient $h_1$'); ax.legend()

Complexity and caveats
----------------------

* Cost :math:`O(N^2)` per sample, against :math:`O(N)` for LMS.
* The recursion for :math:`\mathbf{P}_k` can lose symmetry and positive
  definiteness through round-off, especially for :math:`\lambda<1` and
  long runs. Fast (:math:`O(N)`) and numerically robust (QR/lattice)
  versions exist but are outside the scope of this package.

.. admonition:: Pitfall
   :class: pitfall

   RLS is *not* uniformly better: for large :math:`N` its cost may be
   prohibitive, and with :math:`\lambda<1` and noisy data its steady-state
   misadjustment can exceed that of a well-tuned LMS.

Relation with the Kalman filter
-------------------------------

RLS is a special case of the Kalman filter (:doc:`kalman`) in which the state
is the coefficient vector, the "observation matrix" is the regressor
:math:`\mathbf{x}_k^T`, and the state covariance is scaled by
:math:`\lambda^{-1}`. The gain vector :math:`\mathbf{g}_k` is the Kalman gain.

Exercises
---------

.. admonition:: Exercise
   :class: exercise

   For :math:`\lambda=0.99`, what is the approximate memory of RLS? How much
   would you change :math:`\lambda` to track a system that changes ten times
   faster?

.. raw:: html

   <details class="solution"><summary>Solution</summary>

The memory is :math:`1/(1-\lambda)=100` samples. A system that changes ten
times faster calls for a memory about ten times shorter, i.e. :math:`1-\lambda`
ten times larger: :math:`\lambda\approx0.9`, at the cost of a noisier estimate.

.. raw:: html

   </details>

.. seealso::

   :func:`~asp.algorithms.rls`,
   :func:`~asp.algorithms.rls_apriori`,
   :func:`~asp.algorithms.rls_aposteriori`, notebook 6.
