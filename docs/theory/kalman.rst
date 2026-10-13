Kalman filters
==============

.. admonition:: Learning objectives
   :class: objectives

   * Write a dynamical system in state-space form with process and
     measurement noise.
   * Derive the prediction and update steps of the Kalman filter and interpret
     the Kalman gain.
   * Explain how the extended Kalman filter (EKF) handles nonlinear models.
   * Apply
     :func:`~asp.algorithms.kalman_filter` and
     :func:`~asp.algorithms.extended_kalman_filter`.

State-space model
-----------------

Many problems are naturally described by a hidden **state** that evolves in
time and is observed through noisy measurements,

.. math::
   :label: state-space

   \mathbf{x}[k+1]=\mathbf{A}\,\mathbf{x}[k]+\mathbf{n}[k],\qquad
   \mathbf{y}[k]=\mathbf{C}\,\mathbf{x}[k]+\mathbf{v}[k],

with zero-mean white Gaussian noises
:math:`\mathbf{n}\sim\mathcal{N}(\mathbf{0},\mathbf{R}_n)` (process: models
what the dynamics do not capture) and
:math:`\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{R}_v)` (measurement).
Examples: the position and velocity of a mass-spring system measured by a
noisy sensor; the position of a vehicle measured by a radar.

The goal is the minimum mean-square error estimate
:math:`\hat{\mathbf{x}}[k]=\operatorname{E}\left[\mathbf{x}[k]\mid
\mathbf{y}[0],\dots,\mathbf{y}[k]\right]` **recursively**, i.e. updating
the previous estimate when a new measurement arrives.

Prediction and update
---------------------

The filter keeps a Gaussian belief about the state: a mean
:math:`\hat{\mathbf{x}}` and a covariance :math:`\mathbf{P}`. At each
instant it performs two steps.

**1. Prediction** — push the belief through the dynamics (uncertainty grows):

.. math::

   \hat{\mathbf{x}}^{-}[k]=\mathbf{A}\hat{\mathbf{x}}[k-1],\qquad
   \mathbf{P}^{-}[k]=\mathbf{A}\mathbf{P}[k-1]\mathbf{A}^T+\mathbf{R}_n .

**2. Update** — correct the prediction with the measurement (uncertainty
shrinks):

.. math::
   :label: kalman-update

   \begin{aligned}
   \mathbf{K}[k]&=\mathbf{P}^{-}\mathbf{C}^T\left(\mathbf{C}\mathbf{P}^{-}\mathbf{C}^T+\mathbf{R}_v\right)^{-1}\\
   \hat{\mathbf{x}}[k]&=\hat{\mathbf{x}}^{-}+\mathbf{K}\left(\mathbf{y}[k]-\mathbf{C}\hat{\mathbf{x}}^{-}\right)\\
   \mathbf{P}[k]&=(\mathbf{I}-\mathbf{K}\mathbf{C})\mathbf{P}^{-}
   \end{aligned}

.. admonition:: Key idea
   :class: keyidea

   The term :math:`\mathbf{y}[k]-\mathbf{C}\hat{\mathbf{x}}^{-}` is the
   **innovation**: the part of the measurement that the prediction could not
   explain. The **Kalman gain** :math:`\mathbf{K}` decides how much to trust it.
   If the measurement noise :math:`\mathbf{R}_v` is small, :math:`\mathbf{K}` is
   large and the estimate follows the data; if the model is accurate
   (:math:`\mathbf{R}_n` small), :math:`\mathbf{K}` is small and the estimate
   follows the model. The filter is the optimal compromise between the two.

.. plot::
   :caption: Kalman filter tracking the position of a mass-spring system
             from noisy position measurements. The estimate is much closer to
             the true trajectory than the raw measurements.

   dt, n = 0.05, 400
   rng = np.random.default_rng(10)
   A = np.array([[1.0, dt], [-2 * dt, 1.0]])      # discretised mass-spring
   C = np.array([[1.0, 0.0]])                      # only the position is measured
   x = np.zeros((2, n)); s = np.array([1.0, 0.0])
   for k in range(n):
       s = A @ s + 0.01 * rng.standard_normal(2)
       x[:, k] = s
   y = C @ x + 0.4 * rng.standard_normal((1, n))

   x_hat = alg.kalman_filter(A, C, 1e-4 * np.eye(2), 0.16 * np.eye(1),
                             np.array([[1.0], [0.0]]), y)

   fig, ax = plt.subplots(figsize=(6.4, 3))
   ax.plot(y[0], '.', ms=2, color='0.6', label='measurements')
   ax.plot(x[0], 'k', lw=1, label='true position')
   ax.plot(x_hat[0], 'r', lw=1, label='Kalman estimate')
   ax.set(xlabel='$k$'); ax.legend(loc='upper right')

.. admonition:: Pitfall
   :class: pitfall

   The initial covariance :math:`\mathbf{P}[-1]` expresses how confident you are
   about the initial state. In
   :func:`~asp.algorithms.kalman_filter` it defaults to
   :math:`\hat{\mathbf{x}}_0\hat{\mathbf{x}}_0^T`, which is the **zero matrix**
   if the initial state is zero — the filter would then believe it knows the
   state exactly and ignore all measurements. Pass ``P_init`` explicitly (for
   example a multiple of the identity) in that case.

Optimality and limits
---------------------

For linear models with Gaussian noise the Kalman filter is the optimal
estimator (it computes the exact posterior mean). If the noise is not
Gaussian it is still the best **linear** estimator. It requires knowing
:math:`\mathbf{A}`, :math:`\mathbf{C}`, :math:`\mathbf{R}_n` and
:math:`\mathbf{R}_v`; if they are wrong, the estimate degrades (and the
computed covariance :math:`\mathbf{P}` is no longer reliable).

The extended Kalman filter
--------------------------

Real systems are often nonlinear,

.. math::

   \mathbf{x}[k+1]=f(\mathbf{x}[k])+\mathbf{n}[k],\qquad
   \mathbf{y}[k]=h(\mathbf{x}[k])+\mathbf{v}[k] .

The **EKF** *linearizes* :math:`f` and :math:`h` around the current estimate
using their Jacobians :math:`\mathbf{F}=\partial f/\partial\mathbf{x}` and
:math:`\mathbf{H}=\partial h/\partial\mathbf{x}` and applies the Kalman
equations to the linearized model. The mean is propagated through the
nonlinear functions themselves:

.. math::

   \hat{\mathbf{x}}^{-}=f(\hat{\mathbf{x}}),\quad
   \mathbf{P}^{-}=\mathbf{F}\mathbf{P}\mathbf{F}^T+\mathbf{R}_n,\quad
   \hat{\mathbf{x}}=\hat{\mathbf{x}}^{-}+\mathbf{K}\left(\mathbf{y}-h(\hat{\mathbf{x}}^{-})\right),

with :math:`\mathbf{K}` computed as in :eq:`kalman-update` using
:math:`\mathbf{H}` instead of :math:`\mathbf{C}`. A classical example is the
re-entry of a body in the atmosphere, tracked by a radar that measures the
slant range — see notebook 7. In the EKF, the quality of the linearization
matters: a poor initial estimate can make the filter diverge.

.. plot::
   :caption: EKF estimating a scalar state :math:`x` that evolves as a
             random walk and is observed through the nonlinear sensor
             :math:`y=x^3/10`.

   rng = np.random.default_rng(11)
   n = 300
   x = 1.5 + np.cumsum(0.02 * rng.standard_normal(n))
   y = (x**3 / 10 + 0.05 * rng.standard_normal(n))[None, :]

   x_hat = alg.extended_kalman_filter(
       f=lambda s: s, h=lambda s: s**3 / 10,
       F_jac=lambda s: np.eye(1), H_jac=lambda s: 0.3 * s**2,
       Rn=np.array([[4e-4]]), Rv=np.array([[2.5e-3]]),
       x_init=np.array([[1.0]]), y=y, P_init=np.eye(1))

   fig, ax = plt.subplots(figsize=(6.4, 2.8))
   ax.plot(x, 'k', label='true state')
   ax.plot(x_hat[0], 'r', label='EKF')
   ax.set(xlabel='$k$'); ax.legend()

Exercises
---------

.. admonition:: Exercise
   :class: exercise

   A scalar random walk :math:`x[k+1]=x[k]+n[k]` (:math:`\sigma_n^2=0`) is
   observed with :math:`y[k]=x[k]+v[k]`, :math:`\sigma_v^2=R`. Starting with
   :math:`P=P_0`, show that :math:`P[k]=\left(P_0^{-1}+k/R\right)^{-1}`
   (after :math:`k` updates) and discuss the gain.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

With :math:`A=C=1` and no process noise :math:`P^-=P`, the gain is
:math:`K=P/(P+R)` and the update gives :math:`P_{\text{new}}=(1-K)P=PR/(P+R)`,
i.e. :math:`1/P_{\text{new}}=1/P+1/R`: each measurement adds an "information"
:math:`1/R`. Thus :math:`P[k]=(P_0^{-1}+k/R)^{-1}\to R/k`. The gain
:math:`K[k]\approx1/k` decays and the estimator tends to the running average
of the measurements — the sample mean.

.. raw:: html

   </details>

.. seealso::

   :func:`~asp.algorithms.kalman_filter`,
   :func:`~asp.algorithms.extended_kalman_filter`,
   notebook 7.
