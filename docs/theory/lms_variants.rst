Variants of the LMS algorithm
=============================

.. admonition:: Learning objectives
   :class: objectives

   * Explain the two main weaknesses of LMS: sensitivity to the input power
     and to the eigenvalue spread.
   * Derive the normalized LMS (NLMS) and the LMS-Newton updates.
   * Understand how transform-domain (TD-LMS) and frequency-domain (FD-LMS)
     algorithms trade complexity for convergence speed.
   * Choose a variant for a given problem.

Why variants?
-------------

Plain LMS (:doc:`lms`) has two limitations:

1. the stable range of :math:`\mu` depends on the input power
   (:math:`\mu<1/(N\sigma_x^2)`), which may be unknown or time-varying;
2. convergence is slow for inputs with large eigenvalue spread.

The variants below attack one or both.

Normalized LMS
--------------

NLMS divides the step by the instantaneous energy of the regressor:

.. math::
   :label: nlms

   \mathbf{h}_{k+1}=\mathbf{h}_k+\frac{\mu}{\gamma+\|\mathbf{x}_k\|^2}\,
   e[k]\,\mathbf{x}_k .

It is the solution of the *minimum disturbance* problem

.. math::

   \min_{\mathbf{h}_{k+1}}\|\mathbf{h}_{k+1}-\mathbf{h}_k\|^2
   \quad\text{s.t.}\quad d[k]=\mathbf{h}_{k+1}^T\mathbf{x}_k

(for :math:`\mu=1`): change the coefficients as little as possible while
zeroing the a posteriori error. It is stable for :math:`0<\mu<2`
**regardless of the input power**, and :math:`\gamma>0` only avoids dividing
by zero. Function:
:func:`~asp.algorithms.nlms`.

.. plot::
   :caption: LMS and NLMS when the input amplitude suddenly increases tenfold
             at :math:`k=2000`. LMS (with a step size adequate for the first
             half) becomes unstable; NLMS is not affected.

   rng = np.random.default_rng(6)
   g = np.array([1.0, 0.5, -0.3, 0.1])
   x = rng.standard_normal(4000)
   x[2000:] *= 10
   d = np.convolve(x, g)[:len(x)] + 1e-3 * rng.standard_normal(len(x))

   with np.errstate(all='ignore'):
       *_, e_lms, _ = alg.lms(x, d, 4, 0.02)
       *_, e_nlms, _ = alg.nlms(x, d, 4, 0.5)

   fig, ax = plt.subplots(figsize=(6.4, 3))
   ax.semilogy(e_nlms + 1e-30, label='NLMS')
   ax.semilogy(np.minimum(e_lms, 1e30) + 1e-30, label='LMS')
   ax.set(xlabel='$k$', ylabel='$e^2[k]$', ylim=(1e-8, 1e12)); ax.legend()

LMS-Newton
----------

Newton's method (:eq:`newton`) multiplies the gradient by
:math:`\mathbf{R}^{-1}`, which removes the effect of the eigenvalue spread
(all modes then converge at the same rate). LMS-Newton keeps the instantaneous
gradient of LMS but uses a recursive estimate of the correlation matrix,

.. math::

   \hat{\mathbf{R}}_k=(1-\alpha)\hat{\mathbf{R}}_{k-1}+\alpha\,\mathbf{x}_k\mathbf{x}_k^T ,

and propagates its **inverse** with the matrix inversion lemma
:math:`(\mathbf{A}+\mathbf{u}\mathbf{u}^T)^{-1}=\mathbf{A}^{-1}-
\dfrac{\mathbf{A}^{-1}\mathbf{u}\mathbf{u}^T\mathbf{A}^{-1}}
{1+\mathbf{u}^T\mathbf{A}^{-1}\mathbf{u}}`:

.. math::

   \hat{\mathbf{R}}_k^{-1}=\frac{1}{1-\alpha}\left[\hat{\mathbf{R}}_{k-1}^{-1}-
   \frac{\hat{\mathbf{R}}_{k-1}^{-1}\mathbf{x}_k\mathbf{x}_k^T\hat{\mathbf{R}}_{k-1}^{-1}}
   {\frac{1-\alpha}{\alpha}+\mathbf{x}_k^T\hat{\mathbf{R}}_{k-1}^{-1}\mathbf{x}_k}\right],
   \qquad
   \mathbf{h}_{k+1}=\mathbf{h}_k+\mu\,e[k]\,\hat{\mathbf{R}}_k^{-1}\mathbf{x}_k .

Cost: :math:`O(N^2)` per sample. Function:
:func:`~asp.algorithms.lms_newton`.

.. plot::
   :caption: Coloured input (AR(1), pole 0.95). LMS needs thousands of
             samples; LMS-Newton and TD-LMS (with a DCT) converge much faster
             because they equalize the modes.

   from scipy.fft import dct
   rng = np.random.default_rng(7)
   n = 6000
   w = rng.standard_normal(n)
   x = np.zeros(n)
   for k in range(1, n):
       x[k] = 0.95 * x[k - 1] + w[k]
   x /= x.std()
   g = np.array([1.0, -0.5, 0.25, 0.1])
   d = np.convolve(x, g)[:n] + 1e-3 * rng.standard_normal(n)

   M = dct(np.eye(4), norm='ortho', axis=0)
   *_, e_lms, _ = alg.lms(x, d, 4, 0.005)
   *_, e_nlms, _ = alg.nlms(x, d, 4, 0.1)
   *_, e_newton, _ = alg.lms_newton(x, d, 4, 0.05, 0.01)
   *_, e_td, _ = alg.tdlms(x, d, 4, 0.02, M)

   sm = lambda e: np.convolve(e, np.ones(40) / 40, mode='same') + 1e-30
   fig, ax = plt.subplots(figsize=(6.4, 3.2))
   for e, lab in ((e_lms, 'LMS'), (e_nlms, 'NLMS'), (e_newton, 'LMS-Newton'), (e_td, 'TD-LMS')):
       ax.semilogy(sm(e), label=lab)
   ax.set(xlabel='$k$', ylabel='smoothed $e^2[k]$', ylim=(1e-8, 10)); ax.legend()

Transform-domain LMS
--------------------

LMS-Newton uses a full matrix. A cheaper idea is to *decorrelate* the
regressor with a fixed orthogonal transform :math:`\mathbf{M}` (DCT, DFT,
Hartley, …), :math:`\mathbf{s}_k=\mathbf{M}\mathbf{x}_k`. The correlation
matrix of :math:`\mathbf{s}_k` is then nearly diagonal, and it suffices to
normalize each component by its own power:

.. math::

   p_i[k]=\alpha s_i^2[k]+(1-\alpha)p_i[k-1],\qquad
   \mathbf{w}_{k+1}=\mathbf{w}_k+\mu\,e[k]\operatorname{diag}
   \!\left(\frac{1}{p_i[k]+\gamma}\right)\mathbf{s}_k .

The equivalent time-domain filter is :math:`\mathbf{h}=\mathbf{M}^T\mathbf{w}`.
Cost: :math:`O(N)` per sample plus the transform. How well it works depends on
how well :math:`\mathbf{M}` decorrelates the actual input. Function:
:func:`~asp.algorithms.tdlms`.

Frequency-domain LMS
--------------------

For long filters (hundreds or thousands of coefficients) even :math:`O(N)` per
sample is costly. FD-LMS processes the signal in blocks with the
**overlap-and-save** method: a block of :math:`N_{FFT}` samples is
transformed, multiplied by the frequency response of the filter, and brought
back to the time domain, where the first :math:`N-1` samples (corrupted by the
circular convolution) are discarded. The gradient is also computed in the
frequency domain:

.. math::

   \mathbf{y}=\operatorname{IFFT}(\mathbf{X}\odot\mathbf{H}),\qquad
   \mathbf{H}\leftarrow\mathbf{H}+\frac{\mu}{\sqrt{N_{FFT}}}
   \operatorname{FFT}(\mathbf{e})\odot\mathbf{X}^* .

The cost per sample falls to :math:`O(\log N)`, the coefficients are updated
once per block, and each frequency bin can be given its own step size, which
also helps with coloured inputs. Function:
:func:`~asp.algorithms.fdlms`.

Which one to use?
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 32 32
   :class: algorithm-table

   * - Algorithm
     - Cost / sample
     - Strength
     - Weakness
   * - LMS
     - :math:`O(N)`
     - simplest, robust
     - slow for coloured inputs; :math:`\mu` depends on input power
   * - NLMS
     - :math:`O(N)`
     - power-independent step size
     - still sensitive to eigenvalue spread
   * - LMS-Newton
     - :math:`O(N^2)`
     - convergence independent of eigenvalue spread
     - cost; needs :math:`\alpha` tuning
   * - TD-LMS
     - :math:`O(N)` + transform
     - cheap decorrelation
     - depends on a good fixed transform
   * - FD-LMS
     - :math:`O(\log N)`
     - very long filters
     - block update (latency); extra constraints for exactness

Exercises
---------

.. admonition:: Exercise
   :class: exercise

   Show that for :math:`\mu=1` and :math:`\gamma=0` the a posteriori error of
   NLMS is zero.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

With :math:`\mathbf{h}_{k+1}=\mathbf{h}_k+\mu\,e[k]\,\mathbf{x}_k/
\|\mathbf{x}_k\|^2`, the a posteriori error is
:math:`d[k]-\mathbf{h}_{k+1}^T\mathbf{x}_k=e[k]-\mu\,e[k]
\,\|\mathbf{x}_k\|^2/\|\mathbf{x}_k\|^2=(1-\mu)e[k]`, which vanishes for
:math:`\mu=1`. For :math:`0<\mu<2` its magnitude is smaller than that of the
a priori error, hence the stability range.

.. raw:: html

   </details>

.. seealso::

   :func:`~asp.algorithms.nlms`,
   :func:`~asp.algorithms.lms_newton`,
   :func:`~asp.algorithms.tdlms`,
   :func:`~asp.algorithms.fdlms`, notebook 5.
