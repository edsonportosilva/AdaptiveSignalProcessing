Blind adaptive filtering and the CMA
====================================

.. admonition:: Learning objectives
   :class: objectives

   * Explain why *blind* adaptation is needed in communications.
   * Derive the constant modulus algorithm (CMA) from the dispersion cost.
   * Identify the ambiguities of blind equalization (delay, phase) and the
     role of the initialization.
   * Use :func:`~asp.algorithms.cma` with real and
     complex signals.

The problem
-----------

In supervised adaptation (LMS, RLS) a **desired signal** :math:`d[k]` is
required. In a communication system that means transmitting a *training
sequence*, which wastes bandwidth and is impossible when the receiver must
re-synchronise after a link failure or when listening to a broadcast. *Blind*
algorithms adapt using only the received signal plus some **known property of
the transmitted signal**.

The constant modulus property
-----------------------------

Many modulation formats (PSK, FM, QPSK, …) transmit symbols with constant
modulus, :math:`|s[k]|=\text{const}`. The channel's intersymbol interference
(ISI) adds neighbouring symbols to each sample, so the modulus of the
received signal *fluctuates*. An equalizer that makes the output modulus
constant again therefore removes ISI. This is quantified by the **dispersion
(Godard / CMA) cost**

.. math::
   :label: cma-cost

   J_{CMA}(\mathbf{h})=\operatorname{E}\left[\left(|y[k]|^2-R_2\right)^2\right],
   \qquad y[k]=\mathbf{h}^T\mathbf{x}_k,\qquad
   R_2=\frac{\operatorname{E}|s|^4}{\operatorname{E}|s|^2}.

For constant-modulus constellations :math:`R_2=|s|^2` (1 for unit amplitude).

The algorithm
-------------

Stochastic gradient descent on :eq:`cma-cost` gives

.. math::
   :label: cma

   \begin{aligned}
   y[k]&=\mathbf{h}_k^T\mathbf{x}_k,\\
   \mathbf{h}_{k+1}&=\mathbf{h}_k-\mu\left(|y[k]|^2-R_2\right)y[k]\,\mathbf{x}_k^{*}.
   \end{aligned}

Compare it with LMS: the "error" :math:`(|y|^2-R_2)\,y` is computed from the
output itself, so **no reference signal is needed**.

.. plot::
   :caption: CMA equalizing a BPSK signal distorted by an ISI channel
             without any training sequence. Left: received signal (the eye is
             closed). Right: the CMA output after convergence clusters
             around :math:`\pm1`.

   rng = np.random.default_rng(12)
   s = rng.choice([-1.0, 1.0], 20000)
   r = np.convolve(s, [1.0, 0.5, -0.2])[:len(s)]

   out, h, cost, H = alg.cma(r, 7, 1e-3)

   fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
   ax[0].plot(r[-300:], '.', ms=3); ax[0].set(title='received', xlabel='$k$')
   ax[1].plot(out[-300:], '.', ms=3); ax[1].set(title='CMA output', xlabel='$k$')
   ax[2].semilogy(np.convolve(cost, np.ones(100) / 100, mode='same') + 1e-12)
   ax[2].set(title='CMA cost', xlabel='$k$')
   fig.tight_layout()

Properties and pitfalls
-----------------------

* **Phase/sign ambiguity.** :math:`J_{CMA}` depends only on :math:`|y|`, so
  any constant rotation of the output (:math:`\pm1` for real signals,
  an arbitrary angle for complex ones) is also a minimum. The ambiguity is
  resolved afterwards (differential encoding, a few pilot symbols, or a
  carrier-recovery loop). The same property makes the CMA tolerant to a
  carrier phase offset.
* **Delay ambiguity.** The equalizer may converge to any delayed inverse of
  the channel.
* **Non-convexity.** :math:`J_{CMA}` has local minima, and the result depends
  on the initialization. The standard choice (used here) is the
  *single-spike* initialization: :math:`h_0=1`, all the other taps 0.
* **Slow convergence and sensitivity to** :math:`\mu`. The cost is of fourth
  order in :math:`\mathbf{h}`; a step size that is too large diverges.
* **Constant-modulus assumption.** For multilevel constellations such as
  16-QAM, :math:`|s|` is not constant; the CMA still works as a coarse
  equalization stage (with the proper :math:`R_2`, see the ``R2`` argument)
  but leaves a residual error — a decision-directed algorithm usually
  follows.

.. plot::
   :caption: Complex (QPSK) signal with a channel and a 40-degree carrier
             phase offset. The CMA output is on a circle (constant modulus
             recovered) rotated by an arbitrary angle — the phase ambiguity.

   rng = np.random.default_rng(13)
   n = 30000
   s = (rng.choice([-1, 1], n) + 1j * rng.choice([-1, 1], n)) / np.sqrt(2)
   r = np.convolve(s, [1.0, 0.25 + 0.15j])[:n] * np.exp(1j * np.deg2rad(40))

   out, *_ = alg.cma(r, 5, 5e-3)

   fig, ax = plt.subplots(1, 2, figsize=(7, 3.3))
   for a, sig, t in zip(ax, (r, out), ('received', 'CMA output')):
       a.plot(sig[-3000:].real, sig[-3000:].imag, '.', ms=2)
       a.set(title=t, aspect='equal', xlabel='I', ylabel='Q')
   fig.tight_layout()

Exercises
---------

.. admonition:: Exercise
   :class: exercise

   Compute :math:`R_2` for 4-PAM with amplitudes :math:`\{\pm1,\pm3\}` and
   equiprobable symbols.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

:math:`\operatorname{E}|s|^2=(1+9)/2=5` and
:math:`\operatorname{E}|s|^4=(1+81)/2=41`, so
:math:`R_2=41/5=8.2`. (Not constant modulus, therefore :math:`R_2\ne|s|^2`.)

.. raw:: html

   </details>

.. seealso::

   :func:`~asp.algorithms.cma`, notebook 9.
   Godard (1980), Treichler & Agee (1983).
