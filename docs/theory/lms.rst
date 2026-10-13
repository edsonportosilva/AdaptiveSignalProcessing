The LMS algorithm
=================

.. admonition:: Learning objectives
   :class: objectives

   * Derive the LMS update from steepest descent by using an instantaneous
     gradient estimate.
   * State the convergence condition on the step size and explain the
     trade-off between convergence speed and steady-state error.
   * Identify unknown systems and equalize channels with
     :func:`~asp.algorithms.lms`.
   * Explain why LMS can *track* time-varying systems.

From steepest descent to LMS
----------------------------

Steepest descent on the MSE (:doc:`optimization`) needs
:math:`\nabla J=-2\mathbf{p}+2\mathbf{R}\mathbf{h}`, i.e. the statistics of
the signals. Widrow and Hoff (1960) observed that the *instantaneous* values
:math:`\hat{\mathbf{R}}=\mathbf{x}_k\mathbf{x}_k^T` and
:math:`\hat{\mathbf{p}}=d[k]\mathbf{x}_k` give an unbiased, though noisy,
estimate of the gradient:

.. math::

   \hat\nabla J=-2d[k]\mathbf{x}_k+2\mathbf{x}_k\mathbf{x}_k^T\mathbf{h}_k
   =-2e[k]\,\mathbf{x}_k .

Plugging it into :math:`\mathbf{h}_{k+1}=\mathbf{h}_k-\mu\hat\nabla J` yields
the **LMS algorithm**:

.. math::
   :label: lms

   \begin{aligned}
   y[k] &= \mathbf{h}_k^T\mathbf{x}_k,\\
   e[k] &= d[k]-y[k],\\
   \mathbf{h}_{k+1} &= \mathbf{h}_k+2\mu\,e[k]\,\mathbf{x}_k .
   \end{aligned}

.. admonition:: Key idea
   :class: keyidea

   LMS needs **no statistics, no matrix inversion and no knowledge of the
   system**: just :math:`2N` multiplications and additions per sample. It
   "learns" the Wiener solution by averaging, over many samples, the noisy
   gradient estimates. That is why it is the workhorse of adaptive filtering.

Convergence
-----------

*In the mean.* Assuming :math:`\mathbf{h}_k` is independent of
:math:`\mathbf{x}_k` (independence assumption), taking expectations of
:eq:`lms` gives the same recursion as steepest descent,

.. math::

   \operatorname{E}[\mathbf{v}_{k+1}]=(\mathbf{I}-2\mu\mathbf{R})
   \operatorname{E}[\mathbf{v}_k],\qquad \mathbf{v}_k=\mathbf{h}_k-\mathbf{h}_o,

so the coefficients converge to :math:`\mathbf{h}_o` in the mean if

.. math::
   :label: lms-mu

   0<\mu<\frac{1}{\lambda_{\max}}
   \quad\Longleftarrow\quad
   \mu<\frac{1}{N\sigma_x^2}\quad(\text{since }\lambda_{\max}\le
   \operatorname{tr}\mathbf{R}=N\sigma_x^2).

Convergence in the mean-square sense (what we actually need) is more
demanding: roughly :math:`\mu<1/(3\operatorname{tr}\mathbf{R})`.

*Speed.* Mode :math:`i` decays as :math:`(1-2\mu\lambda_i)^k`: the slowest
mode has time constant :math:`\tau_{\max}\approx 1/(2\mu\lambda_{\min})`.
Inputs with a large eigenvalue spread are therefore slow to converge.

*Steady state.* Because the gradient estimate is noisy, the coefficients
keep fluctuating around :math:`\mathbf{h}_o` and the MSE is larger than
:math:`J_{\min}` by the *excess MSE*. For small :math:`\mu`,

.. math::

   J_{\text{ex}}\approx\mu\operatorname{tr}(\mathbf{R})\,J_{\min},
   \qquad \mathcal{M}=\frac{J_{\text{ex}}}{J_{\min}}\approx
   \mu\operatorname{tr}(\mathbf{R})\quad(\text{misadjustment}).

.. admonition:: Key idea: the fundamental trade-off
   :class: keyidea

   A **large** :math:`\mu` converges **fast** but leaves a **large**
   steady-state error and risks instability; a **small** :math:`\mu` is
   accurate but slow. The simulation below shows the three regimes.

.. plot::
   :caption: LMS identifying a 4-tap system observed in noise
             (:math:`\sigma_v^2=10^{-2}`) for three step sizes. The smoothed
             squared error is shown; the dashed line is :math:`J_{\min}`.

   rng = np.random.default_rng(3)
   g = np.array([1.0, 0.5, -0.3, 0.1])
   x = rng.standard_normal(6000)
   d = np.convolve(x, g)[:len(x)] + 0.1 * rng.standard_normal(len(x))

   fig, ax = plt.subplots(figsize=(6.4, 3.2))
   for μ in (0.0005, 0.005, 0.05):
       *_, sq_err, _ = alg.lms(x, d, 4, μ)
       smooth = np.convolve(sq_err, np.ones(50) / 50, mode='same')
       ax.semilogy(smooth, label=fr'$\mu={μ}$')
   ax.axhline(0.01, color='k', ls='--', lw=0.8)
   ax.set(xlabel='iteration $k$', ylabel='smoothed $e^2[k]$')
   ax.legend()

Application: system identification
-----------------------------------

The input :math:`x` excites an unknown system :math:`g`; the system output
(plus noise) is the desired signal. After convergence :math:`\mathbf{h}\approx
\mathbf{g}`. This is the experiment of the first figure of this site and of
notebook 4.

Application: channel equalization
---------------------------------

A symbol sequence :math:`s[k]` is distorted by a channel with memory,
producing intersymbol interference. Feeding the channel output to the filter
and using a delayed copy of the transmitted symbols as desired signal,
:math:`d[k]=s[k-\Delta]`, makes the filter approximate the channel inverse
(during a *training* phase).

.. plot::
   :caption: LMS equalizer for a BPSK signal distorted by the channel
             :math:`G(z)=1+0.9z^{-1}-0.4z^{-2}`. Left: received samples.
             Right: equalizer output after training.

   rng = np.random.default_rng(4)
   s = rng.choice([-1.0, 1.0], 8000)
   channel = np.array([1.0, 0.9, -0.4])
   r = np.convolve(s, channel)[:len(s)] + 0.05 * rng.standard_normal(len(s))

   Δ = 5
   d = np.concatenate((np.zeros(Δ), s[:-Δ]))       # delayed symbols
   out, h, *_ = alg.lms(r, d, 11, 0.005)

   fig, ax = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
   ax[0].plot(r[-500:], '.', ms=3); ax[0].set(title='received', xlabel='$k$')
   ax[1].plot(out[-500:], '.', ms=3); ax[1].set(title='equalized', xlabel='$k$')
   for a in ax:
       a.axhline(1, color='k', lw=0.5); a.axhline(-1, color='k', lw=0.5)

Tracking time-varying systems
-----------------------------

Since LMS never stops adapting, it can follow a system whose coefficients
drift, provided the drift is slow compared with the convergence time. Here the
coefficients are generated with
:func:`~asp.algorithms.time_varying_filter`:

.. plot::
   :caption: LMS tracking two time-varying coefficients (dashed: true values).

   n, Ntaps = 8000, 3
   rng = np.random.default_rng(5)
   x = rng.standard_normal(n)
   t = np.arange(n)
   G = np.zeros((n, Ntaps))
   G[:, 0] = 1.0
   G[:, 1] = 0.5 * np.sin(2 * np.pi * t / n * 2)
   G[:, 2] = 0.25 + 0.25 * np.cos(2 * np.pi * t / n * 3)
   d = alg.time_varying_filter(x, G)

   *_, H = alg.lms(x, d, Ntaps, 0.01)
   fig, ax = plt.subplots(figsize=(6.4, 3))
   ax.plot(G, 'k--', lw=0.8)
   ax.plot(H, lw=0.8)
   ax.set(xlabel='iteration $k$', title='coefficients')

.. admonition:: Pitfall
   :class: pitfall

   A step size that is too large for the input power makes LMS **diverge**
   (the coefficients grow without bound). Always check :math:`\mu` against
   :eq:`lms-mu` — or use the normalized version
   (:func:`~asp.algorithms.nlms`) when the input power
   is unknown or varies.

Exercises
---------

.. admonition:: Exercise 1
   :class: exercise

   An LMS filter with :math:`N=8` taps processes a white input with
   :math:`\sigma_x^2=0.5`. What is the largest "safe" step size? Estimate the
   misadjustment for :math:`\mu=0.01`.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

:math:`\mu<1/(N\sigma_x^2)=1/4=0.25` (a more conservative mean-square
choice would be :math:`1/(3N\sigma_x^2)\approx0.083`). With
:math:`\operatorname{tr}\mathbf{R}=N\sigma_x^2=4`, the misadjustment is
:math:`\mathcal{M}\approx\mu\operatorname{tr}\mathbf{R}=0.04`, i.e. the steady
state MSE is about 4 % above :math:`J_{\min}`.

.. raw:: html

   </details>

.. admonition:: Exercise 2
   :class: exercise

   Use :func:`~asp.algorithms.lms` to run the
   identification experiment with a coloured input (an AR(1) process with
   pole 0.95). Compare the convergence with that obtained with a white input
   of the same power and explain the difference.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

The coloured input has a large eigenvalue spread (for the AR(1) process the
spread of the :math:`N`-th order correlation matrix approaches
:math:`\left(\frac{1+0.95}{1-0.95}\right)^2\approx1520`), so the mode
associated with :math:`\lambda_{\min}` converges :math:`\approx1520` times
more slowly than for the white input for the same stability margin. See
:doc:`lms_variants` for algorithms that mitigate this problem.

.. raw:: html

   </details>

.. seealso::

   :func:`~asp.algorithms.lms`,
   :func:`~asp.algorithms.time_varying_filter`,
   notebook 4. Original reference: Widrow & Hoff (1960), Haykin (2014).
