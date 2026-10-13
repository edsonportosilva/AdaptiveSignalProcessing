Multi-layer perceptrons
=======================

.. admonition:: Learning objectives
   :class: objectives

   * Describe the forward computation of a multi-layer perceptron (MLP).
   * Derive back-propagation from the chain rule and recognise LMS as its
     one-layer special case.
   * Train a network with
     :func:`~asp.algorithms.fit` and use it for
     nonlinear regression, system identification and equalization.

Why neural networks?
--------------------

Linear adaptive filters can only model *linear* systems. When the system or
the channel has a nonlinearity (saturation, a power amplifier, a nonlinear
sensor), the optimal estimator is no longer linear. An MLP is a universal
function approximator that can be trained with the same philosophy used so
far: minimize a squared-error cost with gradient descent.

Forward propagation
-------------------

A network with :math:`L` layers computes, for an input :math:`\mathbf{z}_0=\mathbf{x}`,

.. math::

   \mathbf{a}_l=\mathbf{W}_l\mathbf{z}_{l-1}+\mathbf{b}_l,\qquad
   \mathbf{z}_l=\phi(\mathbf{a}_l),\quad l=1,\dots,L-1,\qquad
   \hat{\mathbf{y}}=\mathbf{a}_L ,

where :math:`\phi` is a nonlinear activation applied element-wise (the output
layer is linear for regression). Available activations:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Name
     - :math:`\phi(a)`
     - :math:`\phi'(a)`
   * - ``tanh``
     - :math:`\tanh a`
     - :math:`1-\tanh^2 a`
   * - ``sigmoid``
     - :math:`1/(1+e^{-a})`
     - :math:`\sigma(a)(1-\sigma(a))`
   * - ``ReLU``
     - :math:`\max(0,a)`
     - :math:`1` if :math:`a>0`, else :math:`0`

.. plot::
   :caption: Activation functions and their derivatives.

   a = np.linspace(-4, 4, 400)
   fig, ax = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
   for axi, (name, f, df) in zip(ax, (('tanh', np.tanh, alg.dtanh),
                                       ('sigmoid', alg.sigmoid, alg.dsigmoid),
                                       ('ReLU', alg.ReLU, alg.dReLU))):
       axi.plot(a, f(a), label=r'$\phi$')
       axi.plot(a, df(a), '--', label=r"$\phi'$")
       axi.set(title=name, xlabel='$a$', ylim=(-1.3, 1.6))
   ax[0].legend()
   fig.tight_layout()

Back-propagation
----------------

Training minimizes the squared error of a mini-batch,
:math:`J=\frac12\sum\|\hat{\mathbf{y}}-\mathbf{y}\|^2`. The chain rule gives
the gradient layer by layer, starting at the output with
:math:`\mathbf{e}_L=\hat{\mathbf{y}}-\mathbf{y}`:

.. math::
   :label: backprop

   \boldsymbol{\delta}_l=\mathbf{e}_l\odot\phi'(\mathbf{a}_l),\qquad
   \frac{\partial J}{\partial\mathbf{W}_l}=\boldsymbol{\delta}_l\mathbf{z}_{l-1}^T,\qquad
   \frac{\partial J}{\partial\mathbf{b}_l}=\sum_{\text{batch}}\boldsymbol{\delta}_l,\qquad
   \mathbf{e}_{l-1}=\mathbf{W}_l^T\boldsymbol{\delta}_l ,

with :math:`\phi'=1` at the (linear) output layer. The error is thus
*propagated backwards* through the transposed weights. The parameters are then
updated by gradient descent,
:math:`\mathbf{W}_l\leftarrow\mathbf{W}_l-\eta\,\partial J/\partial\mathbf{W}_l`.

.. admonition:: Key idea
   :class: keyidea

   For :math:`L=1` (no hidden layer, linear output) the equations reduce to
   :math:`\mathbf{W}\leftarrow\mathbf{W}-\eta\,\mathbf{e}\,\mathbf{x}^T`:
   exactly the **LMS algorithm**. Back-propagation is "LMS with the chain
   rule". The package's test-suite verifies this equivalence, as well as the
   gradients against finite differences.

Example: nonlinear regression
-----------------------------

.. plot::
   :caption: A network with 12 hidden ``tanh`` neurons learns
             :math:`y=\sin(x)` from 400 samples. Left: cost per epoch.
             Right: network output.

   rng = np.random.default_rng(0)
   X = rng.uniform(-3, 3, (1, 400))
   Y = np.sin(X)

   weights, biases = alg.initialize_nn([1, 12, 1])
   cost, weights, biases = alg.fit(X, Y, weights, biases, 'tanh',
                                   learning_rate=1e-3, epochs=300,
                                   batch_size=10, verbose=False)

   xt = np.linspace(-3, 3, 300)[None, :]
   fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
   ax[0].semilogy(cost); ax[0].set(xlabel='epoch', title='cost')
   ax[1].plot(X[0], Y[0], '.', ms=2, color='0.6', label='data')
   ax[1].plot(xt[0], alg.predict(xt, weights, biases, 'tanh')[0], 'r', label='MLP')
   ax[1].legend(); ax[1].set(xlabel='$x$')
   fig.tight_layout()

Nonlinear filters from MLPs
---------------------------

Feeding the network with a window of ``N`` consecutive samples of a signal
(see
:func:`~asp.algorithms.sliding_window_matrix`) turns it
into a **nonlinear FIR filter**: the output depends on
:math:`x[k+N/2],\dots,x[k-N/2]`. Trained to approximate a desired signal
it becomes a nonlinear system identifier or a nonlinear equalizer — notebook 8
compares it with the linear LMS filter in a nonlinear channel.

Practical remarks
-----------------

* Weights are initialised with small random values (not zero) to break the
  symmetry between neurons; biases start at zero.
* The learning rate :math:`\eta` plays the role of :math:`\mu` in LMS. Too
  large and the cost diverges; too small and training is slow.
* ``ReLU`` hidden units can "die" (zero output for all inputs) if the
  learning rate is too high; ``tanh`` may saturate for large inputs: scale
  the data to a moderate range.
* The gradient is summed over the examples of a mini-batch, so the effective
  step grows with ``batch_size``.

Exercises
---------

.. admonition:: Exercise
   :class: exercise

   Verify with pencil and paper that for a network with a single linear
   neuron :math:`\hat y=\mathbf{w}^T\mathbf{x}` and one example, the update
   from :eq:`backprop` is the LMS rule.

.. raw:: html

   <details class="solution"><summary>Solution</summary>

Here :math:`e=\hat y-y`, :math:`\phi'=1`, so
:math:`\delta=e` and
:math:`\partial J/\partial\mathbf{w}=e\,\mathbf{x}`. The update is
:math:`\mathbf{w}\leftarrow\mathbf{w}-\eta\,e\,\mathbf{x}=
\mathbf{w}+\eta\,(y-\hat y)\mathbf{x}`, i.e. LMS with the desired signal
:math:`d=y` and :math:`2\mu=\eta`.

.. raw:: html

   </details>

.. seealso::

   :func:`~asp.algorithms.initialize_nn`,
   :func:`~asp.algorithms.fit`,
   :func:`~asp.algorithms.predict`, notebook 8.
