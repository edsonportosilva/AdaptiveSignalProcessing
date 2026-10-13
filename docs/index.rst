Adaptive Signal Processing
==========================

*Course material and reference implementations — Graduate Program in
Electrical Engineering, Federal University of Campina Grande (UFCG).*

Adaptive filters are systems whose coefficients are tuned **automatically**,
sample by sample, from the data they process. They identify unknown systems,
cancel interference, equalize communication channels, predict signals and
track environments that change over time. This site is the companion of the
course notebooks: it explains the *theory* behind every algorithm in
:mod:`asp.algorithms` and documents the code.

.. admonition:: How to use this material
   :class: objectives

   1. Read a chapter of the :doc:`theory <theory/index>` to learn where an
      algorithm comes from and when it works.
   2. Run the matching :doc:`notebook <notebooks>` to experiment with it.
   3. Look at the :doc:`API reference <api/index>` for the exact signature
      and the conventions of each function.
   4. Test your understanding with the exercises at the end of each chapter.

.. figure:: https://github.com/edsonportosilva/AdaptiveSignalProcessing/raw/main/notebooks/figures/general-adaptive-filter.png
   :align: center
   :width: 80%
   :alt: General adaptive filter

   The general adaptive filter: the coefficients :math:`\mathbf{h}_k` are
   updated by an algorithm that minimizes a function of the error
   :math:`e[k] = d[k] - y[k]`.

A first example
---------------

Identify an unknown FIR system from its input and output with the LMS
algorithm:

.. plot::
   :caption: LMS identifying a 4-tap system. The coefficients converge to the
             true values (dashed) and the squared error decays.

   rng = np.random.default_rng(0)
   g = np.array([1.0, 0.5, -0.3, 0.1])          # unknown system
   x = rng.standard_normal(3000)                 # input
   d = np.convolve(x, g)[:len(x)]                # desired signal = system output

   out, h, sq_err, H = alg.lms(x, d, Ntaps=4, μ=0.01)

   fig, ax = plt.subplots(1, 2, figsize=(8, 3))
   ax[0].plot(H)
   ax[0].hlines(g, 0, len(x), colors='k', linestyles='--', lw=0.8)
   ax[0].set(xlabel='iteration $k$', title='coefficients $h_k$')
   ax[1].semilogy(sq_err + 1e-12)
   ax[1].set(xlabel='iteration $k$', title='squared error $e^2[k]$')
   fig.tight_layout()

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   installation
   theory/index
   notebooks
   api/index
   references
