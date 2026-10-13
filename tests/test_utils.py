"""Tests for ``asp.utils``."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import sympy as sp

from asp import utils


class TestRollZeropad:
    def test_positive_and_negative_shift(self):
        a = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(utils.roll_zeropad(a, 1), [0, 1, 2, 3])
        np.testing.assert_array_equal(utils.roll_zeropad(a, -1), [2, 3, 4, 0])

    def test_zero_shift_and_large_shift(self):
        a = np.array([1, 2, 3])
        np.testing.assert_array_equal(utils.roll_zeropad(a, 0), a)
        np.testing.assert_array_equal(utils.roll_zeropad(a, 10), 0)

    def test_axis(self):
        a = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(utils.roll_zeropad(a, 1, axis=1), [[0, 0, 1], [0, 3, 4]])


class TestDiscreteConvolution:
    def test_matches_numpy_convolve(self):
        # x and h have the same length N; with delay D = N the result is the
        # full linear convolution
        x = np.array([1.0, 2.0, 3.0, 4.0])
        h = np.array([0.5, 0.25, 0.125, 0.0625])
        y = utils.discreteConvolution(x, h, steps=7, D=4)
        np.testing.assert_allclose(y, np.convolve(x, h))


class TestRandomSquareSignal:
    def test_binary_and_length(self):
        np.random.seed(0)
        s = utils.random_square_signal(1000, 20, 0.5)
        assert s.shape == (1000,)
        assert set(np.unique(s)) <= {0.0, 1.0}
        assert 0.1 < s.mean() < 0.9


class TestSymbolic:
    def test_round_expr(self):
        x = sp.Symbol("x")
        assert float(utils.round_expr(sp.Float(1.23456) * x, 2).coeff(x)) == pytest.approx(1.23, abs=1e-3)

    def test_symdisp_runs(self):
        utils.symdisp("a = ", sp.Matrix([[1.23456, 2]]), numDig=2)


class TestAnimations:
    def test_gen_taps_update_gif(self, tmp_path):
        pillow = pytest.importorskip("PIL")
        H = np.cumsum(np.ones((10, 3)) * 0.1, axis=0)
        out = tmp_path / "taps.gif"
        utils.genTapsUpdateGIF(H, str(out), reference=np.array([1.0, 1.0, 1.0]),
                               xlabel="k", inter=20, writer="pillow")
        assert out.exists() and out.stat().st_size > 0

    def test_gen_conv_gif(self, tmp_path):
        pytest.importorskip("PIL")
        out = tmp_path / "conv.gif"
        utils.genConvGIF(
            np.ones(3) / 3, np.exp(-0.5 * np.arange(4)), -2, 6, str(out),
            xlabel="n", ylabel=["x[n-k]", "h[k]", "y[n]"], inter=50, writer="pillow",
        )
        assert out.exists() and out.stat().st_size > 0
