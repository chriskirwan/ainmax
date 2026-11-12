import jax.numpy as jnp
import numpy as np
import pytest
from ainmax._src.axis import Axis
from ainmax._src.feidhm.constructors import array
from ainmax._src.feidhm.signal import (
    bartlett,
    blackman,
    convolve,
    correlate,
    hamming,
    hanning,
    kaiser,
    unwrap,
)
from hypothesis import given, strategies as st


def assert_named(res, axes, expected=None):
    assert res.axes == tuple(axes)
    assert res.array.shape == tuple(ax.size for ax in axes)
    if expected is not None:
        np.testing.assert_array_equal(res.array, expected)


# Window operations moved from operations.py


class TestWindows:
    @pytest.mark.parametrize(
        ("fn", "jfn"),
        [
            (bartlett, jnp.bartlett),
            (blackman, jnp.blackman),
            (hanning, jnp.hanning),
            (hamming, jnp.hamming),
        ],
    )
    @pytest.mark.parametrize("n", [1, 2, 7, 16])
    def test_matches_jnp(self, fn, jfn, n):
        res = fn(Axis(n, "t"))
        assert_named(res, (Axis(n, "t"),))
        np.testing.assert_allclose(res.array, jfn(n), atol=1e-6)

    @pytest.mark.parametrize("beta", [0.0, 2.0, 8.6])
    def test_kaiser_with_python_and_named_beta(self, beta):
        expected = jnp.kaiser(9, beta)
        np.testing.assert_allclose(kaiser(Axis(9, "t"), beta).array, expected, rtol=1e-5)
        np.testing.assert_allclose(kaiser(Axis(9, "t"), array(jnp.array(beta), ())).array, expected, rtol=1e-5)

    @pytest.mark.parametrize("fn", [bartlett, blackman, hanning])
    @given(n=st.integers(min_value=2, max_value=32))
    def test_symmetric_and_bounded(self, fn, n):
        w = fn(Axis(n, "t")).array
        np.testing.assert_allclose(w, w[::-1], atol=1e-6)
        assert float(w.max()) <= 1.0 + 1e-6

    def test_axis_object_is_preserved(self):
        axis = Axis(7, "time")
        for fn in (bartlett, blackman, hanning, hamming):
            assert fn(axis).axes == (axis,)

    def test_kaiser_preserves_axis_and_rejects_non_scalar_beta(self):
        axis = Axis(9, "time")
        result = kaiser(axis, array(jnp.array(2.0), ()))
        assert_named(result, (axis,), jnp.kaiser(9, 2.0))

        with pytest.raises(TypeError, match="incompatible shapes"):
            kaiser(axis, array(jnp.ones(2), (Axis(2, "beta"),)))


class TestUnwrap:
    def test_default_and_named_axis_match_jnp(self):
        axes = (Axis(2, "batch"), Axis(5, "phase"))
        values = jnp.array([[0.0, 2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0, 9.0]])
        signal = array(values, axes)

        for axis in (-1, "phase"):
            result = unwrap(signal, axis=axis)
            assert result.axes == axes
            np.testing.assert_allclose(result.array, jnp.unwrap(values, axis=1))

    def test_custom_period_and_discontinuity_match_jnp(self):
        values = jnp.array([0.0, 200.0, 400.0, 600.0])
        signal = array(values, (Axis(4, "angle"),))
        result = unwrap(signal, discont=150.0, axis="angle", period=360.0)
        np.testing.assert_allclose(result.array, jnp.unwrap(values, discont=150.0, period=360.0))

    def test_unknown_axis_raises(self, batch_array):
        with pytest.raises(ValueError, match="axis 'missing' not found"):
            unwrap(batch_array, axis="missing")


class TestConvolve:
    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_modes_match_jnp_and_keep_common_axis_name(self, mode):
        a = array(jnp.array([1.0, 2.0, 3.0, 4.0]), (Axis(4, "time"),))
        v = array(jnp.array([2.0, -1.0, 3.0]), (Axis(3, "time"),))
        result = convolve(a, v, mode)

        assert result.axes == (Axis(jnp.convolve(a.array, v.array, mode=mode).size, "time"),)
        np.testing.assert_allclose(result.array, jnp.convolve(a.array, v.array, mode=mode))

    def test_output_axis_name_is_inferred_from_the_named_input(self):
        a = array(jnp.array([1.0, 2.0, 3.0]), (Axis(3),))
        v = array(jnp.array([2.0, 1.0]), (Axis(2, "kernel"),))
        assert convolve(a, v).axes == (Axis(4, "kernel"),)

    def test_incompatible_axis_names_and_non_1d_inputs_raise(self):
        a = array(jnp.arange(3.0), (Axis(3, "left"),))
        b = array(jnp.arange(2.0), (Axis(2, "right"),))
        with pytest.raises(ValueError, match="incompatible names"):
            convolve(a, b)

        two_dimensional = array(jnp.ones((2, 3)), (Axis(2, "x"), Axis(3, "y")))
        with pytest.raises(ValueError, match="one-dimensional"):
            convolve(two_dimensional, b)


class TestCorrelate:
    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_modes_match_jnp(self, mode):
        a = array(jnp.array([1.0, 2.0, 3.0, 4.0]), (Axis(4, "time"),))
        v = array(jnp.array([2.0, -1.0, 3.0]), (Axis(3, "time"),))
        result = correlate(a, v, mode)

        expected = jnp.correlate(a.array, v.array, mode=mode)
        assert result.axes == (Axis(expected.size, "time"),)
        np.testing.assert_allclose(result.array, expected)

    def test_incompatible_names_and_non_1d_inputs_raise(self):
        a = array(jnp.arange(3.0), (Axis(3, "left"),))
        b = array(jnp.arange(2.0), (Axis(2, "right"),))
        with pytest.raises(ValueError, match="incompatible names"):
            correlate(a, b)

        with pytest.raises(ValueError, match="one-dimensional"):
            correlate(array(jnp.ones((2, 3)), (Axis(2, "x"), Axis(3, "y"))), b)
