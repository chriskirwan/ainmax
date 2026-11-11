import jax.numpy as jnp
import numpy as np
import pytest
from ainmax._src.axis import Axis
from ainmax._src.feipy.constructors import array
from ainmax._src.feipy.signal import (
    bartlett,
    bincount,
    blackman,
    convolve,
    correlate,
    hamming,
    hanning,
    histogram,
    histogram2d,
    histogram_bin_edges,
    histogramdd,
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


class TestBincount:
    def test_counts_minlength_and_named_output_axis(self):
        values = array(jnp.array([0, 1, 1, 3, 3, 3]), (Axis(6, "sample"),))
        result = bincount(values, minlength=5, axis="bin")

        assert_named(result, (Axis(5, "bin"),), jnp.bincount(values.array, minlength=5))

    def test_weighted_counts_and_fixed_length(self):
        values = array(jnp.array([0, 1, 1, 2]), (Axis(4, "sample"),))
        weights = array(jnp.array([0.5, 2.0, 3.0, 4.0]), (Axis(4, "weight"),))
        result = bincount(values, weights, length=5, axis=Axis(5, "bin"))

        expected = jnp.bincount(values.array, weights=weights.array, length=5)
        assert_named(result, (Axis(5, "bin"),), expected)

    def test_bad_weights_axes_and_non_1d_inputs_raise(self):
        values = array(jnp.array([0, 1, 1]), (Axis(3, "sample"),))
        with pytest.raises(ValueError, match="same number of elements"):
            bincount(values, array(jnp.ones(2), (Axis(2, "weight"),)))

        with pytest.raises(ValueError, match="axis has size 3, expected 2"):
            bincount(values, length=2, axis=Axis(3, "bin"))

        with pytest.raises(ValueError, match="one-dimensional"):
            bincount(array(jnp.ones((2, 2), dtype=jnp.int32), (Axis(2, "x"), Axis(2, "y"))))


class TestHistogram:
    def test_counts_and_edges_match_jnp(self):
        values = array(jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]), (Axis(5, "sample"),))
        counts, edges = histogram(values, bins=2, range=(0.0, 2.0), axis="bin")
        expected_counts, expected_edges = jnp.histogram(values.array, bins=2, range=(0.0, 2.0))

        assert_named(counts, (Axis(2, "bin"),), expected_counts)
        assert_named(edges, (Axis(3, "bin"),), expected_edges)

    def test_named_bins_infer_output_axis_name_and_weights_are_applied(self):
        values = array(jnp.array([0.0, 1.0, 2.0, 3.0]), (Axis(4, "sample"),))
        weights = array(jnp.array([1.0, 2.0, 3.0, 4.0]), (Axis(4, "weight"),))
        bins = array(jnp.array([0.0, 2.0, 4.0]), (Axis(3, "edges"),))
        counts, edges = histogram(values, bins=bins, weights=weights)

        expected_counts, expected_edges = jnp.histogram(values.array, bins=bins.array, weights=weights.array)
        assert_named(counts, (Axis(2, "edges"),), expected_counts)
        assert_named(edges, (Axis(3, "edges"),), expected_edges)

    def test_density_and_bad_weight_or_axis_sizes(self):
        values = array(jnp.array([0.0, 1.0, 2.0, 3.0]), (Axis(4, "sample"),))
        weights = array(jnp.ones(3), (Axis(3, "weight"),))
        with pytest.raises(ValueError, match="same number of elements"):
            histogram(values, weights=weights)

        with pytest.raises(ValueError, match="axis has size"):
            histogram(values, bins=2, axis=Axis(3, "bin"))

        counts, _ = histogram(values, bins=2, density=True)
        expected, _ = jnp.histogram(values.array, bins=2, density=True)
        np.testing.assert_allclose(counts.array, expected)


class TestHistogramBinEdges:
    def test_edges_match_jnp_and_named_bin_edges_infer_axis(self):
        values = array(jnp.array([0.0, 1.0, 2.0, 3.0]), (Axis(4, "sample"),))
        edges = histogram_bin_edges(values, bins=4, range=(0.0, 4.0), axis="bin")
        expected = jnp.histogram_bin_edges(values.array, bins=4, range=(0.0, 4.0))
        assert_named(edges, (Axis(5, "bin"),), expected)

        named_bins = array(jnp.array([0.0, 1.0, 3.0]), (Axis(3, "limits"),))
        inferred = histogram_bin_edges(values, bins=named_bins)
        assert inferred.axes == (Axis(3, "limits"),)

    def test_axis_object_must_match_edge_count(self):
        values = array(jnp.arange(4.0), (Axis(4, "sample"),))
        with pytest.raises(ValueError, match="axis has size 4, expected 3"):
            histogram_bin_edges(values, bins=2, axis=Axis(4, "bin"))


class TestHistogram2D:
    def test_counts_and_edges_match_jnp(self):
        x = array(jnp.array([0.0, 0.5, 1.0, 1.5]), (Axis(4, "x_sample"),))
        y = array(jnp.array([0.0, 1.0, 0.5, 1.5]), (Axis(4, "y_sample"),))
        result, xedges, yedges = histogram2d(
            x, y, bins=(2, 3), range=((0.0, 2.0), (0.0, 2.0)), axes=("x_bin", "y_bin")
        )
        expected, expected_xedges, expected_yedges = jnp.histogram2d(
            x.array, y.array, bins=(2, 3), range=((0.0, 2.0), (0.0, 2.0))
        )

        assert_named(result, (Axis(2, "x_bin"), Axis(3, "y_bin")), expected)
        assert_named(xedges, (Axis(3, "x_bin"),), expected_xedges)
        assert_named(yedges, (Axis(4, "y_bin"),), expected_yedges)

    def test_weights_density_and_invalid_sizes(self):
        x = array(jnp.array([0.0, 0.5, 1.0]), (Axis(3, "x"),))
        y = array(jnp.array([0.0, 0.5, 1.0]), (Axis(3, "y"),))
        weights = array(jnp.array([1.0, 2.0, 3.0]), (Axis(3, "weight"),))
        result, _, _ = histogram2d(x, y, bins=2, weights=weights, density=True, axes=("row", "col"))
        expected, _, _ = jnp.histogram2d(x.array, y.array, bins=2, weights=weights.array, density=True)
        assert_named(result, (Axis(2, "row"), Axis(2, "col")), expected)

        with pytest.raises(ValueError, match="same number of elements"):
            histogram2d(x, array(jnp.arange(2.0), (Axis(2, "y"),)))
        with pytest.raises(ValueError, match="axis has size"):
            histogram2d(x, y, bins=2, axes=(Axis(3, "row"), "col"))


class TestHistogramDD:
    def test_counts_and_edges_match_jnp(self):
        samples = jnp.array([[0.0, 0.0], [0.25, 0.5], [0.75, 0.25], [1.0, 1.0]])
        named_samples = array(samples, (Axis(4, "sample"), Axis(2, "coordinate")))
        result, edges = histogramdd(named_samples, bins=(2, 3), axes=("x_bin", "y_bin"))
        expected, expected_edges = jnp.histogramdd(samples, bins=(2, 3))

        assert_named(result, (Axis(2, "x_bin"), Axis(3, "y_bin")), expected)
        assert len(edges) == 2
        assert_named(edges[0], (Axis(3, "x_bin"),), expected_edges[0])
        assert_named(edges[1], (Axis(4, "y_bin"),), expected_edges[1])

    def test_weighted_density_histogram(self):
        samples = array(
            jnp.array([[0.0, 0.0], [0.25, 0.5], [0.75, 0.25], [1.0, 1.0]]), (Axis(4, "sample"), Axis(2, "coordinate"))
        )
        weights = array(jnp.array([1.0, 2.0, 3.0, 4.0]), (Axis(4, "weight"),))
        result, edges = histogramdd(samples, bins=2, weights=weights, density=True, axes=("row", "col"))
        expected, _ = jnp.histogramdd(samples.array, bins=2, weights=weights.array, density=True)
        assert_named(result, (Axis(2, "row"), Axis(2, "col")), expected)
        assert [edge.axes for edge in edges] == [(Axis(3, "row"),), (Axis(3, "col"),)]

    def test_invalid_sample_weights_and_axis_count_raise(self):
        samples = array(jnp.ones((3, 2)), (Axis(3, "sample"), Axis(2, "coordinate")))
        with pytest.raises(ValueError, match="one entry per sample"):
            histogramdd(samples, weights=array(jnp.ones(2), (Axis(2, "weight"),)))

        with pytest.raises(ValueError, match="sample must have shape"):
            histogramdd(array(jnp.ones(3), (Axis(3, "sample"),)))

        with pytest.raises(ValueError, match="got 1 output axes"):
            histogramdd(samples, bins=2, axes=("only_one",))
