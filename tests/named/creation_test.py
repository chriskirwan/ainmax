import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st
from jax.sharding import SingleDeviceSharding
from lynqx._src.axis import Axis
from lynqx._src.named.constructors import array
from lynqx._src.named.creation import (
    arange,
    delta,
    empty,
    empty_like,
    full,
    full_like,
    geomspace,
    identity,
    iota,
    linspace,
    logspace,
    ones,
    ones_like,
    tri,
    tril,
    triu,
    zeros,
    zeros_like,
)
from lynqx._src.partition import PM
from strategies import disjoint_named_axis_tuples, named_array_and_axis_pair, named_arrays


def assert_named(res, axes, expected=None):
    assert res.axes == tuple(axes)
    assert res.array.shape == tuple(ax.size for ax in axes)
    if expected is not None:
        np.testing.assert_array_equal(res.array, expected)


class TestZeros:
    def test_shape_and_axes(self, x4, y8):
        got = zeros({"x": 4, "y": 8})
        assert got.axes == (x4, y8)
        assert got.array.shape == (4, 8)
        assert jnp.all(got.array == 0)

    def test_single_axis_not_wrapped_in_a_sequence(self, x4):
        got = zeros(x4)
        assert got.axes == (x4,)

    def test_explicit_dtype(self, x4):
        got = zeros(x4, dtype=jnp.int32)
        assert got.dtype == jnp.int32

    def test_default_dtype_matches_jnp(self, x4):
        got = zeros(x4)
        assert got.array.dtype == jnp.zeros((4,)).dtype


class TestOnes:
    def test_shape_and_values(self, x4, y8):
        got = ones({"x": 4, "y": 8})
        assert got.axes == (x4, y8)
        assert jnp.all(got.array == 1)

    def test_explicit_dtype(self, x4):
        got = ones(x4, dtype=jnp.int32)
        assert got.dtype == jnp.int32


class TestEmpty:
    def test_is_zero_initialized(self, x4):
        got = empty(x4)
        assert jnp.all(got.array == 0)


class TestFull:
    def test_fills_with_scalar_value(self, x4, y8):
        got = full({"x": 4, "y": 8}, 3.5)
        assert got.axes == (x4, y8)
        assert jnp.array_equal(got.array, jnp.full((4, 8), 3.5))

    def test_fills_with_named_array_scalar(self, x4):
        fill = array(jnp.array(9.0), ())
        got = full(x4, fill)
        assert jnp.array_equal(got.array, jnp.full((4,), 9.0))

    def test_duplicate_axis_names_raise(self):
        with pytest.raises(ValueError, match="Duplicate axis names"):
            full((Axis(4, "x"), Axis(8, "x")), 0.0)


LIKE_FUNCTIONS = (
    (zeros_like, 0.0),
    (ones_like, 1.0),
    (empty_like, 0.0),
)
LIKE_FUNCTION_IDS = ["zeros_like", "ones_like", "empty_like"]


class TestLikeFunctions:
    @pytest.mark.parametrize("fn,expected_fill", LIKE_FUNCTIONS, ids=LIKE_FUNCTION_IDS)
    def test_shape_and_axes_match_source(self, fn, expected_fill, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        got = fn(a)
        assert got.axes == a.axes
        assert jnp.array_equal(got.array, jnp.full((4, 8), expected_fill))

    @pytest.mark.parametrize("fn,_expected_fill", LIKE_FUNCTIONS, ids=LIKE_FUNCTION_IDS)
    def test_shape_argument_overrides_source_axes(self, fn, _expected_fill, x4, y8, z16):
        a = ones((x4, y8))
        got = fn(a, shape=(z16,))
        assert got.axes == (z16,)

    @pytest.mark.parametrize("fn,_expected_fill", LIKE_FUNCTIONS, ids=LIKE_FUNCTION_IDS)
    def test_dtype_defaults_to_source_array_dtype(self, fn, _expected_fill, x4):
        a = array(jnp.arange(4, dtype=jnp.int32), (x4,))
        got = fn(a)
        assert got.dtype == jnp.int32


class TestFullLike:
    def test_uses_given_fill_value_and_source_axes(self, x4, y8):
        a = zeros((x4, y8))
        got = full_like(a, 3.0)
        assert got.axes == a.axes
        assert jnp.array_equal(got.array, jnp.full((4, 8), 3.0))

    def test_shape_argument_overrides_source_axes(self, x4, y8, z16):
        a = zeros((x4, y8))
        got = full_like(a, 3.0, shape=(z16,))
        assert got.axes == (z16,)

    def test_dtype_defaults_to_source_array_dtype(self, x4):
        a = array(jnp.arange(4, dtype=jnp.int32), (x4,))
        got = full_like(a, 5)
        assert got.dtype == jnp.int32


class TestIota:
    def test_1d_matches_lax_iota(self, x4):
        got = iota(x4, dtype=jnp.int32)
        assert got.axes == (x4,)
        assert jnp.array_equal(got.array, jax.lax.iota(jnp.int32, 4))

    def test_1d_rejects_an_explicit_axis(self, x4):
        with pytest.raises(ValueError, match="must not be provided"):
            iota(x4, axis="x", dtype=jnp.int32)

    def test_multi_axis_requires_an_axis_argument(self, x4, y8):
        with pytest.raises(ValueError):
            iota((x4, y8), dtype=jnp.int32)

    def test_multi_axis_increments_along_the_named_axis(self, x4, y8):
        got = iota((x4, y8), axis="y", dtype=jnp.int32)
        want = jax.lax.broadcasted_iota(jnp.int32, (4, 8), 1)
        assert got.axes == (x4, y8)
        assert jnp.array_equal(got.array, want)

    def test_multi_axis_accepts_a_positional_axis_selector(self, x4, y8):
        got = iota((x4, y8), axis=0, dtype=jnp.int32)
        want = jax.lax.broadcasted_iota(jnp.int32, (4, 8), 0)
        assert jnp.array_equal(got.array, want)

    def test_unresolvable_axis_raises(self, x4, y8):
        with pytest.raises(ValueError):
            iota((x4, y8), axis="nope", dtype=jnp.int32)

    def test_default_dtype_matches_canonicalized_none(self, x4):
        got = iota(x4)
        assert got.array.dtype == jax.dtypes.canonicalize_dtype(None)


class TestArange:
    def test_default_start_and_step(self, x4):
        got = arange(x4, dtype=jnp.float32)
        assert got.axes == (x4,)
        assert jnp.array_equal(got.array, jnp.arange(4, dtype=jnp.float32))

    def test_explicit_start_and_step(self, x4):
        got = arange(x4, start=2, step=3, dtype=jnp.float32)
        want = 2 + 3 * jnp.arange(4, dtype=jnp.float32)
        assert jnp.array_equal(got.array, want)

    def test_dtype_inferred_as_floating_from_float_start_and_step(self, x4):
        got = arange(x4, start=1.0, step=0.5)
        assert jnp.issubdtype(got.array.dtype, jnp.floating)

    def test_dtype_inferred_as_integer_from_int_start_and_step(self, x4):
        got = arange(x4, start=1, step=2)
        assert jnp.issubdtype(got.array.dtype, jnp.integer)

    def test_complex_start_raises(self, x4):
        with pytest.raises(ValueError, match="complex"):
            arange(x4, start=1 + 2j)

    def test_complex_step_raises(self, x4):
        with pytest.raises(ValueError, match="complex"):
            arange(x4, step=1j)


class TestCreationProperties:
    @given(disjoint_named_axis_tuples(min_size=1, max_size=3))
    def test_zeros_axes_and_shape_match_spec(self, axes):
        got = zeros(axes)
        assert got.axes == axes
        assert got.array.shape == tuple(ax.size for ax in axes)
        assert jnp.all(got.array == 0)

    @given(disjoint_named_axis_tuples(min_size=1, max_size=3))
    def test_full_matches_jnp_full(self, axes):
        got = full(axes, 3.5)
        want = jnp.full(tuple(ax.size for ax in axes), 3.5)
        assert got.axes == axes
        assert jnp.array_equal(got.array, want)


class TestOutSharding:
    def test_zeros_respects_out_sharding_inside_a_mesh(self, mesh_1d):
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            got = zeros({"batch": 8}, out_sharding=pm)
        assert got.axes == (Axis(8, "batch"),)


# ``device`` kwarg (jax / Array API compatibility)

_X = Axis(4, "x")
_Y = Axis(8, "y")
_T = Axis(5, "t")


def _proto():
    return array(jnp.arange(32.0).reshape(4, 8), (_X, _Y))


# (factory(device) -> NamedArray, expected fill value; ``None`` where the contents are unspecified)
FILLED_FACTORIES = [
    pytest.param(lambda d: zeros((_X, _Y), device=d), 0.0, id="zeros"),
    pytest.param(lambda d: ones((_X, _Y), device=d), 1.0, id="ones"),
    pytest.param(lambda d: empty((_X, _Y), device=d), None, id="empty"),
    pytest.param(lambda d: full((_X, _Y), 3.0, device=d), 3.0, id="full"),
    pytest.param(lambda d: zeros_like(_proto(), device=d), 0.0, id="zeros_like"),
    pytest.param(lambda d: ones_like(_proto(), device=d), 1.0, id="ones_like"),
    pytest.param(lambda d: empty_like(_proto(), device=d), None, id="empty_like"),
    pytest.param(lambda d: full_like(_proto(), 3.0, device=d), 3.0, id="full_like"),
]


PLACEMENT_FACTORIES = [pytest.param(p.values[0], id=p.id) for p in FILLED_FACTORIES]

# Constructors that accept both ``dtype`` and ``device``.
DTYPE_FACTORIES = [
    pytest.param(lambda d: zeros(_X, jnp.int32, device=d), id="zeros"),
    pytest.param(lambda d: ones(_X, jnp.int32, device=d), id="ones"),
    pytest.param(lambda d: empty(_X, jnp.int32, device=d), id="empty"),
    pytest.param(lambda d: full(_X, 2, jnp.int32, device=d), id="full"),
]

needs_two_devices = pytest.mark.skipif(len(jax.devices()) < 2, reason="requires at least two devices")


class TestDeviceKwarg:
    @pytest.mark.parametrize("factory", PLACEMENT_FACTORIES)
    def test_result_is_committed_to_the_requested_device(self, factory):
        device = jax.devices()[0]
        assert factory(device).array.devices() == {device}

    @pytest.mark.parametrize("factory", PLACEMENT_FACTORIES)
    def test_accepts_a_single_device_sharding(self, factory):
        device = jax.devices()[0]
        assert factory(SingleDeviceSharding(device)).array.devices() == {device}

    @pytest.mark.parametrize("factory", PLACEMENT_FACTORIES)
    def test_none_places_on_the_default_device(self, factory):
        assert factory(None).array.devices() == jnp.zeros(()).devices()

    @pytest.mark.parametrize("factory", PLACEMENT_FACTORIES)
    def test_device_does_not_change_axes_or_dtype(self, factory):
        got, default = factory(jax.devices()[0]), factory(None)
        assert got.axes == default.axes
        assert got.array.shape == default.array.shape
        assert got.array.dtype == default.array.dtype

    @pytest.mark.parametrize("factory,fill", FILLED_FACTORIES)
    def test_fill_values_are_unchanged_by_device(self, factory, fill):
        if fill is None:
            pytest.skip("contents of an uninitialised array are unspecified")
        got = factory(jax.devices()[0])
        np.testing.assert_array_equal(got.array, np.full((4, 8), fill))

    @pytest.mark.parametrize("factory", DTYPE_FACTORIES)
    def test_device_combines_with_dtype(self, factory):
        device = jax.devices()[0]
        got = factory(device)
        assert got.array.dtype == jnp.int32
        assert got.array.devices() == {device}

    @pytest.mark.parametrize("fn", [zeros_like, ones_like, empty_like])
    def test_like_functions_combine_device_with_shape_override(self, fn):
        device = jax.devices()[0]
        got = fn(_proto(), shape=(Axis(16, "z"),), device=device)
        assert got.axes == (Axis(16, "z"),)
        assert got.array.devices() == {device}


class TestDeviceAndOutShardingAreExclusive:
    """Mirrors ``jax.numpy``: passing both ``device`` and ``out_sharding`` is an error."""

    EXCLUSIVE_FACTORIES = [
        pytest.param(lambda **kw: zeros({"batch": 8}, **kw), id="zeros"),
        pytest.param(lambda **kw: ones({"batch": 8}, **kw), id="ones"),
        pytest.param(lambda **kw: empty({"batch": 8}, **kw), id="empty"),
        pytest.param(lambda **kw: full({"batch": 8}, 1.0, **kw), id="full"),
    ]

    @pytest.mark.parametrize("factory", EXCLUSIVE_FACTORIES)
    def test_both_set_raises(self, factory, mesh_1d):
        with jax.set_mesh(mesh_1d):
            with pytest.raises(ValueError, match="device"):
                factory(device=jax.devices()[0], out_sharding=PM({"batch": "x"}))


@needs_two_devices
class TestNonDefaultDevice:
    """Only meaningful with several devices, e.g. ``XLA_FLAGS=--xla_force_host_platform_device_count=2``."""

    @pytest.mark.parametrize("factory", PLACEMENT_FACTORIES)
    def test_result_lands_on_a_non_default_device(self, factory):
        device = jax.devices()[-1]
        assert factory(device).array.devices() == {device}

    @pytest.mark.parametrize("factory", PLACEMENT_FACTORIES)
    def test_none_respects_the_default_device_context(self, factory):
        device = jax.devices()[-1]
        with jax.default_device(device):
            assert factory(None).array.devices() == {device}


# Creation operations moved from operations.py


class TestLinspace:
    def test_scalar_endpoints(self):
        res = linspace(0.0, 1.0, Axis(5, "t"))
        assert_named(res, (Axis(5, "t"),))
        np.testing.assert_allclose(res.array, jnp.linspace(0.0, 1.0, 5))

    def test_string_axis_uses_numpy_default_size(self):
        res = linspace(0.0, 1.0, "t")
        assert res.axes == (Axis(50, "t"),)

    def test_anonymous_axis(self):
        res = linspace(0.0, 1.0, Axis(5))
        assert res.axes == (Axis(5),)

    def test_endpoint_false(self):
        res = linspace(0.0, 1.0, Axis(4, "t"), endpoint=False)
        np.testing.assert_allclose(res.array, jnp.array([0.0, 0.25, 0.5, 0.75]))

    def test_dtype(self):
        assert linspace(0.0, 1.0, Axis(4, "t"), dtype=jnp.float16).array.dtype == jnp.float16

    def test_retstep(self):
        res, step = linspace(0.0, 1.0, Axis(5, "t"), retstep=True)
        assert res.axes == (Axis(5, "t"),)
        assert step.axes == ()
        np.testing.assert_allclose(step.array, 0.25)

    def test_new_axis_comes_first_then_input_axes(self):
        start = zeros(Axis(3, "x"))
        res = linspace(start, 1.0, Axis(5, "t"))
        assert_named(res, (Axis(5, "t"), Axis(3, "x")))
        np.testing.assert_allclose(res.array, jnp.linspace(jnp.zeros(3), 1.0, 5))

    def test_start_and_stop_with_different_axes_broadcast(self):
        s = jnp.arange(3.0)
        e = jnp.arange(4.0) + 10
        res = linspace(array(s, (Axis(3, "x"),)), array(e, (Axis(4, "y"),)), Axis(5, "t"))
        assert res.axes == (Axis(5, "t"), Axis(3, "x"), Axis(4, "y"))
        np.testing.assert_allclose(res.array, jnp.linspace(s[:, None], e[None, :], 5))

    def test_retstep_has_broadcast_axes(self):
        start = zeros(Axis(3, "x"))
        _, step = linspace(start, 1.0, Axis(5, "t"), retstep=True)
        assert step.axes == (Axis(3, "x"),)

    def test_axis_name_clashing_with_input_raises(self):
        start = zeros(Axis(3, "x"))
        with pytest.raises(ValueError, match="already exists"):
            linspace(start, 1.0, Axis(5, "x"))

    @given(named_arrays(min_dims=0, max_dims=2), st.integers(min_value=2, max_value=10))
    def test_endpoints_are_start_and_stop(self, start, n):
        stop = array(start.array + 1.0, start.axes)
        res = linspace(start, stop, Axis(n, "t"))
        assert res.axes == (Axis(n, "t"), *start.axes)
        np.testing.assert_allclose(res.array[0], start.array, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(res.array[-1], stop.array, rtol=1e-5, atol=1e-5)


class TestLogspace:
    def test_values_and_axes(self):
        res = logspace(0.0, 2.0, Axis(3, "t"))
        assert_named(res, (Axis(3, "t"),))
        np.testing.assert_allclose(res.array, jnp.array([1.0, 10.0, 100.0]), rtol=1e-5)

    def test_string_axis_default_size(self):
        assert logspace(0.0, 1.0, "t").axes == (Axis(50, "t"),)

    def test_base(self):
        res = logspace(0.0, 2.0, Axis(3, "t"), base=2.0)
        np.testing.assert_allclose(res.array, jnp.array([1.0, 2.0, 4.0]), rtol=1e-5)

    def test_endpoint_false(self):
        res = logspace(0.0, 2.0, Axis(2, "t"), endpoint=False)
        np.testing.assert_allclose(res.array, jnp.logspace(0.0, 2.0, 2, endpoint=False), rtol=1e-5)

    def test_array_inputs_add_trailing_axes(self):
        start = zeros(Axis(3, "x"))
        res = logspace(start, 2.0, Axis(4, "t"))
        assert res.axes == (Axis(4, "t"), Axis(3, "x"))
        np.testing.assert_allclose(res.array, jnp.logspace(jnp.zeros(3), 2.0, 4), rtol=1e-5)

    def test_axis_name_clashing_with_input_raises(self):
        start = zeros(Axis(3, "x"))
        with pytest.raises(ValueError, match="already exists"):
            logspace(start, 1.0, Axis(4, "x"))


class TestGeomspace:
    def test_values_and_axes(self):
        res = geomspace(1.0, 100.0, Axis(3, "t"))
        assert_named(res, (Axis(3, "t"),))
        np.testing.assert_allclose(res.array, jnp.array([1.0, 10.0, 100.0]), rtol=1e-5)

    def test_string_axis_default_size(self):
        assert geomspace(1.0, 10.0, "t").axes == (Axis(50, "t"),)

    def test_endpoint_false(self):
        res = geomspace(1.0, 100.0, Axis(2, "t"), endpoint=False)
        np.testing.assert_allclose(res.array, jnp.geomspace(1.0, 100.0, 2, endpoint=False), rtol=1e-5)

    def test_array_inputs(self):
        start = array(jnp.array([1.0, 2.0]), (Axis(2, "x"),))
        res = geomspace(start, 64.0, Axis(4, "t"))
        assert res.axes == (Axis(4, "t"), Axis(2, "x"))
        np.testing.assert_allclose(res.array, jnp.geomspace(jnp.array([1.0, 2.0]), 64.0, 4), rtol=1e-5)

    def test_axis_name_clashing_with_input_raises(self):
        start = ones(Axis(2, "x"))
        with pytest.raises(ValueError, match="already exists"):
            geomspace(start, 10.0, Axis(4, "x"))


class TestDelta:
    def test_two_axes_is_identity_matrix(self):
        res = delta({"x": 3, "y": 3}, ("x", "y"))
        assert_named(res, (Axis(3, "x"), Axis(3, "y")), jnp.eye(3))

    def test_default_dtype_is_float32(self):
        assert delta({"x": 2, "y": 2}, ("x", "y")).array.dtype == jnp.float32

    def test_explicit_dtype(self):
        assert delta({"x": 2, "y": 2}, ("x", "y"), dtype=jnp.int32).array.dtype == jnp.int32

    def test_three_axes_is_true_only_on_the_triple_diagonal(self):
        res = delta({"x": 3, "y": 3, "z": 3}, ("x", "y", "z"))
        expected = jnp.zeros((3, 3, 3)).at[jnp.arange(3), jnp.arange(3), jnp.arange(3)].set(1.0)
        assert_named(res, (Axis(3, "x"), Axis(3, "y"), Axis(3, "z")), expected)

    def test_other_axes_are_broadcast(self):
        res = delta({"b": 4, "x": 2, "y": 2}, ("x", "y"))
        assert_named(res, (Axis(4, "b"), Axis(2, "x"), Axis(2, "y")), jnp.broadcast_to(jnp.eye(2), (4, 2, 2)))

    def test_axis_order_in_selection_is_irrelevant(self):
        a = delta({"x": 3, "y": 3}, ("x", "y"))
        b = delta({"x": 3, "y": 3}, ("y", "x"))
        np.testing.assert_array_equal(a.array, b.array)

    def test_selection_by_integer_position(self):
        res = delta({"x": 3, "y": 3}, (0, 1))
        np.testing.assert_array_equal(res.array, jnp.eye(3))

    def test_non_square_gives_rectangular_identity(self):
        res = delta({"x": 2, "y": 3}, ("x", "y"))
        np.testing.assert_array_equal(res.array, jnp.eye(2, 3))

    def test_single_axis_raises(self):
        with pytest.raises(ValueError, match="at least 2 axes"):
            delta({"x": 3, "y": 3}, "x")

    def test_duplicate_axes_raise(self):
        with pytest.raises(ValueError, match="distinct axes"):
            delta({"x": 3, "y": 3}, ("x", "x"))

    @given(st.integers(min_value=1, max_value=8), st.integers(min_value=2, max_value=4))
    def test_total_mass_equals_axis_size(self, n, rank):
        names = ["x", "y", "z", "w"][:rank]
        res = delta({name: n for name in names}, tuple(names))
        assert float(res.array.sum()) == n


class TestIdentity:
    def test_matches_eye(self):
        res = identity({"x": 4, "y": 4})
        assert_named(res, (Axis(4, "x"), Axis(4, "y")), jnp.eye(4))

    def test_defaults_to_trailing_axes(self):
        res = identity({"b": 2, "x": 3, "y": 3})
        assert_named(res, (Axis(2, "b"), Axis(3, "x"), Axis(3, "y")), jnp.broadcast_to(jnp.eye(3), (2, 3, 3)))

    def test_custom_axes(self):
        res = identity({"x": 3, "b": 2, "y": 3}, ("x", "y"))
        np.testing.assert_array_equal(res.array, jnp.broadcast_to(jnp.eye(3)[:, None, :], (3, 2, 3)))

    def test_dtype(self):
        assert identity({"x": 2, "y": 2}, dtype=jnp.int32).array.dtype == jnp.int32

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            identity({"x": 3, "y": 4})

    def test_wrong_number_of_axes_raises(self):
        with pytest.raises(ValueError, match="exactly 2 axes"):
            identity({"x": 3, "y": 3, "z": 3}, ("x", "y", "z"))

    @given(st.integers(min_value=1, max_value=12))
    def test_matches_eye_for_any_size(self, n):
        np.testing.assert_array_equal(identity({"x": n, "y": n}).array, jnp.eye(n))


class TestTriuTrilTri:
    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_triu_matches_jnp(self, square, k):
        np.testing.assert_array_equal(triu(square, "x", "w", k).array, jnp.triu(square.array, k))

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_tril_matches_jnp(self, square, k):
        np.testing.assert_array_equal(tril(square, "x", "w", k).array, jnp.tril(square.array, k))

    @pytest.mark.parametrize("k", [-1, 0, 1])
    def test_tri_matches_jnp(self, k):
        res = tri({"x": 3, "w": 3}, ("x", "w"), k=k)
        np.testing.assert_array_equal(res.array, jnp.tri(3, 3, k))

    def test_swapping_axis_roles_turns_triu_into_tril(self, square):
        np.testing.assert_array_equal(triu(square, "w", "x").array, tril(square, "x", "w").array)

    def test_batch_axes_are_masked_independently(self):
        a = array(jnp.arange(18.0).reshape(2, 3, 3) + 1, (Axis(2, "b"), Axis(3, "x"), Axis(3, "w")))
        np.testing.assert_array_equal(tril(a, "x", "w").array, jnp.tril(a.array))

    def test_non_square_axes(self, mat):
        np.testing.assert_array_equal(triu(mat, "x", "y").array, jnp.triu(mat.array))

    @pytest.mark.parametrize("fn", [triu, tril])
    def test_axes_and_dtype_are_preserved(self, square, fn):
        res = fn(square, "x", "w")
        assert res.axes == square.axes
        assert res.array.dtype == square.array.dtype

    @pytest.mark.parametrize("fn", [triu, tril])
    def test_idempotent(self, square, fn):
        once = fn(square, "x", "w")
        np.testing.assert_array_equal(fn(once, "x", "w").array, once.array)

    @pytest.mark.parametrize("fn", [triu, tril])
    def test_same_axis_twice_raises(self, square, fn):
        with pytest.raises(ValueError):
            fn(square, "x", "x")

    def test_tri_dtype(self):
        assert tri({"x": 3, "w": 3}, ("x", "w"), dtype=jnp.int32).array.dtype == jnp.int32

    @given(named_array_and_axis_pair())
    def test_triu_plus_strict_tril_reassembles_the_array(self, arr_axes):
        a, ax1, ax2 = arr_axes
        upper = triu(a, ax1.name, ax2.name)
        lower = tril(a, ax1.name, ax2.name, -1)
        np.testing.assert_array_equal(upper.array + lower.array, a.array)
