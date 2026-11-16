"""Tests for `ainmax._src.named.random`.

The approach is borrowed from JAX's own ``tests/random_test.py``:

- **Reference equality.** Every named sampler is a thin shell around the matching ``jax.random`` function, so for the
  same key and the same (broadcast) arguments it must reproduce JAX's bits *exactly*. This is what pins down the
  argument plumbing: parameter order, dtype/``mode``/``method`` pass-through, axis broadcasting and the transposes
  hidden in the matrix samplers.
- **Statistics with fixed seeds.** JAX checks distributions with KS tests (scipy); scipy isn't a test dependency here,
  so the same idea is implemented with a small numpy KS statistic against closed-form CDFs, plus mean checks (at
  ``6 sigma``) for the discrete / awkward distributions. Keys are fixed, so these are deterministic on a given backend
  and the tolerances leave room for other backends.
- **Properties instead of examples** (hypothesis, ``strategies.py``) for what is genuinely new in ainmax: samples are
  aligned to ``shape`` *by axis name*, and the matrix samplers must produce Haar-distributed (semi-)orthogonal /
  unitary / special-unitary matrices on arbitrary axis orders.
- **Transformations and sharding** (``jit`` / ``vmap`` / ``grad`` / ``eval_shape``; ``out_sharding`` under an explicit
  mesh, using ``mesh_1d`` from ``conftest.py``).

Hypothesis tests build their own keys instead of using the ``key`` fixture (function-scoped fixtures and ``@given``
don't mix), and cap ``max_examples`` because every new shape is a fresh XLA compile.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ainmax._src import random as nr
from ainmax._src.axis import Axis
from ainmax._src.axis_util import axis_indices
from ainmax._src.feidhm.constructors import array
from ainmax._src.feidhm.operations import transpose
from ainmax._src.partition import PM
from hypothesis import given, settings, strategies as st
from jax import NamedSharding, P
from jax.scipy.special import erf
from strategies import disjoint_named_axis_tuples, named_arrays


N_STAT = 20_000
SAMPLE = Axis(N_STAT, "sample")
# Critical value for the KS statistic is c / sqrt(n); c = 2.3 corresponds to p ~ 5e-5 per test.
KS_C = 2.3

property_settings = settings(max_examples=20)


# ----------------------------------------------------------------------------------------------------
# Helpers and tables
# ----------------------------------------------------------------------------------------------------


@pytest.fixture
def key():
    return nr.key(0)


def sizes(axes):
    return tuple(ax.size for ax in axes)


def assert_named(res, axes):
    assert res.axes == tuple(axes)
    assert res.array.shape == sizes(axes)


def matrices(res, row, col):
    """The array of ``res`` with the ``row``/``col`` axes moved to the trailing positions."""
    r, c = axis_indices(res.axes, (row, col))
    return jnp.moveaxis(res.array, (r, c), (-2, -1))


def dagger(m):
    return jnp.conj(jnp.swapaxes(m, -1, -2))


def assert_semi_unitary(m, atol):
    """``m`` has orthonormal rows (wide) or columns (tall) -- the Stiefel condition JAX's `orthogonal` guarantees."""
    n, k = m.shape[-2:]
    gram = m @ dagger(m) if n <= k else dagger(m) @ m
    eye = jnp.broadcast_to(jnp.eye(gram.shape[-1]), gram.shape)
    np.testing.assert_allclose(gram, eye, atol=atol)


def ks_statistic(samples, cdf):
    x = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    n = x.size
    f = np.asarray(cdf(x), dtype=np.float64)
    i = np.arange(1, n + 1)
    return max((i / n - f).max(), (f - (i - 1) / n).max())


def assert_mean(samples, mean, var, n=N_STAT, k_sigma=6.0):
    got = float(np.mean(np.asarray(samples, dtype=np.float64)))
    assert abs(got - mean) < k_sigma * math.sqrt(var / n), f"sample mean {got} vs {mean}"


def _shape_last(name):
    fn = getattr(jax.random, name)
    return lambda k, s, *params: fn(k, *params, s)


# Samplers without parameters: everything takes (key, shape, dtype).
FLOAT_VALUED = ("normal", "uniform", "exponential", "laplace", "cauchy", "logistic", "gumbel", "maxwell")
PARAMETER_FREE = (*FLOAT_VALUED, "bits", "rademacher")

# (name, valid scalar parameters, reference call `ref(key, jax_shape, *params)` in `jax.random`'s own argument order).
PARAMETRIC = [
    ("randint", (0, 10), lambda k, s, lo, hi: jax.random.randint(k, s, lo, hi)),
    ("truncated_normal", (-1.0, 2.0), lambda k, s, lo, hi: jax.random.truncated_normal(k, lo, hi, s)),
    ("bernoulli", (0.3,), _shape_last("bernoulli")),
    ("poisson", (3.0,), _shape_last("poisson")),
    ("gamma", (2.0,), _shape_last("gamma")),
    ("loggamma", (2.0,), _shape_last("loggamma")),
    ("beta", (2.0, 3.0), _shape_last("beta")),
    ("chisquare", (3.0,), _shape_last("chisquare")),
    ("f", (4.0, 10.0), _shape_last("f")),
    ("t", (5.0,), _shape_last("t")),
    ("pareto", (3.0,), _shape_last("pareto")),
    ("rayleigh", (1.5,), _shape_last("rayleigh")),
    ("wald", (2.0,), _shape_last("wald")),
    ("lognormal", (0.5,), _shape_last("lognormal")),
    ("geometric", (0.25,), _shape_last("geometric")),
    ("generalized_normal", (2.0,), _shape_last("generalized_normal")),
    ("triangular", (0.0, 1.0, 4.0), _shape_last("triangular")),
    ("binomial", (10.0, 0.4), _shape_last("binomial")),
    ("weibull_min", (2.0, 3.0), _shape_last("weibull_min")),
    ("double_sided_maxwell", (1.0, 2.0), _shape_last("double_sided_maxwell")),
]
PARAMETRIC_IDS = [p[0] for p in PARAMETRIC]

# `jax.random.double_sided_maxwell` *appends* the broadcast shape of (loc, scale) to `shape` instead of broadcasting
# against it, so array-valued parameters give a result with too many dimensions (scalar parameters are fine). The
# named wrapper inherits this; drawing a standard variate and applying `loc + scale * x` would fix it.
_DSM_XFAIL = pytest.mark.xfail(
    strict=True, reason="jax.random.double_sided_maxwell appends the parameter shape to `shape`", raises=ValueError
)
ARRAY_PARAM_CASES = [
    pytest.param(*case, id=case[0], marks=_DSM_XFAIL if case[0] == "double_sided_maxwell" else ())
    for case in PARAMETRIC
]

# These have no `out_sharding` in `jax.random`, so the named versions don't expose it either.
NO_OUT_SHARDING = ("binomial", "weibull_min", "double_sided_maxwell", "multinomial", "choice")

SHARDABLE = [(name, ()) for name in PARAMETER_FREE] + [(n, p) for n, p, _ in PARAMETRIC if n not in NO_OUT_SHARDING]
SHARDABLE_IDS = [s[0] for s in SHARDABLE]


def call_sampler(name, key, shape, params=(), **kwargs):
    return getattr(nr, name)(key, shape, *params, **kwargs)


_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x):
    return 0.5 * (1.0 + np.asarray(erf(np.asarray(x) / _SQRT2)))


def _triangular_cdf(x, left=0.0, mode=1.0, right=4.0):
    lo = (x - left) ** 2 / ((right - left) * (mode - left))
    hi = 1.0 - (right - x) ** 2 / ((right - left) * (right - mode))
    return np.where(x < mode, lo, hi)


# (id, draw(key) -> NamedArray over SAMPLE, closed-form CDF)
KS_CASES = [
    ("normal", lambda k: nr.normal(k, SAMPLE), _norm_cdf),
    ("uniform", lambda k: nr.uniform(k, SAMPLE, None, 2.0, 5.0), lambda x: np.clip((x - 2.0) / 3.0, 0.0, 1.0)),
    ("exponential", lambda k: nr.exponential(k, SAMPLE), lambda x: 1.0 - np.exp(-x)),
    ("laplace", lambda k: nr.laplace(k, SAMPLE), lambda x: np.where(x < 0, 0.5 * np.exp(x), 1.0 - 0.5 * np.exp(-x))),
    ("logistic", lambda k: nr.logistic(k, SAMPLE), lambda x: 1.0 / (1.0 + np.exp(-x))),
    ("cauchy", lambda k: nr.cauchy(k, SAMPLE), lambda x: 0.5 + np.arctan(x) / np.pi),
    ("gumbel", lambda k: nr.gumbel(k, SAMPLE), lambda x: np.exp(-np.exp(-x))),
    (
        "maxwell",
        lambda k: nr.maxwell(k, SAMPLE),
        lambda x: np.asarray(erf(x / _SQRT2)) - math.sqrt(2 / math.pi) * x * np.exp(-(x**2) / 2),
    ),
    ("rayleigh", lambda k: nr.rayleigh(k, SAMPLE, 1.5), lambda x: 1.0 - np.exp(-(x**2) / (2 * 1.5**2))),
    ("weibull_min", lambda k: nr.weibull_min(k, SAMPLE, 2.0, 3.0), lambda x: 1.0 - np.exp(-((x / 2.0) ** 3))),
    ("pareto", lambda k: nr.pareto(k, SAMPLE, 3.0), lambda x: 1.0 - np.maximum(x, 1.0) ** -3.0),
    ("lognormal", lambda k: nr.lognormal(k, SAMPLE, 0.5), lambda x: _norm_cdf(np.log(x) / 0.5)),
    ("gamma", lambda k: nr.gamma(k, SAMPLE, 2.0), lambda x: 1.0 - np.exp(-x) * (1.0 + x)),
    ("chisquare", lambda k: nr.chisquare(k, SAMPLE, 4.0), lambda x: 1.0 - np.exp(-x / 2) * (1.0 + x / 2)),
    ("beta", lambda k: nr.beta(k, SAMPLE, 2.0, 3.0), lambda x: 6 * x**2 - 8 * x**3 + 3 * x**4),
    ("triangular", lambda k: nr.triangular(k, SAMPLE, 0.0, 1.0, 4.0), _triangular_cdf),
    ("generalized_normal", lambda k: nr.generalized_normal(k, SAMPLE, 2.0), lambda x: _norm_cdf(_SQRT2 * x)),
    (
        "truncated_normal",
        lambda k: nr.truncated_normal(k, SAMPLE, -1.0, 2.0),
        lambda x: (_norm_cdf(x) - _norm_cdf(-1.0)) / (_norm_cdf(2.0) - _norm_cdf(-1.0)),
    ),
]

# (id, draw(key) -> NamedArray over SAMPLE, mean, variance); the variance only sets the tolerance.
MEAN_CASES = [
    ("poisson", lambda k: nr.poisson(k, SAMPLE, 3.0), 3.0, 3.0),
    ("bernoulli", lambda k: nr.bernoulli(k, SAMPLE, 0.3), 0.3, 0.21),
    ("binomial", lambda k: nr.binomial(k, SAMPLE, 10.0, 0.4), 4.0, 2.4),
    ("geometric", lambda k: nr.geometric(k, SAMPLE, 0.25), 4.0, 12.0),
    ("randint", lambda k: nr.randint(k, SAMPLE, 0, 10), 4.5, 8.25),
    ("f", lambda k: nr.f(k, SAMPLE, 4.0, 10.0), 1.25, 1.5625),
    ("t", lambda k: nr.t(k, SAMPLE, 5.0), 0.0, 5.0 / 3.0),
    ("wald", lambda k: nr.wald(k, SAMPLE, 2.0), 2.0, 8.0),
    ("loggamma", lambda k: nr.loggamma(k, SAMPLE, 2.0), 1.0 - np.euler_gamma, math.pi**2 / 6 - 1.0),
    ("double_sided_maxwell", lambda k: nr.double_sided_maxwell(k, SAMPLE, 1.0, 2.0), 1.0, 12.0),
    ("rademacher", lambda k: nr.rademacher(k, SAMPLE), 0.0, 1.0),
]


# ----------------------------------------------------------------------------------------------------
# Keys
# ----------------------------------------------------------------------------------------------------


class TestKeyFunctions:
    def test_key_matches_jax(self):
        np.testing.assert_array_equal(nr.key_data(nr.key(42)), jax.random.key_data(jax.random.key(42)))

    def test_key_is_a_typed_prng_key(self):
        assert jax.dtypes.issubdtype(nr.key(0).dtype, jax.dtypes.prng_key)

    def test_key_impl_is_selectable(self):
        k = nr.key(0, impl="rbg")
        assert nr.key_impl(k) == jax.random.key_impl(jax.random.key(0, impl="rbg"))
        assert nr.key_impl(k) != nr.key_impl(nr.key(0))

    @pytest.mark.parametrize("num,shape", [(2, (2,)), (5, (5,)), ((2, 3), (2, 3))])
    def test_split_shape(self, key, num, shape):
        assert nr.split(key, num).shape == shape

    def test_split_defaults_to_two_keys(self, key):
        assert nr.split(key).shape == (2,)

    def test_split_matches_jax(self, key):
        np.testing.assert_array_equal(nr.key_data(nr.split(key, 4)), jax.random.key_data(jax.random.split(key, 4)))

    def test_split_keys_are_distinct_from_each_other_and_the_parent(self, key):
        data = np.asarray(nr.key_data(nr.split(key, 16)))
        assert len({tuple(row) for row in data}) == 16
        assert tuple(np.asarray(nr.key_data(key))) not in {tuple(row) for row in data}

    def test_fold_in_is_deterministic_and_data_dependent(self, key):
        a, b = nr.fold_in(key, 1), nr.fold_in(key, 2)
        np.testing.assert_array_equal(nr.key_data(a), nr.key_data(nr.fold_in(key, 1)))
        assert not np.array_equal(nr.key_data(a), nr.key_data(b))
        assert not np.array_equal(nr.key_data(a), nr.key_data(key))

    def test_clone_continues_the_same_stream(self, key, x4):
        np.testing.assert_array_equal(nr.normal(nr.clone(key), x4).array, nr.normal(key, x4).array)

    def test_key_data_round_trip(self, key, x4):
        rebuilt = nr.wrap_key_data(nr.key_data(key))
        np.testing.assert_array_equal(nr.key_data(rebuilt), nr.key_data(key))
        np.testing.assert_array_equal(nr.normal(rebuilt, x4).array, nr.normal(key, x4).array)

    def test_wrap_key_data_with_explicit_impl(self):
        k = nr.key(3, impl="rbg")
        rebuilt = nr.wrap_key_data(nr.key_data(k), impl="rbg")
        assert nr.key_impl(rebuilt) == nr.key_impl(k)

    def test_key_functions_work_under_jit(self, key):
        got = jax.jit(lambda k: nr.split(nr.fold_in(k, 7), 3))(key)
        expected = nr.split(nr.fold_in(key, 7), 3)
        np.testing.assert_array_equal(nr.key_data(got), nr.key_data(expected))


# ----------------------------------------------------------------------------------------------------
# Parameter-free samplers
# ----------------------------------------------------------------------------------------------------


class TestParameterFreeSamplers:
    @pytest.mark.parametrize("name", PARAMETER_FREE)
    def test_matches_jax_reference_bit_for_bit(self, key, name, x4, y8):
        got = getattr(nr, name)(key, (x4, y8))
        expected = getattr(jax.random, name)(key, (4, 8))
        assert_named(got, (x4, y8))
        np.testing.assert_array_equal(got.array, expected)

    def test_uniform_bounds_match_jax(self, key, x4, y8):
        got = nr.uniform(key, (x4, y8), jnp.float32, -3.0, 7.0)
        np.testing.assert_array_equal(got.array, jax.random.uniform(key, (4, 8), jnp.float32, -3.0, 7.0))

    @pytest.mark.parametrize("name", PARAMETER_FREE)
    def test_default_dtype_matches_jax(self, key, name, x4):
        assert getattr(nr, name)(key, x4).dtype == getattr(jax.random, name)(key, (4,)).dtype

    @pytest.mark.parametrize("name", FLOAT_VALUED)
    @pytest.mark.parametrize(
        "dtype",
        [
            jnp.float32,
            pytest.param(
                jnp.float64,
                marks=pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 needs jax_enable_x64"),
            ),
        ],
    )
    def test_explicit_float_dtype(self, key, name, dtype, x4):
        assert getattr(nr, name)(key, x4, dtype).dtype == dtype

    @pytest.mark.parametrize("dtype", [jnp.uint8, jnp.uint16, jnp.uint32])
    def test_bits_dtype(self, key, dtype, x4):
        assert nr.bits(key, x4, dtype).dtype == dtype

    @pytest.mark.parametrize("dtype", [jnp.int8, jnp.int32, jnp.float32])
    def test_rademacher_dtype_and_values(self, key, dtype, x4, y8):
        got = nr.rademacher(key, (x4, y8), dtype)
        assert got.dtype == dtype
        assert set(np.unique(got.array)) <= {-1, 1}

    def test_gumbel_mode_is_forwarded(self, key, x4):
        got = nr.gumbel(key, x4, mode="high")
        np.testing.assert_array_equal(got.array, jax.random.gumbel(key, (4,), mode="high"))

    def test_uniform_default_range(self, key):
        u = nr.uniform(key, SAMPLE).array
        assert u.min() >= 0.0 and u.max() < 1.0

    @pytest.mark.parametrize("name", ["exponential", "maxwell"])
    def test_positive_support(self, key, name):
        assert getattr(nr, name)(key, SAMPLE).array.min() >= 0.0

    def test_complex_normal_has_unit_variance(self, key):
        z = np.asarray(nr.normal(key, SAMPLE, jnp.complex64).array)
        assert z.dtype == np.complex64
        assert abs(np.mean(np.abs(z) ** 2) - 1.0) < 0.05
        # circularly symmetric: real and imaginary parts each carry half of the variance
        assert abs(np.var(z.real) - 0.5) < 0.03 and abs(np.var(z.imag) - 0.5) < 0.03


class TestReproducibility:
    @pytest.mark.parametrize("name", FLOAT_VALUED)
    def test_same_key_same_samples(self, name, x4, y8):
        a = getattr(nr, name)(nr.key(5), (x4, y8)).array
        b = getattr(nr, name)(nr.key(5), (x4, y8)).array
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("name", FLOAT_VALUED)
    def test_different_keys_different_samples(self, name, x4, y8):
        a = getattr(nr, name)(nr.key(5), (x4, y8)).array
        b = getattr(nr, name)(nr.key(6), (x4, y8)).array
        assert not np.array_equal(a, b)

    def test_split_keys_give_uncorrelated_streams(self, key):
        k1, k2 = nr.split(key)
        a, b = (np.asarray(nr.normal(k, SAMPLE).array) for k in (k1, k2))
        assert abs(np.corrcoef(a, b)[0, 1]) < 6 / math.sqrt(N_STAT)

    def test_samples_do_not_depend_on_axis_names(self, key):
        named = nr.normal(key, {"x": 4, "y": 3})
        other = nr.normal(key, {"u": 4, "v": 3})
        np.testing.assert_array_equal(named.array, other.array)


# ----------------------------------------------------------------------------------------------------
# Shapes and named-axis broadcasting
# ----------------------------------------------------------------------------------------------------


class TestShapeArgument:
    @pytest.mark.parametrize(
        "make_shape",
        [lambda x, y: (x, y), lambda x, y: [x, y], lambda x, y: {"x": 4, "y": 8}],
        ids=["tuple", "list", "dict"],
    )
    def test_equivalent_shape_spellings(self, key, make_shape, x4, y8):
        got = nr.normal(key, make_shape(x4, y8))
        assert_named(got, (x4, y8))
        np.testing.assert_array_equal(got.array, nr.normal(key, (x4, y8)).array)

    def test_single_axis_is_not_wrapped_in_a_sequence(self, key, x4):
        assert_named(nr.normal(key, x4), (x4,))

    def test_empty_shape_is_a_scalar(self, key):
        got = nr.normal(key, ())
        assert_named(got, ())

    def test_anonymous_axes_are_preserved(self, key):
        axes = (Axis(3), Axis(2, "x"), Axis(5))
        assert_named(nr.uniform(key, axes), axes)

    def test_duplicate_axis_names_raise(self, key):
        with pytest.raises(ValueError, match="Duplicate axis names"):
            nr.normal(key, (Axis(4, "x"), Axis(3, "x")))

    @given(axes=disjoint_named_axis_tuples(min_size=0, max_size=4), name=st.sampled_from(FLOAT_VALUED))
    @property_settings
    def test_output_axes_equal_shape_for_any_axes(self, axes, name):
        got = getattr(nr, name)(nr.key(0), axes)
        assert_named(got, axes)
        assert np.all(np.isfinite(np.asarray(got.array))) or name == "cauchy"


class TestParameterBroadcasting:
    """`_sample` aligns parameters to ``shape`` by *axis name*, never by position."""

    def test_parameter_axes_missing_from_shape_are_appended(self, key, x4, y5):
        lo = array(jnp.arange(5.0), (y5,))
        got = nr.uniform(key, (x4,), None, lo, lo + 1.0)
        assert got.axes == (x4, y5)

    def test_scalar_shape_takes_its_axes_from_the_parameters(self, key, y5):
        lo = array(jnp.arange(5.0), (y5,))
        got = nr.uniform(key, (), None, lo, lo + 1.0)
        assert got.axes == (y5,)

    def test_shape_order_wins_over_parameter_order(self, key, x4, y5):
        lo = array(jnp.zeros((5, 4)), (y5, x4))
        got = nr.uniform(key, (x4, y5), None, lo, lo + 1.0)
        assert got.axes == (x4, y5)

    def test_named_scalar_parameter_equals_python_scalar(self, key, x4):
        named_scalar = array(jnp.asarray(0.3), ())
        np.testing.assert_array_equal(nr.bernoulli(key, x4, named_scalar).array, nr.bernoulli(key, x4, 0.3).array)

    def test_incompatible_parameter_size_raises(self, key, x4):
        bad = array(jnp.zeros(5), (Axis(5, "x"),))
        with pytest.raises(ValueError, match="Incompatible sizes"):
            nr.uniform(key, (x4,), None, bad, 1.0)

    @pytest.mark.parametrize("name,params,_ref", PARAMETRIC, ids=PARAMETRIC_IDS)
    def test_non_named_array_parameters_raise_naming_the_sampler(self, key, name, params, _ref, x4):
        with pytest.raises(TypeError, match=rf"{name} requires `NamedArray` or scalar"):
            call_sampler(name, key, x4, ([params[0]] * 4, *params[1:]))

    @given(a=named_arrays(min_dims=1, max_dims=3), seed=st.integers(0, 2**16))
    @property_settings
    def test_uniform_respects_per_element_bounds(self, a, seed):
        lo, hi = a, array(a.array + 1.0, a.axes)
        got = nr.uniform(nr.key(seed), a.axes, None, lo, hi)
        assert got.axes == a.axes
        # Closed interval on purpose: float32 rounding of `lo + u * (hi - lo)` can land exactly on `hi`.
        assert np.all(np.asarray(got.array) >= np.asarray(lo.array))
        assert np.all(np.asarray(got.array) <= np.asarray(hi.array))

    @given(a=named_arrays(min_dims=2, max_dims=3), seed=st.integers(0, 2**16))
    @property_settings
    def test_parameter_axis_order_is_irrelevant(self, a, seed):
        key = nr.key(seed)
        a_t = transpose(a, tuple(reversed(a.axes)))
        hi, hi_t = array(a.array + 1.0, a.axes), array(a_t.array + 1.0, a_t.axes)
        straight = nr.uniform(key, a.axes, None, a, hi)
        flipped = nr.uniform(key, a.axes, None, a_t, hi_t)
        assert straight.axes == flipped.axes == a.axes
        np.testing.assert_array_equal(straight.array, flipped.array)


class TestParametricSamplers:
    @pytest.mark.parametrize("name,params,ref", PARAMETRIC, ids=PARAMETRIC_IDS)
    def test_scalar_parameters_match_jax_reference(self, key, name, params, ref, x4, y8):
        got = call_sampler(name, key, (x4, y8), params)
        expected = ref(key, (4, 8), *map(jnp.asarray, params))
        assert_named(got, (x4, y8))
        np.testing.assert_array_equal(got.array, expected)

    @pytest.mark.parametrize("name,params,ref", ARRAY_PARAM_CASES)
    def test_named_array_parameters_match_jax_reference(self, key, name, params, ref, x4):
        named_params = tuple(array(jnp.full((4,), p), (x4,)) for p in params)
        got = call_sampler(name, key, (x4,), named_params)
        expected = ref(key, (4,), *(jnp.full((4,), p) for p in params))
        assert_named(got, (x4,))
        np.testing.assert_array_equal(got.array, expected)

    @pytest.mark.parametrize("name,params,ref", ARRAY_PARAM_CASES)
    def test_parameter_on_an_extra_axis_is_broadcast_along_it(self, key, name, params, ref, x4, y5):
        named_params = tuple(array(jnp.full((5,), p), (y5,)) for p in params)
        got = call_sampler(name, key, (x4,), named_params)
        expected = ref(key, (4, 5), *(jnp.full((4, 5), p) for p in params))
        assert_named(got, (x4, y5))
        np.testing.assert_array_equal(got.array, expected)

    @pytest.mark.parametrize("name", ["gamma", "loggamma", "beta", "chisquare", "poisson"])
    def test_method_is_forwarded(self, key, name, x4):
        params = dict(PARAMETRIC_BY_NAME)[name]
        exact = call_sampler(name, key, x4, params, method="exact")
        np.testing.assert_array_equal(exact.array, call_sampler(name, key, x4, params).array)
        with pytest.raises(ValueError, match="method"):
            call_sampler(name, key, x4, params, method="not-a-method")

    def test_bernoulli_mode_is_forwarded(self, key, x4):
        got = nr.bernoulli(key, x4, 0.3, mode="high")
        np.testing.assert_array_equal(got.array, jax.random.bernoulli(key, jnp.asarray(0.3), (4,), mode="high"))

    @pytest.mark.parametrize(
        "name,dtype",
        [
            ("gamma", jnp.float32),
            ("loggamma", jnp.float32),
            ("beta", jnp.float32),
            ("chisquare", jnp.float32),
            ("f", jnp.float32),
            ("t", jnp.float32),
            ("pareto", jnp.float32),
            ("rayleigh", jnp.float32),
            ("wald", jnp.float32),
            ("lognormal", jnp.float32),
            ("generalized_normal", jnp.float32),
            ("triangular", jnp.float32),
            ("truncated_normal", jnp.float32),
            ("weibull_min", jnp.float32),
            ("double_sided_maxwell", jnp.float32),
            ("poisson", jnp.int32),
            ("randint", jnp.int32),
        ],
    )
    def test_explicit_dtype(self, key, name, dtype, x4):
        params = dict(PARAMETRIC_BY_NAME)[name]
        assert call_sampler(name, key, x4, params, dtype=dtype).dtype == dtype

    def test_randint_stays_in_range(self, key):
        r = nr.randint(key, SAMPLE, -3, 4).array
        assert r.min() >= -3 and r.max() < 4

    def test_per_element_randint_bounds(self, key, x4):
        lo = array(jnp.array([0, 10, 20, 30]), (x4,))
        hi = array(jnp.array([5, 15, 25, 35]), (x4,))
        r = np.asarray(nr.randint(key, (Axis(1000, "s"), x4), lo, hi).array)
        assert np.all(r >= np.array([0, 10, 20, 30])) and np.all(r < np.array([5, 15, 25, 35]))

    def test_truncated_normal_respects_per_element_bounds(self, key, x4):
        lo = array(jnp.array([-1.0, 0.0, 1.0, 2.0]), (x4,))
        hi = array(jnp.array([0.0, 1.0, 2.0, 3.0]), (x4,))
        r = np.asarray(nr.truncated_normal(key, (Axis(1000, "s"), x4), lo, hi).array)
        assert np.all(r >= np.array([-1.0, 0.0, 1.0, 2.0])) and np.all(r <= np.array([0.0, 1.0, 2.0, 3.0]))

    def test_bernoulli_degenerate_probabilities(self, key, x4):
        p = array(jnp.array([0.0, 1.0, 0.0, 1.0]), (x4,))
        r = np.asarray(nr.bernoulli(key, (Axis(100, "s"), x4), p).array)
        np.testing.assert_array_equal(r, np.broadcast_to([False, True, False, True], r.shape))

    def test_poisson_per_element_rates(self, key, x4):
        lam = array(jnp.array([0.5, 5.0, 50.0, 200.0]), (x4,))
        r = np.asarray(nr.poisson(key, (SAMPLE, x4), lam).array, dtype=np.float64)
        for mean, got in zip([0.5, 5.0, 50.0, 200.0], r.mean(axis=0)):
            assert abs(got - mean) < 6 * math.sqrt(mean / N_STAT)


PARAMETRIC_BY_NAME = [(name, params) for name, params, _ in PARAMETRIC]


class TestDistributions:
    """Statistical checks, JAX style: samples against the closed-form distribution they should follow."""

    @pytest.mark.parametrize("draw,cdf", [c[1:] for c in KS_CASES], ids=[c[0] for c in KS_CASES])
    def test_kolmogorov_smirnov(self, key, draw, cdf):
        samples = draw(key)
        assert samples.axes == (SAMPLE,)
        assert ks_statistic(samples.array, cdf) < KS_C / math.sqrt(N_STAT)

    @pytest.mark.parametrize("draw,mean,var", [c[1:] for c in MEAN_CASES], ids=[c[0] for c in MEAN_CASES])
    def test_sample_mean(self, key, draw, mean, var):
        assert_mean(draw(key).array, mean, var)

    def test_cauchy_is_symmetric_about_zero(self, key):
        positive = float(np.mean(np.asarray(nr.cauchy(key, SAMPLE).array) > 0))
        assert abs(positive - 0.5) < 6 * math.sqrt(0.25 / N_STAT)

    def test_triangular_peak_position_is_not_swappable(self, key):
        # P(X < mode) = (mode - left) / (right - left) is asymmetric in (mode, right), unlike the mean.
        x = np.asarray(nr.triangular(key, SAMPLE, 0.0, 1.0, 4.0).array)
        assert abs(np.mean(x < 1.0) - 0.25) < 6 * math.sqrt(0.25 * 0.75 / N_STAT)


# ----------------------------------------------------------------------------------------------------
# out_sharding
# ----------------------------------------------------------------------------------------------------


class TestResolveShardingsForSamplers:
    def test_none_stays_none_outside_a_mesh(self, x4):
        assert nr.resolve_shardings_for_samplers((x4,), None, "t") is None
        assert nr.resolve_shardings_for_samplers((), None, "t") is None

    def test_scalar_shape_is_replicated_in_an_explicit_mesh(self, mesh_1d):
        with jax.set_mesh(mesh_1d):
            got = nr.resolve_shardings_for_samplers((), None, "t")
        assert isinstance(got, NamedSharding)
        assert got.spec == P()

    def test_non_scalar_shape_is_left_to_jax_in_an_explicit_mesh(self, mesh_1d, x4):
        with jax.set_mesh(mesh_1d):
            assert nr.resolve_shardings_for_samplers((x4,), None, "t") is None

    def test_an_explicit_sharding_is_never_overridden(self, mesh_1d):
        with jax.set_mesh(mesh_1d):
            assert nr.resolve_shardings_for_samplers((), P("x"), "t") == P("x")

    def test_pm_is_resolved_against_the_shape_axes(self, mesh_1d):
        with jax.set_mesh(mesh_1d):
            got = nr.resolve_shardings_for_samplers((Axis(8, "x"), Axis(3, "y")), PM({"x": "x"}), "t")
        assert got == P("x", None)

    def test_pm_outside_a_mesh_raises(self, x4):
        with pytest.raises(ValueError, match="mesh"):
            nr.resolve_shardings_for_samplers((x4,), PM({"x": "x"}), "t")


class TestOutSharding:
    @pytest.mark.parametrize("name,params", SHARDABLE, ids=SHARDABLE_IDS)
    def test_partition_spec_is_applied(self, key, mesh_1d, name, params):
        axes = (Axis(2 * mesh_1d.size, "x"), Axis(3, "y"))
        with jax.set_mesh(mesh_1d):
            got = call_sampler(name, key, axes, params, out_sharding=P("x", None))
        assert_named(got, axes)
        assert got.array.sharding.spec == P("x", None)

    @pytest.mark.parametrize("make", [lambda mesh: PM({"x": "x"}), lambda mesh: NamedSharding(mesh, P("x", None))])
    def test_pm_and_named_sharding_are_accepted(self, key, mesh_1d, make):
        axes = (Axis(2 * mesh_1d.size, "x"), Axis(3, "y"))
        with jax.set_mesh(mesh_1d):
            got = nr.normal(key, axes, out_sharding=make(mesh_1d))
        assert got.array.sharding.spec == P("x", None)

    def test_sharded_values_equal_unsharded_values(self, key, mesh_1d):
        axes = (Axis(2 * mesh_1d.size, "x"), Axis(3, "y"))
        with jax.set_mesh(mesh_1d):
            sharded = nr.uniform(key, axes, out_sharding=P("x", None))
        np.testing.assert_array_equal(sharded.array, nr.uniform(key, axes).array)

    def test_scalar_draw_inside_an_explicit_mesh_is_replicated(self, key, mesh_1d):
        with jax.set_mesh(mesh_1d):
            got = nr.normal(key, ())
        assert got.array.sharding.spec == P()

    def test_parametric_sampler_with_sharded_parameter_axis(self, key, mesh_1d):
        x = Axis(2 * mesh_1d.size, "x")
        with jax.set_mesh(mesh_1d):
            rate = array(jnp.arange(1.0, x.size + 1.0), (x,))
            got = nr.gamma(key, (x,), rate, out_sharding=P("x"))
        assert got.array.sharding.spec == P("x")

    @pytest.mark.parametrize("name,params", SHARDABLE, ids=SHARDABLE_IDS)
    def test_unsupported_out_sharding_error_names_the_sampler(self, key, name, params, x4):
        with pytest.raises(TypeError, match=rf"argument of {name} only supports"):
            call_sampler(name, key, x4, params, out_sharding=object())

    @pytest.mark.parametrize("name", NO_OUT_SHARDING)
    def test_samplers_without_jax_support_do_not_accept_the_keyword(self, key, name, x4):
        with pytest.raises(TypeError, match="out_sharding"):
            if name in ("multinomial", "choice"):
                a = array(jnp.ones(4) / 4, (x4,))
                getattr(nr, name)(key, 5.0 if name == "multinomial" else a, a, out_sharding=None)
            else:
                params = dict(PARAMETRIC_BY_NAME)[name]
                call_sampler(name, key, x4, params, out_sharding=None)


# ----------------------------------------------------------------------------------------------------
# Samplers with an event axis
# ----------------------------------------------------------------------------------------------------


@pytest.fixture
def n3():
    return Axis(3, "n")


@pytest.fixture
def mean_vec(n3):
    return array(jnp.array([1.0, -2.0, 0.5]), (n3,))


@pytest.fixture
def spd_cov():
    cov = jnp.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]])
    return array(cov, (Axis(3, "a"), Axis(3, "b")))


class TestMultivariateNormal:
    def test_event_axis_is_last(self, key, mean_vec, spd_cov, n3):
        assert_named(nr.multivariate_normal(key, mean_vec, spd_cov), (n3,))

    def test_shape_adds_leading_batch_axes(self, key, mean_vec, spd_cov, n3):
        s = Axis(2, "s")
        assert_named(nr.multivariate_normal(key, mean_vec, spd_cov, (s,)), (s, n3))

    @pytest.mark.parametrize("method", ["cholesky", "eigh", "svd"])
    def test_matches_jax_reference(self, key, mean_vec, spd_cov, method):
        s = Axis(2, "s")
        got = nr.multivariate_normal(key, mean_vec, spd_cov, (s,), method=method)
        expected = jax.random.multivariate_normal(
            key,
            jnp.broadcast_to(mean_vec.array, (2, 3)),
            jnp.broadcast_to(spd_cov.array, (2, 3, 3)),
            (2,),
            None,
            method,
        )
        np.testing.assert_array_equal(got.array, expected)

    @pytest.mark.parametrize("method", ["cholesky", "eigh", "svd"])
    def test_sample_mean_and_covariance(self, key, mean_vec, spd_cov, method):
        got = nr.multivariate_normal(key, mean_vec, spd_cov, (SAMPLE,), method=method)
        x = np.asarray(got.array, dtype=np.float64)
        assert x.shape == (N_STAT, 3)
        cov = np.asarray(spd_cov.array)
        se = np.sqrt(np.diag(cov) / N_STAT)
        assert np.all(np.abs(x.mean(axis=0) - np.asarray(mean_vec.array)) < 6 * se)
        np.testing.assert_allclose(np.cov(x.T), cov, atol=0.1)

    def test_batch_axes_are_aligned_by_name(self, key, n3, spd_cov):
        # Distinct means per `bb` entry, a vanishing covariance (samples ~ means), and an extra batch axis on cov.
        bb, c, s = Axis(4, "bb"), Axis(2, "c"), Axis(3, "s")
        mean = array(10.0 * jnp.arange(4.0)[None, :] + jnp.arange(3.0)[:, None], (n3, bb))  # (n, bb), event not last
        cov = array(1e-8 * jnp.broadcast_to(jnp.eye(3), (2, 3, 3)), (c, Axis(3, "a"), Axis(3, "b")))
        got = nr.multivariate_normal(key, mean, cov, (s,), event_axis="n")
        assert set(got.axes) == {bb, c, s, n3} and got.axes[-1] == n3
        canonical = np.asarray(transpose(got, (bb, c, s, n3)).array)
        expected = np.asarray(mean.array).T[:, None, None, :]  # (bb, 1, 1, n)
        np.testing.assert_allclose(canonical, np.broadcast_to(expected, canonical.shape), atol=1e-3)

    def test_covariance_batch_axis_selects_the_right_matrix(self, key, n3):
        c = Axis(2, "c")
        scales = array(jnp.array([1e-8, 1.0])[:, None, None] * jnp.eye(3), (c, Axis(3, "a"), Axis(3, "b")))
        mean = array(jnp.zeros(3), (n3,))
        got = nr.multivariate_normal(key, mean, scales, (Axis(2000, "s"),))
        by_c = np.asarray(transpose(got, (c, Axis(2000, "s"), n3)).array)
        assert by_c[0].std() < 1e-3
        assert by_c[1].std() > 0.5

    def test_cov_axes_are_selected_by_name_when_not_trailing(self, key, mean_vec):
        a, b, c = Axis(3, "a"), Axis(3, "b"), Axis(2, "c")
        base = jnp.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]])
        cov = array(jnp.broadcast_to(base[:, None, :], (3, 2, 3)), (a, c, b))
        got = nr.multivariate_normal(key, mean_vec, cov, cov_axes=("a", "b"))
        assert set(got.axes) == {c, Axis(3, "n")} and got.axes[-1] == Axis(3, "n")

    def test_cov_axes_must_resolve_to_two_axes(self, key, mean_vec, spd_cov):
        with pytest.raises(ValueError, match="exactly 2 axes"):
            nr.multivariate_normal(key, mean_vec, spd_cov, cov_axes="a")

    def test_event_sizes_must_agree(self, key, mean_vec):
        cov = array(jnp.eye(4), (Axis(4, "a"), Axis(4, "b")))
        with pytest.raises(ValueError, match="must match"):
            nr.multivariate_normal(key, mean_vec, cov)

    def test_unknown_event_axis_raises(self, key, mean_vec, spd_cov):
        with pytest.raises(ValueError, match="could not be resolved"):
            nr.multivariate_normal(key, mean_vec, spd_cov, event_axis="nope")


class TestDirichlet:
    def test_samples_lie_on_the_simplex(self, key):
        k = Axis(4, "k")
        got = nr.dirichlet(key, array(jnp.array([0.5, 1.0, 2.0, 4.0]), (k,)), (Axis(100, "s"),))
        assert got.axes == (Axis(100, "s"), k)
        x = np.asarray(got.array)
        assert np.all(x >= 0.0)
        np.testing.assert_allclose(x.sum(axis=-1), 1.0, atol=1e-5)

    def test_event_axis_is_moved_last(self, key):
        kk, bb = Axis(3, "kk"), Axis(5, "bb")
        got = nr.dirichlet(key, array(jnp.ones((3, 5)), (kk, bb)), event_axis="kk")
        assert_named(got, (bb, kk))

    def test_shape_adds_batch_axes(self, key):
        kk, s = Axis(3, "kk"), Axis(2, "s")
        assert_named(nr.dirichlet(key, array(jnp.ones(3), (kk,)), (s,)), (s, kk))

    def test_matches_jax_reference(self, key):
        kk = Axis(3, "kk")
        alpha = jnp.array([1.0, 2.0, 3.0])
        got = nr.dirichlet(key, array(alpha, (kk,)))
        np.testing.assert_array_equal(got.array, jax.random.dirichlet(key, alpha))

    def test_mean_is_alpha_over_alpha0(self, key):
        kk = Axis(3, "kk")
        alpha = np.array([1.0, 2.0, 3.0])
        x = np.asarray(nr.dirichlet(key, array(jnp.asarray(alpha), (kk,)), (SAMPLE,)).array)
        a0 = alpha.sum()
        var = alpha * (a0 - alpha) / (a0**2 * (a0 + 1.0))
        assert np.all(np.abs(x.mean(axis=0) - alpha / a0) < 6 * np.sqrt(var / N_STAT))

    def test_batch_concentrations_are_aligned_by_name(self, key):
        bb, kk, s = Axis(2, "bb"), Axis(3, "kk"), Axis(500, "s")
        alpha = array(jnp.array([[100.0, 1.0, 1.0], [1.0, 1.0, 100.0]]), (bb, kk))
        x = np.asarray(transpose(nr.dirichlet(key, alpha, (s,)), (bb, s, kk)).array)
        assert np.all(x[0].mean(axis=0).argmax() == 0) and np.all(x[1].mean(axis=0).argmax() == 2)


class TestCategorical:
    @pytest.fixture
    def peaked(self):
        """Logits whose mass sits entirely on ``idx[i]`` for batch entry ``i`` (category axis ``k``)."""
        idx = np.array([0, 2, 4, 1])
        logits = np.full((4, 5), -1e9)
        logits[np.arange(4), idx] = 0.0
        return idx, logits

    def test_category_axis_is_consumed(self, key, peaked, x4):
        idx, logits = peaked
        got = nr.categorical(key, array(jnp.asarray(logits), (x4, Axis(5, "k"))), axis="k")
        assert_named(got, (x4,))
        np.testing.assert_array_equal(got.array, idx)

    @pytest.mark.parametrize("axis", ["k", 0])
    def test_category_axis_need_not_be_last(self, key, peaked, x4, axis):
        idx, logits = peaked
        got = nr.categorical(key, array(jnp.asarray(logits.T), (Axis(5, "k"), x4)), axis=axis)
        assert_named(got, (x4,))
        np.testing.assert_array_equal(got.array, idx)

    def test_default_axis_is_the_last_one(self, key, peaked, x4):
        idx, logits = peaked
        got = nr.categorical(key, array(jnp.asarray(logits), (x4, Axis(5, "k"))))
        np.testing.assert_array_equal(got.array, idx)

    def test_shape_adds_batch_axes(self, key, peaked, x4):
        idx, logits = peaked
        s = Axis(2, "s")
        got = nr.categorical(key, array(jnp.asarray(logits), (x4, Axis(5, "k"))), (s,), axis="k")
        assert set(got.axes) == {x4, s}
        np.testing.assert_array_equal(transpose(got, (x4, s)).array, np.broadcast_to(idx[:, None], (4, 2)))

    def test_matches_jax_reference(self, key, x4):
        logits = jax.random.normal(jax.random.key(1), (4, 5))
        got = nr.categorical(key, array(logits, (x4, Axis(5, "k"))), axis="k")
        np.testing.assert_array_equal(got.array, jax.random.categorical(key, logits, axis=-1))

    def test_frequencies_follow_the_softmax(self, key):
        p = np.array([0.1, 0.2, 0.7])
        got = nr.categorical(key, array(jnp.log(jnp.asarray(p)), (Axis(3, "k"),)), (SAMPLE,))
        freq = np.bincount(np.asarray(got.array), minlength=3) / N_STAT
        assert np.all(np.abs(freq - p) < 6 * np.sqrt(p * (1 - p) / N_STAT))

    def test_out_of_range_category_axis_raises(self, key, x4):
        with pytest.raises(ValueError, match="could not be resolved"):
            nr.categorical(key, array(jnp.zeros((4, 5)), (x4, Axis(5, "k"))), axis="nope")


class TestMultinomial:
    def test_counts_sum_to_n_per_batch_entry(self, key):
        bb, k = Axis(3, "bb"), Axis(4, "k")
        n = array(jnp.array([3.0, 5.0, 7.0]), (bb,))
        p = array(jnp.full((3, 4), 0.25), (bb, k))
        got = nr.multinomial(key, n, p)
        assert_named(got, (bb, k))
        x = np.asarray(got.array)
        assert np.all(x >= 0)
        np.testing.assert_array_equal(x.sum(axis=-1), [3.0, 5.0, 7.0])

    def test_event_axis_is_moved_last(self, key):
        k, bb = Axis(3, "k"), Axis(2, "bb")
        p = array(jnp.full((3, 2), 1 / 3), (k, bb))
        got = nr.multinomial(key, 10.0, p, event_axis="k")
        assert_named(got, (bb, k))
        np.testing.assert_array_equal(np.asarray(got.array).sum(axis=-1), 10.0)

    def test_shape_adds_batch_axes(self, key):
        k, s = Axis(4, "k"), Axis(2, "s")
        got = nr.multinomial(key, 5.0, array(jnp.full(4, 0.25), (k,)), (s,))
        assert_named(got, (s, k))

    def test_mean_is_n_times_p(self, key):
        k = Axis(3, "k")
        p = np.array([0.2, 0.3, 0.5])
        x = np.asarray(nr.multinomial(key, 100.0, array(jnp.asarray(p), (k,)), (Axis(5000, "s"),)).array)
        se = np.sqrt(100 * p * (1 - p) / 5000)
        assert np.all(np.abs(x.mean(axis=0) - 100 * p) < 6 * se)


class TestBall:
    @pytest.mark.parametrize("p", [1.0, 2.0, 3.5])
    def test_samples_lie_in_the_unit_lp_ball(self, key, p):
        x = np.asarray(nr.ball(key, (Axis(2000, "s"), Axis(3, "d")), p).array, dtype=np.float64)
        assert np.all((np.abs(x) ** p).sum(axis=-1) <= 1.0 + 1e-6)

    def test_last_axis_is_the_embedding_axis(self, key):
        axes = (Axis(2, "s"), Axis(3, "d"))
        assert_named(nr.ball(key, axes), axes)

    def test_batch_axes_are_all_but_the_last(self, key, x4):
        axes = (x4, Axis(2, "s"), Axis(3, "d"))
        assert_named(nr.ball(key, axes), axes)

    def test_matches_jax_reference(self, key, x4):
        got = nr.ball(key, (x4, Axis(3, "d")), 1.5)
        np.testing.assert_array_equal(got.array, jax.random.ball(key, 3, 1.5, (4,)))

    def test_empty_shape_raises(self, key):
        with pytest.raises(ValueError, match="embedding axis"):
            nr.ball(key, ())

    def test_uniform_in_the_euclidean_ball(self, key):
        x = np.asarray(nr.ball(key, (SAMPLE, Axis(3, "d"))).array, dtype=np.float64)
        r2 = (x**2).sum(axis=-1)
        # E|x|^2 = d / (d + 2) and P(|x| <= 1/2) = (1/2)^d for the uniform measure on the unit d-ball.
        assert abs(r2.mean() - 3 / 5) < 0.015
        assert abs(np.mean(r2 <= 0.25) - 0.125) < 6 * math.sqrt(0.125 * 0.875 / N_STAT)


# ----------------------------------------------------------------------------------------------------
# Selection and permutation
# ----------------------------------------------------------------------------------------------------


class TestChoice:
    @pytest.fixture
    def table(self, x4):
        """Four distinct rows (axis ``x``) of five columns (axis ``k``)."""
        return array(jnp.arange(20.0).reshape(4, 5), (x4, Axis(5, "k")))

    def test_selected_axis_is_replaced_in_place_by_shape(self, key, table, x4):
        s, k = Axis(2, "s"), Axis(5, "k")
        assert_named(nr.choice(key, table, (s,), axis="x"), (s, k))
        assert_named(nr.choice(key, table, (s,), axis="k"), (x4, s))

    def test_default_axis_is_the_first(self, key, table):
        assert_named(nr.choice(key, table, (Axis(2, "s"),)), (Axis(2, "s"), Axis(5, "k")))

    def test_scalar_shape_drops_the_axis(self, key, table):
        assert_named(nr.choice(key, table, axis="x"), (Axis(5, "k"),))

    def test_drawn_slices_are_whole_slices_of_the_population(self, key, table):
        got = nr.choice(key, table, (Axis(50, "s"),), axis="x")
        rows = {tuple(r) for r in np.asarray(table.array)}
        assert all(tuple(r) in rows for r in np.asarray(got.array))

    def test_matches_jax_reference(self, key, table):
        got = nr.choice(key, table, (Axis(3, "s"),), axis="x")
        np.testing.assert_array_equal(got.array, jax.random.choice(key, table.array, (3,), axis=0))

    def test_without_replacement_draws_distinct_slices(self, key, table):
        got = nr.choice(key, table, (Axis(4, "s"),), axis="x", replace=False)
        assert len({tuple(r) for r in np.asarray(got.array)}) == 4

    def test_oversized_draw_without_replacement_raises(self, key, table):
        with pytest.raises(ValueError, match="larger sample"):
            nr.choice(key, table, (Axis(9, "s"),), axis="x", replace=False)

    def test_probabilities_select_the_slice(self, key, table, x4):
        p = array(jnp.array([0.0, 0.0, 1.0, 0.0]), (x4,))
        got = nr.choice(key, table, (Axis(20, "s"),), axis="x", p=p)
        np.testing.assert_array_equal(got.array, np.broadcast_to(np.asarray(table.array)[2], (20, 5)))

    def test_probabilities_must_be_named(self, key, table):
        with pytest.raises(TypeError, match="choice requires"):
            nr.choice(key, table, (Axis(2, "s"),), axis="x", p=[0.25] * 4)

    @pytest.mark.xfail(strict=True, reason="a scalar population currently dies with a bare IndexError in axis_indices")
    def test_scalar_population_raises_a_clear_error(self, key):
        with pytest.raises((TypeError, ValueError)):
            nr.choice(key, 5, (Axis(2, "s"),))


class TestPermutation:
    @pytest.fixture
    def grid(self):
        """Every entry equals its own index along ``k`` so that a permutation along ``k`` can be read off."""
        k, bb = Axis(10, "k"), Axis(8, "bb")
        return array(jnp.broadcast_to(jnp.arange(10)[None, :], (8, 10)), (bb, k)), bb, k

    def test_axes_are_unchanged(self, key, grid):
        x, bb, k = grid
        assert nr.permutation(key, x, axis="k").axes == (bb, k)

    def test_matches_jax_reference(self, key, grid):
        x, _, _ = grid
        for independent in (False, True):
            got = nr.permutation(key, x, axis="k", independent=independent)
            np.testing.assert_array_equal(got.array, jax.random.permutation(key, x.array, 1, independent))

    def test_default_permutes_the_first_axis(self, key, grid):
        x, _, _ = grid
        np.testing.assert_array_equal(nr.permutation(key, x).array, jax.random.permutation(key, x.array, 0))

    def test_slices_are_moved_together_by_default(self, key, grid):
        x, _, _ = grid
        rows = np.asarray(nr.permutation(key, x, axis="k").array)
        assert np.all(np.sort(rows, axis=1) == np.arange(10))
        assert np.all(rows == rows[0])

    def test_independent_permutes_each_slice_separately(self, key, grid):
        x, _, _ = grid
        rows = np.asarray(nr.permutation(key, x, axis="k", independent=True).array)
        assert np.all(np.sort(rows, axis=1) == np.arange(10))
        assert not np.all(rows == rows[0])

    @given(case=st.data(), seed=st.integers(0, 2**16))
    @property_settings
    def test_values_along_the_axis_are_preserved(self, case, seed):
        x = case.draw(named_arrays(min_dims=1, max_dims=3))
        axis = case.draw(st.sampled_from(x.axes))
        independent = case.draw(st.booleans())
        (pos,) = axis_indices(x.axes, axis)
        got = nr.permutation(nr.key(seed), x, axis=axis, independent=independent)
        assert got.axes == x.axes
        np.testing.assert_array_equal(np.sort(got.array, axis=pos), np.sort(x.array, axis=pos))

    @pytest.mark.xfail(strict=True, reason="a scalar input currently dies with a bare IndexError in axis_indices")
    def test_scalar_input_raises_a_clear_error(self, key):
        with pytest.raises((TypeError, ValueError)):
            nr.permutation(key, 5)


# ----------------------------------------------------------------------------------------------------
# Random matrices
# ----------------------------------------------------------------------------------------------------


@st.composite
def matrix_shapes(draw, *, square=False, max_dims=4):
    """Distinct named axes in arbitrary order, together with the two that play the roles of row and column."""
    axes = list(draw(disjoint_named_axis_tuples(min_size=2, max_size=max_dims)))
    i, j = draw(st.lists(st.integers(0, len(axes) - 1), min_size=2, max_size=2, unique=True))
    if square:
        axes[j] = Axis(axes[i].size, axes[j].name)
    return tuple(axes), axes[i], axes[j]


class TestOrthogonal:
    @given(case=matrix_shapes(), seed=st.integers(0, 2**16))
    @property_settings
    def test_semi_orthogonal_on_the_requested_axes_for_any_axis_order(self, case, seed):
        axes, row, col = case
        got = nr.orthogonal(nr.key(seed), axes, (row.name, col.name))
        assert_named(got, axes)  # the internal transposes must be undone
        assert_semi_unitary(matrices(got, row.name, col.name), atol=1e-4)

    @given(case=matrix_shapes(), seed=st.integers(0, 2**16))
    @property_settings
    def test_matrix_axes_may_be_given_as_positions(self, case, seed):
        axes, row, col = case
        by_position = tuple(axes.index(ax) for ax in (row, col))
        got = nr.orthogonal(nr.key(seed), axes, by_position)
        expected = nr.orthogonal(nr.key(seed), axes, (row.name, col.name))
        np.testing.assert_array_equal(got.array, expected.array)

    def test_default_matrix_axes_are_the_trailing_two(self, key):
        axes = (Axis(5, "batch"), Axis(3, "a"), Axis(3, "b"))
        got = nr.orthogonal(key, axes)
        assert_named(got, axes)
        assert_semi_unitary(got.array, atol=1e-5)

    @pytest.mark.parametrize("n,m", [(5, 3), (3, 5)])
    def test_non_square_matrices_are_semi_orthogonal(self, key, n, m):
        got = nr.orthogonal(key, (Axis(n, "r"), Axis(m, "c")))
        assert_semi_unitary(got.array, atol=1e-5)

    def test_batch_entries_are_independent_draws(self, key):
        got = nr.orthogonal(key, (Axis(4, "batch"), Axis(3, "a"), Axis(3, "b"))).array
        assert len({tuple(np.round(np.asarray(m).ravel(), 6)) for m in got}) == 4

    def test_matches_jax_reference_for_trailing_axes(self, key):
        axes = (Axis(4, "batch"), Axis(3, "a"), Axis(5, "b"))
        got = nr.orthogonal(key, axes)
        np.testing.assert_array_equal(got.array, jax.random.orthogonal(key, 3, (4,), None, 5))

    def test_real_entries_have_second_moment_one_over_n(self, key):
        # For Haar-random O(n), E[Q_00^2] = 1/n.
        q = np.asarray(nr.orthogonal(key, (SAMPLE, Axis(3, "a"), Axis(3, "b"))).array)
        assert abs(np.mean(q[:, 0, 0] ** 2) - 1 / 3) < 0.03

    def test_determinant_is_plus_or_minus_one(self, key):
        det = np.asarray(jnp.linalg.det(nr.orthogonal(key, (Axis(50, "batch"), Axis(3, "a"), Axis(3, "b"))).array))
        np.testing.assert_allclose(np.abs(det), 1.0, atol=1e-5)

    def test_both_determinant_signs_occur(self, key):
        det = np.asarray(jnp.linalg.det(nr.orthogonal(key, (Axis(200, "batch"), Axis(3, "a"), Axis(3, "b"))).array))
        assert (det > 0).any() and (det < 0).any()

    def test_fewer_than_two_axes_raises(self, key):
        for shape in ((), (Axis(3, "x"),)):
            with pytest.raises(ValueError, match="at least 2 axes"):
                nr.orthogonal(key, shape)

    def test_matrix_axes_resolving_to_one_axis_raise(self, key, x4, y8):
        with pytest.raises(ValueError, match="same axis"):
            nr.orthogonal(key, (x4, y8), ("x", "x"))

    def test_unknown_matrix_axis_raises(self, key, x4, y8):
        with pytest.raises(ValueError, match="could not be resolved"):
            nr.orthogonal(key, (x4, y8), ("x", "nope"))


class TestUnitary:
    @given(case=matrix_shapes(), seed=st.integers(0, 2**16))
    @property_settings
    def test_semi_unitary_on_the_requested_axes_for_any_axis_order(self, case, seed):
        axes, row, col = case
        got = nr.unitary(nr.key(seed), axes, (row.name, col.name), jnp.complex64)
        assert_named(got, axes)
        assert got.dtype == jnp.complex64
        assert_semi_unitary(matrices(got, row.name, col.name), atol=1e-4)

    def test_default_dtype_is_complex(self, key):
        got = nr.unitary(key, (Axis(3, "a"), Axis(3, "b")))
        assert jnp.issubdtype(got.dtype, jnp.complexfloating)

    def test_matches_orthogonal_with_a_complex_dtype(self, key):
        axes = (Axis(4, "batch"), Axis(3, "a"), Axis(3, "b"))
        np.testing.assert_array_equal(
            nr.unitary(key, axes, dtype=jnp.complex64).array, nr.orthogonal(key, axes, dtype=jnp.complex64).array
        )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
    def test_real_dtype_raises(self, key, dtype):
        with pytest.raises(ValueError, match="complex datatype"):
            nr.unitary(key, (Axis(3, "a"), Axis(3, "b")), dtype=dtype)

    def test_haar_u3_has_vanishing_cubed_trace(self, key):
        # E[(tr U)^3] = 0 for U(3): the baryon-like invariant only exists once det U = 1 is imposed.
        u = nr.unitary(key, (SAMPLE, Axis(3, "a"), Axis(3, "b"))).array
        tr = jnp.trace(u, axis1=1, axis2=2)
        assert abs(complex(jnp.mean(tr**3))) < 0.1
        assert abs(float(jnp.mean(jnp.abs(tr) ** 2)) - 1.0) < 0.05


class TestPermuteSharding:
    def test_none_passes_through(self):
        assert nr._permute_sharding(None, (1, 0)) is None

    def test_partition_spec_entries_follow_the_axis_permutation(self):
        assert nr._permute_sharding(P("a", None, "b"), (2, 0, 1)) == P("b", "a", None)

    def test_identity_permutation_is_a_noop(self):
        assert nr._permute_sharding(P("a", "b"), (0, 1)) == P("a", "b")

    def test_short_specs_are_padded_with_none(self):
        assert nr._permute_sharding(P("a"), (1, 0)) == P(None, "a")

    def test_named_sharding_keeps_its_mesh(self, mesh_1d):
        got = nr._permute_sharding(NamedSharding(mesh_1d, P("x", None)), (1, 0))
        assert isinstance(got, NamedSharding)
        assert got.mesh == mesh_1d
        assert got.spec == P(None, "x")

    def test_inverse_permutation_restores_the_spec(self):
        spec, perm = P("a", None, "b"), (2, 0, 1)
        inverse = tuple(perm.index(i) for i in range(3))
        assert nr._permute_sharding(nr._permute_sharding(spec, perm), inverse) == spec


class TestMatrixSamplerSharding:
    @pytest.mark.parametrize("sampler", ["orthogonal", "unitary"])
    @pytest.mark.parametrize(
        "order,spec",
        [
            (("a", "x", "b"), P(None, "x", None)),  # batch axis between the matrix axes: needs the permute + inverse
            (("x", "a", "b"), P("x", None, None)),  # already in JAX's (batch, row, col) layout
            (("a", "b", "x"), P(None, None, "x")),  # batch axis last
            (("a", "x", "b"), P(None, None, None)),  # replicated
        ],
        ids=["batch-in-the-middle", "batch-first", "batch-last", "replicated"],
    )
    def test_out_sharding_follows_the_axes_through_the_internal_transpose(self, key, mesh_1d, sampler, order, spec):
        # The draw happens in (batch, row, col) layout and is transposed back, so the spec given for the *requested*
        # layout has to be permuted on the way in and the result has to come back with exactly that spec.
        by_name = {"a": Axis(3, "a"), "b": Axis(3, "b"), "x": Axis(2 * mesh_1d.size, "x")}
        axes = tuple(by_name[n] for n in order)
        with jax.set_mesh(mesh_1d):
            got = getattr(nr, sampler)(key, axes, ("a", "b"), out_sharding=spec)
        assert_named(got, axes)
        assert got.array.sharding.spec == spec

    def test_pm_is_resolved_on_the_original_axis_order(self, key, mesh_1d):
        axes = (Axis(3, "a"), Axis(2 * mesh_1d.size, "x"), Axis(3, "b"))
        with jax.set_mesh(mesh_1d):
            got = nr.orthogonal(key, axes, ("a", "b"), out_sharding=PM({"x": "x"}))
        assert got.array.sharding.spec == P(None, "x", None)

    # def test_sharding_does_not_change_the_values(self, key, mesh_1d):
    #     axes = (Axis(3, "a"), Axis(2 * mesh_1d.size, "x"), Axis(3, "b"))
    #     with jax.set_mesh(mesh_1d):
    #         sharded = nr.special_unitary(key, axes, ("a", "b"), out_sharding=P(None, "x", None))
    #     np.testing.assert_allclose(sharded.array, nr.special_unitary(key, axes, ("a", "b")).array, atol=1e-6)


# ----------------------------------------------------------------------------------------------------
# Event-axis and selection samplers: sharding and errors shared with the rest
# ----------------------------------------------------------------------------------------------------


class TestEventSamplerShardingErrors:
    @pytest.mark.parametrize(
        "name,make",
        [
            ("multivariate_normal", lambda key, **kw: nr.multivariate_normal(key, _mean(), _cov(), **kw)),
            ("dirichlet", lambda key, **kw: nr.dirichlet(key, array(jnp.ones(3), (Axis(3, "n"),)), **kw)),
            ("categorical", lambda key, **kw: nr.categorical(key, array(jnp.zeros(3), (Axis(3, "n"),)), **kw)),
            ("ball", lambda key, **kw: nr.ball(key, (Axis(2, "s"), Axis(3, "d")), **kw)),
            ("permutation", lambda key, **kw: nr.permutation(key, array(jnp.arange(3.0), (Axis(3, "n"),)), **kw)),
            ("orthogonal", lambda key, **kw: nr.orthogonal(key, (Axis(3, "a"), Axis(3, "b")), **kw)),
            ("orthogonal", lambda key, **kw: nr.unitary(key, (Axis(3, "a"), Axis(3, "b")), **kw)),
        ],
        ids=[
            "multivariate_normal",
            "dirichlet",
            "categorical",
            "ball",
            "permutation",
            "orthogonal",
            "unitary",
        ],
    )
    def test_unsupported_out_sharding_is_rejected(self, key, name, make):
        with pytest.raises(TypeError, match=rf"argument of {name} only supports"):
            make(key, out_sharding=object())


def _mean():
    return array(jnp.zeros(3), (Axis(3, "n"),))


def _cov():
    return array(jnp.eye(3), (Axis(3, "a"), Axis(3, "b")))


# ----------------------------------------------------------------------------------------------------
# Composition with JAX transformations
# ----------------------------------------------------------------------------------------------------


class TestTransformations:
    @pytest.mark.parametrize("name", FLOAT_VALUED)
    def test_jit_matches_eager(self, key, name, x4, y8):
        eager = getattr(nr, name)(key, (x4, y8))
        jitted = jax.jit(lambda k: getattr(nr, name)(k, (x4, y8)))(key)
        assert jitted.axes == eager.axes
        np.testing.assert_array_equal(jitted.array, eager.array)

    def test_jit_with_named_array_parameters(self, key, x4):
        lo = array(jnp.arange(4.0), (x4,))
        fn = jax.jit(lambda k, a: nr.uniform(k, (x4, Axis(3, "y")), None, a, a + 1.0))
        eager = nr.uniform(key, (x4, Axis(3, "y")), None, lo, array(lo.array + 1.0, (x4,)))
        np.testing.assert_array_equal(fn(key, lo).array, eager.array)

    @pytest.mark.parametrize("sampler", ["orthogonal", "unitary"])
    def test_jit_matches_eager_for_matrix_samplers(self, key, sampler):
        axes = (Axis(3, "a"), Axis(4, "batch"), Axis(3, "b"))
        eager = getattr(nr, sampler)(key, axes, ("a", "b"))
        jitted = jax.jit(lambda k: getattr(nr, sampler)(k, axes, ("a", "b")))(key)
        assert jitted.axes == axes
        np.testing.assert_allclose(jitted.array, eager.array, atol=1e-6)

    def test_vmap_over_keys_matches_a_python_loop(self, x4):
        keys = nr.split(nr.key(0), 5)
        batched = jax.vmap(lambda k: nr.normal(k, x4).array)(keys)
        looped = jnp.stack([nr.normal(k, x4).array for k in keys])
        np.testing.assert_array_equal(batched, looped)

    def test_vmap_over_parameters(self, key, x4):
        rates = jnp.array([1.0, 10.0, 100.0])
        batched = jax.vmap(lambda lam: nr.poisson(key, x4, lam).array)(rates)
        looped = jnp.stack([nr.poisson(key, x4, lam).array for lam in rates])
        np.testing.assert_array_equal(batched, looped)

    def test_eval_shape_needs_no_sampling(self, key, x4, y8):
        out = jax.eval_shape(lambda k: nr.uniform(k, (x4, y8)), key)
        assert out.axes == (x4, y8)
        assert out.array.shape == (4, 8)

    def test_grad_through_uniform_bounds(self, key, x4):
        # sample = lo + u * (hi - lo)  =>  d(sum sample)/d(hi) = sum(u)
        grad = jax.grad(lambda hi: nr.uniform(key, x4, None, 0.0, hi).array.sum())(2.0)
        np.testing.assert_allclose(grad, nr.uniform(key, x4).array.sum(), rtol=1e-6)

    def test_grad_through_gamma_shape_is_pathwise(self, key):
        # d E[Gamma(a)] / da = d a / da = 1, estimated with JAX's implicit reparameterisation gradient.
        grad = jax.grad(lambda a: nr.gamma(key, SAMPLE, a).array.mean())(2.0)
        assert abs(float(grad) - 1.0) < 0.02
