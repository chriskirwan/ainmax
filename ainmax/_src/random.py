from typing import Literal

import jax.numpy as jnp
import jax.random
from jax import Array, NamedSharding, P
from jax.sharding import get_abstract_mesh
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import Key

from ainmax._src.axis import AxisSelection, AxisSelector, AxisShape
from ainmax._src.axis_util import axis_indices, axis_shape_to_tuple, axis_sizes, remove_axes
from ainmax._src.feidhm import array, broadcast_shapes, broadcast_to, ensure_named, ensure_named_tuple, transpose
from ainmax._src.sharding import canonicalize_sharding, ShardingLike
from ainmax._src.typing import NamedArray, NamedArrayLike
from ainmax._src.util import default_floating_dtype


PRNGKey = Key[Array, ""]
RealNamedArrayLike = NamedArrayLike
IntNamedArrayLike = NamedArrayLike


def resolve_shardings_for_samplers(shape, out_sharding, name):
    out_sharding = canonicalize_sharding(shape, out_sharding, name)
    cur_mesh = get_abstract_mesh()
    if cur_mesh.are_all_axes_explicit and out_sharding is None and not shape:
        # when shape is empty i.e. scalar, we can choose a replicated sharding.
        out_sharding = NamedSharding(cur_mesh, P())
    return out_sharding


def _align(param, out_axes):
    """Broadcast `param` to `out_axes` by name. Scalars are left alone, JAX broadcasts those for free."""
    return param if not param.axes else broadcast_to(param, out_axes)


def _finish(name, draw, out_axes, out_sharding):
    """Canonicalize the sharding for `out_axes`, call `draw(jax_sharding)` and wrap the result."""
    jax_sharding = resolve_shardings_for_samplers(out_axes, out_sharding, name)
    return array(draw(jax_sharding), out_axes)


def _sample(name, sampler, shape, params, out_sharding):
    """Shared body for elementwise-parametrised samplers.

    `sampler(jax_shape, jax_sharding, *param_arrays)` must return a jax array of shape `jax_shape`.
    """
    shape_axes = axis_shape_to_tuple(shape)
    named = ensure_named_tuple(name, params)
    if named:
        out_axes = broadcast_shapes(shape_axes, *(p.axes for p in named))
        named = tuple(_align(p, out_axes) for p in named)
    else:
        out_axes = shape_axes
    arrays = [p.array for p in named]
    jax_shape = axis_sizes(out_axes)
    return _finish(name, lambda sh: sampler(jax_shape, sh, *arrays), out_axes, out_sharding)


def _event_batch(x, event_axis, shape):
    """Split `x` into its event axis and its batch axes broadcast against `shape`."""
    (pos,) = axis_indices(x.axes, event_axis)
    batch = broadcast_shapes(remove_axes(x.axes, event_axis), axis_shape_to_tuple(shape))
    return x.axes[pos], batch


def _complex_dtype(dtype):
    if dtype is None:
        dtype = default_floating_dtype().complex
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        raise ValueError("Must be a complex datatype")
    return dtype


def _permute_sharding(sharding, perm):
    """Re-order the entries of a sharding spec the same way the axes were permuted."""
    if sharding is None:
        return None
    spec = sharding.spec if isinstance(sharding, NamedSharding) else sharding
    entries = (*spec, *((None,) * (len(perm) - len(spec))))
    permuted = P(*(entries[i] for i in perm))
    return NamedSharding(sharding.mesh, permuted) if isinstance(sharding, NamedSharding) else permuted


# Key


def key(seed: ArrayLike, dtype=None, *, impl=None) -> Array:
    return jax.random.key(seed, dtype=dtype, impl=impl)


def split(key: PRNGKey, num: int | tuple[int, ...] = 2) -> Array:
    return jax.random.split(key, num)


def fold_in(key: PRNGKey, data: ArrayLike) -> Array:
    return jax.random.fold_in(key, data)


def clone(key: PRNGKey) -> Array:
    return jax.random.clone(key)


def key_data(keys: PRNGKey) -> Array:
    return jax.random.key_data(keys)


def wrap_key_data(key_bits_array: Array, *, impl=None, dtype=None) -> Array:
    return jax.random.wrap_key_data(key_bits_array, impl=impl, dtype=dtype)


def key_impl(keys: PRNGKey):
    return jax.random.key_impl(keys)


# Parameter-free sampler


def uniform(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    minval: RealNamedArrayLike = 0.0,
    maxval: RealNamedArrayLike = 1.0,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """
    Sample uniform random values in [minval, maxval).

    Args:
        key: a PRNG key used as the random key.
        shape: An `Axis` or `Sequence[Axis]` representing the result shape. Defaults to ().
        dtype: An optional float dtype for the returned values. Defaults to `float64` if `jax_enable_x64` is true,
            otherwise `float32`.
        minval: A minimum (inclusive) value, broadcast against `shape` by axis name. Defaults to 0.
        maxval: A maximum (exclusive) value, broadcast against `shape` by axis name. Defaults to 1.
        out_sharding: Optional sharding of the result.

    Returns:
        A `NamedArray` of uniform samples.
    """

    def draw(s, sh, lo, hi):
        return jax.random.uniform(key, s, dtype, lo, hi, out_sharding=sh)

    return _sample("uniform", draw, shape, (minval, maxval), out_sharding)


def normal(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Standard normal samples (complex dtypes give unit-variance complex normals)."""

    def draw(s, sh):
        return jax.random.normal(key, s, dtype, out_sharding=sh)

    return _sample("normal", draw, shape, (), out_sharding)


def bits(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Random unsigned-integer bits."""

    def draw(s, sh):
        return jax.random.bits(key, s, dtype, out_sharding=sh)

    return _sample("bits", draw, shape, (), out_sharding)


def exponential(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Standard exponential samples."""

    def draw(s, sh):
        return jax.random.exponential(key, s, dtype, out_sharding=sh)

    return _sample("exponential", draw, shape, (), out_sharding)


def laplace(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Standard Laplace samples."""

    def draw(s, sh):
        return jax.random.laplace(key, s, dtype, out_sharding=sh)

    return _sample("laplace", draw, shape, (), out_sharding)


def cauchy(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Standard Cauchy samples."""

    def draw(s, sh):
        return jax.random.cauchy(key, s, dtype, out_sharding=sh)

    return _sample("cauchy", draw, shape, (), out_sharding)


def logistic(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Standard logistic samples."""

    def draw(s, sh):
        return jax.random.logistic(key, s, dtype, out_sharding=sh)

    return _sample("logistic", draw, shape, (), out_sharding)


def gumbel(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    mode: str | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Standard Gumbel samples."""

    def draw(s, sh):
        return jax.random.gumbel(key, s, dtype, mode, out_sharding=sh)

    return _sample("gumbel", draw, shape, (), out_sharding)


def maxwell(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Maxwell samples (chi distribution with 3 degrees of freedom)."""

    def draw(s, sh):
        return jax.random.maxwell(key, s, dtype, out_sharding=sh)

    return _sample("maxwell", draw, shape, (), out_sharding)


def rademacher(
    key: PRNGKey,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Samples of +1 / -1 with equal probability."""

    def draw(s, sh):
        return jax.random.rademacher(key, s, dtype, out_sharding=sh)

    return _sample("rademacher", draw, shape, (), out_sharding)


# Samplers with named-array parameters (broadcast against `shape` by axis name


def randint(
    key: PRNGKey,
    shape: AxisShape,
    minval: IntNamedArrayLike,
    maxval: IntNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Random integers in [minval, maxval)."""

    def draw(s, sh, lo, hi):
        return jax.random.randint(key, s, lo, hi, dtype, out_sharding=sh)

    return _sample("randint", draw, shape, (minval, maxval), out_sharding)


def truncated_normal(
    key: PRNGKey,
    shape: AxisShape,
    lower: RealNamedArrayLike,
    upper: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Standard normal samples truncated to [lower, upper]."""

    def draw(s, sh, lo, hi):
        return jax.random.truncated_normal(key, lo, hi, s, dtype, out_sharding=sh)

    return _sample("truncated_normal", draw, shape, (lower, upper), out_sharding)


def bernoulli(
    key: PRNGKey,
    shape: AxisShape,
    p: RealNamedArrayLike,
    mode: Literal["high", "low"] = "low",
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Bernoulli samples with success probability `p`."""

    def draw(s, sh, p_):
        return jax.random.bernoulli(key, p_, s, mode, out_sharding=sh)

    return _sample("bernoulli", draw, shape, (p,), out_sharding)


def poisson(
    key: PRNGKey,
    shape: AxisShape,
    lam: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Poisson samples with rate `lam`."""

    def draw(s, sh, lam_):
        return jax.random.poisson(key, lam_, s, dtype, method=method, out_sharding=sh)

    return _sample("poisson", draw, shape, (lam,), out_sharding)


def gamma(
    key: PRNGKey,
    shape: AxisShape,
    a: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Gamma(a, 1) samples."""

    def draw(s, sh, a_):
        return jax.random.gamma(key, a_, s, dtype, method=method, out_sharding=sh)

    return _sample("gamma", draw, shape, (a,), out_sharding)


def loggamma(
    key: PRNGKey,
    shape: AxisShape,
    a: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Log of Gamma(a, 1) samples, computed stably."""

    def draw(s, sh, a_):
        return jax.random.loggamma(key, a_, s, dtype, method=method, out_sharding=sh)

    return _sample("loggamma", draw, shape, (a,), out_sharding)


def beta(
    key: PRNGKey,
    shape: AxisShape,
    a: RealNamedArrayLike,
    b: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Beta(a, b) samples."""

    def draw(s, sh, a_, b_):
        return jax.random.beta(key, a_, b_, s, dtype, method=method, out_sharding=sh)

    return _sample("beta", draw, shape, (a, b), out_sharding)


def chisquare(
    key: PRNGKey,
    shape: AxisShape,
    df: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    method: str = "exact",
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Chi-square samples with `df` degrees of freedom."""

    def draw(s, sh, df_):
        return jax.random.chisquare(key, df_, s, dtype, method=method, out_sharding=sh)

    return _sample("chisquare", draw, shape, (df,), out_sharding)


def f(
    key: PRNGKey,
    shape: AxisShape,
    dfnum: RealNamedArrayLike,
    dfden: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """F-distribution samples."""

    def draw(s, sh, num, den):
        return jax.random.f(key, num, den, s, dtype, out_sharding=sh)

    return _sample("f", draw, shape, (dfnum, dfden), out_sharding)


def t(
    key: PRNGKey,
    shape: AxisShape,
    df: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Student's t samples with `df` degrees of freedom."""

    def draw(s, sh, df_):
        return jax.random.t(key, df_, s, dtype, out_sharding=sh)

    return _sample("t", draw, shape, (df,), out_sharding)


def pareto(
    key: PRNGKey,
    shape: AxisShape,
    b: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Pareto samples with shape parameter `b`."""

    def draw(s, sh, b_):
        return jax.random.pareto(key, b_, s, dtype, out_sharding=sh)

    return _sample("pareto", draw, shape, (b,), out_sharding)


def rayleigh(
    key: PRNGKey,
    shape: AxisShape,
    scale: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Rayleigh samples."""

    def draw(s, sh, scale_):
        return jax.random.rayleigh(key, scale_, s, dtype, out_sharding=sh)

    return _sample("rayleigh", draw, shape, (scale,), out_sharding)


def wald(
    key: PRNGKey,
    shape: AxisShape,
    mean: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Wald (inverse Gaussian) samples."""

    def draw(s, sh, mean_):
        return jax.random.wald(key, mean_, s, dtype, out_sharding=sh)

    return _sample("wald", draw, shape, (mean,), out_sharding)


def lognormal(
    key: PRNGKey,
    shape: AxisShape = (),
    sigma: RealNamedArrayLike = 1.0,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Log-normal samples (underlying normal has mean 0 and standard deviation `sigma`)."""

    def draw(s, sh, sigma_):
        return jax.random.lognormal(key, sigma_, s, dtype, out_sharding=sh)

    return _sample("lognormal", draw, shape, (sigma,), out_sharding)


def geometric(
    key: PRNGKey,
    shape: AxisShape,
    p: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Geometric samples (number of trials up to and including the first success)."""

    def draw(s, sh, p_):
        return jax.random.geometric(key, p_, s, dtype, out_sharding=sh)

    return _sample("geometric", draw, shape, (p,), out_sharding)


def generalized_normal(
    key: PRNGKey,
    shape: AxisShape,
    p: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Generalized normal samples with shape parameter `p`."""

    def draw(s, sh, p_):
        return jax.random.generalized_normal(key, p_, s, dtype, out_sharding=sh)

    return _sample("generalized_normal", draw, shape, (p,), out_sharding)


def triangular(
    key: PRNGKey,
    shape: AxisShape,
    left: RealNamedArrayLike,
    mode: RealNamedArrayLike,
    right: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Triangular samples on [left, right] with peak at `mode`."""

    def draw(s, sh, left_, mode_, right_):
        return jax.random.triangular(key, left_, mode_, right_, s, dtype, out_sharding=sh)

    return _sample("triangular", draw, shape, (left, mode, right), out_sharding)


# The following have no `out_sharding` in jax.random.


def binomial(
    key: PRNGKey,
    shape: AxisShape,
    n: RealNamedArrayLike,
    p: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
) -> NamedArray:
    """Binomial samples with `n` trials and success probability `p`."""

    def draw(s, sh, n_, p_):
        return jax.random.binomial(key, n_, p_, s, dtype)

    return _sample("binomial", draw, shape, (n, p), None)


def weibull_min(
    key: PRNGKey,
    shape: AxisShape,
    scale: RealNamedArrayLike,
    concentration: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
) -> NamedArray:
    """Weibull (minimum) samples."""

    def draw(s, sh, scale_, conc_):
        return jax.random.weibull_min(key, scale_, conc_, s, dtype)

    return _sample("weibull_min", draw, shape, (scale, concentration), None)


def double_sided_maxwell(
    key: PRNGKey,
    shape: AxisShape,
    loc: RealNamedArrayLike,
    scale: RealNamedArrayLike,
    dtype: DTypeLike | None = None,
) -> NamedArray:
    """Double-sided Maxwell samples."""

    def draw(s, sh, loc_, scale_):
        return jax.random.double_sided_maxwell(key, loc_, scale_, s, dtype)

    return _sample("double_sided_maxwell", draw, shape, (loc, scale), None)


# Samplers with an event axi


def multivariate_normal(
    key: PRNGKey,
    mean: NamedArrayLike,
    cov: NamedArrayLike,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    event_axis: AxisSelector = -1,
    cov_axes: AxisSelection = (-2, -1),
    method: Literal["cholesky", "eigh", "svd"] = "cholesky",
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """
    Multivariate normal samples. The event axis is last in the result.

    Args:
        key: a PRNG key used as the random key.
        mean: NamedArray whose `event_axis` (size n) is the mean vector; any other axes are batch axes.
        cov: NamedArray whose `cov_axes` (a pair of size-n axes) is the covariance matrix; any other axes are
            batch axes, broadcast against mean's batch axes (and `shape`) by name.
        shape: extra batch axes to broadcast the result to.
        dtype: optional float dtype of the result.
        event_axis: selector (name/position) for mean's event axis.
        cov_axes: selectors for cov's two event axes. Defaults to the trailing two, matching
            `identity`/`delta`'s convention. Both must match `event_axis` in size.
        method: factorization of `cov`: "cholesky", "eigh" or "svd".
        out_sharding: optional sharding of the result (which includes the event axis).
    """
    mean, cov = ensure_named("multivariate_normal", mean, cov)

    n_axis, mean_batch = _event_batch(mean, event_axis, ())

    cov_positions = axis_indices(cov.axes, cov_axes)
    if len(cov_positions) != 2:
        raise ValueError(f"multivariate_normal: `cov_axes` must resolve to exactly 2 axes, got {cov_axes!r}")
    cov_row, cov_col = (cov.axes[i] for i in cov_positions)
    if cov_row.size != n_axis.size or cov_col.size != n_axis.size:
        raise ValueError(
            f"multivariate_normal: cov's event axes {cov_row}/{cov_col} must match mean's event axis {n_axis} in size"
        )

    cov_batch = remove_axes(cov.axes, cov_axes)
    batch_axes = broadcast_shapes(mean_batch, cov_batch, axis_shape_to_tuple(shape))
    out_axes = (*batch_axes, n_axis)

    mean = broadcast_to(mean, out_axes)
    cov = broadcast_to(cov, (*batch_axes, cov_row, cov_col))
    batch_sizes = axis_sizes(batch_axes)

    def draw(sh):
        return jax.random.multivariate_normal(key, mean.array, cov.array, batch_sizes, dtype, method, out_sharding=sh)

    return _finish("multivariate_normal", draw, out_axes, out_sharding)


def dirichlet(
    key: PRNGKey,
    alpha: RealNamedArrayLike,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    event_axis: AxisSelector = -1,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """
    Dirichlet samples. `alpha`'s `event_axis` holds the concentrations; other axes (and `shape`) are batch axes.
    The event axis is last in the result.
    """
    alpha = ensure_named("dirichlet", alpha)
    event, batch = _event_batch(alpha, event_axis, shape)
    out_axes = (*batch, event)
    alpha = broadcast_to(alpha, out_axes)
    batch_sizes = axis_sizes(batch)

    def draw(sh):
        return jax.random.dirichlet(key, alpha.array, batch_sizes, dtype, out_sharding=sh)

    return _finish("dirichlet", draw, out_axes, out_sharding)


def categorical(
    key: PRNGKey,
    logits: RealNamedArrayLike,
    shape: AxisShape = (),
    *,
    axis: AxisSelector = -1,
    mode: str | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """
    Sample category indices (with replacement) from unnormalized log-probabilities.

    `logits`' `axis` is the category axis and is consumed; its other axes (and `shape`) are batch axes of the
    result. Sampling without replacement (`replace=False` in JAX) is not exposed.
    """
    logits = ensure_named("categorical", logits)
    category, batch = _event_batch(logits, axis, shape)
    logits = broadcast_to(logits, (*batch, category))
    batch_sizes = axis_sizes(batch)

    def draw(sh):
        return jax.random.categorical(key, logits.array, -1, batch_sizes, mode=mode, out_sharding=sh)

    return _finish("categorical", draw, batch, out_sharding)


def multinomial(
    key: PRNGKey,
    n: RealNamedArrayLike,
    p: RealNamedArrayLike,
    shape: AxisShape = (),
    dtype: DTypeLike | None = None,
    *,
    event_axis: AxisSelector = -1,
    unroll: int | bool = 1,
) -> NamedArray:
    """
    Multinomial counts for `n` trials with category probabilities `p` along `p`'s `event_axis`.
    The event axis is last in the result.
    """
    n, p = ensure_named("multinomial", n, p)
    event, p_batch = _event_batch(p, event_axis, shape)
    batch = broadcast_shapes(p_batch, n.axes)
    out_axes = (*batch, event)
    p = broadcast_to(p, out_axes)
    n = _align(n, batch)
    batch_sizes = axis_sizes(batch)

    def draw(sh):
        return jax.random.multinomial(key, n.array, p.array, shape=batch_sizes, dtype=dtype, unroll=unroll)

    return _finish("multinomial", draw, out_axes, None)


def ball(
    key: PRNGKey,
    shape: AxisShape,
    p: float = 2,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Uniform samples from the unit l^p ball. The *last* axis of `shape` is the embedding dimension."""
    axis_tuple = axis_shape_to_tuple(shape)
    if not axis_tuple:
        raise ValueError("ball: `shape` must end with the embedding axis")
    *batch, dim = axis_tuple
    batch_sizes = axis_sizes(batch)

    def draw(sh):
        return jax.random.ball(key, dim.size, p, batch_sizes, dtype, out_sharding=sh)

    return _finish("ball", draw, axis_tuple, out_sharding)


# Selection and permutatio


def choice(
    key: PRNGKey,
    a: NamedArrayLike,
    shape: AxisShape = (),
    *,
    axis: AxisSelector = 0,
    replace: bool = True,
    p: NamedArrayLike | None = None,
) -> NamedArray:
    """
    Draw samples along `axis` of `a`; that axis is replaced in the result by the axes of `shape`.

    `p`, if given, must be 1-D with the same length as `a`'s `axis`. Integer `a` (as in `jax.random.choice`) is not
    supported because it would have no axis name; pass `named.arange(...)` instead.
    """
    a = ensure_named("choice", a)
    (pos,) = axis_indices(a.axes, axis)
    shape_axes = axis_shape_to_tuple(shape)
    out_axes = (*a.axes[:pos], *shape_axes, *a.axes[pos + 1 :])
    p_array = None if p is None else ensure_named("choice", p).array
    sizes = axis_sizes(shape_axes)

    def draw(sh):
        return jax.random.choice(key, a.array, sizes, replace=replace, p=p_array, axis=pos)

    return _finish("choice", draw, out_axes, None)


def permutation(
    key: PRNGKey,
    x: NamedArrayLike,
    axis: AxisSelector = 0,
    independent: bool = False,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Randomly permute `x` along `axis` (independently along every other axis if `independent`)."""
    x = ensure_named("permutation", x)
    (pos,) = axis_indices(x.axes, axis)

    def draw(sh):
        return jax.random.permutation(key, x.array, pos, independent, out_sharding=sh)

    return _finish("permutation", draw, x.axes, out_sharding)


# Random matrice


def orthogonal(
    key: PRNGKey,
    shape: AxisShape,
    matrix_axes: tuple[AxisSelector, AxisSelector] = (-2, -1),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """
    Haar-distributed random orthogonal (unitary for complex `dtype`) matrices on the `matrix` axes of `shape`.
    Non-square matrices give random semi-orthogonal frames. All other axes are batch axes.
    """
    axis_tuple = axis_shape_to_tuple(shape)
    if len(axis_tuple) < 2:
        raise ValueError("must have at least 2 axes")

    row_pos, col_pos = axis_indices(axis_tuple, matrix_axes)
    if row_pos == col_pos:
        raise ValueError(f"`matrix` {matrix_axes} resolve to the same axis (position {row_pos})")

    # JAX wants (*batch, row, col); draw in that layout and permute back.
    batch_pos = [i for i in range(len(axis_tuple)) if i not in (row_pos, col_pos)]
    perm = (*batch_pos, row_pos, col_pos)
    *batch_axes, row_axis, col_axis = (axis_tuple[i] for i in perm)

    jax_sharding = resolve_shardings_for_samplers(axis_tuple, out_sharding, "orthogonal")
    jax_array = jax.random.orthogonal(
        key,
        row_axis.size,
        axis_sizes(batch_axes),
        dtype,
        col_axis.size,
        out_sharding=_permute_sharding(jax_sharding, perm),
    )
    drawn = array(jax_array, (*batch_axes, row_axis, col_axis))

    if perm == tuple(range(len(axis_tuple))):
        return drawn
    # `perm[k]` is the original position of the k-th drawn axis, so the way back is its inverse.
    inverse = [0] * len(perm)
    for k, original in enumerate(perm):
        inverse[original] = k
    return transpose(drawn, tuple(inverse))


def unitary(
    key: PRNGKey,
    shape: AxisShape,
    matrix_axes: tuple[AxisSelector, AxisSelector] = (-2, -1),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Haar-distributed random unitary matrices (`orthogonal` with a complex dtype)."""
    return orthogonal(key, shape, matrix_axes, _complex_dtype(dtype), out_sharding=out_sharding)
