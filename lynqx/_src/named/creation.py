from typing import Literal, overload

import jax.lax
import jax.numpy as jnp
from jax.typing import DTypeLike

from lynqx._src.axis_util import (
    axis_index,
    axis_shape_to_tuple,
    axis_sizes,
    check_unique_axis_names,
)
from lynqx._src.named import constructors, util
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import (
    Axis,
    AxisLike,
    AxisSelector,
    AxisShape,
    DuckTypedNamedArray,
    NamedArray,
    NamedArrayLike,
    ShardingLike,
)


def zeros(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of zeros.

    Args:
        shape: specify the shape of the created array.
        dtype: optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.
    Returns:
        Named array of the specified shape and dtype, with the given sharding if specified.
    """

    return full(shape, 0.0, dtype, out_sharding=out_sharding)


def ones(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones.

    Args:
        shape: specify the shape of the created array.
        dtype: optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, with the given device/sharding if specified.
    """

    return full(shape, 1.0, dtype, out_sharding=out_sharding)


def empty(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an empty array.

    Args:
        shape: specify the shape of the created array.
        dtype: optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.
    Returns:
        A ``NamedArray`` of the specified shape and dtype, with the given device/sharding if specified.
    """

    return full(shape, 0.0, dtype, out_sharding=out_sharding)


def full(
    shape: AxisShape,
    fill_value: NamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array filled with a specified value.

    Args:
        shape: specify the shape of the created array.
        fill_value: scalar or array with which to fill the created array.
        dtype: optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, with the given device/sharding if specified.
    """

    target = axis_shape_to_tuple(shape)
    target_sizes = axis_sizes(target)

    jax_sharding = canonicalize_sharding(target, out_sharding, "full")

    if isinstance(fill_value, NamedArray):
        fill_value = fill_value.array

    jax_array = jnp.full(
        target_sizes,
        fill_value,
        dtype=dtype,
        out_sharding=jax_sharding,
    )

    return constructors.array(jax_array, target)


def zeros_like(
    a: NamedArrayLike | DuckTypedNamedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of zeros with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: Optionally override the shape of the created array.
        dtype: Optionally override the dtype of the created array.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, on the specified device if specified.
    """

    return full_like(a, 0.0, shape=shape, dtype=dtype, out_sharding=out_sharding)


def ones_like(
    a: NamedArrayLike | DuckTypedNamedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: Optionally override the shape of the created array.
        dtype: Optionally override the dtype of the created array.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, on the specified device if specified.
    """
    return full_like(a, 1.0, shape=shape, dtype=dtype, out_sharding=out_sharding)


def empty_like(
    a: NamedArrayLike | DuckTypedNamedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an empty array with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: Optionally override the shape of the created array.
        dtype: Optionally override the dtype of the created array.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, on the specified device if specified.
    """

    return full_like(a, 0.0, shape=shape, dtype=dtype, out_sharding=out_sharding)


def full_like(
    a: NamedArrayLike | DuckTypedNamedArray,
    fill_value: NamedArrayLike,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        fill_value: Scalar or array with which to fill the created array.
        shape: Optionally override the shape of the created array.
        dtype: Optionally override the dtype of the created array.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, on the specified device if specified.
    """

    named, fill_named = util.ensure_named("full_like", a, fill_value)
    dtype = named.dtype if dtype is None else dtype
    fill_shape = named.axes if shape is None else shape

    return full(fill_shape, fill_value, dtype, out_sharding=out_sharding)


def iota(
    shape: AxisShape,
    axis: AxisSelector | None = None,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
):
    """
    Create an array starting at zero and incrementing by one along the specified axis.

    Args:
        shape: the shape of the created array.
        axis: optional axis to increment along. Must be specified if `shape` has more than one axis.
        dtype: optional dtype for the created array.

    Returns:
        A ``NamedArray`` of ``axis.size`` spaced values, separated by `1`.
    """

    shape = axis_shape_to_tuple(shape)
    dtype = jax.dtypes.canonicalize_dtype(dtype)

    if len(shape) == 1:
        # redo this check for compatible axes instead
        if axis is not None:
            raise ValueError(f"axis={axis!r} must not be provided for 1-D shape={shape!r}")
        jax_array = jax.lax.iota(dtype, shape[0].size)
        return constructors.array(jax_array, shape, out_sharding=out_sharding)

    if axis is None:
        raise ValueError()

    axis_idx = axis_index(shape, axis)

    if axis_idx is None:
        raise ValueError()

    jax_shape = axis_sizes(shape)
    jax_sharding = canonicalize_sharding(shape, out_sharding, "iota")
    jax_array = jax.lax.broadcasted_iota(dtype, jax_shape, axis_idx, out_sharding=jax_sharding)

    return constructors.array(jax_array, shape)


def arange(
    axis: Axis,
    start: NamedArrayLike | None = None,
    step: NamedArrayLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """
    Create an array of evenly-spaced values.

    Args:
        axis: axis to apply the values along.
        start: optional start of the interval, inclusive
        step: optional end on the interval, exclusive.

    Returns:
        A ``NamedArray`` of ``axis.size`` spaced values, separated by ``step`` (defaults to ``1.0``).
    """

    if start is None:
        start = 0
    if step is None:
        step = 1

    start, step = util.ensure_scalar("arange", start, step)

    if dtype is None:
        dtype = jax.dtypes.result_type(start, step)

    if jnp.iscomplexobj(start) or jnp.iscomplexobj(step):
        raise ValueError("`arange` does not support complex start/step")

    return iota(axis, dtype=dtype, out_sharding=out_sharding) * step + start


def meshgrid(
    *xi: NamedArrayLike, sparse: bool = False, indexing: Literal["xy", "ij"] = "xy"
) -> tuple[NamedArray, ...]:
    """Return coordinate arrays over the grid spanned by one-dimensional coordinate vectors.

    Like `jax.numpy.meshgrid`, but every input carries its own `Axis`, so the grid axes are the inputs' axes and
    no extra bookkeeping is needed to say which output dimension belongs to which coordinate.

    Args:
        *xi: One-dimensional ``NamedArray``s of coordinates. Their axes must be distinct (both by value and by name).
        sparse: If ``True``, return the inputs unchanged. Broadcasting by axis name already makes them behave like the
            sparse grid of NumPy, so no size-1 axes are needed.
        indexing: ``"xy"`` (the ``jax.numpy`` default) puts the second input's axis first and the first input's axis
            second in the output, with the remaining axes in input order. ``"ij"`` keeps input order. This only
            changes the memory layout of the outputs, not which axis carries which coordinate.

    Returns:
        A tuple of ``len(xi)`` arrays, one per input, each broadcast over the axes of all inputs (or the inputs
        themselves if ``sparse``).

    Raises:
        ValueError: If an input is not one-dimensional, if two inputs share an axis, or if ``indexing`` is invalid.
    """
    xi = util.ensure_named_tuple("meshgrid", xi)
    if indexing not in ("xy", "ij"):
        raise ValueError(f"meshgrid: `indexing` must be 'xy' or 'ij', got {indexing!r}")
    for pos, x in enumerate(xi):
        if x.ndim != 1:
            raise ValueError(f"meshgrid: inputs must be one-dimensional, got shape {x.shape} at position {pos}")

    axes = tuple(x.axes[0] for x in xi)
    if len(set(axes)) != len(axes):
        raise ValueError(f"meshgrid: inputs must have distinct axes, got {axes}")
    check_unique_axis_names(axes)

    if sparse:
        return xi

    grids = jnp.meshgrid(*(x.array for x in xi), indexing=indexing)
    if indexing == "xy" and len(axes) > 1:
        axes = (axes[1], axes[0], *axes[2:])
    return tuple(constructors.array(g, axes) for g in grids)


@overload
def indices(
    dimensions: AxisShape, axis: AxisLike | None = None, dtype: DTypeLike | None = None, sparse: Literal[False] = False
) -> NamedArray: ...


@overload
def indices(
    dimensions: AxisShape, axis: AxisLike | None = None, dtype: DTypeLike | None = None, sparse: Literal[True] = True
) -> tuple[NamedArray, ...]: ...


@overload
def indices(
    dimensions: AxisShape, axis: AxisLike | None = None, dtype: DTypeLike | None = None, sparse: bool = False
) -> NamedArray | tuple[NamedArray, ...]: ...


def indices(
    dimensions: AxisShape, axis: AxisLike | None = None, dtype: DTypeLike | None = None, sparse: bool = False
) -> NamedArray | tuple[NamedArray, ...]:
    """Generate arrays of grid indices.

    Args:
        dimensions: The shape of the grid.
        axis: Optionally specify the axis to stack the generated index arrays. Raises when specified if
            ``sparse = True``.
        dtype: The dtype of the indices; Defaults to ``integer``
        sparse: If ``True``, then return sparse indices, otherwise returns dense indices. Default is ``False``.

    Returns:
        A ``NamedArray`` of shape ``(axis, *dimensions)`` if ``sparse`` is ``False``, or a sequence of rank-1 arrays
        of the same length as ``dimensions`` if ``sparse`` is ``True``.
    """

    dim_axes = axis_shape_to_tuple(dimensions)
    dim_tuple = axis_sizes(dim_axes)

    dim_axis = util.resolve_new_axis(axis, len(dim_tuple), "indices", "axis")

    jax_indices = jnp.indices(dim_tuple, dtype, sparse=sparse)

    out_axes = (dim_axis,) + dim_axes
    if sparse:
        return tuple(constructors.array(idx, ax) for idx, ax in zip(jax_indices, dim_tuple))
    return constructors.array(jax_indices, out_axes)
