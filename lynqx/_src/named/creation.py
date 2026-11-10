import operator
from functools import reduce
from typing import Literal, overload

import equinox as eqx
import jax.lax
import jax.numpy as jnp
from jax import Device
from jax.sharding import Sharding
from jax.typing import DTypeLike

from lynqx._src.axis import Axis, AxisLike, AxisSelection, AxisSelector, AxisShape
from lynqx._src.axis_util import (
    axis_index,
    axis_indices,
    axis_names,
    axis_shape_to_tuple,
    axis_sizes,
    check_unique_axis_names,
)
from lynqx._src.named import constructors, util
from lynqx._src.partition import ShardingLike
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import (
    DuckTypedNamedArray,
    NamedArray,
    NamedArrayLike,
)


def zeros(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    device: Device | Sharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of zeros.

    Args:
        shape: Specify the shape of the created array.
        dtype: Optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        device: Optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
            committed. This argument exists for compatibility with the :ref:`python-array-api`.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.
    Returns:
        A ``NamedArray`` of the specified shape and dtype, with the given sharding if specified.
    """

    target = axis_shape_to_tuple(shape)
    target_sizes = axis_sizes(target)

    jax_sharding = canonicalize_sharding(target, out_sharding, "full")

    jax_array = jnp.zeros(target_sizes, dtype=dtype, out_sharding=jax_sharding, device=device)

    return constructors.array(jax_array, target)


def ones(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    device: Device | Sharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones.

    Args:
        shape: Specify the shape of the created array.
        dtype: Optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        device: Optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
            committed. This argument exists for compatibility with the :ref:`python-array-api`.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.
    Returns:
        A ``NamedArray`` of the specified shape and dtype, with the given sharding if specified.
    """

    target = axis_shape_to_tuple(shape)
    target_sizes = axis_sizes(target)

    jax_sharding = canonicalize_sharding(target, out_sharding, "full")

    jax_array = jnp.ones(target_sizes, dtype=dtype, device=device, out_sharding=jax_sharding)

    return constructors.array(jax_array, target)


def empty(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    device: Device | Sharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an empty array.

    Args:
        shape: Specify the shape of the created array.
        dtype: Optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        device: Optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
            committed. This argument exists for compatibility with the :ref:`python-array-api`.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.
    Returns:
        A ``NamedArray`` of the specified shape and dtype, with the given sharding if specified.
    """

    target = axis_shape_to_tuple(shape)
    target_sizes = axis_sizes(target)

    jax_sharding = canonicalize_sharding(target, out_sharding, "full")

    jax_array = jnp.empty(target_sizes, dtype=dtype, device=device, out_sharding=jax_sharding)

    return constructors.array(jax_array, target)


def full(
    shape: AxisShape,
    fill_value: NamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    device: Device | Sharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array filled with a specified value.

    Args:
        shape: Specify the shape of the created array.
        fill_value: Scalar or array with which to fill the created array.
        dtype: Optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        device: Optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
                    committed. This argument exists for compatibility with the :ref:`python-array-api`.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
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
        device=device,
        out_sharding=jax_sharding,
    )

    return constructors.array(jax_array, target)


def zeros_like(
    prototype: NamedArrayLike | DuckTypedNamedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    device: Device | Sharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of zeros with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: Optionally override the shape of the created array.
        dtype: Optionally override the dtype of the created array.
        device: Optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
                    committed. This argument exists for compatibility with the :ref:`python-array-api`.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, on the specified device if specified.
    """
    prototype = util.ensure_named("full_like", prototype)
    dtype = prototype.dtype if dtype is None else dtype
    target_shape = axis_shape_to_tuple(prototype.axes if shape is None else shape)

    jax_sharding = canonicalize_sharding(target_shape, out_sharding, "zeros_like")
    jax_array = jnp.zeros_like(
        prototype, shape=axis_sizes(target_shape), dtype=dtype, device=device, out_sharding=jax_sharding
    )

    return constructors.array(jax_array, target_shape)


def ones_like(
    prototype: NamedArrayLike | DuckTypedNamedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    device: Device | Sharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: Optionally override the shape of the created array.
        dtype: Optionally override the dtype of the created array.
        device: Optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
                    committed. This argument exists for compatibility with the :ref:`python-array-api`.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, on the specified device if specified.
    """
    prototype = util.ensure_named("full_like", prototype)
    dtype = prototype.dtype if dtype is None else dtype
    target_shape = axis_shape_to_tuple(prototype.axes if shape is None else shape)

    jax_sharding = canonicalize_sharding(target_shape, out_sharding, "zeros_like")
    jax_array = jnp.ones_like(
        prototype, shape=axis_sizes(target_shape), dtype=dtype, device=device, out_sharding=jax_sharding
    )

    return constructors.array(jax_array, target_shape)


def empty_like(
    prototype: NamedArrayLike | DuckTypedNamedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    device: Device | Sharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an empty array with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: Optionally override the shape of the created array.
        dtype: Optionally override the dtype of the created array.
        device: Optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
                    committed. This argument exists for compatibility with the :ref:`python-array-api`.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, on the specified device if specified.
    """
    if device is not None and out_sharding is not None:
        raise ValueError(f"Only one of `device` or `out_sharding` can be set. Got {device} and {out_sharding}")

    prototype = util.ensure_named("full_like", prototype)
    dtype = prototype.dtype if dtype is None else dtype
    target_shape = axis_shape_to_tuple(prototype.axes if shape is None else shape)

    jax_array = jnp.empty_like(prototype, shape=axis_sizes(target_shape), dtype=dtype, device=device)

    return constructors.array(jax_array, target_shape, out_sharding=out_sharding)


def full_like(
    prototype: NamedArrayLike | DuckTypedNamedArray,
    fill_value: NamedArrayLike,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    device: Device | Sharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        fill_value: Scalar or array with which to fill the created array.
        shape: Optionally override the shape of the created array.
        dtype: Optionally override the dtype of the created array.
        device: Optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
                    committed. This argument exists for compatibility with the :ref:`python-array-api`.
        out_sharding: Optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        A ``NamedArray`` of the specified shape and dtype, on the specified device if specified.
    """

    prototype, fill_named = util.ensure_named("full_like", prototype, fill_value)
    dtype = prototype.dtype if dtype is None else dtype
    target_shape = axis_shape_to_tuple(prototype.axes if shape is None else shape)

    jax_sharding = canonicalize_sharding(target_shape, out_sharding, "zeros_like")
    jax_array = jnp.full_like(
        prototype,
        fill_named.array,
        shape=axis_sizes(target_shape),
        dtype=dtype,
        device=device,
        out_sharding=jax_sharding,
    )

    return constructors.array(jax_array, target_shape)


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


# TODO: add `device` argument when `jnp.arange` support both kwargs
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


# Array creation operations


@overload
def linspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray: ...


@overload
def linspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    *,
    retstep: Literal[True],
    dtype: DTypeLike | None = None,
    out_sharding: ShardingLike | None = None,
) -> tuple[NamedArray, NamedArray]: ...


@overload
def linspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray | tuple[NamedArray, NamedArray]: ...


# TODO: add `device` argument when `linspce` support both kwargs
def linspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray | tuple[NamedArray, NamedArray]:
    """Return evenly spaced values over a specified interval along a new axis.

    Values are generated over ``[start, stop]`` (inclusive when ``endpoint`` is ``True``) and returned along the new
    ``axis``, with all remaining axes coming from broadcasting ``start`` and ``stop``.

    Args:
        start: The starting value(s).
        stop: The ending value(s).
        axis: The new axis along which the samples are generated. A string is treated as an unnamed axis of size
            50 (matching NumPy's default); an `Axis` supplies both size and optional name.
        endpoint: If ``True`` (default), ``stop`` is included.
        retstep: If ``True``, also return the step size between samples.
        dtype: Optional dtype of the output.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` of samples, or a tuple ``(samples, step)`` when ``retstep=True``.
    """

    if isinstance(axis, str):
        axis = Axis(50, axis)  # numpy default

    from lynqx._src.named.operations import broadcast_arrays_and_shape

    (start, stop), common_axes = broadcast_arrays_and_shape(start, stop)
    if axis.name in axis_names(common_axes):
        raise ValueError(f"Axis {axis} already exists in one of the input arrays.")

    jax_array = jnp.linspace(start.array, stop.array, axis.size, endpoint=endpoint, retstep=retstep, dtype=dtype)

    out_axes = (axis,) + common_axes
    step = None
    if retstep:
        jax_array, jax_array_step = jax_array
        step = constructors.array(jax_array_step, common_axes, out_sharding=out_sharding)

    result = constructors.array(jax_array, out_axes, out_sharding=out_sharding)

    return (result, step) if step is not None else result


# TODO: add `device` argument when `jnp.logspce` support both kwargs
def logspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    base: NamedArrayLike = 10.0,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Return numbers spaced evenly on a log scale along a new axis.

    Args:
        start: ``base ** start`` is the starting value.
        stop: ``base ** stop`` is the final value.
        axis: The new axis along which the samples are generated. A string is treated as an unnamed axis of size 50;
            an `Axis` supplies size and optional name.
        endpoint: If ``True`` (default), ``stop`` is included.
        base: The base of the log space. Defaults to 10.
        dtype: Optional dtype of the output.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` of samples on a log scale.
    """
    if isinstance(axis, str):
        axis = Axis(50, axis)  # numpy default

    from lynqx._src.named import operations

    (start, stop, base), common_axes = operations.broadcast_arrays_and_shape(start, stop, base)
    if axis.name in axis_names(common_axes):
        raise ValueError(f"Axis {axis} already exists in one of the input arrays")

    jax_array = jnp.logspace(start.array, stop.array, axis.size, endpoint=endpoint, base=base.array, dtype=dtype)

    out_axes = (axis,) + common_axes
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


# TODO: add `device` argument when `jnp.geomspace` support both kwargs
def geomspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Return numbers spaced evenly on a geometric progression along a new axis.

    Args:
        start: The starting value(s) of the sequence.
        stop: The final value(s) of the sequence.
        axis: The new axis along which the samples are generated. A string is treated as an unnamed axis of size 50;
            an `Axis` supplies size and optional name.
        endpoint: If ``True`` (default), ``stop`` is included.
        dtype: Optional dtype of the output.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` of samples on a geometric progression.
    """
    if isinstance(axis, str):
        axis = Axis(50, axis)  # numpy default

    from lynqx._src.named import operations

    (start, stop), common_axes = operations.broadcast_arrays_and_shape(start, stop)
    if axis.name in axis_names(common_axes):
        raise ValueError(f"Axis {axis} already exists in one of the input arrays")

    jax_array = jnp.geomspace(start.array, stop.array, axis.size, endpoint=endpoint, dtype=dtype)

    out_axes = (axis,) + common_axes
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def delta(
    shape: AxisShape,
    axes: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Return a Kronecker delta over the selected axes of the given shape.

    The result equals 1 where all selected axes share the same index and 0 elsewhere.

    Args:
        shape: The output shape.
        axes: The axes to compare for equality. Defaults to all axes.
        dtype: Optional dtype of the output. Defaults to ``float32``.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` with the given shape.
    """
    shape = axis_shape_to_tuple(shape)
    indices = axis_indices(shape, axes)

    if len(indices) < 2:
        raise ValueError(f"delta requires at least 2 axes to compare, got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"delta requires distinct axes, got {axes!r} resolving to duplicate positions {indices}")

    indices = tuple(sorted(indices))  # normalize to shape order; equality is symmetric, so this is free
    spec = tuple(shape[i] for i in indices)

    iotas = [iota(spec, i, dtype=jnp.uint32) for i in range(len(spec))]
    eyes = [i1 == i2 for i1, i2 in zip(iotas[:-1], iotas[1:])]
    result = reduce(operator.and_, eyes)

    new_dtype = jnp.float32 if dtype is None else dtype
    casted_result = eqx.tree_at(lambda m: m.array, result, replace_fn=lambda x: jnp.astype(x, new_dtype))
    from lynqx._src.named.operations import broadcast_to

    return broadcast_to(casted_result, shape, out_sharding=out_sharding)


def identity(
    shape: AxisShape,
    axes: AxisSelection = (-1, -2),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Return a square identity matrix over two selected axes of the given shape.

    Args:
        shape: The output shape.
        axes: The two axes forming the matrix. Defaults to the last two axes.
        dtype: Optional dtype of the output.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` with the given shape whose two selected axes form an identity matrix.
    """
    resolved = axis_shape_to_tuple(shape)
    indices = axis_indices(resolved, axes)

    if len(indices) != 2:
        raise ValueError(f"identity requires exactly 2 axes, got {len(indices)} from {axes!r}")

    a, b = resolved[indices[0]], resolved[indices[1]]
    if a.size != b.size:
        raise ValueError(
            f"identity requires square axes, got sizes {a.size} and {b.size}; "
            "use delta(...) directly for a non-square diagonal tensor"
        )

    return delta(resolved, axes, dtype=dtype, out_sharding=out_sharding)


# Diagonal and triangular matrix operations


def _tri_mask(array: NamedArray, axis1: AxisSelector, axis2: AxisSelector, k: int, *, upper: bool) -> NamedArray:
    """Build a mask selecting the upper or lower triangle of a matrix-like array."""
    idx0, idx1 = axis_indices(array.axes, (axis1, axis2))
    if idx0 == idx1:
        raise ValueError(f"axis1 and axis2 must be distinct, both resolved to position {idx0}")

    from lynqx._src.named.operations import broadcast_to

    idx0, idx1 = axis_indices(array.axes, (axis1, axis2))
    if idx0 == idx1:
        raise ValueError(f"axis1 and axis2 must be distinct, both resolved to position {idx0}")

    ax1 = array.axes[idx0]  # row axis, as named by axis1
    ax2 = array.axes[idx1]  # column axis, as named by axis2

    # Rank-1 iotas along each named axis, then extended-broadcast each to the
    # full array axes.  broadcast_to transposes/expands, so the two results
    # have identical axes in identical order and subtract cleanly via ufuncs.
    row = broadcast_to(iota(ax1, dtype=jnp.int32), array.axes)
    col = broadcast_to(iota(ax2, dtype=jnp.int32), array.axes)

    diff = col - row  # ufuncs.subtract: same axes, elementwise
    mask = (diff >= k) if upper else (diff <= k)
    return mask


def triu(a: NamedArrayLike, axis1: AxisSelector, axis2: AxisSelector, k: int = 0) -> NamedArray:
    """Return the upper triangle of an array.

    Args:
        a: The input array.
        axis1: The first axis of the matrix.
        axis2: The second axis of the matrix.
        k: Diagonal above which to keep elements. Defaults to 0.

    Returns:
        A ```NamedArray``` with elements below the ``k``-th diagonal zeroed out.
    """

    named = util.ensure_named("triu", a)
    mask = _tri_mask(named, axis1, axis2, k, upper=True)
    return constructors.array(jnp.where(mask.array, named.array, 0), named.axes)


def tril(a: NamedArrayLike, axis1: AxisSelector, axis2: AxisSelector, k: int = 0) -> NamedArray:
    """Return the lower triangle of an array.

    Args:
        a: The input array.
        axis1: The first axis of the matrix.
        axis2: The second axis of the matrix.
        k: Diagonal below which to keep elements. Defaults to 0.

    Returns:
        A ```NamedArray``` with elements above the ``k``-th diagonal zeroed out.
    """
    named = util.ensure_named("tril", a)
    mask = _tri_mask(named, axis1, axis2, k, upper=False)
    return constructors.array(jnp.where(mask.array, named.array, 0), named.axes)


def tri(shape: AxisShape, axes: AxisSelection = (-1, -2), k: int = 0, dtype: DTypeLike | None = None) -> NamedArray:
    """Return an array with ones at and below the ``k``-th diagonal and zeros elsewhere.

    Args:
        shape: The output shape.
        axes: The two axes forming the matrix. Defaults to the last two axes.
        k: The diagonal offset. Defaults to 0.
        dtype: Optional dtype of the output.

    Returns:
        A `NamedArray` with the given shape.
    """
    resolved = axis_shape_to_tuple(shape)
    indices = axis_indices(resolved, axes)

    a, b = indices

    diag = ones(shape, dtype=dtype)
    mask = _tri_mask(diag, a, b, k, upper=False)
    return mask * diag


def tril_indices(
    n: int, k: int = 0, m: int | None = None, axis: AxisLike | tuple[AxisLike, AxisLike] | None = None
) -> tuple[NamedArray, NamedArray]:
    """Return the row and column indices of a lower-triangular matrix.

    Args:
        n: The row dimension of the matrix.
        k: Diagonal offset. Defaults to ``0``.
        m: The column dimension. Defaults to ``n``.
        axis: Optional axis or pair of axes to attach to the returned index arrays.

    Returns:
        A pair of ``NamedArray`` objects containing the row and column indices of the selected lower triangle.
    """

    m = n if m is None else m
    row_jax, col_jax = jnp.tril_indices(n, k, m)

    axis_tuple = axis if isinstance(axis, tuple) else (axis, axis)
    size_tuple = [row_jax.shape[0], col_jax.shape[0]]
    row_axis, col_axis = [
        util.resolve_new_axis(ax, sz, "tril_indices", "axis") for ax, sz in zip(axis_tuple, size_tuple)
    ]

    return constructors.array(row_jax, row_axis), constructors.array(col_jax, col_axis)


def triu_indices(
    n: int, k: int = 0, m: int | None = None, axis: AxisLike | tuple[AxisLike, AxisLike] | None = None
) -> tuple[NamedArray, NamedArray]:
    """Return the row and column indices of an upper-triangular matrix.

    Args:
        n: The row dimension of the matrix.
        k: Diagonal offset. Defaults to ``0``.
        m: The column dimension. Defaults to ``n``.
        axis: Optional axis or pair of axes to attach to the returned index arrays.

    Returns:
        A pair of ``NamedArray`` objects containing the row and column indices of the selected upper triangle.
    """

    row_jax, col_jax = jnp.triu_indices(n, k, m)

    axis_tuple = axis if isinstance(axis, tuple) else (axis, axis)
    size_tuple = [row_jax.shape[0], col_jax.shape[0]]
    row_axis, col_axis = [
        util.resolve_new_axis(ax, sz, "tril_indices", "axis") for ax, sz in zip(axis_tuple, size_tuple)
    ]

    return constructors.array(row_jax, row_axis), constructors.array(col_jax, col_axis)


def tril_indices_from(
    a: NamedArrayLike,
    k: int = 0,
    axis: AxisLike | tuple[AxisLike, AxisLike] | None = None,
) -> tuple[NamedArray, NamedArray]:
    """Return the indices of the lower triangle of a 2-D array.

    Args:
        a: Input array. Must have ``a.ndim == 2`
        k: Optional value that specifies the sub-diagonal on and above which the indices of the upper triangle are
            returned. ``k=0`` refers to the main diagonal. ``k<0`` refers to the sub-diagonal below the main diagonal
            and ``k>0`` refers to the sub-diagonal above the main diagonal.

    Returns:
        A tuple of two ``NamedArrays`` containing the indices of the lower triangle, one along each axis.
    """
    named = util.ensure_named("tril_indices_from", a)

    if named.ndim != 2:
        raise ValueError("tril_indices_from requires a two-dimensional array")

    return tril_indices(named.axes[0].size, k, named.axes[1].size, axis)


def triu_indices_from(
    a: NamedArrayLike,
    k: int = 0,
    axis: AxisLike | tuple[AxisLike, AxisLike] | None = None,
) -> tuple[NamedArray, NamedArray]:
    """Return the indices of the upper triangle of a 2-D array.

    Args:
        a: Input array. Must have ``a.ndim == 2`
        k: Optional value that specifies the sub-diagonal on and above which the indices of the upper triangle are
            returned. ``k=0`` refers to the main diagonal. ``k<0`` refers to the sub-diagonal below the main diagonal
            and ``k>0`` refers to the sub-diagonal above the main diagonal.

    Returns:
        A tuple of two ``NamedArrays`` containing the indices of the upper triangle, one along each axis.

    """

    named = util.ensure_named("triu_indices_from", a)

    if named.ndim != 2:
        raise ValueError("triu_indices_from requires a two-dimensional array")

    return triu_indices(named.axes[0].size, k, named.axes[1].size, axis)
