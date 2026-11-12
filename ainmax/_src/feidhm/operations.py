from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, overload

import equinox as eqx
import jax.lax
import jax.numpy as jnp
from jax import Device
from jax.typing import DTypeLike

from ainmax._src.axis import Axis, AxisLike, AxisSelection, AxisSelector, AxisShape
from ainmax._src.axis_util import (
    axis_index,
    axis_indices,
    axis_names,
    axis_shape_to_tuple,
    axis_sizes,
    match_axes,
    remove_axes,
    resolve_axes,
    validate_unique_axes,
)
from ainmax._src.feidhm import constructors, reductions, ufuncs, util
from ainmax._src.partition import ShardingLike
from ainmax._src.sharding import canonicalize_sharding
from ainmax._src.typing import NamedArray, NamedArrayLike


# Utility functions


# Dtype and rank predicates
def iscomplex(a: NamedArrayLike) -> NamedArray:
    """Return a boolean array showing where the input is complex.

    Args:
        a: The input array or scalar.

    Returns:
        A boolean ``NamedArray`` indicating whether each element of ``a`` is complex.
    """

    a = util.ensure_named("iscomplex", a)
    return constructors.array(jnp.iscomplex(a.array), a.axes)


def iscomplexobj(x: Any) -> bool:
    """Check whether the input is a complex number or contains complex elements.

    The decision is based on the input's dtype rather than its values, so arrays with zero imaginary parts are still
    considered complex.

    Args:
        x: The input to test.

    Returns:
        ``True`` if ``x`` has a complex dtype, otherwise ``False``.
    """

    return jnp.iscomplexobj(x)


def isreal(a: NamedArrayLike) -> NamedArray:
    """Return a boolean array showing where the input is real.

    Args:
        a: The input array or scalar.

    Returns:
        A boolean ``NamedArray`` indicating whether ``a`` is elementwise real.
    """

    a = util.ensure_named("isreal", a)
    return constructors.array(jnp.isreal(a.array), a.axes)


def isrealobj(a: Any) -> bool:
    """Check whether the input has a non-complex dtype.

    The decision is based on the input's dtype rather than its values, so arrays with zero imaginary parts are still
    considered complex.

    Args:
        a: The input to test.

    Returns:
        ``True`` if ``a`` does not have a complex dtype, otherwise ``False``.
    """
    return jnp.isrealobj(a)


def ndim(x: NamedArrayLike) -> int:
    """Return the number of axes in an array.

    Args:
        x: The array or scalar to inspect.

    Returns:
        The number of dimensions in ``x``.
    """
    if isinstance(x, NamedArray):
        return x.array.ndim
    return jnp.ndim(x)


# Broadcasting


@validate_unique_axes(arg_names=("shape",))
def broadcast_to(
    a: NamedArrayLike,
    shape: AxisShape,
    *,
    out_sharding: ShardingLike | None = None,
):
    """Broadcast an array to a specified shape.

    Lynqx uses extended broadcasting semantics that allow broadcasting to a shape with different axes, as long as the
    existing axes are preserved and any new axes are added with size 1. This is similar to NumPy's broadcasting rules,
    but with the added constraint that named axes must match.

    Args:
        a: The array to broadcast.
        shape: The target shape to broadcast to.
        out_sharding: Optional sharding specification for the output array. If not specified, it will be determined
            automatically by the compiler.

    Returns:
        A copy of ``a`` broadcasted to the specified shape.
    """
    named = util.ensure_named("broadcast_to", a)
    target = axis_shape_to_tuple(shape)

    matched = match_axes(
        named.axes,
        target,
        allow_positional_fallback=True,
    )

    if matched.unmatched_source:
        missing = [named.axes[i] for i in matched.unmatched_source]
        raise ValueError(f"broadcast_to cannot drop axes {missing}; every existing axis must appear in `shape`")

    # Validate that every matched axis is broadcast-compatible.
    for m in matched.matches:
        src_ax = named.axes[m.source]
        tgt_ax = target[m.target]

        if src_ax.size != tgt_ax.size and src_ax.size != 1:
            raise ValueError(f"Cannot broadcast axis {src_ax.name!r} of size {src_ax.size} to {tgt_ax.size}")

    target_sizes = axis_sizes(target)
    jax_sharding = canonicalize_sharding(
        target,
        out_sharding,
        "broadcast_to",
    )

    if named.array.ndim == 0:
        jax_array = jnp.broadcast_to(named.array, target_sizes, out_sharding=jax_sharding)
        return constructors.array(jax_array, target)

    broadcast_dimensions = tuple(m.target for m in matched.matches)

    # If source axes are not already in target order, transpose first.
    source_order = tuple(m.source for m in sorted(matched.matches, key=lambda m: m.target))
    identity = tuple(range(named.array.ndim))

    if source_order != identity:
        array = jnp.transpose(named.array, source_order)

        # After transposing, matches are in target order.
        broadcast_dimensions = tuple(m.target for m in sorted(matched.matches, key=lambda m: m.target))
    else:
        array = named.array

    jax_array = jax.lax.broadcast_in_dim(
        array,
        target_sizes,
        broadcast_dimensions=broadcast_dimensions,
    )

    return constructors.array(jax_array, target, out_sharding=out_sharding)


def broadcast_arrays_and_shape(
    *arrays: NamedArrayLike, out_sharding: ShardingLike | None = None
) -> tuple[tuple[NamedArray, ...], tuple[Axis, ...]]:
    """Broadcast multiple arrays to a common shape and return both.

    Args:
        *arrays: The arrays to broadcast.
        out_sharding: Optional sharding specification for the output arrays.

    Returns:
        A tuple ``(broadcasted_arrays, common_shape)`` where ``broadcasted_arrays`` is a tuple of ``NamedArray``
        sharing ``common_shape``.
    """

    named_arrays = util.ensure_named_tuple("broadcast_arrays", arrays)
    if not named_arrays:
        return (), ()

    target_shape = named_arrays[0].axes
    for arr in named_arrays[1:]:
        _, _, target_shape = util.align_shapes_for_broadcast(target_shape, arr.axes)

    broadcasted_arrays = tuple(broadcast_to(arr, target_shape, out_sharding=out_sharding) for arr in arrays)
    return broadcasted_arrays, target_shape


def broadcast_arrays(*arrays: NamedArrayLike, out_sharding: ShardingLike | None = None):
    """Broadcast multiple NamedArrays to a common shape.

    Args:
        *arrays: The arrays to broadcast.
        out_sharding: Optional sharding specification for the output arrays.

    Returns:
        A tuple of broadcasted ``NamedArrays``.
    """
    return broadcast_arrays_and_shape(*arrays, out_sharding=out_sharding)[0]


def broadcast_shapes(*shapes: AxisShape) -> tuple[Axis, ...]:
    """Broadcast multiple shapes to a common shape.

    Args:
        *shapes: The shapes to broadcast.

    Returns:
        The common broadcasted shape.
    """
    if not shapes:
        return ()

    target_shape = axis_shape_to_tuple(shapes[0])
    for shape in shapes[1:]:
        _, _, target_shape = util.align_shapes_for_broadcast(target_shape, axis_shape_to_tuple(shape))

    return target_shape


# Axis and shape transformations


def rename(a: NamedArrayLike, axis: AxisSelector, name: str | None) -> NamedArray:
    """Rename an axis of an array.

    Args:
        a: The NamedArray to rename.
        axis: The axis to rename.
        name: The new name for the axis. If None, the axis will be unnamed.

    Returns:
        A ``NamedArray`` with the specified axis renamed.
    """
    named = util.ensure_named("rename", a)
    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"rename: axis {axis!r} not found in {named.axes}")

    new_axes = list(named.axes)
    new_axes[idx] = Axis(new_axes[idx].size, name)
    return constructors.array(named.array, new_axes)


def reshape(a: NamedArrayLike, shape: AxisShape, *, out_sharding: ShardingLike | None = None):
    """Reshape an array to a new shape.

    The total number of elements must be preserved, and the ordering of axes follows the flattened (row-major)
    ordering of the input.

    Args:
        a: The ``NamedArray`` to reshape.
        shape: The target shape.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` with the requested shape.
    """
    a = util.ensure_named("reshape", a)

    shape_tuple = axis_shape_to_tuple(shape)
    shape_sizes = axis_sizes(shape_tuple)
    if prod(shape_sizes) != a.size:
        raise ValueError()

    jax_sharding = canonicalize_sharding(shape_tuple, out_sharding, "reshape")
    jax_array = jnp.reshape(a.array, shape_sizes, out_sharding=jax_sharding)

    return constructors.array(jax_array, shape_tuple)


def transpose(a: NamedArrayLike, permutation: AxisSelection | None = None):
    """Transpose a NamedArray according to the given axis permutation.

    Args:
        a: The NamedArray to transpose.
        permutation: A sequence of axis names or indices specifying the new order of axes.
                     If None, reverses the order of axes.

    Returns:
        A ```NamedArray``` with axes transposed according to the specified permutation.
    """
    named = util.ensure_named("transpose", a)

    if permutation is None:
        permutes_axes = tuple(reversed(named.axes))
    else:
        permutes_axes = resolve_axes(named.axes, permutation)

    matched = match_axes(named.axes, permutes_axes, allow_positional_fallback=True)
    if matched.unmatched_source or matched.unmatched_target:
        raise ValueError("Invalid permutation: all axes must be accounted for in the permutation.")

    tgt_to_src = {m.target: m.source for m in matched.matches}
    axis_permutation = tuple(tgt_to_src[i] for i in range(len(permutes_axes)))

    jax_array = jnp.transpose(named.array, axes=axis_permutation)
    return constructors.array(jax_array, permutes_axes)


def swapaxes(a: NamedArrayLike, axis1: AxisSelector, axis2: AxisSelector):
    """Swap two axes of a NamedArray.

    Args:
        a: The NamedArray to swap axes.
        axis1: The first axis to swap.
        axis2: The second axis to swap.

    Returns:
        A new NamedArray with the specified axes swapped.
    """
    named = util.ensure_named("swapaxes", a)
    idx0, idx1 = axis_indices(named.axes, (axis1, axis2))

    if idx0 == idx1:
        raise ValueError(f"swapaxes: axis1 and axis2 must be distinct, both resolved to position {idx0}")

    new_axes = list(named.axes)
    new_axes[idx0], new_axes[idx1] = new_axes[idx1], new_axes[idx0]

    jax_array = jnp.swapaxes(named.array, idx0, idx1)
    return constructors.array(jax_array, tuple(new_axes))


def roll(a: NamedArrayLike, shift: NamedArrayLike | int | Sequence[int], axis: AxisSelection):
    """
    Roll a NamedArray along specified axes.

    Args:
        a: The NamedArray to roll.
        shift: The number of positions by which elements are shifted. Can be an integer or a sequence of integers. If
            a sequence, its length must match the number of axes specified.
        axis: The axes along which to roll. Can be a single axis or a sequence of axes.

    Returns:
        A new NamedArray with the same shape as `a`, but with elements rolled along the specified axes.
    """

    named = util.ensure_named("roll", a)
    resolved_axes = resolve_axes(named.axes, axis)

    if isinstance(shift, int):
        shifts = (shift,) * len(resolved_axes)
    elif isinstance(shift, Sequence):
        if len(shift) != len(resolved_axes):
            raise ValueError(f"roll: shift length {len(shift)} does not match number of axes {len(resolved_axes)}")
        shifts = tuple(shift)
    else:
        raise TypeError(f"roll: shift must be int or sequence of ints, got {type(shift).__name__}")

    jax_array = jnp.roll(named.array, shift=shifts, axis=axis_indices(named.axes, resolved_axes))
    return constructors.array(jax_array, named.axes)


PadValue = int | tuple[int, int]
PadMapping = Mapping[AxisSelector, PadValue]
PadSpec = PadValue | PadMapping


def _resolve_pad_width(val: PadValue) -> tuple[int, int]:
    """Normalize a per-axis pad width to a ``(before, after)`` tuple."""
    if isinstance(val, int):
        if val < 0:
            raise ValueError(f"Pad width must be non-negative, got {val}")
        return (val, val)
    else:
        lo, hi = val
        if lo < 0 or hi < 0:
            raise ValueError(f"Pad width must be non-negative, got ({lo}, {hi})")
        return val


def _normalize_pad_width(axes: tuple[Axis, ...], pad_width: PadSpec) -> tuple[tuple[int, int], ...]:
    """Normalize padding values for all axes in an array."""
    if not isinstance(pad_width, Mapping):
        if isinstance(pad_width, tuple) and len(pad_width) != 2:
            raise ValueError(
                f"pad_width tuple must be (before, after), got {pad_width!r}; use a mapping for per-axis widths"
            )
        return (_resolve_pad_width(pad_width),) * len(axes)

    widths: list[tuple[int, int] | None] = [None] * len(axes)
    for key, val in pad_width.items():
        idx = axis_index(axes, key)
        if idx is None:  # drop this check if axis_index raises on failure
            raise ValueError(f"pad_width key {key!r} does not match any axis in {tuple(a.name for a in axes)}")
        if widths[idx] is not None:
            raise ValueError(f"axis {axes[idx].name!r} appears more than once in pad_width")
        widths[idx] = _resolve_pad_width(val)
    return tuple(w if w is not None else (0, 0) for w in widths)


def _pad_callable(named, widths, fn, kwargs):
    """Apply a padding callable one axis at a time while validating the output shape."""
    out = named
    for pos, (ax, (lo, hi)) in enumerate(zip(named.axes, widths)):
        if lo == 0 and hi == 0:
            continue
        out = util.ensure_named("pad", fn(out, ax, (lo, hi), **kwargs))
        got = resolve_axes(out.axes, ax.name)[0] if ax.name is not None else out.axes[pos]
        expected = ax.size + lo + hi
        if got.size != expected:
            raise ValueError(f"pad callable returned size {got} for axis {ax.name}, expected {expected}")
    return out


def pad(
    array: NamedArrayLike, pad_width: PadSpec, mode: str | Callable[..., Any] = "constant", **kwargs
) -> NamedArray:
    """Pad an array along specified axes.

    Args:
        array: The array to pad.
        pad_width: A mapping from axes to pad widths, or a single pad width to apply to all axes. Each pad width can
        be an int (same padding before and after) or a tuple of two ints (padding before, padding after).
        mode: The padding mode. Can be a string (e.g. ``'constant'``, ``'edge'``, ``'reflect'``) or a callable taking
            a `NamedArray` and returning a padded `NamedArray`.
        **kwargs: Extra keyword arguments forwarded to the padding function.

    Returns:
        A new `NamedArray` padded according to the specified widths and mode.
    """
    named = util.ensure_named("pad", array)
    widths = _normalize_pad_width(named.axes, pad_width)
    out_axes = tuple(Axis(ax.size + lo + hi, ax.name) for ax, (lo, hi) in zip(named.axes, widths))

    if callable(mode):
        return _pad_callable(named, widths, mode, kwargs)

    jax_array = jnp.pad(named.array, widths, mode=mode, **kwargs)
    return constructors.array(jax_array, out_axes)


def ravel(a: NamedArrayLike, name: AxisLike | None = None) -> NamedArray:
    """Flatten a NamedArray into a rank-1 array while preserving its axes.

    Args:
        a: The array to flatten.
        name: Optional name for the new flattened axis. If None, the axis will be unnamed.

    Returns:
        A new NamedArray that is a flattened version of the input, with a single axis.
    """
    named = util.ensure_named("ravel", a)

    if isinstance(name, Axis):
        if name.size != named.size:
            raise ValueError(
                f"ravel: provided axis {name!r} has size {name.size}, but flattened array has size {named.size}"
            )
        flat_axis = name
    else:
        flat_axis = Axis(named.size, name)
    jax_array = jnp.ravel(named.array)
    return constructors.array(jax_array, (flat_axis,))


def flatten(a: NamedArrayLike, name: AxisLike | None = None):
    """Flatten an array into a single axis.

    This is a convenience alias for :func:`ravel`.

    Args:
        a: The array to flatten.
        name: Optional name for the resulting single axis.

    Returns:
        A ``NamedArray`` with one axis containing all elements of ``a``.
    """
    return ravel(a, name)


# Array assembly and repetition


def _normalize_insert_index(index: int, ndim: int, fn_name: str) -> int:
    """Normalise the insertion position of a new axis to ``[0, ndim]``.

    `ndim` is the rank of the *inputs*, so the output has rank `ndim + 1` and valid values are
    `-(ndim + 1) <= index <= ndim`; `-1` appends.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{fn_name}: `index` must be an int, got {type(index).__name__}")
    if not -(ndim + 1) <= index <= ndim:
        raise ValueError(
            f"{fn_name}: `index` {index} is out of range for inserting into {ndim} existing axes; "
            f"expected {-(ndim + 1)} <= index <= {ndim}"
        )
    return index + ndim + 1 if index < 0 else index


def _align_permutation(ref: Sequence[Axis], axes: Sequence[Axis], fn_name: str, position: int) -> tuple[int, ...]:
    """Find ``perm`` such that ``axes[perm[i]]`` corresponds to ``ref[i]``.

    Named axes correspond by name. Anonymous axes correspond by order of appearance, which never needs to compare
    sizes, so repeated anonymous axes of the same size are not ambiguous and anonymous axes of different sizes can
    still be paired up.
    """
    if len(ref) != len(axes):
        raise ValueError(
            f"{fn_name}: array at position {position} has {len(axes)} axes but the first array has {len(ref)}; "
            f"all arrays must have the same axes"
        )

    name_to_pos = {ax.name: i for i, ax in enumerate(axes) if ax.name is not None}
    anonymous = iter(i for i, ax in enumerate(axes) if ax.name is None)

    perm: list[int] = []
    for ax in ref:
        if ax.name is None:
            nxt = next(anonymous, None)
            if nxt is None:
                raise ValueError(
                    f"{fn_name}: array at position {position} has fewer anonymous axes than the first array"
                )
            perm.append(nxt)
        elif ax.name in name_to_pos:
            perm.append(name_to_pos[ax.name])
        else:
            raise ValueError(
                f"{fn_name}: array at position {position} has no axis named {ax.name!r}; its axes are "
                f"{tuple(a.label for a in axes)}"
            )
    return tuple(perm)


def _canonicalize_arrays(fn_name: str, arrays: Sequence[NamedArray], *, free: int | None = None):
    """Permute every array's axes into the order of the first array and check sizes."""
    ref = arrays[0].axes
    identity = tuple(range(len(ref)))

    jax_arrays = []
    for position, arr in enumerate(arrays):
        perm = _align_permutation(ref, arr.axes, fn_name, position)
        for i, p in enumerate(perm):
            if i != free and arr.axes[p].size != ref[i].size:
                raise ValueError(
                    f"{fn_name}: axis {ref[i].label!r} has size {arr.axes[p].size} in the array at position "
                    f"{position} but size {ref[i].size} in the first array"
                )
        jax_arrays.append(arr.array if perm == identity else jnp.transpose(arr.array, perm))

    return ref, jax_arrays


def stack(
    arrays: Sequence[NamedArrayLike],
    axis: AxisLike,
    dtype: DTypeLike | None = None,
    *,
    index: int = 0,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Join arrays along a new axis.

    All arrays must have the same axes. Named axes are matched by name (so they may appear in different orders);
    anonymous axes are matched by order of appearance. The result uses the axis order of the first array.

    Args:
        arrays: The arrays to stack.
        axis: The new axis. A string creates a new axis of that name with size ``len(arrays)``; an `Axis` must have
            size ``len(arrays)``.
        dtype: Optional dtype of the result.
        index: Position at which the new axis is inserted into the result. Negative values count from the end of
            the *result*, so ``-1`` appends the new axis last. Defaults to ``0``.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` with one more axis than the inputs.
    """
    named = util.ensure_named_tuple("stack", tuple(arrays))
    if not named:
        raise ValueError("stack requires at least one array")

    if isinstance(axis, str):
        axis = Axis(len(named), axis)
    elif axis.size != len(named):
        raise ValueError(f"stack: new axis {axis!r} has size {axis.size} but {len(named)} arrays were provided")

    ref, jax_arrays = _canonicalize_arrays("stack", named)

    if axis.name is not None and axis.name in axis_names(ref):
        raise ValueError(f"stack: new axis {axis.name!r} already exists in the input arrays")

    position = _normalize_insert_index(index, len(ref), "stack")
    jax_array = jnp.stack(jax_arrays, axis=position, dtype=dtype)

    out_axes = ref[:position] + (axis,) + ref[position:]
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def unstack(a: NamedArrayLike, axis: AxisSelector = 0) -> tuple[NamedArray, ...]:
    """Split an array into a tuple of arrays along an axis, removing that axis.

    This is the inverse of `stack`.

    Args:
        a: The array to split.
        axis: The axis to split along. Defaults to the first axis.

    Returns:
        A tuple of ``a.axes[axis].size`` arrays, each with the selected axis removed.
    """
    named = util.ensure_named("unstack", a)

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"unstack: axis {axis!r} not found in {named.axes}")

    out_axes = named.axes[:idx] + named.axes[idx + 1 :]
    return tuple(constructors.array(x, out_axes) for x in jnp.unstack(named.array, axis=idx))


TileReps = int | Sequence[int] | Mapping[AxisSelector, int]


def tile(a: NamedArrayLike, repeats: TileReps, *, out_sharding: ShardingLike | None = None) -> NamedArray:
    """Repeat an array along its existing axes.

    Unlike `numpy.tile`, this never prepends axes: ``repeats`` is always interpreted per existing axis, and each
    axis of size ``n`` with repetition ``r`` becomes an axis of size ``n * r`` (names are preserved).

    Args:
        a: The array to tile.
        repeats: Either a single int applied to every axis, a sequence with one entry per axis (in axis order), or
            a mapping from axis selectors to repetitions, where unlisted axes are repeated once.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        The tiled array, with the same axis names and order as ``a``.
    """
    named = util.ensure_named("tile", a)
    ndim = len(named.axes)

    if isinstance(repeats, Mapping):
        per_axis = [1] * ndim
        seen: set[int] = set()
        for key, r in repeats.items():
            idx = axis_index(named.axes, key)
            if idx is None:
                raise ValueError(f"tile: axis {key!r} not found in {named.axes}")
            if idx in seen:
                raise ValueError(f"tile: axis {named.axes[idx].label!r} appears more than once in `repeats`")
            seen.add(idx)
            per_axis[idx] = r
        rep_tuple = tuple(per_axis)
    elif isinstance(repeats, int):
        rep_tuple = (repeats,) * ndim
    else:
        rep_tuple = tuple(repeats)
        if len(rep_tuple) != ndim:
            raise ValueError(
                f"tile: `repeats` has {len(rep_tuple)} entries but the array has {ndim} axes; "
                f"use a mapping to repeat only some axes"
            )

    for r in rep_tuple:
        if isinstance(r, bool) or not isinstance(r, int) or r < 1:
            raise ValueError(f"tile: repetitions must be positive integers, got {rep_tuple}")

    if ndim == 0:
        return named

    out_axes = tuple(ax.resize(ax.size * r) for ax, r in zip(named.axes, rep_tuple))
    jax_array = jnp.tile(named.array, rep_tuple)
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def concatenate(
    arrays: Sequence[NamedArrayLike],
    axis: AxisSelector = 0,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Join arrays along an existing axis.

    All arrays must have the same axes, and sizes must agree on every axis except the one being joined. Named
    axes are matched by name; anonymous axes by order of appearance. The result uses the axis order of the first
    array.

    Args:
        arrays: The arrays to concatenate.
        axis: The axis to join along. Selectors are resolved against the first array. Defaults to the first axis.
        dtype: Optional dtype of the result.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` whose size along ``axis`` is the sum of the input sizes.
    """
    named = util.ensure_named_tuple("concatenate", tuple(arrays))
    if not named:
        raise ValueError("concatenate requires at least one array")
    if not named[0].axes:
        raise ValueError("concatenate: zero-dimensional arrays cannot be concatenated")

    idx = axis_index(named[0].axes, axis)
    if idx is None:
        raise ValueError(f"concatenate: axis {axis!r} not found in {named[0].axes}")

    ref, jax_arrays = _canonicalize_arrays("concatenate", named, free=idx)
    jax_array = jnp.concatenate(jax_arrays, axis=idx, dtype=dtype)

    out_axes = ref[:idx] + (ref[idx].resize(jax_array.shape[idx]),) + ref[idx + 1 :]
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def concat(
    arrays: Sequence[NamedArrayLike],
    axis: AxisSelector = 0,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Alias of `concatenate`.

    Args:
        arrays: The arrays to concatenate.
        axis: The axis along which to concatenate.
        dtype: Optional dtype of the result.
        out_sharding: Optional sharding specification for the result.

    Returns:
        A ``NamedArray`` containing the concatenated arrays.

    Raises:
        ValueError: If the arrays cannot be concatenated along ``axis``.
    """
    return concatenate(arrays, axis, dtype, out_sharding=out_sharding)


BlockSpec = NamedArrayLike | Sequence[NamedArrayLike]


def _block_depth(x: Any) -> int:
    """Determine the nesting depth of a block specification.

    Args:
        x: A block or nested sequence of blocks.

    Returns:
        The maximum nesting depth of the block specification.
    """
    if isinstance(x, (list, tuple)):
        if not x:
            raise ValueError("block: empty lists are not allowed")
        depths = {_block_depth(e) for e in x}
        if len(depths) != 1:
            raise ValueError("block: all lists must be nested to the same depth")
        return depths.pop() + 1
    return 0


def block(
    arrays: BlockSpec,
    axes: Sequence[AxisSelector] | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Assemble an array from nested lists of blocks.

    The list nesting depth ``d`` maps onto ``axes[d]`` (outermost list first): the innermost lists are
    concatenated along ``axes[-1]``, the next level along ``axes[-2]``, and so on.

    For example, a 2x2 block matrix is ``block([[A, B], [C, D]], axes=("i", "j"))``: ``A, B`` and ``C, D`` are
    joined along ``"j"`` and the two results are then joined along ``"i"``.

    Args:
        arrays: A block array, or arbitrarily nested lists (all the same depth) of blocks.
        axes: One axis selector per nesting level, outermost first. Defaults to the trailing axes in order
            (``(-depth, ..., -1)``), as in `numpy.block`. Each selector is resolved per concatenation against the
            first array in that group.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        The assembled ``NamedArray``.
    """
    depth = _block_depth(arrays)
    if depth == 0:
        return util.ensure_named("block", arrays)

    if axes is None:
        axes = tuple(range(-depth, 0))
    elif len(axes) != depth:
        raise ValueError(f"block: `arrays` is nested {depth} deep but {len(axes)} axes were given")

    def build(x: Any, level: int, sharding: ShardingLike | None) -> NamedArray:
        """Recursively assemble a nested block array into a single ``NamedArray``.

        Args:
            x: The current block subtree being assembled.
            level: The current nesting depth within the block specification.
            sharding: Optional sharding to apply to the final concatenation.

        Returns:
            The assembled array at or below the current nesting level.
        """
        if level == depth:
            return util.ensure_named("block", x)
        parts = [build(e, level + 1, None) for e in x]
        return concatenate(parts, axes[level], out_sharding=sharding)

    return build(arrays, 0, out_sharding)


def repeat(
    a: NamedArrayLike,
    repeats: NamedArrayLike,
    axis: AxisSelector | None = None,
    *,
    total_repeat_length: int | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Repeat each element of an axis, like `jax.numpy.repeat` (each element is repeated in place, unlike `tile`).

    Unlike NumPy, ``axis=None`` does not flatten the array (that would silently drop every axis name): it is only
    accepted for 1-D arrays; use `ravel` first otherwise. The repeated axis keeps its name and order.

    Args:
        a: The array whose elements are repeated.
        repeats: Non-negative repetition counts: a Python int / 0-d array applied to every element, or a 1-D array
            with one count per element of ``axis`` (or a single count). If the 1-D array has a named axis, that
            name must match the repeated axis.
        axis: The axis to repeat along. May be omitted for 1-D arrays.
        total_repeat_length: Static size of the repeated axis. Required whenever it cannot be inferred from
            concrete values, e.g. ``repeats`` is an array inside ``jit``. If it differs from ``sum(repeats)`` the
            output is truncated or padded with the last element, as in JAX.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        An array with the axes of ``a`` where ``axis`` has size ``sum(repeats)`` (``total_repeat_length`` if given).
    """
    named = util.ensure_named("repeat", a)
    if axis is None:
        if named.ndim != 1:
            raise ValueError(
                f"repeat: `axis` must be given for arrays with {named.ndim} axes (axis=None would flatten and drop "
                "axis names; call `ravel` first if that is intended)"
            )
        idx = 0
    else:
        idx = axis_index(named.axes, axis)
        if idx is None:
            raise ValueError(f"repeat: axis {axis!r} not found in {named.axes}")
    target = named.axes[idx]

    if isinstance(total_repeat_length, bool) or not (
        total_repeat_length is None or isinstance(total_repeat_length, int)
    ):
        raise TypeError(f"repeat: `total_repeat_length` must be an int or None, got {total_repeat_length!r}")
    if total_repeat_length is not None and total_repeat_length < 0:
        raise ValueError(f"repeat: `total_repeat_length` must be non-negative, got {total_repeat_length}")

    repeats = util.ensure_named("repeat", repeats)
    if repeats.ndim > 1:
        raise ValueError(f"repeat: `repeats` must be a scalar or 1-D, got axes {repeats.axes}")
    if repeats.ndim == 1:
        r_ax = repeats.axes[0]
        if r_ax.size not in (1, target.size):
            raise ValueError(
                f"repeat: `repeats` has size {r_ax.size} but axis {target.label!r} has size {target.size}"
            )
        if r_ax.name is not None and target.name is not None and r_ax.name != target.name:
            raise ValueError(f"repeat: `repeats` axis {r_ax.name!r} does not match the repeated axis {target.name!r}")
    if not (jnp.issubdtype(repeats.dtype, jnp.integer) or repeats.dtype == jnp.bool_):
        raise TypeError(f"repeat: `repeats` must have an integer dtype, got {repeats.dtype}")

    # The output size must be static because it becomes an `Axis`.
    if total_repeat_length is not None:
        out_size = total_repeat_length
    else:
        try:
            counts = repeats.array
        except jax.errors.TracerArrayConversionError as err:
            raise ValueError(
                "repeat: the output size depends on traced values of `repeats`; pass `total_repeat_length`"
            ) from err
        if any(counts < 0):
            raise ValueError("repeat: `repeats` must be non-negative")
        out_size = int(counts.sum() * (target.size if counts.size == 1 else 1))

    if out_size == 0:
        raise ValueError("repeat: the result would have a zero-size axis, which `Axis` does not support")

    out_axes = tuple(ax.resize(out_size) if i == idx else ax for i, ax in enumerate(named.axes))
    jax_sharding = canonicalize_sharding(out_axes, out_sharding, "repeat")
    jax_array = jnp.repeat(
        named.array,
        repeats.array,
        axis=idx,
        total_repeat_length=total_repeat_length if repeats.ndim else None,
        out_sharding=jax_sharding,
    )

    return constructors.array(jax_array, out_axes)


# Value conversions and comparisons


def astype(
    a: NamedArrayLike, dtype: DTypeLike | None, *, copy: bool = False, device: Device | ShardingLike | None = None
) -> NamedArray:
    """Convert an array to a specified dtype.

    This is implemented via :func:`jax.lax.convert_element_type`, which may have slightly different behavior than
    :func:`numpy.astype` in some cases. In particular, the details of float-to-int and int-to-float casts are
    implementation dependent.

    Args:
        a: Input array to convert.
        dtype: Output datatype.
        copy: If ``True``, always return a copy. If ``False`` (default), only return a copy if necessary.
        device: Optionally specify the device to which the output will be committed.

    Returns:
        An array with the same shape as ``a``, containing values of the
        specified dtype.
    """
    a = util.ensure_named("astype", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.astype(x, dtype, copy=copy, device=device))


def allclose(
    a: NamedArrayLike,
    b: NamedArrayLike,
    rtol: NamedArrayLike = 1e-05,
    atol: NamedArrayLike = 1e-08,
    equal_nan: bool = False,
) -> NamedArray:
    r"""Check if two arrays are element-wise approximately equal within a tolerance.

    Essentially this function evaluates the following condition:

    .. math::

     |a - b| \le \mathtt{atol} + \mathtt{rtol} * |b|

    ``jnp.inf`` in ``a`` will be considered equal to ``jnp.inf`` in ``b``.

    Args:
        a: First input array to compare.
        b: Second input array to compare.
        rtol: Relative tolerance used for approximate equality. Default = 1e-05.
        atol: Absolute tolerance used for approximate equality. Default = 1e-08.
        equal_nan: Boolean. If ``True``, NaNs in ``a`` will be considered equal to NaNs in ``b``. Default is ``False``.

    Returns:
        Boolean scalar ``NamedArray` indicating whether the input arrays are element-wise approximately equal within
        the specified tolerances.
    """

    a, b, rtol, atol = broadcast_arrays(a, b, rtol, atol)
    jax_array = jnp.allclose(a.array, b.array, rtol.array, atol.array, equal_nan)
    return constructors.asnamedarray(jax_array, ())


def isclose(
    a: NamedArrayLike,
    b: NamedArrayLike,
    rtol: NamedArrayLike = 1e-05,
    atol: NamedArrayLike = 1e-08,
    equal_nan: bool = False,
) -> NamedArray:
    r"""Check if the elements of two arrays are approximately equal within a tolerance.

    Essentially this function evaluates the following condition:

    .. math::

     |a - b| \le \mathtt{atol} + \mathtt{rtol} * |b|

    ``jnp.inf`` in ``a`` will be considered equal to ``jnp.inf`` in ``b``.

    Args:
        a: First input array to compare.
        b: Second input array to compare.
        rtol: Relative tolerance used for approximate equality. Default = 1e-05.
        atol: Absolute tolerance used for approximate equality. Default = 1e-08.
        equal_nan: Boolean. If ``True``, NaNs in ``a`` will be considered equal to NaNs in ``b``. Default is ``False``.

    Returns:
        A `NamedArray` containing boolean values indicating whether the input arrays are element-wise approximately
        equal within the specified tolerances.
    """

    (a, b, rtol, atol), common_shape = broadcast_arrays_and_shape(a, b, rtol, atol)
    jax_array = jnp.isclose(a.array, b.array, rtol.array, atol.array, equal_nan)
    return constructors.asnamedarray(jax_array, common_shape)


def array_equal(a: NamedArrayLike, b: NamedArrayLike, equal_nan: bool = False) -> NamedArray:
    """Check if two `NamedArray`s are element-wise equal.

    Args:
        a: First input array.
        b: Second input array.
        equal_nan: If ``True``, NaNs in ``a`` are considered equal to NaNs in
            ``b``. Defaults to ``False``.

    Returns:
        A zero-dimensional boolean `NamedArray`.
    """

    a, b = util.ensure_named("array_equal", a, b)
    if set(a.axes) != set(b.axes):
        return constructors.array(False, ())
    b = transpose(b, a.axes)
    return constructors.array(jnp.array_equal(a.array, b.array, equal_nan), ())


def array_equiv(a: NamedArrayLike, b: NamedArrayLike) -> NamedArray:
    """Check if two arrays are broadcast-equivalent.

    Returns ``False`` if the inputs cannot be broadcast to a common shape.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        A zero-dimensional boolean `NamedArray`.
    """
    try:
        eq = ufuncs.equal(a, b)
    except ValueError:
        return constructors.array(False, ())
    return reductions.all(eq)


def clip(a: NamedArrayLike, *, min: NamedArrayLike | None = None, max: NamedArrayLike | None = None) -> NamedArray:
    """Clip array values to a specified range.

    Args:
        a: Array to be clipped.
        min: Optional minimum value of the clipped range; if ``None`` (default), the result will not be clipped to any
            minimum value. If specified, it should be broadcast-compatible with ``a`` and ``max``.
        max: Optional maximum value of the clipped range; if ``None`` (default), the result will not be clipped to any
        maximum value. If specified, it should be broadcast-compatible with ``a`` and ``min``.

    Returns:
        A ``NamedArray`` containing values from ``a``, with values smaller than ``min`` set to ``min``, and values
        larger than ``max`` set to ``max``. Wherever ``min`` is larger than ``max``, the value of ``max`` is returned.
    """
    a = util.ensure_named("clip", a)

    if any(iscomplexobj(t) for t in (a, min, max)):
        raise ValueError(
            "Clip received a complex value either through the input or the min/max keywords. Complex values have no "
            "ordering and cannot be clipped. Please convert to a real value or array by taking the real or imaginary "
            "components via jax.numpy.real/imag respectively."
        )

    if min is not None:
        a = ufuncs.maximum(min, a)
    if max is not None:
        a = ufuncs.minimum(max, a)

    return constructors.array(a.array, a.axes)


def round(a: NamedArrayLike, decimals: int = 0) -> NamedArray:
    """Round input evenly to the given number of decimals.

    Args:
        a: Input array or scalar.
        decimals: Number of decimal points to which the input needs to be rounded. It must be specified statically.
            Not implemented for ``decimals < 0``.

    Returns:
        A ``NamedArray`` containing the rounded values to the specified decimals with same shape and dtype as ``a``
    """

    a = util.ensure_named("round", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.round(x, decimals))


def around(a: NamedArrayLike, decimals: int = 0) -> NamedArray:
    """Alias of ``round``.

    Args:
        a: Input array or scalar.
        decimals: Number of decimal points to which the input is rounded.

    Returns:
        A ``NamedArray`` containing the rounded values.
    """

    return round(a, decimals)


def nan_to_num(
    x: NamedArrayLike, nan: NamedArrayLike, posinf: NamedArrayLike | None = None, neginf: NamedArrayLike | None = None
) -> NamedArray:
    """Replace NaN and infinite entries in an array.

    Args:
        x: The input array.
        nan: Value used to replace NaN entries.
        posinf: Value used to replace positive infinity entries, if specified.
        neginf: Value used to replace negative infinity entries, if specified.

    Returns:
        A ``NamedArray`` with NaN and infinite entries replaced.
    """
    inputs = [x, nan]
    if posinf is not None:
        inputs.append(posinf)
    if neginf is not None:
        inputs.append(neginf)

    arrays, axes = broadcast_arrays_and_shape(*inputs)

    x_array, nan_array = arrays[:2]
    cursor = 2

    posinf_array = None
    if posinf is not None:
        posinf_array = arrays[cursor].array
        cursor += 1

    neginf_array = None
    if neginf is not None:
        neginf_array = arrays[cursor].array

    result = jnp.nan_to_num(
        x_array.array,
        nan=nan_array.array,
        posinf=posinf_array,
        neginf=neginf_array,
    )

    return constructors.array(result, axes)


# Complex-value transforms


def angle(a: NamedArrayLike, deg: bool = False) -> NamedArray:
    """Return the angle of a complex-valued number or array.

    Args:
        a: A complex number or an array of complex numbers.
        deg: If ``True``, returns the result in degrees, otherwise returns in radians. Defaults to ``False``.

    Returns:
        A ``NamedArray`` of counterclockwise angles of each element of ``a``, with the same shape as ``a`` and dtype
        ``float``.
    """

    a = util.ensure_named("angle", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.angle(x, deg))


# Copies and axis adjustments


def copy(a: NamedArrayLike, order: str = "K") -> NamedArray:
    """Return a copy of an array.

    Args:
        a: The array to copy.
        order: Memory layout order for the copy, as in ``jax.numpy.copy``.

    Returns:
        A copy of ``a`` with the same axes.
    """
    a = util.ensure_named("copy", a)
    return constructors.array(jnp.copy(a.array, order=order), a.axes)


def append(
    arr: NamedArrayLike,
    values: NamedArrayLike,
    axis: AxisSelector | None = None,
) -> NamedArray:
    """Append values to an array.

    When ``axis`` is None, both inputs are flattened first.

    Args:
        arr: The array to append to.
        values: The values to append.
        axis: The axis along which to append. If ``None``, both inputs are
            flattened first.

    Returns:
        A ``NamedArray`` containing ``arr`` followed by ``values``.

    Raises:
        ValueError: If the arrays cannot be concatenated along ``axis``.
    """
    if axis is None:
        return concatenate((ravel(arr), ravel(values)), axis=0)
    return concatenate((arr, values), axis=axis)


def expand_dims(
    a: NamedArrayLike,
    axis: AxisLike,
    *,
    index: int = 0,
) -> NamedArray:
    """Insert a new singleton axis.

    ``axis`` describes the new axis, while ``index`` determines its position.

    Args:
        a: The input array.
        axis: The new singleton axis.
        index: Position at which to insert the axis.

    Returns:
        A ``NamedArray`` with the new singleton axis inserted.

    Raises:
        TypeError: If ``index`` is not an integer.
        ValueError: If the new axis name already exists or ``index`` is out of
            range.
    """
    named = util.ensure_named("expand_dims", a)
    new_axis = util.resolve_new_axis(axis, 1, "expand_dims", "axis")

    if new_axis.name is not None and new_axis.name in axis_names(named.axes):
        raise ValueError(f"expand_dims: axis name {new_axis.name!r} already exists")

    position = _normalize_insert_index(index, named.ndim, "expand_dims")
    array = jnp.expand_dims(named.array, axis=position)
    axes = named.axes[:position] + (new_axis,) + named.axes[position:]

    return constructors.array(array, axes)


def squeeze(
    a: NamedArrayLike,
    axis: AxisSelection | None = None,
) -> NamedArray:
    """Remove singleton axes.

    If ``axis`` is specified, every selected axis must have size one.

    Args:
        a: The input array.
        axis: The singleton axis or axes to remove. If ``None``, all singleton
            axes are removed.

    Returns:
        A ``NamedArray`` with the selected axes removed.

    Raises:
        ValueError: If any selected axis does not have size one.
    """
    named = util.ensure_named("squeeze", a)

    if axis is None:
        indices = tuple(i for i, ax in enumerate(named.axes) if ax.size == 1)
    else:
        indices = axis_indices(named.axes, axis)

        for i in indices:
            if named.axes[i].size != 1:
                raise ValueError(f"squeeze: axis {named.axes[i].label!r} has size {named.axes[i].size}, expected 1")

    filtered_axes = remove_axes(named.axes, indices)
    jax_squeeze = jnp.squeeze(named.array, axis=indices if axis is not None else None)

    return constructors.array(jax_squeeze, filtered_axes)


def flip(
    m: NamedArrayLike,
    axis: AxisSelection | None = None,
) -> NamedArray:
    """Reverse the order of elements along selected axes.

    Args:
        m: The input array.
        axis: The axis or axes to reverse. If ``None``, all axes are reversed.

    Returns:
        A ``NamedArray`` with the selected axes reversed.
    """
    named = util.ensure_named("flip", m)
    indices = None if axis is None else axis_indices(named.axes, axis)

    return constructors.array(jnp.flip(named.array, axis=indices), named.axes)


def moveaxis(
    a: NamedArrayLike,
    source: AxisSelection,
    destination: int | Sequence[int],
) -> NamedArray:
    """Move selected axes to new positions.

    Sources are resolved as named-axis selectors. Destinations are positional
    indices, following NumPy's convention.

    Args:
        a: The input array.
        source: The axis or axes to move.
        destination: The positional destination or destinations for the source
            axes.

    Returns:
        A ``NamedArray`` with the selected axes moved.

    Raises:
        TypeError: If a destination is not an integer.
        ValueError: If source and destination counts differ, an axis or
            destination is repeated, or a destination is out of range.
    """
    named = util.ensure_named("moveaxis", a)

    sources = axis_indices(named.axes, source)
    destinations = (destination,) if isinstance(destination, int) else tuple(destination)

    if len(sources) != len(destinations):
        raise ValueError(f"moveaxis: got {len(sources)} source axes and {len(destinations)} destination positions")

    if len(set(sources)) != len(sources):
        raise ValueError("moveaxis: source axes must be distinct")

    ndim = named.ndim
    normalized = []

    for destination in destinations:
        if isinstance(destination, bool) or not isinstance(destination, int):
            raise TypeError("moveaxis: destinations must be integers")
        if not -ndim <= destination < ndim:
            raise ValueError(f"moveaxis: destination {destination} is out of range for {ndim} axes")
        normalized.append(destination % ndim)

    if len(set(normalized)) != len(normalized):
        raise ValueError("moveaxis: destination positions must be distinct")

    order = [i for i in range(ndim) if i not in sources]

    for destination, source in sorted(zip(normalized, sources)):
        order.insert(destination, source)

    return constructors.array(jnp.transpose(named.array, order), tuple(named.axes[i] for i in order))


# Selection, filtering, and search operations


def compress(
    condition: NamedArrayLike,
    a: NamedArrayLike,
    axis: AxisSelector | None = None,
    *,
    size: int | None = None,
    fill_value: NamedArrayLike = 0.0,
) -> NamedArray:
    """Select the slices of ``a`` along an axis where a 1D boolean ``condition`` is true.

    Axis sizes are static in ainmax, so the number of selected slices must be known when the result is built.
    Outside of `jax.jit` it is computed from ``condition``; inside `jax.jit` pass ``size`` explicitly (extra
    slots are filled with ``fill_value``, surplus selections are dropped).

    Args:
        condition: Rank-1 boolean array. Entries beyond its length are treated as ``False``.
        a: The array to select from.
        axis: The axis of ``a`` to select along. If omitted, ``a`` must be 1D, or ``condition``'s axis must be
            named so it can be looked up by name in ``a``.
        size: The static number of selected slices.
        fill_value: Value used to pad the result when ``size`` exceeds the number of selected slices.

    Returns:
        A ``NamedArray`` like ``a`` with the selected axis resized to the number of selected slices.
    """
    cond, named, fill_value = util.ensure_named("compress", condition, a, fill_value)
    if cond.ndim != 1:
        raise ValueError(f"compress: `condition` must be 1D, got {cond.ndim} axes")

    if axis is None:
        if named.ndim == 1:
            axis = 0
        elif cond.axes[0].name is not None:
            axis = cond.axes[0].name
        else:
            raise ValueError("compress: `axis` is required unless `a` is 1D or `condition` has a named axis")

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"compress: axis {axis!r} not found in {named.axes}")

    cond_name, target = cond.axes[0].name, named.axes[idx]
    if cond_name is not None and target.name is not None and cond_name != target.name:
        raise ValueError(f"compress: `condition` axis {cond_name!r} does not match the selected axis {target.name!r}")

    if size is None:
        try:
            size = int(jnp.count_nonzero(cond.array))
        except jax.errors.ConcretizationTypeError as err:
            raise ValueError("compress: pass an explicit `size` when calling under `jax.jit`") from err
    if size < 1:
        raise ValueError(f"compress: result would have {size} slices along {target.label!r}; axes must be non-empty")

    jax_array = jnp.compress(cond.array, named.array, axis=idx, size=size, fill_value=fill_value.array)
    out_axes = named.axes[:idx] + (target.resize(size),) + named.axes[idx + 1 :]
    return constructors.array(jax_array, out_axes)


def extract(
    condition: NamedArrayLike,
    a: NamedArrayLike,
    *,
    size: int | None = None,
    fill_value: NamedArrayLike = 0.0,
):
    """Extract elements from an array where a boolean condition is ``True``.

    Args:
        condition: One-dimensional boolean mask selecting entries along the target axis.
        a: The array to extract from.
        size: Optional static output length. If omitted, it is inferred from the number of ``True`` values.
        fill_value: Value used to pad the extracted result to ``size``.

    Returns:
        A ``NamedArray`` formed by extracting the selected entries along the condition axis.
    """
    cond, a, fill_value = util.ensure_named("extract", condition, a, fill_value)
    if cond.ndim != 1:
        raise ValueError(f"compress: `condition` must be 1D, got {cond.ndim} axes")

    idx = axis_index(a.axes, cond.axes[0])
    if idx is None:
        raise ValueError(f"extract: axis {cond.axes[0]} not found in {a.axes}")

    cond_name, target = cond.axes[0].name, a.axes[idx]
    if cond_name is not None and target.name is not None and cond_name != target.name:
        raise ValueError(f"compress: `condition` axis {cond_name!r} does not match the selected axis {target.name!r}")

    if size is None:
        try:
            size = int(jnp.count_nonzero(cond.array))
        except jax.errors.ConcretizationTypeError as err:
            raise ValueError("compress: pass an explicit `size` when calling under `jax.jit`") from err
    if size < 1:
        raise ValueError(f"compress: result would have {size} slices along {target.label!r}; axes must be non-empty")

    jax_array = jnp.extract(cond.array, a.array, size=size, fill_value=fill_value.array)
    out_axes = a.axes[:idx] + (target.resize(size),) + a.axes[idx + 1 :]
    return constructors.array(jax_array, out_axes)


def select(
    condlist: Sequence[NamedArrayLike],
    choicelist: Sequence[NamedArrayLike],
    default: NamedArrayLike = 0,
) -> NamedArray:
    """Select values from choices according to a list of conditions.

    Args:
        condlist: Boolean arrays specifying which values to select.
        choicelist: Arrays supplying values for the corresponding conditions.
        default: Value used where no condition is true.

    Returns:
        A ``NamedArray`` of values selected from ``choicelist`` and ``default``,
        broadcast to a common shape.

    Raises:
        ValueError: If no conditions are provided or the condition and choice
            lists have different lengths.
    """
    condlist = tuple(condlist)
    choicelist = tuple(choicelist)

    if not condlist:
        raise ValueError("select requires at least one condition")
    if len(condlist) != len(choicelist):
        raise ValueError(f"select: got {len(condlist)} conditions and {len(choicelist)} choices")

    arrays, axes = broadcast_arrays_and_shape(*condlist, *choicelist, default)

    n = len(condlist)
    conditions = arrays[:n]
    choices = arrays[n:-1]
    default = arrays[-1]

    result = jnp.select(
        [condition.array for condition in conditions],
        [choice.array for choice in choices],
        default=default.array,
    )

    return constructors.array(result, axes)


@overload
def where(condition: NamedArrayLike, x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray: ...


@overload
def where(
    condition: NamedArrayLike,
    x: None = None,
    y: None = None,
    /,
    *,
    size: int | Axis | None = None,
    fill_value: Any = None,
) -> tuple[NamedArray, ...]: ...


@overload
def where(
    condition: NamedArrayLike,
    x: NamedArrayLike | None = None,
    y: NamedArrayLike | None = None,
    /,
    *,
    size: int | Axis | None = None,
    fill_value: Any = None,
) -> NamedArray | tuple[NamedArray, ...]: ...


def where(
    condition: NamedArrayLike,
    x: NamedArrayLike | None = None,
    y: NamedArrayLike | None = None,
    /,
    *,
    size: int | Axis | None = None,
    fill_value: Any = None,
) -> NamedArray | tuple[NamedArray, ...]:
    """Select elements by condition, or find the indices where a condition holds.

    Three-argument form: ``where(condition, x, y)``: broadcast the three arguments and take elements of ``x``
    where ``condition`` is true and of ``y`` elsewhere. ``size`` and ``fill_value`` are not allowed here.

    One-argument form: ``where(condition)``: like `numpy.nonzero`, return one integer index array per axis of
    ``condition`` (in ``condition.axes`` order). Each index array has a single new axis enumerating the true
    elements.

    Axis sizes are static in ainmax, so the number of true elements must be known when the result is built.
    Outside of `jax.jit` it is computed from ``condition``; inside `jax.jit` pass ``size`` explicitly.

    Args:
        condition: Boolean (or truthy) array.
        x: Values where ``condition`` is true (three-argument form only).
        y: Values where ``condition`` is false (three-argument form only).
        size: One-argument form only. The static length of the index axis, as an int (anonymous axis) or an `Axis`
            (to also name it). Surplus true elements are dropped; missing slots are filled with ``fill_value``.
        fill_value: One-argument form only. Index used to pad up to ``size``: a scalar, or one scalar per axis of
            ``condition``. Defaults to 0.

    Returns:
        A ``NamedArray`` with the broadcast axes of the inputs (three-argument form), or a tuple of integer
        ``NamedArray`` index arrays (one-argument form).
    """
    if x is None and y is None:
        return _where_indices(condition, size, fill_value)

    if x is None or y is None:
        raise ValueError("where: either both or neither of `x` and `y` must be provided")
    if size is not None or fill_value is not None:
        raise ValueError("where: `size` and `fill_value` cannot be used in the three-argument form")

    condition, x, y = broadcast_arrays(condition, x, y)

    jax_array = jnp.where(condition.array, x.array, y.array)
    return constructors.array(jax_array, x.axes)


def _where_indices(condition: NamedArrayLike, size: int | Axis | None, fill_value: Any) -> tuple[NamedArray, ...]:
    """Return index arrays for the true positions of a boolean condition."""
    cond = util.ensure_named("where", condition)
    if cond.ndim == 0:
        raise ValueError("where: the one-argument form is not defined for zero-dimensional conditions")

    if isinstance(size, Axis):
        index_axis = size
    else:
        if size is None:
            try:
                size = int(jnp.count_nonzero(cond.array))
            except jax.errors.ConcretizationTypeError as err:
                raise ValueError("where: pass an explicit `size` when calling under `jax.jit`") from err
        elif isinstance(size, bool) or not isinstance(size, int):
            raise TypeError(f"where: `size` must be an int or Axis, got {type(size).__name__}")
        if size < 1:
            raise ValueError(f"where: result would have {size} elements; axes must be non-empty")
        index_axis = Axis(size)

    indices = jnp.nonzero(cond.array, size=index_axis.size, fill_value=fill_value)
    return tuple(constructors.array(i, (index_axis,)) for i in indices)


# Splitting and partitioning


def _split_named(
    ary: NamedArrayLike,
    indices_or_sections: int | Sequence[int],
    axis: AxisSelector,
    *,
    equal: bool,
) -> list[NamedArray]:
    """Shared implementation for split and array_split."""
    named = util.ensure_named("array_split" if not equal else "split", ary)

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"split: axis {axis!r} not found in {named.axes}")

    if isinstance(indices_or_sections, bool):
        raise TypeError("split: indices_or_sections must not be bool")

    fn = jnp.split if equal else jnp.array_split
    parts = fn(named.array, indices_or_sections, axis=idx)

    result = []
    for part in parts:
        size = part.shape[idx]
        if size == 0:
            raise ValueError("split: zero-sized axes are not supported")

        axes = named.axes[:idx] + (named.axes[idx].resize(size),) + named.axes[idx + 1 :]
        result.append(constructors.array(part, axes))

    return result


def split(
    ary: NamedArrayLike,
    indices_or_sections: int | Sequence[int],
    axis: AxisSelector = 0,
) -> list[NamedArray]:
    """Split an array into equally sized or explicitly indexed sections.

    Args:
        ary: The array to split.
        indices_or_sections: Number of equal sections or split indices.
        axis: The axis along which to split.

    Returns:
        A list of ``NamedArray`` sections.

    Raises:
        TypeError: If ``indices_or_sections`` is a boolean.
        ValueError: If ``axis`` is not present, the split is invalid, or a
            resulting section would have size zero.
    """
    return _split_named(ary, indices_or_sections, axis, equal=True)


def array_split(
    ary: NamedArrayLike,
    indices_or_sections: int | Sequence[int],
    axis: AxisSelector = 0,
) -> list[NamedArray]:
    """Split an array into sections, allowing unequal section sizes.

    Args:
        ary: The array to split.
        indices_or_sections: Number of sections or split indices.
        axis: The axis along which to split.

    Returns:
        A list of ``NamedArray`` sections.

    Raises:
        TypeError: If ``indices_or_sections`` is a boolean.
        ValueError: If ``axis`` is not present or a resulting section would
            have size zero.
    """
    return _split_named(ary, indices_or_sections, axis, equal=False)


# Bit packing


def packbits(
    a: NamedArrayLike,
    axis: AxisSelector | None = None,
    bitorder: str = "big",
) -> NamedArray:
    """Pack binary-valued elements into uint8 values.

    Args:
        a: The input array.
        axis: Axis along which to pack bits. If ``None``, the input is flattened.
        bitorder: Bit order within each output byte, either ``"big"`` or
            ``"little"``.

    Returns:
        A ``NamedArray`` of packed ``uint8`` values.

    Raises:
        ValueError: If ``axis`` is not present in ``a``.
    """
    named = util.ensure_named("packbits", a)

    if axis is None:
        result = jnp.packbits(named.array, axis=None, bitorder=bitorder)
        return constructors.array(result, (Axis(result.size),))

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"packbits: axis {axis!r} not found in {named.axes}")

    result = jnp.packbits(named.array, axis=idx, bitorder=bitorder)
    axes = named.axes[:idx] + (named.axes[idx].resize(result.shape[idx]),) + named.axes[idx + 1 :]
    return constructors.array(result, axes)


def unpackbits(
    a: NamedArrayLike,
    axis: AxisSelector | None = None,
    count: int | None = None,
    bitorder: str = "big",
) -> NamedArray:
    """Unpack elements of a uint8 array into binary-valued elements.

    Args:
        a: The input array of bytes.
        axis: Axis along which to unpack bits. If ``None``, the input is
            flattened.
        count: Number of bits to unpack along ``axis``.
        bitorder: Bit order within each input byte, either ``"big"`` or
            ``"little"``.

    Returns:
        A ``NamedArray`` of unpacked binary-valued elements.

    Raises:
        ValueError: If ``axis`` is not present in ``a`` or unpacking would
            produce a zero-sized axis.
    """
    named = util.ensure_named("unpackbits", a)

    if axis is None:
        result = jnp.unpackbits(named.array, axis=None, count=count, bitorder=bitorder)
        if result.size == 0:
            raise ValueError("unpackbits: zero-sized axes are not supported")
        return constructors.array(result, (Axis(result.size),))

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"unpackbits: axis {axis!r} not found in {named.axes}")

    result = jnp.unpackbits(named.array, axis=idx, count=count, bitorder=bitorder)
    if result.shape[idx] == 0:
        raise ValueError("unpackbits: zero-sized axes are not supported")

    axes = named.axes[:idx] + (named.axes[idx].resize(result.shape[idx]),) + named.axes[idx + 1 :]
    return constructors.array(result, axes)


def top_k(
    a: NamedArrayLike, k: int, axis: AxisSelector, new_axis: AxisLike | None = None
) -> tuple[NamedArray, NamedArray]:
    """Return the top ``k`` values and their indices along an axis.

    Args:
        a: The input array.
        k: The number of top elements to return.
        axis: The axis along which to find the top ``k`` elements.
        new_axis: The axis descriptor for the output. If a string, creates an
            anonymous axis of size ``k`` with that name; if ``None``, the
            existing axis is resized to ``k``.

    Returns:
        A tuple ``(values, indices)`` of `NamedArray`s sharing the same axes.
    """
    named = util.ensure_named("top_k", a)

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"Axis {axis} not found in {a}")

    values, indices = jax.lax.top_k(named.array, k=k, axis=idx)

    if new_axis is None:
        axis = resolve_axes(named.axes, axis)[0]
        new_axis = axis.resize(k)
    elif isinstance(new_axis, str):
        new_axis = Axis(k, new_axis)

    updated_axes = named.axes[:idx] + (new_axis,) + named.axes[idx + 1 :]
    return constructors.array(values, updated_axes), constructors.array(indices, updated_axes)


def searchsorted(
    a: NamedArrayLike,
    v: NamedArrayLike,
    side: str = "left",
    sorter: NamedArrayLike | None = None,
    *,
    method: str = "scan",
) -> NamedArray:
    """Find indices where elements should be inserted to keep ``a`` sorted.

    Args:
        a: A 1-D array of sorted values to search in.
        v: The values whose insertion indices are sought.
        side: ``'left'`` or ``'right'``. Defaults to ``'left'``.
        sorter: Optional array of indices that sorts ``a``.
        method: Search method passed to :func:`jax.numpy.searchsorted`. Defaults to ``'scan'``.

    Returns:
        A `NamedArray` with the same axes as ``v``.
    """
    if sorter is None:
        a, v = util.ensure_named("searchsorted", a, v)
    else:
        a, v, sorter = util.ensure_named("searchsorted", a, v, sorter)

    if a.ndim != 1:
        raise ValueError("`seachsorted` only supports rank-1 `a`.")

    jax_sorter = sorter.array if sorter is not None else None
    jax_array = jnp.searchsorted(a.array, v.array, side=side, sorter=jax_sorter, method=method)

    return constructors.array(jax_array, v.axes)


def isin(
    element: NamedArrayLike,
    test_elements: NamedArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    method: str = "auto",
) -> NamedArray:
    """Test whether each element occurs in a collection of values.

    Args:
        element: Values to test.
        test_elements: Collection of values to search.
        assume_unique: If ``True``, assume both inputs contain unique values.
        invert: If ``True``, return the complement of the membership test.
        method: Search method, as accepted by ``jax.numpy.isin``.

    Returns:
        A boolean ``NamedArray`` with the same axes as ``element``.
    """
    element, test_elements = util.ensure_named("isin", element, test_elements)

    result = jnp.isin(
        element.array,
        test_elements.array,
        assume_unique=assume_unique,
        invert=invert,
        method=method,
    )

    return constructors.array(result, element.axes)


def bincount(
    x: NamedArrayLike,
    weights: NamedArrayLike | None = None,
    minlength: int = 0,
    *,
    length: int | None = None,
    axis: AxisLike | None = None,
) -> NamedArray:
    """Count occurrences of non-negative integers.

    Each input value contributes one count to its corresponding bin. If
    ``weights`` is supplied, its values are accumulated instead of unit counts.

    Args:
        x: One-dimensional array of non-negative integer values to count.
        weights: Optional array containing one weight for each value in ``x``.
        minlength: Minimum number of bins in the result.
        length: Optional explicit number of bins in the result.
        axis: Optional name or descriptor for the result's bin axis. If an
            ``Axis`` is supplied, its size must match the result length.

    Returns:
        A one-dimensional ``NamedArray`` containing the count or weighted sum
        for each bin.

    Raises:
        ValueError: If ``x`` is not one-dimensional, ``weights`` does not
            contain one value per input, or ``axis`` has the wrong size.
    """
    x = util.ensure_named("bincount", x)
    if x.ndim != 1:
        raise ValueError(f"expected a one-dimensional array, got {x.ndim} dimensions")

    if weights is not None:
        weights = util.ensure_named("bincount", weights)
        if x.ndim != 1:
            raise ValueError(f"expected a one-dimensional array, got {x.ndim} dimensions")
        if weights.size != x.size:
            raise ValueError("bincount: weights must have the same number of elements as x")

    result = jnp.bincount(
        x.array,
        weights=None if weights is None else weights.array,
        minlength=minlength,
        length=length,
    )

    return constructors.array(result, (_result_axis(axis, result.size, "bincount"),))


def digitize(
    x: NamedArrayLike,
    bins: NamedArrayLike,
    right: bool = False,
    *,
    method: str | None = None,
) -> NamedArray:
    """Return the indices of the bins to which values belong.

    Args:
        x: Values to assign to bins.
        bins: One-dimensional array of bin boundaries.
        right: If ``True``, use the right side of each bin for comparisons.
        method: Optional method for locating bins, as accepted by
            ``jax.numpy.digitize``.

    Returns:
        An integer ``NamedArray`` with the same axes as ``x``.

    Raises:
        ValueError: If ``bins`` is not one-dimensional.
    """
    x, bins = util.ensure_named("digitize", x, bins)

    if bins.ndim != 1:
        raise ValueError("digitize: `bins` must be one-dimensional")

    result = jnp.digitize(x.array, bins.array, right=right, method=method)
    return constructors.array(result, x.axes)


def _result_axis(
    specification: AxisLike | None,
    size: int,
    name: str,
) -> Axis:
    """Construct an output axis, preserving an explicitly supplied name."""
    if isinstance(specification, Axis):
        if specification.size != size:
            raise ValueError(f"{name}: axis has size {specification.size}, expected {size}")
        return specification

    return Axis(size, specification)


def _histogram_inputs(
    a: NamedArrayLike,
    weights: NamedArrayLike | None,
) -> tuple[NamedArray, Any]:
    """Flatten histogram samples and optional weights."""
    a = util.ensure_named("histogram", a)

    if weights is None:
        return a, None

    weights = util.ensure_named("histogram", weights)
    if weights.size != a.size:
        raise ValueError("histogram: weights must have the same number of elements as the input")

    return a, weights.array.reshape(-1)


def _histogram_axis_pair(
    axis: AxisLike | None,
    bins: int,
    name: str,
) -> tuple[Axis, Axis]:
    """Construct matching bin and bin-edge axes."""
    if isinstance(axis, Axis):
        if axis.size != bins:
            raise ValueError(f"{name}: bin axis has size {axis.size}, expected {bins}")
        label = axis.name
    else:
        label = axis

    return Axis(bins, label), Axis(bins + 1, label)


def histogram(
    a: NamedArrayLike,
    bins: int | NamedArrayLike = 10,
    range: Sequence[Any] | None = None,
    weights: NamedArrayLike | None = None,
    density: bool | None = None,
    *,
    axis: AxisLike | None = None,
) -> tuple[NamedArray, NamedArray]:
    """Compute a one-dimensional histogram.

    The input and optional weights are flattened before computing the
    histogram. When ``bins`` is an array of bin edges with a named axis and
    ``axis`` is not specified, that name is used for the histogram's bin axis.

    Args:
        a: Input values to bin.
        bins: Number of equally spaced bins, or an array of bin edges.
        range: Optional ``(minimum, maximum)`` range used when ``bins`` is an
            integer.
        weights: Optional weights with one value per input element.
        density: If true, normalize the result to form a probability density.
        axis: Optional name or descriptor for the histogram's bin axis. If an
            ``Axis`` is supplied, its size must match the number of bins. The
            returned edge array uses the same name with one additional element.

    Returns:
        A tuple ``(hist, bin_edges)``. ``hist`` contains the bin counts or
        weighted sums, and ``bin_edges`` contains the bin boundaries.

    Raises:
        ValueError: If ``weights`` does not contain one value per input
            element, or ``axis`` has the wrong size.
    """
    a, weights_array = _histogram_inputs(a, weights)

    if isinstance(bins, NamedArray):
        bins_array = bins.array
        inferred_name = bins.axes[0].name if bins.ndim == 1 else None
    else:
        bins_array = bins
        inferred_name = None

    counts, edges = jnp.histogram(
        a.array.reshape(-1),
        bins=bins_array,
        range=range,
        weights=weights_array,
        density=density,
    )

    specification = axis if axis is not None else inferred_name
    bin_axis, edge_axis = _histogram_axis_pair(specification, counts.size, "histogram")

    return (
        constructors.array(counts, (bin_axis,)),
        constructors.array(edges, (edge_axis,)),
    )


def histogram_bin_edges(
    a: NamedArrayLike,
    bins: int | NamedArrayLike = 10,
    range: Sequence[Any] | None = None,
    weights: NamedArrayLike | None = None,
    *,
    axis: AxisLike | None = None,
) -> NamedArray:
    """Compute the bin edges for a one-dimensional histogram.

    The input and optional weights are flattened before computing the edges.
    When ``bins`` is an array of edges with a named axis and ``axis`` is not
    specified, that name is used for the output axis.

    Args:
        a: Input values used to determine the bin edges.
        bins: Number of equally spaced bins, or an array of bin edges.
        range: Optional ``(minimum, maximum)`` range used when ``bins`` is an
            integer.
        weights: Optional weights with one value per input element, passed to
            the bin-edge calculation.
        axis: Optional name or descriptor for the output edge axis. If an
            ``Axis`` is supplied, its size must match the number of edges.

    Returns:
        A one-dimensional ``NamedArray`` containing the bin edges.

    Raises:
        ValueError: If ``weights`` does not contain one value per input
            element, or ``axis`` has the wrong size.
    """
    a, weights_array = _histogram_inputs(a, weights)

    if isinstance(bins, NamedArray):
        bins_array = bins.array
        inferred_name = bins.axes[0].name if bins.ndim == 1 else None
    else:
        bins_array = bins
        inferred_name = None

    edges = jnp.histogram_bin_edges(
        a.array.reshape(-1),
        bins=bins_array,
        range=range,
        weights=weights_array,
    )

    specification = axis if axis is not None else inferred_name
    edge_axis = _result_axis(specification, edges.size, "histogram_bin_edges")

    return constructors.array(edges, (edge_axis,))


def histogram2d(
    x: NamedArrayLike,
    y: NamedArrayLike,
    bins: Any = 10,
    range: Sequence[Any] | None = None,
    weights: NamedArrayLike | None = None,
    density: bool | None = None,
    *,
    axes: tuple[AxisLike | None, AxisLike | None] | None = None,
) -> tuple[NamedArray, NamedArray, NamedArray]:
    """Compute a two-dimensional histogram.

    ``x`` and ``y`` are flattened and treated as paired coordinates. The
    optional ``axes`` names the two histogram dimensions; each returned edge
    array uses the name of its corresponding histogram axis.

    Args:
        x: Values for the first coordinate.
        y: Values for the second coordinate.
        bins: Bin count or bin edges for both dimensions, or a pair of bin
            specifications.
        range: Optional ranges for the two dimensions.
        weights: Optional weight for each paired coordinate.
        density: If true, normalize the result to form a probability density.
        axes: Optional pair of names or axis descriptors for the histogram's
            first and second dimensions. Supplied ``Axis`` sizes must match
            the corresponding bin counts.

    Returns:
        A tuple ``(hist, x_edges, y_edges)`` containing the two-dimensional
        histogram and the bin edges for each dimension.

    Raises:
        ValueError: If ``x`` and ``y`` have different numbers of elements,
            ``weights`` does not contain one value per coordinate, or an
            output axis has the wrong size.
    """
    x = util.ensure_named("histogram2d", x)
    y = util.ensure_named("histogram2d", y)

    if x.size != y.size:
        raise ValueError("histogram2d: x and y must contain the same number of elements")

    weights_array = None
    if weights is not None:
        weights = util.ensure_named("histogram2d", weights)
        if weights.size != x.size:
            raise ValueError("histogram2d: weights must have the same number of elements as x and y")
        weights_array = weights.array.reshape(-1)

    result, xedges, yedges = jnp.histogram2d(
        x.array.reshape(-1),
        y.array.reshape(-1),
        bins=bins,
        range=range,
        weights=weights_array,
        density=density,
    )

    axis_specs = axes if axes is not None else (None, None)
    x_axis = _result_axis(axis_specs[0], result.shape[0], "histogram2d")
    y_axis = _result_axis(axis_specs[1], result.shape[1], "histogram2d")

    return (
        constructors.array(result, (x_axis, y_axis)),
        constructors.array(xedges, (x_axis.resize(xedges.size),)),
        constructors.array(yedges, (y_axis.resize(yedges.size),)),
    )


def histogramdd(
    sample: NamedArrayLike,
    bins: Any = 10,
    range: Sequence[Any] | None = None,
    weights: NamedArrayLike | None = None,
    density: bool | None = None,
    *,
    axes: Sequence[AxisLike | None] | None = None,
) -> tuple[NamedArray, list[NamedArray]]:
    """Compute an N-dimensional histogram from a sample matrix.

    Each row of ``sample`` is one observation and each column is a dimension.

    Args:
        sample: A two-dimensional array with shape ``(num_samples,
            num_dimensions)``.
        bins: Bin count or bin edges, either shared across dimensions or
            specified separately for each dimension.
        range: Optional range for each sample dimension.
        weights: Optional weight for each sample row.
        density: If true, normalize the result to form a probability density.
        axes: Optional name or axis descriptor for each histogram dimension.
            Each supplied ``Axis`` must have the corresponding bin count.

    Returns:
        A tuple ``(hist, edges)``. ``hist`` is the N-dimensional histogram;
        ``edges`` is a list of one-dimensional ``NamedArray`` bin-edge arrays,
        one for each sample dimension.

    Raises:
        ValueError: If ``sample`` is not two-dimensional, ``weights`` does
            not contain one value per sample, or ``axes`` has the wrong
            number of entries or a supplied axis has the wrong size.
    """
    sample = util.ensure_named("histogramdd", sample)

    if sample.ndim != 2:
        raise ValueError("histogramdd: sample must have shape (N, D)")

    weights_array = None
    if weights is not None:
        weights = util.ensure_named("histogramdd", weights)
        if weights.size != sample.axes[0].size:
            raise ValueError("histogramdd: weights must have one entry per sample")
        weights_array = weights.array.reshape(-1)

    result, edges = jnp.histogramdd(
        sample.array,
        bins=bins,
        range=range,
        weights=weights_array,
        density=density,
    )

    if axes is None:
        axis_specs = (None,) * result.ndim
    else:
        axis_specs = tuple(axes)
        if len(axis_specs) != result.ndim:
            raise ValueError(
                f"histogramdd: got {len(axis_specs)} output axes for a {result.ndim}-dimensional histogram"
            )

    result_axes = tuple(_result_axis(spec, size, "histogramdd") for spec, size in zip(axis_specs, result.shape))

    edge_arrays = [constructors.array(edge, (result_axes[i].resize(edge.size),)) for i, edge in enumerate(edges)]

    return constructors.array(result, result_axes), edge_arrays
