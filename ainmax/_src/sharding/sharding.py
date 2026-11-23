from collections.abc import Callable
from functools import partial, wraps
from typing import Any

import jax

from ainmax._src.filters import is_named_array
from ainmax._src.sharding import partition, util


PyTree = Any


def reshard(xs: PyTree, out_shardings: partition.ShardingLike) -> PyTree:
    """Reshard the array leaves of a pytree.

    NamedArray leaves use their logical axes to resolve ``PM`` specifications.

    Args:
        xs: Pytree whose array leaves should be resharded.
        out_shardings: Sharding specification for the output leaves. A
            ``PM`` is resolved using each NamedArray's logical axes.

    Returns:
        A pytree with the same structure as ``xs`` and the requested sharding.
    """
    return util._map_canonicalized("reshard", xs, out_shardings, jax.reshard)


def device_put(x, device=None, *, src=None, may_alias=False, donate=False):
    """Copy array leaves of a pytree to a device or sharding.

    This follows :func:`jax.device_put` for individual leaves while preserving
    NamedArray metadata and supporting ``PM`` specifications for NamedArrays.

    Args:
        x: Value or pytree to place on a device.
        device: Device or sharding specification for the output leaves. If
            omitted, JAX chooses the placement for each leaf.
        src: Optional source device or sharding, as in ``jax.device_put``.
        may_alias: Whether the input and output buffers may alias.
        donate: Whether to donate the input buffers.

    Returns:
        ``x`` placed on the requested device or sharding.
    """
    op = partial(jax.device_put, src=src, may_alias=may_alias, donate=donate)
    if device is None:
        return jax.tree.map(op, x, is_leaf=is_named_array)
    return util._map_canonicalized("device_put", x, device, op)


def with_sharding_constraint(x: PyTree, shardings: PyTree) -> PyTree:
    """Constrain the sharding of each array leaf in a pytree.

    This follows :func:`jax.lax.with_sharding_constraint` for individual
    leaves. ``PM`` specifications are resolved using the logical axes of
    NamedArray leaves.

    Args:
        x: Pytree whose array leaves should be constrained.
        shardings: Pytree of sharding specifications with a structure matching
            ``x``. A ``PM`` is resolved using each NamedArray's logical axes.

    Returns:
        A pytree with the same structure as ``x`` and constrained sharding.
    """
    return util._map_canonicalized("with_sharding_constraint", x, shardings, jax.lax.with_sharding_constraint)


def auto_mesh_axes(fn: Callable[..., Any], out_sharding: partition.ShardingLike | PyTree | None, **hoist_kwargs):
    """Switches some or all mesh axis types to `Auto`, which allows adding an `out_sharding` argument to any
    composition of operations.

    Mesh axes can be either `Auto` or `Explicit`, but not `Manual`.

    A ``PM`` ``out_sharding`` is resolved against the *output* axes of ``fn``. Those are only known once ``fn`` has
    been traced, so the wrapper first runs :func:`jax.eval_shape` on sharding-free abstract inputs (no FLOPs, and
    it never sees the explicit mesh, so ops that cannot run on explicitly sharded axes still trace) and resolves the
    ``PM`` against the resulting axes at call time. ``out_sharding`` may also be a pytree (prefix) of shardings
    matching the output. ``NamedSharding``/``P`` shardings need no tracing and are applied directly.

    Keyword arguments in ``hoist_kwargs`` are bound to ``fn`` before applying
    JAX's automatic mesh transformation.
    """
    f = partial(fn, **hoist_kwargs)

    if out_sharding is None:
        return f

    if not util.has_pm(out_sharding):
        return jax.sharding.auto_axes(f, out_sharding=out_sharding)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        out_shape = jax.eval_shape(f, *util.shape_dtype_struct(args), **util.shape_dtype_struct(kwargs))
        _, resolved, treedef = util.flatten_pytree_shardings("auto_mesh", out_shape, out_sharding)
        shardings = jax.tree.unflatten(treedef, resolved)
        return jax.sharding.auto_axes(f, out_sharding=shardings)(*args, **kwargs)

    return wrapper
