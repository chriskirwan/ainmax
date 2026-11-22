from collections.abc import Callable
from typing import Any

import jax.sharding
import jax.tree
from jax import Device, NamedSharding, P
from jax.api_util import flatten_axes
from jaxtyping import PyTree

from ainmax._src.axis import AxisSpec
from ainmax._src.basearray import NamedArray
from ainmax._src.filters import assert_single_array_leaf, is_named_array
from ainmax._src.sharding.partition import PM, ShardingLike


def resolve_sharding(axes: AxisSpec, sharding: ShardingLike | None, fn_name: str) -> NamedSharding | P | None:
    """Convert an ainmax sharding specification to a JAX sharding specification.

    ``PM`` specifications are resolved against the current JAX mesh and the
    supplied logical axes.
    """
    if sharding is None:
        return None

    if isinstance(sharding, (NamedSharding, P)):
        return sharding

    if not isinstance(sharding, ShardingLike):
        raise TypeError(
            f"`out_sharding` argument of {fn_name} only supports instances of "
            f"`NamedSharding`, `PartitionSpec`, or `PM`. Got {sharding} of type: {type(sharding)}"
        )

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        raise ValueError(
            "Using `PM` when you are not inside a mesh context is not allowed. Please pass a "
            "`NamedSharding` instance or enter into a mesh context via `jax.set_mesh`"
        )

    partition_spec = sharding.partition_spec(axes)
    return partition_spec


def _has_pm(sharding: Any) -> bool:
    return any(isinstance(leaf, PM) for leaf in jax.tree.leaves(sharding, is_leaf=lambda x: isinstance(x, PM)))


def _shape_dtype_struct(tree: PyTree) -> PyTree:
    """Replace array leaves by ``ShapeDtypeStruct``"""

    def leaf(x):
        return jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, "shape") and hasattr(x, "dtype") else x

    return jax.tree.map(leaf, tree)


def _resolve_leaf_sharding(
    fn_name: str, x: PyTree, sharding: Device | ShardingLike | None
) -> Device | NamedSharding | P | None:
    """Canonicalize a per-leaf sharding, resolving PM for NamedArray leaves."""
    if isinstance(x, NamedArray):
        return resolve_sharding(x.axes, sharding, fn_name)
    if isinstance(sharding, PM):
        raise TypeError(
            f"{fn_name}: `PM` PartitionSpec supplied for a non-NamedArray leaf "
            f"(type {type(x).__name__}); use `jax.P` or `jax.NamedSharding` instead."
        )
    return sharding


def _flatten_pytree_shardings(
    fn_name: str,
    xs: PyTree,
    shardings: PyTree,
) -> tuple[list[PyTree], list[Device | NamedSharding | P | None], PyTree]:
    flat_xs, treedef = jax.tree.flatten(xs, is_leaf=is_named_array)
    for x in flat_xs:
        assert_single_array_leaf(x, fn_name)
    flat_shardings = flatten_axes(fn_name, treedef, shardings)
    canonical_shardings = [_resolve_leaf_sharding(fn_name, x, shard) for x, shard in zip(flat_xs, flat_shardings)]
    return flat_xs, canonical_shardings, treedef


def _map_canonicalized(fn_name: str, xs: PyTree, shardings: PyTree, op: Callable[[Any, Any], Any]) -> PyTree:
    flat_xs, canonical, treedef = _flatten_pytree_shardings(fn_name, xs, shardings)
    flat_out = [op(xi, c) for xi, c in zip(flat_xs, canonical)]
    return jax.tree.unflatten(treedef, flat_out)
