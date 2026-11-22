import math
from collections.abc import Callable, Hashable, Set
from functools import partial, update_wrapper
from typing import overload, TypeVar

import equinox as eqx
import jax.sharding
import jax.tree
from jax import P, shard_map as jax_shard_map
from jaxtyping import PyTree

from ainmax._src.array import NamedArrayImpl
from ainmax._src.axis import Axis
from ainmax._src.filters import is_arrayish, is_named_array
from ainmax._src.sharding import _flatten_pytree_shardings, _has_pm


T = TypeVar("T")
F = TypeVar("F", bound=Callable)
G = TypeVar("G", bound=Callable)

MeshLike = jax.sharding.Mesh | jax.sharding.AbstractMesh


def _shard_factor(entry, sizes: dict[str, int]) -> int:
    names = () if entry is None else ((entry,) if isinstance(entry, str) else tuple(entry))
    return math.prod(sizes.get(n, 1) for n in names)


def _resize_axes(axes: tuple[Axis, ...], spec, sizes: dict[str, int], op: str) -> tuple[Axis, ...]:
    spec = tuple(spec) + (None,) * (len(axes) - len(spec))
    out = []
    for a, entry in zip(axes, spec):
        k = _shard_factor(entry, sizes)
        if op == "local" and a.size % k:
            raise ValueError(f"shard_map: axis {a} is not divisible by the {k} shards of mesh axes {entry!r}")
        out.append(Axis(a.size // k if op == "local" else a.size * k, a.name))
    return tuple(out)


@overload
def shard_map(
    f: F,
    /,
    *,
    out_specs: PyTree,
    in_specs: PyTree | None = ...,
    mesh: MeshLike | None = ...,
    axis_names: Set[Hashable] = ...,
    check_vma: bool = ...,
) -> F: ...


@overload
def shard_map(
    f: None = None,
    /,
    *,
    out_specs: PyTree,
    in_specs: PyTree | None = ...,
    mesh: MeshLike | None = ...,
    axis_names: Set[Hashable] = ...,
    check_vma: bool = ...,
) -> Callable[[G], G]: ...


def shard_map(
    f: F | None = None,
    /,
    *,
    out_specs: PyTree,
    in_specs: PyTree | None = None,
    mesh: MeshLike | None = None,
    axis_names: Set[Hashable] = frozenset(),
    check_vma: bool = True,
) -> F | Callable[[G], G]:
    """NamedArray-aware wrapper for `jax.shard_map`.

    Map a function over shards of data using a mesh of devices.

    Args:
        f: callable to be mapped. Each application of `f`, or “instance” of `f`, takes as input a shard of the
            mapped-over arguments and produces a shard of the output.
        mesh: (optional, default None) a `jax.sharding.Mesh` representing the array of devices over which to shard the
            data and on which to execute instances of `f`. The names of the `Mesh` can be used in collective
            communication operations in ``f``. If mesh is None, it will be inferred from the context which can be set
            via `jax.set_mesh` context manager.
        in_specs: (optional, default `Infer`) a pytree with ``jax.sharding.PartitionSpec`` instances as leaves, with a
            tree structure that is a tree prefix of the args tuple to be mapped over. Similar to
            `jax.sharding.NamedSharding`, each ``PartitionSpec`` represents how the corresponding argument (or subtree
            of arguments) should be sharded along the named axes of `mesh`. In each `PartitionSpec`, mentioning a
            `mesh` axis name at a position expresses sharding the corresponding argument array axis along that
            positional axis; not mentioning an axis name expresses replication.

            If ``Infer``, all mesh axes must be of type `Explicit`, in which case the in_specs are inferred from the
            argument types. If ``None``, inputs will be treated as static.
        out_specs: a pytree with `PartitionSpec` instances as leaves, with a tree structure that is a tree prefix of
            the output of `f`. Each `PartitionSpec` represents how the corresponding output shards should be
            concatenated. In each `PartitionSpec`, mentioning a `mesh` axis name at a position expresses concatenation
            of that mesh axis's shards along the corresponding positional axis; not mentioning a `mesh` axis name
            expresses a promise that the output values are equal along that mesh axis, and that rather than
            concatenating only a single value should be produced.
        axis_names: (optional, default set()) set of axis names from `mesh` over which the function `f` is manual. If
            empty, `f`, is manual over all mesh axes.
        check_vma: (optional) boolean (default True) representing whether to enable additional validity checks and
            automatic differentiation optimizations. The validity checks concern whether any mesh axis names not
            mentioned in `out_specs` are consistent with how the outputs of `f` are replicated.

    Returns:
        A callable representing a mapped version of `f`, which accepts positional arguments corresponding to those of
        `f` and produces output corresponding to that of `f`.
    """

    if f is None:
        return partial(
            shard_map,
            out_specs=out_specs,
            in_specs=in_specs,
            mesh=mesh,
            axis_names=axis_names,
            check_vma=check_vma,
        )

    needs_pm_discovery = _has_pm(out_specs)

    def wrapped(*args):
        mesh_ = mesh if mesh is not None else jax.sharding.get_abstract_mesh()
        manual = set(axis_names) or set(mesh_.axis_names)
        sizes = {n: s for n, s in mesh_.shape.items() if n in manual}

        in_named, in_static = eqx.partition(args, is_arrayish, is_leaf=is_named_array)
        in_flat, in_treedef = jax.tree.flatten(in_named, is_leaf=is_named_array)
        in_arrays = [x.array if is_named_array(x) else x for x in in_flat]

        if in_specs is not None:
            _, in_specs_flat, _ = _flatten_pytree_shardings("shard_map", in_named, in_specs)
        else:
            in_specs_flat = [jax.typeof(a).sharding.spec for a in in_arrays]

        arrays = jax.tree.unflatten(in_treedef, in_arrays)
        in_specs_tree = jax.tree.unflatten(in_treedef, in_specs_flat)

        captured: dict = {}

        def inner(tree):
            # `f` sees *local* NamedArrays: sharded axes divided by the shard count.
            leaves = jax.tree.leaves(tree)
            local = [
                NamedArrayImpl(a, _resize_axes(meta.axes, spec, sizes, "local")) if is_named_array(meta) else a
                for meta, a, spec in zip(in_flat, leaves, in_specs_flat, strict=True)
            ]
            full_args = eqx.combine(jax.tree.unflatten(in_treedef, local), in_static, is_leaf=is_named_array)
            result = f(*full_args)
            out_named, _ = eqx.partition(result, is_arrayish, is_leaf=is_named_array)
            out_flat, out_treedef = jax.tree.flatten(out_named, is_leaf=is_named_array)
            captured["treedef"] = out_treedef
            captured["axes"] = [x.axes if is_named_array(x) else None for x in out_flat]
            return [x.array if is_named_array(x) else x for x in out_flat]

        if needs_pm_discovery:
            discovery = jax_shard_map(
                inner, mesh=mesh, in_specs=(in_specs_tree,), out_specs=P(), axis_names=axis_names, check_vma=False
            )
            local_shapes = jax.eval_shape(discovery, arrays)
            meta = [
                NamedArrayImpl(s, axes) if axes is not None else s
                for s, axes in zip(local_shapes, captured["axes"], strict=True)
            ]
            meta_tree = jax.tree.unflatten(captured["treedef"], meta)
            _, out_specs_flat, _ = _flatten_pytree_shardings("shard_map", meta_tree, out_specs)
            resolved_out_specs = list(out_specs_flat)
        else:
            resolved_out_specs = out_specs

        mapped = jax_shard_map(
            inner,
            mesh=mesh,
            in_specs=(in_specs_tree,),
            out_specs=resolved_out_specs,
            axis_names=axis_names,
            check_vma=check_vma,
        )
        raw_out = mapped(arrays)

        out_treedef, out_axes = captured["treedef"], captured["axes"]
        if needs_pm_discovery:
            flat_out_specs = resolved_out_specs
        else:
            meta = [
                NamedArrayImpl(arr, axes) if axes is not None else arr
                for arr, axes in zip(raw_out, out_axes, strict=True)
            ]
            meta_tree = jax.tree.unflatten(out_treedef, meta)
            _, flat_out_specs, _ = _flatten_pytree_shardings("shard_map", meta_tree, out_specs)
        rebuilt = [
            NamedArrayImpl(arr, _resize_axes(axes, spec, sizes, "global")) if axes is not None else arr
            for arr, axes, spec in zip(raw_out, out_axes, flat_out_specs, strict=True)
        ]
        return jax.tree.unflatten(out_treedef, rebuilt)

    return update_wrapper(wrapped, f)


## collectives from jax/_src/lax/parallel.py
