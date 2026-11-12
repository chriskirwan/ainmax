from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, wraps
from math import isqrt
from typing import Any, cast, Literal

import jax
import jax.numpy as jnp

from ainmax._src.axis import Axis, AxisSelection
from ainmax._src.axis_util import axis_index, axis_selection_to_tuple, check_unique_axis_names
from ainmax._src.feidhm import array, transpose
from ainmax._src.typing import NamedArray


def _is_named(x):
    return isinstance(x, NamedArray)


def _resolve_positions(leaf: NamedArray, refs) -> tuple[int | None, ...]:
    positions = []
    for r in refs:
        p = axis_index(leaf.axes, r)  # raises on ambiguous names
        if p is None and isinstance(r, int):
            raise ValueError(f"vmap: positional axis {r} is out of range for axes {leaf.axes}")
        positions.append(p)
    found = [p for p in positions if p is not None]
    if len(found) != len(set(found)):
        raise ValueError(f"vmap: several selectors in {refs} resolve to the same axis of {leaf.axes}")
    return tuple(positions)


def _level_in_axis(positions, k):
    """in_axes entry for level k: outer levels have already stripped their axes."""
    p = positions[k]
    if p is None:
        return None
    return p - sum(q is not None and q < p for q in positions[:k])


def _unify_mapped_axes(leaves, all_positions, refs) -> tuple[Axis, ...]:
    mapped: list[Axis | None] = [None] * len(refs)
    for leaf, positions in zip(leaves, all_positions):
        if positions is None:
            continue
        for k, p in enumerate(positions):
            if p is None:
                continue
            ax, cur = leaf.axes[p], mapped[k]
            if cur is None:
                mapped[k] = ax
                continue
            if cur.size != ax.size:
                raise ValueError(f"vmap: axis {refs[k]!r} has inconsistent sizes: {cur.size} vs {ax.size}")
            if cur.name is not None and ax.name is not None and cur.name != ax.name:
                raise ValueError(f"vmap: selector {refs[k]!r} hits differently named axes {cur.name!r}/{ax.name!r}")
            if cur.name is None:
                mapped[k] = ax
    missing = [refs[k] for k, m in enumerate(mapped) if m is None]
    if missing:
        raise ValueError(f"vmap: no argument has axis {missing}")
    return tuple(mapped)  # type: ignore[arg-type]


def vmap[F: Callable](fn: F, in_axes: AxisSelection, out_axes: AxisSelection | None = None) -> F:
    """Vectorize `fn` over one or more axes of its `NamedArray` arguments.

    Args:
        fn: function to vectorize. Arguments and outputs may be arbitrary pytrees whose `NamedLike` leaves carry
            the axes being mapped.
        in_axes: axes to map over, selected by name, `Axis` or position. The first selector becomes the outermost
            mapped axis in the output.
        out_axes: optional reordering of each `NamedLike` output's axes, applied via `transpose` after mapping.
            Unlike `jax.vmap`, this is *not* the position at which the mapped axes are inserted.
    """

    if not callable(fn):
        raise TypeError(f"vmap: expected a callable, got {type(fn).__name__}")
    refs = axis_selection_to_tuple(in_axes)
    if not refs:
        raise ValueError("vmap: `axis` must select at least one axis")

    @wraps(fn)
    def wrapped(*args, **kwargs):
        leaves, treedef = jax.tree.flatten((args, kwargs), is_leaf=_is_named)
        positions = [_resolve_positions(l, refs) if _is_named(l) else None for l in leaves]
        mapped = _unify_mapped_axes(leaves, positions, refs)
        raw_leaves = [l.array if _is_named(l) else l for l in leaves]
        out_cell: list = []

        def core(*raw):
            rebuilt = []
            for leaf, pos, r in zip(leaves, positions, raw):
                if pos is None:
                    rebuilt.append(r)
                    continue
                drop = {p for p in pos if p is not None}
                rebuilt.append(array(r, tuple(a for i, a in enumerate(leaf.axes) if i not in drop)))
            a, kw = jax.tree.unflatten(treedef, rebuilt)
            out_leaves, out_def = jax.tree.flatten(fn(*a, **kw), is_leaf=_is_named)
            out_cell[:] = [out_def, [o.axes if _is_named(o) else None for o in out_leaves]]
            return [o.array if _is_named(o) else o for o in out_leaves]

        f = core
        for k in reversed(range(len(refs))):
            in_axes = tuple(None if pos is None else _level_in_axis(pos, k) for pos in positions)
            f = jax.vmap(f, in_axes=in_axes, out_axes=0, axis_name=mapped[k].name)
        raw_out = f(*raw_leaves)

        out_def, out_axes_info = out_cell
        wrapped_out = []
        for o, info in zip(raw_out, out_axes_info):
            if info is None:  # plain array: mapped axes named, the rest anonymous
                info = tuple(Axis(s) for s in o.shape[len(mapped) :])
            axes = mapped + tuple(info)
            check_unique_axis_names(axes)
            wrapped_out.append(array(o, axes))
        out = jax.tree.unflatten(out_def, wrapped_out)

        if out_axes is not None:
            out = jax.tree.map(lambda l: transpose(l, out_axes) if _is_named(l) else l, out, is_leaf=_is_named)
        return out

    return cast(F, wrapped)


# --------------------------------------------------------------------------------------------------------------------
# scan / fold
# --------------------------------------------------------------------------------------------------------------------

CheckpointPolicy = Callable[..., bool]
"""A `jax.checkpoint_policies.*` policy (or any callable accepted by `jax.checkpoint(policy=...)`)."""


def _as_policy(remat: bool | CheckpointPolicy) -> CheckpointPolicy | None:
    return remat if callable(remat) else None


def _scan_length(xs, length: int | None) -> int:
    leaves = jax.tree.leaves(xs)
    if length is None:
        if not leaves:
            raise ValueError("checkpointed_scan: `length` is required when `xs` has no leaves")
        length = leaves[0].shape[0]
    for leaf in leaves:
        if leaf.shape[:1] != (length,):
            raise ValueError(f"checkpointed_scan: expected leading dimension {length}, got shape {leaf.shape}")
    return int(length)


def checkpointed_scan(
    f: Callable,
    init: Any,
    xs: Any = None,
    length: int | None = None,
    *,
    block_size: int | Literal["sqrt"] = "sqrt",
    policy: CheckpointPolicy | None = None,
    reverse: bool = False,
    unroll: int = 1,
):
    """`jax.lax.scan` that stores one carry per block of `block_size` steps and recomputes the rest.

    The scan is split into blocks. An outer `lax.scan` runs over the blocks with a `jax.checkpoint`-ed body, and an
    inner `lax.scan` runs over the steps of one block. In reverse mode only the carries at block boundaries are kept;
    each block is recomputed once when its cotangent is needed. With `block_size ~ sqrt(n)` that is O(sqrt(n)) stored
    carries plus O(sqrt(n)) live residuals during the recompute, for about one extra forward pass. If `block_size`
    does not divide `n`, the remaining `n % block_size` steps run in a plain, uncheckpointed scan.

    Args:
        f, init, xs, length, reverse, unroll: as in `jax.lax.scan`. `unroll` applies to the inner scan.
        block_size: steps per checkpointed block, or `"sqrt"` for `isqrt(n)`. A value `>= n` degrades to
            checkpointing every step.
        policy: a `jax.checkpoint_policies` policy for what to save *inside* a block (default: save nothing).
    """
    n = _scan_length(xs, length)
    b = isqrt(n) if block_size == "sqrt" else block_size
    if not isinstance(b, int) or (n > 0 and b < 1):
        raise ValueError(f"checkpointed_scan: block_size must be a positive int or 'sqrt', got {block_size!r}")

    ckpt = partial(jax.checkpoint, policy=policy, prevent_cse=False)  # scan already prevents CSE
    if n == 0 or b >= n:
        return jax.lax.scan(ckpt(f), init, xs, length=n, reverse=reverse, unroll=unroll)

    n_blocks, rem = divmod(n, b)
    main = n_blocks * b
    head = jax.tree.map(lambda x: x[:main].reshape(n_blocks, b, *x.shape[1:]), xs)
    tail = jax.tree.map(lambda x: x[main:], xs)

    @ckpt
    def block(carry, xb):
        return jax.lax.scan(f, carry, xb, length=b, reverse=reverse, unroll=unroll)

    def run_head(carry):
        carry, ys = jax.lax.scan(block, carry, head, length=n_blocks, reverse=reverse)
        return carry, jax.tree.map(lambda y: y.reshape(main, *y.shape[2:]), ys)

    def run_tail(carry):
        return jax.lax.scan(f, carry, tail, length=rem, reverse=reverse, unroll=unroll)

    if rem == 0:
        return run_head(init)
    if reverse:  # the tail holds the *last* steps, which a reverse scan visits first
        carry, ys_tail = run_tail(init)
        carry, ys_head = run_head(carry)
    else:
        carry, ys_head = run_head(init)
        carry, ys_tail = run_tail(carry)
    return carry, jax.tree.map(lambda h, t: jnp.concatenate([h, t]), ys_head, ys_tail)


@partial(jax.tree_util.register_dataclass, data_fields=["array"], meta_fields=["axes"])
@dataclass(frozen=True)
class _Staged:
    """An array plus its axes, as a pytree node that does *not* validate shape against axes.

    `jax.lax.scan` stacks `ys` along a new leading dimension that the body's `NamedArray` axes do not describe. Staged
    values travel through JAX with their axes intact, and get the scanned axis prepended once the scan is done.
    """

    array: Any
    axes: tuple[Axis, ...]


def _is_staged(x) -> bool:
    return isinstance(x, _Staged)


def _stage(leaf) -> _Staged:
    if _is_named(leaf):
        return _Staged(leaf.array, leaf.axes)
    leaf = jnp.asarray(leaf)
    return _Staged(leaf, tuple(Axis(s) for s in leaf.shape))


def scan(
    fn: Callable,
    axis: AxisSelection,
    *,
    remat: bool | CheckpointPolicy = False,
    block_size: int | Literal["sqrt"] | None = None,
    reverse: bool = False,
    unroll: int = 1,
    out_axes: AxisSelection | None = None,
) -> Callable:
    """Scan `fn` over one axis of its `NamedArray` arguments. Curried: `scan(fn, axis)(init, *xs, **kw)`.

    `fn(carry, *x, **kw) -> (carry, y)`, with the scanned axis removed from every `NamedArray` that had it. As in
    `vmap`, arguments without the axis (including plain arrays) are not scanned; they are passed to every step
    unchanged. The carry's pytree structure, including its `axes`, must be the same on input and output.
    Every `NamedArray` (and plain array) in `y` is stacked with the scanned axis first.

    Args:
        fn: step function.
        axis: the axis to scan over, by name, `Axis` or position. Pass an `Axis` (not a name) to scan with no scanned
            inputs, like `lax.scan(..., length=n)`.
        remat: `False`: save everything JAX would save. `True`: `jax.checkpoint` each step, so only the step inputs
            (carry and `x`) are stored. A `jax.checkpoint_policies` policy: `jax.checkpoint` each step with that
            policy. Policies that refer to names can use `jax.ad_checkpoint.checkpoint_name` inside `fn`.
        block_size: if set, checkpoint blocks of this many steps instead of single steps (see `checkpointed_scan`);
            `"sqrt"` picks `isqrt(n)`. Implies checkpointing; `remat` then only chooses the policy used within a block.
        reverse, unroll: as in `jax.lax.scan`.
        out_axes: optional reordering of every `NamedArray` in `ys`, applied with `transpose` after scanning.
    """
    if not callable(fn):
        raise TypeError(f"scan: expected a callable, got {type(fn).__name__}")
    refs = axis_selection_to_tuple(axis)
    if len(refs) != 1:
        raise ValueError(f"scan: expected exactly one axis, got {refs}; nest scans to scan over several axes")
    (ref,) = refs

    @wraps(fn)
    def wrapped(init, *args, **kwargs):
        leaves, treedef = jax.tree.flatten((args, kwargs), is_leaf=_is_named)
        positions = [_resolve_positions(l, refs) if _is_named(l) else None for l in leaves]
        scanned = [i for i, p in enumerate(positions) if p is not None and p[0] is not None]
        if scanned:
            (mapped,) = _unify_mapped_axes(leaves, positions, refs)
        elif isinstance(ref, Axis):
            mapped = ref
        else:
            raise ValueError(f"scan: no argument has axis {ref!r}; pass an `Axis` to scan without inputs")

        xs = []
        for i in scanned:
            position = positions[i]
            assert position is not None and position[0] is not None
            xs.append(jnp.moveaxis(leaves[i].array, position[0], 0))

        def step(carry, x_slices):
            rebuilt = list(leaves)
            for i, x in zip(scanned, x_slices):
                position = positions[i]
                assert position is not None and position[0] is not None
                leaf, p = leaves[i], position[0]
                rebuilt[i] = array(x, leaf.axes[:p] + leaf.axes[p + 1 :])
            a, kw = jax.tree.unflatten(treedef, rebuilt)
            carry, y = fn(carry, *a, **kw)
            return carry, jax.tree.map(_stage, y, is_leaf=_is_named)

        policy = _as_policy(remat)
        if block_size is not None:
            carry, ys = checkpointed_scan(
                step, init, xs, mapped.size, block_size=block_size, policy=policy, reverse=reverse, unroll=unroll
            )
        else:
            body = jax.checkpoint(step, policy=policy, prevent_cse=False) if remat is not False else step
            carry, ys = jax.lax.scan(body, init, xs, length=mapped.size, reverse=reverse, unroll=unroll)

        def finish(s: _Staged):
            axes = (mapped, *s.axes)
            check_unique_axis_names(axes)
            return array(s.array, axes)

        ys = jax.tree.map(finish, ys, is_leaf=_is_staged)
        if out_axes is not None:
            ys = jax.tree.map(lambda l: transpose(l, out_axes) if _is_named(l) else l, ys, is_leaf=_is_named)
        return carry, ys

    return wrapped


def fold(
    fn: Callable,
    axis: AxisSelection,
    *,
    remat: bool | CheckpointPolicy = False,
    block_size: int | Literal["sqrt"] | None = None,
    reverse: bool = False,
    unroll: int = 1,
) -> Callable:
    """Like `scan`, but `fn(carry, *x, **kw) -> carry` and only the final carry is returned (no `ys`)."""

    def step(carry, *args, **kwargs):
        return fn(carry, *args, **kwargs), None

    scanned = scan(step, axis, remat=remat, block_size=block_size, reverse=reverse, unroll=unroll)

    @wraps(fn)
    def wrapped(init, *args, **kwargs):
        return scanned(init, *args, **kwargs)[0]

    return wrapped
