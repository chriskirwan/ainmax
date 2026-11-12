import math
from collections.abc import Sequence
from typing import Literal, NamedTuple, overload

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from ainmax._src.axis import Axis, AxisLike, AxisSelection, AxisSelector
from ainmax._src.axis_util import (
    axis_indices,
    check_unique_axis_names,
    match_axes,
    remove_axes,
)
from ainmax._src.feidhm import constructors, operations, util
from ainmax._src.hof import vmap
from ainmax._src.typing import NamedArray, NamedArrayLike


MatrixSelection = tuple[AxisSelector, AxisSelector]
AxisPair = tuple[AxisLike, AxisLike]
Trans = Literal[0, 1, 2, "N", "T", "C"]


# result types


class EighResult(NamedTuple):
    """Return type of :func:`eigh`."""

    eigenvalues: NamedArray
    eigenvectors: NamedArray


class EigResult(NamedTuple):
    """Return type of :func:`eig`."""

    eigenvalues: NamedArray
    eigenvectors: NamedArray


class SVDResult(NamedTuple):
    """Return type of :func:`svd`."""

    u: NamedArray
    s: NamedArray
    vh: NamedArray


class QRResult(NamedTuple):
    """Return type of :func:`qr`."""

    q: NamedArray
    r: NamedArray


class QRPivotResult(NamedTuple):
    """Return type of :func:`qr` with ``pivoting=True``: ``a[..., p] == q @ r``."""

    q: NamedArray
    r: NamedArray
    p: NamedArray


class RPivotResult(NamedTuple):
    """Return type of :func:`qr` with ``mode="r", pivoting=True``."""

    r: NamedArray
    p: NamedArray


class LUResult(NamedTuple):
    """Return type of :func:`lu_general` (the ``lax.linalg.lu`` triple)."""

    lu: NamedArray
    pivots: NamedArray
    permutation: NamedArray


class PLUResult(NamedTuple):
    """Return type of :func:`lu`: ``a == p @ l @ u``."""

    p: NamedArray
    l: NamedArray  # noqa: E741
    u: NamedArray


class PLUPermutedResult(NamedTuple):
    """Return type of :func:`lu` with ``permute_l=True``: ``a == pl @ u``."""

    pl: NamedArray
    u: NamedArray


class LUFactorResult(NamedTuple):
    """Return type of :func:`lu_factor`; feed it to :func:`lu_solve`."""

    lu: NamedArray
    pivots: NamedArray


class CholFactorResult(NamedTuple):
    """Return type of :func:`cho_factor`; feed it to :func:`cho_solve`."""

    c: NamedArray
    lower: bool


class SLogDetResult(NamedTuple):
    sign: NamedArray
    logabsdet: NamedArray


class LstsqResult(NamedTuple):
    """Return type of :func:`lstsq`."""

    solution: NamedArray
    residuals: NamedArray
    rank: NamedArray
    singular_values: NamedArray


class PolarResult(NamedTuple):
    """Return type of :func:`polar`: ``a == u @ p`` (``side="right"``) or ``a == p @ u`` (``side="left"``)."""

    u: NamedArray
    p: NamedArray


class SchurResult(NamedTuple):
    """Return type of :func:`schur`: ``a == z @ t @ z^H``."""

    t: NamedArray
    z: NamedArray


class HessenbergResult(NamedTuple):
    """Return type of :func:`hessenberg` with ``calc_q=True``: ``a == q @ h @ q^H``."""

    h: NamedArray
    q: NamedArray


class ExpmFrechetResult(NamedTuple):
    """Return type of :func:`expm_frechet` with ``compute_expm=True``."""

    expm: NamedArray
    frechet: NamedArray


# internal helpers


def _namedarray_to_matrix_shape(x: NamedArray, axes: MatrixSelection) -> tuple[NamedArray, tuple[Axis, ...]]:
    """Move the two matrix axes to the trailing positions, keeping batch axes in their original relative order.

    Works on positions rather than `Axis` values, so anonymous axes of equal size are not ambiguous.
    """
    if x.ndim < 2:
        raise ValueError(f"Input array must have at least 2 dimensions, got {x.ndim}")

    old_axes = x.axes
    row_pos, col_pos = axis_indices(x.axes, axes)
    if row_pos == col_pos:
        raise ValueError(f"`matrix_axes` {axes} resolve to the same axis (position {row_pos})")

    batch_pos = [i for i in range(x.ndim) if i not in (row_pos, col_pos)]
    perm = (*batch_pos, row_pos, col_pos)

    if perm == tuple(range(x.ndim)):
        return x, old_axes

    return operations.transpose(x, perm), old_axes


def ensure_matrix_shape(
    fn_name: str, array: NamedArrayLike, matrix_axes: MatrixSelection
) -> tuple[NamedArray, tuple[Axis, ...]]:
    """Return `array` with its matrix axes moved last, plus the original axes."""
    named_array = util.ensure_named(fn_name, array)
    return _namedarray_to_matrix_shape(named_array, matrix_axes)


def canonicalize_matrix_input(
    fn: str, a: NamedArrayLike, matrix_axes: MatrixSelection, *, square: bool = False, inexact: bool = True
) -> tuple[NamedArray, tuple[Axis, ...], Axis, Axis]:
    """Canonicalise a matrix argument: returns ``(a_canonical, batch_axes, row_axis, col_axis)``."""
    if inexact:
        (a,) = util.promote_dtypes_inexact(a)
    named, _ = ensure_matrix_shape(fn, a, matrix_axes)
    *batch, row, col = named.axes
    if square and row.size != col.size:
        raise ValueError(f"`{fn}` requires square matrices, got {row.size} x {col.size}")
    return named, tuple(batch), row, col


def _pair(spec: AxisPair | None, sizes: tuple[int, int], fn: str, param: str) -> tuple[Axis, Axis]:
    if spec is None:
        return Axis(sizes[0]), Axis(sizes[1])
    if len(spec) != 2:
        raise ValueError(f"{fn}: `{param}` must be a pair of axes, got {spec!r}")
    return (
        util.resolve_new_axis(spec[0], sizes[0], fn, param),
        util.resolve_new_axis(spec[1], sizes[1], fn, param),
    )


def check_out_axes(*axes: Sequence[Axis]) -> None:
    for ax in axes:
        check_unique_axis_names(tuple(ax))


def _pairs_from_matrix_axes(a_axes: MatrixSelection, b_axes: MatrixSelection | None) -> MatrixSelection:
    return a_axes if b_axes is None else b_axes


def _align_batch(a: NamedArray, b: NamedArray, k: int):
    """Broadcast the batch axes (all but the last ``k``) of two canonicalised arrays onto each other.

    Returns raw arrays plus the common batch axes.
    """
    a_batch, b_batch = a.axes[:-k], b.axes[:-k]
    _, _, out_batch = util.align_shapes_for_broadcast(a_batch, b_batch)
    a_full = operations.broadcast_to(a, (*out_batch, *a.axes[-k:]))
    b_full = operations.broadcast_to(b, (*out_batch, *b.axes[-k:]))
    return a_full, b_full, out_batch


# ---- right-hand sides (solve & friends) ---------------------------------------------------------------------


class _RHS(NamedTuple):
    array: jax.Array  # (*batch, n, K)
    free_axes: tuple[Axis, ...]


def _prepare_rhs(
    fn: str,
    batch_axes: tuple[Axis, ...],
    n: int,
    match_axis: Axis,
    b: NamedArray,
    b_axis: AxisSelector | None,
) -> _RHS:
    """Bring a right-hand side to ``(*batch_axes, n, K)``.

    ``b_axis`` is the axis of ``b`` that lives in the matrix space being solved against; ``None`` picks the axis of
    ``b`` that has the same name as ``match_axis`` and falls back to the last axis. Axes of ``b`` that match a batch
    axis of the matrix (by name, anonymous ones right-aligned) are batch axes; all remaining axes of ``b`` are free
    right-hand-side axes and are flattened into ``K`` so that the factorisation is shared between them.
    """
    if b.ndim == 0:
        raise ValueError(f"{fn}: right-hand side must have at least one axis")
    if b_axis is None:
        idx = b.ndim - 1
        if match_axis.name is not None:
            idx = next((i for i, ax in enumerate(b.axes) if ax.name == match_axis.name), idx)
    else:
        (idx,) = axis_indices(b.axes, (b_axis,))
    if b.axes[idx].size != n:
        raise ValueError(f"{fn}: right-hand side axis {b.axes[idx]} does not have the matrix size {n}")

    rest_pos = [i for i in range(b.ndim) if i != idx]
    rest_axes = tuple(b.axes[i] for i in rest_pos)
    res = match_axes(batch_axes, rest_axes, allow_positional_fallback=True)
    by_src = {m.source: m.target for m in res.matches}
    for s, t in by_src.items():
        if batch_axes[s].size != rest_axes[t].size:
            raise ValueError(f"{fn}: batch axis size mismatch: matrix {batch_axes[s]} vs rhs {rest_axes[t]}")

    free_t = list(res.unmatched_target)
    matched_order = [by_src[s] for s in sorted(by_src)]
    perm = [rest_pos[t] for t in matched_order] + [idx] + [rest_pos[t] for t in free_t]
    arr = jnp.transpose(b.array, perm)
    k = math.prod(rest_axes[t].size for t in free_t)
    arr = arr.reshape(*(batch_axes[s].size for s in sorted(by_src)), n, k)
    arr = arr.reshape(*(batch_axes[i].size if i in by_src else 1 for i in range(len(batch_axes))), n, k)
    arr = jnp.broadcast_to(arr, (*(ax.size for ax in batch_axes), n, k))
    return _RHS(arr, tuple(rest_axes[t] for t in free_t))


def _finish_rhs(x: jax.Array, batch_axes: tuple[Axis, ...], free_axes: tuple[Axis, ...], x_axis: Axis) -> NamedArray:
    """Inverse of `_prepare_rhs`: ``(*batch, m, K)`` -> axes ``(*batch, *free, x_axis)``."""
    nb, nf = len(batch_axes), len(free_axes)
    x = x.reshape(*x.shape[: nb + 1], *(ax.size for ax in free_axes))
    x = jnp.transpose(x, (*range(nb), *range(nb + 1, nb + 1 + nf), nb))
    out_axes = (*batch_axes, *free_axes, x_axis)
    check_unique_axis_names(out_axes)
    return constructors.array(x, out_axes)


def _trans_flag(trans: Trans) -> int:
    t = {"N": 0, "T": 1, "C": 2}.get(trans, trans) if isinstance(trans, str) else trans
    if t not in (0, 1, 2):
        raise ValueError(f"`trans` must be one of 0, 1, 2, 'N', 'T', 'C', got {trans!r}")
    return t


# eigendecompositions


def eigh_general(
    a: NamedArrayLike,
    eigval_axis: AxisLike | None = None,
    eigvec_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    lower: bool = True,
    symmetrize_input: bool = True,
    sort_eigenvalues: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    implementation: lax.linalg.EighImplementation | None = None,
) -> tuple[NamedArray, NamedArray]:
    """Thin wrapper over `lax.linalg.eigh`. Returns ``(eigenvalues, eigenvectors)``.

    Eigenvectors are the *columns* of the vector matrix: ``eigvec_axes = (row, col)`` with sizes ``(n, k)``.
    ``subset_by_index`` is only implemented on TPU.
    """
    named, batch_axes, row_axis, col_axis = canonicalize_matrix_input(
        "eigh", a, matrix_axes, square=True, inexact=False
    )
    n = row_axis.size

    if subset_by_index is None:
        k = n
    else:
        lo, hi = subset_by_index
        if not 0 <= lo < hi <= n:
            raise ValueError(f"`subset_by_index` must satisfy 0 <= start < stop <= {n}, got {subset_by_index}")
        k = hi - lo

    eigval_ax = util.resolve_new_axis(eigval_axis, k, "eigh", "eigval_axis")
    eigvec_ax = _pair(eigvec_axes, (n, k), "eigh", "eigvec_axes")

    val_axes = (*batch_axes, eigval_ax)
    vec_axes = (*batch_axes, *eigvec_ax)
    check_out_axes(val_axes, vec_axes)

    jax_eigvecs, jax_eigvals = lax.linalg.eigh(
        named.array,
        lower=lower,
        symmetrize_input=symmetrize_input,
        sort_eigenvalues=sort_eigenvalues,
        subset_by_index=subset_by_index,
        implementation=implementation,
    )

    return constructors.array(jax_eigvals, val_axes), constructors.array(jax_eigvecs, vec_axes)


LeftOrRightEigvecAxes = AxisPair | tuple[AxisPair, AxisPair]


def eig_general(
    a: NamedArrayLike,
    eigval_axis: AxisLike | None = None,
    eigvec_axes: LeftOrRightEigvecAxes | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    compute_left_eigenvectors: bool = True,
    compute_right_eigenvectors: bool = True,
    enable_eigvec_derivs: bool = False,
    implementation: lax.linalg.EigImplementation | None = None,
) -> list[NamedArray]:
    """Thin wrapper over `lax.linalg.eig` (CPU only). Returns ``[w, (vl), (vr)]``, omitting what is not requested.

    ``eigvec_axes`` is either a single ``(row, col)`` pair (applied to whichever eigenvectors are computed; if both
    are, they share it) or ``(left_pair, right_pair)``.
    """
    named, batch_axes, row_axis, _ = canonicalize_matrix_input("eig", a, matrix_axes, square=True)
    n = row_axis.size

    eigval_ax = util.resolve_new_axis(eigval_axis, n, "eig", "eigval_axis")

    if eigvec_axes is None:
        left_ax, right_ax = _pair(None, (n, n), "eig", "eigvec_axes"), _pair(None, (n, n), "eig", "eigvec_axes")
    elif len(eigvec_axes) == 2 and isinstance(eigvec_axes[0], (str, Axis)):
        left_ax = right_ax = _pair(eigvec_axes, (n, n), "eig", "eigvec_axes")  # type: ignore[arg-type]
    elif len(eigvec_axes) == 2:
        left_ax = _pair(eigvec_axes[0], (n, n), "eig", "eigvec_axes[0]")  # type: ignore[arg-type]
        right_ax = _pair(eigvec_axes[1], (n, n), "eig", "eigvec_axes[1]")  # type: ignore[arg-type]
    else:
        raise ValueError(
            f"`eigvec_axes` must be a (row, col) pair or a (left, right) pair of pairs, got {eigvec_axes!r}"
        )

    val_axes = (*batch_axes, eigval_ax)
    check_out_axes(val_axes)
    if compute_left_eigenvectors:
        check_out_axes((*batch_axes, *left_ax))
    if compute_right_eigenvectors:
        check_out_axes((*batch_axes, *right_ax))

    # `lax.linalg.eig` returns [w, (vl), (vr)] -- only what was requested.
    it = iter(
        lax.linalg.eig(
            named.array,
            compute_left_eigenvectors=compute_left_eigenvectors,
            compute_right_eigenvectors=compute_right_eigenvectors,
            enable_eigvec_derivs=enable_eigvec_derivs,
            implementation=implementation,
        )
    )
    out = [constructors.array(next(it), val_axes)]
    if compute_left_eigenvectors:
        out.append(constructors.array(next(it), (*batch_axes, *left_ax)))
    if compute_right_eigenvectors:
        out.append(constructors.array(next(it), (*batch_axes, *right_ax)))
    return out


def _uplo_to_lower(UPLO: str | None) -> bool:
    if UPLO is None:
        return True
    if UPLO.upper() not in ("L", "U"):
        raise ValueError(f"`UPLO` must be 'L' or 'U', got {UPLO!r}")
    return UPLO.upper() == "L"


def eigh(
    a: NamedArrayLike,
    eigval_axis: AxisLike | None = None,
    eigvec_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    UPLO: str | None = None,
    symmetrize_input: bool = True,
) -> EighResult:
    """Eigendecomposition of a Hermitian matrix (see `jax.numpy.linalg.eigh`); eigenvalues ascending."""
    (a,) = util.promote_dtypes_inexact(a)
    w, v = eigh_general(
        a,
        eigval_axis=eigval_axis,
        eigvec_axes=eigvec_axes,
        matrix_axes=matrix_axes,
        lower=_uplo_to_lower(UPLO),
        symmetrize_input=symmetrize_input,
    )
    return EighResult(w, v)


def eigvalsh(
    a: NamedArrayLike,
    eigval_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    UPLO: str | None = None,
    symmetrize_input: bool = True,
) -> NamedArray:
    """Eigenvalues of a Hermitian matrix, ascending."""
    (a,) = util.promote_dtypes_inexact(a)
    w, _ = eigh_general(
        a,
        eigval_axis=eigval_axis,
        matrix_axes=matrix_axes,
        lower=_uplo_to_lower(UPLO),
        symmetrize_input=symmetrize_input,
    )
    return w


def eig(
    a: NamedArrayLike,
    eigval_axis: AxisLike | None = None,
    eigvec_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
) -> EigResult:
    """Eigendecomposition of a general square matrix (right eigenvectors only). CPU only, like `jnp.linalg.eig`."""
    (a,) = util.promote_dtypes_inexact(a)
    w, v = eig_general(a, eigval_axis, eigvec_axes, matrix_axes, compute_left_eigenvectors=False)
    return EigResult(w, v)


def eigvals(
    a: NamedArrayLike,
    eigval_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
) -> NamedArray:
    """Eigenvalues of a general square matrix. CPU only, like `jnp.linalg.eigvals`."""
    (a,) = util.promote_dtypes_inexact(a)
    (w,) = eig_general(
        a, eigval_axis, matrix_axes=matrix_axes, compute_right_eigenvectors=False, compute_left_eigenvectors=False
    )
    return w


# SVD


@overload
def svd(
    a: NamedArrayLike,
    s_axis: AxisLike | None = None,
    u_axes: AxisPair | None = None,
    vh_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    compute_uv: Literal[False],
    full_matrices: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> NamedArray: ...


@overload
def svd(
    a: NamedArrayLike,
    s_axis: AxisLike | None = None,
    u_axes: AxisPair | None = None,
    vh_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    compute_uv: Literal[True] = True,
    full_matrices: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> SVDResult: ...


def svd(
    a: NamedArrayLike,
    s_axis: AxisLike | None = None,
    u_axes: AxisPair | None = None,
    vh_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    compute_uv: bool = True,
    full_matrices: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> NamedArray | SVDResult:
    """Singular value decomposition ``a = u @ diag(s) @ vh`` (see `jax.numpy.linalg.svd`).

    Shapes for an ``n x m`` matrix with ``k = min(n, m)``: ``s: (k,)``, ``u: (n, n | k)``, ``vh: (k | m, m)``
    (the larger variant if ``full_matrices``). ``subset_by_index`` (TPU only) requires ``full_matrices=False``.
    """
    if not compute_uv and (u_axes is not None or vh_axes is not None):
        raise ValueError("svd: `u_axes` / `vh_axes` were given but `compute_uv=False`")
    if subset_by_index is not None and full_matrices:
        raise ValueError("svd: `subset_by_index` requires `full_matrices=False`")

    a, batch_axes, row_axis, col_axis = canonicalize_matrix_input("svd", a, matrix_axes)
    n, m = row_axis.size, col_axis.size
    if hermitian and n != m:
        raise ValueError(f"svd: `hermitian=True` requires a square matrix, got {n} x {m}")

    k = min(n, m)
    if subset_by_index is not None:
        lo, hi = subset_by_index
        if not 0 <= lo < hi <= k:
            raise ValueError(f"`subset_by_index` must satisfy 0 <= start < stop <= {k}, got {subset_by_index}")
        k = hi - lo

    s_ax = util.resolve_new_axis(s_axis, k, "svd", "s_axis")
    s_axes = (*batch_axes, s_ax)

    if not compute_uv:
        check_out_axes(s_axes)
        jax_s = jnp.linalg.svd(
            a.array, full_matrices, compute_uv=False, hermitian=hermitian, subset_by_index=subset_by_index
        )
        return constructors.array(jax_s, s_axes)

    u_ax = _pair(u_axes, (n, n if full_matrices else k), "svd", "u_axes")
    vh_ax = _pair(vh_axes, (m if full_matrices else k, m), "svd", "vh_axes")
    check_out_axes(s_axes, (*batch_axes, *u_ax), (*batch_axes, *vh_ax))

    jax_u, jax_s, jax_vh = jnp.linalg.svd(
        a.array, full_matrices, compute_uv=True, hermitian=hermitian, subset_by_index=subset_by_index
    )
    return SVDResult(
        constructors.array(jax_u, (*batch_axes, *u_ax)),
        constructors.array(jax_s, s_axes),
        constructors.array(jax_vh, (*batch_axes, *vh_ax)),
    )


def svdvals(
    a: NamedArrayLike,
    s_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
) -> NamedArray:
    """Singular values of ``a``, descending (see `jax.numpy.linalg.svdvals`)."""
    return svd(a, s_axis, matrix_axes=matrix_axes, compute_uv=False)


# QR


QRMode = Literal["reduced", "complete", "r"]


@overload
def qr(
    a: NamedArrayLike,
    q_axes: None = None,
    r_axes: AxisPair | None = None,
    p_axis: None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    mode: Literal["r"],
    pivoting: Literal[False] = False,
) -> NamedArray: ...


@overload
def qr(
    a: NamedArrayLike,
    q_axes: None = None,
    r_axes: AxisPair | None = None,
    p_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    mode: Literal["r"],
    pivoting: Literal[True],
) -> RPivotResult: ...


@overload
def qr(
    a: NamedArrayLike,
    q_axes: AxisPair | None = None,
    r_axes: AxisPair | None = None,
    p_axis: None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    mode: Literal["reduced", "complete"] = "reduced",
    pivoting: Literal[False] = False,
) -> QRResult: ...


@overload
def qr(
    a: NamedArrayLike,
    q_axes: AxisPair | None = None,
    r_axes: AxisPair | None = None,
    p_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    mode: Literal["reduced", "complete"] = "reduced",
    pivoting: Literal[True],
) -> QRPivotResult: ...


@overload
def qr(
    a: NamedArrayLike,
    q_axes: AxisPair | None = None,
    r_axes: AxisPair | None = None,
    p_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    mode: QRMode = "reduced",
    pivoting: bool = False,
) -> NamedArray | RPivotResult | QRResult | QRPivotResult: ...


def qr(
    a: NamedArrayLike,
    q_axes: AxisPair | None = None,
    r_axes: AxisPair | None = None,
    p_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    mode: QRMode = "reduced",
    pivoting: bool = False,
) -> NamedArray | RPivotResult | QRResult | QRPivotResult:
    """QR decomposition ``a = q @ r`` (see `jax.numpy.linalg.qr`, `jax.scipy.linalg.qr`).

    For an ``n x m`` matrix with ``k = min(n, m)``:

    * ``mode="reduced"``: ``q: (n, k)``, ``r: (k, m)``
    * ``mode="complete"``: ``q: (n, n)``, ``r: (n, m)``
    * ``mode="r"``: only ``r: (k, m)``

    With ``pivoting=True`` a column permutation ``p: (m,)`` is also returned, such that ``a[..., :, p] == q @ r``
    (column-pivoted QR; CPU/GPU only). Note that LAPACK does not fix the signs of ``diag(r)``.

    Args:
        a: Input array.
        q_axes: ``(row, col)`` axes of ``q`` (``None`` -> anonymous). Must be ``None`` for ``mode="r"``.
        r_axes: ``(row, col)`` axes of ``r``.
        p_axis: Axis of the permutation (only with ``pivoting=True``).
        matrix_axes: The two matrix axes of ``a``; all other axes are batch axes.
        mode: ``"reduced"`` (default), ``"complete"`` or ``"r"``.
        pivoting: Whether to use column pivoting.
    """
    if mode not in ("reduced", "complete", "r"):
        raise ValueError(f"qr: `mode` must be 'reduced', 'complete' or 'r', got {mode!r}")
    if mode == "r" and q_axes is not None:
        raise ValueError("qr: `q_axes` was given but `mode='r'` does not compute q")
    if not pivoting and p_axis is not None:
        raise ValueError("qr: `p_axis` was given but `pivoting=False`")

    a, batch_axes, row_axis, col_axis = canonicalize_matrix_input("qr", a, matrix_axes)
    n, m = row_axis.size, col_axis.size
    k = min(n, m)
    full = mode == "complete"

    r_ax = _pair(r_axes, (n if full else k, m), "qr", "r_axes")
    r_out = (*batch_axes, *r_ax)
    p_out = (*batch_axes, util.resolve_new_axis(p_axis, m, "qr", "p_axis")) if pivoting else None
    q_out = None
    if mode != "r":
        q_out = (*batch_axes, *_pair(q_axes, (n, n if full else k), "qr", "q_axes"))
    check_out_axes(r_out, *(ax for ax in (q_out, p_out) if ax is not None))

    if pivoting:
        jax_q, jax_r, jax_p = lax.linalg.qr(a.array, pivoting=True, full_matrices=full)
    else:
        jax_q, jax_r = lax.linalg.qr(a.array, pivoting=False, full_matrices=full)
        jax_p = None
    r = constructors.array(jax_r, r_out)
    p = constructors.array(jax_p, p_out) if p_out is not None and jax_p is not None else None

    if q_out is None:
        return RPivotResult(r, p) if p is not None else r  # type: ignore[arg-type]
    q = constructors.array(jax_q, q_out)
    return QRPivotResult(q, r, p) if p is not None else QRResult(q, r)


# Cholesky / LU


def cholesky(
    a: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    upper: bool = False,
    symmetrize_input: bool = True,
) -> NamedArray:
    """Cholesky factor; result keeps the (row, col) axes of ``a`` in canonical layout."""
    a, _, _, _ = canonicalize_matrix_input("cholesky", a, matrix_axes, square=True)
    jax_array = jnp.linalg.cholesky(a.array, upper=upper, symmetrize_input=symmetrize_input)
    return constructors.array(jax_array, a.axes)


def cho_factor(
    a: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    lower: bool = False,
) -> CholFactorResult:
    """Cholesky factorisation for use with :func:`cho_solve` (see `jax.scipy.linalg.cho_factor`)."""
    a, _, _, _ = canonicalize_matrix_input("cho_factor", a, matrix_axes, square=True)
    c, low = jsl.cho_factor(a.array, lower=lower)
    return CholFactorResult(constructors.array(c, a.axes), low)


def cho_solve(
    c_and_lower: tuple[NamedArrayLike, bool],
    b: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    b_axis: AxisSelector | None = None,
) -> NamedArray:
    """Solve ``A x = b`` from a Cholesky factor. See :func:`solve` for the axis conventions of ``b``."""
    c, lower = c_and_lower
    c, b = util.promote_dtypes_inexact(c, b)
    c, batch, row, col = canonicalize_matrix_input("cho_solve", c, matrix_axes, square=True, inexact=False)
    rhs = _prepare_rhs("cho_solve", batch, row.size, row, b, b_axis)
    x = jsl.cho_solve((c.array, lower), rhs.array)
    return _finish_rhs(x, batch, rhs.free_axes, Axis(col.size, col.name))


def lu_general(
    a: NamedArrayLike,
    pivot_axis: AxisLike | None = None,
    permutation_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
) -> LUResult:
    """Thin wrapper over `lax.linalg.lu`: packed LU (axes of ``a``), ``pivots: (k,)`` and ``permutation: (n,)``."""
    a, batch_axes, row_axis, col_axis = canonicalize_matrix_input("lu", a, matrix_axes)
    n, m = row_axis.size, col_axis.size
    piv_ax = util.resolve_new_axis(pivot_axis, min(n, m), "lu", "pivot_axis")
    perm_ax = util.resolve_new_axis(permutation_axis, n, "lu", "permutation_axis")
    check_out_axes((*batch_axes, piv_ax), (*batch_axes, perm_ax))

    lu_arr, pivots, perm = lax.linalg.lu(a.array)
    return LUResult(
        constructors.array(lu_arr, a.axes),
        constructors.array(pivots, (*batch_axes, piv_ax)),
        constructors.array(perm, (*batch_axes, perm_ax)),
    )


@overload
def lu(
    a: NamedArrayLike,
    p_axes: AxisPair | None = None,
    l_axes: AxisPair | None = None,
    u_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    permute_l: Literal[False] = False,
) -> PLUResult: ...


@overload
def lu(
    a: NamedArrayLike,
    p_axes: None = None,
    l_axes: AxisPair | None = None,
    u_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    permute_l: Literal[True],
) -> PLUPermutedResult: ...


def lu(
    a: NamedArrayLike,
    p_axes: AxisPair | None = None,
    l_axes: AxisPair | None = None,
    u_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    permute_l: bool = False,
) -> PLUResult | PLUPermutedResult:
    """LU decomposition with partial pivoting ``a = p @ l @ u`` (see `jax.scipy.linalg.lu`).

    For an ``n x m`` matrix with ``k = min(n, m)``: ``p: (n, n)``, ``l: (n, k)``, ``u: (k, m)``.
    With ``permute_l=True`` returns ``(p @ l, u)`` instead.
    """
    if permute_l and p_axes is not None:
        raise ValueError("lu: `p_axes` was given but `permute_l=True` does not return p")
    a, batch_axes, row_axis, col_axis = canonicalize_matrix_input("lu", a, matrix_axes)
    n, m = row_axis.size, col_axis.size
    k = min(n, m)

    l_ax = (*batch_axes, *_pair(l_axes, (n, k), "lu", "l_axes"))
    u_ax = (*batch_axes, *_pair(u_axes, (k, m), "lu", "u_axes"))
    check_out_axes(l_ax, u_ax)

    if permute_l:
        pl, u = jsl.lu(a.array, permute_l=True)
        return PLUPermutedResult(constructors.array(pl, l_ax), constructors.array(u, u_ax))

    p_ax = (*batch_axes, *_pair(p_axes, (n, n), "lu", "p_axes"))
    check_out_axes(p_ax)
    p, l, u = jsl.lu(a.array)  # noqa: E741
    return PLUResult(constructors.array(p, p_ax), constructors.array(l, l_ax), constructors.array(u, u_ax))


def lu_factor(
    a: NamedArrayLike,
    pivot_axis: AxisLike | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
) -> LUFactorResult:
    """Packed LU factorisation for use with :func:`lu_solve` (see `jax.scipy.linalg.lu_factor`)."""
    lu_arr, pivots, _ = lu_general(a, pivot_axis, matrix_axes=matrix_axes)
    return LUFactorResult(lu_arr, pivots)


def lu_solve(
    lu_and_piv: tuple[NamedArray, NamedArray],
    b: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    b_axis: AxisSelector | None = None,
    *,
    trans: Trans = 0,
) -> NamedArray:
    """Solve ``A x = b`` (``trans=0``), ``A^T x = b`` (1) or ``A^H x = b`` (2) from :func:`lu_factor`.

    ``pivots`` must carry the batch axes of ``lu`` followed by the pivot axis (as returned by `lu_factor`).
    See :func:`solve` for the axis conventions of ``b``.
    """
    lu_arr, piv = lu_and_piv
    t = _trans_flag(trans)
    lu_arr, b = util.promote_dtypes_inexact(lu_arr, b)
    lu_c, batch, row, col = canonicalize_matrix_input("lu_solve", lu_arr, matrix_axes, square=True, inexact=False)
    piv = operations.broadcast_to(util.ensure_named("lu_solve", piv), (*batch, piv.axes[-1]))
    # x lives in the column space for A x = b and in the row space for A^T x = b.
    match, x_ax = (row, col) if t == 0 else (col, row)
    rhs = _prepare_rhs("lu_solve", batch, match.size, match, b, b_axis)
    x = jsl.lu_solve((lu_c.array, piv.array), rhs.array, trans=t)
    return _finish_rhs(x, batch, rhs.free_axes, Axis(x_ax.size, x_ax.name))


# determinants, traces, norms, conditioning


def det(a: NamedArrayLike, matrix_axes: MatrixSelection = (-2, -1)) -> NamedArray:
    a, batch_axes, _, _ = canonicalize_matrix_input("det", a, matrix_axes, square=True)
    return constructors.array(jnp.linalg.det(a.array), batch_axes)


def slogdet(
    a: NamedArrayLike, matrix_axes: MatrixSelection = (-2, -1), *, method: Literal["lu", "qr"] | None = None
) -> SLogDetResult:
    """Sign and log|det| (see `jax.numpy.linalg.slogdet`)."""
    a, batch_axes, _, _ = canonicalize_matrix_input("slogdet", a, matrix_axes, square=True)
    sign, logabsdet = jnp.linalg.slogdet(a.array, method=method)
    return SLogDetResult(constructors.array(sign, batch_axes), constructors.array(logabsdet, batch_axes))


def _keepdims_axes(axes: tuple[Axis, ...], idxs: Sequence[int]) -> tuple[Axis, ...]:
    return tuple(Axis(1, ax.name) if i in idxs else ax for i, ax in enumerate(axes))


def matrix_rank(
    a: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    rtol: NamedArrayLike | None = None,
    *,
    hermitian: bool = False,
    tol: NamedArrayLike | None = None,
) -> NamedArray:
    """Numerical rank. ``rtol`` / ``tol`` may be scalars or arrays over (a subset of) the batch axes."""
    a, batch_axes, _, _ = canonicalize_matrix_input("matrix_rank", a, matrix_axes)

    def _tol(x: NamedArrayLike | None):
        return None if x is None else operations.broadcast_to(util.ensure_named("matrix_rank", x), batch_axes).array

    jax_array = jnp.linalg.matrix_rank(a.array, rtol=_tol(rtol), hermitian=hermitian, tol=_tol(tol))
    return constructors.array(jax_array, batch_axes)


def matrix_norm(
    a: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    keepdims: bool = False,
    ord: str | int | float = "fro",
) -> NamedArray:
    a, batch_axes, row, col = canonicalize_matrix_input("matrix_norm", a, matrix_axes, inexact=False)
    jax_norm = jnp.linalg.matrix_norm(a.array, keepdims=keepdims, ord=ord)  # type: ignore[arg-type]
    axes = _keepdims_axes(a.axes, (a.ndim - 2, a.ndim - 1)) if keepdims else batch_axes
    return constructors.array(jax_norm, axes)


def vector_norm(
    a: NamedArrayLike,
    vector_axis: AxisSelection | None = -1,
    *,
    keepdims: bool = False,
    ord: str | int | float = 2,
) -> NamedArray:
    """Vector norm over ``vector_axis`` (one or several axes; ``None`` = all axes)."""
    a = util.ensure_named("vector_norm", a)
    idx = None if vector_axis is None else axis_indices(a.axes, vector_axis)
    jax_norm = jnp.linalg.vector_norm(a.array, axis=idx, keepdims=keepdims, ord=ord)
    red = tuple(range(a.ndim)) if idx is None else idx
    out_axes = _keepdims_axes(a.axes, red) if keepdims else remove_axes(a.axes, red)
    return constructors.array(jax_norm, out_axes)


def norm(
    a: NamedArrayLike,
    axis: AxisSelection | None = None,
    ord: str | int | float | None = None,
    *,
    keepdims: bool = False,
) -> NamedArray:
    """`jax.numpy.linalg.norm`: one axis -> vector norm, two axes -> matrix norm (first = row, second = col).

    ``axis=None`` follows NumPy: all axes of a 1-D / 2-D array (the flattened 2-norm if ``ord is None``).
    """
    a = util.ensure_named("norm", a)
    idx = None if axis is None else axis_indices(a.axes, axis)
    if idx is not None and len(idx) not in (1, 2):
        raise ValueError(f"norm: `axis` must select one or two axes, got {axis!r}")
    jax_norm = jnp.linalg.norm(
        a.array, ord=ord, axis=idx[0] if idx is not None and len(idx) == 1 else idx, keepdims=keepdims
    )
    red = tuple(range(a.ndim)) if idx is None else idx
    out_axes = _keepdims_axes(a.axes, red) if keepdims else remove_axes(a.axes, red)
    return constructors.array(jax_norm, out_axes)


def cond(a: NamedArrayLike, p: str | int | float | None = None, matrix_axes: MatrixSelection = (-2, -1)) -> NamedArray:
    a, batch_axes, _, _ = canonicalize_matrix_input("cond", a, matrix_axes)
    return constructors.array(jnp.linalg.cond(a.array, p=p), batch_axes)


# inverses and linear solves


def inv(a: NamedArrayLike, matrix_axes: MatrixSelection = (-2, -1)) -> NamedArray:
    """Matrix inverse; keeps the (row, col) roles of ``a``."""
    a, _, _, _ = canonicalize_matrix_input("inv", a, matrix_axes, square=True)
    return constructors.array(jnp.linalg.inv(a.array), a.axes)


def matrix_power(a: NamedArrayLike, n: int, *, matrix_axes: MatrixSelection = (-2, -1)) -> NamedArray:
    a, _, _, _ = canonicalize_matrix_input("matrix_power", a, matrix_axes, square=True)
    return constructors.array(jnp.linalg.matrix_power(a.array, n), a.axes)


def _role_transposed(row: Axis, col: Axis) -> tuple[Axis, Axis]:
    """Axes of the transpose of an ``row x col`` matrix: roles (names) stay in place, sizes swap."""
    return Axis(col.size, row.name), Axis(row.size, col.name)


def pinv(
    a: NamedArrayLike,
    rtol: NamedArrayLike | None = None,
    hermitian: bool = False,
    matrix_axes: MatrixSelection = (-2, -1),
) -> NamedArray:
    """Moore-Penrose pseudo-inverse. For an ``n x m`` matrix the result is ``m x n`` (roles keep their names)."""
    a, batch_axes, row, col = canonicalize_matrix_input("pinv", a, matrix_axes)
    jax_rtol = None
    if rtol is not None:
        # rtol is batch shaped (or broadcastable to the batch shape) in jax
        jax_rtol = operations.broadcast_to(util.ensure_named("pinv", rtol), batch_axes).array
    jax_pinv = jnp.linalg.pinv(a.array, jax_rtol, hermitian)
    return constructors.array(jax_pinv, (*batch_axes, *_role_transposed(row, col)))


def solve(
    a: NamedArrayLike,
    b: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    b_axis: AxisSelector | None = None,
    *,
    assume_a: Literal["gen", "sym", "her", "pos"] = "gen",
    lower: bool = False,
) -> NamedArray:
    """Solve ``a x = b`` for square ``a`` (see `jax.numpy.linalg.solve`, `jax.scipy.linalg.solve`).

    ``b_axis`` is the axis of ``b`` in the *row* space of ``a`` (``None``: the axis of ``b`` named like the row axis
    of ``a``, else the last one). Other axes of ``b`` that match a batch axis of ``a`` (by name, anonymous ones
    right-aligned) are batch axes; all the rest are free right-hand-side axes that share one factorisation.
    The result has axes ``(*batch, *free, col_axis)``.
    """
    a, b = util.promote_dtypes_inexact(a, b)
    a, batch, row, col = canonicalize_matrix_input("solve", a, matrix_axes, square=True, inexact=False)
    rhs = _prepare_rhs("solve", batch, row.size, row, b, b_axis)
    x = jsl.solve(a.array, rhs.array, lower=lower, assume_a=assume_a)
    return _finish_rhs(x, batch, rhs.free_axes, Axis(col.size, col.name))


def solve_triangular(
    a: NamedArrayLike,
    b: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    b_axis: AxisSelector | None = None,
    *,
    trans: Trans = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> NamedArray:
    """Solve ``op(a) x = b`` for triangular ``a`` (see `jax.scipy.linalg.solve_triangular`).

    With ``trans=0`` ``b`` lives in the row space and ``x`` in the column space; for ``trans`` 1 / 2 (``a^T`` /
    ``a^H``) the roles are exchanged. See :func:`solve` for the remaining axis conventions.
    """
    t = _trans_flag(trans)
    a, b = util.promote_dtypes_inexact(a, b)
    a, batch, row, col = canonicalize_matrix_input("solve_triangular", a, matrix_axes, square=True, inexact=False)
    match, x_ax = (row, col) if t == 0 else (col, row)
    rhs = _prepare_rhs("solve_triangular", batch, match.size, match, b, b_axis)
    x = jsl.solve_triangular(a.array, rhs.array, trans=t, lower=lower, unit_diagonal=unit_diagonal)
    return _finish_rhs(x, batch, rhs.free_axes, Axis(x_ax.size, x_ax.name))


def _rhs_axis_index(fn: str, b: NamedArray, b_axis: AxisSelector | None, match_axis: Axis) -> int:
    """Position of the axis of ``b`` that lives in the matrix space being solved against.

    ``None`` picks the axis named like ``match_axis`` and falls back to the last axis.
    """
    if b.ndim == 0:
        raise ValueError(f"{fn}: right-hand side must have at least one axis")
    if b_axis is not None:
        (idx,) = axis_indices(b.axes, (b_axis,))
        return idx
    if match_axis.name is not None:
        return next((i for i, ax in enumerate(b.axes) if ax.name == match_axis.name), b.ndim - 1)
    return b.ndim - 1


def lstsq(
    a: NamedArrayLike,
    b: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    b_axis: AxisSelector | None = None,
    s_axis: AxisLike | None = None,
    *,
    rcond: float | None = None,
) -> LstsqResult:
    """Least squares ``min_x |a x - b|`` (see `jax.numpy.linalg.lstsq`), batched over the batch axes of ``a``.

    ``b_axis`` is the axis of ``b`` in the row space of ``a`` (``None``: the axis named like the row axis of ``a``,
    else the last one). Axes of ``b`` that are also batch axes of ``a`` are mapped over together with ``a`` (``b``
    is broadcast over batch axes it lacks); all remaining axes of ``b`` are free right-hand-side axes sharing one
    factorisation.

    Returns the solution (axes ``(*batch, *free, col_axis)``), the squared residuals ``(*batch, *free)`` (always
    computed), the rank ``(*batch,)`` and the singular values ``(*batch, s_axis)``.
    """
    a, b = util.promote_dtypes_inexact(a, b)
    a, batch_axes, row, col = canonicalize_matrix_input("lstsq", a, matrix_axes, inexact=False)
    # Resolve against the full `b` now: inside `vmap` the mapped axes are gone and positions shift.
    b_ax = b.axes[_rhs_axis_index("lstsq", b, b_axis, row)]
    s_ax = util.resolve_new_axis(s_axis, min(row.size, col.size), "lstsq", "s_axis")
    x_ax = Axis(col.size, col.name)

    def _lstsq(a_i: NamedArray, b_i: NamedArray) -> LstsqResult:
        rhs = _prepare_rhs("lstsq", (), row.size, row, b_i, b_ax)
        x, res, rank, s = jnp.linalg.lstsq(a_i.array, rhs.array, rcond=rcond)
        free_sizes = tuple(ax.size for ax in rhs.free_axes)
        return LstsqResult(
            _finish_rhs(x, (), rhs.free_axes, x_ax),
            constructors.array(res.reshape(free_sizes), rhs.free_axes),
            constructors.array(rank, ()),
            constructors.array(s, (s_ax,)),
        )

    return vmap(_lstsq, batch_axes)(a, b) if batch_axes else _lstsq(a, b)


# matrix functions & other factorisations


def expm(
    a: NamedArrayLike,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    upper_triangular: bool = False,
    max_squarings: int = 16,
) -> NamedArray:
    """Matrix exponential (see `jax.scipy.linalg.expm`); keeps the (row, col) roles of ``a``."""
    a, _, _, _ = canonicalize_matrix_input("expm", a, matrix_axes, square=True)
    return constructors.array(
        jsl.expm(a.array, upper_triangular=upper_triangular, max_squarings=max_squarings), a.axes
    )


matrix_exp = expm


@overload
def expm_frechet(
    a: NamedArrayLike,
    e: NamedArrayLike,
    a_matrix_axes: MatrixSelection = (-2, -1),
    e_matrix_axes: MatrixSelection | None = None,
    *,
    compute_expm: Literal[True] = True,
) -> ExpmFrechetResult: ...


@overload
def expm_frechet(
    a: NamedArrayLike,
    e: NamedArrayLike,
    a_matrix_axes: MatrixSelection = (-2, -1),
    e_matrix_axes: MatrixSelection | None = None,
    *,
    compute_expm: Literal[False],
) -> NamedArray: ...


def expm_frechet(
    a: NamedArrayLike,
    e: NamedArrayLike,
    a_matrix_axes: MatrixSelection = (-2, -1),
    e_matrix_axes: MatrixSelection | None = None,
    *,
    compute_expm: bool = True,
) -> ExpmFrechetResult | NamedArray:
    """Fréchet derivative of ``expm`` at ``a`` in direction ``e`` (see `jax.scipy.linalg.expm_frechet`, batched via
    ``vmap``).

    Batch axes of ``a`` and ``e`` are broadcast against each other. Axes of the result are those of ``a``.
    """
    a, e = util.promote_dtypes_inexact(a, e)
    a_c, _, row, col = canonicalize_matrix_input("expm_frechet", a, a_matrix_axes, square=True, inexact=False)

    e_c, _, e_row, e_col = canonicalize_matrix_input(
        "expm_frechet", e, _pairs_from_matrix_axes(a_matrix_axes, e_matrix_axes), square=True, inexact=False
    )
    if e_row.size != row.size:
        raise ValueError(f"expm_frechet: size mismatch between a ({row.size}) and e ({e_row.size})")
    a_arr, e_arr, batch = _align_batch(a_c, e_c, 2)

    def _expm_frechet(a_i, b_i):
        jax_res = jsl.expm_frechet(a_i.array, b_i.array, compute_expm=compute_expm)
        if compute_expm:
            return ExpmFrechetResult(
                constructors.array(jax_res[0], (row, col)), constructors.array(jax_res[1], (row, col))
            )
        return constructors.array(jax_res, (row, col))

    return vmap(_expm_frechet, batch)(a_arr, e_arr) if batch else _expm_frechet(a_arr, e_arr)


def sqrtm(a: NamedArrayLike, matrix_axes: MatrixSelection = (-2, -1), *, blocksize: int = 1) -> NamedArray:
    """Matrix square root (see `jax.scipy.linalg.sqrtm`, CPU only); batched via ``vmap``."""
    a, batch_axes, _, _ = canonicalize_matrix_input("sqrtm", a, matrix_axes, square=True)

    def _sqrtm(x: NamedArray) -> NamedArray:
        return constructors.array(jsl.sqrtm(x.array, blocksize=blocksize), x.axes)

    return vmap(_sqrtm, batch_axes)(a) if batch_axes else _sqrtm(a)


def polar(
    a: NamedArrayLike,
    p_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    side: Literal["right", "left"] = "right",
    method: str = "qdwh",
    eps: float | None = None,
    max_iterations: int | None = None,
) -> PolarResult:
    """Polar decomposition ``a = u p`` (``side="right"``) or ``a = p u`` (``"left"``).

    ``u`` has the axes of ``a`` (and is the closest unitary / isometry to ``a``), ``p`` is positive semi-definite
    with axes ``p_axes`` of size ``m x m`` (right) or ``n x n`` (left); default anonymous.
    """
    a, batch_axes, row, col = canonicalize_matrix_input("polar", a, matrix_axes)
    size = col.size if side == "right" else row.size
    p_ax = (*batch_axes, *_pair(p_axes, (size, size), "polar", "p_axes"))
    check_out_axes(p_ax)
    u, p = jsl.polar(a.array, side=side, method=method, eps=eps, max_iterations=max_iterations)
    return PolarResult(constructors.array(u, a.axes), constructors.array(p, p_ax))


def schur(
    a: NamedArrayLike,
    t_axes: AxisPair | None = None,
    z_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    output: Literal["real", "complex"] = "real",
) -> SchurResult:
    """Schur decomposition ``a = z t z^H`` (see `jax.scipy.linalg.schur`, CPU only)."""
    a, batch_axes, row, _ = canonicalize_matrix_input("schur", a, matrix_axes, square=True)
    n = row.size
    t_ax = (*batch_axes, *_pair(t_axes, (n, n), "schur", "t_axes"))
    z_ax = (*batch_axes, *_pair(z_axes, (n, n), "schur", "z_axes"))
    check_out_axes(t_ax, z_ax)
    t, z = jsl.schur(a.array, output=output)
    return SchurResult(constructors.array(t, t_ax), constructors.array(z, z_ax))


@overload
def hessenberg(
    a: NamedArrayLike,
    h_axes: AxisPair | None = None,
    q_axes: None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    calc_q: Literal[False] = False,
) -> NamedArray: ...


@overload
def hessenberg(
    a: NamedArrayLike,
    h_axes: AxisPair | None = None,
    q_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    calc_q: Literal[True],
) -> HessenbergResult: ...


def hessenberg(
    a: NamedArrayLike,
    h_axes: AxisPair | None = None,
    q_axes: AxisPair | None = None,
    matrix_axes: MatrixSelection = (-2, -1),
    *,
    calc_q: bool = False,
) -> NamedArray | HessenbergResult:
    """Upper Hessenberg form ``a = q h q^H`` (see `jax.scipy.linalg.hessenberg`, CPU only)."""
    if not calc_q and q_axes is not None:
        raise ValueError("hessenberg: `q_axes` was given but `calc_q=False`")
    a, batch_axes, row, _ = canonicalize_matrix_input("hessenberg", a, matrix_axes, square=True)
    n = row.size
    h_ax = (*batch_axes, *_pair(h_axes, (n, n), "hessenberg", "h_axes"))
    check_out_axes(h_ax)
    if not calc_q:
        return constructors.array(jsl.hessenberg(a.array, calc_q=False), h_ax)
    q_ax = (*batch_axes, *_pair(q_axes, (n, n), "hessenberg", "q_axes"))
    check_out_axes(q_ax)
    h, q = jsl.hessenberg(a.array, calc_q=True)
    return HessenbergResult(constructors.array(h, h_ax), constructors.array(q, q_ax))


def cross(
    a: NamedArrayLike,
    b: NamedArrayLike,
    axis: AxisSelector | tuple[AxisSelector, AxisSelector] = -1,
) -> NamedArray:
    """Cross product of 3-vectors along ``axis`` (a pair selects the axis in ``a`` and ``b`` separately).

    Other axes are broadcast against each other; the result has axes ``(*batch, vector_axis_of_a)``.
    """
    a, b = util.promote_dtypes_inexact(a, b)
    a_sel, b_sel = axis if isinstance(axis, tuple) else (axis, axis)
    (ia,) = axis_indices(a.axes, (a_sel,))
    (ib,) = axis_indices(b.axes, (b_sel,))
    a_c = operations.transpose(a, (*(i for i in range(a.ndim) if i != ia), ia))
    b_c = operations.transpose(b, (*(i for i in range(b.ndim) if i != ib), ib))
    a_named, b_named, batch = _align_batch(a_c, b_c, 1)
    return constructors.array(jnp.linalg.cross(a_named.array, b_named.array), (*batch, a_c.axes[-1]))


# generic matrix operations


def matrix_transpose(a: NamedArrayLike, matrix_axes: MatrixSelection = (-2, -1)) -> NamedArray:
    """Transpose of the matrices: ``out[..., i, j] = a[..., j, i]`` with the (row, col) *roles* kept in place.

    This is different from ``operations.swapaxes``, which swaps the axes *including their names* and is therefore
    only a permutation of axes. Here the result of a matrix with axes ``("a", "b")`` again has axes ``("a", "b")``.
    For non-square matrices the sizes swap (``n x m`` -> ``m x n``) while the names stay in place.
    """
    a = util.ensure_named("matrix_transpose", a)
    if a.ndim < 2:
        raise ValueError(f"matrix_transpose: input must have at least 2 dimensions, got {a.ndim}")
    i, j = axis_indices(a.axes, matrix_axes)
    if i == j:
        raise ValueError(f"matrix_transpose: `matrix_axes` {matrix_axes} resolve to the same axis (position {i})")
    axes = list(a.axes)
    axes[i], axes[j] = (
        Axis(a.axes[j].size, a.axes[i].name),
        Axis(a.axes[i].size, a.axes[j].name),
    )
    return constructors.array(jnp.swapaxes(a.array, i, j), tuple(axes))
