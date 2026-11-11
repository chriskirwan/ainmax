from typing import Any, Sequence

import jax.numpy as jnp
from jax.lax import PrecisionLike
from jax.typing import DTypeLike

from ainmax._src.axis import Axis, AxisLike, AxisSelector
from ainmax._src.axis_util import axis_index
from ainmax._src.feidhm import constructors, util
from ainmax._src.typing import NamedArray, NamedArrayLike


def _signal_1d(name: str, x: NamedArrayLike) -> NamedArray:
    """Validate a one-dimensional signal."""
    x = util.ensure_named(name, x)
    if x.ndim != 1:
        raise ValueError(f"{name}: expected a one-dimensional array, got {x.ndim} dimensions")
    return x


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


def _signal_output_axis(a: NamedArray, b: NamedArray, size: int, name: str) -> Axis:
    """Choose a common axis name for a one-dimensional signal result."""
    ax, bx = a.axes[0], b.axes[0]

    if ax.name is not None and bx.name is not None and ax.name != bx.name:
        raise ValueError(f"{name}: input axes have incompatible names {ax.name!r} and {bx.name!r}")

    return Axis(size, ax.name if ax.name is not None else bx.name)


def unwrap(
    p: NamedArrayLike,
    discont: Any = None,
    axis: AxisSelector = -1,
    period: Any = 2 * jnp.pi,
) -> NamedArray:
    """Unwrap a periodic signal along an axis."""
    named = util.ensure_named("unwrap", p)

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"unwrap: axis {axis!r} not found in {named.axes}")

    result = jnp.unwrap(
        named.array,
        discont=discont,
        axis=idx,
        period=period,
    )

    return constructors.array(result, named.axes)


def convolve(
    a: NamedArrayLike,
    v: NamedArrayLike,
    mode: str = "full",
    *,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> NamedArray:
    """Convolve two one-dimensional signals."""
    a = _signal_1d("convolve", a)
    v = _signal_1d("convolve", v)

    result = jnp.convolve(
        a.array,
        v.array,
        mode=mode,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    axis = _signal_output_axis(a, v, result.size, "convolve")
    return constructors.array(result, (axis,))


def correlate(
    a: NamedArrayLike,
    v: NamedArrayLike,
    mode: str = "valid",
    *,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> NamedArray:
    """Compute the cross-correlation of two one-dimensional signals."""
    a = _signal_1d("correlate", a)
    v = _signal_1d("correlate", v)

    result = jnp.correlate(
        a.array,
        v.array,
        mode=mode,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    axis = _signal_output_axis(a, v, result.size, "correlate")
    return constructors.array(result, (axis,))


def bincount(
    x: NamedArrayLike,
    weights: NamedArrayLike | None = None,
    minlength: int = 0,
    *,
    length: int | None = None,
    axis: AxisLike | None = None,
) -> NamedArray:
    """Count occurrences of non-negative integers.

    ``axis`` optionally names the resulting bin axis.
    """
    x = _signal_1d("bincount", x)

    if weights is not None:
        weights = _signal_1d("bincount", weights)
        if weights.size != x.size:
            raise ValueError("bincount: weights must have the same number of elements as x")

    result = jnp.bincount(
        x.array,
        weights=None if weights is None else weights.array,
        minlength=minlength,
        length=length,
    )

    return constructors.array(result, (_result_axis(axis, result.size, "bincount"),))


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

    ``axis`` optionally names the histogram's bin axis. The returned bin-edge
    array uses the same name with one additional element.
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
    """Compute the bin edges for a one-dimensional histogram."""
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
    """Compute a two-dimensional histogram."""
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
    """Compute an N-dimensional histogram."""
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


# Window functions


def bartlett(M: Axis) -> NamedArray:
    """Return the Bartlett window of length ``M.size`` on axis ``M``.

    Args:
        M: The axis of the window.

    Returns:
        A `NamedArray` of length ``M.size`` on axis ``M``.
    """
    return constructors.array(jnp.bartlett(M.size), M)


def blackman(M: Axis) -> NamedArray:
    """Return the Blackman window of length ``M.size`` on axis ``M``.

    Args:
        M: The axis of the window.

    Returns:
        A `NamedArray` of length ``M.size`` on axis ``M``.
    """
    return constructors.array(jnp.blackman(M.size), M)


def hanning(M: Axis) -> NamedArray:
    """Return the Hann window of length ``M.size`` on axis ``M``.

    Args:
        M: The axis of the window.

    Returns:
        A `NamedArray` of length ``M.size`` on axis ``M``.
    """
    return constructors.array(jnp.hanning(M.size), M)


def hamming(M: Axis) -> NamedArray:
    """Return a Hamming window of size M.

    Args:
        M: The axis of the window.

    Returns:
        An array of size M containing the Hamming window.
    """

    return constructors.array(jnp.hamming(M.size), M)


def kaiser(M: Axis, beta: NamedArrayLike) -> NamedArray:
    """Return the Kaiser window of length ``M.size`` on axis ``M``.

    Args:
        M: The axis of the window.
        beta: The shape parameter of the window.

    Returns:
        A `NamedArray` of length ``M.size`` on axis ``M``.
    """
    beta = util.ensure_named("kaiser", beta)
    return constructors.array(jnp.kaiser(M.size, beta.array), M)
