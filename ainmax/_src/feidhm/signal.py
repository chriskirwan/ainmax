from typing import Any

import jax.numpy as jnp
from jax.lax import PrecisionLike
from jax.typing import DTypeLike

from ainmax._src.axis import Axis, AxisSelector
from ainmax._src.axis_util import axis_index
from ainmax._src.feidhm import constructors, util
from ainmax._src.typing import NamedArray, NamedArrayLike


def _signal_1d(name: str, x: NamedArrayLike) -> NamedArray:
    """Validate a one-dimensional signal."""
    x = util.ensure_named(name, x)
    if x.ndim != 1:
        raise ValueError(f"{name}: expected a one-dimensional array, got {x.ndim} dimensions")
    return x


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


# TODO: scipy ND convolve: Does not support preferred_element_type, so maybe plug into `tensor_contractions.dot` for
# the non "fft" mode?
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


# TODO: scipy ND correlate: Does not support preferred_element_type, so maybe plug into `tensor_contractions.dot` for
# the non "fft" mode?
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
