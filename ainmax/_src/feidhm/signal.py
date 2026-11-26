from typing import Any, Literal

import jax.numpy as jnp
import jax.scipy.signal as jsignal
from jax import lax
from jax.lax import PrecisionLike
from jax.typing import DTypeLike

from ainmax._src.axis import Axis, AxisLike, AxisSelector
from ainmax._src.axis_util import axis_index
from ainmax._src.feidhm import constructors, util
from ainmax._src.typing import NamedArray, NamedArrayLike


# TODO: once a `fft` submodule exists, build `fftconvolve` and the spectral estimators (`csd`, `welch`, `stft`,
# `istft`) on it instead of wrapping `jax.scipy.signal`.


def _signal_axis(name: str, x: NamedArray, axis: AxisSelector) -> int:
    """Resolve ``axis`` to a position in ``x``."""
    idx = axis_index(x.axes, axis)
    if idx is None:
        raise ValueError(f"{name}: axis {axis!r} not found in {x.axes}")
    return idx


def _align_signals(name: str, a: NamedArray, b: NamedArray) -> tuple[NamedArray, NamedArray]:
    """Pair the axes of two signals.

    Axes are paired by position, as in `scipy.signal`. The one exception is when every axis of both signals is named
    and ``b`` has the same names as ``a`` in a different order: ``b`` is then transposed to ``a``'s order.
    """
    if a.ndim != b.ndim:
        raise ValueError(f"{name}: inputs must have the same number of dimensions, got {a.ndim} and {b.ndim}")

    a_names = [ax.name for ax in a.axes]
    b_names = [ax.name for ax in b.axes]
    if None not in a_names and None not in b_names and a_names != b_names and set(a_names) == set(b_names):
        perm = [b_names.index(n) for n in a_names]
        b = constructors.array(jnp.transpose(b.array, perm), tuple(b.axes[i] for i in perm))

    return a, b


def _merge_names(name: str, a: NamedArray, b: NamedArray) -> tuple[str | None, ...]:
    """Combine the axis names of two aligned signals: they must agree wherever both are named."""
    names = []
    for ax, bx in zip(a.axes, b.axes):
        if ax.name is not None and bx.name is not None and ax.name != bx.name:
            raise ValueError(f"{name}: input axes have incompatible names {ax.name!r} and {bx.name!r}")
        names.append(ax.name if ax.name is not None else bx.name)
    return tuple(names)


def _scalar(name: str, value: NamedArrayLike) -> Any:
    """The raw array of a scalar argument such as the sampling frequency."""
    return util.ensure_named(name, value).array


def _window_array(window: Any) -> Any:
    """Windows may be given by name, as a ``(name, *parameters)`` tuple, or as an explicit array."""
    return window.array if isinstance(window, NamedArray) else window


def unwrap(
    p: NamedArrayLike,
    discont: Any = None,
    axis: AxisSelector = -1,
    period: Any = 2 * jnp.pi,
) -> NamedArray:
    """Unwrap a periodic signal along an axis."""
    named = util.ensure_named("unwrap", p)
    idx = _signal_axis("unwrap", named, axis)

    result = jnp.unwrap(
        named.array,
        discont=discont,
        axis=idx,
        period=period,
    )

    return constructors.array(result, named.axes)


def _convolve_direct(
    x: Any,
    kernel: Any,
    mode: str,
    precision: PrecisionLike,
    preferred_element_type: DTypeLike | None,
) -> Any:
    """N-dimensional direct convolution, lowered to `jax.lax.conv_general_dilated`.

    This is `jax.scipy.signal.convolve` with ``method="direct"``, except that ``preferred_element_type`` can be
    passed through (``jax.scipy.signal`` doesn't expose it).
    """
    if mode not in ("full", "same", "valid"):
        raise ValueError(f"mode must be one of ['full', 'same', 'valid'], got {mode!r}")

    dtype = jnp.result_type(x, kernel)
    if preferred_element_type is None and not jnp.issubdtype(dtype, jnp.inexact):
        # Like `jnp.convolve`: promote to a floating type unless the caller says how to accumulate.
        dtype = jnp.result_type(dtype, float)
    x, kernel = x.astype(dtype), kernel.astype(dtype)

    if mode == "valid":
        # Convolution commutes, so the smaller input can always play the kernel.
        if all(s >= k for s, k in zip(x.shape, kernel.shape)):
            pass
        elif all(s <= k for s, k in zip(x.shape, kernel.shape)):
            x, kernel = kernel, x
        else:
            raise ValueError("mode 'valid' requires one input to be at least as large as the other in every dimension")
        padding = [(0, 0)] * x.ndim
    elif mode == "same":
        # Centered slice of the full result, with the size of the first input.
        padding = [(k - 1 - (k - 1) // 2, (k - 1) // 2) for k in kernel.shape]
    else:
        padding = [(k - 1, k - 1) for k in kernel.shape]

    result = lax.conv_general_dilated(
        x[None, None],
        jnp.flip(kernel)[None, None],
        (1,) * x.ndim,
        padding,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    return result[0, 0]


ModeStr = Literal["full", "same", "valid"]


def _convolve_named(
    fn: str,
    a: NamedArrayLike,
    v: NamedArrayLike,
    mode: ModeStr,
    method: str,
    precision: PrecisionLike,
    preferred_element_type: DTypeLike | None,
    *,
    conjugate_flip: bool,
) -> NamedArray:
    """Shared implementation of `convolve` and `correlate` (``conjugate_flip=True``)."""
    if method not in ("auto", "direct", "fft"):
        raise ValueError(f"{fn}: method must be one of ['auto', 'direct', 'fft'], got {method!r}")

    a, v = util.ensure_named(fn, a, v)
    a, v = _align_signals(fn, a, v)
    names = _merge_names(fn, a, v)

    kernel = jnp.flip(jnp.conj(v.array)) if conjugate_flip else v.array

    if method == "fft":
        if precision is not None or preferred_element_type is not None:
            raise ValueError(f"{fn}: precision and preferred_element_type are not supported with method='fft'")
        result = jsignal.fftconvolve(a.array, kernel, mode=mode)
    else:
        result = _convolve_direct(a.array, kernel, mode, precision, preferred_element_type)

    return constructors.array(result, tuple(Axis(size, n) for size, n in zip(result.shape, names)))


def convolve(
    a: NamedArrayLike,
    v: NamedArrayLike,
    mode: ModeStr = "full",
    method: str = "auto",
    *,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> NamedArray:
    """Convolve two N-dimensional signals.

    This follows `scipy.signal.convolve`: the inputs must have the same number of dimensions and the axes are
    convolved pairwise by position. Axis names must agree wherever both inputs are named, and the result takes the name
    from whichever input has one. If every axis of both inputs is named and ``v`` has the same names as ``a`` in a
    different order, ``v`` is transposed to match ``a``.

    Args:
        a: The first signal.
        v: The second signal, with ``v.ndim == a.ndim``.
        mode: ``"full"`` for the full convolution, ``"same"`` for the centered part of it with the shape of ``a``, or
            ``"valid"`` for the part that doesn't depend on zero-padding. ``"valid"`` requires one input to be at
            least as large as the other in every dimension.
        method: ``"direct"`` (``"auto"`` also selects this) lowers to `jax.lax.conv_general_dilated`; ``"fft"`` uses
            `fftconvolve`, which doesn't support ``precision`` or ``preferred_element_type``.
        precision: The precision of the direct computation.
        preferred_element_type: The element type used to accumulate and return the result. When given, integer inputs
            are not promoted to floating point.

    Returns:
        The convolution as a `NamedArray`.

    Note:
        For one-dimensional inputs this agrees with `jax.numpy.convolve` except that ``mode="same"`` always has the
        size of ``a``, where NumPy returns the size of the larger input.
    """
    return _convolve_named("convolve", a, v, mode, method, precision, preferred_element_type, conjugate_flip=False)


def correlate(
    a: NamedArrayLike,
    v: NamedArrayLike,
    mode: ModeStr = "valid",
    method: str = "auto",
    *,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> NamedArray:
    """Cross-correlate two N-dimensional signals.

    Equivalent to ``convolve(a, flip(conj(v)), ...)``: in one dimension, ``c[k] = sum_n a[n + k] * conj(v[n])``. The
    pairing of axes and the handling of names are as for `convolve`.

    Args:
        a: The first signal.
        v: The second signal, with ``v.ndim == a.ndim``.
        mode: ``"full"``, ``"same"`` or ``"valid"``, as for `convolve`.
        method: ``"direct"`` (also ``"auto"``) or ``"fft"``, as for `convolve`.
        precision: The precision of the direct computation.
        preferred_element_type: The element type used to accumulate and return the result.

    Returns:
        The cross-correlation as a `NamedArray`.
    """
    return _convolve_named("correlate", a, v, mode, method, precision, preferred_element_type, conjugate_flip=True)


def fftconvolve(a: NamedArrayLike, v: NamedArrayLike, mode: ModeStr = "full") -> NamedArray:
    """Convolve two N-dimensional signals with the FFT.

    The inputs are paired and named as in `convolve`, and all axes are convolved. To convolve only some axes, map over
    the others.

    Args:
        a: The first signal.
        v: The second signal, with ``v.ndim == a.ndim``.
        mode: ``"full"``, ``"same"`` or ``"valid"``, as for `convolve`.

    Returns:
        The convolution as a `NamedArray`.
    """
    a, v = util.ensure_named("fftconvolve", a, v)
    a, v = _align_signals("fftconvolve", a, v)
    names = _merge_names("fftconvolve", a, v)

    result = jsignal.fftconvolve(a.array, v.array, mode=mode)
    return constructors.array(result, tuple(Axis(size, n) for size, n in zip(result.shape, names)))


def detrend(
    data: NamedArrayLike,
    axis: AxisSelector = -1,
    type: str = "linear",
    bp: int | tuple[int, ...] = 0,
) -> NamedArray:
    """Remove a linear or constant trend along an axis.

    Args:
        data: The signal.
        axis: The axis to detrend.
        type: ``"linear"`` subtracts a least-squares line, ``"constant"`` subtracts the mean.
        bp: Breakpoints (positions along ``axis``) at which a separate line is fitted to each piece.

    Returns:
        The detrended signal, with the axes of ``data``.
    """
    named = util.ensure_named("detrend", data)
    idx = _signal_axis("detrend", named, axis)
    return constructors.array(jsignal.detrend(named.array, axis=idx, type=type, bp=bp), named.axes)


# Spectral analysis
#
# In the signal's own units the transforms replace ``axis`` by a frequency axis (in place) and, for `stft`, append a
# segment axis. The new axes are anonymous unless named through ``freq_axis`` / ``time_axis``.


def csd(
    x: NamedArrayLike,
    y: NamedArrayLike | None = None,
    fs: NamedArrayLike = 1.0,
    window: str | tuple | NamedArrayLike = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: AxisSelector = -1,
    average: str = "mean",
    *,
    freq_axis: AxisLike | None = None,
) -> tuple[NamedArray, NamedArray]:
    """Estimate the cross power spectral density with Welch's method.

    Args:
        x: The first signal.
        y: The second signal. If ``None``, the power spectral density of ``x`` is computed. ``x`` and ``y`` are paired
            as in `convolve`, except that axes of size one broadcast.
        fs: The sampling frequency.
        window: A window name, a ``(name, *parameters)`` tuple, or a window array of length ``nperseg``.
        nperseg: The length of each segment.
        noverlap: The number of points shared by adjacent segments.
        nfft: The FFT length.
        detrend: How each segment is detrended (``"constant"``, ``"linear"`` or ``False``).
        return_onesided: Return only the non-negative frequencies of a real signal.
        scaling: ``"density"`` or ``"spectrum"``.
        axis: The axis of the signal along which to compute the spectrum.
        average: ``"mean"`` or ``"median"``, how the segments are averaged.
        freq_axis: The name of the output frequency axis, or an `Axis` of the right size.

    Returns:
        The frequencies, on the frequency axis, and the cross spectral density, which has the axes of ``x`` with
        ``axis`` replaced by the frequency axis.
    """
    x = util.ensure_named("csd", x)
    idx = _signal_axis("csd", x, axis)

    names = tuple(ax.name for ax in x.axes)
    y_array = None
    if y is not None:
        y = util.ensure_named("csd", y)
        x, y = _align_signals("csd", x, y)
        names = _merge_names("csd", x, y)
        y_array = y.array

    freqs, result = jsignal.csd(
        x.array,
        y_array,
        fs=_scalar("csd", fs),
        window=_window_array(window),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,  # pyrefly: ignore[bad-argument-type]
        return_onesided=return_onesided,
        scaling=scaling,
        axis=idx,
        average=average,
    )

    freq = util.resolve_new_axis(freq_axis, freqs.size, "csd", "freq_axis")
    axes = tuple(Axis(size, n) for size, n in zip(result.shape, names))
    axes = axes[:idx] + (freq,) + axes[idx + 1 :]
    return constructors.array(freqs, (freq,)), constructors.array(result, axes)


def welch(
    x: NamedArrayLike,
    fs: NamedArrayLike = 1.0,
    window: str | tuple | NamedArrayLike = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool = "constant",
    return_onesided: bool = True,
    scaling: str = "density",
    axis: AxisSelector = -1,
    average: str = "mean",
    *,
    freq_axis: AxisLike | None = None,
) -> tuple[NamedArray, NamedArray]:
    """Estimate the power spectral density with Welch's method.

    The arguments are those of `csd`.

    Returns:
        The frequencies, on the frequency axis, and the power spectral density, which has the axes of ``x`` with
        ``axis`` replaced by the frequency axis.
    """
    x = util.ensure_named("welch", x)
    idx = _signal_axis("welch", x, axis)

    freqs, result = jsignal.welch(
        x.array,
        fs=_scalar("welch", fs),
        window=_window_array(window),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,  # pyrefly: ignore[bad-argument-type]
        return_onesided=return_onesided,
        scaling=scaling,
        axis=idx,
        average=average,
    )

    freq = util.resolve_new_axis(freq_axis, freqs.size, "welch", "freq_axis")
    axes = x.axes[:idx] + (freq,) + x.axes[idx + 1 :]
    return constructors.array(freqs, (freq,)), constructors.array(result, axes)


def stft(
    x: NamedArrayLike,
    fs: NamedArrayLike = 1.0,
    window: str | tuple | NamedArrayLike = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | bool = False,
    return_onesided: bool = True,
    boundary: str | None = "zeros",
    padded: bool = True,
    axis: AxisSelector = -1,
    *,
    freq_axis: AxisLike | None = None,
    time_axis: AxisLike | None = None,
) -> tuple[NamedArray, NamedArray, NamedArray]:
    """Compute the short-time Fourier transform.

    Args:
        x: The signal.
        fs: The sampling frequency.
        window: A window name, a ``(name, *parameters)`` tuple, or a window array of length ``nperseg``.
        nperseg: The length of each segment.
        noverlap: The number of points shared by adjacent segments.
        nfft: The FFT length.
        detrend: How each segment is detrended (``"constant"``, ``"linear"`` or ``False``).
        return_onesided: Return only the non-negative frequencies of a real signal.
        boundary: How the signal is extended at both ends (``"zeros"``, ``"even"``, ``"odd"``, ``"constant"``) or
            ``None`` for no extension.
        padded: Zero-pad the signal so that it fits a whole number of segments.
        axis: The axis of the signal along which to transform.
        freq_axis: The name of the output frequency axis, or an `Axis` of the right size.
        time_axis: The name of the output segment axis, or an `Axis` of the right size. Defaults to the name of
            ``axis``.

    Returns:
        The frequencies, the segment times and the transform. The transform has the axes of ``x`` with ``axis``
        replaced by the frequency axis, followed by the segment axis.
    """
    x = util.ensure_named("stft", x)
    idx = _signal_axis("stft", x, axis)

    freqs, times, result = jsignal.stft(
        x.array,
        fs=_scalar("stft", fs),
        window=_window_array(window),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,  # pyrefly: ignore[bad-argument-type]
        return_onesided=return_onesided,
        boundary=boundary,
        padded=padded,
        axis=idx,
    )

    freq = util.resolve_new_axis(freq_axis, freqs.size, "stft", "freq_axis")
    segments = util.resolve_new_axis(
        time_axis if time_axis is not None else x.axes[idx].name, times.size, "stft", "time_axis"
    )
    axes = x.axes[:idx] + (freq,) + x.axes[idx + 1 :] + (segments,)
    return (
        constructors.array(freqs, (freq,)),
        constructors.array(times, (segments,)),
        constructors.array(result, axes),
    )


def istft(
    Zxx: NamedArrayLike,
    fs: NamedArrayLike = 1.0,
    window: str | tuple | NamedArrayLike = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    input_onesided: bool = True,
    boundary: bool = True,
    time_axis: AxisSelector = -1,
    freq_axis: AxisSelector = -2,
) -> tuple[NamedArray, NamedArray]:
    """Invert a short-time Fourier transform.

    Args:
        Zxx: The transform, such as that returned by `stft`.
        fs: The sampling frequency.
        window: A window name, a ``(name, *parameters)`` tuple, or a window array of length ``nperseg``.
        nperseg: The length of each segment.
        noverlap: The number of points shared by adjacent segments.
        nfft: The FFT length.
        input_onesided: Whether ``Zxx`` holds only the non-negative frequencies.
        boundary: Whether the signal was extended at both ends when transformed.
        time_axis: The segment axis of ``Zxx``.
        freq_axis: The frequency axis of ``Zxx``.

    Returns:
        The times and the reconstructed signal. The signal has the axes of ``Zxx`` without the frequency axis, and the
        segment axis, in its position and with its name, now holds the time samples.
    """
    Zxx = util.ensure_named("istft", Zxx)
    t_idx = _signal_axis("istft", Zxx, time_axis)
    f_idx = _signal_axis("istft", Zxx, freq_axis)

    _, signal = jsignal.istft(
        Zxx.array,
        fs=_scalar("istft", fs),
        window=_window_array(window),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=input_onesided,
        boundary=boundary,
        time_axis=t_idx,
        freq_axis=f_idx,
    )

    # Where the time axis ends up once the frequency axis is gone. `jax.scipy.signal.istft` returns its own time vector
    # but sizes it by `signal.shape[0]` instead of the time axis, so it is built here.
    out_idx = t_idx - 1 if f_idx < t_idx else t_idx
    kept = tuple(ax for i, ax in enumerate(Zxx.axes) if i != f_idx)
    time = Axis(signal.shape[out_idx], kept[out_idx].name)
    axes = kept[:out_idx] + (time,) + kept[out_idx + 1 :]

    times = jnp.arange(time.size, dtype=jnp.finfo(signal.dtype).dtype) / _scalar("istft", fs)
    return constructors.array(times, (time,)), constructors.array(signal, axes)


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
