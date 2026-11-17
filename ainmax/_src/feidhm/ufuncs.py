from collections.abc import Callable
from functools import wraps

import jax.lax as lax
import jax.numpy as jnp
from jax import Array

from ainmax._src.feidhm import constructors, operations, util
from ainmax._src.typing import NamedArray, NamedArrayLike


def wrap_elementwise_unary_op(fn: Callable[..., Array]) -> Callable[..., NamedArray]:
    """Wrap a unary JAX function to preserve named axes.

    Args:
        fn: A function mapping a JAX array (plus any extra positional or keyword arguments) to an array of the same
            shape.

    Returns:
        A function taking a ``NamedArrayLike`` first argument, applying ``fn`` to its underlying array, and returning
        a ``NamedArray`` with the same axes. Extra arguments are forwarded to ``fn``.
    """

    @wraps(fn)
    def wrapped(x: NamedArrayLike, *args, **kwargs) -> NamedArray:
        x = util.ensure_named("wrap_elementwise_unary", x)
        return constructors.array(fn(x.array, *args, **kwargs), x.axes)

    return wrapped


def wrap_elementwise_binary_op(fn: Callable[..., Array]) -> Callable[..., NamedArray]:
    """Wrap a binary JAX function with named-axis broadcasting.

    Args:
        fn: A function mapping two broadcast-compatible JAX arrays (plus any extra positional or keyword arguments)
            to an array of the broadcast shape.

    Returns:
        A function taking two ``NamedArrayLike`` arguments, broadcasting them to a common set of axes by name,
        applying ``fn``, and returning a ``NamedArray`` on those axes. Extra arguments are forwarded to ``fn``.
    """

    @wraps(fn)
    def wrapped(x: NamedArrayLike, y: NamedArrayLike, *args, **kwargs):
        x, y = util.ensure_named("wrap_elementwise_binary_op", x, y)
        (a, b), axes = operations.broadcast_arrays_and_shape(x, y)
        return constructors.array(fn(a.array, b.array, *args, **kwargs), axes)

    return wrapped


def abs(x: NamedArrayLike) -> NamedArray:
    r"""Compute the absolute value of each element.

    For complex input this is the magnitude

    .. math::

        |a + bi| = \sqrt{a^2 + b^2}

    and the result has the corresponding real dtype.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the absolute value of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.abs)(x)


def absolute(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`abs`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`abs`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.absolute)(x)


def acos(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse cosine of each element, in radians.

    Real inputs outside ``[-1, 1]`` give ``nan``. Complex inputs use the principal branch.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the inverse cosine, in the range ``[0, pi]`` for real input, with the same axes as
        ``x``.
    """
    return wrap_elementwise_unary_op(jnp.acos)(x)


def acosh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic cosine of each element.

    Real inputs less than ``1`` give ``nan``. Complex inputs use the principal branch.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the inverse hyperbolic cosine, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.acosh)(x)


def arccos(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`acos`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`acos`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.arccos)(x)


def arccosh(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`acosh`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`acosh`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.arccosh)(x)


def arcsin(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`asin`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`asin`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.arcsin)(x)


def arcsinh(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`asinh`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`asinh`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.arcsinh)(x)


def arctan(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`atan`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`atan`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.arctan)(x)


def arctanh(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`atanh`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`atanh`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.arctanh)(x)


def asin(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse sine of each element, in radians.

    Real inputs outside ``[-1, 1]`` give ``nan``. Complex inputs use the principal branch.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the inverse sine, in the range ``[-pi/2, pi/2]`` for real input, with the same axes
        as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.asin)(x)


def asinh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic sine of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the inverse hyperbolic sine, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.asinh)(x)


def atan(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse tangent of each element, in radians.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the inverse tangent, in the range ``[-pi/2, pi/2]`` for real input, with the same
        axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.atan)(x)


def atanh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic tangent of each element.

    Real inputs outside ``[-1, 1]`` give ``nan``, and ``+1`` and ``-1`` give ``inf`` and ``-inf``.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the inverse hyperbolic tangent, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.atanh)(x)


def bitwise_invert(x: NamedArrayLike) -> NamedArray:
    """Invert the bits of each integer element.

    For signed integers this is ``-x - 1``; for unsigned integers it flips every bit within the dtype's width. Boolean
    inputs are logically negated.

    Args:
        x: Input array or scalar of integer or boolean dtype.

    Returns:
        A ``NamedArray`` containing the bitwise complement of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.bitwise_invert)(x)


def bitwise_count(x: NamedArrayLike, /) -> NamedArray:
    """Count the set bits in each integer element.

    For signed integers the count is taken over the absolute value.

    Args:
        x: Input array or scalar of integer dtype.

    Returns:
        A ``NamedArray`` of ``uint8`` counts of set bits, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.bitwise_count)(x)


def bitwise_not(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`bitwise_invert`.

    Args:
        x: Input array or scalar of integer or boolean dtype.

    Returns:
        A ``NamedArray`` containing the result of :func:`bitwise_invert`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.bitwise_not)(x)


def cbrt(x: NamedArrayLike) -> NamedArray:
    """Compute the real cube root of each element.

    Unlike ``x ** (1 / 3)``, this is defined for negative inputs.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing the cube root of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.cbrt)(x)


def ceil(x: NamedArrayLike) -> NamedArray:
    """Round each element up to the nearest integer.

    The result keeps its floating-point dtype.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing the smallest integer not less than each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.ceil)(x)


def conj(x: NamedArrayLike) -> NamedArray:
    """Compute the complex conjugate of each element.

    Real inputs are returned unchanged, promoted to a complex dtype where JAX does so.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the complex conjugate of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.conj)(x)


def conjugate(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`conj`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`conj`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.conjugate)(x)


def cos(x: NamedArrayLike) -> NamedArray:
    """Compute the cosine of each element, interpreted in radians.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the cosine of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.cos)(x)


def cosh(x: NamedArrayLike) -> NamedArray:
    """Compute the hyperbolic cosine of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the hyperbolic cosine of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.cosh)(x)


def deg2rad(x: NamedArrayLike) -> NamedArray:
    r"""Convert each angle from degrees to radians.

    .. math::

        \theta_{\mathrm{rad}} = \theta_{\mathrm{deg}} \cdot \frac{\pi}{180}

    Args:
        x: Angles in degrees.

    Returns:
        A ``NamedArray`` containing the angles in radians, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.deg2rad)(x)


def degrees(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`rad2deg`.

    Args:
        x: Angles in radians.

    Returns:
        A ``NamedArray`` containing the result of :func:`rad2deg`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.degrees)(x)


def exp(x: NamedArrayLike) -> NamedArray:
    """Compute the exponential of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`e^x`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.exp)(x)


def exp2(x: NamedArrayLike) -> NamedArray:
    """Compute two raised to the power of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`2^x`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.exp2)(x)


def expm1(x: NamedArrayLike) -> NamedArray:
    """Compute the exponential of each element minus one.

    This retains precision for small ``x``, where ``exp(x) - 1`` suffers from cancellation.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`e^x - 1`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.expm1)(x)


def fabs(x: NamedArrayLike) -> NamedArray:
    """Compute the absolute value of each element as a floating-point value.

    Integer inputs are promoted to a floating-point dtype. Complex inputs are not supported; use :func:`abs` for those.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` of floating-point absolute values, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.fabs)(x)


def floor(x: NamedArrayLike) -> NamedArray:
    """Round each element down to the nearest integer.

    The result keeps its floating-point dtype.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing the largest integer not greater than each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.floor)(x)


def frexp(x: NamedArrayLike) -> tuple[NamedArray, NamedArray]:
    r"""Decompose each element into a mantissa and a base-two exponent.

    .. math::

        x = m \cdot 2^e

    where the magnitude of the mantissa satisfies :math:`0.5 \le |m| < 1` (or ``m`` is zero). This is the inverse of
    :func:`ldexp`.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A tuple ``(mantissa, exponent)`` of ``NamedArray`` values, each with the same axes as ``x``. The mantissa has
        a floating-point dtype and the exponent an integer dtype.
    """
    x = util.ensure_named("frexp", x)
    m, e = jnp.frexp(x.array)
    return constructors.array(m, x.axes), constructors.array(e, x.axes)


def i0(x: NamedArrayLike) -> NamedArray:
    r"""Compute the modified Bessel function of the first kind of order zero.

    .. math::

        I_0(x) = \sum_{k=0}^{\infty} \frac{(x / 2)^{2k}}{(k!)^2}

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`I_0(x)`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.i0)(x)


def imag(x: NamedArrayLike) -> NamedArray:
    """Return the imaginary part of each element.

    Real inputs give zeros.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the imaginary part of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.imag)(x)


def invert(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`bitwise_invert`.

    Args:
        x: Input array or scalar of integer or boolean dtype.

    Returns:
        A ``NamedArray`` containing the result of :func:`bitwise_invert`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.invert)(x)


def isfinite(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is finite.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` of booleans that are ``True`` where the element is neither infinite nor NaN, with the same
        axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.isfinite)(x)


def isinf(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is infinite.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` of booleans that are ``True`` where the element is positive or negative infinity, with the
        same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.isinf)(x)


def isnan(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is NaN.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` of booleans that are ``True`` where the element is NaN, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.isnan)(x)


def isneginf(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is negative infinity.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` of booleans that are ``True`` where the element is ``-inf``, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.isneginf)(x)


def isposinf(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is positive infinity.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` of booleans that are ``True`` where the element is ``+inf``, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.isposinf)(x)


def log(x: NamedArrayLike) -> NamedArray:
    r"""Compute the natural logarithm of each element.

    Zero gives ``-inf``, and negative real inputs give ``nan``. Complex inputs use the principal branch.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\ln x`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.log)(x)


def log10(x: NamedArrayLike) -> NamedArray:
    r"""Compute the base-10 logarithm of each element.

    Zero gives ``-inf``, and negative real inputs give ``nan``.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\log_{10} x`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.log10)(x)


def log1p(x: NamedArrayLike) -> NamedArray:
    r"""Compute the natural logarithm of one plus each element.

    This retains precision for small ``x``, where ``log(1 + x)`` suffers from cancellation.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\ln(1 + x)`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.log1p)(x)


def log2(x: NamedArrayLike) -> NamedArray:
    r"""Compute the base-two logarithm of each element.

    Zero gives ``-inf``, and negative real inputs give ``nan``.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\log_2 x`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.log2)(x)


def logical_not(x: NamedArrayLike) -> NamedArray:
    """Compute the logical negation of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` of booleans that are ``True`` where the element is zero (falsy), with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.logical_not)(x)


def modf(x: NamedArrayLike, /) -> tuple[NamedArray, NamedArray]:
    """Split each element into its fractional and integral parts.

    Both parts have the same sign as the input element.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A tuple ``(fractional, integral)`` of floating-point ``NamedArray`` values, each with the same axes as ``x``.
    """
    x = util.ensure_named("modf", x)
    out0, out1 = jnp.modf(x.array)
    return constructors.array(out0, x.axes), constructors.array(out1, x.axes)


def negative(x: NamedArrayLike) -> NamedArray:
    """Compute the numerical negative of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing ``-x``, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.negative)(x)


def positive(x: NamedArrayLike) -> NamedArray:
    """Return each element unchanged.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing ``+x``, the same values as ``x``, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.positive)(x)


def rad2deg(x: NamedArrayLike) -> NamedArray:
    r"""Convert each angle from radians to degrees.

    .. math::

        \theta_{\mathrm{deg}} = \theta_{\mathrm{rad}} \cdot \frac{180}{\pi}

    Args:
        x: Angles in radians.

    Returns:
        A ``NamedArray`` containing the angles in degrees, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.rad2deg)(x)


def radians(x: NamedArrayLike) -> NamedArray:
    """Alias of :func:`deg2rad`.

    Args:
        x: Angles in degrees.

    Returns:
        A ``NamedArray`` containing the result of :func:`deg2rad`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.radians)(x)


def real(x: NamedArrayLike) -> NamedArray:
    """Return the real part of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the real part of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.real)(x)


def reciprocal(x: NamedArrayLike) -> NamedArray:
    """Compute the reciprocal of each element.

    Integer inputs use integer division semantics, so only ``+/-1`` give non-zero results; convert to floating point
    first for the usual reciprocal.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`1 / x`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.reciprocal)(x)


def rint(x: NamedArrayLike) -> NamedArray:
    """Round each element to the nearest integer, preserving its dtype.

    Halves are rounded to the nearest even integer, so ``0.5`` and ``2.5`` both round to even values (``0.`` and
    ``2.``).

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing the rounded values, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.rint)(x)


def rsqrt(x: NamedArrayLike) -> NamedArray:
    r"""Compute the reciprocal of the square root of each element.

    This is implemented with :func:`jax.lax.rsqrt`, which can be faster and more accurate than ``1 / sqrt(x)``. Inputs
    must have a floating-point or complex dtype.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`1 / \sqrt{x}`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(lax.rsqrt)(x)


def sign(x: NamedArrayLike) -> NamedArray:
    r"""Compute the sign of each element.

    For real input the result is ``-1``, ``0`` or ``1``. For complex input it is

    .. math::

        \mathrm{sign}(z) = \frac{z}{|z|}

    with ``sign(0) = 0``.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the sign of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.sign)(x)


def signbit(x: NamedArrayLike) -> NamedArray:
    """Test whether the sign bit is set for each element.

    Unlike ``x < 0``, this is ``True`` for negative zero and for NaNs with the sign bit set.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` of booleans that are ``True`` where the sign bit is set, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.signbit)(x)


def sin(x: NamedArrayLike) -> NamedArray:
    """Compute the sine of each element, interpreted in radians.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the sine of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.sin)(x)


def sinc(x: NamedArrayLike) -> NamedArray:
    r"""Compute the normalized sinc function elementwise.

    .. math::

        \mathrm{sinc}(x) = \frac{\sin(\pi x)}{\pi x}

    with :math:`\mathrm{sinc}(0) = 1`. This is the normalized sinc, as in NumPy, not :math:`\sin(x) / x`.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{sinc}(x)`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.sinc)(x)


def sinh(x: NamedArrayLike) -> NamedArray:
    """Compute the hyperbolic sine of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the hyperbolic sine of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.sinh)(x)


def sort_complex(x: NamedArrayLike) -> NamedArray:
    """Sort complex values by real part, then by imaginary part.

    The sort is along the last axis of ``x``. Real inputs are promoted to complex.

    Args:
        x: Input array of real or complex values.

    Returns:
        A ``NamedArray`` of sorted values with a complex dtype, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.sort_complex)(x)


def spacing(x: NamedArrayLike, /) -> NamedArray:
    """Compute the distance to the adjacent representable value.

    The result is the gap between each element and the next representable floating-point value of larger magnitude.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing the spacing of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.spacing)(x)


def square(x: NamedArrayLike) -> NamedArray:
    """Compute the square of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`x^2`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.square)(x)


def sqrt(x: NamedArrayLike) -> NamedArray:
    r"""Compute the non-negative square root of each element.

    Negative real inputs give ``nan``. Complex inputs use the principal branch.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\sqrt{x}`, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.sqrt)(x)


def tan(x: NamedArrayLike) -> NamedArray:
    """Compute the tangent of each element, interpreted in radians.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the tangent of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.tan)(x)


def tanh(x: NamedArrayLike) -> NamedArray:
    """Compute the hyperbolic tangent of each element.

    Args:
        x: Input array or scalar.

    Returns:
        A ``NamedArray`` containing the hyperbolic tangent of each element, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.tanh)(x)


def trunc(x: NamedArrayLike) -> NamedArray:
    """Round each element toward zero.

    The result keeps its floating-point dtype.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing each element with its fractional part discarded, with the same axes as ``x``.
    """
    return wrap_elementwise_unary_op(jnp.trunc)(x)


# Binary Elementwise Operations
def add(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Add corresponding elements.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` containing ``x + y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.add)(x, y)


def atan2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    r"""Compute the elementwise four-quadrant inverse tangent, in radians.

    .. math::

        \mathrm{atan2}(x, y) = \arg(y + i x)

    Following the usual convention, the first argument is the y-coordinate and the second is the x-coordinate, so for a
    point :math:`(X, Y)` the angle is ``atan2(Y, X)``. The signs of both arguments are used to select the quadrant.

    Args:
        x: y-coordinates. Real-valued input array or scalar.
        y: x-coordinates. Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing the angle in the range ``[-pi, pi]``, broadcast to the common axes of ``x`` and
        ``y``.
    """
    return wrap_elementwise_binary_op(jnp.atan2)(x, y)


def arctan2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Alias of :func:`atan2`.

    Args:
        x: y-coordinates. Real-valued input array or scalar.
        y: x-coordinates. Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`atan2`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.arctan2)(x, y)


def bitwise_and(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the bitwise AND of corresponding integer elements.

    Args:
        x: First input array or scalar of integer or boolean dtype.
        y: Second input array or scalar of integer or boolean dtype.

    Returns:
        A ``NamedArray`` containing ``x & y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.bitwise_and)(x, y)


def bitwise_left_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Shift the bits of each first operand left by the second operand.

    Bits shifted past the width of the dtype are discarded.

    Args:
        x: Array or scalar of integers to shift.
        y: Array or scalar of non-negative integer shift amounts.

    Returns:
        A ``NamedArray`` containing ``x << y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.bitwise_left_shift)(x, y)


def bitwise_or(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the bitwise OR of corresponding integer elements.

    Args:
        x: First input array or scalar of integer or boolean dtype.
        y: Second input array or scalar of integer or boolean dtype.

    Returns:
        A ``NamedArray`` containing ``x | y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.bitwise_or)(x, y)


def bitwise_right_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Shift the bits of each first operand right by the second operand.

    Signed integers use an arithmetic shift, which preserves the sign; unsigned integers use a logical shift.

    Args:
        x: Array or scalar of integers to shift.
        y: Array or scalar of non-negative integer shift amounts.

    Returns:
        A ``NamedArray`` containing ``x >> y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.bitwise_right_shift)(x, y)


def bitwise_xor(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the bitwise XOR of corresponding integer elements.

    Args:
        x: First input array or scalar of integer or boolean dtype.
        y: Second input array or scalar of integer or boolean dtype.

    Returns:
        A ``NamedArray`` containing ``x ^ y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.bitwise_xor)(x, y)


def copysign(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the magnitude of the first operand with the sign of the second.

    The sign is taken from the sign bit, so a negative-zero ``y`` gives a negative result.

    Args:
        x: Array or scalar supplying the magnitudes. Real-valued.
        y: Array or scalar supplying the signs. Real-valued.

    Returns:
        A ``NamedArray`` with the magnitudes of ``x`` and the signs of ``y``, broadcast to the common axes of ``x`` and
        ``y``.
    """
    return wrap_elementwise_binary_op(jnp.copysign)(x, y)


def divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Divide corresponding elements.

    This is true division: integer inputs are promoted to a floating-point dtype. Division by zero gives ``inf``,
    ``-inf`` or ``nan`` rather than raising.

    Args:
        x: Dividend array or scalar.
        y: Divisor array or scalar.

    Returns:
        A ``NamedArray`` containing ``x / y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.divide)(x, y)


def divmod(x: NamedArrayLike, y: NamedArrayLike, /) -> tuple[NamedArray, NamedArray]:
    """Return the elementwise quotient and remainder.

    The quotient is :func:`floor_divide` and the remainder is :func:`mod`, so that ``x == q * y + r``, and ``r`` has
    the sign of the divisor ``y``.

    Args:
        x: Dividend array or scalar.
        y: Divisor array or scalar.

    Returns:
        A tuple ``(quotient, remainder)`` of ``NamedArray`` values, each broadcast to the common axes of ``x`` and
        ``y``.
    """
    a, b = util.ensure_named("divmod", x, y)
    _, _, axes = util.align_shapes_for_broadcast(a.axes, b.axes)
    a_array, b_array = jnp.divmod(a.array, b.array)
    return constructors.array(a_array, axes), constructors.array(b_array, axes)


def equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether corresponding elements are equal.

    NaN compares unequal to everything, including itself. To compare whole arrays, see :func:`array_equal`.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans containing ``x == y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.equal)(x, y)


def float_power(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Raise the first operand to the power of the second using floating-point arithmetic.

    Unlike :func:`power`, inputs are always promoted to a floating-point dtype first, so integer inputs give floating-
    point results.

    Args:
        x: Base array or scalar.
        y: Exponent array or scalar.

    Returns:
        A ``NamedArray`` containing ``x ** y`` computed in floating point, broadcast to the common axes of ``x`` and
        ``y``.
    """
    return wrap_elementwise_binary_op(jnp.float_power)(x, y)


def floor_divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    r"""Divide corresponding elements and round the result down.

    This rounds toward negative infinity, not toward zero, and is consistent with :func:`mod` so that ``x ==
    floor_divide(x, y) * y + mod(x, y)``.

    Args:
        x: Dividend array or scalar.
        y: Divisor array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\lfloor x / y \rfloor`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.floor_divide)(x, y)


def fmod(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the remainder from division, with the sign of the dividend.

    This differs from :func:`mod` for operands of opposite sign: the result takes the sign of ``x`` rather than ``y``.

    Args:
        x: Dividend array or scalar.
        y: Divisor array or scalar.

    Returns:
        A ``NamedArray`` containing ``x - y * trunc(x / y)``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.fmod)(x, y)


def gcd(a: NamedArrayLike, b: NamedArrayLike, /) -> NamedArray:
    """Compute the greatest common divisor of corresponding integer elements.

    The result is always non-negative, and ``gcd(0, 0)`` is ``0``.

    Args:
        a: First input array or scalar of integer dtype.
        b: Second input array or scalar of integer dtype.

    Returns:
        A ``NamedArray`` containing the greatest common divisor, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.gcd)(a, b)


def greater(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether each first-operand element is greater than the corresponding second element.

    Args:
        x: Real-valued first input array or scalar.
        y: Real-valued second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans containing ``x > y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.greater)(x, y)


def greater_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether each first-operand element is greater than or equal to the corresponding second element.

    Args:
        x: Real-valued first input array or scalar.
        y: Real-valued second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans containing ``x >= y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.greater_equal)(x, y)


def heaviside(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    r"""Evaluate the Heaviside step function using the second operand at zero.

    .. math::

        H(x, y) = \begin{cases} 0 & x < 0 \\ y & x = 0 \\ 1 & x > 0 \end{cases}

    Args:
        x: Input values. Real-valued array or scalar.
        y: Value of the step function where ``x == 0``. Real-valued array or scalar.

    Returns:
        A ``NamedArray`` containing the step function values, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.heaviside)(x, y)


def hypot(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    r"""Compute the hypotenuse from corresponding elements.

    This avoids intermediate overflow and underflow.

    Args:
        x: First leg length. Real-valued array or scalar.
        y: Second leg length. Real-valued array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\sqrt{x^2 + y^2}`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.hypot)(x, y)


def lcm(a: NamedArrayLike, b: NamedArrayLike, /) -> NamedArray:
    """Compute the least common multiple of corresponding integer elements.

    The result is always non-negative, and is ``0`` if either input is ``0``.

    Args:
        a: First input array or scalar of integer dtype.
        b: Second input array or scalar of integer dtype.

    Returns:
        A ``NamedArray`` containing the least common multiple, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.lcm)(a, b)


def ldexp(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    r"""Multiply each first-operand element by two raised to the second operand.

    This is the inverse of :func:`frexp`.

    Args:
        x: Array or scalar of mantissas. Real-valued.
        y: Array or scalar of integer exponents.

    Returns:
        A ``NamedArray`` containing :math:`x \cdot 2^y`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.ldexp)(x, y)


def left_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Alias of :func:`bitwise_left_shift`.

    Args:
        x: Array or scalar of integers to shift.
        y: Array or scalar of non-negative integer shift amounts.

    Returns:
        A ``NamedArray`` containing the result of :func:`bitwise_left_shift`, broadcast to the common axes of ``x`` and
        ``y``.
    """
    return wrap_elementwise_binary_op(jnp.left_shift)(x, y)


def less(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether each first-operand element is less than the corresponding second element.

    Args:
        x: Real-valued first input array or scalar.
        y: Real-valued second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans containing ``x < y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.less)(x, y)


def less_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether each first-operand element is less than or equal to the corresponding second element.

    Args:
        x: Real-valued first input array or scalar.
        y: Real-valued second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans containing ``x <= y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.less_equal)(x, y)


def logaddexp(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    r"""Compute the logarithm of the sum of exponentials, using base e.

    This is computed in a numerically stable way, and is the two-argument form of
    ``logsumexp``.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\log(e^x + e^y)`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.logaddexp)(x, y)


def logaddexp2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    r"""Compute the logarithm of the sum of powers of two.

    This is computed in a numerically stable way.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\log_2(2^x + 2^y)`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.logaddexp2)(x, y)


def logical_and(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the logical AND of corresponding elements.

    Inputs are interpreted as booleans, with zero being ``False``.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans containing ``x and y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.logical_and)(x, y)


def logical_or(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the logical OR of corresponding elements.

    Inputs are interpreted as booleans, with zero being ``False``.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans containing ``x or y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.logical_or)(x, y)


def logical_xor(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the logical XOR of corresponding elements.

    Inputs are interpreted as booleans, with zero being ``False``.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans that are ``True`` where exactly one of ``x`` and ``y`` is true, broadcast to the
        common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.logical_xor)(x, y)


def maximum(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the elementwise maximum, propagating NaNs.

    If either element of a pair is NaN, the result is NaN. See :func:`clip` for limiting values to a range.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` containing the larger of each pair of elements, broadcast to the common axes of ``x`` and
        ``y``.
    """
    return wrap_elementwise_binary_op(jnp.maximum)(x, y)


def minimum(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the elementwise minimum, propagating NaNs.

    If either element of a pair is NaN, the result is NaN. See :func:`clip` for limiting values to a range.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` containing the smaller of each pair of elements, broadcast to the common axes of ``x`` and
        ``y``.
    """
    return wrap_elementwise_binary_op(jnp.minimum)(x, y)


def mod(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the remainder from division, with the sign of the divisor.

    This is the Python ``%`` operator. For a result with the sign of the dividend, use :func:`fmod`.

    Args:
        x: Dividend array or scalar.
        y: Divisor array or scalar.

    Returns:
        A ``NamedArray`` containing ``x - y * floor(x / y)``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.mod)(x, y)


def multiply(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Multiply corresponding elements.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` containing ``x * y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.multiply)(x, y)


def nextafter(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the next representable value from the first operand toward the second.

    Where ``x == y`` the result is ``y``.

    Args:
        x: Starting values. Real-valued array or scalar.
        y: Direction values. Real-valued array or scalar.

    Returns:
        A ``NamedArray`` containing the adjacent representable values, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.nextafter)(x, y)


def not_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether corresponding elements are unequal.

    NaN compares unequal to everything, including itself.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` of booleans containing ``x != y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.not_equal)(x, y)


def pow(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Raise each first-operand element to the power of the second.

    Args:
        x: Base array or scalar.
        y: Exponent array or scalar.

    Returns:
        A ``NamedArray`` containing ``x ** y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.pow)(x, y)


def power(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Alias of :func:`pow`.

    Args:
        x: Base array or scalar.
        y: Exponent array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`pow`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.power)(x, y)


def remainder(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Alias of :func:`mod`.

    Args:
        x: Dividend array or scalar.
        y: Divisor array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`mod`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.remainder)(x, y)


def right_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Alias of :func:`bitwise_right_shift`.

    Args:
        x: Array or scalar of integers to shift.
        y: Array or scalar of non-negative integer shift amounts.

    Returns:
        A ``NamedArray`` containing the result of :func:`bitwise_right_shift`, broadcast to the common axes of ``x``
        and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.right_shift)(x, y)


def subtract(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Subtract the second operand from the first elementwise.

    Args:
        x: First input array or scalar.
        y: Second input array or scalar.

    Returns:
        A ``NamedArray`` containing ``x - y``, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.subtract)(x, y)


def true_divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Alias of :func:`divide`.

    Args:
        x: Dividend array or scalar.
        y: Divisor array or scalar.

    Returns:
        A ``NamedArray`` containing the result of :func:`divide`, broadcast to the common axes of ``x`` and ``y``.
    """
    return wrap_elementwise_binary_op(jnp.true_divide)(x, y)
