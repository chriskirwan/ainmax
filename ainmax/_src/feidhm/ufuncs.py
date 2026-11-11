from collections.abc import Callable
from functools import wraps

import jax.lax as lax
import jax.numpy as jnp
from jax import Array

from ainmax._src.feidhm import constructors, operations, util
from ainmax._src.typing import NamedArray, NamedArrayLike


def wrap_elementwise_unary_op(fn: Callable[..., Array]) -> Callable[..., NamedArray]:
    """Wrap a unary JAX function to preserve named axes."""

    @wraps(fn)
    def wrapped(x: NamedArrayLike, *args, **kwargs) -> NamedArray:
        x = util.ensure_named("wrap_elementwise_unary", x)
        return constructors.array(fn(x.array, *args, **kwargs), x.axes)

    return wrapped


def wrap_elementwise_binary_op(fn: Callable[..., Array]) -> Callable[..., NamedArray]:
    """Wrap a binary JAX function with named-axis broadcasting."""

    @wraps(fn)
    def wrapped(x: NamedArrayLike, y: NamedArrayLike, *args, **kwargs):
        x, y = util.ensure_named("wrap_elementwise_binary_op", x, y)
        (a, b), axes = operations.broadcast_arrays_and_shape(x, y)
        return constructors.array(fn(a.array, b.array, *args, **kwargs), axes)

    return wrapped


def abs(x: NamedArrayLike) -> NamedArray:
    """Compute the absolute value of each element."""
    return wrap_elementwise_unary_op(jnp.abs)(x)


def absolute(x: NamedArrayLike) -> NamedArray:
    """Compute the absolute value of each element."""
    return wrap_elementwise_unary_op(jnp.absolute)(x)


def acos(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse cosine of each element, in radians."""
    return wrap_elementwise_unary_op(jnp.acos)(x)


def acosh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic cosine of each element."""
    return wrap_elementwise_unary_op(jnp.acosh)(x)


def arccos(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse cosine of each element, in radians."""
    return wrap_elementwise_unary_op(jnp.arccos)(x)


def arccosh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic cosine of each element."""
    return wrap_elementwise_unary_op(jnp.arccosh)(x)


def arcsin(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse sine of each element, in radians."""
    return wrap_elementwise_unary_op(jnp.arcsin)(x)


def arcsinh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic sine of each element."""
    return wrap_elementwise_unary_op(jnp.arcsinh)(x)


def arctan(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse tangent of each element, in radians."""
    return wrap_elementwise_unary_op(jnp.arctan)(x)


def arctanh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic tangent of each element."""
    return wrap_elementwise_unary_op(jnp.arctanh)(x)


def asin(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse sine of each element, in radians."""
    return wrap_elementwise_unary_op(jnp.asin)(x)


def asinh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic sine of each element."""
    return wrap_elementwise_unary_op(jnp.asinh)(x)


def atan(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse tangent of each element, in radians."""
    return wrap_elementwise_unary_op(jnp.atan)(x)


def atanh(x: NamedArrayLike) -> NamedArray:
    """Compute the inverse hyperbolic tangent of each element."""
    return wrap_elementwise_unary_op(jnp.atanh)(x)


def bitwise_invert(x: NamedArrayLike) -> NamedArray:
    """Invert the bits of each integer element."""
    return wrap_elementwise_unary_op(jnp.bitwise_invert)(x)


def bitwise_count(x: NamedArrayLike, /) -> NamedArray:
    """Count the set bits in each integer element."""
    return wrap_elementwise_unary_op(jnp.bitwise_count)(x)


def bitwise_not(x: NamedArrayLike) -> NamedArray:
    """Invert the bits of each integer element."""
    return wrap_elementwise_unary_op(jnp.bitwise_not)(x)


def cbrt(x: NamedArrayLike) -> NamedArray:
    """Compute the cube root of each element."""
    return wrap_elementwise_unary_op(jnp.cbrt)(x)


def ceil(x: NamedArrayLike) -> NamedArray:
    """Round each element up to the nearest integer."""
    return wrap_elementwise_unary_op(jnp.ceil)(x)


def conj(x: NamedArrayLike) -> NamedArray:
    """Compute the complex conjugate of each element."""
    return wrap_elementwise_unary_op(jnp.conj)(x)


def conjugate(x: NamedArrayLike) -> NamedArray:
    """Compute the complex conjugate of each element."""
    return wrap_elementwise_unary_op(jnp.conjugate)(x)


def cos(x: NamedArrayLike) -> NamedArray:
    """Compute the cosine of each element, interpreted in radians."""
    return wrap_elementwise_unary_op(jnp.cos)(x)


def cosh(x: NamedArrayLike) -> NamedArray:
    """Compute the hyperbolic cosine of each element."""
    return wrap_elementwise_unary_op(jnp.cosh)(x)


def deg2rad(x: NamedArrayLike) -> NamedArray:
    """Convert each angle from degrees to radians."""
    return wrap_elementwise_unary_op(jnp.deg2rad)(x)


def degrees(x: NamedArrayLike) -> NamedArray:
    """Convert each angle from radians to degrees."""
    return wrap_elementwise_unary_op(jnp.degrees)(x)


def exp(x: NamedArrayLike) -> NamedArray:
    """Compute the exponential of each element."""
    return wrap_elementwise_unary_op(jnp.exp)(x)


def exp2(x: NamedArrayLike) -> NamedArray:
    """Compute two raised to the power of each element."""
    return wrap_elementwise_unary_op(jnp.exp2)(x)


def expm1(x: NamedArrayLike) -> NamedArray:
    """Compute the exponential of each element minus one."""
    return wrap_elementwise_unary_op(jnp.expm1)(x)


def fabs(x: NamedArrayLike) -> NamedArray:
    """Compute the absolute value of each element as a floating-point value."""
    return wrap_elementwise_unary_op(jnp.fabs)(x)


def floor(x: NamedArrayLike) -> NamedArray:
    """Round each element down to the nearest integer."""
    return wrap_elementwise_unary_op(jnp.floor)(x)


def frexp(x: NamedArrayLike) -> tuple[NamedArray, NamedArray]:
    """Decompose each element into a mantissa and a base-two exponent."""
    x = util.ensure_named("frexp", x)
    m, e = jnp.frexp(x.array)
    return constructors.array(m, x.axes), constructors.array(e, x.axes)


def i0(x: NamedArrayLike) -> NamedArray:
    """Compute the modified Bessel function of the first kind of order zero."""
    return wrap_elementwise_unary_op(jnp.i0)(x)


def imag(x: NamedArrayLike) -> NamedArray:
    """Return the imaginary part of each element."""
    return wrap_elementwise_unary_op(jnp.imag)(x)


def invert(x: NamedArrayLike) -> NamedArray:
    """Invert the bits of each integer element."""
    return wrap_elementwise_unary_op(jnp.invert)(x)


def isfinite(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is finite."""
    return wrap_elementwise_unary_op(jnp.isfinite)(x)


def isinf(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is infinite."""
    return wrap_elementwise_unary_op(jnp.isinf)(x)


def isnan(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is NaN."""
    return wrap_elementwise_unary_op(jnp.isnan)(x)


def isneginf(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is negative infinity."""
    return wrap_elementwise_unary_op(jnp.isneginf)(x)


def isposinf(x: NamedArrayLike) -> NamedArray:
    """Test whether each element is positive infinity."""
    return wrap_elementwise_unary_op(jnp.isposinf)(x)


def log(x: NamedArrayLike) -> NamedArray:
    """Compute the natural logarithm of each element."""
    return wrap_elementwise_unary_op(jnp.log)(x)


def log10(x: NamedArrayLike) -> NamedArray:
    """Compute the base-10 logarithm of each element."""
    return wrap_elementwise_unary_op(jnp.log10)(x)


def log1p(x: NamedArrayLike) -> NamedArray:
    """Compute the natural logarithm of one plus each element."""
    return wrap_elementwise_unary_op(jnp.log1p)(x)


def log2(x: NamedArrayLike) -> NamedArray:
    """Compute the base-two logarithm of each element."""
    return wrap_elementwise_unary_op(jnp.log2)(x)


def logical_not(x: NamedArrayLike) -> NamedArray:
    """Compute the logical negation of each element."""
    return wrap_elementwise_unary_op(jnp.logical_not)(x)


def modf(x: NamedArrayLike, /) -> tuple[NamedArray, NamedArray]:
    """Split each element into its fractional and integral parts."""
    x = util.ensure_named("modf", x)
    out0, out1 = jnp.modf(x.array)
    return constructors.array(out0, x.axes), constructors.array(out1, x.axes)


def negative(x: NamedArrayLike) -> NamedArray:
    """Compute the numerical negative of each element."""
    return wrap_elementwise_unary_op(jnp.negative)(x)


def positive(x: NamedArrayLike) -> NamedArray:
    """Return each element unchanged."""
    return wrap_elementwise_unary_op(jnp.positive)(x)


def rad2deg(x: NamedArrayLike) -> NamedArray:
    """Convert each angle from radians to degrees."""
    return wrap_elementwise_unary_op(jnp.rad2deg)(x)


def radians(x: NamedArrayLike) -> NamedArray:
    """Convert each angle from degrees to radians."""
    return wrap_elementwise_unary_op(jnp.radians)(x)


def real(x: NamedArrayLike) -> NamedArray:
    """Return the real part of each element."""
    return wrap_elementwise_unary_op(jnp.real)(x)


def reciprocal(x: NamedArrayLike) -> NamedArray:
    """Compute the reciprocal of each element."""
    return wrap_elementwise_unary_op(jnp.reciprocal)(x)


def rint(x: NamedArrayLike) -> NamedArray:
    """Round each element to the nearest integer, preserving its dtype."""
    return wrap_elementwise_unary_op(jnp.rint)(x)


def rsqrt(x: NamedArrayLike) -> NamedArray:
    """Compute the reciprocal of the square root of each element."""
    return wrap_elementwise_unary_op(lax.rsqrt)(x)


def sign(x: NamedArrayLike) -> NamedArray:
    """Compute the sign of each element."""
    return wrap_elementwise_unary_op(jnp.sign)(x)


def signbit(x: NamedArrayLike) -> NamedArray:
    """Test whether the sign bit is set for each element."""
    return wrap_elementwise_unary_op(jnp.signbit)(x)


def sin(x: NamedArrayLike) -> NamedArray:
    """Compute the sine of each element, interpreted in radians."""
    return wrap_elementwise_unary_op(jnp.sin)(x)


def sinc(x: NamedArrayLike) -> NamedArray:
    """Compute the normalized sinc function elementwise."""
    return wrap_elementwise_unary_op(jnp.sinc)(x)


def sinh(x: NamedArrayLike) -> NamedArray:
    """Compute the hyperbolic sine of each element."""
    return wrap_elementwise_unary_op(jnp.sinh)(x)


def sort_complex(x: NamedArrayLike) -> NamedArray:
    """Sort complex values by real part, then by imaginary part."""
    return wrap_elementwise_unary_op(jnp.sort_complex)(x)


def spacing(x: NamedArrayLike, /) -> NamedArray:
    """Compute the distance to the adjacent representable value."""
    return wrap_elementwise_unary_op(jnp.spacing)(x)


def square(x: NamedArrayLike) -> NamedArray:
    """Compute the square of each element."""
    return wrap_elementwise_unary_op(jnp.square)(x)


def sqrt(x: NamedArrayLike) -> NamedArray:
    """Compute the non-negative square root of each element."""
    return wrap_elementwise_unary_op(jnp.sqrt)(x)


def tan(x: NamedArrayLike) -> NamedArray:
    """Compute the tangent of each element, interpreted in radians."""
    return wrap_elementwise_unary_op(jnp.tan)(x)


def tanh(x: NamedArrayLike) -> NamedArray:
    """Compute the hyperbolic tangent of each element."""
    return wrap_elementwise_unary_op(jnp.tanh)(x)


def trunc(x: NamedArrayLike) -> NamedArray:
    """Round each element toward zero."""
    return wrap_elementwise_unary_op(jnp.trunc)(x)


# Binary Elementwise Operations
def add(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Add corresponding elements."""
    return wrap_elementwise_binary_op(jnp.add)(x, y)


def atan2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the elementwise four-quadrant inverse tangent, in radians."""
    return wrap_elementwise_binary_op(jnp.atan2)(x, y)


def arctan2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the elementwise four-quadrant inverse tangent, in radians."""
    return wrap_elementwise_binary_op(jnp.arctan2)(x, y)


def bitwise_and(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the bitwise AND of corresponding integer elements."""
    return wrap_elementwise_binary_op(jnp.bitwise_and)(x, y)


def bitwise_left_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Shift the bits of each first operand left by the second operand."""
    return wrap_elementwise_binary_op(jnp.bitwise_left_shift)(x, y)


def bitwise_or(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the bitwise OR of corresponding integer elements."""
    return wrap_elementwise_binary_op(jnp.bitwise_or)(x, y)


def bitwise_right_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Shift the bits of each first operand right by the second operand."""
    return wrap_elementwise_binary_op(jnp.bitwise_right_shift)(x, y)


def bitwise_xor(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the bitwise XOR of corresponding integer elements."""
    return wrap_elementwise_binary_op(jnp.bitwise_xor)(x, y)


def copysign(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the magnitude of the first operand with the sign of the second."""
    return wrap_elementwise_binary_op(jnp.copysign)(x, y)


def divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Divide corresponding elements."""
    return wrap_elementwise_binary_op(jnp.divide)(x, y)


def divmod(x: NamedArrayLike, y: NamedArrayLike, /) -> tuple[NamedArray, NamedArray]:
    """Return the elementwise quotient and remainder."""
    a, b = util.ensure_named("divmod", x, y)
    _, _, axes = util.align_shapes_for_broadcast(a.axes, b.axes)
    a_array, b_array = jnp.divmod(a.array, b.array)
    return constructors.array(a_array, axes), constructors.array(b_array, axes)


def equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether corresponding elements are equal."""
    return wrap_elementwise_binary_op(jnp.equal)(x, y)


def float_power(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Raise the first operand to the power of the second using floating-point arithmetic."""
    return wrap_elementwise_binary_op(jnp.float_power)(x, y)


def floor_divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Divide corresponding elements and round the result down."""
    return wrap_elementwise_binary_op(jnp.floor_divide)(x, y)


def fmod(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the remainder from division, with the sign of the dividend."""
    return wrap_elementwise_binary_op(jnp.fmod)(x, y)


def gcd(a: NamedArrayLike, b: NamedArrayLike, /) -> NamedArray:
    """Compute the greatest common divisor of corresponding integer elements."""
    return wrap_elementwise_binary_op(jnp.gcd)(a, b)


def greater(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether each first-operand element is greater than the corresponding second element."""
    return wrap_elementwise_binary_op(jnp.greater)(x, y)


def greater_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether each first-operand element is greater than or equal to the corresponding second element."""
    return wrap_elementwise_binary_op(jnp.greater_equal)(x, y)


def heaviside(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    """Evaluate the Heaviside step function using the second operand at zero."""
    return wrap_elementwise_binary_op(jnp.heaviside)(x, y)


def hypot(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the hypotenuse from corresponding elements."""
    return wrap_elementwise_binary_op(jnp.hypot)(x, y)


def lcm(a: NamedArrayLike, b: NamedArrayLike, /) -> NamedArray:
    """Compute the least common multiple of corresponding integer elements."""
    return wrap_elementwise_binary_op(jnp.lcm)(a, b)


def ldexp(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    """Multiply each first-operand element by two raised to the second operand."""
    return wrap_elementwise_binary_op(jnp.ldexp)(x, y)


def left_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Shift the bits of each first operand left by the second operand."""
    return wrap_elementwise_binary_op(jnp.left_shift)(x, y)


def less(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether each first-operand element is less than the corresponding second element."""
    return wrap_elementwise_binary_op(jnp.less)(x, y)


def less_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether each first-operand element is less than or equal to the corresponding second element."""
    return wrap_elementwise_binary_op(jnp.less_equal)(x, y)


def logaddexp(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the logarithm of the sum of exponentials, using base e."""
    return wrap_elementwise_binary_op(jnp.logaddexp)(x, y)


def logaddexp2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the logarithm of the sum of powers of two."""
    return wrap_elementwise_binary_op(jnp.logaddexp2)(x, y)


def logical_and(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the logical AND of corresponding elements."""
    return wrap_elementwise_binary_op(jnp.logical_and)(x, y)


def logical_or(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the logical OR of corresponding elements."""
    return wrap_elementwise_binary_op(jnp.logical_or)(x, y)


def logical_xor(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the logical XOR of corresponding elements."""
    return wrap_elementwise_binary_op(jnp.logical_xor)(x, y)


def maximum(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the elementwise maximum, propagating NaNs."""
    return wrap_elementwise_binary_op(jnp.maximum)(x, y)


def minimum(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Compute the elementwise minimum, propagating NaNs."""
    return wrap_elementwise_binary_op(jnp.minimum)(x, y)


def mod(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the remainder from division, with the sign of the divisor."""
    return wrap_elementwise_binary_op(jnp.mod)(x, y)


def multiply(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Multiply corresponding elements."""
    return wrap_elementwise_binary_op(jnp.multiply)(x, y)


def nextafter(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the next representable value from the first operand toward the second."""
    return wrap_elementwise_binary_op(jnp.nextafter)(x, y)


def not_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Test whether corresponding elements are unequal."""
    return wrap_elementwise_binary_op(jnp.not_equal)(x, y)


def pow(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Raise each first-operand element to the power of the second."""
    return wrap_elementwise_binary_op(jnp.pow)(x, y)


def power(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Raise each first-operand element to the power of the second."""
    return wrap_elementwise_binary_op(jnp.power)(x, y)


def remainder(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Return the remainder from division, with the sign of the divisor."""
    return wrap_elementwise_binary_op(jnp.remainder)(x, y)


def right_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Shift the bits of each first operand right by the second operand."""
    return wrap_elementwise_binary_op(jnp.right_shift)(x, y)


def subtract(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Subtract the second operand from the first elementwise."""
    return wrap_elementwise_binary_op(jnp.subtract)(x, y)


def true_divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    """Divide corresponding elements using true-division semantics."""
    return wrap_elementwise_binary_op(jnp.true_divide)(x, y)
