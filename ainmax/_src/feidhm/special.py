from typing import Literal, overload

import jax.nn as jnn
import jax.scipy.special as jspecial

from ainmax._src.axis import AxisLike, AxisSelection
from ainmax._src.feidhm import constructors, operations, reductions, ufuncs, util
from ainmax._src.typing import NamedArray, NamedArrayLike


def gammaln(x: NamedArrayLike) -> NamedArray:
    r"""Compute the natural logarithm of the absolute value of the gamma function.

    .. math::

        \mathrm{gammaln}(x) = \log|\Gamma(x)|

    where :math:`\Gamma` is the gamma function. Unlike :func:`gamma`, this does not overflow for large ``x``. Only
    real inputs are supported; use :func:`loggamma` for complex arguments.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\log|\Gamma(x)|`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.gammaln)(x)


def gammasgn(x: NamedArrayLike) -> NamedArray:
    r"""Return the sign of the gamma function evaluated at each element.

    .. math::

        \mathrm{gammasgn}(x) = \mathrm{sign}(\Gamma(x))

    Together with :func:`gammaln` this recovers the full gamma function, :math:`\Gamma(x) =
    \mathrm{gammasgn}(x) \exp(\mathrm{gammaln}(x))`.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` of signs (``+1`` or ``-1``) with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.gammasgn)(x)


def loggamma(x: NamedArrayLike) -> NamedArray:
    r"""Compute the principal branch of the logarithm of the gamma function.

    .. math::

        \mathrm{loggamma}(x) = \log \Gamma(x)

    This is the analytic continuation of the log-gamma function to the complex plane, with the branch cut along the
    negative real axis. For real positive ``x`` it agrees with :func:`gammaln`; for real negative ``x`` it differs,
    as :func:`gammaln` returns :math:`\log|\Gamma(x)|` instead.

    Args:
        x: Real- or complex-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing the principal branch of :math:`\log \Gamma(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.loggamma)(x)


def gamma(x: NamedArrayLike) -> NamedArray:
    r"""Compute the gamma function for each element.

    The gamma function is defined for :math:`\mathrm{Re}(x) > 0` by

    .. math::

        \Gamma(x) = \int_0^\infty t^{x - 1} e^{-t} \, dt

    and extended to the rest of the real line by analytic continuation. For positive integers,
    :math:`\Gamma(n) = (n - 1)!`. It has poles at zero and the negative integers.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\Gamma(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.gamma)(x)


def betaln(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    r"""Compute the natural logarithm of the beta function.

    .. math::

        \mathrm{betaln}(x, y) = \log B(x, y) = \log \frac{\Gamma(x) \Gamma(y)}{\Gamma(x + y)}

    Working in log space avoids the overflow and underflow that :func:`beta` suffers for large arguments.

    Args:
        x: First input array or scalar. Real-valued and positive.
        y: Second input array or scalar. Real-valued and positive.

    Returns:
        A ``NamedArray`` containing :math:`\log B(x, y)`, broadcast to the common axes of ``x`` and ``y``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.betaln)(x, y)


def factorial(x: NamedArrayLike, exact: bool = False) -> NamedArray:
    r"""Return the factorial of each element.

    For non-negative ``x`` this is :math:`x! = \Gamma(x + 1)`, so non-integer inputs are evaluated through the gamma
    function. Negative inputs give ``0``.

    Args:
        x: Input array or scalar.
        exact: Whether to compute exact integer values. This is not supported by JAX, so ``True`` raises
            ``NotImplementedError``. Defaults to ``False``.

    Returns:
        A ``NamedArray`` containing the factorial of each element, with the same axes as ``x``. The dtype is floating
        point.

    Raises:
        NotImplementedError: If ``exact`` is ``True``.
    """
    return ufuncs.wrap_elementwise_unary_op(lambda value: jspecial.factorial(value, exact=exact))(x)


def comb(N: NamedArrayLike, k: NamedArrayLike, *, repetition: bool = False) -> NamedArray:
    r"""Compute binomial coefficients with optional repeated selections.

    Without repetition this is the number of ways to choose ``k`` items from ``N`` distinct items,

    .. math::

        \binom{N}{k} = \frac{N!}{k! \, (N - k)!}

    With ``repetition=True`` it is the number of ways to choose ``k`` items with replacement,
    :math:`\binom{N + k - 1}{k}`. Non-integer arguments are evaluated through the gamma function.

    Args:
        N: Number of items to choose from.
        k: Number of items to choose.
        repetition: If ``True``, count selections with repetition. Keyword-only. Defaults to ``False``.

    Returns:
        A ``NamedArray`` of binomial coefficients, broadcast to the common axes of ``N`` and ``k``. The dtype is
        floating point.
    """
    return ufuncs.wrap_elementwise_binary_op(lambda a, b, *, repetition: jspecial.comb(a, b, repetition=repetition))(
        N, k, repetition=repetition
    )


def beta(a: NamedArrayLike, b: NamedArrayLike) -> NamedArray:
    r"""Compute the beta function for matching elementwise inputs.

    .. math::

        B(a, b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a + b)} = \int_0^1 t^{a - 1} (1 - t)^{b - 1} \, dt

    Args:
        a: First input array or scalar. Real-valued and positive.
        b: Second input array or scalar. Real-valued and positive.

    Returns:
        A ``NamedArray`` containing :math:`B(a, b)`, broadcast to the common axes of ``a`` and ``b``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.beta)(a, b)


def betainc(a: NamedArrayLike, b: NamedArrayLike, x: NamedArrayLike) -> NamedArray:
    r"""Compute the regularized incomplete beta function.

    .. math::

        I_x(a, b) = \frac{1}{B(a, b)} \int_0^x t^{a - 1} (1 - t)^{b - 1} \, dt

    where :math:`B(a, b)` is the beta function. This is the cumulative distribution function of the beta distribution.

    Args:
        a: Shape parameter. Real-valued and positive.
        b: Shape parameter. Real-valued and positive.
        x: Upper limit of integration. Real-valued, in the range ``[0, 1]``.

    Returns:
        A ``NamedArray`` containing :math:`I_x(a, b)`, broadcast to the common axes of ``a``, ``b`` and ``x``.
    """
    a, b, x = util.ensure_named("betainc", a, b, x)
    (a_arr, b_arr, x_arr), axes = operations.broadcast_arrays_and_shape(a, b, x)
    return constructors.array(jspecial.betainc(a_arr.array, b_arr.array, x_arr.array), axes)


def digamma(x: NamedArrayLike) -> NamedArray:
    r"""Compute the digamma function, the derivative of the log-gamma function.

    .. math::

        \psi(x) = \frac{d}{dx} \log \Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\psi(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.digamma)(x)


def gammainc(a: NamedArrayLike, x: NamedArrayLike) -> NamedArray:
    r"""Compute the regularized lower incomplete gamma function.

    .. math::

        P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a - 1} e^{-t} \, dt

    This is the cumulative distribution function of the gamma distribution with shape ``a`` and unit scale. It
    satisfies :math:`P(a, x) + Q(a, x) = 1`, where :math:`Q` is :func:`gammaincc`.

    Args:
        a: Shape parameter. Real-valued and positive.
        x: Upper limit of integration. Real-valued and non-negative.

    Returns:
        A ``NamedArray`` containing :math:`P(a, x)`, broadcast to the common axes of ``a`` and ``x``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.gammainc)(a, x)


def gammaincc(a: NamedArrayLike, x: NamedArrayLike) -> NamedArray:
    r"""Compute the regularized upper incomplete gamma function.

    .. math::

        Q(a, x) = \frac{1}{\Gamma(a)} \int_x^\infty t^{a - 1} e^{-t} \, dt = 1 - P(a, x)

    where :math:`P` is :func:`gammainc`. Prefer this over ``1 - gammainc(a, x)`` when ``Q`` is small, as it avoids
    cancellation.

    Args:
        a: Shape parameter. Real-valued and positive.
        x: Lower limit of integration. Real-valued and non-negative.

    Returns:
        A ``NamedArray`` containing :math:`Q(a, x)`, broadcast to the common axes of ``a`` and ``x``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.gammaincc)(a, x)


def gammaincinv(a: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    r"""Compute the inverse regularized lower incomplete gamma function.

    Returns the value :math:`z` such that :math:`P(a, z) = x`, where :math:`P` is :func:`gammainc`.

    Args:
        a: Shape parameter. Real-valued and positive.
        y: Target value of :math:`P(a, \cdot)`. Real-valued, in the range ``[0, 1]``.

    Returns:
        A ``NamedArray`` containing the inverse, broadcast to the common axes of ``a`` and ``x``.

    Raises:
        NotImplementedError: If the installed version of JAX does not provide ``jax.scipy.special.gammaincinv``.
    """
    if not hasattr(jspecial, "gammaincinv"):
        raise NotImplementedError("jax.scipy.special.gammaincinv is not available in JAX")
    return ufuncs.wrap_elementwise_binary_op(jspecial.gammaincinv)(a, y)


def gammainccinv(a: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    r"""Compute the inverse regularized upper incomplete gamma function.

    Returns the value :math:`z` such that :math:`Q(a, z) = x`, where :math:`Q` is :func:`gammaincc`.

    Args:
        a: Shape parameter. Real-valued and positive.
        x: Target value of :math:`Q(a, \cdot)`. Real-valued, in the range ``[0, 1]``.

    Returns:
        A ``NamedArray`` containing the inverse, broadcast to the common axes of ``a`` and ``x``.

    Raises:
        NotImplementedError: If the installed version of JAX does not provide ``jax.scipy.special.gammainccinv``.
    """
    if not hasattr(jspecial, "gammainccinv"):
        raise NotImplementedError("jax.scipy.special.gammainccinv is not available in JAX")
    return ufuncs.wrap_elementwise_binary_op(jspecial.gammainccinv)(a, y)


def erf(x: NamedArrayLike) -> NamedArray:
    r"""Compute the error function of each element.

    .. math::

        \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{erf}(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.erf)(x)


def erfc(x: NamedArrayLike) -> NamedArray:
    r"""Compute the complementary error function of each element.

    .. math::

        \mathrm{erfc}(x) = 1 - \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_x^\infty e^{-t^2} \, dt

    Prefer this over ``1 - erf(x)`` for large ``x``, where the subtraction loses precision.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{erfc}(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.erfc)(x)


def erfinv(x: NamedArrayLike) -> NamedArray:
    r"""Compute the inverse error function.

    Returns the value :math:`y` such that :math:`\mathrm{erf}(y) = x`.

    Args:
        x: Real-valued input array or scalar in the range ``[-1, 1]``.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{erf}^{-1}(x)`, with the same axes as ``x``. The result is
        ``-inf`` and ``inf`` at ``-1`` and ``1`` respectively, and ``nan`` outside ``[-1, 1]``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.erfinv)(x)


def erfcinv(x: NamedArrayLike) -> NamedArray:
    r"""Compute the inverse complementary error function.

    Returns the value :math:`y` such that :math:`\mathrm{erfc}(y) = x`, i.e. :math:`\mathrm{erfinv}(1 - x)`.

    Args:
        x: Real-valued input array or scalar in the range ``[0, 2]``.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{erfc}^{-1}(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.erfcinv)(x)


def erfcx(x: NamedArrayLike) -> NamedArray:
    r"""Compute the scaled complementary error function.

    .. math::

        \mathrm{erfcx}(x) = e^{x^2} \, \mathrm{erfc}(x)

    This avoids the underflow of :func:`erfc` for large positive ``x``, where :math:`\mathrm{erfc}(x)` is tiny but
    the scaled value remains well behaved.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{erfcx}(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.erfcx)(x)


def dawsn(x: NamedArrayLike) -> NamedArray:
    r"""Compute the Dawson integral.

    .. math::

        F(x) = e^{-x^2} \int_0^x e^{t^2} \, dt

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`F(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.dawsn)(x)


def logit(x: NamedArrayLike) -> NamedArray:
    r"""Compute the logit function for each element.

    .. math::

        \mathrm{logit}(x) = \log \frac{x}{1 - x}

    This is the inverse of :func:`expit`.

    Args:
        x: Real-valued input array or scalar in the range ``[0, 1]``.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{logit}(x)`, with the same axes as ``x``. The result is ``-inf`` at
        ``0``, ``inf`` at ``1`` and ``nan`` outside ``[0, 1]``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.logit)(x)


def expit(x: NamedArrayLike) -> NamedArray:
    r"""Compute the logistic sigmoid function for each element.

    .. math::

        \mathrm{expit}(x) = \frac{1}{1 + e^{-x}}

    This is the inverse of :func:`logit`.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{expit}(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.expit)(x)


@overload
def logsumexp(
    a: NamedArrayLike,
    axis: AxisSelection | None = None,
    b: NamedArrayLike | None = None,
    keepdims: bool = False,
    return_sign: Literal[False] = False,
    where: NamedArrayLike | None = None,
) -> NamedArray: ...


@overload
def logsumexp(
    a: NamedArrayLike,
    axis: AxisSelection | None = None,
    b: NamedArrayLike | None = None,
    keepdims: bool = False,
    *,
    return_sign: Literal[True],
    where: NamedArrayLike | None = None,
) -> tuple[NamedArray, NamedArray]: ...


def logsumexp(
    a: NamedArrayLike,
    axis: AxisSelection | None = None,
    b: NamedArrayLike | None = None,
    keepdims: bool = False,
    return_sign: bool = False,
    where: NamedArrayLike | None = None,
) -> NamedArray | tuple[NamedArray, NamedArray]:
    r"""Compute the log of the sum of exponentials over the selected axes.

    .. math::

        \mathrm{logsumexp}(a) = \log \sum_i b_i \exp(a_i)

    where the sum runs over the selected axes and :math:`b_i = 1` if ``b`` is not given. The computation is
    stabilized against overflow by subtracting the maximum before exponentiating.

    Args:
        a: Input array.
        axis: The axis or axes to reduce over, by name or position. If ``None``, all axes are reduced.
        b: Optional array of scaling factors for the exponentials, broadcast against ``a``. May contain negative
            values, in which case the sum can be negative; see ``return_sign``.
        keepdims: If ``True``, the reduced axes are retained with size one. Defaults to ``False``.
        return_sign: If ``True``, return a tuple ``(out, sign)`` where ``out`` is the log of the magnitude of the sum
            and ``sign`` is its sign. Without this, a negative sum (possible only when ``b`` has negative entries)
            gives ``nan``. Defaults to ``False``.
        where: Optional boolean array selecting which elements take part in the sum, broadcast against ``a``.

    Returns:
        A ``NamedArray`` containing the log-sum-exp with the reduced axes removed (or kept with size one if
        ``keepdims`` is ``True``). If ``return_sign`` is ``True``, a tuple ``(out, sign)`` of such arrays.
    """

    return reductions.wrap_reduction_call(jspecial.logsumexp, named_array_kwargs=("b",))(
        a, axis, b=b, keepdims=keepdims, return_sign=return_sign, where=where
    )


def xlogy(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    r"""Compute ``x * log(y)`` with special handling for zero inputs.

    The result is defined to be ``0`` wherever ``x == 0``, even if ``y`` is zero or ``nan``-producing under the
    logarithm. This is the safe form of the product that appears in entropy and likelihood calculations.

    Args:
        x: Multiplier. Input array or scalar.
        y: Argument of the logarithm. Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`x \log y`, broadcast to the common axes of ``x`` and ``y``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.xlogy)(x, y)


def xlog1py(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    r"""Compute ``x * log1p(y)`` with special handling for zero inputs.

    The result is defined to be ``0`` wherever ``x == 0``. Using ``log1p`` retains precision for small ``y``.

    Args:
        x: Multiplier. Input array or scalar.
        y: Argument of the logarithm, shifted by one. Input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`x \log(1 + y)`, broadcast to the common axes of ``x`` and ``y``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.xlog1py)(x, y)


def entr(x: NamedArrayLike) -> NamedArray:
    r"""Compute the elementwise entropy function.

    .. math::

        \mathrm{entr}(x) = \begin{cases}
            -x \log x & x > 0 \\
            0 & x = 0 \\
            -\infty & x < 0
        \end{cases}

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{entr}(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.entr)(x)


def boxcox(x: NamedArrayLike, lmbda: NamedArrayLike) -> NamedArray:
    r"""Apply the Box-Cox transform to each element.

    .. math::

        y = \begin{cases}
            \dfrac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\
            \log x & \lambda = 0
        \end{cases}

    Args:
        x: Input array or scalar. Real-valued and positive.
        lmbda: Transform parameter :math:`\lambda`. Real-valued.

    Returns:
        A ``NamedArray`` containing the transformed values, broadcast to the common axes of ``x`` and ``lmbda``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.boxcox)(x, lmbda)


def boxcox1p(x: NamedArrayLike, lmbda: NamedArrayLike) -> NamedArray:
    r"""Apply the Box-Cox transform to ``1 + x`` for each element.

    .. math::

        y = \begin{cases}
            \dfrac{(1 + x)^\lambda - 1}{\lambda} & \lambda \neq 0 \\
            \log(1 + x) & \lambda = 0
        \end{cases}

    This retains precision for small ``x``, compared with calling :func:`boxcox` on ``1 + x``.

    Args:
        x: Input array or scalar. Real-valued and greater than ``-1``.
        lmbda: Transform parameter :math:`\lambda`. Real-valued.

    Returns:
        A ``NamedArray`` containing the transformed values, broadcast to the common axes of ``x`` and ``lmbda``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.boxcox1p)(x, lmbda)


def multigammaln(a: NamedArrayLike, d: NamedArrayLike) -> NamedArray:
    r"""Compute the log of the multivariate gamma function.

    .. math::

        \log \Gamma_d(a) = \frac{d(d - 1)}{4} \log \pi + \sum_{j=1}^{d} \log \Gamma\left(a + \frac{1 - j}{2}\right)

    Args:
        a: Real-valued input array or scalar. Every element must satisfy ``a > (d - 1) / 2``.
        d: Dimension of the multivariate gamma function. Must be a concrete (non-traced) positive integer.

    Returns:
        A ``NamedArray`` containing :math:`\log \Gamma_d(a)`, broadcast to the common axes of ``a`` and ``d``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.multigammaln)(a, d)


def kl_div(p: NamedArrayLike, q: NamedArrayLike) -> NamedArray:
    r"""Compute the elementwise Kullback-Leibler divergence.

    .. math::

        \mathrm{kl\_div}(p, q) = \begin{cases}
            p \log(p / q) - p + q & p > 0, \, q > 0 \\
            q & p = 0, \, q \geq 0 \\
            \infty & \text{otherwise}
        \end{cases}

    The extra ``-p + q`` terms, relative to :func:`rel_entr`, make the result non-negative for all inputs. Summing over
    normalized distributions, they cancel and the total equals the usual KL divergence.

    Args:
        p: First input array or scalar. Real-valued.
        q: Second input array or scalar. Real-valued.

    Returns:
        A ``NamedArray`` containing the elementwise divergence, broadcast to the common axes of ``p`` and ``q``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.kl_div)(p, q)


def rel_entr(p: NamedArrayLike, q: NamedArrayLike) -> NamedArray:
    r"""Compute the elementwise relative entropy.

    .. math::

        \mathrm{rel\_entr}(p, q) = \begin{cases}
            p \log(p / q) & p > 0, \, q > 0 \\
            0 & p = 0, \, q \geq 0 \\
            \infty & \text{otherwise}
        \end{cases}

    Args:
        p: First input array or scalar. Real-valued.
        q: Second input array or scalar. Real-valued.

    Returns:
        A ``NamedArray`` containing the elementwise relative entropy, broadcast to the common axes of ``p`` and
        ``q``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.rel_entr)(p, q)


def wofz(z: NamedArrayLike) -> NamedArray:
    r"""Compute the Faddeeva function for complex arguments.

    .. math::

        w(z) = e^{-z^2} \, \mathrm{erfc}(-i z)

    Args:
        z: Complex-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`w(z)`, with the same axes as ``z``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.wofz)(z)


def zeta(x: NamedArrayLike, q: NamedArrayLike | None = None) -> NamedArray:
    r"""Compute the Hurwitz zeta function, or the Riemann zeta function when ``q`` is omitted.

    .. math::

        \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}

    The Riemann zeta function is the special case :math:`\zeta(x) = \zeta(x, 1)`.

    Args:
        x: Real-valued input array or scalar, with ``x > 1``.
        q: Optional real-valued offset, with ``q > 0``. If ``None``, the Riemann zeta function is computed.

    Returns:
        A ``NamedArray`` containing the zeta function, broadcast to the common axes of ``x`` and ``q`` (or with the
        axes of ``x`` if ``q`` is omitted).
    """
    if q is None:
        return ufuncs.wrap_elementwise_unary_op(lambda value: jspecial.zeta(value, 1.0))(x)
    return ufuncs.wrap_elementwise_binary_op(jspecial.zeta)(x, q)


def polygamma(n: NamedArrayLike, x: NamedArrayLike) -> NamedArray:
    r"""Compute the polygamma function of order ``n``.

    .. math::

        \psi^{(n)}(x) = \frac{d^{n + 1}}{dx^{n + 1}} \log \Gamma(x)

    Order zero is the digamma function, see :func:`digamma`.

    Args:
        n: Order of the derivative. Integer-valued and non-negative.
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\psi^{(n)}(x)`, broadcast to the common axes of ``n`` and ``x``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.polygamma)(n, x)


def ndtr(x: NamedArrayLike) -> NamedArray:
    r"""Compute the standard normal cumulative distribution function.

    .. math::

        \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2 / 2} \, dt
            = \frac{1}{2} \left[1 + \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)\right]

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\Phi(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.ndtr)(x)


def ndtri(p: NamedArrayLike) -> NamedArray:
    r"""Compute the inverse of the standard normal CDF.

    Returns the value :math:`x` such that :math:`\Phi(x) = p`, also known as the probit or quantile function of the
    standard normal distribution.

    Args:
        p: Real-valued input array or scalar of probabilities, in the range ``[0, 1]``.

    Returns:
        A ``NamedArray`` containing :math:`\Phi^{-1}(p)`, with the same axes as ``p``. The result is ``-inf`` at
        ``0``, ``inf`` at ``1`` and ``nan`` outside ``[0, 1]``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.ndtri)(p)


def log_ndtr(x: NamedArrayLike, series_order: int = 3) -> NamedArray:
    r"""Compute the log of the standard normal CDF.

    .. math::

        \log \Phi(x) = \log \left[\frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2 / 2} \, dt\right]

    This stays accurate for very negative ``x``, where :math:`\Phi(x)` underflows, by switching to an asymptotic
    series expansion.

    Args:
        x: Real-valued input array or scalar.
        series_order: Number of terms in the asymptotic series used for very negative ``x``. Must be a static,
            non-negative integer. Defaults to ``3``.

    Returns:
        A ``NamedArray`` containing :math:`\log \Phi(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(lambda value: jspecial.log_ndtr(value, series_order=series_order))(x)


# Bessel functions


def i0e(x: NamedArrayLike) -> NamedArray:
    r"""Compute the exponentially scaled modified Bessel function of order zero.

    .. math::

        \mathrm{i0e}(x) = e^{-|x|} I_0(x)

    where :math:`I_0` is the modified Bessel function of the first kind, see :func:`i0`. The scaling avoids the
    overflow of :func:`i0` for large ``|x|``.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`e^{-|x|} I_0(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.i0e)(x)


def i0(x: NamedArrayLike) -> NamedArray:
    r"""Compute the modified Bessel function of the first kind of order zero.

    .. math::

        I_0(x) = \sum_{k=0}^{\infty} \frac{(x / 2)^{2k}}{(k!)^2}

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`I_0(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.i0)(x)


def i1e(x: NamedArrayLike) -> NamedArray:
    r"""Compute the exponentially scaled modified Bessel function of order one.

    .. math::

        \mathrm{i1e}(x) = e^{-|x|} I_1(x)

    where :math:`I_1` is the modified Bessel function of the first kind, see :func:`i1`. The scaling avoids the
    overflow of :func:`i1` for large ``|x|``.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`e^{-|x|} I_1(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.i1e)(x)


def i1(x: NamedArrayLike) -> NamedArray:
    r"""Compute the modified Bessel function of the first kind of order one.

    .. math::

        I_1(x) = \frac{x}{2} \sum_{k=0}^{\infty} \frac{(x / 2)^{2k}}{k! \, (k + 1)!}

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`I_1(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.i1)(x)


def bessel_jn(z: NamedArrayLike, axis: AxisLike, *, v: int, n_iter: int = 50) -> NamedArray:
    r"""Compute the Bessel functions of the first kind for all integer orders up to ``v``.

    .. math::

        J_n(z) = \frac{1}{2\pi} \int_{-\pi}^{\pi} e^{i (z \sin\theta - n\theta)} \, d\theta, \qquad n = 0, \dots, v

    The values are computed with a recurrence relation, so the result for all orders ``0`` to ``v`` is produced in a
    single call.

    Args:
        z: Real-valued input array or scalar.
        v: Maximum order of the Bessel function. Keyword-only, and must be a static integer.
        n_iter: Number of iterations of the recurrence used to compute the values. Larger values improve accuracy
            for large ``|z|``. Defaults to ``50``.

    Returns:
        A ``NamedArray`` containing :math:`J_0(z), \dots, J_v(z)`.
    """
    z = util.ensure_named("bessel_jn", z)
    out_axis = util.resolve_new_axis(axis, v + 1, "bernoulli", "axis")

    return constructors.array(jspecial.bessel_jn(z.array, v=v, n_iter=n_iter), (out_axis, *z.axes))


# Legendre and spherical harmonics

# LegendreAxes = tuple[AxisLike, AxisLike, AxisLike]
# ALFAxes = LegendreAxes | tuple[LegendreAxes, LegendreAxes]


# def lpmn(m: int, n: int, z: NamedArrayLike, out_axes: ALFAxes | None = None) -> tuple[NamedArray, NamedArray]:
#     """Compute the associated Legendre polynomials and their derivatives."""
#     z = util.ensure_named("lpnm", z)

#     p_vals, p_derivatives = jspecial.lpmn(m, n, z.array)
#     p_val_axes = util.resolve_new_axis(ax)

#     return


# # def lpmn_values(n: int, x: NamedArrayLike) -> tuple[NamedArray, NamedArray]:
# #     """Compute the values of the associated Legendre polynomials."""
# #     x = util.ensure_named("lpnm", x)


def sph_harm_y(
    n: NamedArrayLike,
    m: NamedArrayLike,
    theta: NamedArrayLike,
    phi: NamedArrayLike,
    n_max: int | None = None,
) -> NamedArray:
    r"""Compute spherical harmonics of degree ``n`` and order ``m``.

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n + 1}{4\pi} \frac{(n - m)!}{(n + m)!}} \, P_n^m(\cos\theta) \, e^{i m \phi}

    where :math:`P_n^m` is the associated Legendre function. Here :math:`\theta` is the polar (colatitude) angle and
    :math:`\phi` the azimuthal angle, following the convention of ``scipy.special.sph_harm_y``.

    Args:
        n: Degree of the harmonic. Integer-valued and non-negative.
        m: Order of the harmonic. Integer-valued, with ``|m| <= n``.
        theta: Polar angle in radians, in the range ``[0, pi]``.
        phi: Azimuthal angle in radians, in the range ``[0, 2 * pi]``.
        n_max: Optional static upper bound on the degree ``n``, forwarded to JAX. Must be at least ``max(n)``.

    Returns:
        A complex ``NamedArray`` containing :math:`Y_n^m(\theta, \phi)`, broadcast to the common axes of ``n``, ``m``,
        ``theta`` and ``phi``.
    """
    n, m, theta, phi = util.ensure_named("sph_harm_y", n, m, theta, phi)
    (n_arr, m_arr, theta_arr, phi_arr), axes = operations.broadcast_arrays_and_shape(n, m, theta, phi)
    return constructors.array(
        jspecial.sph_harm_y(n_arr.array, m_arr.array, theta_arr.array, phi_arr.array, n_max=n_max), axes
    )


def expi(x: NamedArrayLike) -> NamedArray:
    r"""Compute the exponential integral function.

    .. math::

        \mathrm{Ei}(x) = -\mathrm{PV} \int_{-x}^{\infty} \frac{e^{-t}}{t} \, dt

    where PV denotes the Cauchy principal value. For negative ``x`` this equals :math:`-E_1(-x)`, see :func:`exp1`.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{Ei}(x)`, with the same axes as ``x``.
    """
    return ufuncs.wrap_elementwise_unary_op(jspecial.expi)(x)


def sici(x: NamedArrayLike) -> tuple[NamedArray, NamedArray]:
    r"""Compute the sine and cosine integrals for each element.

    .. math::

        \mathrm{Si}(x) = \int_0^x \frac{\sin t}{t} \, dt, \qquad
        \mathrm{Ci}(x) = \gamma + \log x + \int_0^x \frac{\cos t - 1}{t} \, dt

    where :math:`\gamma` is the Euler-Mascheroni constant.

    Args:
        x: Real-valued input array or scalar.

    Returns:
        A tuple ``(si, ci)`` of ``NamedArray`` values holding :math:`\mathrm{Si}(x)` and :math:`\mathrm{Ci}(x)`, each
        with the same axes as ``x``.
    """
    named = util.ensure_named("sici", x)
    si, ci = jspecial.sici(named.array)
    return constructors.array(si, named.axes), constructors.array(ci, named.axes)


def expn(n: NamedArrayLike, x: NamedArrayLike) -> NamedArray:
    r"""Compute the generalized exponential integral function.

    .. math::

        E_n(x) = \int_1^{\infty} \frac{e^{-x t}}{t^n} \, dt

    Args:
        n: Order of the integral. Integer-valued and non-negative.
        x: Real-valued argument, with ``x >= 0``.

    Returns:
        A ``NamedArray`` containing :math:`E_n(x)`, broadcast to the common axes of ``n`` and ``x``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.expn)(n, x)


def exp1(x: NamedArrayLike) -> NamedArray:
    r"""Compute the exponential integral function of order one.

    .. math::

        E_1(x) = \int_1^{\infty} \frac{e^{-x t}}{t} \, dt

    This is :func:`expn` with ``n=1``.

    Args:
        x: Real-valued argument, with ``x >= 0``.

    Returns:
        A ``NamedArray`` containing :math:`E_1(x)`, with the same axes as ``x``.
    """

    return expn(1, x)


def spence(x: NamedArrayLike) -> NamedArray:
    r"""Compute Spence's function, also known as the dilogarithm for real values.

    .. math::

        \mathrm{spence}(x) = \int_1^x \frac{\log t}{1 - t} \, dt = \mathrm{Li}_2(1 - x)

    where :math:`\mathrm{Li}_2` is the dilogarithm.

    Args:
        x: Real-valued input array or scalar, with ``x >= 0``.

    Returns:
        A ``NamedArray`` containing :math:`\mathrm{spence}(x)`, with the same axes as ``x``.
    """

    return ufuncs.wrap_elementwise_unary_op(jspecial.spence)(x)


def bernoulli(n: int, axis: AxisLike) -> NamedArray:
    r"""Generate the Bernoulli numbers :math:`B_0` through :math:`B_n`, inclusive.

    The Bernoulli numbers are defined by the generating function

    .. math::

        \frac{t}{e^t - 1} = \sum_{k=0}^{\infty} B_k \frac{t^k}{k!}

    Args:
        n: Highest index to generate. Must be a static non-negative integer.
        axis: The axis of the result. A string creates an axis of that name with size ``n + 1``; an ``Axis`` must have
            size ``n + 1``.

    Returns:
        A one-dimensional ``NamedArray`` holding :math:`B_0, \dots, B_n` along ``axis``.
    """

    out_axis = util.resolve_new_axis(axis, n, "bernoulli", "axis")

    return constructors.array(jspecial.bernoulli(n), out_axis)


def poch(z: NamedArrayLike, m: NamedArrayLike) -> NamedArray:
    r"""Compute the Pochhammer symbol, or rising factorial.

    .. math::

        (z)_m = \frac{\Gamma(z + m)}{\Gamma(z)}

    For non-negative integer ``m`` this is the product :math:`z (z + 1) \cdots (z + m - 1)`.

    Args:
        z: Base of the symbol. Real-valued input array or scalar.
        m: Number of factors. Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`(z)_m`, broadcast to the common axes of ``z`` and ``m``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.poch)(z, m)


def hyp1f1(a: NamedArrayLike, b: NamedArrayLike, x: NamedArrayLike) -> NamedArray:
    r"""Compute the confluent hypergeometric function :math:`{}_1F_1`.

    .. math::

        {}_1F_1(a; b; x) = \sum_{k=0}^{\infty} \frac{(a)_k}{(b)_k} \frac{x^k}{k!}

    where :math:`(a)_k` is the Pochhammer symbol, see :func:`poch`.

    Args:
        a: First parameter. Real-valued input array or scalar.
        b: Second parameter. Real-valued input array or scalar. Must not be a non-positive integer.
        x: Argument. Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`{}_1F_1(a; b; x)`, broadcast to the common axes of ``a``, ``b`` and ``x``.
    """
    (a, b, x), shape = operations.broadcast_arrays_and_shape(a, b, x)
    jax_hyp1f1 = jspecial.hyp1f1(a.array, b.array, x.array)

    return constructors.array(jax_hyp1f1, shape)


def hyp2f1(a: NamedArrayLike, b: NamedArrayLike, c: NamedArrayLike, x: NamedArrayLike) -> NamedArray:
    r"""Compute the Gauss hypergeometric function :math:`{}_2F_1`.

    .. math::

        {}_2F_1(a, b; c; x) = \sum_{k=0}^{\infty} \frac{(a)_k (b)_k}{(c)_k} \frac{x^k}{k!}

    where :math:`(a)_k` is the Pochhammer symbol, see :func:`poch`.

    Args:
        a: First numerator parameter. Real-valued input array or scalar.
        b: Second numerator parameter. Real-valued input array or scalar.
        c: Denominator parameter. Real-valued input array or scalar. Must not be a non-positive integer.
        x: Argument. Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`{}_2F_1(a, b; c; x)`, broadcast to the common axes of ``a``, ``b``, ``c``
        and ``x``.
    """
    (a, b, c, x), shape = operations.broadcast_arrays_and_shape(a, b, c, x)
    jax_hyp2f1 = jspecial.hyp2f1(a.array, b.array, c.array, x.array)

    return constructors.array(jax_hyp2f1, shape)


def owens_t(h: NamedArrayLike, a: NamedArrayLike) -> NamedArray:
    r"""Compute Owen's T function.

    .. math::

        T(h, a) = \frac{1}{2\pi} \int_0^a \frac{e^{-\frac{1}{2} h^2 (1 + x^2)}}{1 + x^2} \, dx

    Args:
        h: Real-valued input array or scalar.
        a: Real-valued input array or scalar.

    Returns:
        A ``NamedArray`` containing :math:`T(h, a)`, broadcast to the common axes of ``h`` and ``a``.
    """
    return ufuncs.wrap_elementwise_binary_op(jspecial.owens_t)(h, a)


def softmax(x: NamedArrayLike, axis: AxisSelection, *, where: NamedArrayLike | None = None) -> NamedArray:
    r"""Compute the softmax function along the selected axes.

    .. math::

        \mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}

    where the sum runs over the selected axes. The computation is stabilized against overflow by subtracting the
    maximum before exponentiating.

    Args:
        x: Input array.
        axis: The axis or axes to normalize over, by name or position.
        where: Optional boolean array selecting which elements take part in the normalization, broadcast against
            ``x``. Masked-out elements have probability zero.

    Returns:
        A ``NamedArray`` with the same axes as ``x``, whose values are non-negative and sum to one over ``axis``.
    """
    return reductions.wrap_axiswise_call(jnn.softmax)(x, axis, where=where)


def log_softmax(x: NamedArrayLike, axis: AxisSelection, *, where: NamedArrayLike | None = None) -> NamedArray:
    r"""Compute the log of the softmax function along the selected axes.

    .. math::

        \mathrm{log\_softmax}(x)_i = x_i - \log \sum_j e^{x_j}

    where the sum runs over the selected axes. This is more accurate than taking the log of :func:`softmax`.

    Args:
        x: Input array.
        axis: The axis or axes to normalize over, by name or position.
        where: Optional boolean array selecting which elements take part in the normalization, broadcast against
            ``x``. Masked-out elements are ``-inf`` in the output.

    Returns:
        A ``NamedArray`` with the same axes as ``x`` containing the log-probabilities.
    """
    return reductions.wrap_axiswise_call(jnn.log_softmax)(x, axis, where=where)
