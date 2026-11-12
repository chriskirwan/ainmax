import jax.numpy as jnp

from ainmax._src.axis import AxisSelector
from ainmax._src.axis_util import axis_index, match_axes, remove_axes
from ainmax._src.feidhm import constructors, operations, util
from ainmax._src.typing import NamedArray, NamedArrayLike


def trapezoid(
    a: NamedArrayLike,
    x: NamedArrayLike | None = None,
    dx: NamedArrayLike = 1.0,
    axis: AxisSelector = -1,
) -> NamedArray:
    """Integrate along an axis using the composite trapezoidal rule.

    If ``x`` is provided, its values specify the sample coordinates. Otherwise, equally spaced samples with spacing
    ``dx`` are assumed.

    Args:
        a: The array to integrate.
        x: Optional sample coordinates. May be a rank-1 array corresponding to the integration axis, or a
        higher-dimensional array broadcastable to ``a`` that includes the integration axis.
        dx: Spacing between samples when ``x`` is ``None``. May be a scalar or an array broadcastable to the
            remaining axes of ``a``.
        axis: The axis along which to integrate. Defaults to the last axis.

    Returns:
        A ``NamedArray`` with the integration axis removed.
    """

    a = util.ensure_named("trapezoid", a)

    idx = axis_index(a.axes, axis)
    if idx is None:
        raise ValueError(f"trapezoid: axis {axis!r} not found in {a.axes}")

    integrated_axis = a.axes[idx]
    out_axes = remove_axes(a.axes, idx)

    if x is None:
        dx = util.ensure_named("trapezoid", dx)
        dx = operations.broadcast_to(dx, out_axes)

        result = jnp.trapezoid(a.array, axis=idx) * dx.array

    else:
        x = util.ensure_named("trapezoid", x)

        if x.ndim == 0:
            raise ValueError("trapezoid: `x` must contain the integration axis")

        if x.ndim == 1:
            x_axis = x.axes[0]
            if x_axis.size != integrated_axis.size:
                raise ValueError(
                    f"trapezoid: `x` has {x_axis.size} samples, but axis {integrated_axis.label!r} has size "
                    f"{integrated_axis.size}"
                )
            if x_axis.name is not None and x_axis.name != integrated_axis.name:
                raise ValueError(
                    f"trapezoid: `x` axis {x_axis.name!r} does not match integration axis {integrated_axis.name!r}"
                )

            x_array = operations.broadcast_to(x, a.axes)

        else:
            # Match the coordinate array's axes against the input axes.
            matched = match_axes(x.axes, a.axes, allow_positional_fallback=True)
            axis_match = next((m for m in matched.matches if m.target == idx), None)

            if axis_match is None:
                raise ValueError(f"trapezoid: `x` must contain integration axis {integrated_axis.label!r}")

            x_axis = x.axes[axis_match.source]
            if x_axis.size != integrated_axis.size:
                raise ValueError(
                    f"trapezoid: `x` has {x_axis.size} samples along {integrated_axis.label!r}, "
                    f"expected {integrated_axis.size}"
                )
            x_array = operations.broadcast_to(x, a.axes)

        result = jnp.trapezoid(a.array, x=x_array.array, axis=idx)

    return constructors.array(result, out_axes)
