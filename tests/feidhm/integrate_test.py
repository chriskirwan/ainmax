import jax.numpy as jnp
import pytest
from ainmax._src.axis import Axis
from ainmax._src.feidhm.constructors import array
from ainmax._src.feidhm.integrate import trapezoid


class TestTrapezoid:
    def test_named_coordinate_vector_and_named_axis(self, batch_array):
        a = batch_array
        x = array(jnp.asarray([0.0, 0.5, 2.0, 4.0]), (Axis(4, "x"),))

        got = trapezoid(a, x=x, axis="x")
        want = jnp.trapezoid(a.array, x=x.array, axis=1)

        assert got.axes == (Axis(3, "batch"), Axis(5, "y"))
        assert jnp.allclose(got.array, want)

    def test_default_coordinates_and_named_dx(self, batch_array):
        a = batch_array
        dx = array(jnp.asarray(2.0), ())

        got = trapezoid(a, dx=dx, axis=1)
        want = jnp.trapezoid(a.array, dx=2.0, axis=1)

        assert got.axes == (Axis(3, "batch"), Axis(5, "y"))
        assert jnp.allclose(got.array, want)

    def test_full_named_coordinates_are_aligned_to_input_axes(self, batch_array):
        a = batch_array
        x_data = jnp.broadcast_to(jnp.arange(4.0)[None, :, None], a.shape)
        x = array(x_data, (Axis(3, "batch"), Axis(4, "x"), Axis(5, "y")))

        got = trapezoid(a, x=x, axis="x")
        want = jnp.trapezoid(a.array, x=x_data, axis=1)

        assert jnp.allclose(got.array, want)

    def test_scalar_coordinate_keeps_jax_validation(self):
        x = array(jnp.asarray(2.0), ())

        with pytest.raises(ValueError, match="must contain the integration axis"):
            trapezoid(array(jnp.arange(4.0), (Axis(4, "x"),)), x=x, axis="x")

    def test_unresolved_axis_raises(self, batch_array):
        with pytest.raises(ValueError, match="axis 'missing' not found"):
            trapezoid(batch_array, axis="missing")
