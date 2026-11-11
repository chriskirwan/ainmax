import jax.numpy as jnp
import pytest
from ainmax._src.axis import Axis
from ainmax._src.feipy.constructors import array


@pytest.fixture
def x4():
    return Axis(4, "x")


@pytest.fixture
def y8():
    return Axis(8, "y")


@pytest.fixture
def z16():
    return Axis(16, "z")


@pytest.fixture
def batch3():
    return Axis(3, "batch")


@pytest.fixture
def y5():
    return Axis(5, "y")


@pytest.fixture
def mat():
    return array(jnp.arange(12.0).reshape(3, 4), (Axis(3, "x"), Axis(4, "y")))


@pytest.fixture
def square():
    return array(jnp.arange(9.0).reshape(3, 3), (Axis(3, "x"), Axis(3, "w")))


@pytest.fixture
def named_3d(x4, y8, z16):
    data = jnp.arange(4 * 8 * 16, dtype=jnp.float32).reshape((4, 8, 16))
    return array(data, (x4, y8, z16))


@pytest.fixture
def batch_array(batch3, x4, y5):
    data = jnp.arange(1, batch3.size * x4.size * y5.size + 1, dtype=jnp.float32).reshape(
        (batch3.size, x4.size, y5.size)
    )
    return array(data, (batch3, x4, y5))
