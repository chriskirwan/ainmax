import os


os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import pytest
from ainmax._src.axis import Axis
from ainmax._src.feidhm.constructors import array
from hypothesis import HealthCheck, settings


# JIT compilation can comfortably blow past hypothesis's default 200ms deadline -- that's a compile-time cost, not a
# regression, and treating it as one just makes the suite flaky.
settings.register_profile("default", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("ci", deadline=None, max_examples=200)
settings.register_profile("dev", deadline=None, max_examples=25)
settings.load_profile("default")


@pytest.fixture(scope="session")
def mesh_1d():
    """A 1D mesh named ``x`` spanning every (simulated) device."""
    return jax.make_mesh((len(jax.devices()),), ("x",))


@pytest.fixture(scope="session")
def mesh_2d():
    """A 2D ``(x, y)`` mesh, factored as evenly as the device count allows."""
    n = len(jax.devices())
    for x in range(int(n**0.5), 0, -1):
        if n % x == 0:
            return jax.make_mesh((x, n // x), ("x", "y"))
    return jax.make_mesh((1, n), ("x", "y"))


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
