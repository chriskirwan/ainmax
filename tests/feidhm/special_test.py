from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial
import jax.test_util as jtu
import numpy as np
import pytest
from ainmax._src.array import NamedArrayImpl
from ainmax._src.axis import Axis
from ainmax._src.feidhm import special


jax.config.update("jax_enable_x64", True)

FLOAT_DTYPES = (jnp.float32, jnp.float64)
DEFAULT_TOL = {jnp.float32: 1e-5, jnp.float64: 1e-12}
AXIS_SHAPE = (("batch", 3), ("x", 4))


@dataclass(frozen=True)
class SpecialUnaryRecord:
    name: str
    domain: tuple[float, float] = (-2.0, 2.0)
    dtypes: tuple = FLOAT_DTYPES
    tol: dict = field(default_factory=lambda: dict(DEFAULT_TOL))
    test_grad: bool = True


@dataclass(frozen=True)
class SpecialBinaryRecord:
    name: str
    domain: tuple[float, float] = (-2.0, 2.0)
    dtypes: tuple = FLOAT_DTYPES
    tol: dict = field(default_factory=lambda: dict(DEFAULT_TOL))
    test_grad: bool = True


UNARY_RECORDS = [
    SpecialUnaryRecord("gammaln", domain=(0.5, 5.0)),
    SpecialUnaryRecord("gammasgn", domain=(0.5, 5.0), test_grad=False),
    SpecialUnaryRecord("loggamma", domain=(0.5, 5.0)),
    SpecialUnaryRecord("gamma", domain=(0.5, 5.0)),
    SpecialUnaryRecord("digamma", domain=(0.5, 5.0)),
    SpecialUnaryRecord("erf", domain=(-3.0, 3.0)),
    SpecialUnaryRecord("erfc", domain=(-3.0, 3.0)),
    SpecialUnaryRecord("erfcx", domain=(-3.0, 3.0)),
    SpecialUnaryRecord("dawsn", domain=(-3.0, 3.0)),
    SpecialUnaryRecord("logit", domain=(0.1, 0.9)),
    SpecialUnaryRecord("expit", domain=(-4.0, 4.0)),
    SpecialUnaryRecord("entr", domain=(0.1, 3.0)),
    SpecialUnaryRecord("ndtr", domain=(-4.0, 4.0)),
    SpecialUnaryRecord("ndtri", domain=(0.01, 0.99)),
    SpecialUnaryRecord("i0", domain=(-5.0, 5.0)),
    SpecialUnaryRecord("i1", domain=(-5.0, 5.0)),
]

BINARY_RECORDS = [
    SpecialBinaryRecord("betaln", domain=(0.5, 3.0)),
    SpecialBinaryRecord("beta", domain=(0.5, 3.0)),
    SpecialBinaryRecord("xlogy", domain=(0.1, 2.0)),
    SpecialBinaryRecord("xlog1py", domain=(0.1, 2.0)),
    SpecialBinaryRecord("boxcox", domain=(0.2, 2.0)),
    SpecialBinaryRecord("boxcox1p", domain=(0.1, 2.0)),
    SpecialBinaryRecord("kl_div", domain=(0.2, 2.0)),
    SpecialBinaryRecord("rel_entr", domain=(0.2, 2.0)),
    SpecialBinaryRecord("owens_t", domain=(-3.0, 3.0)),
    SpecialBinaryRecord("gammainc", domain=(0.5, 3.0)),
    SpecialBinaryRecord("gammaincc", domain=(0.5, 3.0)),
]


def _sample(key: jax.Array, shape: tuple[int, ...], dtype, domain: tuple[float, float]):
    if jnp.issubdtype(dtype, jnp.integer):
        lo, hi = int(domain[0]), int(domain[1])
        return jax.random.randint(key, shape, lo, hi + 1, dtype=dtype)
    lo, hi = domain
    return jax.random.uniform(key, shape, minval=lo, maxval=hi, dtype=dtype)


def _make_named(key: jax.Array, dtype, domain: tuple[float, float]) -> NamedArrayImpl:
    axes = tuple(Axis(size, name) for name, size in AXIS_SHAPE)
    shape = tuple(size for _, size in AXIS_SHAPE)
    return NamedArrayImpl(_sample(key, shape, dtype, domain), axes)


def _tol_for(record, dtype) -> float:
    return record.tol.get(dtype, 1e-6)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("record", UNARY_RECORDS, ids=lambda r: r.name)
def test_unary_special_matches_reference(record: SpecialUnaryRecord, dtype) -> None:
    fn = getattr(special, record.name)
    ref = getattr(jspecial, record.name)

    x = _make_named(jax.random.key(0), dtype, record.domain)
    got = fn(x)
    want = ref(x.array)

    tol = _tol_for(record, dtype)
    np.testing.assert_allclose(got.array, want, rtol=tol, atol=tol)
    assert got.axes == x.axes


@pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("record", BINARY_RECORDS, ids=lambda r: r.name)
def test_binary_special_matches_reference(record: SpecialBinaryRecord, dtype) -> None:
    fn = getattr(special, record.name)
    ref = getattr(jspecial, record.name)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = _make_named(k1, dtype, record.domain)
    y = _make_named(k2, dtype, record.domain)

    got = fn(x, y)
    want = ref(x.array, y.array)

    tol = _tol_for(record, dtype)
    np.testing.assert_allclose(got.array, want, rtol=tol, atol=tol)
    assert got.axes == tuple(sorted(set(x.axes) | set(y.axes), key=lambda a: a.name))


@pytest.mark.parametrize(
    "record",
    [r for r in UNARY_RECORDS if r.test_grad and jnp.float64 in r.dtypes],
    ids=lambda r: r.name,
)
def test_unary_special_grad(record: SpecialUnaryRecord) -> None:
    x = _make_named(jax.random.key(1), jnp.float64, record.domain)

    def f(arr):
        return getattr(special, record.name)(NamedArrayImpl(arr, x.axes)).array.sum()

    jtu.check_grads(f, (x.array,), order=1, modes=["rev"])


@pytest.mark.parametrize(
    "record",
    [r for r in BINARY_RECORDS if r.test_grad and jnp.float64 in r.dtypes],
    ids=lambda r: r.name,
)
def test_binary_special_grad(record: SpecialBinaryRecord) -> None:
    k1, k2 = jax.random.split(jax.random.key(1))
    x = _make_named(k1, jnp.float64, record.domain)
    y = _make_named(k2, jnp.float64, record.domain)

    def f(a, b):
        return getattr(special, record.name)(NamedArrayImpl(a, x.axes), NamedArrayImpl(b, y.axes)).array.sum()

    jtu.check_grads(f, (x.array, y.array), order=1, modes=["rev"])


def test_logsumexp_matches_reference() -> None:
    x = NamedArrayImpl(jnp.arange(12.0, dtype=jnp.float64).reshape(3, 4), (Axis(3, "batch"), Axis(4, "x")))
    b = NamedArrayImpl(jnp.linspace(1.0, 2.0, 4, dtype=jnp.float64), (Axis(4, "x"),))

    got = special.logsumexp(x, axis="x", b=b)
    want = jspecial.logsumexp(x.array, axis=1, b=b.array)
    np.testing.assert_allclose(got.array, want)
    assert got.axes == (Axis(3, "batch"),)

    out, sign = special.logsumexp(x, axis="x", b=b, return_sign=True)
    want_out, want_sign = jspecial.logsumexp(x.array, axis=1, b=b.array, return_sign=True)
    np.testing.assert_allclose(out.array, want_out)
    np.testing.assert_allclose(sign.array, want_sign)
