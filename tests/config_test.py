import threading

import ainmax
import jax
import jax.numpy as jnp
import pytest
from ainmax._src.config import LocalState, parse_bool


@pytest.fixture(autouse=True)
def _restore_config():
    saved = ainmax.config.values()
    yield
    for name, value in saved.items():
        ainmax.config.update(name, value)


def test_default_is_short():
    assert ainmax.config.short_arrays is True


def test_context_sets_and_restores():
    before = ainmax.config.short_arrays
    with ainmax.short_arrays(not before):
        assert ainmax.config.short_arrays is (not before)
    assert ainmax.config.short_arrays is before


def test_context_restores_on_exception():
    before = ainmax.config.short_arrays
    with pytest.raises(RuntimeError):
        with ainmax.short_arrays(not before):
            raise RuntimeError
    assert ainmax.config.short_arrays is before


def test_nested_contexts():
    with ainmax.short_arrays(True):
        with ainmax.short_arrays(False):
            assert ainmax.config.short_arrays is False
        assert ainmax.config.short_arrays is True


def test_update_and_attribute_assignment_are_global():
    ainmax.config.update("short_arrays", True)
    assert ainmax.config.short_arrays is True
    ainmax.config.short_arrays = False
    assert ainmax.config.short_arrays is False


def test_context_overrides_global_but_does_not_clobber_it():
    ainmax.config.short_arrays = False
    with ainmax.short_arrays(True):
        ainmax.config.short_arrays = False  # global write inside a context is shadowed by the override
        assert ainmax.config.short_arrays is True
    assert ainmax.config.short_arrays is False


def test_contexts_are_isolated_between_threads():
    seen: list[bool] = []
    ainmax.config.short_arrays = False
    with ainmax.short_arrays(True):
        t = threading.Thread(target=lambda: seen.append(ainmax.config.short_arrays))
        t.start()
        t.join()
    assert seen == [False]


def test_decorator_use():
    @ainmax.short_arrays(True)
    def f() -> bool:
        return ainmax.config.short_arrays

    ainmax.config.short_arrays = False
    assert f() is True
    assert ainmax.config.short_arrays is False


def test_unknown_option_raises():
    with pytest.raises(AttributeError, match="Unknown ainmax config option"):
        ainmax.config.update("verbos_arrays", True)
    with pytest.raises(AttributeError):
        ainmax.config.verbos_arrays = True
    with pytest.raises(AttributeError):
        _ = ainmax.config.nope
    assert not hasattr(ainmax.config, "nope")


def test_bool_validation():
    with pytest.raises(TypeError):
        ainmax.config.short_arrays = 1
    with pytest.raises(TypeError):
        with ainmax.short_arrays("yes"):
            pass


def test_x64_forwards_to_jax_and_is_context_aware():
    with ainmax.enable_x64(False):
        assert ainmax.config.enable_x64 is False
        assert jnp.zeros(()).dtype == jnp.float32
        with ainmax.enable_x64(True):
            assert ainmax.config.enable_x64 is True
            assert jnp.zeros(()).dtype == jnp.float64
        assert ainmax.config.enable_x64 is False


def test_x64_update_is_visible_to_jax():
    ainmax.config.enable_x64 = True
    assert jax.config.jax_enable_x64 is True


def test_override_multiple_options():
    with ainmax.config.override(enable_x64=True, short_arrays=False):
        assert ainmax.config.enable_x64 is True
        assert ainmax.config.short_arrays is False
    assert ainmax.config.short_arrays is True


def test_override_unwinds_on_bad_option():
    before = ainmax.config.short_arrays
    with pytest.raises(AttributeError):
        with ainmax.config.override(short_arrays=not before, nope=1):
            pass
    assert ainmax.config.short_arrays is before


def test_dtype_promotion_and_matmul_precision_contexts():
    with ainmax.numpy_dtype_promotion("strict"):
        assert ainmax.config.numpy_dtype_promotion == "strict"
    with ainmax.default_matmul_precision("highest"):
        assert ainmax.config.default_matmul_precision == "highest"


@pytest.mark.parametrize(("text", "expected"), [("1", True), ("TRUE", True), (" off ", False), ("0", False)])
def test_parse_bool(text, expected):
    assert parse_bool(text) is expected


def test_parse_bool_rejects_garbage():
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_env_var_sets_initial_value(monkeypatch):
    monkeypatch.setenv("LYNQX_TEST_FLAG", "true")
    state = LocalState[bool]("flag", False, validate=lambda v: v, env="LYNQX_TEST_FLAG", parse=parse_bool)
    assert state.get() is True
