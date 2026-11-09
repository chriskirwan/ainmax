import threading

import jax
import jax.numpy as jnp
import lynqx
import pytest
from lynqx._src.config import LocalState, parse_bool


@pytest.fixture(autouse=True)
def _restore_config():
    saved = lynqx.config.values()
    yield
    for name, value in saved.items():
        lynqx.config.update(name, value)


def test_default_is_short():
    assert lynqx.config.short_arrays is True


def test_context_sets_and_restores():
    before = lynqx.config.short_arrays
    with lynqx.short_arrays(not before):
        assert lynqx.config.short_arrays is (not before)
    assert lynqx.config.short_arrays is before


def test_context_restores_on_exception():
    before = lynqx.config.short_arrays
    with pytest.raises(RuntimeError):
        with lynqx.short_arrays(not before):
            raise RuntimeError
    assert lynqx.config.short_arrays is before


def test_nested_contexts():
    with lynqx.short_arrays(True):
        with lynqx.short_arrays(False):
            assert lynqx.config.short_arrays is False
        assert lynqx.config.short_arrays is True


def test_update_and_attribute_assignment_are_global():
    lynqx.config.update("short_arrays", True)
    assert lynqx.config.short_arrays is True
    lynqx.config.short_arrays = False
    assert lynqx.config.short_arrays is False


def test_context_overrides_global_but_does_not_clobber_it():
    lynqx.config.short_arrays = False
    with lynqx.short_arrays(True):
        lynqx.config.short_arrays = False  # global write inside a context is shadowed by the override
        assert lynqx.config.short_arrays is True
    assert lynqx.config.short_arrays is False


def test_contexts_are_isolated_between_threads():
    seen: list[bool] = []
    lynqx.config.short_arrays = False
    with lynqx.short_arrays(True):
        t = threading.Thread(target=lambda: seen.append(lynqx.config.short_arrays))
        t.start()
        t.join()
    assert seen == [False]


def test_decorator_use():
    @lynqx.short_arrays(True)
    def f() -> bool:
        return lynqx.config.short_arrays

    lynqx.config.short_arrays = False
    assert f() is True
    assert lynqx.config.short_arrays is False


def test_unknown_option_raises():
    with pytest.raises(AttributeError, match="Unknown lynqx config option"):
        lynqx.config.update("verbos_arrays", True)
    with pytest.raises(AttributeError):
        lynqx.config.verbos_arrays = True
    with pytest.raises(AttributeError):
        _ = lynqx.config.nope
    assert not hasattr(lynqx.config, "nope")


def test_bool_validation():
    with pytest.raises(TypeError):
        lynqx.config.short_arrays = 1
    with pytest.raises(TypeError):
        with lynqx.short_arrays("yes"):
            pass


def test_x64_forwards_to_jax_and_is_context_aware():
    with lynqx.enable_x64(False):
        assert lynqx.config.enable_x64 is False
        assert jnp.zeros(()).dtype == jnp.float32
        with lynqx.enable_x64(True):
            assert lynqx.config.enable_x64 is True
            assert jnp.zeros(()).dtype == jnp.float64
        assert lynqx.config.enable_x64 is False


def test_x64_update_is_visible_to_jax():
    lynqx.config.enable_x64 = True
    assert jax.config.jax_enable_x64 is True


def test_override_multiple_options():
    with lynqx.config.override(enable_x64=True, short_arrays=False):
        assert lynqx.config.enable_x64 is True
        assert lynqx.config.short_arrays is False
    assert lynqx.config.short_arrays is True


def test_override_unwinds_on_bad_option():
    before = lynqx.config.short_arrays
    with pytest.raises(AttributeError):
        with lynqx.config.override(short_arrays=not before, nope=1):
            pass
    assert lynqx.config.short_arrays is before


def test_dtype_promotion_and_matmul_precision_contexts():
    with lynqx.numpy_dtype_promotion("strict"):
        assert lynqx.config.numpy_dtype_promotion == "strict"
    with lynqx.default_matmul_precision("highest"):
        assert lynqx.config.default_matmul_precision == "highest"


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
