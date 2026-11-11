import contextlib
import contextvars
import os
from collections.abc import Callable, Generator
from typing import Any, Final

import jax


_UNSET: Final = object()
_TRUE: Final = frozenset({"1", "true", "yes", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "off"})


def parse_bool(text: str) -> bool:
    """Parse an environment-variable style boolean (no regex, just a lookup)."""
    value = text.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"Cannot interpret {text!r} as a boolean; use one of {sorted(_TRUE | _FALSE)}.")


def _check_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Expected a bool, got {type(value).__name__}: {value!r}")
    return value


class ConfigState[T]:
    """One configuration option: readable, settable, and usable as a context manager."""

    name: str
    help: str

    def get(self) -> T:
        raise NotImplementedError

    def set(self, value: T) -> None:
        """Set the process-wide value (the analogue of ``jax.config.update``)."""
        raise NotImplementedError

    def context(self, value: T) -> contextlib.AbstractContextManager[None]:
        """Temporarily override the option inside a ``with`` block."""
        raise NotImplementedError

    def __call__(self, value: T) -> contextlib.AbstractContextManager[None]:
        return self.context(value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name}={self.get()!r})"


class LocalState[T](ConfigState[T]):
    """An option owned by ainmax."""

    def __init__(
        self,
        name: str,
        default: T,
        *,
        validate: Callable[[Any], T],
        env: str | None = None,
        parse: Callable[[str], T] | None = None,
        help: str = "",
    ) -> None:
        self.name = name
        self.help = help
        self._validate = validate
        value: Any = default
        if env is not None and (raw := os.environ.get(env)) is not None:
            value = parse(raw) if parse is not None else raw
        self._global: T = validate(value)
        self._override: contextvars.ContextVar[Any] = contextvars.ContextVar(f"lynqx_{name}", default=_UNSET)

    def get(self) -> T:
        override = self._override.get()
        return self._global if override is _UNSET else override

    def set(self, value: T) -> None:
        self._global = self._validate(value)

    @contextlib.contextmanager
    def context(self, value: T) -> Generator[None]:
        token = self._override.set(self._validate(value))
        try:
            yield
        finally:
            self._override.reset(token)


class JaxState[T](ConfigState[T]):
    """An option forwarded to ``jax.config`` and JAX's own (thread-local) context manager."""

    def __init__(
        self,
        name: str,
        jax_name: str,
        jax_context: Callable[[Any], contextlib.AbstractContextManager[Any]],
        *,
        help: str = "",
    ) -> None:
        self.name = name
        self.jax_name = jax_name
        self.help = help
        self._jax_context = jax_context

    def get(self) -> T:
        return getattr(jax.config, self.jax_name)

    def set(self, value: T) -> None:
        jax.config.update(self.jax_name, value)

    def context(self, value: T) -> contextlib.AbstractContextManager[Any]:
        return self._jax_context(value)


class Config:
    """Registry of options with attribute access, ``update`` and ``override``."""

    def __init__(self, *states: ConfigState[Any]) -> None:
        object.__setattr__(self, "_states", {state.name: state for state in states})

    def _state(self, name: str) -> ConfigState[Any]:
        states: dict[str, ConfigState[Any]] = object.__getattribute__(self, "_states")
        try:
            return states[name]
        except KeyError:
            raise AttributeError(
                f"Unknown ainmax config option {name!r}. Available options: {sorted(states)}"
            ) from None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._state(name).get()

    def __setattr__(self, name: str, value: Any) -> None:
        # Unknown names raise instead of silently creating an attribute, so typos are caught.
        self._state(name).set(value)

    def update(self, name: str, value: Any) -> None:
        """Set an option process-wide, e.g. ``config.update("verbose_arrays", True)``."""
        self._state(name).set(value)

    @contextlib.contextmanager
    def override(self, **options: Any) -> Generator[None]:
        """Temporarily override several options at once; all are restored on exit."""
        with contextlib.ExitStack() as stack:
            for name, value in options.items():
                stack.enter_context(self._state(name).context(value))
            yield

    def values(self) -> dict[str, Any]:
        states: dict[str, ConfigState[Any]] = object.__getattribute__(self, "_states")
        return {name: state.get() for name, state in states.items()}

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *object.__getattribute__(self, "_states")})

    def __repr__(self) -> str:
        body = ", ".join(f"{name}={value!r}" for name, value in self.values().items())
        return f"ainmax.config({body})"


class AinmaxConfig(Config):
    # Annotations only (no values), so type checkers / IDEs know the attribute types; access goes via __getattr__.
    short_arrays: bool
    enable_x64: bool
    default_matmul_precision: str | None
    numpy_dtype_promotion: str
    debug_nans: bool


short_arrays: Final = LocalState[bool](
    "short_arrays",
    True,
    validate=_check_bool,
    env="AIMNAX_SHORT_ARRAYS",
    parse=parse_bool,
    help="Print NamedArrays with the full array contents (JAX-style) instead of the short `f32[x:8, ?:2]` form.",
)
enable_x64: Final = JaxState[bool](
    "enable_x64", "jax_enable_x64", jax.enable_x64, help="Forwarded to jax_enable_x64 (64-bit dtypes)."
)
default_matmul_precision: Final = JaxState[str | None](
    "default_matmul_precision",
    "jax_default_matmul_precision",
    jax.default_matmul_precision,
    help="Forwarded to jax_default_matmul_precision (e.g. 'highest' to avoid TF32/bf16 matmuls on GPU/TPU).",
)
numpy_dtype_promotion: Final = JaxState[str](
    "numpy_dtype_promotion",
    "jax_numpy_dtype_promotion",
    jax.numpy_dtype_promotion,
    help="Forwarded to jax_numpy_dtype_promotion ('standard' or 'strict').",
)
debug_nans: Final = JaxState[bool](
    "debug_nans", "jax_debug_nans", jax.debug_nans, help="Forwarded to jax_debug_nans (raise on the first NaN)."
)

config: Final = AinmaxConfig(short_arrays, enable_x64, default_matmul_precision, numpy_dtype_promotion, debug_nans)
