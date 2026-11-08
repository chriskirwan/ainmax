import re
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    NamedTuple,
    Protocol,
    TYPE_CHECKING,
)

import numpy as np
from jaxtyping import ScalarLike
from jaxtyping._storage import get_shape_memo

from lynqx._src.axis import Axis, validate_axis_name
from lynqx._src.basearray import NamedArray, NamedArrayLike as NamedArrayLike


class NamedArrayCheckError(TypeError):
    """Base class for all NamedArray type/axis check errors."""

    pass


class NamedArrayDTypeError(NamedArrayCheckError):
    """Raised when a NamedArray has an incompatible dtype."""

    pass


class NamedArrayAxisError(NamedArrayCheckError):
    """Raised when NamedArray axes or sizes do not satisfy the specification."""

    pass


@dataclass(frozen=True, slots=True)
class NamedAxis:
    """A pattern describing a single axis (name and optional size/symbol)."""

    name: str | None = None
    size: int | str | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            validate_axis_name(self.name)
        if isinstance(self.size, str) and not self.size.isidentifier():
            raise ValueError(f"Symbolic size must be an identifier, got {self.size!r}")
        if isinstance(self.size, int) and self.size <= 0:
            raise ValueError(f"NamedAxis size must be a positive int, got {self.size!r}")

    @classmethod
    def of(cls, value: "NamedAxis | Axis | str | int") -> "NamedAxis":
        if isinstance(value, cls):
            return value
        if isinstance(value, Axis):
            return cls(value.name, value.size)
        if isinstance(value, str):
            return _parse_string_token(value.strip())
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(None, value)
        raise TypeError(f"Cannot interpret {value!r} as an axis specification")


class _AxisToken(NamedTuple):
    name: str | None
    size: int | str | None


_SIZE_RE = re.compile(r"[1-9][0-9]*")


def _parse_string_token(tok: str) -> NamedAxis:
    name, sep, size_str = tok.partition(":")
    if sep == "" and name.isdigit():
        if not _SIZE_RE.fullmatch(name):
            raise ValueError(f"Invalid size {name!r}")
        return NamedAxis(None, int(name))

    if name == "_" or not name:
        parsed_name = None
    else:
        try:
            validate_axis_name(name)
        except ValueError as exc:
            raise ValueError(f"Invalid axis token {tok!r}") from exc
        parsed_name = name

    parsed_size: int | str | None = None
    if sep == ":":
        if size_str == "*":
            parsed_size = None
        elif _SIZE_RE.fullmatch(size_str):
            parsed_size = int(size_str)
        elif size_str.isidentifier():
            parsed_size = size_str
        else:
            raise ValueError(f"Invalid size {size_str!r} in token {tok!r}")

    return NamedAxis(parsed_name, parsed_size)


class NamedAxisSpec:
    """Parsed, hashable specification for NamedArray axes supporting strings and NamedAxis objects."""

    __slots__ = (
        "tokens",
        "ordered",
        "leading_extra",
        "trailing_extra",
        "is_wildcard",
        "_raw",
        "_key",
        "_describe",
    )

    def __init__(self, spec: Any) -> None:
        if isinstance(spec, NamedAxisSpec):
            # Copy constructor
            for slot in self.__slots__:
                setattr(self, slot, getattr(spec, slot))
            return

        # Normalize spec input into a raw string representation or handle sequence directly
        if isinstance(spec, str):
            self._raw = spec.strip()
            tokens, ordered, leading, trailing, wildcard = self._parse_string_spec(spec)
        elif isinstance(spec, (tuple, list, set, frozenset)):
            ordered = not isinstance(spec, (set, frozenset))
            named_axes = [NamedAxis.of(x) for x in spec]
            if not ordered and any(axis.name is None for axis in named_axes):
                raise ValueError("An unordered specification cannot contain an anonymous axis")
            _validate_unique_axis_names(named_axes)
            tokens = tuple(_AxisToken(a.name, a.size) for a in named_axes)
            leading = trailing = not ordered
            wildcard = False
            self._raw = str(spec)
        else:
            # Single axis/int/NamedAxis
            single = NamedAxis.of(spec)
            tokens = (_AxisToken(single.name, single.size),)
            ordered = True
            leading = trailing = wildcard = False
            self._raw = str(spec)

        self.tokens = tokens
        self.ordered = ordered
        self.leading_extra = leading
        self.trailing_extra = trailing
        self.is_wildcard = wildcard
        self._finalize()

    @staticmethod
    def _parse_string_spec(dim_str: str) -> tuple[tuple[_AxisToken, ...], bool, bool, bool, bool]:
        if dim_str.strip() == "...":
            return (), False, True, True, True
        if dim_str == "":
            return (), True, False, False, False
        if not dim_str.strip():
            return (), False, True, True, False

        rest = dim_str.strip()
        has_brace = rest.startswith("{") or rest.endswith("}")
        if has_brace and not (rest.startswith("{") and rest.endswith("}")):
            raise ValueError(f"Invalid braced axis specification {rest!r}")

        if rest.startswith("{"):
            ordered = False
            inner = rest[1:-1].strip()
            if inner == "...":
                return (), False, True, True, True
            if "..." in inner:
                raise ValueError("Ellipsis is not meaningful inside a braced axis specification")
            raw_tokens = inner.split()
            leading = trailing = True
        else:
            ordered = True
            unbraced = rest.split()
            leading = bool(unbraced) and unbraced[0] == "..."
            if leading:
                unbraced = unbraced[1:]
            trailing = bool(unbraced) and unbraced[-1] == "..."
            if trailing:
                unbraced = unbraced[:-1]
            raw_tokens = unbraced

            if "..." in unbraced:
                raise ValueError("Ellipsis may only appear at the very start or end of an ordered spec")
            if leading and trailing:
                raise ValueError(
                    "An ordered spec cannot allow extra axes on both ends; use a contiguous-subsequence-anywhere spec"
                )

        named_axes = [_parse_string_token(token) for token in raw_tokens]
        if not ordered and any(axis.name is None for axis in named_axes):
            raise ValueError("An unordered specification cannot contain an anonymous axis")
        _validate_unique_axis_names(named_axes)
        parsed = tuple(_AxisToken(axis.name, axis.size) for axis in named_axes)
        return parsed, ordered, leading, trailing, False

    def _finalize(self) -> None:
        self._key = (self.ordered, self.leading_extra, self.trailing_extra, self.is_wildcard, self.tokens)
        if self.is_wildcard:
            self._describe = "any axes (wildcard)"
        elif not self.tokens and self.ordered:
            self._describe = "scalar (no axes)"
        elif not self.ordered:
            self._describe = "any axes (unordered, open)"
        else:
            axes = " ".join(
                (token.name if token.name is not None else "_") + (f":{token.size}" if token.size is not None else "")
                for token in self.tokens
            )
            if self.leading_extra:
                self._describe = f"ordered axes that end with {axes}"
            elif self.trailing_extra:
                self._describe = f"ordered axes that start with {axes}"
            else:
                self._describe = f"ordered, exact axes {axes}"

    def describe(self) -> str:
        return self._describe

    def __repr__(self) -> str:
        return f"NamedAxisSpec({self._raw!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NamedAxisSpec) and self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)


def _bind_axis(name: str, size: int) -> None:
    single_memo, _, _, _ = get_shape_memo()
    existing = single_memo.get(name)
    if existing is None:
        single_memo[name] = size
    elif existing != size:
        raise NamedArrayAxisError(
            f"Size mismatch for axis {name!r}: got {size}, but a previous argument bound {name!r}={existing}"
        )


def _validate_unique_axis_names(axes: Iterable[NamedAxis]) -> None:
    seen: set[str] = set()
    for axis in axes:
        if axis.name is not None:
            if axis.name in seen:
                raise ValueError(f"Duplicate named axis {axis.name!r}")
            seen.add(axis.name)


def _token_matches(axis: Axis, token: _AxisToken) -> bool:
    if (axis.name is None) != (token.name is None):
        return False
    if token.name is not None and axis.name != token.name:
        return False
    if token.size is None:
        return True
    if isinstance(token.size, int):
        return axis.size == token.size
    # Symbolic size check handled via memo
    return True


def _validate_ordered(axes: tuple[Axis, ...], spec: NamedAxisSpec) -> None:
    n = len(spec.tokens)
    if not spec.leading_extra and not spec.trailing_extra:
        window, ok = axes, len(axes) == n
    elif spec.leading_extra:
        window, ok = (axes[-n:] if n else ()), len(axes) >= n
    else:
        window, ok = axes[:n], len(axes) >= n

    if ok:
        ok = all(_token_matches(a, t) for a, t in zip(window, spec.tokens))

    if not ok:
        raise NamedArrayAxisError(f"NamedArray axes {list(axes)} do not satisfy spec {spec._raw!r}")

    for a, t in zip(window, spec.tokens):
        if t.name is not None:
            if isinstance(t.size, str):
                _bind_axis(t.size, a.size)
            _bind_axis(t.name, a.size)


def _validate_unordered(axes: tuple[Axis, ...], spec: NamedAxisSpec) -> None:
    named = {a.name: a.size for a in axes if a.name is not None}
    for t in spec.tokens:
        if t.name not in named:
            raise NamedArrayAxisError(f"NamedArray missing axis {t.name!r} for spec {spec._raw!r}")
        size = named[t.name]
        if t.size is not None and isinstance(t.size, int) and size != t.size:
            raise NamedArrayAxisError(f"Axis {t.name!r} has size {size}, expected {t.size}")
        if isinstance(t.size, str):
            _bind_axis(t.size, size)
        _bind_axis(t.name, size)


def _validate(instance: Any, valid_dtypes: frozenset[np.dtype] | None, spec: NamedAxisSpec) -> None:
    if valid_dtypes is not None:
        try:
            actual_dtype = np.dtype(instance.dtype)
        except Exception as exc:
            raise NamedArrayDTypeError(f"Could not resolve dtype: {exc}") from exc
        if actual_dtype not in valid_dtypes:
            raise NamedArrayDTypeError(f"NamedArray dtype mismatch: got {actual_dtype!r}")

    if spec.is_wildcard:
        return

    if not hasattr(instance, "axes"):
        raise NamedArrayAxisError(f"Expected NamedArray with .axes attribute, got {type(instance)}")

    axes: tuple[Axis, ...] = instance.axes
    if spec.ordered:
        _validate_ordered(axes, spec)
    else:
        _validate_unordered(axes, spec)


# ---------------------------------------------------------------------------
# Dtype annotations
# ---------------------------------------------------------------------------


class _MetaNamedArrayType(type):
    _dtype_name: str
    _valid_dtypes: frozenset[np.dtype] | None
    _axis_spec: NamedAxisSpec

    def __instancecheck__(cls, instance: Any) -> bool:
        if not isinstance(instance, NamedArray):
            return False
        try:
            _validate(instance, cls._valid_dtypes, cls._axis_spec)
        except NamedArrayCheckError:
            return False
        return True

    def __repr__(cls) -> str:
        return f"<{cls.__name__} — {cls._axis_spec._raw!r}>"


_type_cache: weakref.WeakValueDictionary[tuple[type, NamedAxisSpec], type] = weakref.WeakValueDictionary()


def _make_named_array_type(dtype_class: Any, axis_spec_input: Any) -> type:
    axis_spec = NamedAxisSpec(axis_spec_input)
    cache_key = (dtype_class, axis_spec)
    if (cached := _type_cache.get(cache_key)) is not None:
        return cached

    raw_dtypes = getattr(dtype_class, "dtypes", None)
    valid_dtypes = (
        frozenset(np.dtype(d) for d in raw_dtypes)
        if isinstance(raw_dtypes, Iterable) and not isinstance(raw_dtypes, (str, bytes))
        else None
    )

    new_type = _MetaNamedArrayType(
        f"{dtype_class.__name__}[NamedArray, {axis_spec._raw!r}]",
        (NamedArray,),
        {
            "__module__": __name__,
            "_dtype_name": dtype_class.__name__,
            "_valid_dtypes": valid_dtypes,
            "_axis_spec": axis_spec,
        },
    )
    _type_cache[cache_key] = new_type
    return new_type


_specifier_cache: weakref.WeakValueDictionary[type, type] = weakref.WeakValueDictionary()


def _make_dtype(jt_dtype: Any) -> type:
    if (cached := _specifier_cache.get(jt_dtype)) is not None:
        return cached

    class _NamedAwareDtype:
        dtypes: ClassVar[Any] = getattr(jt_dtype, "dtypes", None)

        def __class_getitem__(cls, item: tuple[type, Any]) -> Any:
            array_type, spec_input = item
            is_named = array_type is NamedArray or (
                isinstance(array_type, type) and issubclass(array_type, NamedArray)
            )
            if is_named:
                return _make_named_array_type(jt_dtype, spec_input)
            if isinstance(spec_input, str) and ("{" in spec_input or "}" in spec_input):
                raise TypeError("Braced axis syntax is only valid for NamedArray")
            return jt_dtype[array_type, spec_input]

    _NamedAwareDtype.__name__ = jt_dtype.__name__
    _specifier_cache[jt_dtype] = _NamedAwareDtype
    return _NamedAwareDtype


if TYPE_CHECKING:
    from jaxtyping import (
        BFloat16 as BFloat16,
        Bool as Bool,
        Complex as Complex,
        Complex64 as Complex64,
        Complex128 as Complex128,
        Float as Float,
        Float8e4m3b11fnuz as Float8e4m3b11fnuz,
        Float8e4m3fn as Float8e4m3fn,
        Float8e4m3fnuz as Float8e4m3fnuz,
        Float8e5m2 as Float8e5m2,
        Float8e5m2fnuz as Float8e5m2fnuz,
        Float16 as Float16,
        Float32 as Float32,
        Float64 as Float64,
        Inexact as Inexact,
        Int as Int,
        Int2 as Int2,
        Int4 as Int4,
        Int8 as Int8,
        Int16 as Int16,
        Int32 as Int32,
        Int64 as Int64,
        Integer as Integer,
        Key as Key,
        Num as Num,
        Real as Real,
        Scalar as Scalar,
        ScalarLike as ScalarLike,
        Shaped as Shaped,
        UInt as UInt,
        UInt2 as UInt2,
        UInt4 as UInt4,
        UInt8 as UInt8,
        UInt16 as UInt16,
        UInt32 as UInt32,
        UInt64 as UInt64,
    )

else:
    import jaxtyping as jt

    Bool = _make_dtype(jt.Bool)

    Int = _make_dtype(jt.Int)
    Integer = _make_dtype(jt.Integer)
    Int2 = _make_dtype(jt.Int2)
    Int4 = _make_dtype(jt.Int4)
    Int8 = _make_dtype(jt.Int8)
    Int16 = _make_dtype(jt.Int16)
    Int32 = _make_dtype(jt.Int32)
    Int64 = _make_dtype(jt.Int64)
    UInt = _make_dtype(jt.UInt)
    UInt2 = _make_dtype(jt.UInt2)
    UInt4 = _make_dtype(jt.UInt4)
    UInt8 = _make_dtype(jt.UInt8)
    UInt16 = _make_dtype(jt.UInt16)
    UInt32 = _make_dtype(jt.UInt32)
    UInt64 = _make_dtype(jt.UInt64)

    BFloat16 = _make_dtype(jt.BFloat16)
    Float = _make_dtype(jt.Float)
    Float8e4m3b11fnuz = _make_dtype(jt.Float8e4m3b11fnuz)
    Float8e4m3fnuz = _make_dtype(jt.Float8e4m3fnuz)
    Float8e4m3fn = _make_dtype(jt.Float8e4m3fn)
    Float8e5m2 = _make_dtype(jt.Float8e5m2)
    Float8e5m2fnuz = _make_dtype(jt.Float8e5m2fnuz)
    Float16 = _make_dtype(jt.Float16)
    Float32 = _make_dtype(jt.Float32)
    Float64 = _make_dtype(jt.Float64)

    Complex = _make_dtype(jt.Complex)
    Complex64 = _make_dtype(jt.Complex64)
    Complex128 = _make_dtype(jt.Complex128)

    Inexact = _make_dtype(jt.Inexact)
    Real = _make_dtype(jt.Real)
    Num = _make_dtype(jt.Num)
    Shaped = _make_dtype(jt.Shaped)

    Key = _make_dtype(jt.Key)


class DuckTypedNamedArray(Protocol):
    @property
    def dtype(self) -> np.dtype: ...

    @property
    def shape(self) -> Sequence[Axis | Any]: ...


NamedScalar = Shaped[NamedArray, ""]
NamedScalarLike = NamedScalar | ScalarLike
