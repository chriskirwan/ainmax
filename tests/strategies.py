import jax.numpy as jnp
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from lynqx._src.axis import Axis
from lynqx._src.named import array


axis_names = st.sampled_from(["x", "y", "z", "batch", "link"])
axis_sizes = st.integers(min_value=1, max_value=16)


@st.composite
def named_axes(draw):
    return Axis(draw(axis_sizes), draw(axis_names))


@st.composite
def anonymous_axes(draw):
    return Axis(draw(axis_sizes))


@st.composite
def disjoint_named_axis_tuples(draw, min_size=0, max_size=4):
    """A tuple of named axes with no repeated name -- the precondition
    `concatenate_axes` and friends actually want."""
    names = draw(st.lists(axis_names, min_size=min_size, max_size=max_size, unique=True))
    return tuple(Axis(draw(axis_sizes), name) for name in names)


@st.composite
def disjoint_named_axis_tuple_pairs(draw, max_size=3):
    """Two tuples of named axes, disjoint both internally and from each
    other -- for testing operations like `concatenate_axes`/`union_axes`
    where cross-input disjointedness is the interesting precondition.

    """
    names = draw(st.lists(axis_names, min_size=0, max_size=max_size * 2, unique=True))
    split = draw(st.integers(min_value=0, max_value=len(names)))
    a = tuple(Axis(draw(axis_sizes), name) for name in names[:split])
    b = tuple(Axis(draw(axis_sizes), name) for name in names[split:])
    return a, b


@st.composite
def duplicate_named_axis_pair(draw):
    """Two named axes sharing a name -- sizes may or may not match, since
    a duplicate *name* is the violation `check_unique_axis_names` cares
    about, independent of whether the sizes happen to agree (that's a
    separate concern, already covered by union_axes/intersect_axes's own
    "conflicting sizes" checks elsewhere)."""
    name = draw(axis_names)
    return Axis(draw(axis_sizes), name), Axis(draw(axis_sizes), name)


@st.composite
def mixed_anonymous_and_duplicate_named_tuples(draw):
    """Generates tuples containing a mix of anonymous axes and colliding named axes."""
    anon = draw(st.lists(anonymous_axes(), min_size=1, max_size=4))
    dupes = draw(duplicate_named_axis_pair())
    combined = list(anon) + list(dupes)
    draw(st.randoms()).shuffle(combined)
    return tuple(combined)


@st.composite
def named_arrays(draw, *, min_dims=1, max_dims=3, dtype=jnp.float32, min_value=-100.0, max_value=100.0):
    """A `NamedArray` with unique axis names and finite float entries.

    Subnormals are excluded because XLA on CPU may flush them to zero, which would make exact-equality properties
    flaky. Values are bounded so that elementwise identities (``round``, ``clip``, ...) are never dominated by overflow
    or by the loss of precision that comes with huge magnitudes.
    """
    axes = draw(disjoint_named_axis_tuples(min_size=min_dims, max_size=max_dims))
    shape = tuple(ax.size for ax in axes)
    elements = st.floats(min_value, max_value, width=32, allow_nan=False, allow_infinity=False, allow_subnormal=False)
    data = draw(hnp.arrays(dtype, shape, elements=elements))
    return array(jnp.asarray(data), axes)


@st.composite
def named_array_and_axis(draw, **kwargs):
    """A (non-scalar) `NamedArray` together with one of its axes."""
    kwargs.setdefault("min_dims", 1)
    a = draw(named_arrays(**kwargs))
    return a, draw(st.sampled_from(a.axes))


@st.composite
def named_array_and_axis_pair(draw, **kwargs):
    """A `NamedArray` with at least two axes, together with two *distinct* axes of it."""
    kwargs.setdefault("min_dims", 2)
    a = draw(named_arrays(**kwargs))
    first, second = draw(st.lists(st.sampled_from(a.axes), min_size=2, max_size=2, unique=True))
    return a, first, second


axes = st.one_of(named_axes(), anonymous_axes())
