import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ainmax._src.axis import Axis
from ainmax._src.feidhm.constructors import array
from ainmax._src.feidhm.creation import ones
from ainmax._src.feidhm.operations import (
    allclose,
    angle,
    around,
    array_equal,
    array_equiv,
    astype,
    bincount,
    block,
    broadcast_arrays,
    broadcast_arrays_and_shape,
    broadcast_shapes,
    broadcast_to,
    clip,
    compress,
    concat,
    concatenate,
    histogram,
    histogram2d,
    histogram_bin_edges,
    histogramdd,
    isclose,
    iscomplex,
    iscomplexobj,
    pad,
    ravel,
    rename,
    reshape,
    roll,
    round,
    searchsorted,
    stack,
    swapaxes,
    tile,
    top_k,
    transpose,
    unstack,
    where,
)
from hypothesis import assume, given, strategies as st
from strategies import (
    disjoint_named_axis_tuples,
    named_array_and_axis,
    named_array_and_axis_pair,
    named_arrays,
)


@pytest.fixture
def named_with_anon():
    data = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    return array(data, (Axis(2, "batch"), Axis(3), Axis(4, "channel")))


class TestBroadcastTo:
    def test_broadcast_expands_unit_axes(self):
        a = ones((Axis(1, "x"), Axis(8, "y")))
        target_shape = (Axis(4, "x"), Axis(8, "y"))

        res = broadcast_to(a, target_shape)

        assert res.axes == target_shape
        assert res.array.shape == (4, 8)
        assert jnp.array_equal(res.array, jnp.ones((4, 8)))

    def test_broadcast_adds_new_leading_or_trailing_axes(self):
        a = ones(Axis(8, "y"))
        target_shape = (Axis(4, "x"), Axis(8, "y"), Axis(16, "z"))

        res = broadcast_to(a, target_shape)

        assert res.axes == target_shape
        assert res.array.shape == (4, 8, 16)

    def test_broadcast_dict_shape_specifier(self):
        a = ones((Axis(1, "x"), Axis(8, "y")))
        res = broadcast_to(a, {"x": 4, "y": 8, "z": 16})

        assert res.axes == (Axis(4, "x"), Axis(8, "y"), Axis(16, "z"))
        assert res.array.shape == (4, 8, 16)

    def test_broadcast_dropping_existing_axis_raises(self, named_3d):
        # Target shape missing 'z'
        target_shape = (Axis(4, "x"), Axis(8, "y"))
        with pytest.raises(ValueError, match="broadcast_to cannot drop axes"):
            broadcast_to(named_3d, target_shape)

    def test_broadcast_incompatible_non_unit_size_raises(self):
        a = ones((Axis(4, "x"), Axis(8, "y")))
        target_shape = (Axis(8, "x"), Axis(8, "y"))

        with pytest.raises(ValueError, match="Cannot broadcast axis 'x' of size 4 to 8"):
            broadcast_to(a, target_shape)

    def test_broadcast_preserves_array_values(self):
        data = jnp.array([1.0, 2.0, 3.0])
        a = array(data, (Axis(3, "x"),))
        res = broadcast_to(a, (Axis(2, "batch"), Axis(3, "x")))

        expected = jnp.broadcast_to(data[None, :], (2, 3))
        assert jnp.array_equal(res.array, expected)

    def test_broadcast_reorders_source_and_inserts_axis_between_matched_axes(self, x4, z16):
        data = jnp.arange(16 * 4).reshape(16, 4)  # axes (z, x), in that order
        a = array(data, (z16, x4))

        target_shape = (x4, Axis(1, "w"), z16)
        res = broadcast_to(a, target_shape)

        assert res.axes == target_shape
        assert res.array.shape == (4, 1, 16)

        # Not just shape: confirm z's data actually landed on the z axis
        # (position 2) and not on w (position 1) as the bug would have it.
        expected = jnp.transpose(data)[:, None, :]  # (x, w, z)
        assert jnp.array_equal(res.array, expected)

    def test_broadcast_dropping_unit_axis_raises(self):
        a = ones((Axis(1, "x"), Axis(3, "y")))
        with pytest.raises(ValueError, match="broadcast_to cannot drop axes"):
            broadcast_to(a, (Axis(3, "y"),))

    def test_broadcast_matches_anonymous_axis_positionally(self, named_with_anon):
        target_shape = (Axis(2, "batch"), Axis(3), Axis(4, "channel"))
        res = broadcast_to(named_with_anon, target_shape)
        assert res.axes == target_shape

    def test_broadcast_duplicate_axis_name_in_shape_raises(self):
        a = ones(Axis(8, "x"))
        with pytest.raises(ValueError, match="Duplicate axis names"):
            broadcast_to(a, (Axis(8, "x"), Axis(5, "x")))

    def test_broadcast_anonymous_trailing_alignment(self):
        a = ones(Axis(3))
        target_shape = (Axis(2, "batch"), Axis(3))
        res = broadcast_to(a, target_shape)

        assert res.axes == target_shape
        assert res.array.shape == (2, 3)

    def test_broadcast_anonymous_cannot_match_named_target(self):
        a = ones((Axis(4), Axis(3, "x")))
        target_shape = (Axis(3, "x"), Axis(4, "y"))

        with pytest.raises(ValueError, match="broadcast_to cannot drop axes"):
            broadcast_to(a, target_shape)


class TestBroadcastArrays:
    def test_empty_input_returns_empty_tuple(self):
        assert broadcast_arrays() == ()

    def test_single_array_returns_unchanged(self, x4):
        a = ones(x4)
        (res,) = broadcast_arrays(a)

        assert res.axes == (x4,)
        assert jnp.array_equal(res.array, a.array)

    def test_broadcasts_two_disjoint_named_arrays(self, x4, y8):
        a = ones(x4)
        b = ones(y8)

        res_a, res_b = broadcast_arrays(a, b)

        expected_axes = (x4, y8)
        assert res_a.axes == expected_axes
        assert res_b.axes == expected_axes
        assert res_a.array.shape == (4, 8)
        assert res_b.array.shape == (4, 8)

    def test_broadcasts_three_arrays_with_mixed_and_unit_axes(self):
        ax_x1 = Axis(1, "x")
        ax_x4 = Axis(4, "x")
        ax_y8 = Axis(8, "y")
        ax_z16 = Axis(16, "z")

        a = ones((ax_x1, ax_y8))
        b = ones((ax_x4, ax_y8))
        c = ones(ax_z16)

        res_a, res_b, res_c = broadcast_arrays(a, b, c)

        expected_axes = (ax_x4, ax_y8, ax_z16)
        assert res_a.axes == expected_axes
        assert res_b.axes == expected_axes
        assert res_c.axes == expected_axes

        assert res_a.array.shape == (4, 8, 16)
        assert res_b.array.shape == (4, 8, 16)
        assert res_c.array.shape == (4, 8, 16)

    def test_broadcasts_values_correctly(self):
        a = array(jnp.array([1.0, 2.0, 3.0]), (Axis(3, "x"),))
        b = array(jnp.array([10.0, 20.0]), (Axis(2, "y"),))

        res_a, res_b = broadcast_arrays(a, b)

        expected_a = jnp.tile(jnp.array([1.0, 2.0, 3.0])[:, None], (1, 2))
        expected_b = jnp.tile(jnp.array([10.0, 20.0])[None, :], (3, 1))

        assert jnp.array_equal(res_a.array, expected_a)
        assert jnp.array_equal(res_b.array, expected_b)

    def test_incompatible_axis_sizes_raises_value_error(self):
        a = ones(Axis(4, "x"))
        b = ones(Axis(8, "x"))

        with pytest.raises(ValueError, match="Incompatible sizes for axis 'x'"):
            broadcast_arrays(a, b)

    def test_incompatible_anonymous_axis_sizes_raises_value_error(self):
        a = ones(Axis(4))
        b = ones(Axis(8))

        with pytest.raises(ValueError, match="Cannot align anonymous axes"):
            broadcast_arrays(a, b)

    @given(disjoint_named_axis_tuples(min_size=1, max_size=3))
    def test_broadcast_arrays_idempotent_shapes(self, axes):
        shape = tuple(ax.size for ax in axes)
        arr = jnp.ones(shape)
        named_arr = array(arr, axes)

        (res,) = broadcast_arrays(named_arr)
        assert res.axes == named_arr.axes
        assert res.array.shape == named_arr.array.shape

    def test_broadcast_arrays_preserves_left_hand_axis_order(self):
        a = ones((Axis(4, "x"), Axis(8, "y"), Axis(3, "i")))
        b = ones(Axis(3, "i"))

        res_a, res_b = broadcast_arrays(a, b)

        expected_axes = (Axis(4, "x"), Axis(8, "y"), Axis(3, "i"))  # NOT (i, x, y)
        assert res_a.axes == expected_axes
        assert res_b.axes == expected_axes


class TestTranspose:
    def test_default_transpose_reverses_axes(self, named_3d, x4, y8, z16):
        res = transpose(named_3d)

        assert res.axes == (z16, y8, x4)
        assert res.array.shape == (16, 8, 4)
        assert jnp.array_equal(res.array, jnp.transpose(named_3d.array))

    def test_transpose_with_explicit_string_names(self, named_3d, x4, y8, z16):
        res = transpose(named_3d, ("z", "x", "y"))

        assert res.axes == (z16, x4, y8)
        assert res.array.shape == (16, 4, 8)
        assert jnp.array_equal(res.array, jnp.transpose(named_3d.array, (2, 0, 1)))

    def test_transpose_with_integer_indices(self, named_3d, x4, y8, z16):
        res = transpose(named_3d, (2, 0, 1))

        assert res.axes == (z16, x4, y8)
        assert res.array.shape == (16, 4, 8)
        assert jnp.array_equal(res.array, jnp.transpose(named_3d.array, (2, 0, 1)))

    def test_transpose_with_negative_integer_indices(self, named_3d, x4, y8, z16):
        res = transpose(named_3d, (-1, 0, -2))

        assert res.axes == (z16, x4, y8)
        assert res.array.shape == (16, 4, 8)

    def test_transpose_with_mixed_selectors(self, named_with_anon):
        res = transpose(named_with_anon, ("channel", 1, "batch"))

        assert res.axes == (Axis(4, "channel"), Axis(3), Axis(2, "batch"))
        assert res.array.shape == (4, 3, 2)
        assert jnp.array_equal(res.array, jnp.transpose(named_with_anon.array, (2, 1, 0)))

    def test_transpose_identity_permutation(self, named_3d):
        res = transpose(named_3d, ("x", "y", "z"))

        assert res.axes == named_3d.axes
        assert jnp.array_equal(res.array, named_3d.array)

    def test_unresolved_axis_raises_value_error(self, named_3d):
        with pytest.raises(ValueError):
            transpose(named_3d, ("x", "unknown"))

    def test_duplicate_axis_raises_value_error(self, named_3d):
        with pytest.raises(ValueError):
            transpose(named_3d, ("x", "x", "y"))

    def test_wrong_number_of_axes_raises_value_error(self, named_3d):
        with pytest.raises(ValueError):
            transpose(named_3d, ("x", "y"))


def assert_named(res, axes, expected=None):
    """Check the axes of ``res`` and (optionally) the exact values of its underlying array."""
    assert res.axes == tuple(axes)
    assert res.array.shape == tuple(ax.size for ax in axes)
    if expected is not None:
        np.testing.assert_array_equal(res.array, expected)


@pytest.fixture
def cplx():
    data = (jnp.arange(6.0) + 1j * jnp.arange(6.0)[::-1]).reshape(2, 3).astype(jnp.complex64)
    return array(data, (Axis(2, "x"), Axis(3, "y")))


# utility functions


class TestIsComplex:
    def test_python_scalars(self):
        assert bool(iscomplex(1j).array)
        assert not bool(iscomplex(1.0).array)

    def test_elementwise_result_keeps_axes(self):
        x = array(jnp.array([1 + 0j, 1 + 1j, 2 + 0j]), (Axis(3, "x"),))
        res = iscomplex(x)
        assert_named(res, x.axes, jnp.array([False, True, False]))


class TestIsComplexObj:
    def test_by_dtype_not_value(self):
        # zero imaginary part still counts as complex
        assert iscomplexobj(ones(Axis(3, "x"), dtype=jnp.complex64))
        assert not iscomplexobj(ones(Axis(3, "x")))

    def test_plain_inputs(self):
        assert iscomplexobj(1 + 0j)
        assert iscomplexobj(jnp.zeros(2, dtype=jnp.complex64))
        assert not iscomplexobj(1.0)
        assert not iscomplexobj(None)


# broadcasting support


class TestBroadcastShapes:
    def test_no_shapes_gives_empty_tuple(self):
        assert broadcast_shapes() == ()

    def test_single_shape_is_normalised_to_axis_tuple(self):
        assert broadcast_shapes({"x": 3, "y": 4}) == (Axis(3, "x"), Axis(4, "y"))

    def test_disjoint_shapes_are_concatenated_in_order(self, x4, y8, z16):
        assert broadcast_shapes((x4,), (y8,), (z16,)) == (x4, y8, z16)

    def test_unit_axis_expands_to_matching_size(self):
        assert broadcast_shapes((Axis(1, "x"), Axis(8, "y")), (Axis(4, "x"),)) == (Axis(4, "x"), Axis(8, "y"))

    def test_mixed_shape_specifiers(self, y8):
        assert broadcast_shapes({"x": 4}, (y8,)) == (Axis(4, "x"), y8)

    def test_incompatible_sizes_raise(self):
        with pytest.raises(ValueError, match="Incompatible sizes for axis 'x'"):
            broadcast_shapes((Axis(4, "x"),), (Axis(8, "x"),))

    @given(disjoint_named_axis_tuples(min_size=0, max_size=4))
    def test_idempotent(self, axes):
        assert broadcast_shapes(axes, axes) == axes
        assert broadcast_shapes(axes) == axes

    @given(disjoint_named_axis_tuples(min_size=0, max_size=4))
    def test_matches_broadcast_arrays_and_shape(self, axes):
        arr = ones(axes)
        _, shape = broadcast_arrays_and_shape(arr, arr)
        assert shape == broadcast_shapes(axes, axes)


class TestBroadcastArraysAndShape:
    def test_empty(self):
        assert broadcast_arrays_and_shape() == ((), ())

    def test_returns_arrays_and_common_shape(self, x4, y8):
        a = ones(x4)
        b = ones(y8)

        (ra, rb), shape = broadcast_arrays_and_shape(a, b)

        assert shape == (x4, y8)
        assert ra.axes == rb.axes == shape

    def test_python_scalars_are_broadcast(self, x4):
        a = array(jnp.arange(4.0), (x4,))
        (ra, rb), shape = broadcast_arrays_and_shape(a, 2.0)

        assert shape == (x4,)
        assert_named(rb, (x4,), jnp.full((4,), 2.0))

    @given(named_arrays(min_dims=0, max_dims=3))
    def test_broadcast_with_scalar_is_noop_on_array(self, a):
        (ra, rs), shape = broadcast_arrays_and_shape(a, 1.0)
        assert shape == a.axes
        np.testing.assert_array_equal(ra.array, a.array)
        np.testing.assert_array_equal(rs.array, jnp.ones(a.array.shape))


# shape/axis-wise operations


class TestRename:
    def test_renames_axis_and_keeps_data(self, mat):
        res = rename(mat, "x", "q")
        assert_named(res, (Axis(3, "q"), Axis(4, "y")), mat.array)

    def test_none_makes_axis_anonymous(self, mat):
        assert rename(mat, "x", None).axes == (Axis(3), Axis(4, "y"))

    def test_select_by_position(self, mat):
        assert rename(mat, -1, "q").axes == (Axis(3, "x"), Axis(4, "q"))

    def test_missing_axis_raises(self, mat):
        with pytest.raises(ValueError, match="not found"):
            rename(mat, "nope", "q")

    @given(named_array_and_axis())
    def test_roundtrip(self, arr_axis):
        a, ax = arr_axis
        back = rename(rename(a, ax.name, "tmp"), "tmp", ax.name)
        assert back.axes == a.axes
        np.testing.assert_array_equal(back.array, a.array)


class TestReshape:
    def test_values_follow_row_major_order(self, mat):
        res = reshape(mat, {"p": 2, "q": 6})
        assert_named(res, (Axis(2, "p"), Axis(6, "q")), mat.array.reshape(2, 6))

    def test_flatten_to_one_axis(self, mat):
        assert_named(reshape(mat, {"f": 12}), (Axis(12, "f"),), mat.array.ravel())

    def test_axis_tuple_shape(self, mat):
        res = reshape(mat, (Axis(6), Axis(2, "q")))
        assert res.axes == (Axis(6), Axis(2, "q"))

    def test_size_mismatch_raises(self, mat):
        with pytest.raises(ValueError):
            reshape(mat, {"p": 5})

    @given(named_arrays(min_dims=1, max_dims=3))
    def test_roundtrip_through_flat(self, a):
        flat = reshape(a, {"flat": a.array.size})
        back = reshape(flat, a.axes)
        assert back.axes == a.axes
        np.testing.assert_array_equal(back.array, a.array)


class TestSwapaxes:
    def test_matches_jnp(self, mat):
        res = swapaxes(mat, "x", "y")
        assert_named(res, (Axis(4, "y"), Axis(3, "x")), jnp.swapaxes(mat.array, 0, 1))

    def test_by_position(self, mat):
        assert swapaxes(mat, 0, -1).axes == (Axis(4, "y"), Axis(3, "x"))

    def test_only_named_axes_move_in_3d(self, named_3d, x4, y8, z16):
        res = swapaxes(named_3d, "x", "z")
        assert res.axes == (z16, y8, x4)
        np.testing.assert_array_equal(res.array, jnp.swapaxes(named_3d.array, 0, 2))

    def test_same_axis_raises(self, mat):
        with pytest.raises(ValueError, match="must be distinct"):
            swapaxes(mat, "x", "x")

    def test_unknown_axis_raises(self, mat):
        with pytest.raises(ValueError):
            swapaxes(mat, "x", "nope")

    @given(named_array_and_axis_pair())
    def test_involution(self, arr_axes):
        a, ax1, ax2 = arr_axes
        back = swapaxes(swapaxes(a, ax1.name, ax2.name), ax1.name, ax2.name)
        assert back.axes == a.axes
        np.testing.assert_array_equal(back.array, a.array)


class TestRoll:
    def test_single_axis(self, mat):
        res = roll(mat, 1, "x")
        assert_named(res, mat.axes, jnp.roll(mat.array, 1, axis=0))

    def test_negative_shift(self, mat):
        np.testing.assert_array_equal(roll(mat, -1, "y").array, jnp.roll(mat.array, -1, axis=1))

    def test_int_shift_applies_to_every_selected_axis(self, mat):
        res = roll(mat, 1, ("x", "y"))
        np.testing.assert_array_equal(res.array, jnp.roll(mat.array, 1, axis=(0, 1)))

    def test_sequence_of_shifts(self, mat):
        res = roll(mat, (1, 2), ("x", "y"))
        np.testing.assert_array_equal(res.array, jnp.roll(mat.array, (1, 2), axis=(0, 1)))

    def test_axis_by_position(self, mat):
        np.testing.assert_array_equal(roll(mat, 2, -1).array, jnp.roll(mat.array, 2, axis=1))

    def test_shift_longer_than_axis_wraps(self, mat):
        np.testing.assert_array_equal(roll(mat, 7, "x").array, jnp.roll(mat.array, 7, axis=0))

    def test_shift_length_mismatch_raises(self, mat):
        with pytest.raises(ValueError, match="does not match number of axes"):
            roll(mat, (1, 2), "x")

    def test_non_integer_shift_type_raises(self, mat):
        with pytest.raises(TypeError, match="shift must be int"):
            roll(mat, 1.5, "x")

    @given(named_array_and_axis(), st.integers(min_value=-20, max_value=20))
    def test_inverse_and_full_cycle(self, arr_axis, shift):
        a, ax = arr_axis
        back = roll(roll(a, shift, ax.name), -shift, ax.name)
        np.testing.assert_array_equal(back.array, a.array)
        np.testing.assert_array_equal(roll(a, ax.size, ax.name).array, a.array)


class TestPad:
    def test_int_pads_every_axis_symmetrically(self, mat):
        res = pad(mat, 1)
        assert_named(res, (Axis(5, "x"), Axis(6, "y")), jnp.pad(mat.array, 1))

    def test_tuple_is_before_after_for_every_axis(self, mat):
        res = pad(mat, (1, 2))
        assert_named(res, (Axis(6, "x"), Axis(7, "y")), jnp.pad(mat.array, (1, 2)))

    def test_mapping_by_name_leaves_others_untouched(self, mat):
        res = pad(mat, {"x": (1, 2)})
        assert_named(res, (Axis(6, "x"), Axis(4, "y")), jnp.pad(mat.array, ((1, 2), (0, 0))))

    def test_mapping_by_position_and_int_width(self, mat):
        res = pad(mat, {0: 1, -1: 2})
        assert res.axes == (Axis(5, "x"), Axis(8, "y"))

    def test_zero_pad_is_identity(self, mat):
        res = pad(mat, 0)
        assert_named(res, mat.axes, mat.array)

    def test_constant_values_kwarg(self):
        a = array(jnp.array([1.0, 2.0]), (Axis(2, "x"),))
        np.testing.assert_array_equal(pad(a, 1, constant_values=9.0).array, jnp.array([9.0, 1.0, 2.0, 9.0]))

    @pytest.mark.parametrize("mode", ["edge", "reflect", "wrap", "symmetric"])
    def test_numpy_modes(self, mode):
        data = jnp.arange(5.0)
        res = pad(array(data, (Axis(5, "x"),)), 2, mode=mode)
        np.testing.assert_array_equal(res.array, jnp.pad(data, 2, mode=mode))
        assert res.axes == (Axis(9, "x"),)

    def test_negative_width_raises(self, mat):
        with pytest.raises(ValueError, match="non-negative"):
            pad(mat, -1)
        with pytest.raises(ValueError, match="non-negative"):
            pad(mat, {"x": (1, -1)})

    def test_bad_tuple_length_raises(self, mat):
        with pytest.raises(ValueError, match=r"\(before, after\)"):
            pad(mat, (1, 2, 3))

    def test_unknown_axis_key_raises(self, mat):
        with pytest.raises(ValueError):
            pad(mat, {"nope": 1})

    def test_axis_listed_twice_raises(self, mat):
        with pytest.raises(ValueError, match="more than once"):
            pad(mat, {"x": 1, 0: 2})

    def test_callable_mode_is_applied_axis_by_axis(self, mat):
        calls = []

        def pad_fn(arr, ax, width):
            calls.append((ax.name, width))
            return pad(arr, {ax.name: width})

        res = pad(mat, 1, mode=pad_fn)

        assert calls == [("x", (1, 1)), ("y", (1, 1))]
        assert res.axes == (Axis(5, "x"), Axis(6, "y"))

    @given(
        named_arrays(min_dims=1, max_dims=3),
        st.integers(min_value=0, max_value=3),
        st.integers(min_value=0, max_value=3),
    )
    def test_interior_is_original_and_border_is_zero(self, a, lo, hi):
        res = pad(a, (lo, hi))
        assert res.axes == tuple(Axis(ax.size + lo + hi, ax.name) for ax in a.axes)

        interior = tuple(slice(lo, lo + ax.size) for ax in a.axes)
        np.testing.assert_array_equal(res.array[interior], a.array)
        # everything outside the interior is the constant pad value (zero)
        assert not bool(res.array.at[interior].set(0.0).any())


class TestRavel:
    def test_row_major_order(self, mat):
        assert_named(ravel(mat), (Axis(12),), jnp.arange(12.0))

    def test_name_string(self, mat):
        assert ravel(mat, "flat").axes == (Axis(12, "flat"),)

    def test_axis_object(self, mat):
        assert ravel(mat, Axis(12, "flat")).axes == (Axis(12, "flat"),)

    def test_axis_object_with_wrong_size_raises(self, mat):
        with pytest.raises(ValueError, match="has size 5"):
            ravel(mat, Axis(5, "flat"))

    def test_scalar_becomes_length_one(self):
        res = ravel(array(jnp.array(3.0), ()))
        assert_named(res, (Axis(1),), jnp.array([3.0]))

    @given(named_arrays(min_dims=1, max_dims=3))
    def test_matches_jnp_ravel(self, a):
        res = ravel(a, "f")
        assert res.axes == (Axis(a.array.size, "f"),)
        np.testing.assert_array_equal(res.array, jnp.ravel(a.array))


class TestStack:
    def test_new_axis_goes_first_by_default(self, mat):
        res = stack([mat, mat + 1], "s")
        assert_named(res, (Axis(2, "s"), *mat.axes), jnp.stack([mat.array, mat.array + 1]))

    def test_index_positions(self, mat):
        other = mat + 1
        for index in (0, 1, 2, -1, -2, -3):
            res = stack([mat, other], "s", index=index)
            expected = jnp.stack([mat.array, other.array], axis=index % 3)
            np.testing.assert_array_equal(res.array, expected)
            assert res.axes[index % 3] == Axis(2, "s")

    def test_axis_object_with_matching_size(self, mat):
        assert stack([mat, mat], Axis(2, "s")).axes[0] == Axis(2, "s")

    def test_anonymous_new_axis(self, mat):
        assert stack([mat, mat], Axis(2)).axes[0] == Axis(2)

    def test_named_axes_are_matched_by_name_not_order(self, mat):
        res = stack([mat, transpose(mat)], "s")
        assert res.axes == (Axis(2, "s"), *mat.axes)
        np.testing.assert_array_equal(res.array[1], mat.array)

    def test_anonymous_axes_match_by_order_of_appearance(self):
        a = array(jnp.arange(6.0).reshape(2, 3), (Axis(2), Axis(3)))
        b = array(jnp.arange(6.0).reshape(2, 3) * 2, (Axis(2), Axis(3)))
        res = stack([a, b], "s")
        np.testing.assert_array_equal(res.array, jnp.stack([a.array, b.array]))

    def test_dtype(self, mat):
        assert stack([mat, mat], "s", dtype=jnp.int32).array.dtype == jnp.int32

    def test_scalars(self):
        res = stack([array(jnp.array(1.0), ()), array(jnp.array(2.0), ())], "s")
        assert_named(res, (Axis(2, "s"),), jnp.array([1.0, 2.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one array"):
            stack([], "s")

    def test_new_axis_size_mismatch_raises(self, mat):
        with pytest.raises(ValueError, match="has size 3 but 2 arrays"):
            stack([mat, mat], Axis(3, "s"))

    def test_new_axis_name_clash_raises(self, mat):
        with pytest.raises(ValueError, match="already exists"):
            stack([mat, mat], "x")

    def test_mismatched_axes_raise(self, mat):
        other = ones((Axis(3, "x"), Axis(5, "z")))
        with pytest.raises(ValueError, match="no axis named 'y'"):
            stack([mat, other], "s")

    def test_mismatched_sizes_raise(self, mat):
        other = ones((Axis(3, "x"), Axis(5, "y")))
        with pytest.raises(ValueError, match="has size 5"):
            stack([mat, other], "s")

    def test_different_rank_raises(self, mat):
        with pytest.raises(ValueError, match="same axes"):
            stack([mat, ones(Axis(3, "x"))], "s")

    def test_index_out_of_range_raises(self, mat):
        with pytest.raises(ValueError, match="out of range"):
            stack([mat, mat], "s", index=3)
        with pytest.raises(ValueError, match="out of range"):
            stack([mat, mat], "s", index=-4)

    @pytest.mark.parametrize("index", [1.0, True])
    def test_non_int_index_raises(self, mat, index):
        with pytest.raises(TypeError, match="must be an int"):
            stack([mat, mat], "s", index=index)

    @given(named_arrays(min_dims=0, max_dims=3), st.integers(min_value=1, max_value=4), st.data())
    def test_matches_jnp_stack_at_any_valid_index(self, a, n, data):
        index = data.draw(st.integers(min_value=-(a.ndim + 1), max_value=a.ndim))
        res = stack([a] * n, "stacked", index=index)
        np.testing.assert_array_equal(res.array, jnp.stack([a.array] * n, axis=index))
        assert res.ndim == a.ndim + 1


class TestUnstack:
    def test_default_axis_is_first(self, mat):
        parts = unstack(mat)
        assert len(parts) == 3
        for i, p in enumerate(parts):
            assert_named(p, (Axis(4, "y"),), mat.array[i])

    def test_by_name(self, mat):
        parts = unstack(mat, "y")
        assert len(parts) == 4
        assert parts[0].axes == (Axis(3, "x"),)
        np.testing.assert_array_equal(parts[2].array, mat.array[:, 2])

    def test_missing_axis_raises(self, mat):
        with pytest.raises(ValueError, match="not found"):
            unstack(mat, "nope")

    @given(named_array_and_axis(), st.data())
    def test_inverse_of_stack(self, arr_axis, data):
        a, ax = arr_axis
        parts = unstack(a, ax.name)
        assert len(parts) == ax.size
        back = stack(parts, ax, index=a.axes.index(ax))
        assert back.axes == a.axes
        np.testing.assert_array_equal(back.array, a.array)


class TestTile:
    def test_int_applies_to_every_axis(self, mat):
        res = tile(mat, 2)
        assert_named(res, (Axis(6, "x"), Axis(8, "y")), jnp.tile(mat.array, (2, 2)))

    def test_sequence_is_per_axis_in_axis_order(self, mat):
        res = tile(mat, [2, 3])
        assert_named(res, (Axis(6, "x"), Axis(12, "y")), jnp.tile(mat.array, (2, 3)))

    def test_mapping_leaves_unlisted_axes_alone(self, mat):
        res = tile(mat, {"y": 2})
        assert_named(res, (Axis(3, "x"), Axis(8, "y")), jnp.tile(mat.array, (1, 2)))

    def test_mapping_by_position(self, mat):
        assert tile(mat, {-1: 3}).axes == (Axis(3, "x"), Axis(12, "y"))

    def test_never_prepends_axes(self, mat):
        assert tile(mat, 1).axes == mat.axes

    def test_scalar_is_returned_unchanged(self):
        s = array(jnp.array(1.0), ())
        assert tile(s, 3).axes == ()

    def test_sequence_length_mismatch_raises(self, mat):
        with pytest.raises(ValueError, match="has 3 entries but the array has 2 axes"):
            tile(mat, [1, 2, 3])

    @pytest.mark.parametrize("reps", [0, -1, True, [1, 0], [1.5, 1], {"x": 0}])
    def test_non_positive_or_non_int_reps_raise(self, mat, reps):
        with pytest.raises(ValueError, match="positive integers"):
            tile(mat, reps)

    def test_scalar_float_reps_raise(self, mat):
        # currently a TypeError from `tuple(1.5)` rather than the friendlier ValueError used for other bad reps
        with pytest.raises((TypeError, ValueError)):
            tile(mat, 1.5)

    def test_unknown_key_raises(self, mat):
        with pytest.raises(ValueError, match="not found"):
            tile(mat, {"nope": 2})

    def test_axis_listed_twice_raises(self, mat):
        with pytest.raises(ValueError, match="more than once"):
            tile(mat, {"x": 2, 0: 3})

    @given(named_arrays(min_dims=1, max_dims=3), st.data())
    def test_matches_jnp_tile(self, a, data):
        reps = tuple(data.draw(st.integers(min_value=1, max_value=3)) for _ in a.axes)
        res = tile(a, reps)
        assert res.axes == tuple(Axis(ax.size * r, ax.name) for ax, r in zip(a.axes, reps))
        np.testing.assert_array_equal(res.array, jnp.tile(a.array, reps))


class TestConcatenate:
    def test_along_first_axis_by_default(self, mat):
        res = concatenate([mat, mat + 100])
        assert_named(res, (Axis(6, "x"), Axis(4, "y")), jnp.concatenate([mat.array, mat.array + 100]))

    def test_along_named_axis_with_different_sizes(self, mat):
        other = ones((Axis(3, "x"), Axis(2, "y")))
        res = concatenate([mat, other], "y")
        assert_named(res, (Axis(3, "x"), Axis(6, "y")), jnp.concatenate([mat.array, other.array], axis=1))

    def test_axis_by_negative_position(self, mat):
        assert concatenate([mat, mat], -1).axes == (Axis(3, "x"), Axis(8, "y"))

    def test_inputs_with_permuted_axes_are_realigned(self, mat):
        other = ones((Axis(3, "x"), Axis(2, "y")))
        res = concatenate([mat, transpose(other)], "y")
        assert res.axes == (Axis(3, "x"), Axis(6, "y"))
        np.testing.assert_array_equal(res.array, jnp.concatenate([mat.array, other.array], axis=1))

    def test_many_arrays(self, mat):
        assert concatenate([mat] * 4, "x").axes[0] == Axis(12, "x")

    def test_dtype(self, mat):
        assert concatenate([mat, mat], "x", dtype=jnp.int32).array.dtype == jnp.int32

    def test_concat_is_an_alias(self, mat):
        a = concatenate([mat, mat], "y")
        b = concat([mat, mat], "y")
        assert a.axes == b.axes
        np.testing.assert_array_equal(a.array, b.array)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one array"):
            concatenate([])

    def test_zero_dimensional_raises(self):
        with pytest.raises(ValueError, match="zero-dimensional"):
            concatenate([array(jnp.array(1.0), ())])

    def test_unknown_axis_raises(self, mat):
        with pytest.raises(ValueError, match="not found"):
            concatenate([mat, mat], "nope")

    def test_size_mismatch_on_non_joined_axis_raises(self, mat):
        with pytest.raises(ValueError, match="axis 'x'"):
            concatenate([mat, ones((Axis(5, "x"), Axis(4, "y")))], "y")

    def test_mismatched_axis_names_raise(self, mat):
        other = ones((Axis(3, "x"), Axis(4, "z")))
        with pytest.raises(ValueError, match="no axis named 'y'"):
            concatenate([mat, other], "x")

    @given(named_array_and_axis(), st.data())
    def test_inverse_of_splitting(self, arr_axis, data):
        a, ax = arr_axis
        assume(ax.size >= 2)
        idx = a.axes.index(ax)
        cut = data.draw(st.integers(min_value=1, max_value=ax.size - 1))
        left = array(
            jnp.take(a.array, jnp.arange(cut), axis=idx),
            tuple(x if x is not ax else Axis(cut, ax.name) for x in a.axes),
        )
        right = array(
            jnp.take(a.array, jnp.arange(cut, ax.size), axis=idx),
            tuple(x if x is not ax else Axis(ax.size - cut, ax.name) for x in a.axes),
        )
        res = concatenate([left, right], ax.name)
        assert res.axes == a.axes
        np.testing.assert_array_equal(res.array, a.array)


class TestBlock:
    def test_bare_array_is_returned(self, mat):
        res = block(mat)
        assert_named(res, mat.axes, mat.array)

    def test_one_dimensional_nesting_defaults_to_last_axis(self, mat):
        res = block([mat, mat])
        assert_named(res, (Axis(3, "x"), Axis(8, "y")), jnp.block([mat.array, mat.array]))

    def test_two_by_two_with_explicit_axes(self, mat):
        a, b, c, d = mat, mat + 1, mat + 2, mat + 3
        res = block([[a, b], [c, d]], axes=("x", "y"))
        expected = jnp.block([[a.array, b.array], [c.array, d.array]])
        assert_named(res, (Axis(6, "x"), Axis(8, "y")), expected)

    def test_two_by_two_default_axes_are_the_trailing_pair(self, mat):
        res = block([[mat, mat], [mat, mat]])
        assert res.axes == (Axis(6, "x"), Axis(8, "y"))

    def test_tuples_work_like_lists(self, mat):
        assert block((mat, mat), axes=("y",)).axes == (Axis(3, "x"), Axis(8, "y"))

    def test_uneven_nesting_raises(self, mat):
        with pytest.raises(ValueError, match="same depth"):
            block([[mat], mat])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty lists"):
            block([])

    def test_axes_length_must_match_depth(self, mat):
        with pytest.raises(ValueError, match="nested 2 deep but 1 axes"):
            block([[mat, mat]], axes=("x",))


# array value operations


class TestAstype:
    def test_changes_dtype_and_keeps_axes(self, mat):
        res = astype(mat, jnp.int32)
        assert res.axes == mat.axes
        assert res.array.dtype == jnp.int32
        np.testing.assert_array_equal(res.array, mat.array.astype(jnp.int32))

    def test_to_complex(self, mat):
        assert astype(mat, jnp.complex64).array.dtype == jnp.complex64

    def test_same_dtype_is_value_preserving(self, mat):
        res = astype(mat, mat.array.dtype, copy=True)
        np.testing.assert_array_equal(res.array, mat.array)

    @given(named_arrays(min_dims=0, max_dims=3))
    def test_roundtrip_through_wider_dtype(self, a):
        back = astype(astype(a, jnp.float32), jnp.float32)
        assert back.axes == a.axes
        np.testing.assert_array_equal(back.array, a.array)


class TestIsclose:
    def test_values_and_axes(self, mat):
        res = isclose(mat, mat + 1e-9)
        assert_named(res, mat.axes, jnp.ones((3, 4), dtype=bool))

    def test_tolerances(self, mat):
        assert not isclose(mat, mat + 0.1).array.any()
        assert isclose(mat, mat + 0.1, atol=0.2).array.all()
        assert isclose(mat + 0.0, mat * (1 + 1e-3), rtol=1e-2).array.all()

    def test_equal_nan(self):
        x = array(jnp.array([jnp.nan, 1.0]), (Axis(2, "x"),))
        np.testing.assert_array_equal(isclose(x, x).array, jnp.array([False, True]))
        np.testing.assert_array_equal(isclose(x, x, equal_nan=True).array, jnp.array([True, True]))

    def test_broadcasts_over_disjoint_axes(self):
        a = array(jnp.arange(3.0), (Axis(3, "x"),))
        b = array(jnp.arange(4.0), (Axis(4, "y"),))
        res = isclose(a, b)
        assert res.axes == (Axis(3, "x"), Axis(4, "y"))
        np.testing.assert_array_equal(res.array, jnp.isclose(jnp.arange(3.0)[:, None], jnp.arange(4.0)[None, :]))

    @given(named_arrays(min_dims=0, max_dims=3))
    def test_reflexive(self, a):
        assert bool(isclose(a, a).array.all())


class TestAllclose:
    def test_scalar_inputs(self):
        assert bool(allclose(1.0, 1.0).array)
        assert not bool(allclose(1.0, 2.0).array)

    def test_arrays_reduce_to_a_scalar(self, mat):
        res = allclose(mat, mat + 1e-9)
        assert_named(res, (), jnp.array(True))

    def test_arrays_detect_a_difference(self, mat):
        assert not bool(allclose(mat, mat + 1.0).array)

    @given(named_arrays(min_dims=1, max_dims=3))
    def test_reflexive(self, a):
        assert bool(allclose(a, a).array)


class TestArrayEqual:
    def test_equal_arrays(self, mat):
        res = array_equal(mat, mat + 0)
        assert res.axes == ()
        assert bool(res.array)

    def test_different_values(self, mat):
        assert not bool(array_equal(mat, mat + 1).array)

    def test_axis_order_is_irrelevant(self, mat):
        assert bool(array_equal(mat, transpose(mat)).array)

    def test_different_axes_are_not_equal(self, mat):
        assert not bool(array_equal(mat, ones(Axis(5, "z"))).array)

    def test_same_names_different_sizes_are_not_equal(self):
        a = ones(Axis(3, "x"))
        b = ones(Axis(4, "x"))
        assert not bool(array_equal(a, b).array)

    def test_nan_handling(self):
        x = array(jnp.array([jnp.nan, 1.0]), (Axis(2, "x"),))
        assert not bool(array_equal(x, x).array)
        assert bool(array_equal(x, x, equal_nan=True).array)

    @given(named_arrays(min_dims=1, max_dims=3))
    def test_reflexive_and_sensitive_to_a_change(self, a):
        assert bool(array_equal(a, a).array)
        assert not bool(array_equal(a, a + 1.0).array)


class TestArrayEquiv:
    def test_equal(self, mat):
        res = array_equiv(mat, mat + 0)
        assert res.axes == ()
        assert bool(res.array)

    def test_different_values(self, mat):
        assert not bool(array_equiv(mat, mat + 1).array)

    def test_axis_order_is_irrelevant(self, mat):
        assert bool(array_equiv(mat, transpose(mat)).array)

    def test_incompatible_sizes_give_false_rather_than_raising(self):
        a = ones(Axis(3, "x"))
        b = ones(Axis(4, "x"))
        res = array_equiv(a, b)
        assert res.axes == ()
        assert not bool(res.array)


class TestClip:
    def test_both_bounds_match_jnp(self, mat):
        res = clip(mat, min=2.0, max=9.0)
        assert_named(res, mat.axes, jnp.clip(mat.array, 2.0, 9.0))

    def test_min_only(self, mat):
        np.testing.assert_array_equal(clip(mat, min=5.0).array, jnp.maximum(mat.array, 5.0))

    def test_max_only(self, mat):
        np.testing.assert_array_equal(clip(mat, max=5.0).array, jnp.minimum(mat.array, 5.0))

    def test_no_bounds_is_identity(self, mat):
        assert_named(clip(mat), mat.axes, mat.array)

    def test_min_greater_than_max_returns_max(self, mat):
        np.testing.assert_array_equal(clip(mat, min=5.0, max=2.0).array, jnp.full((3, 4), 2.0))

    def test_array_bounds_broadcast_along_named_axis(self, mat):
        lo = array(jnp.array([0.0, 5.0, 10.0]), (Axis(3, "x"),))
        res = clip(mat, min=lo)
        assert res.axes == mat.axes
        np.testing.assert_array_equal(res.array, jnp.maximum(mat.array, lo.array[:, None]))

    def test_complex_input_raises(self, cplx):
        with pytest.raises(ValueError, match="Complex values have no ordering"):
            clip(cplx, min=0.0)

    def test_complex_bound_raises(self, mat):
        with pytest.raises(ValueError, match="Complex values have no ordering"):
            clip(mat, max=1j)

    @given(named_arrays(min_dims=0, max_dims=3), st.floats(-50, 0, width=32), st.floats(0, 50, width=32))
    def test_result_is_within_bounds_and_idempotent(self, a, lo, hi):
        res = clip(a, min=lo, max=hi)
        assert float(res.array.min()) >= lo
        assert float(res.array.max()) <= hi
        np.testing.assert_array_equal(res.array, jnp.clip(a.array, lo, hi))
        np.testing.assert_array_equal(clip(res, min=lo, max=hi).array, res.array)


class TestRound:
    def test_default_decimals(self):
        a = array(jnp.array([0.5, 1.5, 2.5, 1.4]), (Axis(4, "x"),))
        assert_named(round(a), a.axes, jnp.array([0.0, 2.0, 2.0, 1.0]))  # round half to even

    def test_decimals(self):
        a = array(jnp.array([1.234, 2.345]), (Axis(2, "x"),))
        np.testing.assert_allclose(round(a, 2).array, jnp.round(a.array, 2))

    def test_around_is_an_alias(self, mat):
        np.testing.assert_array_equal(around(mat + 0.3).array, round(mat + 0.3).array)
        assert around(mat).axes == mat.axes

    @given(named_arrays(min_dims=0, max_dims=3), st.integers(min_value=0, max_value=3))
    def test_matches_jnp_and_is_idempotent(self, a, decimals):
        res = round(a, decimals)
        assert res.axes == a.axes
        np.testing.assert_array_equal(res.array, jnp.round(a.array, decimals))
        np.testing.assert_array_equal(round(res, decimals).array, res.array)


class TestAngle:
    def test_angle_matches_jnp(self, cplx):
        res = angle(cplx)
        assert res.axes == cplx.axes
        np.testing.assert_allclose(res.array, jnp.angle(cplx.array))

    def test_angle_in_degrees(self, cplx):
        np.testing.assert_allclose(angle(cplx, deg=True).array, jnp.angle(cplx.array, deg=True), rtol=1e-5)

    def test_angle_of_real_input_is_zero_or_pi(self):
        a = array(jnp.array([-1.0, 1.0]), (Axis(2, "x"),))
        np.testing.assert_allclose(angle(a).array, jnp.array([jnp.pi, 0.0]), rtol=1e-6)


class TestCompress:
    cond = array(jnp.array([True, False, True]), (Axis(3, "x"),))

    def test_selects_slices(self, mat):
        res = compress(self.cond, mat, "x")
        assert_named(res, (Axis(2, "x"), Axis(4, "y")), mat.array[jnp.array([0, 2])])

    def test_axis_is_taken_from_condition_name(self, mat):
        res = compress(self.cond, mat)
        assert res.axes == (Axis(2, "x"), Axis(4, "y"))

    def test_axis_is_inferred_for_one_dimensional_input(self):
        a = array(jnp.array([10.0, 20.0, 30.0]), (Axis(3, "x"),))
        res = compress(array(jnp.array([False, True, True]), (Axis(3),)), a)
        assert_named(res, (Axis(2, "x"),), jnp.array([20.0, 30.0]))

    def test_other_axis(self, mat):
        cond = array(jnp.array([True, True, False, True]), (Axis(4, "y"),))
        res = compress(cond, mat, "y")
        assert_named(res, (Axis(3, "x"), Axis(3, "y")), mat.array[:, jnp.array([0, 1, 3])])

    def test_size_pads_with_fill_value(self, mat):
        cond = array(jnp.array([True, False, False]), (Axis(3, "x"),))
        res = compress(cond, mat, "x", size=2, fill_value=-1.0)
        np.testing.assert_array_equal(res.array, jnp.stack([mat.array[0], jnp.full(4, -1.0)]))

    def test_size_truncates_surplus_selections(self, mat):
        res = compress(self.cond, mat, "x", size=1)
        np.testing.assert_array_equal(res.array, mat.array[:1])

    def test_works_under_jit_with_explicit_size(self, mat):
        res = jax.jit(lambda c, m: compress(c, m, "x", size=2))(self.cond, mat)
        np.testing.assert_array_equal(res.array, mat.array[jnp.array([0, 2])])

    def test_under_jit_without_size_raises(self, mat):
        with pytest.raises(ValueError, match="explicit `size`"):
            jax.jit(lambda c, m: compress(c, m, "x"))(self.cond, mat)

    def test_nothing_selected_raises(self, mat):
        cond = array(jnp.array([False, False, False]), (Axis(3, "x"),))
        with pytest.raises(ValueError, match="non-empty"):
            compress(cond, mat, "x")

    def test_non_1d_condition_raises(self, mat):
        with pytest.raises(ValueError, match="must be 1D"):
            compress(ones(mat.axes, dtype=bool), mat, "x")

    def test_axis_required_when_it_cannot_be_inferred(self, mat):
        with pytest.raises(ValueError, match="`axis` is required"):
            compress(array(jnp.array([True, False, True]), (Axis(3),)), mat)

    def test_unknown_axis_raises(self, mat):
        with pytest.raises(ValueError, match="not found"):
            compress(self.cond, mat, "nope")

    def test_condition_name_must_match_selected_axis(self, mat):
        with pytest.raises(ValueError, match="does not match"):
            compress(array(jnp.array([True, False, True, True]), (Axis(4, "z"),)), mat, "y")

    @given(named_array_and_axis(), st.data())
    def test_result_size_is_the_number_of_true_entries(self, arr_axis, data):
        a, ax = arr_axis
        flags = data.draw(st.lists(st.booleans(), min_size=ax.size, max_size=ax.size))
        assume(any(flags))
        cond = array(jnp.array(flags), (Axis(ax.size, ax.name),))
        res = compress(cond, a, ax.name)
        idx = a.axes.index(ax)
        assert res.axes[idx] == Axis(sum(flags), ax.name)
        np.testing.assert_array_equal(res.array, jnp.compress(jnp.array(flags), a.array, axis=idx))


class TestWhere:
    def test_three_argument_form_matches_jnp(self, mat):
        cond = array(mat.array > 5, mat.axes)
        res = where(cond, mat, -1.0)
        assert_named(res, mat.axes, jnp.where(mat.array > 5, mat.array, -1.0))

    def test_three_argument_form_broadcasts_named_axes(self, mat):
        cond = array(jnp.array([True, False, True]), (Axis(3, "x"),))
        res = where(cond, mat, 0.0)
        assert res.axes == mat.axes
        np.testing.assert_array_equal(res.array, jnp.where(cond.array[:, None], mat.array, 0.0))

    def test_three_argument_form_with_scalar_branches(self):
        cond = array(jnp.array([True, False]), (Axis(2, "x"),))
        assert_named(where(cond, 1.0, 2.0), (Axis(2, "x"),), jnp.array([1.0, 2.0]))

    def test_one_argument_form_matches_nonzero(self, mat):
        cond = array(mat.array > 8, mat.axes)
        rows, cols = where(cond)
        exp_rows, exp_cols = jnp.nonzero(mat.array > 8)
        assert rows.axes == cols.axes == (Axis(3),)
        np.testing.assert_array_equal(rows.array, exp_rows)
        np.testing.assert_array_equal(cols.array, exp_cols)

    def test_one_argument_form_returns_one_index_array_per_axis(self):
        cond = array(
            jnp.zeros((2, 3, 4), dtype=bool).at[1, 2, 3].set(True), (Axis(2, "a"), Axis(3, "b"), Axis(4, "c"))
        )
        idx = where(cond)
        assert [int(i.array[0]) for i in idx] == [1, 2, 3]

    def test_size_as_int_pads_with_fill_value(self, mat):
        cond = array(mat.array > 8, mat.axes)
        rows, cols = where(cond, size=5, fill_value=-1)
        assert rows.axes == (Axis(5),)
        np.testing.assert_array_equal(rows.array, jnp.array([2, 2, 2, -1, -1]))
        np.testing.assert_array_equal(cols.array, jnp.array([1, 2, 3, -1, -1]))

    def test_size_as_axis_names_the_index_axis(self, mat):
        cond = array(mat.array > 8, mat.axes)
        rows, cols = where(cond, size=Axis(3, "hits"))
        assert rows.axes == cols.axes == (Axis(3, "hits"),)

    def test_one_argument_form_under_jit_with_size(self, mat):
        cond = array(mat.array > 8, mat.axes)
        rows, _ = jax.jit(lambda c: where(c, size=3))(cond)
        np.testing.assert_array_equal(rows.array, jnp.array([2, 2, 2]))

    def test_one_argument_form_under_jit_without_size_raises(self, mat):
        cond = array(mat.array > 8, mat.axes)
        with pytest.raises(ValueError, match="explicit `size`"):
            jax.jit(lambda c: where(c))(cond)

    def test_nothing_true_raises(self, mat):
        with pytest.raises(ValueError, match="non-empty"):
            where(array(mat.array > 100, mat.axes))

    def test_scalar_condition_raises(self):
        with pytest.raises(ValueError, match="zero-dimensional"):
            where(array(jnp.array(True), ()))

    @pytest.mark.parametrize("size", [1.5, True])
    def test_non_integer_size_raises(self, mat, size):
        with pytest.raises(TypeError, match="`size` must be an int or Axis"):
            where(array(mat.array > 5, mat.axes), size=size)

    def test_only_one_of_x_and_y_raises(self, mat):
        with pytest.raises(ValueError, match="both or neither"):
            where(array(mat.array > 5, mat.axes), mat)

    def test_size_with_three_arguments_raises(self, mat):
        cond = array(mat.array > 5, mat.axes)
        with pytest.raises(ValueError, match="cannot be used in the three-argument form"):
            where(cond, mat, mat, size=2)

    @given(named_arrays(min_dims=1, max_dims=3))
    def test_indices_point_at_true_entries(self, a):
        mask = a.array > 0
        assume(bool(mask.any()))
        idx = where(array(mask, a.axes))
        assert len(idx) == a.ndim
        picked = tuple(i.array for i in idx)
        assert bool(mask[picked].all())
        assert idx[0].array.shape == (int(mask.sum()),)


class TestTopK:
    def test_values_and_indices(self):
        a = array(jnp.array([3.0, 1.0, 2.0, 5.0]), (Axis(4, "y"),))
        values, indices = top_k(a, 2, "y")
        assert_named(values, (Axis(2, "y"),), jnp.array([5.0, 3.0]))
        assert_named(indices, (Axis(2, "y"),), jnp.array([3, 0]))

    def test_other_axes_are_kept_in_place(self, mat):
        values, indices = top_k(mat, 2, "x")
        assert values.axes == indices.axes == (Axis(2, "x"), Axis(4, "y"))
        np.testing.assert_array_equal(values.array, jnp.array([mat.array[2], mat.array[1]]))

    def test_new_axis_by_name_and_object(self, mat):
        values, _ = top_k(mat, 2, "y", "k")
        assert values.axes == (Axis(3, "x"), Axis(2, "k"))
        values, _ = top_k(mat, 2, "y", Axis(2, "best"))
        assert values.axes == (Axis(3, "x"), Axis(2, "best"))

    def test_unknown_axis_raises(self, mat):
        with pytest.raises(ValueError, match="not found"):
            top_k(mat, 1, "nope")

    @given(named_array_and_axis(), st.data())
    def test_values_are_sorted_descending_and_match_indices(self, arr_axis, data):
        a, ax = arr_axis
        k = data.draw(st.integers(min_value=1, max_value=ax.size))
        values, indices = top_k(a, k, ax.name)
        idx = a.axes.index(ax)
        assert values.axes[idx].size == k
        assert bool((jnp.diff(values.array, axis=idx) <= 0).all())
        np.testing.assert_array_equal(values.array, jnp.take_along_axis(a.array, indices.array, axis=idx))


class TestSearchsorted:
    sorted_a = array(jnp.arange(5.0), (Axis(5, "z"),))

    def test_matches_jnp_and_takes_axes_from_v(self):
        v = array(jnp.array([1.5, 2.5, 9.0]), (Axis(3, "x"),))
        res = searchsorted(self.sorted_a, v)
        assert_named(res, v.axes, jnp.searchsorted(jnp.arange(5.0), v.array))

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_side(self, side):
        v = array(jnp.array([0.0, 2.0, 4.0]), (Axis(3, "x"),))
        res = searchsorted(self.sorted_a, v, side=side)
        np.testing.assert_array_equal(res.array, jnp.searchsorted(jnp.arange(5.0), v.array, side=side))

    def test_scalar_v(self):
        res = searchsorted(self.sorted_a, 2.5)
        assert_named(res, (), jnp.array(3))

    def test_with_sorter(self):
        a = array(jnp.array([3.0, 1.0, 2.0]), (Axis(3, "x"),))
        sorter = array(jnp.array([1, 2, 0]), (Axis(3, "x"),))
        v = array(jnp.array([1.5]), (Axis(1, "v"),))
        res = searchsorted(a, v, sorter=sorter)
        np.testing.assert_array_equal(res.array, jnp.searchsorted(a.array, v.array, sorter=sorter.array))

    def test_multidimensional_a_raises(self, mat):
        with pytest.raises(ValueError, match="rank-1"):
            searchsorted(mat, mat)

    @given(st.lists(st.floats(-50, 50, width=32), min_size=1, max_size=10), st.floats(-60, 60, width=32))
    def test_insertion_keeps_order(self, values, query):
        data = jnp.sort(jnp.array(values, dtype=jnp.float32))
        a = array(data, (Axis(len(values), "x"),))
        pos = int(searchsorted(a, query).array)
        merged = jnp.insert(data, pos, jnp.float32(query))
        assert bool((jnp.diff(merged) >= 0).all())


class TestBincount:
    def test_counts_minlength_and_named_output_axis(self):
        values = array(jnp.array([0, 1, 1, 3, 3, 3]), (Axis(6, "sample"),))
        result = bincount(values, minlength=5, axis="bin")

        assert_named(result, (Axis(5, "bin"),), jnp.bincount(values.array, minlength=5))

    def test_weighted_counts_and_fixed_length(self):
        values = array(jnp.array([0, 1, 1, 2]), (Axis(4, "sample"),))
        weights = array(jnp.array([0.5, 2.0, 3.0, 4.0]), (Axis(4, "weight"),))
        result = bincount(values, weights, length=5, axis=Axis(5, "bin"))

        expected = jnp.bincount(values.array, weights=weights.array, length=5)
        assert_named(result, (Axis(5, "bin"),), expected)

    def test_bad_weights_axes_and_non_1d_inputs_raise(self):
        values = array(jnp.array([0, 1, 1]), (Axis(3, "sample"),))
        with pytest.raises(ValueError, match="same number of elements"):
            bincount(values, array(jnp.ones(2), (Axis(2, "weight"),)))

        with pytest.raises(ValueError, match="axis has size 3, expected 2"):
            bincount(values, length=2, axis=Axis(3, "bin"))

        with pytest.raises(ValueError, match="one-dimensional"):
            bincount(array(jnp.ones((2, 2), dtype=jnp.int32), (Axis(2, "x"), Axis(2, "y"))))


class TestHistogram:
    def test_counts_and_edges_match_jnp(self):
        values = array(jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]), (Axis(5, "sample"),))
        counts, edges = histogram(values, bins=2, range=(0.0, 2.0), axis="bin")
        expected_counts, expected_edges = jnp.histogram(values.array, bins=2, range=(0.0, 2.0))

        assert_named(counts, (Axis(2, "bin"),), expected_counts)
        assert_named(edges, (Axis(3, "bin"),), expected_edges)

    def test_named_bins_infer_output_axis_name_and_weights_are_applied(self):
        values = array(jnp.array([0.0, 1.0, 2.0, 3.0]), (Axis(4, "sample"),))
        weights = array(jnp.array([1.0, 2.0, 3.0, 4.0]), (Axis(4, "weight"),))
        bins = array(jnp.array([0.0, 2.0, 4.0]), (Axis(3, "edges"),))
        counts, edges = histogram(values, bins=bins, weights=weights)

        expected_counts, expected_edges = jnp.histogram(values.array, bins=bins.array, weights=weights.array)
        assert_named(counts, (Axis(2, "edges"),), expected_counts)
        assert_named(edges, (Axis(3, "edges"),), expected_edges)

    def test_density_and_bad_weight_or_axis_sizes(self):
        values = array(jnp.array([0.0, 1.0, 2.0, 3.0]), (Axis(4, "sample"),))
        weights = array(jnp.ones(3), (Axis(3, "weight"),))
        with pytest.raises(ValueError, match="same number of elements"):
            histogram(values, weights=weights)

        with pytest.raises(ValueError, match="axis has size"):
            histogram(values, bins=2, axis=Axis(3, "bin"))

        counts, _ = histogram(values, bins=2, density=True)
        expected, _ = jnp.histogram(values.array, bins=2, density=True)
        np.testing.assert_allclose(counts.array, expected)


class TestHistogramBinEdges:
    def test_edges_match_jnp_and_named_bin_edges_infer_axis(self):
        values = array(jnp.array([0.0, 1.0, 2.0, 3.0]), (Axis(4, "sample"),))
        edges = histogram_bin_edges(values, bins=4, range=(0.0, 4.0), axis="bin")
        expected = jnp.histogram_bin_edges(values.array, bins=4, range=(0.0, 4.0))
        assert_named(edges, (Axis(5, "bin"),), expected)

        named_bins = array(jnp.array([0.0, 1.0, 3.0]), (Axis(3, "limits"),))
        inferred = histogram_bin_edges(values, bins=named_bins)
        assert inferred.axes == (Axis(3, "limits"),)

    def test_axis_object_must_match_edge_count(self):
        values = array(jnp.arange(4.0), (Axis(4, "sample"),))
        with pytest.raises(ValueError, match="axis has size 4, expected 3"):
            histogram_bin_edges(values, bins=2, axis=Axis(4, "bin"))


class TestHistogram2D:
    def test_counts_and_edges_match_jnp(self):
        x = array(jnp.array([0.0, 0.5, 1.0, 1.5]), (Axis(4, "x_sample"),))
        y = array(jnp.array([0.0, 1.0, 0.5, 1.5]), (Axis(4, "y_sample"),))
        result, xedges, yedges = histogram2d(
            x, y, bins=(2, 3), range=((0.0, 2.0), (0.0, 2.0)), axes=("x_bin", "y_bin")
        )
        expected, expected_xedges, expected_yedges = jnp.histogram2d(
            x.array, y.array, bins=(2, 3), range=((0.0, 2.0), (0.0, 2.0))
        )

        assert_named(result, (Axis(2, "x_bin"), Axis(3, "y_bin")), expected)
        assert_named(xedges, (Axis(3, "x_bin"),), expected_xedges)
        assert_named(yedges, (Axis(4, "y_bin"),), expected_yedges)

    def test_weights_density_and_invalid_sizes(self):
        x = array(jnp.array([0.0, 0.5, 1.0]), (Axis(3, "x"),))
        y = array(jnp.array([0.0, 0.5, 1.0]), (Axis(3, "y"),))
        weights = array(jnp.array([1.0, 2.0, 3.0]), (Axis(3, "weight"),))
        result, _, _ = histogram2d(x, y, bins=2, weights=weights, density=True, axes=("row", "col"))
        expected, _, _ = jnp.histogram2d(x.array, y.array, bins=2, weights=weights.array, density=True)
        assert_named(result, (Axis(2, "row"), Axis(2, "col")), expected)

        with pytest.raises(ValueError, match="same number of elements"):
            histogram2d(x, array(jnp.arange(2.0), (Axis(2, "y"),)))
        with pytest.raises(ValueError, match="axis has size"):
            histogram2d(x, y, bins=2, axes=(Axis(3, "row"), "col"))


class TestHistogramDD:
    def test_counts_and_edges_match_jnp(self):
        samples = jnp.array([[0.0, 0.0], [0.25, 0.5], [0.75, 0.25], [1.0, 1.0]])
        named_samples = array(samples, (Axis(4, "sample"), Axis(2, "coordinate")))
        result, edges = histogramdd(named_samples, bins=(2, 3), axes=("x_bin", "y_bin"))
        expected, expected_edges = jnp.histogramdd(samples, bins=(2, 3))

        assert_named(result, (Axis(2, "x_bin"), Axis(3, "y_bin")), expected)
        assert len(edges) == 2
        assert_named(edges[0], (Axis(3, "x_bin"),), expected_edges[0])
        assert_named(edges[1], (Axis(4, "y_bin"),), expected_edges[1])

    def test_weighted_density_histogram(self):
        samples = array(
            jnp.array([[0.0, 0.0], [0.25, 0.5], [0.75, 0.25], [1.0, 1.0]]), (Axis(4, "sample"), Axis(2, "coordinate"))
        )
        weights = array(jnp.array([1.0, 2.0, 3.0, 4.0]), (Axis(4, "weight"),))
        result, edges = histogramdd(samples, bins=2, weights=weights, density=True, axes=("row", "col"))
        expected, _ = jnp.histogramdd(samples.array, bins=2, weights=weights.array, density=True)
        assert_named(result, (Axis(2, "row"), Axis(2, "col")), expected)
        assert [edge.axes for edge in edges] == [(Axis(3, "row"),), (Axis(3, "col"),)]

    def test_invalid_sample_weights_and_axis_count_raise(self):
        samples = array(jnp.ones((3, 2)), (Axis(3, "sample"), Axis(2, "coordinate")))
        with pytest.raises(ValueError, match="one entry per sample"):
            histogramdd(samples, weights=array(jnp.ones(2), (Axis(2, "weight"),)))

        with pytest.raises(ValueError, match="sample must have shape"):
            histogramdd(array(jnp.ones(3), (Axis(3, "sample"),)))

        with pytest.raises(ValueError, match="got 1 output axes"):
            histogramdd(samples, bins=2, axes=("only_one",))
