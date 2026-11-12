import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ainmax._src.axis import Axis
from ainmax._src.feidhm.constructors import array
from ainmax._src.hof import checkpointed_scan, fold, scan, vmap


def _named(data, *axes):
    return array(jnp.asarray(data), axes)


def _assert_close(got, want, tol=1e-5):
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=tol, atol=tol)


class TestVmap:
    def test_maps_named_axis_and_preserves_output_axes(self):
        batch, feature = Axis(3, "batch"), Axis(2, "feature")
        x = _named(jnp.arange(6.0).reshape(3, 2), batch, feature)

        result = vmap(lambda row: row * 2, "batch")(x)

        assert result.axes == (batch, feature)
        _assert_close(result.array, x.array * 2)

    def test_maps_multiple_named_axes_in_selector_order(self):
        batch, time, feature = Axis(2, "batch"), Axis(3, "time"), Axis(4, "feature")
        x = _named(jnp.arange(24.0).reshape(2, 3, 4), batch, time, feature)

        result = vmap(lambda value: value + 1, ("time", "batch"))(x)

        assert result.axes == (time, batch, feature)
        _assert_close(result.array, jnp.transpose(x.array, (1, 0, 2)) + 1)

    def test_maps_nested_named_pytrees_and_reuses_unmapped_arguments(self):
        batch, feature = Axis(3, "batch"), Axis(2, "feature")
        x = _named(jnp.arange(6.0).reshape(3, 2), batch, feature)
        y = _named(jnp.arange(6.0, 12.0).reshape(3, 2), batch, feature)
        bias = _named(jnp.array([10.0, 20.0]), feature)

        result = vmap(lambda pair, offset: {"sum": pair[0] + pair[1] + offset}, "batch")((x, y), bias)

        assert result["sum"].axes == (batch, feature)
        _assert_close(result["sum"].array, x.array + y.array + bias.array)

    def test_unmapped_named_input_is_available_in_each_mapped_call(self):
        batch, feature = Axis(3, "batch"), Axis(2, "feature")
        x = _named(jnp.arange(6.0).reshape(3, 2), batch, feature)
        offset = _named(jnp.array([5.0, 7.0]), feature)

        result = vmap(lambda row, bias: row + bias, "batch")(x, offset)

        assert result.axes == (batch, feature)
        _assert_close(result.array, x.array + offset.array)

    def test_out_axes_reorders_named_result_axes(self):
        batch, feature = Axis(3, "batch"), Axis(2, "feature")
        x = _named(jnp.arange(6.0).reshape(3, 2), batch, feature)

        result = vmap(lambda row: row, "batch", out_axes=("feature", "batch"))(x)

        assert result.axes == (feature, batch)
        _assert_close(result.array, x.array.T)

    def test_plain_array_output_gets_anonymous_unmapped_axes(self):
        batch = Axis(3, "batch")
        x = _named(jnp.arange(3.0), batch)

        result = vmap(lambda item: jnp.array([item.array, item.array + 1]), "batch")(x)

        assert result.axes == (batch, Axis(2))
        _assert_close(result.array, jnp.stack((x.array, x.array + 1), axis=-1))

    def test_vmap_composes_with_jit_and_grad(self):
        batch, feature = Axis(3, "batch"), Axis(2, "feature")
        x = _named(jnp.arange(6.0).reshape(3, 2), batch, feature)
        mapped = vmap(lambda row: jnp.sum(row.array**2), "batch")

        result = jax.jit(mapped)(x)
        grad = jax.grad(lambda value: mapped(value).array.sum())(x)

        assert result.axes == (batch,)
        assert grad.axes == x.axes
        _assert_close(result.array, jnp.sum(x.array**2, axis=1))
        _assert_close(grad.array, 2 * x.array)

    def test_invalid_callable_empty_axes_and_missing_axis_raise(self):
        with pytest.raises(TypeError, match="expected a callable"):
            vmap(1, "batch")
        with pytest.raises(ValueError, match="at least one axis"):
            vmap(lambda x: x, ())
        with pytest.raises(ValueError, match="no argument has axis"):
            vmap(lambda x: x, "missing")(_named(jnp.ones(2), Axis(2, "feature")))

    def test_inconsistent_sizes_names_and_duplicate_selectors_raise(self):
        with pytest.raises(ValueError, match="inconsistent sizes"):
            vmap(lambda x, y: x + y, 0)(
                _named(jnp.ones((2, 3)), Axis(2, "batch"), Axis(3, "x")),
                _named(jnp.ones((4, 3)), Axis(4, "batch"), Axis(3, "x")),
            )
        with pytest.raises(ValueError, match="differently named"):
            vmap(lambda x, y: x + y, 0)(
                _named(jnp.ones((2,)), Axis(2, "batch")),
                _named(jnp.ones((2,)), Axis(2, "other")),
            )
        with pytest.raises(ValueError, match="same axis"):
            vmap(lambda x: x, ("batch", 0))(_named(jnp.ones((2, 3)), Axis(2, "batch"), Axis(3, "x")))

    def test_positional_axis_out_of_range_raises(self):
        x = _named(jnp.ones(3), Axis(3, "feature"))
        with pytest.raises(ValueError, match="out of range"):
            vmap(lambda item: item, 1)(x)


class TestScan:
    def test_scan_accumulates_and_stacks_named_output(self):
        time, feature = Axis(4, "time"), Axis(2, "feature")
        xs = _named(jnp.arange(8.0).reshape(4, 2), time, feature)
        init = _named(jnp.zeros(2), feature)

        carry, ys = scan(lambda carry, x: (carry + x, carry + x), "time")(init, xs)

        assert carry.axes == init.axes
        assert ys.axes == (time, feature)
        _assert_close(carry.array, jnp.sum(xs.array, axis=0))
        _assert_close(ys.array, jnp.cumsum(xs.array, axis=0))

    def test_scan_passes_unmapped_arguments_and_supports_kwargs(self):
        time, feature = Axis(3, "time"), Axis(2, "feature")
        xs = _named(jnp.arange(6.0).reshape(3, 2), time, feature)
        offset = _named(jnp.array([2.0, 5.0]), feature)

        carry, ys = scan(lambda c, x, bias: (c + x, x + bias), "time")(_named(jnp.zeros(2), feature), xs, bias=offset)

        assert ys.axes == (time, feature)
        _assert_close(carry.array, xs.array.sum(axis=0))
        _assert_close(ys.array, xs.array + offset.array)

    def test_scan_stacks_pytrees_and_plain_array_outputs(self):
        time = Axis(3, "time")
        xs = _named(jnp.arange(3.0), time)

        _, ys = scan(
            lambda carry, x: (carry + x.array, (x.array, {"plain": jnp.stack((x.array, x.array + 1))})),
            "time",
        )(jnp.array(0.0), xs)

        assert ys[0].axes == (time,)
        assert ys[1]["plain"].axes == (time, Axis(2))
        _assert_close(ys[0].array, xs.array)
        _assert_close(ys[1]["plain"].array, jnp.stack((xs.array, xs.array + 1), axis=-1))

    def test_scan_reverse_and_output_axis_reordering(self):
        time, feature = Axis(3, "time"), Axis(2, "feature")
        xs = _named(jnp.arange(6.0).reshape(3, 2), time, feature)

        carry, ys = scan(
            lambda c, x: (c + x, x),
            "time",
            reverse=True,
            out_axes=("feature", "time"),
        )(_named(jnp.zeros(2), feature), xs)

        assert ys.axes == (feature, time)
        _assert_close(carry.array, xs.array.sum(axis=0))
        _assert_close(ys.array, xs.array.T)

    def test_scan_with_axis_only_runs_given_length(self):
        time = Axis(4, "time")
        carry, ys = scan(lambda c: (c + 1, c), time)(jnp.array(0))

        assert carry.shape == ()
        assert ys.axes == (time,)
        _assert_close(carry, 4)
        _assert_close(ys.array, jnp.arange(4))

    def test_scan_supports_remat_checkpoint_blocks_and_jit(self):
        time, feature = Axis(5, "time"), Axis(2, "feature")
        xs = _named(jnp.arange(10.0).reshape(5, 2), time, feature)
        fn = lambda carry, x: (jnp.sin(carry + x.array), carry + x.array)
        init = jnp.ones(2)

        baseline = scan(fn, "time")(init, xs)
        rematerialized = scan(fn, "time", remat=True)(init, xs)
        blocked = scan(fn, "time", block_size=2)(init, xs)
        sqrt_blocked = scan(fn, "time", block_size="sqrt")(init, xs)
        compiled = jax.jit(scan(fn, "time", block_size=2))(init, xs)

        for result in (rematerialized, blocked, sqrt_blocked, compiled):
            assert result[1].axes == baseline[1].axes
            _assert_close(result[0], baseline[0])
            _assert_close(result[1].array, baseline[1].array)

    def test_scan_gradients_match_uncheckpointed_scan(self):
        time = Axis(5, "time")
        xs = _named(jnp.arange(1.0, 6.0), time)
        fn = lambda carry, x: (carry * 0.5 + x.array, carry * x.array)

        def loss(init, block_size):
            carry, ys = scan(fn, "time", block_size=block_size)(init, xs)
            return carry + ys.array.sum()

        expected = jax.grad(lambda init: loss(init, None))(jnp.array(1.0))
        actual = jax.grad(lambda init: loss(init, 2))(jnp.array(1.0))

        _assert_close(actual, expected)

    def test_scan_rejects_bad_callable_axis_and_missing_input_axis(self):
        with pytest.raises(TypeError, match="expected a callable"):
            scan(1, "time")
        with pytest.raises(ValueError, match="exactly one axis"):
            scan(lambda c, x: (c, x), ("time", "batch"))
        with pytest.raises(ValueError, match="no argument has axis"):
            scan(lambda c: (c, c), "time")(0)

    def test_scan_requires_carry_structure_and_axes_to_remain_stable(self):
        time = Axis(2, "time")
        xs = _named(jnp.ones(2), time)
        initial = _named(jnp.zeros(2), Axis(2, "feature"))
        with pytest.raises((TypeError, ValueError), match="carry"):
            scan(
                lambda carry, x: (_named(carry.array, Axis(2, "other")), x),
                "time",
            )(initial, xs)


class TestFold:
    def test_fold_returns_only_final_carry(self):
        time, feature = Axis(4, "time"), Axis(2, "feature")
        xs = _named(jnp.arange(8.0).reshape(4, 2), time, feature)
        init = _named(jnp.ones(2), feature)

        result = fold(lambda carry, x: carry + x, "time")(init, xs)

        assert result.axes == (feature,)
        _assert_close(result.array, init.array + xs.array.sum(axis=0))

    def test_fold_supports_reverse_remat_and_blocking(self):
        time = Axis(5, "time")
        xs = _named(jnp.arange(5.0), time)
        fn = lambda carry, x: carry * 2 + x.array

        expected = fold(fn, "time", reverse=True)(jnp.array(1.0), xs)
        for kwargs in ({"remat": True}, {"block_size": 2}, {"block_size": "sqrt", "reverse": True}):
            result = fold(fn, "time", **kwargs)(jnp.array(1.0), xs)
            reference = fold(fn, "time", reverse=kwargs.get("reverse", False))(jnp.array(1.0), xs)
            _assert_close(result, reference)
        _assert_close(expected, fold(fn, "time", block_size=2, reverse=True)(jnp.array(1.0), xs))

    def test_fold_without_scanned_inputs_and_invalid_multi_axis(self):
        time = Axis(3, "time")
        result = fold(lambda carry: carry + 2, time)(jnp.array(1))

        _assert_close(result, 7)
        with pytest.raises(ValueError, match="exactly one axis"):
            fold(lambda carry, x: carry + x, ("time", "batch"))


class TestCheckpointedScan:
    @pytest.mark.parametrize("block_size", [1, 2, 3, 5, 8, "sqrt"])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_checkpointed_scan_matches_lax_scan_with_tail(self, block_size, reverse):
        xs = jnp.arange(7.0)
        fn = lambda carry, x: (carry + x, carry - x)

        got = checkpointed_scan(fn, jnp.array(1.0), xs, block_size=block_size, reverse=reverse)
        want = jax.lax.scan(fn, jnp.array(1.0), xs, reverse=reverse)

        _assert_close(got[0], want[0])
        _assert_close(got[1], want[1])

    def test_checkpointed_scan_handles_pytrees_and_empty_sequences(self):
        xs = {"left": jnp.arange(4.0), "right": jnp.arange(4.0) + 10}
        fn = lambda carry, x: (carry + x["left"] + x["right"], x["left"] * x["right"])

        got = checkpointed_scan(fn, jnp.array(0.0), xs, block_size=3)
        want = jax.lax.scan(fn, jnp.array(0.0), xs)
        _assert_close(got[0], want[0])
        _assert_close(got[1], want[1])

        empty_carry, empty_ys = checkpointed_scan(lambda c, x: (c + x, x), 1.0, jnp.empty((0,)), block_size="sqrt")
        _assert_close(empty_carry, 1.0)
        assert empty_ys.shape == (0,)

    def test_checkpointed_scan_accepts_length_without_inputs(self):
        carry, ys = checkpointed_scan(lambda c, _: (c + 1, c), 0, length=3, block_size=2)

        _assert_close(carry, 3)
        _assert_close(ys, jnp.arange(3))

    def test_checkpointed_scan_validates_length_shapes_and_block_size(self):
        with pytest.raises(ValueError, match="length.*required"):
            checkpointed_scan(lambda c, x: (c, x), 0, xs=())
        with pytest.raises(ValueError, match="expected leading dimension"):
            checkpointed_scan(lambda c, x: (c, x), 0, jnp.ones((3,)), length=2)
        with pytest.raises(ValueError, match="positive int"):
            checkpointed_scan(lambda c, x: (c, x), 0, jnp.ones((3,)), block_size=0)
