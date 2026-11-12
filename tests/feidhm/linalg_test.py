import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest
from ainmax._src.axis import Axis
from ainmax._src.feidhm.constructors import array
from ainmax._src.feidhm.linalg import (
    cho_factor,
    cho_solve,
    cholesky,
    cond,
    cross,
    det,
    eig,
    eig_general,
    eigh,
    eigh_general,
    eigvals,
    eigvalsh,
    expm,
    expm_frechet,
    hessenberg,
    inv,
    lstsq,
    lu,
    lu_factor,
    lu_general,
    lu_solve,
    matrix_norm,
    matrix_power,
    matrix_rank,
    matrix_transpose,
    norm,
    pinv,
    polar,
    qr,
    schur,
    slogdet,
    solve,
    solve_triangular,
    sqrtm,
    svd,
    svdvals,
    vector_norm,
)


def _named(data, *axes):
    return array(jnp.asarray(data), axes)


def _assert_close(got, want, tol=1e-5):
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=tol, atol=tol)


class TestEigendecompositions:
    def test_eigh_preserves_batch_axes_and_reconstructs(self):
        batch, row, col, values = Axis(2, "batch"), Axis(3, "row"), Axis(3, "col"), Axis(3, "eigenvalue")
        base = jnp.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
        data = jnp.stack((base, base + jnp.eye(3)))
        a = _named(data, batch, row, col)

        result = eigh(a, eigval_axis=values, eigvec_axes=(row, Axis(3, "eigenvector")))

        assert result.eigenvalues.axes == (batch, values)
        assert result.eigenvectors.axes == (batch, row, Axis(3, "eigenvector"))
        _assert_close(result.eigenvalues.array, jnp.linalg.eigvalsh(data))
        _assert_close(
            data @ result.eigenvectors.array,
            result.eigenvectors.array * result.eigenvalues.array[:, None, :],
        )

    def test_eigh_general_accepts_non_trailing_matrix_axes(self):
        row, col, batch = Axis(2, "row"), Axis(2, "col"), Axis(3, "batch")
        matrix = jnp.array([[2.0, 1.0], [1.0, 4.0]])
        a = _named(jnp.broadcast_to(matrix[:, :, None], (2, 2, 3)), row, col, batch)

        values, vectors = eigh_general(a, matrix_axes=("row", "col"))

        assert values.axes == (batch, Axis(2))
        assert vectors.axes == (batch, Axis(2), Axis(2))
        _assert_close(values.array, jnp.broadcast_to(jnp.linalg.eigvalsh(matrix), (3, 2)))

    def test_eigh_and_eigvals_validate_square_and_uplo(self):
        with pytest.raises(ValueError, match="requires square matrices"):
            eigh(_named(jnp.ones((2, 3)), Axis(2, "r"), Axis(3, "c")))
        with pytest.raises(ValueError, match="UPLO"):
            eigvalsh(_named(jnp.eye(2), Axis(2, "r"), Axis(2, "c")), UPLO="X")

    def test_eig_and_eigvals_match_jax_for_general_matrix(self):
        row, col = Axis(2, "row"), Axis(2, "col")
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = eig(_named(data, row, col), eigvec_axes=(row, col))
        values = eigvals(_named(data, row, col))

        assert result.eigenvalues.axes == values.axes == (Axis(2),)
        _assert_close(result.eigenvalues.array, jnp.linalg.eigvals(data))
        _assert_close(values.array, result.eigenvalues.array)
        _assert_close(data @ result.eigenvectors.array, result.eigenvectors.array * result.eigenvalues.array)

    def test_eig_general_can_omit_both_eigenvector_sets(self):
        a = jnp.diag(jnp.array([2.0, 5.0]))
        out = eig_general(
            _named(a, Axis(2, "r"), Axis(2, "c")),
            compute_left_eigenvectors=False,
            compute_right_eigenvectors=False,
        )

        assert len(out) == 1
        _assert_close(out[0].array, jnp.array([2.0, 5.0]))


class TestSVDAndQR:
    def test_svd_reconstructs_rectangular_matrix_in_both_modes(self):
        row, col = Axis(3, "row"), Axis(2, "col")
        data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
        a = _named(data, row, col)

        for full in (False, True):
            result = svd(a, full_matrices=full)
            assert result.s.axes == (Axis(2),)
            assert result.u.shape == (3, 3 if full else 2)
            assert result.vh.shape == (2 if full else 2, 2)
            _assert_close(result.u.array[:, :2] @ jnp.diag(result.s.array) @ result.vh.array[:2], data)

    def test_svdvals_and_compute_uv_false_match_jax(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])
        named = _named(a, Axis(3, "r"), Axis(2, "c"))

        values = svdvals(named)
        only_values = svd(named, compute_uv=False)

        assert values.axes == only_values.axes == (Axis(2),)
        _assert_close(values.array, jnp.linalg.svd(a, compute_uv=False))
        _assert_close(only_values.array, values.array)

    def test_svd_rejects_incompatible_options(self):
        a = _named(jnp.ones((2, 3)), Axis(2, "r"), Axis(3, "c"))
        with pytest.raises(ValueError, match="subset_by_index"):
            svd(a, subset_by_index=(0, 1))
        with pytest.raises(ValueError, match="compute_uv=False"):
            svd(a, compute_uv=False, u_axes=(Axis(2), Axis(2)))

    def test_qr_reduced_complete_and_r_modes(self):
        data = jnp.array([[1.0, 2.0], [3.0, 5.0], [2.0, 7.0]])
        a = _named(data, Axis(3, "row"), Axis(2, "col"))

        reduced = qr(a)
        complete = qr(a, mode="complete")
        r_only = qr(a, mode="r")

        assert reduced.q.shape == (3, 2)
        assert reduced.r.shape == (2, 2)
        assert complete.q.shape == (3, 3)
        assert complete.r.shape == (3, 2)
        assert r_only.shape == (2, 2)
        _assert_close(reduced.q.array @ reduced.r.array, data)
        _assert_close(complete.q.array @ complete.r.array, data)
        _assert_close(reduced.q.array.T @ reduced.q.array, jnp.eye(2))

    def test_pivoted_qr_returns_permutation_and_reconstructs(self):
        data = jnp.array([[1.0, 0.0, 4.0], [2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
        a = _named(data, Axis(3, "row"), Axis(3, "col"))

        result = qr(a, pivoting=True)
        only_r = qr(a, mode="r", pivoting=True)

        assert result.p.axes == only_r.p.axes == (Axis(3),)
        assert jnp.issubdtype(result.p.dtype, jnp.integer)
        _assert_close(data[:, result.p.array], result.q.array @ result.r.array)
        _assert_close(only_r.r.array, result.r.array)

    def test_qr_rejects_incompatible_options(self):
        with pytest.raises(ValueError, match="does not compute q"):
            qr(jnp.eye(2), q_axes=(Axis(2), Axis(2)), mode="r")
        with pytest.raises(ValueError, match="pivoting=False"):
            qr(jnp.eye(2), p_axis=Axis(2))


class TestCholeskyAndLU:
    def test_cholesky_and_cholesky_factor_solve(self):
        row, col = Axis(2, "row"), Axis(2, "col")
        data = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        a = _named(data, row, col)
        rhs = _named(jnp.array([1.0, 2.0]), row)

        lower = cholesky(a)
        factor = cho_factor(a, lower=True)
        result = cho_solve((factor.c, factor.lower), rhs)

        assert lower.axes == a.axes
        _assert_close(lower.array @ lower.array.T, data)
        assert factor.lower
        assert result.axes == (col,)
        _assert_close(result.array, jnp.linalg.solve(data, rhs.array))

    def test_lu_and_permuted_lu_reconstruct(self):
        row, col = Axis(3, "row"), Axis(3, "col")
        data = jnp.array([[0.0, 2.0, 1.0], [3.0, 1.0, 4.0], [2.0, 5.0, 2.0]])
        a = _named(data, row, col)

        result = lu(a)
        permuted = lu(a, permute_l=True)
        packed = lu_general(a)

        assert result.p.shape == (3, 3)
        assert result.l.shape == result.u.shape == (3, 3)
        _assert_close(result.p.array @ result.l.array @ result.u.array, data)
        _assert_close(permuted.pl.array @ permuted.u.array, data)
        assert packed.lu.axes == a.axes
        assert packed.pivots.shape == packed.permutation.shape == (3,)

    def test_lu_factor_solve_matches_direct_solve(self):
        row, col = Axis(3, "row"), Axis(3, "col")
        data = jnp.array([[0.0, 2.0, 1.0], [3.0, 1.0, 4.0], [2.0, 5.0, 2.0]])
        rhs_data = jnp.array([2.0, -1.0, 3.0])
        a = _named(data, row, col)
        factor = lu_factor(a)
        result = lu_solve((factor.lu, factor.pivots), _named(rhs_data, row))

        assert result.axes == (col,)
        _assert_close(result.array, jnp.linalg.solve(data, rhs_data))

    def test_lu_permute_l_rejects_p_axes(self):
        with pytest.raises(ValueError, match="does not return p"):
            lu(jnp.eye(2), p_axes=(Axis(2), Axis(2)), permute_l=True)


class TestMatrixMetricsAndTransforms:
    def test_determinants_and_slogdet_match_jax(self):
        data = jnp.array([[2.0, 1.0], [4.0, 3.0]])
        a = _named(data, Axis(2, "r"), Axis(2, "c"))

        sign_log = slogdet(a)
        assert det(a).axes == sign_log.sign.axes == sign_log.logabsdet.axes == ()
        _assert_close(det(a).array, jnp.linalg.det(data))
        _assert_close(sign_log.sign.array, jnp.linalg.slogdet(data)[0])
        _assert_close(sign_log.logabsdet.array, jnp.linalg.slogdet(data)[1])

    def test_matrix_rank_and_condition_number(self):
        data = jnp.array([[1.0, 2.0], [2.0, 4.0]])
        a = _named(data, Axis(2, "r"), Axis(2, "c"))

        assert matrix_rank(a).axes == cond(a).axes == ()
        assert matrix_rank(a).array == jnp.linalg.matrix_rank(data)
        _assert_close(cond(a).array, jnp.linalg.cond(data))
        assert matrix_rank(a, tol=0.1).array == 1

    def test_matrix_and_vector_norms_preserve_non_reduced_axes(self):
        batch, row, col = Axis(2, "batch"), Axis(2, "row"), Axis(3, "col")
        data = jnp.arange(12.0).reshape(2, 2, 3)
        a = _named(data, batch, row, col)

        got_matrix = matrix_norm(a, ord="fro")
        got_keep = matrix_norm(a, ord="fro", keepdims=True)
        got_vector = vector_norm(a, vector_axis="col", ord=2)
        got_multi = norm(a, axis=("row", "col"), ord="fro")

        assert got_matrix.axes == got_multi.axes == (batch,)
        assert got_vector.axes == (batch, row)
        assert got_keep.axes == (batch, Axis(1, "row"), Axis(1, "col"))
        _assert_close(got_matrix.array, jnp.linalg.matrix_norm(data, ord="fro"))
        _assert_close(got_keep.array, jnp.linalg.matrix_norm(data, ord="fro", keepdims=True))
        _assert_close(got_vector.array, jnp.linalg.vector_norm(data, axis=2))
        _assert_close(got_multi.array, jnp.linalg.norm(data, axis=(1, 2), ord="fro"))

    def test_inv_matrix_power_pinv_and_nontrailing_matrix_axes(self):
        row, col, batch = Axis(2, "row"), Axis(3, "col"), Axis(2, "batch")
        data = jnp.array([[[1.0, 2.0], [2.0, 1.0], [0.0, 1.0]], [[2.0, 1.0], [1.0, 2.0], [1.0, 0.0]]])
        a = _named(data, row, col, batch)
        pinv_result = pinv(a, matrix_axes=("row", "col"))

        assert pinv_result.axes == (batch, Axis(3, "row"), Axis(2, "col"))
        _assert_close(pinv_result.array, jnp.linalg.pinv(jnp.transpose(data, (2, 0, 1))))

        square = _named(jnp.array([[2.0, 1.0], [1.0, 1.0]]), Axis(2, "r"), Axis(2, "c"))
        _assert_close(inv(square).array, jnp.linalg.inv(square.array))
        _assert_close(matrix_power(square, 2).array, square.array @ square.array)
        with pytest.raises(ValueError, match="requires square matrices"):
            inv(_named(jnp.ones((2, 3)), Axis(2, "r"), Axis(3, "c")))

    def test_matrix_transpose_keeps_axis_roles(self):
        row, col, batch = Axis(2, "row"), Axis(3, "col"), Axis(4, "batch")
        data = jnp.arange(24).reshape(2, 3, 4)
        a = _named(data, row, col, batch)

        result = matrix_transpose(a, matrix_axes=("row", "col"))

        assert result.axes == (Axis(3, "row"), Axis(2, "col"), batch)
        _assert_close(result.array, jnp.swapaxes(data, 0, 1))

    def test_matrix_transpose_rejects_same_matrix_axis(self):
        with pytest.raises(ValueError, match="resolve to the same axis"):
            matrix_transpose(_named(jnp.eye(2), Axis(2, "r"), Axis(2, "c")), matrix_axes=(0, 0))


class TestLinearSolves:
    def test_solve_aligns_batch_axes_and_keeps_free_rhs_axes(self):
        batch, row, col, rhs_axis = Axis(2, "batch"), Axis(2, "row"), Axis(2, "col"), Axis(3, "rhs")
        matrices = jnp.array([[[3.0, 1.0], [1.0, 2.0]], [[2.0, 0.0], [1.0, 4.0]]])
        rhs_data = jnp.arange(12.0).reshape(3, 2, 2) + 1
        a = _named(matrices, batch, row, col)
        b = _named(rhs_data, rhs_axis, batch, row)

        result = solve(a, b)

        assert result.axes == (batch, rhs_axis, col)
        want = jnp.stack([jnp.linalg.solve(matrices[i], rhs_data[:, i, :].T).T for i in range(2)])
        _assert_close(result.array, want)

    def test_solve_can_select_rhs_axis_and_reject_bad_rhs_size(self):
        a = _named(jnp.array([[2.0, 0.0], [0.0, 4.0]]), Axis(2, "r"), Axis(2, "c"))
        b = _named(jnp.array([[2.0, 8.0], [4.0, 12.0]]), Axis(2, "r"), Axis(2, "free"))
        result = solve(a, b, b_axis="r")

        assert result.axes == (Axis(2, "free"), Axis(2, "c"))
        _assert_close(result.array, jnp.linalg.solve(a.array, b.array).T)
        with pytest.raises(ValueError, match="does not have the matrix size"):
            solve(a, _named(jnp.ones(3), Axis(3, "r")))

    def test_solve_triangular_transpose_and_invalid_trans(self):
        row, col = Axis(2, "row"), Axis(2, "col")
        data = jnp.array([[2.0, 0.0], [3.0, 4.0]])
        a = _named(data, row, col)
        b = _named(jnp.array([2.0, 11.0]), col)

        result = solve_triangular(a, b, trans="T", lower=True)

        assert result.axes == (row,)
        _assert_close(result.array, jsl.solve_triangular(data.T, b.array, lower=False))
        with pytest.raises(ValueError, match="`trans` must be"):
            solve_triangular(a, b, trans="bad")

    def test_lstsq_matches_jax_and_reports_rank_and_singular_values(self):
        row, col = Axis(4, "row"), Axis(2, "col")
        data = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
        rhs_data = jnp.array([1.0, 2.0, 2.5, -1.0])
        result = lstsq(_named(data, row, col), _named(rhs_data, row), s_axis=Axis(2, "singular"))
        want = jnp.linalg.lstsq(data, rhs_data)

        assert result.solution.axes == (col,)
        assert result.residuals.axes == ()
        assert result.rank.axes == ()
        assert result.singular_values.axes == (Axis(2, "singular"),)
        _assert_close(result.solution.array, want[0])
        _assert_close(result.residuals.array, want[1])
        assert result.rank.array == want[2]
        _assert_close(result.singular_values.array, want[3])


class TestMatrixFunctions:
    def test_expm_matches_jax_scipy_and_alias(self):
        from ainmax._src.feidhm.linalg import matrix_exp

        data = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        a = _named(data, Axis(2, "r"), Axis(2, "c"))

        result = expm(a)

        assert result.axes == a.axes
        _assert_close(result.array, jsl.expm(data))
        _assert_close(matrix_exp(a).array, result.array)

    def test_expm_frechet_matches_reference_for_both_return_modes(self):
        a_data = jnp.array([[0.0, 0.3], [0.1, -0.2]])
        e_data = jnp.array([[1.0, 0.0], [0.0, 2.0]])
        axes = (Axis(2, "r"), Axis(2, "c"))
        a, e = _named(a_data, *axes), _named(e_data, *axes)

        result = expm_frechet(a, e)
        derivative = expm_frechet(a, e, compute_expm=False)
        expected_exp, expected_derivative = jsl.expm_frechet(a_data, e_data)

        assert result.expm.axes == result.frechet.axes == derivative.axes == axes
        _assert_close(result.expm.array, expected_exp)
        _assert_close(result.frechet.array, expected_derivative)
        _assert_close(derivative.array, expected_derivative)

    def test_sqrtm_polar_schur_and_hessenberg_identities(self):
        data = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        axes = (Axis(2, "row"), Axis(2, "col"))
        a = _named(data, *axes)

        root = sqrtm(a)
        polar_result = polar(a)
        schur_result = schur(a, t_axes=axes, z_axes=axes)
        hessenberg_result = hessenberg(a, h_axes=axes, q_axes=axes, calc_q=True)

        _assert_close(root.array @ root.array, data)
        _assert_close(polar_result.u.array @ polar_result.p.array, data)
        _assert_close(schur_result.z.array @ schur_result.t.array @ schur_result.z.array.T, data)
        _assert_close(
            hessenberg_result.q.array @ hessenberg_result.h.array @ hessenberg_result.q.array.T,
            data,
        )
        assert polar_result.u.axes == a.axes
        assert schur_result.t.axes == schur_result.z.axes == a.axes
        assert hessenberg_result.h.axes == hessenberg_result.q.axes == a.axes

    def test_hessenberg_rejects_q_axes_when_not_computed(self):
        with pytest.raises(ValueError, match="calc_q=False"):
            hessenberg(jnp.eye(2), q_axes=(Axis(2), Axis(2)))


class TestCrossAndTransformValidation:
    def test_cross_matches_jax_and_keeps_vector_axis(self):
        vector, batch = Axis(3, "vector"), Axis(2, "batch")
        left = _named(jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), batch, vector)
        right = _named(jnp.array([0.0, 0.0, 1.0]), vector)

        result = cross(left, right, axis="vector")

        assert result.axes == (batch, vector)
        _assert_close(result.array, jnp.linalg.cross(left.array, jnp.broadcast_to(right.array, (2, 3))))

    def test_non_square_matrix_operations_reject_ambiguous_matrix_axes(self):
        a = _named(jnp.ones((2, 3)), Axis(2, "row"), Axis(3, "col"))
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            matrix_norm(_named(jnp.ones(3), Axis(3, "vector")))
        with pytest.raises(ValueError, match="same axis"):
            matrix_transpose(a, matrix_axes=("row", "row"))

    def test_jitted_solve_preserves_named_axes(self):
        row, col = Axis(2, "row"), Axis(2, "col")
        a = _named(jnp.array([[3.0, 1.0], [1.0, 2.0]]), row, col)
        b = _named(jnp.array([1.0, 4.0]), row)

        result = jax.jit(lambda x, y: solve(x, y))(a, b)

        assert result.axes == (col,)
        _assert_close(result.array, jnp.linalg.solve(a.array, b.array))
