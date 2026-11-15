from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from std.math import cos, pi, sqrt
from std.utils.numerics import nan
import numojo as nm
from numojo.prelude import *

from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.operator import BlockDiagonal, Identity
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    RealArray,
    fget,
    fset,
    real_array,
    real_vector,
    norm2,
)
from mojodec.routines.linalg import (
    DenseCholesky,
    NullSpace,
    StopReason,
    cg,
    eigenvalue_range,
    eigh_generalized,
    jacobi,
    lsqr,
    minres,
    spsolve,
)
from mojodec.routines import dense
from mojodec.routines.dense import EighResult


def laplacian_1d(n: Int) raises -> CSRMatrix:
    var r = List[Int]()
    var c = List[Int]()
    var v = List[Float64]()
    for i in range(n):
        r.append(i)
        c.append(i)
        v.append(2.0)
        if i > 0:
            r.append(i)
            c.append(i - 1)
            v.append(-1.0)
        if i < n - 1:
            r.append(i)
            c.append(i + 1)
            v.append(-1.0)
    return CSRMatrix.from_coo(n, n, r, c, v)


def test_cg() raises:
    var A = laplacian_1d(50)
    nm.random.seed(1)
    var b = nm.random.rand[f64](Shape(50))
    var res = cg(A, b, rtol=1e-12)
    assert_true(res.converged())
    assert_true(res.normal_residual_norm is None)
    var r = A @ res.x
    for i in range(50):
        # cg stops at |r| <= rtol |b|
        assert_almost_equal(
            r.unsafe_get(i), b.unsafe_get(i), atol=1e-12 * norm2(b), rtol=0
        )
    var x = spsolve(A, b)
    for i in range(50):
        # the residual bound times cond(A) ~ 1e3, with |x| ~ 150
        assert_almost_equal(
            x.unsafe_get(i), res.x.unsafe_get(i), atol=1e-10, rtol=0
        )
    # the iteration limit, and an indefinite matrix
    var short = cg(A, b, rtol=1e-12, maxiter=3)
    assert_true(short.stop == StopReason.MAX_ITER)
    assert_equal(short.iterations, 3)
    var neg = cg(A * -1.0, b)
    assert_true(neg.stop == StopReason.BREAKDOWN)
    assert_equal(String(neg.stop), "breakdown")
    var x0 = neg^.take_x()
    assert_equal(x0.size, 50)


def test_lsqr() raises:
    # overdetermined, full column rank: compare with normal equations
    var A = CSRMatrix.from_coo(
        4, 2, [0, 1, 2, 3, 3], [0, 1, 0, 1, 0], [1.0, 1.0, 1.0, 1.0, 1.0]
    )
    var b = nm.fromstring[f64]("[1, 2, 3, 4]")
    var res = lsqr(A, b)
    var AtA = A.T() @ A
    var Atb = A.T() @ b
    var x = spsolve(AtA, Atb)
    for i in range(2):
        # two columns: lsqr is exact after two steps
        assert_almost_equal(
            res.x.unsafe_get(i), x.unsafe_get(i), atol=1e-13, rtol=0
        )
    assert_true(res.converged())
    # the reported norms are those of r = b - A x and A^T r
    var r = b.copy()
    var Ax = A @ res.x
    for i in range(4):
        r.unsafe_set(i, r.unsafe_get(i) - Ax.unsafe_get(i))
    assert_almost_equal(res.residual_norm, norm2(r), atol=1e-14, rtol=0)
    assert_almost_equal(
        res.normal_residual_norm.value(), 0.0, atol=1e-13, rtol=0
    )


def test_dense_pivoting() raises:
    # zero leading pivot: needs row exchanges
    var A = nm.fromstring[f64]("[[0, 1, 1], [1, 0, 1], [1, 1, 0]]")
    var b = nm.fromstring[f64]("[2, 2, 2]")
    var x = dense.solve(A, b)
    for i in range(3):
        assert_almost_equal(x.unsafe_get(i), 1.0, atol=1e-15, rtol=0)
    assert_almost_equal(dense.det(A), 2.0, atol=1e-15, rtol=0)


def test_eigh_generalized() raises:
    var K = nm.fromstring[f64]("[[2, -1, 0], [-1, 2, -1], [0, -1, 2]]")
    var M = nm.fromstring[f64]("[[2, 0, 0], [0, 1, 0], [0, 0, 3]]")
    var wv = eigh_generalized(K, M)
    for j in range(3):
        var lam = wv.values.unsafe_get(j)
        for i in range(3):
            var kx: Float64 = 0
            var mx: Float64 = 0
            for k in range(3):
                kx += fget(K, i, k) * fget(wv.vectors, k, j)
                mx += fget(M, i, k) * fget(wv.vectors, k, j)
            assert_almost_equal(kx, lam * mx, atol=1e-14, rtol=0)
    assert_true(wv.values.unsafe_get(0) <= wv.values.unsafe_get(1))


def test_eigh_generalized_diagonal() raises:
    nm.random.seed(7)
    var n = 12
    var B = nm.random.rand[f64](Shape(n, n))
    var K = nm.zeros[f64](Shape(n, n))
    var M = nm.zeros[f64](Shape(n, n))
    var m = List[Float64]()
    for i in range(n):
        for j in range(n):
            K.unsafe_set(i * n + j, fget(B, i, j) + fget(B, j, i))
        m.append(0.25 + Float64(i))
        M.unsafe_set(i * n + i, m[i])
    var dense_wv = eigh_generalized(K, M)
    var wv = eigh_generalized(K, DiagonalOperator(m.copy()))
    for j in range(n):
        var lam = wv.values.unsafe_get(j)
        # eigenvalues up to ~10: 1e-13 is a few hundred ulps
        assert_almost_equal(
            lam, dense_wv.values.unsafe_get(j), atol=1e-13, rtol=0
        )
        for i in range(n):
            var kx: Float64 = 0
            for k in range(n):
                kx += fget(K, i, k) * fget(wv.vectors, k, j)
            assert_almost_equal(
                kx, lam * m[i] * fget(wv.vectors, i, j), atol=1e-13, rtol=0
            )
        # M-orthonormal columns
        for l in range(n):
            var mxx: Float64 = 0
            for i in range(n):
                mxx += m[i] * fget(wv.vectors, i, j) * fget(wv.vectors, i, l)
            assert_almost_equal(mxx, 1.0 if j == l else 0.0, atol=1e-13, rtol=0)
    m[3] = 0
    with assert_raises(contains="not positive definite"):
        _ = eigh_generalized(K, DiagonalOperator(m.copy()))
    with assert_raises(contains="differ in shape"):
        _ = eigh_generalized(K, DiagonalOperator.identity(n - 1))


def test_dense_eigh() raises:
    nm.random.seed(3)
    var n = 30
    var B = nm.random.rand[f64](Shape(n, n))
    var A = nm.zeros[f64](Shape(n, n))
    for i in range(n):
        for j in range(n):
            A.unsafe_set(i * n + j, fget(B, i, j) + fget(B, j, i))
    var wv = dense.eigh(A)
    for j in range(n):
        if j > 0:
            assert_true(wv.values.unsafe_get(j - 1) <= wv.values.unsafe_get(j))
        for i in range(n):
            var s: Float64 = 0
            for k in range(n):
                s += fget(A, i, k) * fget(wv.vectors, k, j)
            # eigenvalues up to ~30
            assert_almost_equal(
                s,
                wv.values.unsafe_get(j) * fget(wv.vectors, i, j),
                atol=1e-12,
                rtol=0,
            )


def test_empty_and_mismatched_eigenproblems() raises:
    var e = dense.eigh(real_array(0, 0))  # once crashed
    assert_equal(e.values.size, 0)
    var K = real_array(3, 3)
    var M = real_array(2, 2)
    for i in range(3):
        fset(K, i, i, 2.0)
    for i in range(2):
        fset(M, i, i, 4.0)
    with assert_raises(contains="n x n"):
        _ = eigh_generalized(K, M)
    with assert_raises(contains="square"):
        _ = dense.cholesky(real_array(2, 3))


def test_cg_reports_nan_as_breakdown() raises:
    # a NaN p^T A p once passed the `<= 0` test and ran to maxiter
    var A = CSRMatrix.from_coo(
        2, 2, [0, 1], [0, 1], [1.0, nan[DType.float64]()]
    )
    var b = real_vector(2)
    b.unsafe_set(0, 1.0)
    b.unsafe_set(1, 1.0)
    var res = cg(A, b)
    assert_true(res.stop == StopReason.BREAKDOWN)


def _scaled_laplacian(n: Int) raises -> CSRMatrix:
    """`D L D` for the 1-D Laplacian and `D` from 1 to 1000: symmetric
    positive definite, with a diagonal that varies by a factor 10^6."""
    var L = laplacian_1d(n)
    var d = List[Float64]()
    for i in range(n):
        d.append(1.0 + 999.0 * Float64((i * 37) % n) / Float64(n))
    var D = DiagonalOperator(d^)
    return D @ L @ D


def _rhs(n: Int) raises -> RealArray:
    var b = real_vector(n)
    for i in range(n):
        b.unsafe_set(i, Float64((i * 7) % 11) - 5.0)
    return b^


def test_pcg() raises:
    var n = 60
    var A = _scaled_laplacian(n)
    var b = _rhs(n)
    var x = spsolve(A, b)
    var plain = cg(A, b, rtol=1e-12)
    var pre = cg(A, b, M=jacobi(A), rtol=1e-12)
    assert_true(plain.converged())
    assert_true(pre.converged())
    # the diagonal scaling is what Jacobi undoes: 60 iterations, the
    # dimension, against 173
    assert_true(pre.iterations <= n)
    assert_true(pre.iterations * 2 < plain.iterations)
    var r = A @ pre.x - b
    # the stopping test is on the true residual
    assert_true(norm2(r) <= 1e-12 * norm2(b) * 1.01)
    var xnorm = norm2(x)
    for i in range(n):
        # rtol=1e-12 times cond(D^-1 A D^-1)~1e3 relative to |x|
        assert_almost_equal(
            pre.x.unsafe_get(i), x.unsafe_get(i), atol=1e-8 * xnorm, rtol=0
        )
    # the identity preconditioner gives exactly the plain iterates
    var same = cg(A, b, M=Identity(n), rtol=1e-12)
    assert_equal(same.iterations, plain.iterations)
    for i in range(n):
        assert_equal(same.x.unsafe_get(i), plain.x.unsafe_get(i))
    # any LinearOperator: a diagonal system converges in one step per value
    var D = DiagonalOperator([1.0, 2.0, 4.0])
    var bd = real_vector(3)
    for i in range(3):
        bd.unsafe_set(i, 1.0)
    var xd = cg(D, bd, rtol=1e-14)
    assert_true(xd.converged())
    assert_almost_equal(xd.x.unsafe_get(2), 0.25, atol=1e-15, rtol=0)
    # Jacobi on a diagonal system: exact in one iteration
    assert_equal(cg(D, bd, M=jacobi(D.to_csr())).iterations, 1)


def test_minres_indefinite() raises:
    # the 1-D Laplacian shifted into the middle of its spectrum (0, 4)
    var n = 80
    var A = laplacian_1d(n) - CSRMatrix.identity(n) * 1.3
    var b = _rhs(n)
    var x = spsolve(A, b)
    var res = minres(A, b, rtol=1e-12)
    assert_true(res.converged())
    assert_true(res.normal_residual_norm is None)
    # residual_norm is recomputed from x; the recurrence stops at 1e-12
    var r = A @ res.x - b
    assert_almost_equal(res.residual_norm, norm2(r), atol=1e-14, rtol=0)
    assert_true(res.residual_norm <= 2e-12 * norm2(b))
    var xnorm = norm2(x)
    for i in range(n):
        # cond(A) ~ 1e3 (the smallest |eigenvalue| ~ 2e-3)
        assert_almost_equal(
            res.x.unsafe_get(i), x.unsafe_get(i), atol=1e-8 * xnorm, rtol=0
        )
    # cg cannot do this: p^T A p changes sign
    assert_true(cg(A, b).stop == StopReason.BREAKDOWN)
    # the residual never grows: a short run is no worse than a shorter one
    var r5 = minres(A, b, maxiter=5)
    var r10 = minres(A, b, maxiter=10)
    assert_true(r5.stop == StopReason.MAX_ITER)
    assert_equal(r10.iterations, 10)
    assert_true(r10.residual_norm <= r5.residual_norm)
    # b = 0: x = 0 at once
    var z = minres(A, real_vector(n))
    assert_true(z.converged())
    assert_equal(z.iterations, 0)


def neumann_path(n: Int) raises -> CSRMatrix:
    """The Laplacian of a path with free ends: constants in the kernel."""
    var r = List[Int]()
    var c = List[Int]()
    var v = List[Float64]()
    for i in range(n - 1):
        for e in [(i, i, 1.0), (i + 1, i + 1, 1.0), (i, i + 1, -1.0)]:
            r.append(e[0])
            c.append(e[1])
            v.append(e[2])
        r.append(i + 1)
        c.append(i)
        v.append(-1.0)
    return CSRMatrix.from_coo(n, n, r, c, v)


def test_minres_singular_consistent() raises:
    # the Neumann Laplacian of a path: constants in the kernel
    var n = 40
    var r = List[Int]()
    var c = List[Int]()
    var v = List[Float64]()
    for i in range(n - 1):
        for e in [(i, i, 1.0), (i + 1, i + 1, 1.0), (i, i + 1, -1.0)]:
            r.append(e[0])
            c.append(e[1])
            v.append(e[2])
        r.append(i + 1)
        c.append(i)
        v.append(-1.0)
    var A = CSRMatrix.from_coo(n, n, r, c, v)
    var f = _rhs(n)
    var b = A @ f  # in the range of A
    var res = minres(A, b, rtol=1e-12)
    assert_true(res.converged())
    assert_true(res.residual_norm <= 2e-12 * norm2(b))
    # from x0 = 0, the Krylov space is orthogonal to the kernel: x is the
    # solution of least norm, f minus its mean
    var mean: Float64 = 0
    for i in range(n):
        mean += f.unsafe_get(i) / Float64(n)
    for i in range(n):
        # cond ~ 1e3 on the range of A, times |f| ~ 20
        assert_almost_equal(
            res.x.unsafe_get(i), f.unsafe_get(i) - mean, atol=1e-8, rtol=0
        )


def test_preconditioned_minres() raises:
    # D (L - 1.3 I) D: indefinite and badly scaled
    var n = 60
    var L = laplacian_1d(n) - CSRMatrix.identity(n) * 1.3
    var d = List[Float64]()
    for i in range(n):
        d.append(1.0 + 99.0 * Float64((i * 37) % n) / Float64(n))
    var D = DiagonalOperator(d^)
    var A = D @ L @ D
    var b = _rhs(n)
    var plain = minres(A, b, rtol=1e-10)
    var pre = minres(A, b, M=jacobi(A), rtol=1e-10)
    assert_true(plain.converged())
    assert_true(pre.converged())
    # 60 iterations against 159
    assert_true(pre.iterations * 2 < plain.iterations)
    var x = spsolve(A, b)
    var xnorm = norm2(x)
    for i in range(n):
        assert_almost_equal(
            pre.x.unsafe_get(i), x.unsafe_get(i), atol=1e-5 * xnorm, rtol=0
        )
    # the identity preconditioner gives exactly the plain iterates
    var same = minres(A, b, M=Identity(n), rtol=1e-10)
    assert_equal(same.iterations, plain.iterations)
    for i in range(n):
        assert_equal(same.x.unsafe_get(i), plain.x.unsafe_get(i))


def test_dense_cholesky_and_block_diagonal() raises:
    var n = 30
    var A = laplacian_1d(n) + CSRMatrix.identity(n) * 0.5
    var C = DenseCholesky(A)
    var b = _rhs(n)
    var y = real_vector(n)
    C.apply_into(b, y)
    var r = A @ y
    for i in range(n):
        assert_almost_equal(
            r.unsafe_get(i), b.unsafe_get(i), atol=1e-13, rtol=0
        )
    # an exact preconditioner: CG converges in one step
    var res = cg(A, b, M=DenseCholesky(A), rtol=1e-12)
    assert_true(res.converged())
    assert_true(res.iterations <= 1)
    # diag(P1, P2) applies each block to its half
    var P = BlockDiagonal(DiagonalOperator([2.0, 3.0]), DenseCholesky(A))
    assert_equal(P.num_rows(), n + 2)
    var x = real_vector(n + 2)
    for i in range(n + 2):
        x.unsafe_set(i, Float64(i + 1))
    var z = real_vector(n + 2)
    P.apply_into(x, z)
    assert_equal(z.unsafe_get(0), 2.0)
    assert_equal(z.unsafe_get(1), 6.0)
    var tail = real_vector(n)
    for i in range(n):
        tail.unsafe_set(i, Float64(i + 3))
    C.apply_into(tail, y)
    for i in range(n):
        assert_almost_equal(
            z.unsafe_get(2 + i), y.unsafe_get(i), atol=1e-15, rtol=0
        )
    with assert_raises(contains="dimension mismatch"):
        P.apply_into(real_vector(n), z)
    with assert_raises(contains="not positive definite"):
        _ = DenseCholesky(laplacian_1d(n) - CSRMatrix.identity(n) * 5.0)
    with assert_raises(contains="not symmetric"):
        _ = DenseCholesky(
            CSRMatrix.from_dense(nm.fromstring[f64]("[[2, 1], [0, 2]]"))
        )


def test_solver_input_checks() raises:
    var n = 10
    var A = laplacian_1d(n)
    var b = _rhs(n)
    # a non-symmetric matrix is refused before any iteration
    var N = A.copy()
    N.data[1] = -0.5  # entry (0, 1)
    with assert_raises(contains="not symmetric"):
        _ = cg(N, b)
    with assert_raises(contains="not symmetric"):
        _ = minres(N, b)
    with assert_raises(contains="dimension mismatch"):
        _ = minres(A, real_vector(n + 1))
    with assert_raises(contains="not n x n"):
        _ = minres(A, b, M=Identity(n + 1))
    with assert_raises(contains="maxiter"):
        _ = minres(A, b, maxiter=-1)
    # an indefinite preconditioner shows as a breakdown
    var neg = DiagonalOperator(List[Float64](length=n, fill=-1.0))
    assert_true(cg(A, b, M=neg).stop == StopReason.BREAKDOWN)
    assert_true(minres(A, b, M=neg).stop == StopReason.BREAKDOWN)
    # Jacobi needs a square matrix with a nonzero diagonal
    with assert_raises(contains="entry 1 is 0"):
        _ = jacobi(
            CSRMatrix.from_coo(2, 2, [0, 0, 1], [0, 1, 0], [1.0, 2.0, 2.0])
        )
    with assert_raises(contains="not square"):
        _ = jacobi(CSRMatrix(2, 3))
    # 1 / |a_ii|: positive for an indefinite matrix too
    var J = jacobi(A * -2.0)
    assert_equal(J[0], 0.25)


def test_eigen_residuals() raises:
    # backward-stable solvers: tiny backward errors for every pair
    nm.random.seed(5)
    var n = 40
    var B = nm.random.rand[f64](Shape(n, n))
    var A = nm.zeros[f64](Shape(n, n))
    var m = List[Float64]()
    var M = nm.zeros[f64](Shape(n, n))
    for i in range(n):
        for j in range(n):
            A.unsafe_set(i * n + j, fget(B, i, j) + fget(B, j, i))
        m.append(0.5 + Float64(i % 7))
        M.unsafe_set(i * n + i, m[i])
    var std = dense.eigh(A)
    var gen = eigh_generalized(A, M)
    var dia = eigh_generalized(A, DiagonalOperator(m.copy()))
    var e1 = std.residuals(A)
    var e2 = gen.residuals(A, M)
    var e3 = dia.residuals(A, DiagonalOperator(m.copy()))
    var e4 = dia.residuals(CSRMatrix.from_dense(A), DiagonalOperator(m.copy()))
    for i in range(n):
        # n eps = 9e-15
        assert_true(e1.unsafe_get(i) < 1e-14)
        assert_true(e2.unsafe_get(i) < 1e-14)
        assert_true(e3.unsafe_get(i) < 1e-14)
        assert_equal(e3.unsafe_get(i), e4.unsafe_get(i))
    # the definition, on a pair that is off: K = diag(1, 2, 3), M = I,
    # x = e_0 and lambda = 1.5: |1 - 1.5| / ((3 + 1.5) * 1) = 1/9
    var K = CSRMatrix.from_coo(3, 3, [0, 1, 2], [0, 1, 2], [1.0, 2.0, 3.0])
    var vals = real_vector(3)
    vals.unsafe_set(0, 1.5)
    vals.unsafe_set(1, 2.0)
    vals.unsafe_set(2, 3.0)
    var vecs = real_array(3, 3)
    for i in range(3):
        fset(vecs, i, i, 1.0)
    var fake = EighResult(vals^, vecs^)
    var eta = fake.residuals(K, DiagonalOperator.identity(3))
    # ||K|| from 30 power steps: below 3 by a relative (2/3)^60 ~ 3e-11,
    # so eta is above 1/9 by about that much
    assert_true(eta.unsafe_get(0) >= 1.0 / 9.0)
    assert_almost_equal(eta.unsafe_get(0), 1.0 / 9.0, atol=1e-11, rtol=0)
    assert_equal(eta.unsafe_get(1), 0.0)
    assert_equal(eta.unsafe_get(2), 0.0)
    # the matrices must match the eigenpairs
    with assert_raises(contains="K is not 3 x 3"):
        _ = fake.residuals(A)
    with assert_raises(contains="M is not 3 x 3"):
        _ = fake.residuals(K, DiagonalOperator.identity(4))
    var empty = dense.eigh(real_array(0, 0))
    assert_equal(empty.residuals(real_array(0, 0)).size, 0)


def laplacian_2d(m: Int) raises -> CSRMatrix:
    """The 5-point Laplacian on an m x m grid, Dirichlet boundary."""
    var r = List[Int]()
    var c = List[Int]()
    var v = List[Float64]()
    for i in range(m):
        for j in range(m):
            var k = i * m + j
            r.append(k)
            c.append(k)
            v.append(4.0)
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                var ii = i + d[0]
                var jj = j + d[1]
                if ii >= 0 and ii < m and jj >= 0 and jj < m:
                    r.append(k)
                    c.append(ii * m + jj)
                    v.append(-1.0)
    return CSRMatrix.from_coo(m * m, m * m, r, c, v)


def test_starting_guess() raises:
    var n = 60
    var A = _scaled_laplacian(n)
    var f = _rhs(n)
    var b = A @ f
    # the exact solution as the guess: done before the first step
    var exact = cg(A, b, rtol=1e-10, x0=f.copy())
    assert_true(exact.converged())
    assert_equal(exact.iterations, 0)
    for i in range(n):
        assert_equal(exact.x.unsafe_get(i), f.unsafe_get(i))
    # a rough solution as the guess: fewer steps to the same tolerance,
    # which stays relative to ||b|| (on a 2-D grid, where CG converges in
    # fewer than n steps; in 1-D it takes n either way)
    var L = laplacian_2d(30)
    var g = _rhs(900)
    var rough = cg(L, g, rtol=1e-3).take_x()
    var cold = cg(L, g, rtol=1e-10)
    var warm = cg(L, g, rtol=1e-10, x0=rough.copy())
    assert_true(cold.converged() and warm.converged())
    assert_true(warm.iterations < cold.iterations)
    assert_true(warm.residual_norm <= 1e-10 * norm2(g))
    var m_cold = minres(L, g, rtol=1e-10)
    var m_warm = minres(L, g, rtol=1e-10, x0=rough.copy())
    assert_true(m_warm.converged())
    assert_true(m_warm.iterations < m_cold.iterations)
    assert_true(m_warm.residual_norm <= 2e-10 * norm2(g))
    rough = cg(A, b, M=jacobi(A), rtol=1e-3).take_x()
    # minres with a preconditioner
    var mp = minres(A, b, M=jacobi(A), rtol=1e-10, x0=f.copy())
    assert_equal(mp.iterations, 0)
    var mp_warm = minres(A, b, M=jacobi(A), rtol=1e-10, x0=rough.copy())
    assert_true(mp_warm.converged())
    for i in range(n):
        # cond(A) ~ 1e6 for the scaled Laplacian, |f| ~ 5
        assert_almost_equal(
            mp_warm.x.unsafe_get(i), f.unsafe_get(i), atol=1e-3, rtol=0
        )
    with assert_raises(contains="x0 has length 3, not 60"):
        _ = cg(A, b, x0=real_vector(3))
    with assert_raises(contains="x0 has length 3, not 60"):
        _ = minres(A, b, x0=real_vector(3))


def test_nullspace() raises:
    var n = 40
    var A = neumann_path(n)
    var f = _rhs(n)
    var mean: Float64 = 0
    for i in range(n):
        mean += f.unsafe_get(i) / Float64(n)
    # an inconsistent right-hand side: A f plus a constant
    var b = A @ f
    for i in range(n):
        b.unsafe_set(i, b.unsafe_get(i) + 3.0)
    var ns = NullSpace.constants(n)
    assert_equal(len(ns), 1)
    assert_equal(ns.size(), n)
    # without the kernel, CG cannot converge: the constant part of b is
    # not in the range
    assert_true(not cg(A, b, rtol=1e-10, maxiter=400).converged())
    # with it, every solver finds the least-squares solution of least
    # norm, f minus its mean, even from a guess with a constant part and
    # with a preconditioner that is not constant on the diagonal
    var guess = real_vector(n)
    for i in range(n):
        guess.unsafe_set(i, 7.0 + 0.1 * Float64(i))
    var results = List[RealArray]()
    var r1 = cg(A, b, rtol=1e-12, nullspace=ns)
    assert_true(r1.converged())
    results.append(r1^.take_x())
    var r2 = cg(A, b, M=jacobi(A), rtol=1e-12, x0=guess.copy(), nullspace=ns)
    assert_true(r2.converged())
    results.append(r2^.take_x())
    var r3 = minres(A, b, rtol=1e-12, nullspace=ns)
    assert_true(r3.converged())
    results.append(r3^.take_x())
    var r4 = minres(
        A, b, M=jacobi(A), rtol=1e-12, x0=guess.copy(), nullspace=ns
    )
    assert_true(r4.converged())
    results.append(r4^.take_x())
    for ref x in results:
        for i in range(n):
            # cond ~ 1e3 on the range of A, times |f| ~ 20
            assert_almost_equal(
                x.unsafe_get(i), f.unsafe_get(i) - mean, atol=1e-8, rtol=0
            )
    # the basis is orthonormalized, whatever the input
    var w = List[RealArray]()
    var e = real_vector(n)
    e.unsafe_set(0, 1.0)
    w.append(nm.ones[f64](Shape(n)) * 5.0)
    w.append(nm.ones[f64](Shape(n)) + e)
    var two = NullSpace(w)
    var q0 = two.vector(0)
    var q1 = two.vector(1)
    assert_almost_equal(norm2(q0), 1.0, atol=1e-15, rtol=0)
    assert_almost_equal(norm2(q1), 1.0, atol=1e-15, rtol=0)
    var q01: Float64 = 0
    for i in range(n):
        q01 += q0.unsafe_get(i) * q1.unsafe_get(i)
    assert_almost_equal(q01, 0.0, atol=1e-15, rtol=0)
    # what is refused
    with assert_raises(contains="no vectors"):
        _ = NullSpace(List[RealArray]())
    var dep = List[RealArray]()
    dep.append(nm.ones[f64](Shape(n)))
    dep.append(nm.ones[f64](Shape(n)) * 2.0)
    with assert_raises(contains="vector 1 depends on the ones before it"):
        _ = NullSpace(dep)
    var mixed = List[RealArray]()
    mixed.append(nm.ones[f64](Shape(n)))
    mixed.append(real_vector(3))
    with assert_raises(contains="vector 1 has length 3, not 40"):
        _ = NullSpace(mixed)
    with assert_raises(contains="not in the operator's kernel"):
        _ = cg(laplacian_1d(n), b, nullspace=ns)
    with assert_raises(contains="not in the operator's kernel"):
        _ = minres(laplacian_1d(n), b, nullspace=ns)
    with assert_raises(contains="null space has vectors of length 40, not 41"):
        _ = cg(neumann_path(n + 1), real_vector(n + 1), nullspace=ns)


def test_ritz_values() raises:
    # the eigenvalues of the 1-D Laplacian: 2 - 2 cos(k pi / (n + 1))
    var w = dense.tridiagonal_eigenvalues([2.0, 2.0, 2.0], [-1.0, -1.0])
    assert_almost_equal(w.unsafe_get(0), 2 - sqrt(2.0), atol=1e-15, rtol=0)
    assert_almost_equal(w.unsafe_get(1), 2.0, atol=1e-15, rtol=0)
    assert_almost_equal(w.unsafe_get(2), 2 + sqrt(2.0), atol=1e-15, rtol=0)
    assert_equal(dense.tridiagonal_eigenvalues([], []).size, 0)
    with assert_raises(contains="1 off-diagonal entries for 1 diagonal"):
        _ = dense.tridiagonal_eigenvalues([1.0], [1.0])
    var n = 200
    var A = laplacian_1d(n)
    var lo = 2 - 2 * cos(pi / Float64(n + 1))
    var hi = 2 - 2 * cos(Float64(n) * pi / Float64(n + 1))
    # no Ritz values unless asked for
    assert_true(not cg(A, _rhs(n), rtol=1e-8).ritz_values)
    # run to convergence, the Ritz values lie inside the spectrum and the
    # extreme ones converge first
    var res = cg(A, _rhs(n), rtol=1e-12, ritz=True)
    assert_true(res.converged())
    ref t = res.ritz_values.value()
    assert_equal(t.size, res.iterations)
    for i in range(t.size):
        assert_true(t.unsafe_get(i) >= lo - 1e-12)
        assert_true(t.unsafe_get(i) <= hi + 1e-12)
    assert_almost_equal(t.unsafe_get(t.size - 1), hi, atol=1e-8, rtol=0)
    # ten steps: inner bounds, the top within a few percent
    var est = eigenvalue_range(A)
    assert_equal(est.steps, 10)
    assert_true(est.highest <= hi and est.highest >= 0.9 * hi)
    assert_true(est.lowest >= lo)
    # with Jacobi, the spectrum of M A = A / 2
    var pre = eigenvalue_range(A, M=jacobi(A), steps=20)
    assert_true(pre.highest <= hi / 2 + 1e-15)
    assert_true(pre.highest >= 0.95 * hi / 2)
    # reproducible: the same right-hand side every call
    assert_equal(eigenvalue_range(A).highest, est.highest)
    # a singular operator, its kernel excluded; steps capped at n - 1
    var N = neumann_path(6)
    var small = eigenvalue_range(N, steps=50, nullspace=NullSpace.constants(6))
    assert_equal(small.steps, 5)
    # the path's nonzero eigenvalues: 2 - 2 cos(k pi / 6), k = 1..5
    assert_almost_equal(small.lowest, 2 - 2 * cos(pi / 6), atol=1e-12, rtol=0)
    assert_almost_equal(
        small.highest, 2 - 2 * cos(5 * pi / 6), atol=1e-12, rtol=0
    )
    # minres: Ritz values of an indefinite operator, both signs
    var S = A - CSRMatrix.identity(n) * 0.5  # eigenvalues in (-0.5, 3.5)
    var m = minres(S, _rhs(n), rtol=1e-12, ritz=True)
    ref mt = m.ritz_values.value()
    assert_true(mt.unsafe_get(0) < 0 and mt.unsafe_get(mt.size - 1) > 0)
    for i in range(mt.size):
        assert_true(mt.unsafe_get(i) >= lo - 0.5 - 1e-10)
        assert_true(mt.unsafe_get(i) <= hi - 0.5 + 1e-10)
    with assert_raises(contains="CG broke down"):
        _ = eigenvalue_range(S)
    with assert_raises(contains="steps must be positive"):
        _ = eigenvalue_range(A, steps=0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
