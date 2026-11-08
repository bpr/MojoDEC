from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from std.utils.numerics import nan
import numojo as nm
from numojo.prelude import *

from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import fget, fset, real_array, real_vector, norm2
from mojodec.routines.linalg import (
    StopReason,
    cg,
    eigh_generalized,
    lsqr,
    spsolve,
)
from mojodec.routines import dense


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
