from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import fget, real_vector, norm2
from mojodec.routines.linalg import cg, lsqr, spsolve, eigh_generalized
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
    assert_equal(res.info, 0)
    var r = A @ res.x
    for i in range(50):
        assert_almost_equal(r.unsafe_get(i), b.unsafe_get(i), atol=1e-9)
    var x = spsolve(A, b)
    for i in range(50):
        assert_almost_equal(x.unsafe_get(i), res.x.unsafe_get(i), atol=1e-8)


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
        assert_almost_equal(res.x.unsafe_get(i), x.unsafe_get(i), atol=1e-8)


def test_dense_pivoting() raises:
    # zero leading pivot: needs row exchanges
    var A = nm.fromstring[f64]("[[0, 1, 1], [1, 0, 1], [1, 1, 0]]")
    var b = nm.fromstring[f64]("[2, 2, 2]")
    var x = dense.solve(A, b)
    for i in range(3):
        assert_almost_equal(x.unsafe_get(i), 1.0)
    assert_almost_equal(dense.det(A), 2.0)


def test_eigh_generalized() raises:
    var K = nm.fromstring[f64]("[[2, -1, 0], [-1, 2, -1], [0, -1, 2]]")
    var M = nm.fromstring[f64]("[[2, 0, 0], [0, 1, 0], [0, 0, 3]]")
    var wv = eigh_generalized(K, M)
    for j in range(3):
        var lam = wv[0].unsafe_get(j)
        for i in range(3):
            var kx: Float64 = 0
            var mx: Float64 = 0
            for k in range(3):
                kx += fget(K, i, k) * fget(wv[1], k, j)
                mx += fget(M, i, k) * fget(wv[1], k, j)
            assert_almost_equal(kx, lam * mx, atol=1e-9)
    assert_true(wv[0].unsafe_get(0) <= wv[0].unsafe_get(1))


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
            assert_true(wv[0].unsafe_get(j - 1) <= wv[0].unsafe_get(j))
        for i in range(n):
            var s: Float64 = 0
            for k in range(n):
                s += fget(A, i, k) * fget(wv[1], k, j)
            assert_almost_equal(
                s, wv[0].unsafe_get(j) * fget(wv[1], i, j), atol=1e-10
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
