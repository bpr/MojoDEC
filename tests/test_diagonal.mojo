from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
    assert_true,
)

from std.math import sqrt
import numojo as nm

from mojodec import (
    CSRMatrix,
    DiagonalOperator,
    SimplicialComplex,
    d,
    star,
)
from mojodec.core.utils import real_vector_from_list


def test_products() raises:
    var D = DiagonalOperator([1.0, 2.0, 3.0])
    var A = CSRMatrix.from_coo(
        3, 3, [0, 1, 2, 2], [1, 2, 0, 2], [1.0, 1.0, 1.0, 1.0]
    )
    # D @ A scales rows, A @ D scales columns
    var DA = D @ A
    var AD = A @ D
    var Dc = D.to_csr()
    var DA_ref = Dc @ A
    var AD_ref = A @ Dc
    for i in range(3):
        for j in range(3):
            assert_equal(DA[i, j], DA_ref[i, j])
            assert_equal(AD[i, j], AD_ref[i, j])
    assert_equal(DA.nnz(), A.nnz())
    # D @ D stays diagonal
    var DD = D @ D
    assert_equal(DD[2], 9.0)
    assert_equal(DD[0, 1], 0.0)
    # D @ x is elementwise
    var y = D @ real_vector_from_list([1.0, 1.0, 2.0])
    assert_equal(y.unsafe_get(2), 6.0)


def test_inverse_inner_product_norm() raises:
    var D = DiagonalOperator([4.0, 0.5])
    var Di = D.inverse()
    assert_equal(Di[0], 0.25)
    assert_equal((D @ Di)[1], 1.0)
    var x = real_vector_from_list([1.0, 2.0])
    var z = real_vector_from_list([3.0, -1.0])
    assert_equal(D.inner_product(x, z), 4.0 * 3.0 + 0.5 * -2.0)
    assert_equal(D.norm(x), sqrt(4.0 + 0.5 * 4.0))
    with assert_raises():
        _ = DiagonalOperator([1.0, 0.0]).inverse()


def test_arithmetic_and_errors() raises:
    var D = DiagonalOperator([1.0, -2.0])
    var E = 2.0 * D + D * 3.0 - (-D) / 2.0
    assert_equal(E[1], -2.0 * 5.5)
    assert_equal(len(DiagonalOperator.identity(4)), 4)
    var dense = D.todense()
    assert_equal(dense.unsafe_get(3), -2.0)
    with assert_raises():
        _ = D @ DiagonalOperator([1.0])
    with assert_raises():
        _ = D @ CSRMatrix.identity(3)
    with assert_raises():
        _ = D[0, 2]
    with assert_raises(contains="index 2 out of range"):
        _ = D[2]
    with assert_raises(contains="index -1 out of range"):
        D[-1] = 1.0


def test_hodge_stars_are_diagonal() raises:
    var V = nm.fromstring[nm.f64]("[[0,0],[1,0],[1,1],[0,1]]")
    var S = nm.fromstring[DType.int]("[[0,1,3],[1,2,3]]")
    var sc = SimplicialComplex(V, S)
    # star on a cochain agrees with the operator applied to its values
    var c = sc.get_cochain(1)
    for i in range(sc[1].num_simplices):
        c[i] = Float64(i + 1)
    var sc1 = star(c)
    for i in range(sc[1].num_simplices):
        assert_equal(sc1[i], sc[1].star[i] * Float64(i + 1))
    # composed with d, a CSRMatrix comes out, matching the CSR route
    var A = sc[1].star @ sc[0].d
    var B = sc[1].star.to_csr() @ sc[0].d
    for i in range(A.nrows):
        for j in range(A.ncols):
            assert_equal(A[i, j], B[i, j])
    # star on a basis gives the matrix
    var Sb = star(sc.get_cochain_basis(1)).v.copy()
    for i in range(sc[1].num_simplices):
        assert_equal(Sb[i, i], sc[1].star[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
