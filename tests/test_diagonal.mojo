"""Hodge stars as diagonal operators; `DiagonalOperator` itself is tested
in molina."""

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
from mojodec.core.utils import real_array_from_string


def test_hodge_stars_are_diagonal() raises:
    var V = real_array_from_string("[[0,0],[1,0],[1,1],[0,1]]")
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
