"""The typed eigensolver; `eigsh` itself is tested in molina."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    MatrixOp,
    PRIMAL,
    SimplicialComplex,
    TypedComplex,
    eigsh,
    kuhn_grid,
    whitney_innerproduct,
)
from molina.diagonal import DiagonalOperator
from molina.sparse import CSRMatrix
from mojodec.core.utils import RealArray, dot, fget, real_vector
from mojodec.routines.linalg import eigh_generalized


def _worst(eta: RealArray) -> Float64:
    var w: Float64 = 0
    for i in range(eta.size):
        w = max(w, eta.unsafe_get(i))
    return w


def test_typed_cavity() raises:
    # the whole square, no boundary condition: the curl-curl kernel is
    # range(d0), and d0 drops rank only by the constants, so leaving out
    # one vertex gives a G of full rank with the same range
    var tc = TypedComplex[2](kuhn_grid([8, 8]))
    var d1 = tc.d[1, PRIMAL]()
    var K = d1.T() @ whitney_innerproduct[2](tc) @ d1
    var M = whitney_innerproduct[1](tc)
    var cols = List[Int]()
    for v in range(1, tc.sc[0].num_simplices):
        cols.append(v)
    var G = MatrixOp[2, 0, PRIMAL, 1, PRIMAL](
        tc.d[0, PRIMAL]().mat.select(_all(tc.sc[1].num_simplices), cols),
        tc.sc,
    )
    var eig = eigsh(K, M, 4, sigma=-1.0, deflate=G)
    var untyped = eigsh(K.mat, M.mat, 4, sigma=-1.0, deflate=G.mat)
    assert_equal(len(eig), 4)
    var mode = eig.mode(0)  # TypedCochain[2, 1, PRIMAL]
    assert_equal(len(mode), tc.sc[1].num_simplices)
    assert_true(_worst(eig.residuals(K, M)) < 1e-13)
    for i in range(4):
        # the Neumann field's spectrum: pi^2 (1, 1, 2, 4) on a coarse grid
        assert_true(eig.values.unsafe_get(i) > 9.0)
        assert_almost_equal(
            eig.values.unsafe_get(i),
            untyped.values.unsafe_get(i),
            atol=1e-12 * eig.values.unsafe_get(i),
            rtol=0,
        )
    # without the deflation, the kernel (eigenvalue 0) is nearest
    var plain = eigsh(K, M, 2, sigma=-1.0)
    assert_true(abs(plain.values.unsafe_get(0)) < 1e-10)


def _all(n: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(n):
        out.append(i)
    return out^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
