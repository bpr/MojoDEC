from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    DUAL,
    PRIMAL,
    BlockOp,
    CochainPair,
    CSRMatrix,
    MatrixOp,
    TypedComplex,
    minres,
    simplicial_grid_2d,
    spsolve,
)
from mojodec.core.utils import fget, fset


def _grid(n: Int) raises -> TypedComplex[2]:
    """A grid with its interior vertices moved, so every dual edge has
    positive length."""
    var m = simplicial_grid_2d(n)
    var V = m.vertices.copy()
    var h = 1.0 / Float64(n)
    for i in range(1, n):
        for j in range(1, n):
            var v = i * (n + 1) + j
            fset(V, v, 0, fget(V, v, 0) + 0.1 * h * Float64((v * 7) % 5 - 2))
            fset(V, v, 1, fget(V, v, 1) + 0.1 * h * Float64((v * 3) % 5 - 2))
    return TypedComplex[2](V, m.elements)


def _darcy(
    tc: TypedComplex[2],
) raises -> BlockOp[2, 1, PRIMAL, 0, DUAL, 1, DUAL, 2, PRIMAL]:
    """The mixed Darcy operator [[-star1, d1^T], [d1, 0]]."""
    var d1 = tc.d[1, PRIMAL]()
    return BlockOp(
        tc.star[1, PRIMAL]().to_matrix_op() * -1.0,
        d1.T(),
        d1,
        tc.zero[0, DUAL, 2, PRIMAL](),
    )


def _pair(
    tc: TypedComplex[2], seed: Int
) raises -> CochainPair[2, 1, PRIMAL, 0, DUAL]:
    var u = tc.cochain[1, PRIMAL]()
    for i in range(len(u)):
        u[i] = Float64((i * 7 + seed) % 11) - 5.0
    var p = tc.cochain[0, DUAL]()
    for i in range(len(p)):
        p[i] = Float64((i * 5 + seed) % 9) - 4.0
    return CochainPair(u^, p^)


def test_apply_matches_assembled() raises:
    var tc = _grid(4)
    var A = _darcy(tc)
    assert_equal(A.num_unknowns(), tc.num_cells[1, PRIMAL]() + 32)
    assert_equal(A.num_equations(), A.num_unknowns())
    var x = _pair(tc, 1)
    var y = A @ x  # CochainPair[2, 1, DUAL, 2, PRIMAL]
    var ref_ = A.to_csr() @ x.stacked()
    var s = y.stacked()
    for i in range(s.size):
        assert_almost_equal(
            s.unsafe_get(i), ref_.unsafe_get(i), atol=1e-14, rtol=0
        )
    # the blocks keep their kinds: the second equation is the divergence
    var div = tc.d[1, PRIMAL]() @ x.first
    for i in range(len(div)):
        assert_equal(y.second[i], div[i])
    assert_true(String(A).startswith("BlockOp[2]: (primal 1, dual 0)"))


def test_solve_with_fixed_unknowns() raises:
    var tc = _grid(5)
    var A = _darcy(tc)
    var boundary = tc.boundary_cells[1]()
    var pinned = tc.subset[0, DUAL]([0])
    var known = _pair(tc, 3)
    var b = CochainPair(tc.cochain[1, DUAL](), tc.cochain[2, PRIMAL]())
    var x = spsolve(
        A, b, values=known, fixed_first=boundary, fixed_second=pinned
    )
    # the given unknowns are kept
    for i in boundary.indices():
        assert_equal(x.first[i], known.first[i])
    assert_equal(x.second[0], known.second[0])
    # the other equations hold
    var r = A @ x - b
    for i in range(len(r.first)):
        if i not in boundary:
            assert_almost_equal(r.first[i], 0.0, atol=1e-13, rtol=0)
    for i in range(1, len(r.second)):
        assert_almost_equal(r.second[i], 0.0, atol=1e-13, rtol=0)


def test_solve_whole_system() raises:
    var tc = _grid(4)
    var d1 = tc.d[1, PRIMAL]()
    var N2 = tc.num_cells[2, PRIMAL]()
    # [[star1, d1^T], [d1, -I]] is non-singular
    var A = BlockOp(
        tc.star[1, PRIMAL]().to_matrix_op(),
        d1.T(),
        d1,
        tc.dual_diagonal[0, DUAL](
            List[Float64](length=N2, fill=-1.0)
        ).to_matrix_op(),
    )
    var x0 = _pair(tc, 5)
    var b = A @ x0
    var x = spsolve(A, b)
    var e = (x - x0).stacked()
    for i in range(e.size):
        assert_almost_equal(e.unsafe_get(i), 0.0, atol=5e-13, rtol=0)


def test_minres_on_blocks() raises:
    # the saddle-point Darcy system [[-star1, d1^T], [d1, 0]]: symmetric,
    # indefinite, and non-singular (d1 is onto on a disk)
    var tc = _grid(5)
    var A = _darcy(tc)
    var x0 = _pair(tc, 2)
    var b = A @ x0
    var res = minres(A, b, rtol=1e-13)
    assert_true(res.converged())
    var x = res^.take_x()  # CochainPair[2, 1, PRIMAL, 0, DUAL]
    var direct = spsolve(A, b)
    var e = (x - direct).stacked()
    for i in range(e.size):
        # rtol=1e-13 times cond(A) ~ 1e3, |x| ~ 30
        assert_almost_equal(e.unsafe_get(i), 0.0, atol=1e-8, rtol=0)
    # the blocks are applied one by one, the same as assembled
    var y = A.to_csr() @ x0.stacked()
    var z = b.stacked()
    for i in range(y.size):
        assert_almost_equal(
            z.unsafe_get(i), y.unsafe_get(i), atol=1e-14, rtol=0
        )
    # a block system that is not symmetric is refused
    var d1 = tc.d[1, PRIMAL]()
    var N = BlockOp(
        tc.star[1, PRIMAL]().to_matrix_op() * -1.0,
        d1.T() * 2.0,
        d1,
        tc.zero[0, DUAL, 2, PRIMAL](),
    )
    with assert_raises(contains="not symmetric"):
        _ = minres(N, b)


def test_errors() raises:
    var tc = _grid(3)
    var other = _grid(3)
    var d1 = tc.d[1, PRIMAL]()
    with assert_raises(contains="different complexes"):
        _ = BlockOp(
            tc.star[1, PRIMAL]().to_matrix_op(),
            d1.T(),
            other.d[1, PRIMAL](),
            tc.zero[0, DUAL, 2, PRIMAL](),
        )
    with assert_raises(contains="line up"):
        _ = BlockOp(
            tc.star[1, PRIMAL]().to_matrix_op(),
            d1.T(),
            MatrixOp[2, 1, PRIMAL, 2, PRIMAL](CSRMatrix(3, 3)),
            tc.zero[0, DUAL, 2, PRIMAL](),
        )
    with assert_raises(contains="different complexes"):
        _ = CochainPair(tc.cochain[1, PRIMAL](), other.cochain[0, DUAL]())
    with assert_raises(contains="different complexes"):
        _ = _darcy(tc) @ _pair(other, 1)
    # a block of the wrong kind, e.g. d1 (primal 1 -> primal 2) as a12,
    # does not compile; nor does the fixed-unknown solve when an equation
    # is not of the dual kind of its unknown


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
