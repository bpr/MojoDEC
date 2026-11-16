from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    FillOrdering,
    SimplicialComplex,
    SparseCholesky,
    SparseLDLT,
    amd_ordering,
    cg,
    kuhn_grid,
    whitney_innerproduct,
)
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import RealArray, norm2, probe_vector, real_vector
from mojodec.routines.linalg import eigh_generalized, spsolve


def _laplacian(sc: SimplicialComplex, k: Int) raises -> CSRMatrix:
    """The Whitney Hodge stiffness plus mass on k-forms: SPD."""
    var A = whitney_innerproduct(sc, k)
    if k < sc.complex_dimension():
        A = A + sc[k].d.transpose() @ whitney_innerproduct(sc, k + 1) @ sc[k].d
    return A^


def _saddle(
    M: CSRMatrix, B: CSRMatrix, constraint_first: Bool
) raises -> CSRMatrix:
    """[[-M, B^T], [B, 0]], or with the blocks swapped, [[0, B], [B^T, -M]];
    the zero block is not stored."""
    var m = M.nrows
    var c = B.nrows
    var rows = List[Int]()
    var cols = List[Int]()
    var vals = List[Float64]()
    var om = c if constraint_first else 0  # offset of the M block
    var oc = 0 if constraint_first else m  # offset of the constraint block
    for i in range(m):
        for t in range(M.indptr[i], M.indptr[i + 1]):
            rows.append(om + i)
            cols.append(om + M.indices[t])
            vals.append(-M.data[t])
    for i in range(c):
        for t in range(B.indptr[i], B.indptr[i + 1]):
            rows.append(oc + i)
            cols.append(om + B.indices[t])
            vals.append(B.data[t])
            rows.append(om + B.indices[t])
            cols.append(oc + i)
            vals.append(B.data[t])
    return CSRMatrix.from_coo(m + c, m + c, rows, cols, vals)


def _residual(A: CSRMatrix, x: RealArray, b: RealArray) raises -> Float64:
    var r = A @ x
    for i in range(b.size):
        r.unsafe_set(i, r.unsafe_get(i) - b.unsafe_get(i))
    return norm2(r) / norm2(b)


def _max_diff(x: RealArray, y: RealArray) -> Float64:
    var m: Float64 = 0
    for i in range(x.size):
        m = max(m, abs(x.unsafe_get(i) - y.unsafe_get(i)))
    return m


def test_amd_is_a_permutation_with_zero_diagonals_last() raises:
    var sc = SimplicialComplex(kuhn_grid([10, 10]))
    var K = _laplacian(sc, 0)
    var p = amd_ordering(K)
    var n = K.nrows
    var seen = List[Bool](length=n, fill=False)
    for k in range(n):
        assert_true(not seen[p[k]])
        seen[p[k]] = True
    # fewer nonzeros in L than with the natural (banded) order
    var amd = SparseCholesky(K).factor.factor_nonzeros()
    var natural = SparseCholesky(
        K, FillOrdering.NATURAL
    ).factor.factor_nonzeros()
    assert_true(amd < natural)
    # the constraint rows of a saddle point come last
    var A = _saddle(whitney_innerproduct(sc, 1), sc[1].d, True)
    var c = sc[2].num_simplices
    var q = amd_ordering(A)
    for k in range(len(q) - c, len(q)):
        assert_true(q[k] < c)
    assert_equal(len(amd_ordering(CSRMatrix(0, 0))), 0)
    with assert_raises(contains="the matrix is 3 x 4"):
        _ = amd_ordering(CSRMatrix(3, 4))


def _pattern(sc: SimplicialComplex, k: Int) raises -> CSRMatrix:
    ref d = sc[k].d
    return d.transpose() @ d + CSRMatrix.identity(d.ncols)


def _read_perm(path: String) raises -> List[Int]:
    var out = List[Int]()
    with open(path, "r") as f:
        for word in f.read().split():
            out.append(atol(word))
    return out^


def test_amd_matches_faer() raises:
    # tests/data/amd: faer's permutations for the same matrices; the port
    # breaks ties as faer does, so they agree entry for entry
    var g2 = SimplicialComplex(kuhn_grid([20, 20]))
    var g3 = SimplicialComplex(kuhn_grid([14, 14, 14]))
    var n = 400
    var r = List[Int]()
    var c = List[Int]()
    var v = List[Float64]()
    var s: UInt64 = 12345
    for i in range(n):
        r.append(i)
        c.append(i)
        v.append(10.0)
    for _ in range(1200):
        s = s * 6364136223846793005 + 1442695040888963407
        var i = Int((s >> 33) % UInt64(n))
        s = s * 6364136223846793005 + 1442695040888963407
        var j = Int((s >> 33) % UInt64(n))
        if i != j:
            r.append(i)
            c.append(j)
            v.append(-1.0)
            r.append(j)
            c.append(i)
            v.append(-1.0)
    for j in range(1, n):
        r.append(0)
        c.append(j)
        v.append(-0.1)
        r.append(j)
        c.append(0)
        v.append(-0.1)
    # d^T d + I has the pattern of the Whitney stiffness plus mass, which
    # is what faer was given, and is much cheaper to build
    var mats = List[CSRMatrix]()
    mats.append(_pattern(g2, 1))
    mats.append(_pattern(g3, 0))
    mats.append(CSRMatrix.from_coo(n, n, r, c, v))
    var names: List[String] = ["grid2_1", "grid3big_0", "random"]
    for i in range(3):
        var want = _read_perm("tests/data/amd/" + names[i] + ".perm")
        var got = amd_ordering(mats[i])
        assert_equal(len(got), len(want))
        for k in range(len(want)):
            assert_equal(got[k], want[k], String(names[i], " at ", k))


def test_cholesky_solves_dec_systems() raises:
    var grids = List[List[Int]]()
    grids.append([8, 8])
    grids.append([4, 4, 4])
    for ref dims in grids:
        var sc = SimplicialComplex(kuhn_grid(dims))
        for k in range(sc.complex_dimension() + 1):
            var K = _laplacian(sc, k)
            var b = probe_vector(K.nrows, 7)
            var F = SparseCholesky(K)
            var x = F.solve(b)
            assert_true(_residual(K, x, b) < 1e-13)
            # against the dense pivoted LU; cond(K) <= ~1e3 here
            assert_true(_max_diff(x, spsolve(K, b)) < 1e-11 * (1 + norm2(x)))
            var inertia = F.factor.inertia()
            assert_equal(inertia.positive, K.nrows)
            assert_equal(F.factor.num_regularized, 0)
            # an exact preconditioner: CG converges in one step
            var res = cg(K, b, M=F, rtol=1e-12)
            assert_true(res.converged())
            assert_true(res.iterations <= 2)


def test_ldlt_solves_saddle_points_without_pivoting() raises:
    # mixed Darcy: [[-M1, d1^T], [d1, 0]] with the Whitney mass, the zero
    # block absent. AMD orders the faces last, so every pivot is the
    # negative of a mass pivot or of the Schur complement d1 M1^-1 d1^T
    var sc = SimplicialComplex(kuhn_grid([8, 8]))
    var M = whitney_innerproduct(sc, 1)
    var m = M.nrows
    var c = sc[2].num_simplices
    for first in [False, True]:
        var A = _saddle(M, sc[1].d, first)
        var b = probe_vector(m + c, 3)
        var F = SparseLDLT(A)
        assert_equal(F.num_regularized, 0)
        var x = F.solve(b)
        assert_true(_residual(A, x, b) < 1e-12)
        assert_true(_max_diff(x, spsolve(A, b)) < 1e-10 * (1 + norm2(x)))
        var inertia = F.inertia()
        assert_equal(inertia.negative, m)
        assert_equal(inertia.positive, c)
        # not positive definite
        with assert_raises(contains="not positive definite"):
            _ = SparseCholesky(A)
    # in the natural order with the constraints first, the first pivot is
    # the zero block's: refused, naming the row
    var Z = _saddle(M, sc[1].d, True)
    with assert_raises(contains="pivot 0 (row 0 of the matrix) is 0"):
        _ = SparseLDLT(Z, ordering=FillOrdering.NATURAL)


def test_regularization_and_refinement() raises:
    # the constraints-first order again, now with dynamic regularization:
    # pivots of the wrong sign become +-delta, and refinement against the
    # true matrix recovers the solution
    var sc = SimplicialComplex(kuhn_grid([6, 6]))
    var M = whitney_innerproduct(sc, 1)
    var m = M.nrows
    var c = sc[2].num_simplices
    var A = _saddle(M, sc[1].d, True)
    var signs = List[Int]()
    for _ in range(c):
        signs.append(1)  # the constraints' pivots are positive
    for _ in range(m):
        signs.append(-1)
    var b = probe_vector(m + c, 5)
    var F = SparseLDLT(
        A,
        ordering=FillOrdering.NATURAL,
        signs=signs^,
        regularization=1e-10,
        refinement_steps=3,
    )
    assert_true(F.num_regularized > 0)
    var x = F.solve(b)
    assert_true(_residual(A, x, b) < 1e-10)
    # without refinement the regularized factor is off
    F.refinement_steps = 0
    assert_true(_residual(A, F.solve(b), b) > _residual(A, x, b))


def test_ldlt_inertia_of_a_shifted_laplacian() raises:
    # K - s M has as many negative eigenvalues as K x = lambda M x has
    # below s (Sylvester's law of inertia)
    var sc = SimplicialComplex(kuhn_grid([7, 7]))
    var K = sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d
    var M = whitney_innerproduct(sc, 0)
    var lam = eigh_generalized(K.todense(), M.todense()).values.copy()
    for s in [0.5, 7.0, 30.0]:
        var below = 0
        for i in range(lam.size):
            if lam.unsafe_get(i) < s:
                below += 1
        var F = SparseLDLT(K - M * s)
        assert_equal(F.inertia().negative, below)
        assert_equal(F.inertia().positive, K.nrows - below)


def _ints(xs: List[Int]) -> List[Int]:
    return xs.copy()


def test_errors() raises:
    with assert_raises(contains="the matrix is 2 x 3"):
        _ = SparseLDLT(CSRMatrix(2, 3))
    var r: List[Int] = [0, 0, 1]
    var c: List[Int] = [0, 1, 1]
    var v: List[Float64] = [2.0, 1.0, 2.0]
    with assert_raises(contains="not symmetric"):
        _ = SparseLDLT(CSRMatrix.from_coo(2, 2, r, c, v))
    # singular: the Neumann Laplacian of a path, constants in its kernel
    var rs: List[Int] = [0, 0, 1, 1]
    var cs: List[Int] = [0, 1, 0, 1]
    var vs: List[Float64] = [1.0, -1.0, -1.0, 1.0]
    with assert_raises(contains="is 0.0"):
        _ = SparseLDLT(CSRMatrix.from_coo(2, 2, rs, cs, vs))
    var I2 = CSRMatrix.identity(2)
    with assert_raises(contains="1 signs for 2 rows"):
        _ = SparseLDLT(I2, signs=_ints([1]), regularization=1e-8)
    with assert_raises(contains="sign 1 is 0, not +-1"):
        _ = SparseLDLT(I2, signs=_ints([1, 0]), regularization=1e-8)
    with assert_raises(contains="regularization must be >= 0"):
        _ = SparseLDLT(I2, regularization=-1.0)
    var F = SparseLDLT(I2)
    with assert_raises(contains="b has length 3, not 2"):
        _ = F.solve(real_vector(3))
    var y = real_vector(2)
    with assert_raises(contains="vectors of length 3 and 2 for n = 2"):
        F.apply_into(real_vector(3), y)
    # the empty matrix factors, and solves the empty system
    var E = SparseLDLT(CSRMatrix(0, 0))
    assert_equal(E.solve(real_vector(0)).size, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
