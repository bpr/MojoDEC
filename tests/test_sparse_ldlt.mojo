from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    DUAL,
    PRIMAL,
    BlockOp,
    BlockSparseLDLT,
    CochainPair,
    FactorLayout,
    FillOrdering,
    TypedCochain,
    TypedComplex,
    TypedSparseCholesky,
    TypedSparseLDLT,
    SimplicialComplex,
    SparseCholesky,
    SparseLDLT,
    amd_ordering,
    cholesky_nonzeros,
    cg,
    kuhn_grid,
    whitney_innerproduct,
)
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    RealArray,
    fget,
    fset,
    norm2,
    probe_vector,
    real_array,
    real_vector,
)
from mojodec.routines.linalg import eigh_generalized, spsolve
from mojodec.routines.supernodal import etree_postorder


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
        # the default refinement step leaves a smaller residual than
        # the bare triangular solves
        var bare = SparseLDLT(A, refinement_steps=0)
        assert_true(_residual(A, x, b) <= _residual(A, bare.solve(b), b))
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


def test_symbolic_count_and_given_permutation() raises:
    # cholesky_nonzeros counts L from the elimination tree alone; it
    # matches the factor, for AMD and the natural order, in 2D and 3D
    var grids = List[List[Int]]()
    grids.append([9, 7])
    grids.append([4, 5, 3])
    for ref dims in grids:
        var sc = SimplicialComplex(kuhn_grid(dims))
        var K = _laplacian(sc, 1)
        var p = amd_ordering(K)
        var F = SparseLDLT(K, perm=p.copy())
        assert_equal(cholesky_nonzeros(K, p), F.factor_nonzeros())
        assert_equal(F.factor_nonzeros(), SparseLDLT(K).factor_nonzeros())
        var natural = List[Int]()
        for i in range(K.nrows):
            natural.append(i)
        assert_equal(
            cholesky_nonzeros(K, natural),
            SparseLDLT(K, ordering=FillOrdering.NATURAL).factor_nonzeros(),
        )
        # a given permutation is the ordering
        var b = probe_vector(K.nrows, 2)
        var G = SparseCholesky(K, perm=natural^)
        assert_true(_residual(K, G.solve(b), b) < 1e-13)
    var I3 = CSRMatrix.identity(3)
    with assert_raises(contains="perm has 2 entries for 3"):
        _ = SparseLDLT(I3, perm=_ints([0, 1]))
    with assert_raises(contains="perm is not a permutation (entry 2)"):
        _ = cholesky_nonzeros(I3, _ints([0, 1, 1]))
    with assert_raises(contains="perm is not a permutation (entry 0)"):
        _ = SparseLDLT(I3, perm=_ints([3, 1, 0]))


def test_supernodal_agrees_with_simplicial() raises:
    # the same L D L^T in dense blocks: every degree in 2D and 3D, the
    # Darcy saddle point in both block orders, and an indefinite shifted
    # Laplacian; same fill, same inertia, the same solution
    var mats = List[CSRMatrix]()
    var grids = List[List[Int]]()
    grids.append([9, 8])
    grids.append([4, 4, 3])
    for ref dims in grids:
        var sc = SimplicialComplex(kuhn_grid(dims))
        for k in range(sc.complex_dimension() + 1):
            mats.append(_laplacian(sc, k))
    var sc = SimplicialComplex(kuhn_grid([7, 6]))
    var M1 = whitney_innerproduct(sc, 1)
    mats.append(_saddle(M1, sc[1].d, False))
    mats.append(_saddle(M1, sc[1].d, True))
    var K = sc[0].d.transpose() @ M1 @ sc[0].d
    mats.append(K - whitney_innerproduct(sc, 0) * 20.0)
    for ref A in mats:
        var b = probe_vector(A.nrows, 4)
        var S = SparseLDLT(A, layout=FactorLayout.SIMPLICIAL)
        var N = SparseLDLT(A, layout=FactorLayout.SUPERNODAL)
        assert_true(S.layout == FactorLayout.SIMPLICIAL)
        assert_true(N.layout == FactorLayout.SUPERNODAL)
        assert_equal(S.num_supernodes(), 0)
        # (a diagonal matrix, the top-degree mass, has one per column)
        assert_true(0 < N.num_supernodes() and N.num_supernodes() <= A.nrows)
        assert_equal(N.factor_nonzeros(), S.factor_nonzeros())
        assert_true(N.stored_entries() >= N.factor_nonzeros())
        assert_equal(N.inertia().negative, S.inertia().negative)
        assert_equal(N.num_regularized, 0)
        var x = N.solve(b)
        assert_true(_residual(A, x, b) < 1e-13)
        assert_true(_max_diff(x, S.solve(b)) < 1e-11 * (1 + norm2(x)))
        # the postordered permutation is still a permutation
        var seen = List[Bool](length=A.nrows, fill=False)
        for k in range(A.nrows):
            assert_true(not seen[N.perm[k]])
            seen[N.perm[k]] = True
    # an exact preconditioner, as the simplicial one
    var A = _laplacian(SimplicialComplex(kuhn_grid([5, 5, 5])), 1)
    var C = SparseCholesky(A, layout=FactorLayout.SUPERNODAL)
    var b = probe_vector(A.nrows, 2)
    var res = cg(A, b, M=C, rtol=1e-12)
    assert_true(res.converged() and res.iterations <= 2)
    # AUTO: simplicial for a 2D grid's few flops per entry, supernodal
    # for a 3D grid's many
    assert_true(
        SparseLDLT(_laplacian(SimplicialComplex(kuhn_grid([8, 8])), 0)).layout
        == FactorLayout.SIMPLICIAL
    )
    assert_true(SparseLDLT(A).layout == FactorLayout.SUPERNODAL)


def test_supernodal_regularization_and_errors() raises:
    # the constraints-first saddle point in the natural order (postordered,
    # which keeps the constraints' zero pivots early): refused, or
    # regularized as by the simplicial factor
    var sc = SimplicialComplex(kuhn_grid([6, 6]))
    var M = whitney_innerproduct(sc, 1)
    var m = M.nrows
    var c = sc[2].num_simplices
    var Z = _saddle(M, sc[1].d, True)
    with assert_raises(contains="is 0.0; the matrix is singular"):
        _ = SparseLDLT(
            Z, ordering=FillOrdering.NATURAL, layout=FactorLayout.SUPERNODAL
        )
    var signs = List[Int]()
    for _ in range(c):
        signs.append(1)
    for _ in range(m):
        signs.append(-1)
    var b = probe_vector(m + c, 5)
    var F = SparseLDLT(
        Z,
        ordering=FillOrdering.NATURAL,
        signs=signs^,
        regularization=1e-10,
        refinement_steps=3,
        layout=FactorLayout.SUPERNODAL,
    )
    assert_true(F.num_regularized > 0)
    assert_true(_residual(Z, F.solve(b), b) < 1e-10)
    # not positive definite
    with assert_raises(contains="not positive definite"):
        _ = SparseCholesky(Z, layout=FactorLayout.SUPERNODAL)
    # the empty matrix
    var E = SparseLDLT(CSRMatrix(0, 0), layout=FactorLayout.SUPERNODAL)
    assert_equal(E.solve(real_vector(0)).size, 0)


def test_etree_postorder() raises:
    # a forest: 5 has children 3 and 4, 4 has 0 and 2, 2 has 1; 6 has 7
    # (a parent need not have a larger label). Children in increasing
    # order, each node after its subtree
    var etree: List[Int] = [4, 2, 4, 5, 5, -1, -1, 6]
    var post = etree_postorder(etree)
    var want: List[Int] = [3, 0, 1, 2, 4, 5, 7, 6]
    assert_equal(len(post), len(want))
    for k in range(len(want)):
        assert_equal(post[k], want[k])
    assert_equal(len(etree_postorder(List[Int]())), 0)


def test_typed_factorizations() raises:
    var tc = TypedComplex[2](kuhn_grid([6, 5]))
    var d0 = tc.d[0, PRIMAL]()
    var W0 = whitney_innerproduct[0](tc)
    var W1 = whitney_innerproduct[1](tc)
    # stiffness plus mass on 0-forms: primal 0 -> dual 2
    var K = d0.T() @ W1 @ d0 + W0
    var f = W0 @ tc.cochain[0, PRIMAL]()
    for i in range(len(f)):
        f[i] = Float64(i % 5) - 2.0
    var C = TypedSparseCholesky(K)
    var u = C.solve(f)  # TypedCochain[2, 0, PRIMAL]
    var dense = spsolve(K, f)
    for i in range(len(u)):
        assert_almost_equal(u[i], dense[i], atol=1e-12, rtol=0)
    # LDL^T of the indefinite K - s W0: inertia counts eigenvalues below s
    var L = TypedSparseLDLT(K - W0 * 30.0)
    var lam = eigh_generalized(K.mat.todense(), W0.mat.todense()).values.copy()
    var below = 0
    for i in range(lam.size):
        if lam.unsafe_get(i) < 30.0:
            below += 1
    assert_equal(L.inertia().negative, below)
    assert_true(L.factor_nonzeros() > K.mat.nrows)
    var x = L.solve(f)
    var r = (K - W0 * 30.0) @ x
    for i in range(len(r)):
        assert_almost_equal(r[i], f[i], atol=1e-11, rtol=0)
    # the mixed Darcy saddle point with the Whitney mass, factored once:
    # [[-W1, d1^T], [d1, 0]], fluxes primal 1, pressures dual 0
    var d1 = tc.d[1, PRIMAL]()
    var A = BlockOp(W1 * -1.0, d1.T(), d1, tc.zero[0, DUAL, 2, PRIMAL]())
    var F = BlockSparseLDLT(A)
    var m = tc.num_cells[1, PRIMAL]()
    var c = tc.num_cells[2, PRIMAL]()
    assert_equal(F.inertia().negative, m)
    assert_equal(F.inertia().positive, c)
    for seed in range(2):
        var g = tc.cochain[1, DUAL]()
        var h = tc.cochain[2, PRIMAL]()
        for i in range(len(g)):
            g[i] = Float64((i * 3 + seed) % 7) - 3.0
        for i in range(len(h)):
            h[i] = Float64((i + seed) % 4) - 1.5
        var b = CochainPair(g^, h^)
        var y = F.solve(b)  # CochainPair[2, 1, PRIMAL, 0, DUAL]
        var e = (y - spsolve(A, b)).stacked()
        for i in range(e.size):
            assert_almost_equal(e.unsafe_get(i), 0.0, atol=1e-11, rtol=0)
    # cochains from another complex are refused
    var other = TypedComplex[2](kuhn_grid([6, 5]))
    with assert_raises(contains="TypedSparseCholesky.solve: b is on a"):
        _ = C.solve(other.cochain[2, DUAL]())
    with assert_raises(contains="BlockSparseLDLT.solve: b is on a"):
        _ = F.solve(
            CochainPair(other.cochain[1, DUAL](), other.cochain[2, PRIMAL]())
        )


def test_typed_multiple_right_hand_sides() raises:
    # a list of cochains solved as one block: each result the single
    # solve bit for bit, for the three typed wrappers
    var tc = TypedComplex[2](kuhn_grid([7, 6]))
    var d0 = tc.d[0, PRIMAL]()
    var W0 = whitney_innerproduct[0](tc)
    var W1 = whitney_innerproduct[1](tc)
    var K = d0.T() @ W1 @ d0 + W0
    var fs = List[TypedCochain[2, 2, DUAL]]()
    for seed in range(5):
        var f = tc.cochain[2, DUAL]()
        for i in range(len(f)):
            f[i] = Float64((i * 3 + seed) % 7) - 3.0
        fs.append(f^)
    var C = TypedSparseCholesky(K)
    var L = TypedSparseLDLT(K - W0 * 30.0)
    var us = C.solve(fs)  # List[TypedCochain[2, 0, PRIMAL]]
    var xs = L.solve(fs)
    assert_equal(len(us), 5)
    assert_equal(len(xs), 5)
    for c in range(5):
        var u = C.solve(fs[c])
        var x = L.solve(fs[c])
        for i in range(len(u)):
            assert_equal(us[c][i], u[i])
            assert_equal(xs[c][i], x[i])
    assert_equal(len(C.solve(List[TypedCochain[2, 2, DUAL]]())), 0)
    # the mixed Darcy saddle point
    var d1 = tc.d[1, PRIMAL]()
    var A = BlockOp(W1 * -1.0, d1.T(), d1, tc.zero[0, DUAL, 2, PRIMAL]())
    var F = BlockSparseLDLT(A)
    var bs = List[CochainPair[2, 1, DUAL, 2, PRIMAL]]()
    for seed in range(3):
        var g = tc.cochain[1, DUAL]()
        var h = tc.cochain[2, PRIMAL]()
        for i in range(len(g)):
            g[i] = Float64((i * 3 + seed) % 7) - 3.0
        for i in range(len(h)):
            h[i] = Float64((i + seed) % 4) - 1.5
        bs.append(CochainPair(g^, h^))
    var ys = F.solve(bs)
    for c in range(3):
        var y = F.solve(bs[c])
        for i in range(len(y.first)):
            assert_equal(ys[c].first[i], y.first[i])
        for i in range(len(y.second)):
            assert_equal(ys[c].second[i], y.second[i])
    # cochains from another complex are refused
    var other = TypedComplex[2](kuhn_grid([7, 6]))
    var mixed = List[TypedCochain[2, 2, DUAL]]()
    mixed.append(fs[0].copy())
    mixed.append(other.cochain[2, DUAL]())
    with assert_raises(contains="TypedSparseCholesky.solve: b is on a"):
        _ = C.solve(mixed)


def _ints(xs: List[Int]) -> List[Int]:
    return xs.copy()


def test_multiple_right_hand_sides() raises:
    # X = A^-1 B for an n x m B: each column the single-vector solve, bit
    # for bit, for both layouts, the saddle point with refinement, and
    # SparseCholesky
    var sc = SimplicialComplex(kuhn_grid([5, 5, 5]))
    var mats = List[CSRMatrix]()
    mats.append(_laplacian(sc, 1))
    mats.append(_saddle(whitney_innerproduct(sc, 2), sc[2].d, False))
    for t in range(len(mats)):
        ref A = mats[t]
        var n = A.nrows
        for m in [1, 2, 3, 5, 9]:
            var B = real_array(n, m)
            for c in range(m):
                var b = probe_vector(n, UInt64(c + 1))
                for i in range(n):
                    fset(B, i, c, b.unsafe_get(i))
            for layout in [FactorLayout.SIMPLICIAL, FactorLayout.SUPERNODAL]:
                var F = SparseLDLT(A, layout=layout)
                var X = F.solve(B)
                assert_equal(X.rows(), n)
                assert_equal(X.cols(), m)
                for c in range(m):
                    var b = real_vector(n)
                    for i in range(n):
                        b.unsafe_set(i, fget(B, i, c))
                    var x = F.solve(b)
                    for i in range(n):
                        assert_equal(fget(X, i, c), x.unsafe_get(i))
                    assert_true(_residual(A, x, b) < 1e-12)
    var C = SparseCholesky(mats[0])
    var B = real_array(mats[0].nrows, 2)
    for i in range(mats[0].nrows):
        fset(B, i, 0, 1.0)
        fset(B, i, 1, Float64(i))
    var X = C.solve(B)
    var b1 = real_vector(mats[0].nrows)
    for i in range(mats[0].nrows):
        b1.unsafe_set(i, Float64(i))
    var x1 = C.solve(b1)
    for i in range(mats[0].nrows):
        assert_equal(fget(X, i, 1), x1.unsafe_get(i))


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
    with assert_raises(contains="B has 3 rows, not 2"):
        _ = F.solve(real_array(3, 2))
    # the empty matrix factors, and solves the empty system
    var E = SparseLDLT(CSRMatrix(0, 0))
    assert_equal(E.solve(real_vector(0)).size, 0)
    assert_equal(E.solve(real_array(0, 3)).size, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
