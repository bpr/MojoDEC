"""The typed factorizations; the solvers themselves are tested in molina."""

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
from molina.sparse import CSRMatrix
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
from molina.supernodal import etree_postorder


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
