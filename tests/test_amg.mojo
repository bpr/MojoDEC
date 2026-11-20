from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    NullSpace,
    Relaxation,
    SimplicialComplex,
    SmoothedAggregation,
    cg,
    jacobi,
    kuhn_grid,
    whitney_innerproduct,
)
from molina.sparse import CSRMatrix
from mojodec.core.utils import RealArray, dot, probe_vector, real_vector
from molina.amg import _aggregates, _smoothed_prolongation


def _interior(sc: SimplicialComplex, k: Int) raises -> List[Int]:
    var boundary = sc.boundary_cells()[k].copy()
    var out = List[Int]()
    for i in range(sc[k].num_simplices):
        if i not in boundary:
            out.append(i)
    return out^


def _dirichlet_laplacian(dims: List[Int]) raises -> CSRMatrix:
    var sc = SimplicialComplex(kuhn_grid(dims))
    var iv = _interior(sc, 0)
    return (sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d).select(
        iv, iv
    )


def _random_graph() raises -> CSRMatrix:
    var n = 300
    var r = List[Int]()
    var c = List[Int]()
    var v = List[Float64]()
    var s: UInt64 = 7
    for i in range(n):
        r.append(i)
        c.append(i)
        v.append(1.0 if i % 37 == 0 else 3.0 + Float64(i % 5))
    for _ in range(700):
        s = s * 6364136223846793005 + 1442695040888963407
        var i = Int((s >> 33) % UInt64(n))
        s = s * 6364136223846793005 + 1442695040888963407
        var j = Int((s >> 33) % UInt64(n))
        if i != j and i % 37 != 0 and j % 37 != 0:
            var w = -0.1 - Float64((s >> 40) % 10) / 10
            r.append(i)
            c.append(j)
            v.append(w)
            r.append(j)
            c.append(i)
            v.append(w)
            r.append(i)
            c.append(i)
            v.append(-w)
            r.append(j)
            c.append(j)
            v.append(-w)
    return CSRMatrix.from_coo(n, n, r, c, v)


def _words(path: String) raises -> List[List[String]]:
    var out = List[List[String]]()
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            var row = List[String]()
            for w in line.split():
                row.append(String(w))
            out.append(row^)
    return out^


def test_matches_amgcl() raises:
    # tests/data/amgcl: amgcl's aggregates and smoothed P (omega 2/3) for
    # the same matrices; the port agrees bit for bit
    var g8 = SimplicialComplex(kuhn_grid([8, 8]))
    var ie = _interior(g8, 1)
    var mats = List[CSRMatrix]()
    mats.append(_dirichlet_laplacian([20, 20]))
    mats.append(
        (
            g8[1].d.transpose() @ whitney_innerproduct(g8, 2) @ g8[1].d
            + whitney_innerproduct(g8, 1)
        ).select(ie, ie)
    )
    mats.append(_random_graph())
    var names: List[String] = ["grid2", "edges2", "random"]
    for m in range(3):
        ref A = mats[m]
        var t = _words("tests/data/amgcl/" + names[m] + ".out")
        var agg = _aggregates(A, A.diagonal(), 0.08)
        assert_equal(agg.count, atol(t[0][0]))
        for i in range(A.nrows):
            assert_equal(agg.id[i], max(atol(t[1][i]), -1), names[m])
        var P = _smoothed_prolongation(A, agg, 2.0 / 3.0)
        assert_equal(P.nrows, atol(t[2][0]))
        assert_equal(P.ncols, atol(t[2][1]))
        # amgcl leaves each row's columns unsorted; compare per row
        for i in range(P.nrows):
            var lo = atol(t[3][i])
            var hi = atol(t[3][i + 1])
            assert_equal(P.indptr[i + 1] - P.indptr[i], hi - lo)
            for k in range(lo, hi):
                var col = atol(t[4][k])
                var want = atof(t[5][k])
                var found = False
                for q in range(P.indptr[i], P.indptr[i + 1]):
                    if P.indices[q] == col:
                        assert_equal(
                            P.data[q], want, String(names[m], " row ", i)
                        )
                        found = True
                assert_true(found)


def test_iterations_do_not_grow_with_the_mesh() raises:
    # the Dirichlet P1 Laplacian: Jacobi-CG's iterations double with each
    # refinement, AMG-CG's stay put, with either smoother
    for relax in [Relaxation.CHEBYSHEV, Relaxation.GAUSS_SEIDEL]:
        var counts = List[Int]()
        for cells in [16, 32, 64]:
            var K = _dirichlet_laplacian([cells, cells])
            var amg = SmoothedAggregation(K, relaxation=relax, coarse_size=50)
            assert_true(amg.num_levels() >= 2)
            assert_true(amg.operator_complexity() < 1.6)
            var b = probe_vector(K.nrows, 5)
            var res = cg(K, b, M=amg, rtol=1e-8)
            assert_true(res.converged())
            counts.append(res.iterations)
        assert_true(
            counts[2] <= counts[0] + 2,
            String(relax, ": ", counts[0], " ", counts[2]),
        )
        assert_true(counts[2] <= 14)
    var K3 = _dirichlet_laplacian([10, 10, 10])
    var b3 = probe_vector(K3.nrows, 2)
    var res3 = cg(K3, b3, M=SmoothedAggregation(K3, coarse_size=50), rtol=1e-8)
    assert_true(res3.iterations <= 12)
    var jac = cg(K3, b3, M=jacobi(K3), rtol=1e-8)
    assert_true(res3.iterations < jac.iterations)


def test_cycle_is_symmetric_positive_definite() raises:
    # x^T B y = y^T B x and x^T B x > 0: what cg needs of a preconditioner
    var K = _dirichlet_laplacian([30, 30])
    for relax in [Relaxation.CHEBYSHEV, Relaxation.GAUSS_SEIDEL]:
        var amg = SmoothedAggregation(K, relaxation=relax, coarse_size=40)
        var x = probe_vector(K.nrows, 1)
        var y = probe_vector(K.nrows, 2)
        var Bx = real_vector(K.nrows)
        var By = real_vector(K.nrows)
        amg.apply_into(x, Bx)
        amg.apply_into(y, By)
        var a = dot(y, Bx)
        var c = dot(x, By)
        assert_almost_equal(a, c, atol=1e-12 * abs(a), rtol=0)
        assert_true(dot(x, Bx) > 0)


def test_neumann_laplacian_with_its_kernel() raises:
    # singular: the coarse factor is regularized, and cg removes the
    # constants it adds
    var sc = SimplicialComplex(kuhn_grid([40, 40]))
    var K = sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d
    var amg = SmoothedAggregation(K, coarse_size=30)
    var b = probe_vector(K.nrows, 4)
    var res = cg(K, b, M=amg, rtol=1e-8, nullspace=NullSpace.constants(K.nrows))
    assert_true(res.converged())
    assert_true(res.iterations <= 15)


def test_small_matrices_are_factored() raises:
    # at or below coarse_size there is one level, an exact solve
    var K = _dirichlet_laplacian([8, 8])
    var amg = SmoothedAggregation(K)
    assert_equal(amg.num_levels(), 1)
    assert_equal(amg.level_sizes()[0], K.nrows)
    var b = probe_vector(K.nrows, 3)
    var res = cg(K, b, M=amg, rtol=1e-12)
    assert_true(res.iterations <= 2)


def test_errors() raises:
    with assert_raises(contains="the matrix is 2 x 3"):
        _ = SmoothedAggregation(CSRMatrix(2, 3))
    var K = _dirichlet_laplacian([10, 10])
    with assert_raises(contains="strength must be in (0, 1)"):
        _ = SmoothedAggregation(K, strength=1.5)
    with assert_raises(contains="must be positive"):
        _ = SmoothedAggregation(K, degree=0)
    var negative = K * -1.0
    with assert_raises(contains="diagonal entry 0 is"):
        _ = SmoothedAggregation(negative, coarse_size=10)
    var amg = SmoothedAggregation(K, coarse_size=10)
    var y = real_vector(3)
    with assert_raises(contains="vectors of length 3 and 3"):
        amg.apply_into(real_vector(3), y)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
