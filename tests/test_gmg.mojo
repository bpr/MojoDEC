from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    BlockDiagonal,
    GeometricMultigrid,
    PatchSmoother,
    SimplicialComplex,
    cg,
    jacobi,
    kuhn_grid,
    minres,
    refinement_hierarchy,
    whitney_innerproduct,
)
from molina.sparse import CSRMatrix
from mojodec.core.utils import RealArray, dot, probe_vector, real_vector
from mojodec.routines.refinement import Refinement


def _riesz(sc: SimplicialComplex, k: Int, alpha: Float64) raises -> CSRMatrix:
    """`alpha M_k + d^T M_{k+1} d`: the H(grad), H(curl), H(div) norms."""
    var A = whitney_innerproduct(sc, k) * alpha
    if k < sc.complex_dimension():
        A = A + sc[k].d.transpose() @ whitney_innerproduct(sc, k + 1) @ sc[k].d
    return A^


def _prefix(h: List[Refinement], count: Int) -> List[Refinement]:
    var out = List[Refinement]()
    for l in range(count):
        out.append(Refinement(h[l].coarse, h[l].fine, h[l].parent.copy()))
    return out^


def _interior(sc: SimplicialComplex, k: Int) raises -> List[Int]:
    var boundary = sc.boundary_cells()[k].copy()
    var out = List[Int]()
    for i in range(sc[k].num_simplices):
        if i not in boundary:
            out.append(i)
    return out^


def test_iterations_do_not_grow_with_the_mesh_2d() raises:
    # every degree on refinements of a 2 x 2 Kuhn grid; Jacobi's counts
    # double per refinement for k = 0, 1, the vertex-star cycle's do not
    var h = refinement_hierarchy(SimplicialComplex(kuhn_grid([2, 2])), 4)
    for k in range(3):
        for sm in [PatchSmoother.MULTIPLICATIVE, PatchSmoother.ADDITIVE]:
            var counts = List[Int]()
            for refs in range(2, 5):
                ref sc = h[refs - 1].fine
                var A = _riesz(sc, k, 1.0)
                var mg = GeometricMultigrid(A, _prefix(h, refs), k, smoother=sm)
                assert_equal(mg.num_levels(), refs + 1)
                var b = probe_vector(A.nrows, 3)
                var res = cg(A, b, M=mg, rtol=1e-8)
                assert_true(res.converged())
                counts.append(res.iterations)
            var bound = 12 if sm == PatchSmoother.MULTIPLICATIVE else 20
            for c in counts:
                assert_true(c <= bound, String(k, " ", sm, ": ", c))
            assert_true(counts[2] <= counts[0] + 2)
    # the contrast: Jacobi on H(curl) at the finest size
    ref sc = h[3].fine
    var A1 = _riesz(sc, 1, 1.0)
    var b1 = probe_vector(A1.nrows, 3)
    assert_true(cg(A1, b1, M=jacobi(A1), rtol=1e-8).iterations > 400)


def test_three_dimensions() raises:
    var h = refinement_hierarchy(SimplicialComplex(kuhn_grid([2, 2, 2])), 2)
    for k in range(1, 3):
        ref sc = h[1].fine
        var A = _riesz(sc, k, 1.0)
        var mg = GeometricMultigrid(A, _prefix(h, 2), k)
        var res = cg(A, probe_vector(A.nrows, 4), M=mg, rtol=1e-8)
        assert_true(res.iterations <= 12, String(k, ": ", res.iterations))


def test_dirichlet_and_mass_weights() raises:
    # tangential trace zero (boundary edges dropped on every level), and a
    # small or large mass weight: the stiffness or the mass dominates
    var h = refinement_hierarchy(SimplicialComplex(kuhn_grid([2, 2])), 4)
    ref sc = h[3].fine
    var ie = _interior(sc, 1)
    for alpha in [1e-3, 1.0, 1e3]:
        var A = _riesz(sc, 1, alpha).select(ie, ie)
        var mg = GeometricMultigrid(A, _prefix(h, 4), 1, dirichlet=True)
        assert_equal(mg.level_sizes()[0], len(ie))
        var res = cg(A, probe_vector(A.nrows, 5), M=mg, rtol=1e-8)
        assert_true(res.iterations <= 14, String(alpha, ": ", res.iterations))


def test_galerkin_equals_the_coarse_operator() raises:
    # nested Whitney spaces: P^T A_fine P is the operator assembled on the
    # coarser mesh
    var h = refinement_hierarchy(SimplicialComplex(kuhn_grid([3, 2])), 2)
    for k in range(3):
        var A = _riesz(h[1].fine, k, 2.0)
        var mg = GeometricMultigrid(A, _prefix(h, 2), k)
        var want = _riesz(h[0].coarse, k, 2.0).todense()
        var got = mg.coarse.todense()
        var big: Float64 = 0
        for i in range(want.size):
            big = max(big, abs(want.unsafe_get(i)))
        for i in range(want.size):
            assert_almost_equal(
                got.unsafe_get(i), want.unsafe_get(i), atol=1e-12 * big, rtol=0
            )


def test_cycle_is_symmetric_positive_definite() raises:
    var h = refinement_hierarchy(SimplicialComplex(kuhn_grid([2, 2])), 3)
    ref sc = h[2].fine
    for k in range(3):
        var A = _riesz(sc, k, 0.5)
        for sm in [PatchSmoother.MULTIPLICATIVE, PatchSmoother.ADDITIVE]:
            var mg = GeometricMultigrid(A, _prefix(h, 3), k, smoother=sm)
            var x = probe_vector(A.nrows, 1)
            var y = probe_vector(A.nrows, 2)
            var Bx = real_vector(A.nrows)
            var By = real_vector(A.nrows)
            mg.apply_into(x, Bx)
            mg.apply_into(y, By)
            var a = dot(y, Bx)
            var c = dot(x, By)
            assert_almost_equal(a, c, atol=1e-12 * abs(a), rtol=0)
            assert_true(dot(x, Bx) > 0)


def test_mixed_hodge_laplacian_block_preconditioner() raises:
    # Arnold-Falk-Winther with multigrid blocks: MINRES on the mixed
    # 1-form problem stays bounded (measured 26-30 up to 66k unknowns),
    # where Jacobi's count doubles
    var h = refinement_hierarchy(SimplicialComplex(kuhn_grid([2, 2])), 4)
    var its = List[Int]()
    for refs in [2, 4]:
        ref sc = h[refs - 1].fine
        var M0 = whitney_innerproduct(sc, 0)
        var M1 = whitney_innerproduct(sc, 1)
        var B = M1 @ sc[0].d
        var A = sc[1].d.transpose() @ whitney_innerproduct(sc, 2) @ sc[1].d
        var blocks = List[List[Optional[CSRMatrix]]]()
        blocks.append([Optional(M0 * -1.0), Optional(B.transpose())])
        blocks.append([Optional(B.copy()), Optional(A.copy())])
        var K = CSRMatrix.bmat(blocks)
        var b = real_vector(K.nrows)
        for i in range(K.nrows):
            b.unsafe_set(i, Float64((i * 37) % 11) - 5)
        var P = BlockDiagonal(
            GeometricMultigrid(
                M0 + sc[0].d.transpose() @ B, _prefix(h, refs), 0
            ),
            GeometricMultigrid(M1 + A, _prefix(h, refs), 1),
        )
        var res = minres(K, b, M=P, rtol=1e-10)
        assert_true(res.converged())
        its.append(res.iterations)
    assert_true(its[1] <= 35, String(its[1]))
    assert_true(its[1] <= its[0] + 6, String(its[0], " ", its[1]))


def test_errors() raises:
    var h = refinement_hierarchy(SimplicialComplex(kuhn_grid([2, 2])), 2)
    ref sc = h[1].fine
    var A = _riesz(sc, 1, 1.0)
    with assert_raises(contains="the hierarchy is empty"):
        _ = GeometricMultigrid(A, List[Refinement](), 1)
    with assert_raises(contains="k = 3 for dimension 2"):
        _ = GeometricMultigrid(A, _prefix(h, 2), 3)
    with assert_raises(contains="free 0-simplices"):
        _ = GeometricMultigrid(A, _prefix(h, 2), 0)
    var other = refinement_hierarchy(SimplicialComplex(kuhn_grid([2, 2])), 1)
    var broken = _prefix(h, 1)
    broken.append(
        Refinement(other[0].coarse, other[0].fine, other[0].parent.copy())
    )
    with assert_raises(contains="does not refine refinement 0"):
        _ = GeometricMultigrid(A, broken, 1)
    with assert_raises(contains="not positive definite on the star"):
        _ = GeometricMultigrid(A * -1.0, _prefix(h, 2), 1)
    with assert_raises(contains="refinements must be positive"):
        _ = refinement_hierarchy(sc, 0)
    var mg = GeometricMultigrid(A, _prefix(h, 2), 1)
    var y = real_vector(2)
    with assert_raises(contains="vectors of length 2 and 2"):
        mg.apply_into(real_vector(2), y)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
