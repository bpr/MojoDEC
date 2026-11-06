from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    CSRMatrix,
    DUAL,
    MatrixOp,
    PRIMAL,
    SimplicialComplex,
    TypedCochain,
    TypedComplex,
    d,
    delta,
    laplace_derham,
    simplicial_grid_2d,
    spsolve,
    star,
)
from mojodec.core.utils import fget, fset
from mojodec.routines.io.meshio import read_mesh


def _grid(n: Int) raises -> TypedComplex[2]:
    """A grid with its interior vertices moved a little, so no dual edge has
    zero length (as the diagonals of the right triangles would)."""
    var m = simplicial_grid_2d(n)
    var V = m[0].copy()
    var h = 1.0 / Float64(n)
    for i in range(1, n):
        for j in range(1, n):
            var v = i * (n + 1) + j
            fset(
                V, v, 0, fget(V, v, 0) + 0.15 * h * Float64((v * 7) % 5 - 2) / 2
            )
            fset(
                V, v, 1, fget(V, v, 1) + 0.15 * h * Float64((v * 3) % 5 - 2) / 2
            )
    return TypedComplex[2](V, m[1])


def _torus() raises -> TypedComplex[3]:
    return TypedComplex[3](
        SimplicialComplex(read_mesh("tests/meshes/torus_tet.xml"))
    )


def _diff(A: CSRMatrix, B: CSRMatrix) raises -> Float64:
    return (A - B).max_abs()


def _check_level[n: Int, k: Int, primal: Bool](tc: TypedComplex[n]) raises:
    """Typed operators on k-cochains agree with the untyped ones."""
    var basis = tc.sc.get_cochain_basis(k, primal)
    var S = tc.star[k, primal]()
    assert_equal(_diff(S.diag.to_csr(), star(basis).v), 0.0)
    comptime if k < n:
        assert_equal(_diff(tc.d[k, primal]().mat, d(basis).v), 0.0)
    comptime if k > 0:
        assert_equal(_diff(tc.delta[k, primal]().mat, delta(basis).v), 0.0)
    var L = tc.laplacian[k, primal]()
    var Lref = laplace_derham(basis).v.copy()
    assert_true(_diff(L.mat, Lref) <= 1e-9 * Lref.max_abs())
    # applying the operator agrees with the untyped cochain route
    var c = tc.cochain[k, primal]()
    for i in range(len(c)):
        c[i] = Float64((i * 7) % 11) - 5.0
    var Lc = L @ c
    var Lc_ref = laplace_derham(c.untyped())
    for i in range(len(Lc)):
        assert_almost_equal(Lc[i], Lc_ref[i], atol=1e-9 * Lref.max_abs())


def test_operators_match_untyped() raises:
    var g = _grid(4)
    comptime for k in range(3):
        _check_level[2, k, PRIMAL](g)
        _check_level[2, k, DUAL](g)
    var t = _torus()
    comptime for k in range(4):
        _check_level[3, k, PRIMAL](t)
        _check_level[3, k, DUAL](t)


def test_composition() raises:
    var tc = _grid(4)
    # d d = 0, primal and dual
    assert_equal((tc.d[1, PRIMAL]() @ tc.d[0, PRIMAL]()).mat.max_abs(), 0.0)
    assert_equal((tc.d[1, DUAL]() @ tc.d[0, DUAL]()).mat.max_abs(), 0.0)
    # the four-factor Laplacian equals laplacian[0], up to its sign
    var L = (
        tc.star[2, DUAL]()
        @ tc.d[1, DUAL]()
        @ tc.star[1, PRIMAL]()
        @ tc.d[0, PRIMAL]()
    )
    assert_true(_diff(L.mat, -tc.laplacian[0, PRIMAL]().mat) < 1e-12)
    # diagonal compositions: star then inverse is the identity
    var S = tc.star[1, PRIMAL]()
    var I = S.inverse() @ S
    for i in range(len(I.diag)):
        assert_almost_equal(I.diag[i], 1.0)
    # the star on dual 1-cochains is the inverse up to (-1)^(k(n-k)) = -1
    var Sd = tc.star[1, DUAL]()
    for i in range(len(Sd.diag)):
        assert_almost_equal(Sd.diag[i], -S.inverse().diag[i])
    # inner product and norm of the primal star
    var c = tc.cochain[1, PRIMAL]()
    for i in range(len(c)):
        c[i] = 1.0
    var area: Float64 = 0
    for i in range(len(S.diag)):
        area += S.diag[i]
    assert_almost_equal(S.inner_product(c, c), area)
    assert_almost_equal(S.norm(c), area**0.5)
    # arithmetic
    var c2 = 2.0 * c - c + (-c)
    assert_equal(c2[3], 0.0)
    var M = tc.d[0, PRIMAL]() * 3.0 - tc.d[0, PRIMAL]()
    assert_equal(_diff(M.mat, tc.d[0, PRIMAL]().mat * 2.0), 0.0)


def test_dirichlet_poisson() raises:
    var tc = _grid(8)
    var fixed = tc.boundary_subset[0]()
    var A = tc.laplacian[0, PRIMAL]().exclude_subset(fixed) + tc.identity[
        0, PRIMAL
    ]().restrict_to_subset(fixed)
    var g = tc.cochain[0, PRIMAL]()
    ref V = tc.sc.vertices()
    for i in range(len(g)):
        g[i] = 1.0 + fget(V, i, 0) - 2.0 * fget(V, i, 1)
    var b = tc.cochain[0, PRIMAL]()
    b.overwrite(fixed, g)
    var u = tc.cochain[0, PRIMAL](spsolve(A.mat, b.v))
    for i in range(len(u)):
        assert_almost_equal(u[i], g[i], atol=1e-12)


def test_errors() raises:
    var m = simplicial_grid_2d(2)
    with assert_raises():
        _ = TypedComplex[3](m[0], m[1])
    var tc = TypedComplex[2](m[0], m[1])
    with assert_raises():
        _ = TypedCochain[2, 1, PRIMAL](tc.sc.get_cochain(1, False))
    var c = TypedCochain[2, 1, PRIMAL](tc.sc.get_cochain(1))
    assert_equal(len(c), tc.num_cells[1, PRIMAL]())
    var u = c.untyped()
    assert_equal(u.k, 1)
    with assert_raises():
        _ = tc.cochain[0, PRIMAL](c.v.copy())
    # subsets must be of the output cells
    with assert_raises():
        _ = tc.d[0, PRIMAL]().exclude_subset(tc.boundary_subset[0]())
    _ = tc.d[0, PRIMAL]().exclude_subset(tc.boundary_subset[1]())
    with assert_raises():
        _ = tc.star[1, PRIMAL]().restrict_to_subset(tc.boundary_subset[1]())
    _ = tc.star[1, PRIMAL]().restrict_to_subset(tc.boundary_subset[1]().dual())
    var nogeom = TypedComplex[2](m[0], m[1], compute_geometry=False)
    with assert_raises():
        _ = nogeom.star[0, PRIMAL]()
    # cochains from different complexes do not mix
    with assert_raises():
        _ = c + nogeom.cochain[1, PRIMAL]()


def test_operators_keep_their_complex() raises:
    var m = simplicial_grid_2d(2)
    var a = TypedComplex[2](m[0], m[1])
    var b = TypedComplex[2](m[0], m[1])
    var ua = a.cochain[0, PRIMAL]()
    var ub = b.cochain[0, PRIMAL]()
    var L = a.laplacian[0, PRIMAL]()
    _ = L @ ua
    with assert_raises(contains="different complexes"):
        _ = L @ ub
    with assert_raises(contains="different complexes"):
        _ = a.d[0, PRIMAL]() + b.d[0, PRIMAL]()
    with assert_raises(contains="different complexes"):
        _ = a.star[1, PRIMAL]() @ b.d[0, PRIMAL]()
    with assert_raises(contains="different complexes"):
        _ = a.star[0, PRIMAL]().norm(ub)
    # scaling, subsets, and inverses keep the complex
    with assert_raises(contains="different complexes"):
        _ = (-2.0 * a.star[0, PRIMAL]().inverse()) @ b.cochain[2, DUAL]()
    # an operator wrapped from a bare matrix acts on any complex
    var bare = MatrixOp[2, 0, PRIMAL, 1, PRIMAL](a.sc[0].d.copy())
    _ = bare @ ua
    _ = bare @ ub


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
