from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    Primality,
    CSRMatrix,
    DUAL,
    DiagonalOp,
    MatrixOp,
    PRIMAL,
    SimplicialComplex,
    TypedCochain,
    TypedComplex,
    TypedComplexVisitor,
    TypedSubset,
    cg,
    d,
    delta,
    eigh_generalized,
    jacobi,
    laplace_beltrami,
    laplace_derham,
    lsqr,
    minres,
    simplicial_grid_2d,
    spsolve,
    star,
    visit_typed,
    whitney_innerproduct,
)
from mojodec.core.utils import fget, fset, real_vector
from mojodec.routines.io.meshio import read_mesh


def _grid(n: Int) raises -> TypedComplex[2]:
    """A grid with its interior vertices moved a little, so no dual edge has
    zero length (as the diagonals of the right triangles would)."""
    var m = simplicial_grid_2d(n)
    var V = m.vertices.copy()
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
    return TypedComplex[2](V, m.elements)


def _torus() raises -> TypedComplex[3]:
    return TypedComplex[3](
        SimplicialComplex(read_mesh("tests/meshes/torus_tet.xml"))
    )


def _diff(A: CSRMatrix, B: CSRMatrix) raises -> Float64:
    return (A - B).max_abs()


def _check_level[n: Int, k: Int, primal: Bool](tc: TypedComplex[n]) raises:
    """Typed operators on k-cochains agree with the untyped ones."""
    var basis = tc.sc.get_cochain_basis(k, Primality(primal))
    var S = tc.star[k, primal]()
    assert_equal(_diff(S.diag.to_csr(), star(basis).v), 0.0)
    comptime if k < n:
        assert_equal(_diff(tc.d[k, primal]().mat, d(basis).v), 0.0)
    comptime if k > 0:
        assert_equal(_diff(tc.delta[k, primal]().mat, delta(basis).v), 0.0)
    comptime if k < n:
        var LB = tc.laplace_beltrami[k, primal]()
        var LBref = laplace_beltrami(basis).v.copy()
        assert_true(_diff(LB.mat, LBref) <= 1e-9 * LBref.max_abs())
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
        assert_almost_equal(
            Lc[i], Lc_ref[i], atol=1e-13 * Lref.max_abs(), rtol=0
        )


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
        assert_almost_equal(I.diag[i], 1.0, atol=1e-15, rtol=0)
    # the star on dual 1-cochains is the inverse up to (-1)^(k(n-k)) = -1
    var Sd = tc.star[1, DUAL]()
    for i in range(len(Sd.diag)):
        assert_almost_equal(
            Sd.diag[i], -S.inverse().diag[i], atol=1e-13, rtol=0
        )
    # inner product and norm of the primal star
    var c = tc.cochain[1, PRIMAL]()
    for i in range(len(c)):
        c[i] = 1.0
    var area: Float64 = 0
    for i in range(len(S.diag)):
        area += S.diag[i]
    assert_almost_equal(S.inner_product(c, c), area, atol=1e-13, rtol=0)
    assert_almost_equal(S.norm(c), sqrt(area), atol=1e-14, rtol=0)
    # arithmetic
    var c2 = 2.0 * c - c + (-c)
    assert_equal(c2[3], 0.0)
    var M = tc.d[0, PRIMAL]() * 3.0 - tc.d[0, PRIMAL]()
    assert_equal(_diff(M.mat, tc.d[0, PRIMAL]().mat * 2.0), 0.0)


def test_dirichlet_poisson() raises:
    var tc = _grid(8)
    var fixed = tc.boundary_cells[0]()
    var A = tc.laplacian[0, PRIMAL]().exclude_subset(fixed) + tc.identity[
        0, PRIMAL
    ]().restrict_to_subset(fixed)
    var g = tc.cochain[0, PRIMAL]()
    ref V = tc.sc.vertices()
    for i in range(len(g)):
        g[i] = 1.0 + fget(V, i, 0) - 2.0 * fget(V, i, 1)
    var b = tc.cochain[0, PRIMAL]()
    b.overwrite(fixed, g)
    var u: TypedCochain[2, 0, PRIMAL] = spsolve(A, b)
    for i in range(len(u)):
        assert_almost_equal(u[i], g[i], atol=1e-12, rtol=0)


def _stiffness(tc: TypedComplex[2]) raises -> MatrixOp[2, 0, PRIMAL, 2, DUAL]:
    """`d0^T *1 d0`, from primal 0-cochains to dual 2-cochains."""
    var d0 = tc.d[0, PRIMAL]()
    return d0.T() @ tc.star[1, PRIMAL]() @ d0


def test_transpose() raises:
    var tc = _grid(4)
    # d[k, PRIMAL]^T = (-1)^(k+1) d[n-k-1, DUAL], the dual kinds swapped
    var d0t: MatrixOp[2, 1, DUAL, 2, DUAL] = tc.d[0, PRIMAL]().T()
    assert_equal(_diff(d0t.mat, -tc.d[1, DUAL]().mat), 0.0)
    var d1t: MatrixOp[2, 0, DUAL, 1, DUAL] = tc.d[1, PRIMAL]().transpose()
    assert_equal(_diff(d1t.mat, tc.d[0, DUAL]().mat), 0.0)
    var t = _torus()
    var e1t: MatrixOp[3, 1, DUAL, 2, DUAL] = t.d[1, PRIMAL]().T()
    assert_equal(_diff(e1t.mat, t.d[1, DUAL]().mat), 0.0)
    # transposing twice gives back the operator and its kind
    var back: MatrixOp[2, 0, PRIMAL, 1, PRIMAL] = d0t.T()
    assert_equal(_diff(back.mat, tc.d[0, PRIMAL]().mat), 0.0)
    # a Hodge star is its own transpose, kind included
    var S = tc.star[1, PRIMAL]()
    var St: DiagonalOp[2, 1, PRIMAL, 1, DUAL] = S.T()
    for i in range(len(S.diag)):
        assert_equal(St.diag[i], S.diag[i])
    # the stiffness matrix is symmetric, and keeps its complex
    var A = _stiffness(tc)
    assert_equal(_diff(A.mat, A.T().mat), 0.0)
    with assert_raises(contains="different complexes"):
        _ = A @ _grid(4).cochain[0, PRIMAL]()


def test_dual_diagonal() raises:
    var tc = _grid(4)
    var S = tc.star[1, PRIMAL]()
    var w = List[Float64]()
    for i in range(len(S.diag)):
        w.append(S.diag[i])
    var W: DiagonalOp[2, 1, PRIMAL, 1, DUAL] = tc.dual_diagonal[1, PRIMAL](w^)
    for i in range(len(W.diag)):
        assert_equal(W.diag[i], S.diag[i])
    # Dirichlet rows on the boundary vertices of d0^T *1 d0, whose
    # equations are dual 2-cochains
    var fixed = tc.boundary_cells[0]()
    var N0 = tc.num_cells[0, PRIMAL]()
    var A = _stiffness(tc).exclude_subset(fixed.dual()) + tc.dual_diagonal[
        0, PRIMAL
    ](List[Float64](length=N0, fill=1.0)).restrict_to_subset(fixed.dual())
    var g = tc.cochain[0, PRIMAL]()
    ref V = tc.sc.vertices()
    for i in range(N0):
        g[i] = 2.0 * fget(V, i, 0) + fget(V, i, 1)
    var b = tc.cochain[2, DUAL]()
    for i in fixed.indices():
        b[i] = g[i]
    var u = spsolve(A, b)  # a linear function is discretely harmonic
    for i in range(N0):
        assert_almost_equal(u[i], g[i], atol=1e-13, rtol=0)
    with assert_raises(contains="wrong number"):
        _ = tc.dual_diagonal[0, DUAL](List[Float64]())


def test_solvers() raises:
    var tc = _grid(6)
    var A = _stiffness(tc)
    # a consistent right-hand side: the image of a known potential
    var f = tc.cochain[0, PRIMAL]()
    for i in range(len(f)):
        f[i] = Float64((i * 5) % 7) - 3.0
    var b = A @ f  # TypedCochain[2, 2, DUAL]
    var res = cg(A, b, rtol=1e-12)
    assert_true(res.converged())
    var u: TypedCochain[2, 0, PRIMAL] = res^.take_x()
    var r = A @ u - b
    for i in range(len(r)):
        # cg's rtol=1e-12 relative to |b|
        assert_almost_equal(r[i], 0.0, atol=1e-10, rtol=0)
    # the same iterates as the untyped solver
    var raw = cg(A.mat, b.v, rtol=1e-12).take_x()
    for i in range(len(u)):
        assert_equal(u[i], raw.unsafe_get(i))
    # least squares: a potential of the gradient of f
    var w = tc.d[0, PRIMAL]() @ f
    var ls = lsqr(tc.d[0, PRIMAL](), w, atol=1e-12, btol=1e-12)
    var g: TypedCochain[2, 0, PRIMAL] = ls^.take_x()
    var dg = tc.d[0, PRIMAL]() @ g - w
    for i in range(len(dg)):
        # lsqr's atol = btol = 1e-12, relative to |A| |x| and |b|
        assert_almost_equal(dg[i], 0.0, atol=2e-10, rtol=0)
    # operands from another complex are refused
    var other = _grid(6)
    with assert_raises(contains="different complexes"):
        _ = cg(A, other.cochain[2, DUAL]())
    with assert_raises(contains="different complexes"):
        _ = spsolve(A, other.cochain[2, DUAL]())
    # cg(A, f) does not compile: f is primal 0, A gives dual 2
    # the preconditioner maps back: dual 2 -> primal 0
    var J: DiagonalOp[2, 2, DUAL, 0, PRIMAL] = jacobi(A)
    var pre = cg(A, b, M=J, rtol=1e-12)
    assert_true(pre.converged())
    var raw_pre = cg(A.mat, b.v, M=J.diag, rtol=1e-12)
    assert_equal(pre.iterations, raw_pre.iterations)
    for i in range(len(u)):
        assert_equal(pre.x[i], raw_pre.x.unsafe_get(i))
    # minres on the same singular system, plain and preconditioned
    var mr = minres(A, b, rtol=1e-12)
    assert_true(mr.converged())
    var mr_raw = minres(A.mat, b.v, rtol=1e-12)
    for i in range(len(u)):
        assert_equal(mr.x[i], mr_raw.x.unsafe_get(i))
    var mp = minres(A, b, M=J, rtol=1e-12)
    assert_true(mp.converged())
    var rm = A @ mp.x - b
    for i in range(len(rm)):
        # rtol=1e-12 in the M norm; M ~ 1/4 on this grid, |b| ~ 10
        assert_almost_equal(rm[i], 0.0, atol=1e-10, rtol=0)
    with assert_raises(contains="different complexes"):
        _ = minres(A, other.cochain[2, DUAL]())
    with assert_raises(contains="different complexes"):
        _ = cg(A, b, M=jacobi(_stiffness(other)))


def test_whitney_and_eigenproblems() raises:
    var tc = _grid(4)
    var M1 = whitney_innerproduct[1](tc)  # MatrixOp[2, 1, PRIMAL, 1, DUAL]
    assert_equal(_diff(M1.mat, whitney_innerproduct(tc.sc, 1)), 0.0)
    # the resonant cavity: curl-curl against the Whitney 1-form mass matrix
    var K = tc.d[0, DUAL]() @ whitney_innerproduct[2](tc) @ tc.d[1, PRIMAL]()
    var eig = eigh_generalized(K, M1)
    var ref_ = eigh_generalized(K.mat.todense(), M1.mat.todense())
    assert_equal(len(eig), ref_.values.size)
    for i in range(len(eig)):
        assert_equal(eig.values.unsafe_get(i), ref_.values.unsafe_get(i))
    # each mode solves K x = lambda M x
    var last = len(eig) - 1
    var x: TypedCochain[2, 1, PRIMAL] = eig.mode(last)
    var lhs = K @ x
    var rhs = (M1 @ x) * eig.values.unsafe_get(last)
    var scale = eig.values.unsafe_get(last)
    for i in range(len(lhs)):
        # relative to the largest eigenvalue
        assert_almost_equal(lhs[i], rhs[i], atol=1e-13 * scale, rtol=0)
    # with the diagonal Hodge star as the mass matrix
    var S = tc.star[0, PRIMAL]()
    var eig0 = eigh_generalized(S @ tc.laplacian[0, PRIMAL](), S)
    var ref0 = eigh_generalized(
        (S @ tc.laplacian[0, PRIMAL]()).mat.todense(), S.diag
    )
    for i in range(len(eig0)):
        assert_equal(eig0.values.unsafe_get(i), ref0.values.unsafe_get(i))
    assert_almost_equal(eig0.values.unsafe_get(0), 0.0, atol=1e-12, rtol=0)
    with assert_raises(contains="index out of range"):
        _ = eig0.mode(len(eig0))
    with assert_raises(contains="different complexes"):
        _ = eigh_generalized(K, whitney_innerproduct[1](_grid(4)))


struct _Census(TypedComplexVisitor):
    var dims: List[Int]
    var top: List[Int]

    def __init__(out self):
        self.dims = List[Int]()
        self.top = List[Int]()

    def visit[n: Int](mut self, tc: TypedComplex[n]) raises:
        self.dims.append(n)
        self.top.append(tc.num_cells[n, PRIMAL]())
        # code generic over n: the top-degree Laplacian
        _ = tc.laplacian[n, PRIMAL]()


def test_visit_typed() raises:
    var census = _Census()
    visit_typed(_grid(2).sc, census)
    var mesh = read_mesh("tests/meshes/torus_tet.xml")
    var torus = SimplicialComplex(mesh)
    visit_typed(torus, census)
    assert_equal(census.dims, [2, 3])
    assert_equal(census.top[0], 8)
    assert_equal(census.top[1], torus[3].num_simplices)
    # outside the compiled range
    with assert_raises(contains="outside 1..2"):
        visit_typed[max_n=2](torus, census)
    # a TypedComplex straight from a mesh
    var tc = TypedComplex[3](mesh)
    assert_equal(tc.num_cells[0, PRIMAL](), torus[0].num_simplices)
    with assert_raises():
        _ = TypedComplex[2](mesh)


def test_typed_subset() raises:
    var tc = _grid(4)
    var bd = tc.boundary_cells[1]()
    assert_equal(len(bd), 16)
    # the boundary of the boundary edges is the boundary vertices
    var verts = bd.boundary()
    assert_true(verts.untyped() == tc.boundary_cells[0]().untyped())
    # dual() swaps the kind in the type
    var dual_bd: TypedSubset[2, 1, DUAL] = bd.dual()
    assert_equal(len(dual_bd), 16)
    var back: TypedSubset[2, 1, PRIMAL] = dual_bd.dual()
    assert_true(back.untyped() == bd.untyped())
    # set operations and the cofaces of the boundary vertices
    var some = tc.subset[1, PRIMAL]([0, 1, 2])
    assert_equal(len(bd | some), len(bd) + len(some) - len(bd & some))
    assert_true((bd - bd).is_empty())
    assert_equal(len(bd.complement()), tc.num_cells[1, PRIMAL]() - 16)
    var cof = verts.coboundary()
    assert_true(len(cof) >= len(bd))
    assert_equal(len(TypedSubset[2, 2, DUAL].full(tc.sc)), 25)
    # an untyped subset of the wrong kind is refused at run time
    with assert_raises(contains="other cells"):
        _ = TypedSubset[2, 0, PRIMAL](tc.sc, tc.sc.boundary_cells()[1].copy())
    with assert_raises(contains="other cells"):
        _ = TypedSubset[2, 0, DUAL](tc.sc, tc.sc.boundary_cells()[0].copy())
    # subsets of another complex are refused by cochains and operators
    var other = _grid(4)
    var c = tc.cochain[0, PRIMAL]()
    with assert_raises(contains="another complex"):
        c.overwrite(other.boundary_cells[0](), c.copy())
    with assert_raises(contains="different complexes"):
        _ = tc.laplacian[0, PRIMAL]().exclude_subset(other.boundary_cells[0]())


def test_errors() raises:
    var m = simplicial_grid_2d(2)
    with assert_raises():
        _ = TypedComplex[3](m.vertices, m.elements)
    var tc = TypedComplex[2](m.vertices, m.elements)
    with assert_raises():
        _ = TypedCochain[2, 1, PRIMAL](tc.sc.get_cochain(1, .DUAL))
    var c = TypedCochain[2, 1, PRIMAL](tc.sc.get_cochain(1))
    assert_equal(len(c), tc.num_cells[1, PRIMAL]())
    var u = c.untyped()
    assert_equal(u.k, 1)
    with assert_raises():
        _ = tc.cochain[0, PRIMAL](c.v.copy())
    # subsets must be of the output cells; other kinds do not compile:
    # tc.d[0, PRIMAL]().exclude_subset(tc.boundary_cells[0]())
    _ = tc.d[0, PRIMAL]().exclude_subset(tc.boundary_cells[1]())
    _ = tc.star[1, PRIMAL]().restrict_to_subset(tc.boundary_cells[1]().dual())
    var nogeom = TypedComplex[2](m.vertices, m.elements, compute_geometry=False)
    with assert_raises():
        _ = nogeom.star[0, PRIMAL]()
    # cochains from different complexes do not mix
    with assert_raises():
        _ = c + nogeom.cochain[1, PRIMAL]()


def test_operators_keep_their_complex() raises:
    var m = simplicial_grid_2d(2)
    var a = TypedComplex[2](m.vertices, m.elements)
    var b = TypedComplex[2](m.vertices, m.elements)
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
