"""Convergence of Whitney-form approximations on Kuhn grids of the unit
cube in 1, 2, and 3 dimensions, as the cells halve: the L2 projection and
the interpolant W R of a smooth form of every degree, and the Galerkin
solution of Poisson's equation. Rates come from the last two meshes.
Then Poisson's equation on nested meshes refined from an irregular one,
with the error estimated from two levels and the prolongation, and the
mixed formulation of the Hodge Laplacian on 1- and 2-forms.
"""

from std.math import cos, log2, pi, sin, sqrt
from std.testing import TestSuite, assert_true

from mojodec import (
    BlockDiagonal,
    DenseCholesky,
    DiagonalOperator,
    SimplicialComplex,
    cg,
    derham_map,
    form_index_sets,
    jacobi,
    kuhn_grid,
    minres,
    l2_error,
    refine,
    l2_projection,
    whitney_innerproduct,
    whitney_load,
)
from mojodec.core.cochain import Cochain
from mojodec.core.primality import Primality
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import dot, fget, fset, real_vector, rows_of


def _g(x: List[Float64], I: Int) -> Float64:
    """A smooth, non-polynomial component, different for each index I."""
    var c: List[Float64] = [1.3, -0.7, 0.4]
    var s = 0.5 * Float64(I)
    for a in range(len(x)):
        s += c[a] * x[a]
    return sin(2 * s) + 0.5 * cos(3 * x[0])


def _cells(n: Int, j: Int) -> Int:
    """Cells per axis of the j-th mesh: 8.. in 1D, 4.. in 2D, 2.. in 3D."""
    return (8 if n == 1 else (4 if n == 2 else 2)) << j


def test_projection_and_interpolation() raises:
    # Whitney 0-forms are complete linear polynomials: second order in L2.
    # Higher degrees contain only the constants completely: first order.
    # The projection is the best approximation, so never worse than W R.
    for n in range(1, 4):
        for k in range(n + 1):
            var ncomp = len(form_index_sets(n, k))

            def om(x: List[Float64]) raises {ncomp} -> List[Float64]:
                var out = List[Float64]()
                for I in range(ncomp):
                    out.append(_g(x, I))
                return out^

            var proj = List[Float64]()
            var interp = List[Float64]()
            for j in range(3):
                var sc = SimplicialComplex(
                    kuhn_grid(List[Int](length=n, fill=_cells(n, j)))
                )
                var p = l2_error(l2_projection(sc, k, 5, om), 7, om)
                var i = l2_error(derham_map(sc, k, 7, om), 7, om)
                proj.append(p)
                interp.append(i)
                # equal for k = n (both are the elements' means); the load
                # vector's quadrature error (degree 5) is ~1e-11 relative
                assert_true(
                    p <= i * (1 + 1e-9),
                    String("n=", n, " k=", k, ": projection ", p, " > ", i),
                )
            var rate_p = log2(proj[1] / proj[2])
            var rate_i = log2(interp[1] / interp[2])
            # measured: 2.0 for k = 0; 0.90 to 1.0 otherwise
            var want = 1.9 if k == 0 else 0.85
            assert_true(
                rate_p >= want and rate_i >= want,
                String("n=", n, " k=", k, ": rates ", rate_p, ", ", rate_i),
            )


def test_poisson_galerkin() raises:
    # -Laplace u = f on the unit cube, u = 0 on the boundary, with
    # u = prod sin(pi x_a): stiffness d0^T W1 d0 and load int f lambda_v.
    # Second order, within a small factor of the best approximation.
    for n in range(1, 4):
        var dim = n

        def u(x: List[Float64]) raises -> List[Float64]:
            var p: Float64 = 1
            for a in range(len(x)):
                p *= sin(pi * x[a])
            return [p]

        def f(x: List[Float64]) raises {dim} -> List[Float64]:
            var p: Float64 = 1
            for a in range(len(x)):
                p *= sin(pi * x[a])
            return [Float64(dim) * pi * pi * p]

        var err = List[Float64]()
        for j in range(3):
            var sc = SimplicialComplex(
                kuhn_grid(List[Int](length=n, fill=_cells(n, j)))
            )
            var K = sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d
            var b = whitney_load(sc, 0, 5, f)
            var boundary = sc.boundary_cells()[0].copy()
            var free = List[Int]()
            for v in range(sc[0].num_simplices):
                if v not in boundary:
                    free.append(v)
            var bf = real_vector(len(free))
            for i in range(len(free)):
                bf.unsafe_set(i, b.unsafe_get(free[i]))
            var Kf = K.select(free, free)
            var res = cg(Kf, bf, M=jacobi(Kf), rtol=1e-13)
            assert_true(res.converged())
            var uh = sc.get_cochain(0)
            for i in range(len(free)):
                uh.v.unsafe_set(free[i], res.x.unsafe_get(i))
            var e = l2_error(uh, 7, u)
            var best = l2_error(l2_projection(sc, 0, 5, u), 7, u)
            # measured 2.4 to 3.4
            assert_true(
                e <= 5 * best,
                String("n=", n, ": Galerkin ", e, " vs best ", best),
            )
            err.append(e)
        # measured 1.83 (3D) to 2.0 (1D)
        assert_true(
            log2(err[1] / err[2]) >= 1.75,
            String("n=", n, ": rate ", log2(err[1] / err[2])),
        )


def _jittered(n: Int, cells: Int) raises -> SimplicialComplex:
    """A Kuhn grid of the unit cube with its interior vertices moved by up
    to 0.2 cells in each direction, deterministically."""
    var m = kuhn_grid(List[Int](length=n, fill=cells))
    var sc = SimplicialComplex(m)
    var boundary = sc.boundary_cells()[0].copy()
    var h = 1.0 / Float64(cells)
    for v in range(rows_of(m.vertices)):
        if v in boundary:
            continue
        for a in range(n):
            var t = Float64((v * 7 + a * 13) % 11) / 10.0 - 0.5  # -0.5..0.5
            fset(m.vertices, v, a, fget(m.vertices, v, a) + 0.4 * t * h)
    return SimplicialComplex(m)


def _solve_poisson[
    F: def(List[Float64]) raises -> List[Float64]
](sc: SimplicialComplex, f: F) raises -> Cochain:
    """The Whitney Galerkin solution of -Laplace u = f, u = 0 on the
    boundary."""
    var K = sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d
    var b = whitney_load(sc, 0, 5, f)
    var boundary = sc.boundary_cells()[0].copy()
    var free = List[Int]()
    for v in range(sc[0].num_simplices):
        if v not in boundary:
            free.append(v)
    var bf = real_vector(len(free))
    for i in range(len(free)):
        bf.unsafe_set(i, b.unsafe_get(free[i]))
    var Kf = K.select(free, free)
    var res = cg(Kf, bf, M=jacobi(Kf), rtol=1e-13)
    assert_true(res.converged())
    var uh = sc.get_cochain(0)
    for i in range(len(free)):
        uh.v.unsafe_set(free[i], res.x.unsafe_get(i))
    return uh^


def test_poisson_on_nested_meshes() raises:
    # refining an irregular mesh: second order, and the difference between
    # two levels, ||u_h - P u_2h||, estimates the coarse error without u
    for n in range(2, 4):
        var dim = n

        def u(x: List[Float64]) raises -> List[Float64]:
            var p: Float64 = 1
            for a in range(len(x)):
                p *= sin(pi * x[a])
            return [p]

        def f(x: List[Float64]) raises {dim} -> List[Float64]:
            var p: Float64 = 1
            for a in range(len(x)):
                p *= sin(pi * x[a])
            return [Float64(dim) * pi * pi * p]

        var sc = _jittered(n, 3 if n == 2 else 2)
        var prev = _solve_poisson(sc, f)
        var err = List[Float64]()
        err.append(l2_error(prev, 7, u))
        for _ in range(3 if n == 2 else 2):
            var r = refine(sc)
            var uh = _solve_poisson(r.fine, f)
            var e = l2_error(uh, 7, u)
            var diff = uh.v - r.prolongation(0) @ prev.v
            var M0 = whitney_innerproduct(r.fine, 0)
            var est = sqrt(dot(diff, M0 @ diff))
            # second order makes it (1 - 1/4) of the coarse error: measured
            # 0.67 to 0.77
            var ratio = est / err[len(err) - 1]
            assert_true(
                0.6 <= ratio and ratio <= 0.85,
                String("n=", n, ": estimate / error = ", ratio),
            )
            err.append(e)
            prev = uh^
            sc = r.fine
        # measured 1.99 (2D, three refinements) and 1.83 (3D, two)
        var rate = log2(err[len(err) - 2] / err[len(err) - 1])
        assert_true(
            rate >= (1.9 if n == 2 else 1.75), String("n=", n, ": rate ", rate)
        )


def _range(lo: Int, hi: Int) -> List[Int]:
    var out = List[Int](capacity=hi - lo)
    for i in range(lo, hi):
        out.append(i)
    return out^


def _mixed_hodge[
    F: def(List[Float64]) raises -> List[Float64]
](sc: SimplicialComplex, k: Int, f: F) raises -> Tuple[Cochain, Cochain]:
    """The mixed Hodge Laplacian with natural boundary conditions: sigma
    a (k-1)-form, u a k-form, `<sigma, tau> - <u, d tau> = 0` and
    `<d sigma, v> + <d u, d v> = <f, v>`. In matrices, with
    `B = M_k d_(k-1)` and `A = d_k^T M_(k+1) d_k`, the symmetric indefinite
    `[[-M_(k-1), B^T], [B, A]]`, solved by MINRES with the block-diagonal
    preconditioner `diag(M_(k-1))^-1`, `diag(A + B diag(M_(k-1))^-1 B^T)^-1`
    (A is zero for k = n, so plain Jacobi would divide by zero)."""
    var n = sc.complex_dimension()
    var Ms = whitney_innerproduct(sc, k - 1)
    var B = whitney_innerproduct(sc, k) @ sc[k - 1].d
    var blocks = List[List[Optional[CSRMatrix]]]()
    blocks.append([Optional(Ms * -1.0), Optional(B.transpose())])
    if k < n:
        var A = sc[k].d.transpose() @ whitney_innerproduct(sc, k + 1) @ sc[k].d
        blocks.append([Optional(B.copy()), Optional(A^)])
    else:
        blocks.append([Optional(B.copy()), None])
    var K = CSRMatrix.bmat(blocks)
    var b = whitney_load(sc, k, 5, f)
    var ns = sc[k - 1].num_simplices
    var nu = sc[k].num_simplices
    var dm = Ms.diagonal()
    var inv = List[Float64](capacity=ns)
    for i in range(ns):
        inv.append(1 / dm.unsafe_get(i))
    var schur = B @ DiagonalOperator(inv.copy()).to_csr() @ B.transpose()
    if k < n:
        schur = schur + K.select(_range(ns, ns + nu), _range(ns, ns + nu))
    var ds = schur.diagonal()
    for i in range(nu):
        inv.append(1 / ds.unsafe_get(i))
    var rhs = real_vector(ns + nu)
    for i in range(nu):
        rhs.unsafe_set(ns + i, b.unsafe_get(i))
    var res = minres(
        K, rhs, M=DiagonalOperator(inv^), rtol=1e-12, maxiter=100000
    )
    assert_true(res.converged())
    var x = res^.take_x()
    var sigma = sc.get_cochain(k - 1)
    var u = sc.get_cochain(k)
    for i in range(ns):
        sigma.v.unsafe_set(i, x.unsafe_get(i))
    for i in range(nu):
        u.v.unsafe_set(i, x.unsafe_get(ns + i))
    return (sigma^, u^)


def _check_mixed[
    F: def(List[Float64]) raises -> List[Float64],
    S: def(List[Float64]) raises -> List[Float64],
    U: def(List[Float64]) raises -> List[Float64],
    D: def(List[Float64]) raises -> List[Float64],
](n: Int, k: Int, f: F, sigma: S, u: U, du: D) raises:
    """Errors of sigma, u, and d u over three Kuhn grids, halving the
    cells; lowest-order Whitney forms give first order for each (measured
    0.94 to 1.0; sigma, piecewise linear for k = 1, reaches 1.7 to 1.95)."""
    var err = List[List[Float64]]()
    for j in range(3):
        var sc = SimplicialComplex(
            kuhn_grid(List[Int](length=n, fill=(4 if n == 2 else 2) << j))
        )
        var sol = _mixed_hodge(sc, k, f)
        var e: List[Float64] = [
            l2_error(sol[0], 7, sigma),
            l2_error(sol[1], 7, u),
        ]
        if k < n:
            var duh = Cochain(sc, k + 1, Primality.PRIMAL, sc[k].d @ sol[1].v)
            e.append(l2_error(duh, 7, du))
        err.append(e^)
    for q in range(len(err[0])):
        var rate = log2(err[1][q] / err[2][q])
        assert_true(
            rate >= 0.85,
            String("n=", n, " k=", k, ": rate ", rate, " of quantity ", q),
        )


def test_mixed_hodge_laplacian() raises:
    # eigenforms of the Hodge Laplacian on the unit square and cube with
    # natural boundary conditions, so f = lambda u and sigma = delta u
    def p2(x: List[Float64]) -> Float64:
        return cos(pi * x[0]) * cos(pi * x[1])

    def s2(x: List[Float64]) -> Float64:
        return sin(pi * x[0]) * sin(pi * x[1])

    # 2D, 1-forms: u = grad(cos cos) + curl(sin sin), lambda = 2 pi^2
    def u1(x: List[Float64]) raises -> List[Float64]:
        var a = pi * sin(pi * x[0]) * cos(pi * x[1])
        var b = pi * cos(pi * x[0]) * sin(pi * x[1])
        return [-a + a, -b - b]

    def f1(x: List[Float64]) raises -> List[Float64]:
        var v = u1(x)
        return [2 * pi * pi * v[0], 2 * pi * pi * v[1]]

    def sigma1(x: List[Float64]) raises -> List[Float64]:
        return [2 * pi * pi * p2(x)]

    def du1(x: List[Float64]) raises -> List[Float64]:
        return [2 * pi * pi * s2(x)]

    _check_mixed(2, 1, f1, sigma1, u1, du1)

    # 2D, 2-forms: u = sin sin dx^dy, sigma = curl(sin sin)
    def u2(x: List[Float64]) raises -> List[Float64]:
        return [s2(x)]

    def f2(x: List[Float64]) raises -> List[Float64]:
        return [2 * pi * pi * s2(x)]

    def sigma2(x: List[Float64]) raises -> List[Float64]:
        return [
            pi * sin(pi * x[0]) * cos(pi * x[1]),
            -pi * cos(pi * x[0]) * sin(pi * x[1]),
        ]

    _check_mixed(2, 2, f2, sigma2, u2, u2)

    # 3D, 1-forms: u = grad(cos cos cos), lambda = 3 pi^2
    def p3(x: List[Float64]) -> Float64:
        return cos(pi * x[0]) * cos(pi * x[1]) * cos(pi * x[2])

    def u3(x: List[Float64]) raises -> List[Float64]:
        var c0 = cos(pi * x[0])
        var c1 = cos(pi * x[1])
        var c2 = cos(pi * x[2])
        return [
            -pi * sin(pi * x[0]) * c1 * c2,
            -pi * c0 * sin(pi * x[1]) * c2,
            -pi * c0 * c1 * sin(pi * x[2]),
        ]

    def f3(x: List[Float64]) raises -> List[Float64]:
        var v = u3(x)
        return [3 * pi * pi * v[0], 3 * pi * pi * v[1], 3 * pi * pi * v[2]]

    def sigma3(x: List[Float64]) raises -> List[Float64]:
        return [3 * pi * pi * p3(x)]

    def du3(x: List[Float64]) raises -> List[Float64]:
        return [0.0, 0.0, 0.0]

    _check_mixed(3, 1, f3, sigma3, u3, du3)


def test_mixed_block_preconditioner() raises:
    # MINRES on the mixed 1-form system: Jacobi needs more iterations as
    # the mesh is refined (measured 79, 205, 424); the block-diagonal
    # preconditioner with the exact inverses of the blocks' natural norms,
    # M0 + d0^T M1 d0 and M1 + d1^T M2 d1 (Arnold-Falk-Winther), does not
    # (7 at every size)
    var its = List[Int]()
    var jac = List[Int]()
    for j in range(3):
        var sc = SimplicialComplex(kuhn_grid([4 << j, 4 << j]))
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
            DenseCholesky(M0 + sc[0].d.transpose() @ B), DenseCholesky(M1 + A)
        )
        var exact = minres(K, b, M=P, rtol=1e-10)
        var plain = minres(K, b, M=jacobi(K), rtol=1e-10, maxiter=100000)
        assert_true(exact.converged() and plain.converged())
        its.append(exact.iterations)
        jac.append(plain.iterations)
    for j in range(3):
        assert_true(its[j] <= 10, String("block: ", its[j], " iterations"))
    assert_true(jac[2] >= 4 * jac[0], String("Jacobi: ", jac[0], ", ", jac[2]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
