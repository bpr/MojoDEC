"""Convergence of Whitney-form approximations on Kuhn grids of the unit
cube in 1, 2, and 3 dimensions, as the cells halve: the L2 projection and
the interpolant W R of a smooth form of every degree, and the Galerkin
solution of Poisson's equation. Rates come from the last two meshes.
"""

from std.math import cos, log2, pi, sin
from std.testing import TestSuite, assert_true

from mojodec import (
    SimplicialComplex,
    cg,
    derham_map,
    form_index_sets,
    jacobi,
    kuhn_grid,
    l2_error,
    l2_projection,
    whitney_innerproduct,
    whitney_load,
)
from mojodec.core.utils import real_vector


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
