from std.math import asin, atan, cos, pi, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.mesh import SimplicialMesh
from mojodec.core.sparse import CSRMatrix
from mojodec.core.typed import PRIMAL, TypedCochain, TypedComplex
from mojodec.core.utils import dot, real_vector
from mojodec.routines.linalg import cg, eigh_generalized
from mojodec.routines.mesh_generation import loop_subdivision
from mojodec.routines.time import (
    CrankNicolson,
    ImplicitMidpoint,
    Leapfrog,
    TypedCrankNicolson,
    TypedImplicitMidpoint,
    TypedLeapfrog,
)
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.routines.innerproduct import whitney_innerproduct
from mojodec.routines.mesh_generation import kuhn_grid
from mojodec.core.utils import RealArray, fget


def oscillator(dt: Float64) raises -> Leapfrog:
    # 2 q' = p, 3 p' = -q: frequency 1/sqrt(6)
    return Leapfrog(
        DiagonalOperator([2.0]),
        DiagonalOperator([3.0]),
        CSRMatrix.from_dense(nm.fromstring[f64]("[[1]]")),
        dt,
    )


def test_oscillator() raises:
    var dt = 0.2
    var lf = oscillator(dt)
    assert_almost_equal(lf.max_stable_dt, 2 * sqrt(6.0), atol=1e-12, rtol=0)
    var q = real_vector(1)
    var p = real_vector(1)
    q.unsafe_set(0, 1.0)
    var e0 = lf.energy(q, p)
    assert_true(e0 > 0)
    # q_n = cos(n theta) exactly, 2 sin(theta/2) = omega dt
    var theta = 2 * asin(dt / sqrt(6.0) / 2)
    for n in range(1, 1001):
        lf.step(q, p)
        assert_almost_equal(lf.energy(q, p), e0, atol=1e-14, rtol=0)
        assert_almost_equal(
            q.unsafe_get(0), cos(Float64(n) * theta), atol=1e-12, rtol=0
        )
    # steps=k is k single steps
    var q2 = real_vector(1)
    var p2 = real_vector(1)
    q2.unsafe_set(0, 1.0)
    lf.step(q2, p2, steps=1000)
    assert_equal(q2.unsafe_get(0), q.unsafe_get(0))
    assert_equal(p2.unsafe_get(0), p.unsafe_get(0))


def test_invalid_schemes() raises:
    with assert_raises(contains="not below the stability limit"):
        _ = oscillator(5.0)
    with assert_raises(contains="dt must be positive"):
        _ = oscillator(0.0)
    with assert_raises(contains="masses must be positive"):
        _ = Leapfrog(
            DiagonalOperator([-2.0]),
            DiagonalOperator([3.0]),
            CSRMatrix.from_dense(nm.fromstring[f64]("[[1]]")),
            0.1,
        )
    with assert_raises(contains="K is 1 x 2, expected 1 x 1"):
        _ = Leapfrog(
            DiagonalOperator([2.0]),
            DiagonalOperator([3.0]),
            CSRMatrix.from_dense(nm.fromstring[f64]("[[1, 1]]")),
            0.1,
        )
    var lf = oscillator(0.1)
    var q = real_vector(2)
    var p = real_vector(1)
    with assert_raises(contains="lengths"):
        lf.step(q, p)


def equilateral(subdivisions: Int) raises -> TypedComplex[2]:
    """An equilateral triangle of side 1 cut into equilateral triangles."""
    var mesh = SimplicialMesh(
        nm.fromstring[f64]("[[0, 0], [1, 0], [0.5, 0.8660254037844386]]"),
        nm.fromstring[DType.int]("[[0, 1, 2]]"),
    )
    for _ in range(subdivisions):
        mesh = loop_subdivision(mesh)
    return TypedComplex[2](mesh)


def test_maxwell_cavity_mode() raises:
    var tc = equilateral(4)
    # electric field e on the edges, tangential part zero on the boundary
    # (P drops the boundary rows); magnetic flux b on the triangles
    var boundary = tc.boundary_cells[1]()
    var P = tc.identity[1, PRIMAL]().exclude_subset(boundary)
    var d1 = tc.d[1, PRIMAL]()
    var star1 = tc.star[1, PRIMAL]()
    var star2 = tc.star[2, PRIMAL]()
    var K = P.T() @ d1.T() @ star2.to_matrix_op()
    var S = K @ d1 @ P  # stiffness, primal 1 -> dual 1
    var eig = eigh_generalized(S, star1)
    var first = 0
    while eig.values.unsafe_get(first) < 1e-8:
        first += 1
    var lam = eig.values.unsafe_get(first)
    # the lowest Neumann eigenvalue of the unit equilateral triangle
    var exact = (4 * pi / 3) ** 2
    assert_almost_equal(lam, exact, atol=0.005 * exact, rtol=0)

    # the stability limit is set by the largest eigenvalue
    var lam_max = eig.values.unsafe_get(len(eig.values) - 1)
    var dt = 0.5 * 2 / sqrt(lam_max)
    var lf = TypedLeapfrog(star1, star2, K, dt)
    assert_almost_equal(
        lf.max_stable_dt(), 2 / sqrt(lam_max), atol=0, rtol=1e-6
    )

    # start from the mode, b = 0: e(t) = cos(omega_dt t) e0
    var e0: TypedCochain[2, 1, PRIMAL] = eig.mode(first)
    for i in boundary.indices():  # rounding leaves ~1e-59 there
        e0[i] = 0.0
    var e = e0.copy()
    var b = tc.cochain[2, PRIMAL]()
    var E0 = lf.energy(e, b)
    var theta = 2 * asin(sqrt(lam) * dt / 2)
    var norm = dot(e0.v, e0.v)
    for n in range(1, 401):
        lf.step(e, b)
        assert_almost_equal(lf.energy(e, b), E0, atol=1e-13 * E0, rtol=0)
        assert_almost_equal(
            dot(e.v, e0.v) / norm, cos(Float64(n) * theta), atol=1e-9, rtol=0
        )
    # tangential e stays zero on the boundary
    for i in boundary.indices():
        assert_equal(e[i], 0.0)
    with assert_raises(contains="not below the stability limit"):
        _ = TypedLeapfrog(star1, star2, K, 1.01 * 2 / sqrt(lam_max))


def test_midpoint_oscillator() raises:
    # 2 q' = p, 3 p' = -q, omega = 1/sqrt(6): the midpoint rule turns by
    # theta = 2 atan(omega dt / 2) per step, for any dt, energy exact
    var K = CSRMatrix.from_dense(nm.fromstring[f64]("[[1]]"))
    for dt in [0.2, 10.0]:  # 10 is twice the leapfrog's limit
        var im = ImplicitMidpoint(
            DiagonalOperator([2.0]), DiagonalOperator([3.0]), K, dt
        )
        var q = real_vector(1)
        var p = real_vector(1)
        q.unsafe_set(0, 1.0)
        var e0 = im.energy(q, p)
        var theta = 2 * atan(dt / sqrt(6.0) / 2)
        for n in range(1, 201):
            im.step(q, p)
            assert_almost_equal(im.energy(q, p), e0, atol=1e-14, rtol=0)
            assert_almost_equal(
                q.unsafe_get(0), cos(Float64(n) * theta), atol=1e-12, rtol=0
            )


def _interior(sc: SimplicialComplex, k: Int) raises -> List[Int]:
    var boundary = sc.boundary_cells()[k].copy()
    var out = List[Int]()
    for i in range(sc[k].num_simplices):
        if i not in boundary:
            out.append(i)
    return out^


def test_midpoint_maxwell_with_whitney_mass() raises:
    # a Kuhn grid is not well-centered (the diagonals have zero dual
    # length), so the leapfrog refuses its star; the midpoint rule takes
    # the Whitney mass for e and the always-positive star2 for b
    var sc = SimplicialComplex(kuhn_grid([4, 4]))
    var inner = _interior(sc, 1)  # e = 0 on the boundary edges
    var all = List[Int]()
    for t in range(sc[2].num_simplices):
        all.append(t)
    var star2 = sc[2].star.copy()
    var K = (sc[1].d.transpose() @ star2.to_csr()).select(inner, all)
    var Mq = whitney_innerproduct(sc, 1).select(inner, inner)
    with assert_raises(contains="masses must be positive"):
        var star1 = List[Float64]()
        for i in inner:
            star1.append(sc[1].star.values[i])
        _ = Leapfrog(DiagonalOperator(star1^), star2, K, 0.01)
    # the lowest mode: S x = lambda Mq x, S = K star2^-1 K^T
    var S = K @ star2.inverse().to_csr() @ K.transpose()
    var eig = eigh_generalized(S.todense(), Mq.todense())
    var first = 0
    while eig.values.unsafe_get(first) < 1e-8:
        first += 1
    var lam = eig.values.unsafe_get(first)
    assert_almost_equal(lam, pi * pi, atol=0.1 * pi * pi, rtol=0)
    var q0 = real_vector(len(inner))
    for i in range(len(inner)):
        q0.unsafe_set(i, fget_col(eig.vectors, i, first))
    var dt = 0.05
    var im = ImplicitMidpoint(Mq, star2, K, dt)
    var q = q0.copy()
    var p = real_vector(sc[2].num_simplices)
    var E0 = im.energy(q, p)
    var theta = 2 * atan(sqrt(lam) * dt / 2)
    var norm = dot(q0, Mq @ q0)
    for n in range(1, 201):
        im.step(q, p)
        assert_almost_equal(im.energy(q, p), E0, atol=1e-12 * E0, rtol=0)
        assert_almost_equal(
            dot(q, Mq @ q0) / norm,
            cos(Float64(n) * theta),
            atol=1e-10,
            rtol=0,
        )
    assert_true(im.last_iterations > 0)


def fget_col(A: RealArray, i: Int, j: Int) raises -> Float64:
    return fget(A, i, j)


def _heat(cells: Int) raises -> Tuple[CSRMatrix, CSRMatrix]:
    """Mass and stiffness of P1 elements on the interior vertices of a
    Kuhn grid of the unit square (zero boundary values)."""
    var sc = SimplicialComplex(kuhn_grid([cells, cells]))
    var inner = _interior(sc, 0)
    var M = whitney_innerproduct(sc, 0).select(inner, inner)
    var K = (
        sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d
    ).select(inner, inner)
    return (M^, K^)


def test_crank_nicolson_modes_and_decay() raises:
    var mk = _heat(6)
    ref M = mk[0]
    ref K = mk[1]
    var eig = eigh_generalized(K.todense(), M.todense())
    var m = M.nrows
    for dt in [0.01, 1.0]:
        var cn = CrankNicolson(M, K, dt)
        for j in [0, m - 1]:  # the slowest and the stiffest mode
            var lam = eig.values.unsafe_get(j)
            var g = (1 - dt * lam / 2) / (1 + dt * lam / 2)
            var x0 = real_vector(m)
            for i in range(m):
                x0.unsafe_set(i, fget_col(eig.vectors, i, j))
            var u = x0.copy()
            var amp: Float64 = 1
            for _ in range(20):
                cn.step(u)
                amp *= g
                for i in range(m):
                    assert_almost_equal(
                        u.unsafe_get(i),
                        amp * x0.unsafe_get(i),
                        atol=1e-11,
                        rtol=0,
                    )
            if dt == 1.0 and j == m - 1:
                assert_true(g < -0.9)  # stiff modes flip sign: not L-stable
    # a rough start: u^T M u and u^T K u never increase
    var cn = CrankNicolson(M, K, 0.05)
    var u = real_vector(m)
    for i in range(m):
        u.unsafe_set(i, Float64((i * 7) % 5) - 2)
    var e = cn.energy(u)
    var a = dot(u, K @ u)
    for _ in range(30):
        cn.step(u)
        assert_true(cn.energy(u) <= e)
        assert_true(dot(u, K @ u) <= a)
        e = cn.energy(u)
        a = dot(u, K @ u)


def test_crank_nicolson_reaches_steady_state() raises:
    # with a constant source, u tends to the solution of K u = f; a lumped
    # (diagonal) mass gives the same limit
    var mk = _heat(6)
    ref M = mk[0]
    ref K = mk[1]
    var m = M.nrows
    var f = real_vector(m)
    for i in range(m):
        f.unsafe_set(i, 1.0 + 0.1 * Float64(i % 3))
    var exact = cg(K, f, rtol=1e-14).take_x()
    var lumped = List[Float64]()
    for i in range(m):
        var s: Float64 = 0
        for t in range(M.indptr[i], M.indptr[i + 1]):
            s += M.data[t]
        lumped.append(s)
    var full = CrankNicolson(M, K, 0.02)
    var lumped_cn = CrankNicolson(DiagonalOperator(lumped^), K, 0.02)
    for which in range(2):
        var u = real_vector(m)
        if which == 0:
            full.step(u, f, steps=400)
        else:
            lumped_cn.step(u, f, steps=400)
        for i in range(m):
            assert_almost_equal(
                u.unsafe_get(i), exact.unsafe_get(i), atol=1e-12, rtol=0
            )


def test_typed_implicit_schemes() raises:
    var tc = equilateral(2)
    var W1 = whitney_innerproduct[1](tc)
    var star2 = tc.star[2, PRIMAL]()
    var K = tc.d[1, PRIMAL]().T() @ star2.to_matrix_op()
    var im = TypedImplicitMidpoint(W1, star2, K, 0.1)
    var e = tc.cochain[1, PRIMAL]()
    for i in range(e.v.size):
        e[i] = Float64(i % 3) - 1
    var b = tc.cochain[2, PRIMAL]()
    var E0 = im.energy(e, b)
    im.step(e, b, steps=10)
    assert_almost_equal(im.energy(e, b), E0, atol=1e-12 * E0, rtol=0)
    # heat on the vertices with the star as a lumped mass
    var d0 = tc.d[0, PRIMAL]()
    var L = d0.T() @ tc.star[1, PRIMAL]() @ d0
    var cn = TypedCrankNicolson(tc.star[0, PRIMAL](), L, 0.01)
    var u = tc.cochain[0, PRIMAL]()
    for i in range(u.v.size):
        u[i] = Float64(i % 4)
    var e0 = cn.energy(u)
    cn.step(u, steps=5)
    assert_true(cn.energy(u) < e0)
    var other = equilateral(2)
    with assert_raises(contains="different complexes"):
        _ = TypedImplicitMidpoint(W1, other.star[2, PRIMAL](), K, 0.1)


def test_invalid_implicit_schemes() raises:
    var K = CSRMatrix.from_dense(nm.fromstring[f64]("[[1]]"))
    var one = DiagonalOperator([1.0])
    with assert_raises(contains="dt must be positive"):
        _ = ImplicitMidpoint(one, one, K, 0.0)
    with assert_raises(contains="Mp[0] = -1.0"):
        _ = ImplicitMidpoint(one, DiagonalOperator([-1.0]), K, 0.1)
    with assert_raises(contains="expected Mq square and K 1 x 2"):
        _ = ImplicitMidpoint(one, DiagonalOperator([1.0, 1.0]), K, 0.1)
    var skew = CSRMatrix.from_dense(nm.fromstring[f64]("[[2, 1], [0, 2]]"))
    with assert_raises(contains="symmetric"):
        _ = CrankNicolson(skew, skew, 0.1)
    with assert_raises(contains="same square size"):
        _ = CrankNicolson(one, skew, 0.1)
    var cn = CrankNicolson(one, K, 0.1)
    var u = real_vector(2)
    with assert_raises(contains="wrong length"):
        cn.step(u)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
