from std.math import asin, cos, pi, sqrt
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
from mojodec.routines.linalg import eigh_generalized
from mojodec.routines.mesh_generation import loop_subdivision
from mojodec.routines.time import Leapfrog, TypedLeapfrog


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
