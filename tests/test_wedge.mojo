from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.cochain import Cochain, d
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import fget, iget, rows_of
from mojodec.routines.io.meshio import read_mesh
from mojodec.routines.mesh_generation import simplicial_grid_2d
from mojodec.routines.wedge import wedge


def grid() raises -> SimplicialComplex:
    """A 3x3 grid with one vertex moved, so the triangles are not all alike."""
    var vt = simplicial_grid_2d(3)
    var V = vt[0].copy()
    V.unsafe_set(5 * 2, 0.45)
    V.unsafe_set(5 * 2 + 1, 0.2)
    return SimplicialComplex(V, vt[1])


def tets() raises -> SimplicialComplex:
    var mesh = read_mesh("tests/meshes/torus_tet.xml")
    return SimplicialComplex(mesh.vertices, mesh.elements)


def random_cochain(sc: SimplicialComplex, k: Int, seed: Int) raises -> Cochain:
    nm.random.seed(seed)
    var c = sc.get_cochain(k)
    c.v = nm.random.rand[f64](Shape(c.v.size))
    return c^


def coordinate_form(sc: SimplicialComplex, axis: Int) raises -> Cochain:
    """The 1-cochain of the constant form d(x_axis): edge extents along axis."""
    var c = sc.get_cochain(1)
    for e in range(sc[1].num_simplices):
        var a = iget(sc[1].simplices, e, 0)
        var b = iget(sc[1].simplices, e, 1)
        c[e] = fget(sc.vertices(), b, axis) - fget(sc.vertices(), a, axis)
    return c^


def assert_cochains_equal(a: Cochain, b: Cochain, atol: Float64 = 1e-12) raises:
    assert_equal(a.k, b.k)
    assert_equal(a.v.size, b.v.size)
    for i in range(a.v.size):
        assert_almost_equal(a.v.unsafe_get(i), b.v.unsafe_get(i), atol=atol)


def test_dx_wedge_dy_is_signed_area() raises:
    var sc = grid()
    var w = wedge(coordinate_form(sc, 0), coordinate_form(sc, 1))
    assert_equal(w.k, 2)
    for t in range(sc[2].num_simplices):
        assert_almost_equal(w[t], sc[2].primal_volume.unsafe_get(t), atol=1e-14)


def test_functions() raises:
    var sc = grid()
    var f = random_cochain(sc, 0, 1)
    var g = random_cochain(sc, 0, 2)
    var fg = wedge(f, g)
    for v in range(fg.v.size):
        assert_almost_equal(fg[v], f[v] * g[v], atol=1e-15)
    # a 1-cochain times a function: scaled by the function's mean on the edge
    var a = random_cochain(sc, 1, 3)
    var af = wedge(a, f)
    for e in range(sc[1].num_simplices):
        var mean = (
            f[iget(sc[1].simplices, e, 0)] + f[iget(sc[1].simplices, e, 1)]
        ) / 2
        assert_almost_equal(af[e], a[e] * mean, atol=1e-15)


def test_graded_anticommutativity() raises:
    for sc in [grid(), tets()]:
        var n = sc.complex_dimension()
        for k in range(n + 1):
            for l in range(n + 1 - k):
                var a = random_cochain(sc, k, 10 + k)
                var b = random_cochain(sc, l, 20 + l)
                var ab = wedge(a, b)
                var ba = wedge(b, a)
                var sign: Float64 = -1.0 if (k * l) % 2 == 1 else 1.0
                for i in range(ab.v.size):
                    assert_almost_equal(ab[i], sign * ba[i], atol=1e-13)


def test_leibniz() raises:
    # d(a ^ b) = da ^ b + (-1)^k a ^ db
    for sc in [grid(), tets()]:
        var n = sc.complex_dimension()
        for k in range(n):
            for l in range(n - k):
                var a = random_cochain(sc, k, 30 + k)
                var b = random_cochain(sc, l, 40 + l)
                var lhs = d(wedge(a, b))
                var rhs = wedge(d(a), b)
                var sign: Float64 = -1.0 if k % 2 == 1 else 1.0
                rhs = rhs + wedge(a, d(b)).scaled(sign)
                assert_cochains_equal(lhs, rhs, atol=1e-13)


def test_errors() raises:
    var sc = grid()
    var a = random_cochain(sc, 2, 1)
    var b = random_cochain(sc, 1, 2)
    with assert_raises(contains="exceed"):
        _ = wedge(a, b)
    with assert_raises(contains="primal"):
        _ = wedge(sc.get_cochain(1, is_primal=False), b)
    with assert_raises(contains="different complexes"):
        _ = wedge(b, random_cochain(grid(), 0, 3))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
