from std.python import Python
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import RegularCubeMesh
from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import RealArray, fget, real_vector, rows_of
from mojodec.vis.draw import (
    cube_quivers,
    lineplot,
    lineplot2,
    simplex_quivers,
    triplot,
)


def assert_rows(a: RealArray, s: String) raises:
    var b = nm.fromstring[f64](s)
    assert_equal(a.size, b.size)
    for i in range(a.size):
        assert_almost_equal(
            a.unsafe_get(i), b.unsafe_get(i), atol=1e-15, rtol=0
        )


def unit_triangle() raises -> SimplicialComplex:
    return SimplicialComplex(
        nm.fromstring[f64]("[[0,0],[1,0],[0,1]]"),
        nm.fromstring[DType.int]("[[0,1,2]]"),
    )


def test_simplex_quivers() raises:
    var sc = unit_triangle()
    # the Whitney form of edge (0,1) is l0 dl1 - l1 dl0
    var form = real_vector(3)
    form.unsafe_set(0, 1.0)
    var q = simplex_quivers(sc, form)
    assert_rows(q.bases, "[[0.3333333333333333,0.3333333333333333]]")
    assert_rows(q.vectors, "[[0.6666666666666666,0.3333333333333333]]")


def test_simplex_quivers_bad_length() raises:
    var sc = unit_triangle()
    with assert_raises():
        _ = simplex_quivers(sc, real_vector(2))


def test_cube_quivers() raises:
    var rcc = RegularCubeComplex(
        RegularCubeMesh(nm.ones[DType.int](Shape(1, 1)))
    )
    var vals = real_vector(rows_of(rcc[1].cube_array))
    for i in range(vals.size):
        vals.unsafe_set(i, 1.0)
    var q = cube_quivers(rcc, vals)
    assert_rows(q.bases, "[[0.5,0.5]]")
    assert_rows(q.vectors, "[[1,1]]")


def test_draw() raises:
    Python.import_module("matplotlib").use("Agg")
    var plt = Python.import_module("matplotlib.pyplot")
    var sc = unit_triangle()
    _ = plt.figure()
    var tri = triplot(sc.vertices(), sc.simplices(), labels=True)
    assert_equal(Int(py=len(tri.get_paths())), 1)
    var lines = lineplot(sc.vertices(), sc[1].simplices, labels=True)
    assert_equal(Int(py=len(lines.get_paths())), 3)
    var widths = real_vector(3)
    for i in range(3):
        widths.unsafe_set(i, Float64(i + 1))
    lines = lineplot(sc.vertices(), sc[1].simplices, False, widths)
    assert_equal(Float64(py=lines.get_linewidths()[2]), 3.0)
    var tails = nm.fromstring[f64]("[[0,0],[1,1]]")
    var heads = nm.fromstring[f64]("[[1,0],[2,2]]")
    lines = lineplot2(tails, heads, labels=True)
    assert_equal(Int(py=len(lines.get_paths())), 2)
    # one labeled triangle, three labeled edges, two labeled segments
    assert_equal(Int(py=len(plt.gca().texts)), 6)
    plt.close("all")


def test_draw_rejects_3d() raises:
    var V = nm.fromstring[f64]("[[0,0,0],[1,0,0],[0,1,0]]")
    var S = nm.fromstring[DType.int]("[[0,1,2]]")
    with assert_raises(contains="2D points"):
        _ = triplot(V, S)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
