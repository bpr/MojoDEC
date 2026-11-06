from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import RealArray, fget, iget, real_array, rows_of
from mojodec.routines.io.meshio import read_mesh
from mojodec.routines.musical import dual_edge_pieces, flat
from mojodec.routines.quality import well_centeredness
from mojodec.vis.draw import simplex_quivers


def obtuse() raises -> SimplicialComplex:
    """A fan of triangles, several obtuse, so some dual pieces are negative."""
    return SimplicialComplex(
        nm.fromstring[f64](
            "[[0,0],[2,0],[3,1],[1,0.3],[-1,1],[0.9,2],[-0.5,-1]]"
        ),
        nm.fromstring[DType.int](
            "[[0,1,3],[1,2,3],[3,2,5],[4,3,5],[0,3,4],[6,1,0]]"
        ),
    )


def tets() raises -> SimplicialComplex:
    var mesh = read_mesh("tests/meshes/torus_tet.xml")
    return SimplicialComplex(mesh.vertices, mesh.elements)


def constant_field(sc: SimplicialComplex, x: List[Float64]) raises -> RealArray:
    var n = sc.complex_dimension()
    var X = real_array(sc[n].num_simplices, len(x))
    for t in range(rows_of(X)):
        for c in range(len(x)):
            X.unsafe_set(t * len(x) + c, x[c])
    return X^


def test_pieces_sum_to_dual_volumes() raises:
    for sc in [obtuse(), tets()]:
        var n = sc.complex_dimension()
        var P = dual_edge_pieces(sc)
        var total = List[Float64](length=sc[1].num_simplices, fill=0.0)
        for t in range(sc[n].num_simplices):
            var j = 0
            for a in range(n + 1):
                for b in range(a + 1, n + 1):
                    var e = sc[1].simplex_to_index(top_edge(sc, t, a, b))
                    total[e] += fget(P, t, j)
                    j += 1
        for e in range(len(total)):
            assert_almost_equal(
                total[e], sc[1].dual_volume.unsafe_get(e), atol=1e-12
            )


def top_edge(sc: SimplicialComplex, t: Int, a: Int, b: Int) raises -> Simplex:
    """The edge between sorted vertices `a` and `b` of top simplex `t`."""
    var n = sc.complex_dimension()
    return Simplex([iget(sc[n].simplices, t, a), iget(sc[n].simplices, t, b)])


def test_obtuse_mesh_has_negative_pieces() raises:
    var sc = obtuse()
    var w = well_centeredness(sc)
    var obtuse_count = 0
    for i in range(w.size):
        if w.unsafe_get(i) < 0:
            obtuse_count += 1
    assert_true(obtuse_count >= 2)
    var P = dual_edge_pieces(sc)
    var negative = 0
    for i in range(P.size):
        if P.unsafe_get(i) < 0:
            negative += 1
    assert_true(negative >= 1)


def check_constant_field(sc: SimplicialComplex, x: List[Float64]) raises:
    """`flat(X)[e] = X . (e_1 - e_0)` for a constant field `X`."""
    var c = flat(sc, constant_field(sc, x))
    assert_equal(c.k, 1)
    for e in range(sc[1].num_simplices):
        var a = iget(sc[1].simplices, e, 0)
        var b = iget(sc[1].simplices, e, 1)
        var expected: Float64 = 0
        for k in range(len(x)):
            expected += x[k] * (
                fget(sc.vertices(), b, k) - fget(sc.vertices(), a, k)
            )
        assert_almost_equal(c[e], expected, atol=1e-12)


def test_constant_field_is_exact() raises:
    check_constant_field(obtuse(), [0.7, -1.3])
    check_constant_field(tets(), [0.3, 2.0, -0.5])


def test_flat_inverts_whitney_sharp_on_constant_forms() raises:
    var sc = obtuse()
    var c = flat(sc, constant_field(sc, [1.5, 0.25]))
    var X = simplex_quivers(sc, c.v)[1].copy()
    var back = flat(sc, X)
    for e in range(c.v.size):
        assert_almost_equal(back[e], c[e], atol=1e-12)


def test_bad_shape() raises:
    var sc = obtuse()
    with assert_raises(contains="one vector per top simplex"):
        _ = flat(sc, real_array(3, 2))
    with assert_raises(contains="one vector per top simplex"):
        _ = flat(sc, real_array(sc[2].num_simplices, 3))


def test_needs_geometry() raises:
    var m = read_mesh("tests/meshes/torus_tet.xml")
    var sc = SimplicialComplex(m, compute_geometry=False)
    with assert_raises(contains="without geometry"):
        _ = dual_edge_pieces(sc)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
