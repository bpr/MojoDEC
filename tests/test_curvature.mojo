from std.math import cos, isnan, pi, sin, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.abstract_simplicial_complex import AbstractSimplicialComplex
from mojodec.core.mesh import RegularCubeMesh
from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import fget, iget, rows_of
from mojodec.routines.curvature import (
    angle_defect,
    gaussian_curvature,
    mean_curvature,
    mean_curvature_normal,
)
from mojodec.routines.io.meshio import read_mesh
from mojodec.routines.mesh_generation import (
    icosahedron,
    icosphere,
    simplicial_grid_2d,
)
from mojodec.routines.quality import (
    shape_regularity,
    well_centered_fraction,
    well_centeredness,
)


def sphere(subdivisions: Int, radius: Float64) raises -> SimplicialComplex:
    var vt = icosphere(subdivisions, radius)
    return SimplicialComplex(vt)


def total(x: nm.NDArray[f64]) -> Float64:
    var s: Float64 = 0
    for i in range(x.size):
        s += x.unsafe_get(i)
    return s


def test_icosphere_shape() raises:
    for s in range(3):
        var vt = icosphere(s, 2.0)
        assert_equal(rows_of(vt.vertices), 10 * 4**s + 2)
        assert_equal(rows_of(vt.elements), 20 * 4**s)
        for v in range(rows_of(vt.vertices)):
            var r: Float64 = 0
            for k in range(3):
                r += fget(vt.vertices, v, k) ** 2
            assert_almost_equal(sqrt(r), 2.0, atol=1e-14)
        # every triangle's normal points away from the center
        for t in range(rows_of(vt.elements)):
            var p = List[List[Float64]]()
            for c in range(3):
                var i = iget(vt.elements, t, c)
                p.append(
                    [
                        fget(vt.vertices, i, 0),
                        fget(vt.vertices, i, 1),
                        fget(vt.vertices, i, 2),
                    ]
                )
            var u: List[Float64] = [
                p[1][0] - p[0][0],
                p[1][1] - p[0][1],
                p[1][2] - p[0][2],
            ]
            var w: List[Float64] = [
                p[2][0] - p[0][0],
                p[2][1] - p[0][1],
                p[2][2] - p[0][2],
            ]
            var nrm: List[Float64] = [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ]
            var out = nrm[0] * p[0][0] + nrm[1] * p[0][1] + nrm[2] * p[0][2]
            assert_true(out > 0)


def test_euler_characteristic() raises:
    assert_equal(sphere(2, 1.0).euler_characteristic(), 2)
    var vt = simplicial_grid_2d(4)
    assert_equal(SimplicialComplex(vt).euler_characteristic(), 1)
    var torus = read_mesh("tests/meshes/torus_tet.xml")
    assert_equal(
        SimplicialComplex(
            torus.vertices, torus.elements
        ).euler_characteristic(),
        0,
    )
    var bitmap = nm.ones[DType.int](Shape(3, 3))
    bitmap[Item(1, 1)] = 0  # an annulus
    assert_equal(
        RegularCubeComplex(RegularCubeMesh(bitmap)).euler_characteristic(), 0
    )
    # the boundary of a triangle: a circle
    var circle = List[nm.NDArray[DType.int]]()
    circle.append(nm.fromstring[DType.int]("[[0,1],[1,2],[0,2]]"))
    assert_equal(AbstractSimplicialComplex(circle).euler_characteristic(), 0)


def test_gauss_bonnet() raises:
    # the angle defects add up to 2 pi chi, exactly
    for s in range(3):
        var sc = sphere(s, 1.5)
        assert_almost_equal(total(angle_defect(sc)), 4 * pi, atol=1e-12)
    # a square: no interior defect, pi/2 at each corner, 0 elsewhere on the edge
    var vt = simplicial_grid_2d(4)
    var sc = SimplicialComplex(vt)
    var D = angle_defect(sc)
    assert_almost_equal(total(D), 2 * pi, atol=1e-12)
    var corners: List[Int] = [0, 4, 20, 24]
    for v in range(D.size):
        var expected = pi / 2 if v in corners else 0.0
        assert_almost_equal(D.unsafe_get(v), expected, atol=1e-12)


def test_sphere_curvatures() raises:
    for r in [1.0, 2.5]:
        var sc = sphere(3, r)
        var K = gaussian_curvature(sc)
        var H = mean_curvature(sc)
        var Hn = mean_curvature_normal(sc)
        for v in range(K.size):
            # second order: within 0.6% at 3 subdivisions
            assert_almost_equal(K.unsafe_get(v) * r * r, 1.0, atol=6e-3)
            assert_almost_equal(H.unsafe_get(v) * r, 1.0, atol=1e-4)
            # outward
            var dot: Float64 = 0
            for k in range(3):
                dot += fget(Hn, v, k) * fget(sc.vertices(), v, k)
            assert_almost_equal(dot / (H.unsafe_get(v) * r), 1.0, atol=1e-4)
        # K integrates to 4 pi exactly
        var KA: Float64 = 0
        for v in range(K.size):
            KA += K.unsafe_get(v) * sc[0].dual_volume.unsafe_get(v)
        assert_almost_equal(KA, 4 * pi, atol=1e-11)


def test_flat_surface_in_3d() raises:
    # a planar grid lifted into 3D: no curvature at interior vertices
    var vt = simplicial_grid_2d(4)
    var V = nm.zeros[f64](Shape(rows_of(vt.vertices), 3))
    for v in range(rows_of(vt.vertices)):
        V.unsafe_set(3 * v, fget(vt.vertices, v, 0))
        V.unsafe_set(3 * v + 1, 0.5 * fget(vt.vertices, v, 1))
        V.unsafe_set(3 * v + 2, 0.5 * fget(vt.vertices, v, 1))
    var sc = SimplicialComplex(V, vt.elements)
    var Hn = mean_curvature_normal(sc)
    var K = gaussian_curvature(sc)
    for v in range(rows_of(V)):
        var i = v % 5
        var j = v // 5
        if 0 < i < 4 and 0 < j < 4:
            assert_almost_equal(K.unsafe_get(v), 0.0, atol=1e-12)
            for k in range(3):
                assert_almost_equal(fget(Hn, v, k), 0.0, atol=1e-12)


def test_needs_a_surface() raises:
    var torus = read_mesh("tests/meshes/torus_tet.xml")
    var sc = SimplicialComplex(torus.vertices, torus.elements)
    with assert_raises(contains="2-complex"):
        _ = angle_defect(sc)
    with assert_raises(contains="2-complex"):
        _ = mean_curvature_normal(sc)


def test_quality() raises:
    var sc = SimplicialComplex(
        nm.fromstring[f64](
            "[[0,0],[1,0],[0.5,0.8660254037844386],[2,0],[2,1],[4,0],[3,0.2]]"
        ),
        nm.fromstring[DType.int]("[[0,1,2],[1,3,4],[3,5,6]]"),
    )
    var w = well_centeredness(sc)
    assert_almost_equal(w.unsafe_get(0), 1.0 / 3.0, atol=1e-12)  # equilateral
    assert_almost_equal(w.unsafe_get(1), 0.0, atol=1e-12)  # right angle at 3
    assert_true(w.unsafe_get(2) < -0.5)  # obtuse: 157 degrees at (3, 0.2)
    # stored circumcenters (geometry) and recomputed ones (none) agree,
    # whatever the vertex order of the elements
    var verts = sc.vertices().copy()
    var flipped = nm.fromstring[DType.int]("[[2,1,0],[4,1,3],[6,5,3]]")
    var with_geometry = well_centeredness(SimplicialComplex(verts, flipped))
    var without = well_centeredness(
        SimplicialComplex(verts, flipped, compute_geometry=False)
    )
    for t in range(3):
        assert_almost_equal(
            with_geometry.unsafe_get(t), w.unsafe_get(t), atol=1e-12
        )
        assert_almost_equal(without.unsafe_get(t), w.unsafe_get(t), atol=1e-12)
    assert_almost_equal(well_centered_fraction(sc), 1.0 / 3.0, atol=1e-15)
    assert_almost_equal(
        well_centered_fraction(sc, tol=-1e-9), 2.0 / 3.0, atol=1e-15
    )
    var s = shape_regularity(sc)
    assert_almost_equal(s.unsafe_get(0), 1.0, atol=1e-12)
    assert_true(s.unsafe_get(1) < 1)
    var vt = icosahedron()
    var ico = SimplicialComplex(vt)
    assert_almost_equal(well_centered_fraction(ico), 1.0, atol=0)
    var reg = shape_regularity(ico)
    for t in range(reg.size):
        assert_almost_equal(reg.unsafe_get(t), 1.0, atol=1e-12)


def test_unused_vertices_and_small_angles() raises:
    # an unused vertex 0 ahead of the icosahedron: curvature once read
    # level-0 rows as vertex ids
    var ico = icosahedron()
    var nv = rows_of(ico.vertices)
    var V = nm.zeros[DType.float64](Shape(nv + 1, 3))
    for v in range(nv):
        for j in range(3):
            V.unsafe_set((v + 1) * 3 + j, fget(ico.vertices, v, j))
    var T = ico.elements.copy()
    for i in range(T.size):
        T.unsafe_set(i, T.unsafe_get(i) + 1)
    var K = gaussian_curvature(SimplicialComplex(ico))
    var shifted = SimplicialComplex(V, T)
    var K2 = gaussian_curvature(shifted)
    var Hn = mean_curvature_normal(shifted)
    assert_true(isnan(K2.unsafe_get(0)))
    assert_true(isnan(fget(Hn, 0, 0)))
    for v in range(nv):
        assert_almost_equal(
            K2.unsafe_get(v + 1), K.unsafe_get(v), atol=0, rtol=1e-14
        )
    # a flat fan whose sliver has a 1e-8 angle at the center: the angle was
    # computed as 0, leaving a defect of 1e-8
    var t = 0.00000001
    var W = nm.zeros[DType.float64](Shape(5, 2))
    var angles: List[Float64] = [0.0, t, 2.0, 4.0]
    for i in range(4):
        W.unsafe_set((i + 1) * 2, cos(angles[i]))
        W.unsafe_set((i + 1) * 2 + 1, sin(angles[i]))
    var F = nm.zeros[DType.int](Shape(4, 3))
    for i in range(4):
        F.unsafe_set(i * 3 + 1, Scalar[DType.int](i + 1))
        F.unsafe_set(i * 3 + 2, Scalar[DType.int]((i + 1) % 4 + 1))
    var defect = angle_defect(SimplicialComplex(W, F))
    assert_almost_equal(defect.unsafe_get(0), 0.0, atol=1e-15, rtol=0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
