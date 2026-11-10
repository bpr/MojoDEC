from std.math import cos, pi, sin
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.abstract_simplicial_complex import AbstractSimplicialComplex
from mojodec.core.cell_set import CellSet
from mojodec.core.chain_complex import ChainComplex
from mojodec.core.mesh import RegularCubeMesh, SimplicialMesh
from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.rips_complex import RipsComplex
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import int_array, iset, real_array, fset
from mojodec.routines.homology import (
    betti_numbers,
    non_manifold_cells,
    relative_betti_numbers,
)
from mojodec.routines.io.meshio import read_mesh
from mojodec.routines.mesh_generation import (
    cube_grid,
    icosphere,
    simplicial_grid_2d,
    triangulate_ncube,
)


def assert_list(got: List[Int], expected: List[Int]) raises:
    assert_equal(len(got), len(expected), String(got))
    for i in range(len(got)):
        assert_equal(got[i], expected[i], String(got))


def euler(b: List[Int]) -> Int:
    var chi = 0
    for k in range(len(b)):
        chi += -b[k] if k % 2 == 1 else b[k]
    return chi


def torus_surface(n: Int) raises -> SimplicialComplex:
    """An n x n periodic grid of squares, each split into two triangles."""
    var T = int_array(2 * n * n, 3)
    for i in range(n):
        for j in range(n):
            var a = i * n + j
            var b = ((i + 1) % n) * n + j
            var c = ((i + 1) % n) * n + (j + 1) % n
            var d = i * n + (j + 1) % n
            var t = 2 * (i * n + j)
            iset(T, t, 0, a)
            iset(T, t, 1, b)
            iset(T, t, 2, c)
            iset(T, t + 1, 0, a)
            iset(T, t + 1, 1, c)
            iset(T, t + 1, 2, d)
    var V = nm.zeros[f64](Shape(n * n, 3))
    return SimplicialComplex(SimplicialMesh(V, T), compute_geometry=False)


def test_triangle_and_square() raises:
    var tri = SimplicialComplex(
        nm.fromstring[f64]("[[0,0],[1,0],[0,1]]"),
        nm.fromstring[DType.int]("[[0,1,2]]"),
    )
    assert_list(tri.betti_numbers(), [1, 0, 0])
    assert_list(tri.relative_betti_numbers(), [0, 0, 1])
    var grid = SimplicialComplex(simplicial_grid_2d(5))
    assert_list(grid.betti_numbers(), [1, 0, 0])
    assert_list(grid.relative_betti_numbers(), [0, 0, 1])


def test_closed_surfaces() raises:
    var sphere = SimplicialComplex(icosphere(2))
    assert_list(sphere.betti_numbers(), [1, 0, 1])
    # no boundary: relative and absolute agree
    assert_list(sphere.relative_betti_numbers(), [1, 0, 1])
    var torus = torus_surface(4)
    assert_list(torus.betti_numbers(), [1, 2, 1])
    assert_equal(euler(torus.betti_numbers()), torus.euler_characteristic())


def test_solid_torus_and_lefschetz_duality() raises:
    var mesh = read_mesh("tests/meshes/torus_tet.xml")
    var sc = SimplicialComplex(mesh.vertices, mesh.elements)
    var b = sc.betti_numbers()
    assert_list(b, [1, 1, 0, 0])
    assert_equal(euler(b), sc.euler_characteristic())
    # b_k(K, dK) = b_{3-k}(K)
    assert_list(sc.relative_betti_numbers(), [0, 0, 1, 1])
    # its boundary is a torus surface
    assert_list(sc.betti_numbers(sc.boundary_cells()), [1, 2, 1, 0])


def test_links() raises:
    var mesh = read_mesh("tests/meshes/torus_tet.xml")
    var sc = SimplicialComplex(mesh.vertices, mesh.elements)
    var on_boundary = sc.boundary_cells()[0].copy()
    var interior = 0
    for v in range(sc[0].num_simplices):
        var lk = sc.link(Simplex([v]))
        if v in on_boundary:
            assert_list(sc.betti_numbers(lk), [1, 0, 0, 0])  # a disk
        else:
            assert_list(sc.betti_numbers(lk), [1, 0, 1, 0])  # a sphere
            interior += 1
    assert_true(interior > 0)
    var sphere = SimplicialComplex(icosphere(1))
    assert_list(sphere.betti_numbers(sphere.link(Simplex([0]))), [1, 1, 0])


def test_disconnected() raises:
    var V = nm.zeros[f64](Shape(6, 2))
    var S = nm.fromstring[DType.int]("[[0,1,2],[3,4,5]]")
    var sc = SimplicialComplex(SimplicialMesh(V, S), compute_geometry=False)
    assert_list(sc.betti_numbers(), [2, 0, 0])
    # two triangles touching at a vertex: one piece
    var T = nm.fromstring[DType.int]("[[0,1,2],[2,3,4]]")
    var bow = SimplicialComplex(SimplicialMesh(V, T), compute_geometry=False)
    assert_list(bow.betti_numbers(), [1, 0, 0])


def test_projective_plane_has_rational_coefficients() raises:
    # the 6-vertex projective plane: over Z/2 its Betti numbers are
    # [1, 1, 1]; over the rationals, [1, 0, 0]
    var asc = AbstractSimplicialComplex(
        [
            nm.fromstring[DType.int](
                "[[0,1,2],[0,2,3],[0,3,4],[0,4,5],[0,5,1],"
                "[1,2,4],[2,3,5],[3,4,1],[4,5,2],[5,1,3]]"
            )
        ]
    )
    var b = asc.betti_numbers()
    assert_list(b, [1, 0, 0])
    assert_equal(euler(b), asc.euler_characteristic())


def test_cube_complexes() raises:
    var bitmap = nm.ones[DType.int](Shape(3, 3))
    bitmap[Item(1, 1)] = 0
    var annulus = RegularCubeComplex(RegularCubeMesh(bitmap))
    assert_list(annulus.betti_numbers(), [1, 1, 0])
    assert_list(annulus.relative_betti_numbers(), [0, 1, 1])
    var solid = RegularCubeComplex(
        RegularCubeMesh(nm.ones[DType.int](Shape(2, 2, 2)))
    )
    assert_list(solid.betti_numbers(), [1, 0, 0, 0])
    assert_list(solid.relative_betti_numbers(), [0, 0, 0, 1])
    var cube = SimplicialComplex(triangulate_ncube(cube_grid([3, 2])))
    assert_list(cube.betti_numbers(), [1, 0, 0])


def test_rips_circle() raises:
    # 12 points on a circle, joined to their neighbors only: a loop
    var n = 12
    var P = real_array(n, 2)
    for i in range(n):
        fset(P, i, 0, cos(2 * pi * Float64(i) / Float64(n)))
        fset(P, i, 1, sin(2 * pi * Float64(i) / Float64(n)))
    var rips = RipsComplex(P, 0.6)
    assert_list(rips.betti_numbers(), [1, 1])
    # a large radius fills it in
    assert_equal(RipsComplex(P, 2.5).betti_numbers()[1], 0)


def test_chain_complex_and_errors() raises:
    var sc = SimplicialComplex(icosphere(1))
    var cc = sc.chain_complex()
    assert_list(betti_numbers(cc), [1, 0, 1])
    # a set of cells missing faces is not a subcomplex
    var top = CellSet.top_cells(sc)
    with assert_raises(contains="not a subcomplex"):
        _ = sc.betti_numbers(top)
    with assert_raises(contains="not a subcomplex"):
        _ = relative_betti_numbers(cc, top)
    # cells of another complex
    var grid = SimplicialComplex(simplicial_grid_2d(2))
    with assert_raises(contains="count differs"):
        _ = sc.betti_numbers(grid.boundary_cells())
    # an empty subcomplex: relative = absolute
    assert_list(sc.relative_betti_numbers(CellSet.empty(sc)), [1, 0, 1])
    # non-integer boundary entries
    var half = CSRMatrix.from_dense(nm.fromstring[f64]("[[0.5],[-0.5]]"))
    with assert_raises(contains="must be integers"):
        _ = betti_numbers(ChainComplex(2, [half^]))


def test_large_mesh() raises:
    # ~20k triangles: the elimination must stay near linear
    var sc = SimplicialComplex(icosphere(5), compute_geometry=False)
    assert_list(sc.betti_numbers(), [1, 0, 1])
    var grid = SimplicialComplex(
        simplicial_grid_2d(100), compute_geometry=False
    )
    assert_list(grid.betti_numbers(), [1, 0, 0])
    assert_list(grid.relative_betti_numbers(), [0, 0, 1])


def assert_flagged(cells: CellSet, expected: List[List[Int]]) raises:
    var got = cells.indices()
    for k in range(len(got)):
        var want = expected[k].copy() if k < len(expected) else List[Int]()
        assert_list(got[k], want)


def test_manifolds() raises:
    var tri = SimplicialComplex(
        nm.fromstring[f64]("[[0,0],[1,0],[0,1]]"),
        nm.fromstring[DType.int]("[[0,1,2]]"),
    )
    assert_true(tri.is_homology_manifold())
    assert_true(SimplicialComplex(simplicial_grid_2d(4)).is_homology_manifold())
    assert_true(SimplicialComplex(icosphere(2)).is_homology_manifold())
    assert_true(torus_surface(4).is_homology_manifold())
    var mesh = read_mesh("tests/meshes/torus_tet.xml")
    var solid = SimplicialComplex(mesh.vertices, mesh.elements)
    assert_true(solid.is_homology_manifold())
    var bitmap = nm.ones[DType.int](Shape(3, 3))
    bitmap[Item(1, 1)] = 0
    assert_true(
        RegularCubeComplex(RegularCubeMesh(bitmap)).is_homology_manifold()
    )
    # the projective plane is a manifold, though not orientable
    var rp2 = AbstractSimplicialComplex(
        [
            nm.fromstring[DType.int](
                "[[0,1,2],[0,2,3],[0,3,4],[0,4,5],[0,5,1],"
                "[1,2,4],[2,3,5],[3,4,1],[4,5,2],[5,1,3]]"
            )
        ]
    )
    assert_true(rp2.is_homology_manifold())
    # a loop of points on a circle is a 1-manifold
    var n = 12
    var P = real_array(n, 2)
    for i in range(n):
        fset(P, i, 0, cos(2 * pi * Float64(i) / Float64(n)))
        fset(P, i, 1, sin(2 * pi * Float64(i) / Float64(n)))
    assert_true(RipsComplex(P, 0.6).is_homology_manifold())


def test_pinched_meshes() raises:
    # two triangles meeting at vertex 2
    var V = nm.zeros[f64](Shape(7, 3))
    var bow = SimplicialComplex(
        SimplicialMesh(V, nm.fromstring[DType.int]("[[0,1,2],[2,3,4]]")),
        compute_geometry=False,
    )
    assert_false(bow.is_homology_manifold())
    assert_flagged(bow.non_manifold_cells(), [[2]])
    # two tetrahedra meeting at vertex 0
    var cones = SimplicialComplex(
        SimplicialMesh(V, nm.fromstring[DType.int]("[[0,1,2,3],[0,4,6,5]]")),
        compute_geometry=False,
    )
    assert_flagged(cones.non_manifold_cells(), [[0]])
    # two tetrahedra sharing edge 01: every vertex link is a disk, but the
    # edge's link is two separate edges
    var hinge = SimplicialComplex(
        SimplicialMesh(V, nm.fromstring[DType.int]("[[0,1,2,3],[0,1,5,4]]")),
        compute_geometry=False,
    )
    assert_flagged(hinge.non_manifold_cells(), [[], [0]])


def test_non_manifold_abstract_complexes() raises:
    # three triangles on edge 01 (not a pseudomanifold)
    var fan = AbstractSimplicialComplex(
        [nm.fromstring[DType.int]("[[0,1,2],[0,1,3],[0,1,4]]")]
    )
    var bad = fan.non_manifold_cells()
    assert_true(not bad.is_empty())
    assert_true((1, 0) in bad)  # edge 01 is the first edge
    # a triangle with a dangling edge: the edge and its vertices fail
    var flag = AbstractSimplicialComplex(
        [
            nm.fromstring[DType.int]("[[0,1,2]]"),
            nm.fromstring[DType.int]("[[2,3]]"),
        ]
    )
    assert_false(flag.is_homology_manifold())


def test_manifold_check_on_a_large_mesh() raises:
    var sc = SimplicialComplex(icosphere(4), compute_geometry=False)
    assert_true(sc.is_homology_manifold())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
