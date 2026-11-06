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
from mojodec.core.cell_set import CellComplex, CellSet
from mojodec.core.chain_complex import ChainComplex
from mojodec.core.mesh import RegularCubeMesh
from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset
from mojodec.routines.io.meshio import read_mesh
from mojodec.routines.mesh_generation import icosphere, simplicial_grid_2d
from mojodec.routines.topology import (
    boundary_of,
    closure_of,
    interior_of,
    link_of,
    star_of,
)


def triangle() raises -> SimplicialComplex:
    return SimplicialComplex(
        nm.fromstring[f64]("[[0,0],[1,0],[0,1]]"),
        nm.fromstring[DType.int]("[[0,1,2]]"),
    )


def cells[C: CellComplex](cx: C, lists: List[List[Int]]) raises -> CellSet:
    return CellSet.from_lists(cx, lists)


def assert_cells(got: CellSet, expected: List[List[Int]]) raises:
    var g = got.indices()
    assert_equal(len(g), len(expected))
    for k in range(len(g)):
        assert_equal(len(g[k]), len(expected[k]), String("dimension ", k))
        for i in range(len(g[k])):
            assert_equal(g[k][i], expected[k][i])


def assert_simplices(
    sc: SimplicialComplex, cells: CellSet, expected: List[List[Int]]
) raises:
    """The cells, as simplices, are `expected`, lowest dimension first."""
    var got = cells.simplices(sc)
    assert_equal(len(got), len(expected))
    for i in range(len(got)):
        assert_true(got[i].vertices == expected[i], String(got[i]))


def test_triangle() raises:
    var sc = triangle()
    assert_simplices(
        sc, sc.star(Simplex([0])), [[0], [0, 1], [0, 2], [0, 1, 2]]
    )
    assert_simplices(sc, sc.closure(Simplex([0, 1])), [[0], [1], [0, 1]])
    assert_simplices(
        sc,
        sc.closure(Simplex([0, 1, 2])),
        [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]],
    )
    assert_simplices(sc, sc.link(Simplex([0])), [[1], [2], [1, 2]])
    assert_simplices(sc, sc.link(Simplex([0, 1])), [[2]])
    assert_cells(sc.link(Simplex([0, 1, 2])), [[], [], []])
    # orientation does not matter
    assert_true(sc.link(Simplex([1, 0])) == sc.link(Simplex([0, 1])))


def test_sets() raises:
    var sc = triangle()
    # the star of two vertices is the union of their stars
    assert_cells(sc.star(cells(sc, [[0, 1]])), [[0, 1], [0, 1, 2], [0]])
    assert_true(
        sc.star(cells(sc, [[0, 1]]))
        == sc.star(Simplex([0])) | sc.star(Simplex([1]))
    )
    # a shorter list leaves higher dimensions empty; the edges are 01, 02, 12
    assert_cells(sc.closure(cells(sc, [[], [2]])), [[1, 2], [2], []])
    # the link of an edge's two vertices: the opposite vertex, and the
    # edge itself is not in it because it touches them
    assert_cells(sc.link(cells(sc, [[0, 1]])), [[2], [], []])


def test_cell_set() raises:
    var sc = triangle()
    var a = cells(sc, [[0, 2], [1]])
    assert_equal(len(a), 3)
    assert_equal(a.dimension(), 1)
    assert_equal(a.complex_dimension(), 2)
    assert_equal(a.euler_characteristic(), 1)
    assert_true((1, 1) in a)
    assert_false((1, 0) in a)
    assert_false((5, 0) in a)
    assert_true(CellSet.empty(sc).is_empty())
    assert_equal(CellSet.empty(sc).dimension(), -1)
    assert_equal(len(CellSet.full(sc)), 7)
    assert_equal(CellSet.full(sc).euler_characteristic(), 1)
    var b = CellSet.from_simplices(sc, [Simplex([0]), Simplex([2, 0])])
    assert_cells(b, [[0], [1], []])
    assert_cells(a & b, [[0], [1], []])
    assert_cells(a - b, [[2], [], []])
    assert_cells(a | b, [[0, 2], [1], []])
    assert_cells(
        CellSet.from_subset(sc, Subset.from_indices(sc, 1, [2, 0])),
        [[], [0, 2], []],
    )
    assert_cells(CellSet.from_indices(sc, 2, [0]), [[], [], [0]])
    assert_equal(String(a), "CellSet(0: [0, 2], 1: [1], 2: [])")
    # levels are subsets, and can be changed in place
    a[2].add(0)
    assert_equal(a.dimension(), 2)
    with assert_raises(contains="dimension 3 out of range"):
        _ = a[3]
    with assert_raises(contains="does not fit"):
        _ = CellSet.from_subset(sc, Subset.from_indices(sc, 1, [0]).dual())


def test_surface_links_are_circles() raises:
    var vt = simplicial_grid_2d(4)
    var grid = SimplicialComplex(vt)
    var vt2 = icosphere(2)
    var ball = SimplicialComplex(vt2)
    var boundary = List[Int]()
    for ref e in grid.boundary():
        for v in e.vertices:
            boundary.append(v)
    for v in range(25):
        var lk = grid.link(Simplex([v]))
        assert_true(lk[2].is_empty())
        # a circle around an interior vertex, a path from a boundary one
        assert_equal(lk.euler_characteristic(), 1 if v in boundary else 0)
        # the link is a subcomplex
        assert_true(grid.closure(lk) == lk)
    for v in range(ball[0].num_simplices):
        var lk = ball.link(Simplex([v]))
        assert_equal(lk.euler_characteristic(), 0)
        assert_true(len(lk[0]) == 5 or len(lk[0]) == 6)
        assert_equal(len(lk[0]), len(lk[1]))
    # on a closed surface an edge's link is its two opposite vertices
    for e in range(ball[1].num_simplices):
        var lk = ball.link(CellSet.from_indices(ball, 1, [e]))
        assert_equal(len(lk[0]), 2)
        assert_equal(len(lk[1]) + len(lk[2]), 0)


def test_tetrahedral_links_are_spheres_or_disks() raises:
    var mesh = read_mesh("tests/meshes/torus_tet.xml")
    var sc = SimplicialComplex(mesh.vertices, mesh.elements)
    var on_boundary = sc.boundary_cells()[0].copy()
    var interior = 0
    for v in range(sc[0].num_simplices):
        var lk = sc.link(Simplex([v]))
        assert_equal(lk.euler_characteristic(), 1 if v in on_boundary else 2)
        if v not in on_boundary:
            interior += 1
    assert_true(interior > 0)


def test_star_and_closure_laws() raises:
    var vt = icosphere(1)
    var sc = SimplicialComplex(vt)
    var cc = sc.chain_complex()
    var S = cells(sc, [[3, 7], [5], [11]])
    # the free functions agree on the complex and on its chain complex
    var st = star_of(cc, S)
    var cl = closure_of(cc, S)
    assert_true(st == star_of(sc, S))
    assert_true(cl == closure_of(sc, S))
    assert_true((S - st).is_empty() and (S - cl).is_empty())
    # the closure is closed and the star is open: applying them again
    # changes nothing
    assert_true(closure_of(cc, cl) == cl)
    assert_true(star_of(cc, st) == st)
    # the link misses everything touching S
    var lk = link_of(cc, S)
    assert_true(lk == link_of(sc, S))
    assert_true((lk & star_of(cc, cl)).is_empty())


def test_chain_complex() raises:
    var vt = icosphere(1)
    var sc = SimplicialComplex(vt)
    var cc = sc.chain_complex()
    assert_equal(cc.complex_dimension(), 2)
    assert_equal(len(cc), 3)
    for k in range(3):
        assert_equal(cc.num_cells(k), sc[k].num_simplices)
    for k in range(1, 3):
        assert_equal((cc.boundary(k) - sc[k].boundary).max_abs(), 0.0)
        assert_equal((cc.coboundary(k - 1) - sc[k - 1].d).max_abs(), 0.0)
    assert_equal((cc.boundary(1) @ cc.boundary(2)).max_abs(), 0.0)
    # built from boundaries alone, the coboundaries are their transposes
    var again = ChainComplex(
        sc[0].num_simplices, [sc[1].boundary.copy(), sc[2].boundary.copy()]
    )
    assert_equal((again.coboundary(1) - sc[1].d).max_abs(), 0.0)
    with assert_raises(contains="outside 1..2"):
        _ = cc.boundary(0)
    with assert_raises(contains="outside 0..1"):
        _ = cc.coboundary(2)
    with assert_raises(contains="B_2 has"):
        _ = ChainComplex(
            sc[0].num_simplices,
            [sc[1].boundary.copy(), sc[1].boundary.copy()],
        )


def test_other_complexes() raises:
    # a 3x3 grid of squares: the link of the middle vertex is the ring of
    # 8 vertices and 8 edges around its 4 squares
    var rcc = RegularCubeComplex(
        RegularCubeMesh(nm.ones[DType.int](Shape(3, 3)))
    )
    var middle = -1
    for v in range(rcc[0].cube_array.shape[0]):
        if (
            rcc[0].cube_array.unsafe_get(2 * v) == 1
            and rcc[0].cube_array.unsafe_get(2 * v + 1) == 1
        ):
            middle = v
    var lk = rcc.link(cells(rcc, [[middle]]))
    assert_equal(len(lk[0]), 8)
    assert_equal(len(lk[1]), 8)
    assert_equal(len(lk[2]), 0)
    assert_equal(len(rcc.star(cells(rcc, [[middle]]))[2]), 4)
    assert_true(lk == link_of(rcc.chain_complex(), cells(rcc, [[middle]])))
    # the boundary of a triangle, as an abstract complex: a vertex's link
    # is the two other vertices
    var circle = List[nm.NDArray[DType.int]]()
    circle.append(nm.fromstring[DType.int]("[[0,1],[1,2],[0,2]]"))
    var asc = AbstractSimplicialComplex(circle)
    assert_cells(asc.link(cells(asc, [[0]])), [[1, 2], []])
    # top simplices keep their given order, so edge 2 is [0, 2]
    assert_cells(asc.closure(cells(asc, [[], [2]])), [[0, 2], [2]])


def test_boundary_and_interior() raises:
    var sc = triangle()
    # a triangle's boundary is its three edges and vertices
    assert_cells(sc.boundary_cells(), [[0, 1, 2], [0, 1, 2], []])
    assert_cells(sc.interior_cells(), [[], [], [0]])
    assert_simplices(sc, sc.boundary(Simplex([0, 1])), [[0], [1]])
    assert_cells(sc.interior(Simplex([0, 1])), [[], [0], []])
    # vertices alone have no boundary
    assert_cells(sc.boundary(cells(sc, [[0, 2]])), [[], [], []])
    assert_cells(sc.interior(cells(sc, [[0, 2]])), [[0, 2], [], []])
    # on a 4x4 grid the boundary is the 16-gon around it, and the
    # boundary vertices are those of PyDEC's boundary faces
    var vt = simplicial_grid_2d(4)
    var grid = SimplicialComplex(vt)
    var bd = grid.boundary_cells()
    assert_equal(len(bd[0]), 16)
    assert_equal(len(bd[1]), 16)
    assert_equal(len(bd[2]), 0)
    assert_equal(len(grid.boundary()), 16)
    assert_true(
        CellSet.from_simplices(grid, grid.boundary())
        == CellSet.from_subset(grid, bd[1])
    )
    var interior = grid.interior_cells()
    assert_equal(len(interior[0]), 9)
    assert_equal(len(interior[2]), grid[2].num_simplices)
    # boundary and interior split the complex
    assert_true((bd & interior).is_empty())
    assert_true((bd | interior) == CellSet.full(grid))
    # a closed surface has no boundary; its boundary's boundary is empty
    var vt2 = icosphere(1)
    var ball = SimplicialComplex(vt2)
    assert_true(ball.boundary_cells().is_empty())
    var cc = grid.chain_complex()
    assert_true(boundary_of(cc, bd).is_empty())
    # the closed star of a vertex has the link as its boundary and the
    # open star as its interior
    var st = grid.star(Simplex([12]))
    var cl = closure_of(cc, st)
    assert_true(boundary_of(cc, cl) == grid.link(Simplex([12])))
    assert_true(interior_of(cc, cl) == grid.star(Simplex([12])))


def test_boundary_of_a_solid() raises:
    var mesh = read_mesh("tests/meshes/torus_tet.xml")
    var sc = SimplicialComplex(mesh.vertices, mesh.elements)
    var bd = sc.boundary_cells()
    # the boundary of a solid torus is a closed torus
    assert_equal(len(bd[2]), len(sc.boundary()))
    assert_equal(len(bd[3]), 0)
    assert_equal(bd.euler_characteristic(), 0)
    assert_true(sc.boundary(CellSet.from_subset(sc, bd[2])).is_empty())
    # each level is the boundary subset of that dimension
    for k in range(3):
        var faces = Subset.full(sc, 3).manifold_boundary(sc)
        for _ in range(2 - k):
            faces = faces.boundary(sc)
        assert_true(bd[k] == faces)


def test_cube_boundary() raises:
    var rcc = RegularCubeComplex(
        RegularCubeMesh(nm.ones[DType.int](Shape(3, 3)))
    )
    var bd = rcc.boundary_cells()
    assert_equal(len(bd[0]), 12)
    assert_equal(len(bd[1]), 12)
    assert_equal(len(rcc.interior_cells()[0]), 4)
    var circle = List[nm.NDArray[DType.int]]()
    circle.append(nm.fromstring[DType.int]("[[0,1],[1,2],[0,2]]"))
    assert_cells(AbstractSimplicialComplex(circle).boundary_cells(), [[], []])


def test_errors() raises:
    var sc = triangle()
    with assert_raises(contains="not in the complex"):
        _ = sc.star(Simplex([0, 3]))
    with assert_raises(contains="wrong dimension"):
        _ = sc.link(Simplex([0, 1, 2, 3]))
    with assert_raises(contains="no 1-cell with index 7"):
        _ = cells(sc, [[], [7]])
    with assert_raises(contains="dimensions given"):
        _ = cells(sc, [[0], [], [], [0]])
    with assert_raises(contains="not pure: 0-cell 2"):
        _ = sc.boundary(cells(sc, [[2], [0]]))
    # a cell set of another complex
    var vt = simplicial_grid_2d(2)
    var grid = SimplicialComplex(vt)
    with assert_raises(contains="count differs"):
        _ = sc.star(cells(grid, [[0]]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
