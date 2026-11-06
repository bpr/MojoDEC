from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.ncube import NCube
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import iget, fget, row_list, rows_of, cols_of, take_rows
from mojodec.routines.geometry import signed_volume, unsigned_volume
from mojodec.routines.mesh_generation import (
    simplicial_grid_2d,
    cube_grid,
    triangulate_ncube,
    loop_subdivision,
)


def test_simplicial_grid_2d() raises:
    var vi = simplicial_grid_2d(4)
    assert_equal(rows_of(vi.vertices), 25)
    assert_equal(rows_of(vi.elements), 32)
    var total: Float64 = 0
    for t in range(rows_of(vi.elements)):
        var v = signed_volume(take_rows(vi.vertices, row_list(vi.elements, t)))
        assert_true(v > 0)
        total += v
    assert_almost_equal(total, 1.0)
    var sc = SimplicialComplex(vi)
    assert_equal(sc[1].num_simplices, 25 + 32 - 1)


def test_cube_grid() raises:
    var vi = cube_grid([2, 1])
    var V = nm.fromstring[f64]("[[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]]")
    for i in range(V.size):
        assert_equal(vi.vertices.unsafe_get(i), V.unsafe_get(i))
    var expected: List[Int] = [0, 3, 1, 4, 1, 4, 2, 5]
    assert_equal(vi.corners.ndim, 3)
    for i in range(8):
        assert_equal(Int(vi.corners.unsafe_get(i)), expected[i])


def test_triangulate_ncube() raises:
    var vi = cube_grid([3, 2])
    var tv = triangulate_ncube(vi)
    assert_equal(rows_of(tv.elements), 4 * 6)
    var total: Float64 = 0
    for t in range(rows_of(tv.elements)):
        var v = signed_volume(take_rows(tv.vertices, row_list(tv.elements, t)))
        assert_true(v > 0)
        total += v
    assert_almost_equal(total, 6.0)


def test_loop_subdivision() raises:
    var vi = simplicial_grid_2d(2)
    var sub = loop_subdivision(vi)
    assert_equal(rows_of(sub.elements), 4 * rows_of(vi.elements))
    assert_equal(rows_of(sub.vertices), 9 + 16)
    var total: Float64 = 0
    for t in range(rows_of(sub.elements)):
        total += signed_volume(
            take_rows(sub.vertices, row_list(sub.elements, t))
        )
    assert_almost_equal(total, 1.0)


def check_orientation(b: String, expected: Bool) raises:
    var B = nm.fromstring[DType.int](b)
    var A = nm.zeros[DType.int](B.shape)
    for i in range(A.size):
        A.unsafe_set(i, Scalar[DType.int](i))
    var a_cube = NCube(A)
    var b_cube = NCube(B)
    assert_false(a_cube.relative_orientation(a_cube))
    assert_false(b_cube.relative_orientation(b_cube))
    assert_equal(a_cube.relative_orientation(b_cube), expected)
    assert_equal(b_cube.relative_orientation(a_cube), expected)


def test_ncube() raises:
    check_orientation("[0,1]", False)
    check_orientation("[1,0]", True)
    check_orientation("[[0,1],[2,3]]", False)
    check_orientation("[[1,3],[0,2]]", False)
    check_orientation("[[3,2],[1,0]]", False)
    check_orientation("[[2,0],[3,1]]", False)
    check_orientation("[[2,3],[0,1]]", True)
    check_orientation("[[0,2],[1,3]]", True)
    check_orientation("[[1,0],[3,2]]", True)
    check_orientation("[[3,1],[2,0]]", True)
    check_orientation("[[[0,1],[2,3]],[[4,5],[6,7]]]", False)
    check_orientation("[[[1,5],[3,7]],[[0,4],[2,6]]]", False)
    check_orientation("[[[5,4],[7,6]],[[1,0],[3,2]]]", False)
    check_orientation("[[[7,3],[5,1]],[[6,2],[4,0]]]", False)
    check_orientation("[[[4,5],[6,7]],[[0,1],[2,3]]]", True)
    check_orientation("[[[0,4],[2,6]],[[1,5],[3,7]]]", True)
    check_orientation("[[[5,1],[7,3]],[[4,0],[6,2]]]", True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
