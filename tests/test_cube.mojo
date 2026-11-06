from std.testing import TestSuite, assert_equal

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import RegularCubeMesh
from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import IntArray, rows_of, cols_of, take_int_rows
from mojodec.routines.cube_array import cube_array_search, cube_array_boundary


def I(s: String) raises -> NDArray[DType.int]:
    return nm.fromstring[DType.int](s)


def assert_int_array(a: IntArray, s: String) raises:
    var b = I(s)
    assert_equal(rows_of(a), rows_of(b))
    assert_equal(cols_of(a), cols_of(b))
    for i in range(a.size):
        assert_equal(Int(a.unsafe_get(i)), Int(b.unsafe_get(i)))


def assert_dense(A: CSRMatrix, s: String) raises:
    var b = nm.fromstring[f64](s)
    assert_equal(A.nrows, rows_of(b))
    assert_equal(A.ncols, cols_of(b))
    var D = A.todense()
    for i in range(D.size):
        assert_equal(D.unsafe_get(i), b.unsafe_get(i))


def check_search(arr: String, var rows: List[Int]) raises:
    var a = I(arr)
    var r = cube_array_search(a, take_int_rows(a, rows))
    for i in range(len(rows)):
        assert_equal(Int(r.unsafe_get(i)), rows[i])


def test_search() raises:
    check_search("[[0,0,0],[0,0,1],[0,1,0],[1,0,1]]", [0, 2, 1, 1, 3, 0])
    check_search("[[0,1,0],[0,0,2],[0,2,0],[1,0,3]]", [0, 2, 1, 1, 3, 0])
    check_search(
        "[[0,1,0,1],[0,0,1,2],[0,2,0,2],[1,0,0,1]]", [2, 1, 0, 0, 3, 1, 2]
    )


def test_boundary_1d() raises:
    var fb = cube_array_boundary(I("[[0,0],[1,0],[2,0]]"), 1)
    assert_int_array(fb.faces, "[[0],[1],[2],[3]]")
    assert_dense(fb.boundary, "[[-1,0,0],[1,-1,0],[0,1,-1],[0,0,1]]")
    fb = cube_array_boundary(I("[[0,0],[3,0],[5,0]]"), 1)
    assert_int_array(fb.faces, "[[0],[1],[3],[4],[5],[6]]")


def test_boundary_2d() raises:
    var fb = cube_array_boundary(I("[[0,0,0],[0,0,1],[0,1,0],[1,0,1]]"), 1)
    assert_int_array(fb.faces, "[[0,0],[0,1],[1,0],[1,1]]")
    assert_dense(fb.boundary, "[[-1,-1,0,0],[0,1,-1,0],[1,0,0,-1],[0,0,1,1]]")
    fb = cube_array_boundary(I("[[0,0,0,1],[0,1,0,1],[1,0,0,1],[1,1,0,1]]"), 2)
    assert_int_array(
        fb.faces,
        "[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[0,2,0],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[1,2,0],[2,0,1],[2,1,1]]",
    )
    assert_dense(
        fb.boundary,
        "[[1,0,0,0],[-1,0,0,0],[-1,1,0,0],[0,-1,0,0],[0,-1,0,0],[0,0,1,0],[1,0,-1,0],[0,0,-1,1],[0,1,0,-1],[0,0,0,-1],[0,0,1,0],[0,0,0,1]]",
    )
    fb = cube_array_boundary(I("[[0,0,0,1],[2,0,0,1]]"), 2)
    assert_dense(
        fb.boundary, "[[1,0],[-1,0],[-1,0],[1,0],[0,1],[0,-1],[0,-1],[0,1]]"
    )


def test_boundary_3d() raises:
    var fb = cube_array_boundary(I("[[0,0,0,0,1,2]]"), 3)
    assert_int_array(
        fb.faces,
        "[[0,0,0,0,1],[0,0,0,0,2],[0,0,0,1,2],[0,0,1,0,1],[0,1,0,0,2],[1,0,0,1,2]]",
    )
    assert_dense(fb.boundary, "[[-1],[1],[-1],[1],[-1],[1]]")


def test_complex_exactness() raises:
    var bitmap = nm.ones[DType.int](Shape(3, 3, 2))
    bitmap[Item(1, 1, 0)] = 0
    var rcc = RegularCubeComplex(RegularCubeMesh(bitmap))
    var b = rcc.chain_complex()
    for k in range(1, b.complex_dimension()):
        assert_equal((b.boundary(k) @ b.boundary(k + 1)).max_abs(), 0.0)
    assert_equal(rows_of(rcc[3].cube_array), 17)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
