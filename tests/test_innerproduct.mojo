from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import RegularCubeMesh
from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import RealArray, fget, rows_of, cols_of
from mojodec.routines.combinatorial import combinations
from mojodec.routines.geometry import barycentric_gradients
from mojodec.routines.innerproduct import (
    massmatrix_rowcols,
    whitney_innerproduct,
    regular_cube_innerproduct,
)


def I(s: String) raises -> NDArray[DType.int]:
    return nm.fromstring[DType.int](s)


def R(s: String) raises -> NDArray[f64]:
    return nm.fromstring[f64](s)


def assert_close(A: RealArray, B: RealArray) raises:
    assert_equal(A.size, B.size)
    for i in range(A.size):
        assert_almost_equal(A.unsafe_get(i), B.unsafe_get(i), atol=1e-12)


def test_gradients() raises:
    assert_close(barycentric_gradients(R("[[0],[1]]")), R("[[-1],[1]]"))
    assert_close(
        barycentric_gradients(R("[[0,1,2],[1,2,3]]")),
        R(
            "[[-0.3333333333333333,-0.3333333333333333,-0.3333333333333333],[0.3333333333333333,0.3333333333333333,0.3333333333333333]]"
        ),
    )
    assert_close(
        barycentric_gradients(R("[[0,0],[1,0],[0,1]]")),
        R("[[-1,-1],[1,0],[0,1]]"),
    )
    assert_close(
        barycentric_gradients(R("[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]")),
        R("[[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]]"),
    )
    assert_close(
        barycentric_gradients(R("[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]]")),
        R("[[-1,-1,-1,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]]"),
    )


def test_massmatrix_rowcols() raises:
    var sc = SimplicialComplex(R("[[0,0],[1,0],[0,1]]"), I("[[0,1,2]]"))
    var rc = massmatrix_rowcols(sc, 0)
    assert_equal(rc[0], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert_equal(rc[1], [0, 1, 2, 0, 1, 2, 0, 1, 2])
    rc = massmatrix_rowcols(sc, 2)
    assert_equal(rc[0], [0])


def check_whitney(
    k: Int,
    v: String,
    s: String,
    var order: List[List[Int]],
    expected: String,
    factor: Float64 = 1.0,
) raises:
    var sc = SimplicialComplex(R(v), I(s))
    var M = whitney_innerproduct(sc, k)
    var perm = List[Int]()
    for ref o in order:
        perm.append(sc[k].simplex_to_index(Simplex(o)))
    var P = M.select(perm, perm).todense()
    var E = R(expected)
    for i in range(E.size):
        assert_almost_equal(
            P.unsafe_get(i), factor * E.unsafe_get(i), atol=1e-12
        )
    var D = M.todense()
    var n = rows_of(D)
    for i in range(n):
        for j in range(n):
            assert_almost_equal(fget(D, i, j), fget(D, j, i), atol=1e-14)


def test_whitney() raises:
    check_whitney(
        0,
        "[[0],[1]]",
        "[[0,1]]",
        [[0], [1]],
        "[[0.3333333333333333,0.16666666666666666],[0.16666666666666666,0.3333333333333333]]",
    )
    check_whitney(
        0,
        "[[1],[0]]",
        "[[1,0]]",
        [[1], [0]],
        "[[0.3333333333333333,0.16666666666666666],[0.16666666666666666,0.3333333333333333]]",
    )
    check_whitney(
        0,
        "[[-10],[3]]",
        "[[0,1]]",
        [[0], [1]],
        "[[4.333333333333333,2.1666666666666665],[2.1666666666666665,4.333333333333333]]",
    )
    check_whitney(
        0,
        "[[0,0],[1,0],[0,1]]",
        "[[0,1,2]]",
        [[0], [1], [2]],
        "[[2,1,1],[1,2,1],[1,1,2]]",
        1.0 / 24,
    )
    check_whitney(
        0,
        "[[0,0,1],[0,1,0],[1,0,0],[0,0,0]]",
        "[[3,2,1,0]]",
        [[0], [1], [2], [3]],
        "[[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2]]",
        1.0 / 120,
    )
    check_whitney(
        1,
        "[[0,0],[1,0],[0,1]]",
        "[[0,1,2]]",
        [[1, 2], [0, 1], [0, 2]],
        "[[1,0,0],[0,2,1],[0,1,2]]",
        1.0 / 6,
    )
    check_whitney(
        1,
        "[[1,0],[0,1],[0,0]]",
        "[[0,1,2]]",
        [[0, 1], [0, 2], [1, 2]],
        "[[1,0,0],[0,2,1],[0,1,2]]",
        1.0 / 6,
    )
    check_whitney(
        1,
        "[[0,0],[1,0],[0,1]]",
        "[[1,2,0]]",
        [[1, 2], [0, 1], [0, 2]],
        "[[1,0,0],[0,2,1],[0,1,2]]",
        1.0 / 6,
    )
    check_whitney(
        1,
        "[[0,0],[2,0],[0,2]]",
        "[[0,1,2]]",
        [[1, 2], [0, 1], [0, 2]],
        "[[1,0,0],[0,2,1],[0,1,2]]",
        1.0 / 6,
    )
    check_whitney(
        1,
        "[[0,0],[1,0],[0.5,0.8660254037844386]]",
        "[[0,1,2]]",
        [[1, 2], [0, 1], [0, 2]],
        "[[5,-1,1],[-1,5,1],[1,1,5]]",
        3.0**0.5 / 36,
    )
    check_whitney(2, "[[0,0],[1,0],[0,1]]", "[[0,1,2]]", [[0, 1, 2]], "[[2]]")
    check_whitney(2, "[[0,0],[1,0],[0,1]]", "[[1,2,0]]", [[0, 1, 2]], "[[2]]")
    check_whitney(
        1,
        "[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]",
        "[[0,1,2,3]]",
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        "[[10,5,5,0,0,0],[5,10,5,0,0,0],[5,5,10,0,0,0],[0,0,0,4,1,-1],[0,0,0,1,4,1],[0,0,0,-1,1,4]]",
        1.0 / 120,
    )
    check_whitney(
        2,
        "[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]",
        "[[0,1,2,3]]",
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        "[[6,-1,1,-1],[-1,16,4,-4],[1,4,16,4],[-1,-4,4,16]]",
        4.0 / 120,
    )
    check_whitney(
        2,
        "[[0,0,0],[0,1,0],[0,0,1],[1,0,0]]",
        "[[0,3,1,2]]",
        [[1, 2, 3], [0, 1, 2], [0, 3, 2], [0, 3, 1]],
        "[[6,-1,-1,1],[-1,16,-4,4],[-1,-4,16,4],[1,4,4,16]]",
        4.0 / 120,
    )


def get_K(var shape: List[Int], k: Int) raises -> RealArray:
    var bitmap = nm.ones[DType.int](NDArrayShape(shape))
    var rcc = RegularCubeComplex(RegularCubeMesh(bitmap))
    return regular_cube_innerproduct(rcc, k).todense()


def test_cube_innerproduct() raises:
    assert_close(
        get_K([1], 0),
        R(
            "[[0.3333333333333333,0.16666666666666666],[0.16666666666666666,0.3333333333333333]]"
        ),
    )
    assert_close(get_K([1], 1), R("[[1]]"))
    var K = get_K([1, 1], 0)
    var E = R("[[4,2,2,1],[2,4,1,2],[2,1,4,2],[1,2,2,4]]")
    for i in range(E.size):
        assert_almost_equal(K.unsafe_get(i), E.unsafe_get(i) / 36, atol=1e-14)
    K = get_K([1, 1], 1)
    E = R("[[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]]")
    for i in range(E.size):
        assert_almost_equal(K.unsafe_get(i), E.unsafe_get(i) / 6, atol=1e-14)
    assert_close(get_K([1, 1], 2), R("[[1]]"))
    K = get_K([1, 1, 1], 2)
    E = R(
        "[[2,0,0,1,0,0],[0,2,0,0,1,0],[0,0,2,0,0,1],[1,0,0,2,0,0],[0,1,0,0,2,0],[0,0,1,0,0,2]]"
    )
    for i in range(E.size):
        assert_almost_equal(K.unsafe_get(i), E.unsafe_get(i) / 6, atol=1e-14)
    K = get_K([2, 2], 1)
    assert_equal(rows_of(K), 12)
    var local = R("[[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]]")
    var blocks: List[List[Int]] = [
        [0, 1, 2, 6],
        [5, 6, 7, 10],
        [2, 3, 4, 8],
        [7, 8, 9, 11],
    ]
    var expected = nm.zeros[f64](Shape(12, 12))
    for ref b in blocks:
        for i in range(4):
            for j in range(4):
                var off = b[i] * 12 + b[j]
                expected.unsafe_set(
                    off,
                    expected.unsafe_get(off) + local.unsafe_get(i * 4 + j) / 6,
                )
    assert_close(K, expected)


def test_whitney_needs_geometry() raises:
    var sc = SimplicialComplex(
        nm.fromstring[f64]("[[0,0],[1,0],[0,1]]"),
        nm.fromstring[DType.int]("[[0,1,2]]"),
        compute_geometry=False,
    )
    for k in range(3):
        with assert_raises(contains="without geometry"):
            _ = whitney_innerproduct(sc, k)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
