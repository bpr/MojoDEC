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
from mojodec.core.utils import (
    RealArray,
    fget,
    iget,
    real_array_from_rows,
    rows_of,
    cols_of,
)
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
        assert_almost_equal(
            A.unsafe_get(i), B.unsafe_get(i), atol=1e-14, rtol=0
        )


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
    assert_equal(rc.rows, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert_equal(rc.cols, [0, 1, 2, 0, 1, 2, 0, 1, 2])
    rc = massmatrix_rowcols(sc, 2)
    assert_equal(rc.rows, [0])


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
            P.unsafe_get(i), factor * E.unsafe_get(i), atol=5e-12, rtol=0
        )
    var D = M.todense()
    var n = rows_of(D)
    for i in range(n):
        for j in range(n):
            assert_almost_equal(
                fget(D, i, j), fget(D, j, i), atol=1e-15, rtol=0
            )


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
        assert_almost_equal(
            K.unsafe_get(i), E.unsafe_get(i) / 36, atol=1e-15, rtol=0
        )
    K = get_K([1, 1], 1)
    E = R("[[2,0,1,0],[0,2,0,1],[1,0,2,0],[0,1,0,2]]")
    for i in range(E.size):
        assert_almost_equal(
            K.unsafe_get(i), E.unsafe_get(i) / 6, atol=1e-15, rtol=0
        )
    assert_close(get_K([1, 1], 2), R("[[1]]"))
    K = get_K([1, 1, 1], 2)
    E = R(
        "[[2,0,0,1,0,0],[0,2,0,0,1,0],[0,0,2,0,0,1],[1,0,0,2,0,0],[0,1,0,0,2,0],[0,0,1,0,0,2]]"
    )
    for i in range(E.size):
        assert_almost_equal(
            K.unsafe_get(i), E.unsafe_get(i) / 6, atol=1e-15, rtol=0
        )
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


def test_whitney_extreme_degrees() raises:
    # a p-simplex in R^p: the Whitney p-form is 1/volume on it, and the
    # 0-forms are the barycentric coordinates, with mass
    # volume (1 + delta_ij) / ((p + 1)(p + 2)); p = 4 takes the eliminated
    # 4 x 4 determinants
    for p in range(1, 5):
        var v = List[List[Float64]]()
        v.append(List[Float64](length=p, fill=0))
        for i in range(p):
            var row = List[Float64](length=p, fill=0)
            row[i] = Float64(i + 1)
            # shear so that the edges are not orthogonal
            if i > 0:
                row[0] = 0.5
            v.append(row^)
        var vs = String("[")
        for i in range(p + 1):
            vs += String(v[i]) + ("," if i < p else "]")
        var s = String("[")
        for i in range(p + 1):
            s += String(i) + ("," if i < p else "]]")
        var sc = SimplicialComplex(R(vs), I("[" + s))
        var vol = abs(sc[p].primal_volume.unsafe_get(0))
        var top = whitney_innerproduct(sc, p).todense()
        assert_almost_equal(top.unsafe_get(0) * vol, 1.0, atol=1e-14, rtol=0)
        var M0 = whitney_innerproduct(sc, 0).todense()
        for i in range(p + 1):
            for j in range(p + 1):
                var expected = (
                    vol * Float64(1 + Int(i == j)) / Float64((p + 1) * (p + 2))
                )
                assert_almost_equal(
                    fget(M0, i, j), expected, atol=1e-14 * vol, rtol=0
                )


def test_whitney_needs_geometry() raises:
    var sc = SimplicialComplex(
        nm.fromstring[f64]("[[0,0],[1,0],[0,1]]"),
        nm.fromstring[DType.int]("[[0,1,2]]"),
        compute_geometry=False,
    )
    for k in range(3):
        with assert_raises(contains="without geometry"):
            _ = whitney_innerproduct(sc, k)


def test_whitney_on_thin_simplices() raises:
    # the top-form mass matrix of one simplex is 1 / volume; the Gram-matrix
    # gradients and minors this once used were off by 5% for a triangle of
    # height 1e-8 (and called a slightly thicker one degenerate)
    var h = 1e-8
    var tri: List[List[Float64]] = [
        [0.1, 0.2, 0.3],
        [1.3, 0.7, -0.4],
        [0.7 + 0.3 * h, 0.45 + 0.8 * h, -0.05 + 0.5 * h],
    ]
    var sc = SimplicialComplex(real_array_from_rows(tri), I("[[0,1,2]]"))
    var M = whitney_innerproduct(sc, 2).todense().unsafe_get(0)
    var area = abs(sc[2].primal_volume.unsafe_get(0))
    assert_almost_equal(M * area, 1.0, atol=1e-14, rtol=0)
    var tet: List[List[Float64]] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.3, 0.3, 1e-6],
    ]
    var st = SimplicialComplex(real_array_from_rows(tet), I("[[0,1,2,3]]"))
    var M3 = whitney_innerproduct(st, 3).todense().unsafe_get(0)
    var vol = abs(st[3].primal_volume.unsafe_get(0))
    assert_almost_equal(M3 * vol, 1.0, atol=1e-14, rtol=0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
