from std.math import sqrt
from std.memory import bitcast
from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex, _div6
from mojodec.core.utils import real_array, fget, fset, rows_of, cols_of


def I(s: String) raises -> NDArray[DType.int]:
    return nm.fromstring[DType.int](s)


def R(s: String) raises -> NDArray[f64]:
    return nm.fromstring[f64](s)


def idx(sc: SimplicialComplex, s: List[Int]) raises -> Int:
    return sc[len(s) - 1].simplex_to_index(Simplex(s))


def test_duplicate_and_repeated_vertices_raise() raises:
    # once accepted silently: [[0,1,2],[0,1,2]] had an empty boundary and
    # doubled dual volumes
    var V = R("[[0,0],[1,0],[0,1],[1,1]]")
    with assert_raises(contains="have the same vertices"):
        _ = SimplicialComplex(V, I("[[0,1,2],[1,3,2],[2,0,1]]"))
    with assert_raises(contains="repeats vertex 1"):
        _ = SimplicialComplex(V, I("[[0,1,1]]"), compute_geometry=False)
    with assert_raises(contains="have the same vertices"):
        _ = SimplicialComplex(R("[[0],[1]]"), I("[[0,1],[1,0]]"))


def test_find_vertices() raises:
    var sc = SimplicialComplex(
        nm.fromstring[f64]("[[0,0],[1,0],[0,1],[1,1]]"),
        nm.fromstring[DType.int]("[[0,1,3],[0,3,2]]"),
    )
    for k in range(3):
        for i in range(sc[k].num_simplices):
            var s = sc[k].index_to_simplex(i)
            assert_equal(sc[k].find_vertices(s.vertices), i)
    assert_equal(sc[1].find_vertices([1, 2]), -1)
    assert_equal(sc[1].find_vertices([0, 1, 3]), -1)


def test_d_in_1d() raises:
    var sc = SimplicialComplex(R("[[0],[1]]"), I("[[0,1]]"))
    assert_equal(sc[0].d.nrows, 1)
    assert_equal(sc[0].d.ncols, 2)
    assert_equal(sc[0].d[0, 0], -1.0)
    assert_equal(sc[0].d[0, 1], 1.0)
    sc = SimplicialComplex(R("[[0],[1]]"), I("[[1,0]]"))
    assert_equal(sc[0].d[0, 0], 1.0)
    assert_equal(sc[0].d[0, 1], -1.0)


def test_d_one_triangle() raises:
    for orient in range(2):
        var e = I("[[0,1,2]]") if orient == 0 else I("[[0,2,1]]")
        var sc = SimplicialComplex(R("[[0,0],[1,0],[0,1]]"), e)
        assert_equal(sc[0].d.nrows, 3)
        var e01 = idx(sc, [0, 1])
        var e12 = idx(sc, [1, 2])
        var e02 = idx(sc, [0, 2])
        assert_equal(sc[0].d[e01, 0], -1.0)
        assert_equal(sc[0].d[e01, 1], 1.0)
        assert_equal(sc[0].d[e12, 1], -1.0)
        assert_equal(sc[0].d[e12, 2], 1.0)
        assert_equal(sc[0].d[e02, 0], -1.0)
        assert_equal(sc[0].d[e02, 2], 1.0)
        var s: Float64 = 1.0 if orient == 0 else -1.0
        assert_equal(sc[1].d[0, e01], s)
        assert_equal(sc[1].d[0, e02], -s)
        assert_equal(sc[1].d[0, e12], s)


def test_d_two_triangles() raises:
    var sc = SimplicialComplex(
        R("[[0,0],[1,0],[0,1],[1,1]]"), I("[[1,3,2],[2,0,1]]")
    )
    assert_equal(sc[0].d.nrows, 5)
    assert_equal(sc[1].d.nrows, 2)
    var e01 = idx(sc, [0, 1])
    var e02 = idx(sc, [0, 2])
    var e12 = idx(sc, [1, 2])
    var e13 = idx(sc, [1, 3])
    var e23 = idx(sc, [2, 3])
    assert_equal(sc[0].d[e13, 1], -1.0)
    assert_equal(sc[0].d[e13, 3], 1.0)
    assert_equal(sc[1].d[1, e01], 1.0)
    assert_equal(sc[1].d[1, e12], 1.0)
    assert_equal(sc[1].d[1, e02], -1.0)
    assert_equal(sc[1].d[0, e23], -1.0)
    assert_equal(sc[1].d[0, e12], -1.0)
    assert_equal(sc[1].d[0, e13], 1.0)


def test_d_one_tet() raises:
    var sc = SimplicialComplex(
        R("[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]"), I("[[3,2,1,0]]")
    )
    assert_equal(sc[0].d.nrows, 6)
    assert_equal(sc[1].d.nrows, 4)
    assert_equal(sc[2].d.nrows, 1)
    var e01 = idx(sc, [0, 1])
    var e02 = idx(sc, [0, 2])
    var e12 = idx(sc, [1, 2])
    var e03 = idx(sc, [0, 3])
    var e23 = idx(sc, [2, 3])
    var f123 = idx(sc, [1, 2, 3])
    var f023 = idx(sc, [0, 2, 3])
    var f013 = idx(sc, [0, 1, 3])
    var f012 = idx(sc, [0, 1, 2])
    assert_equal(sc[1].d[f012, e12], 1.0)
    assert_equal(sc[1].d[f012, e02], -1.0)
    assert_equal(sc[1].d[f012, e01], 1.0)
    assert_equal(sc[1].d[f023, e23], 1.0)
    assert_equal(sc[1].d[f023, e03], -1.0)
    assert_equal(sc[2].d[0, f123], 1.0)
    assert_equal(sc[2].d[0, f023], -1.0)
    assert_equal(sc[2].d[0, f013], 1.0)
    assert_equal(sc[2].d[0, f012], -1.0)


def test_dd_is_zero() raises:
    var sc = SimplicialComplex(
        R("[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]]"),
        I("[[0,1,2,3],[1,2,3,4]]"),
    )
    assert_equal(sc[0].d.nrows, 9)
    assert_equal(sc[1].d.nrows, 7)
    assert_equal(sc[2].d.nrows, 2)
    for k in range(2):
        var dd = sc[k + 1].d @ sc[k].d
        assert_equal(dd.max_abs(), 0.0)


def check_star(sc: SimplicialComplex, s: List[Int], ratio: Float64) raises:
    ref data = sc[len(s) - 1]
    var i = data.simplex_to_index(Simplex(s))
    assert_almost_equal(data.star[i, i], ratio, atol=1e-14, rtol=0)


def test_hodge_star_regular() raises:
    var r3 = sqrt(3.0)
    var sc = SimplicialComplex(R("[[0],[1]]"), I("[[0,1]]"))
    for i in range(2):
        assert_almost_equal(sc[0].star[i, i], 0.5, atol=1e-15, rtol=0)
    assert_almost_equal(sc[1].star[0, 0], 1.0, atol=1e-15, rtol=0)
    var V = real_array(3, 2)
    fset(V, 1, 0, 1.0)
    fset(V, 2, 0, 0.5)
    fset(V, 2, 1, r3 / 2)
    sc = SimplicialComplex(V, I("[[0,1,2]]"))
    var primal: List[Float64] = [1.0, 1.0, r3 / 4]
    var dual: List[Float64] = [r3 / 12, r3 / 6, 1.0]
    for dim in range(3):
        for i in range(sc[dim].num_simplices):
            assert_almost_equal(
                sc[dim].star[i, i], dual[dim] / primal[dim], atol=1e-14, rtol=0
            )
    sc = SimplicialComplex(
        R("[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]"), I("[[0,1,2,3]]")
    )
    primal = [1.0, sqrt(2.0), sqrt(3.0 / 4), 1.0 / 3]
    dual = [1.0 / 12, sqrt(2.0) / 12, sqrt(3.0 / 36), 1.0]
    for dim in range(4):
        for i in range(sc[dim].num_simplices):
            assert_almost_equal(
                sc[dim].star[i, i], dual[dim] / primal[dim], atol=1e-14, rtol=0
            )


def test_div6_is_division() raises:
    # _div6 (the dual walk's division of its smallest pieces by 3!) rounds
    # as x / 6 does, for magnitudes in [2^-1000, 2^1000]: random
    # significands at every exponent there, and multiples of 3, whose
    # quotients are exact or lie close to midpoints
    comptime V = SIMD[DType.float64, 4]
    var state: UInt64 = 88172645463325252
    for i in range(250_000):
        var x = V(0)
        for l in range(4):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            if i % 2 == 0:
                var e = UInt64(23 + Int((state >> 52) % 2000))
                var sign = (state & 1) << 63
                x[l] = bitcast[DType.float64](
                    sign | (e << 52) | (state & 0xFFFFFFFFFFFFF)
                )
            else:
                var m = Int((state >> 12) % (1 << 50))
                x[l] = Float64(3 * m + 3) * (0.5 if l % 2 == 0 else -0.25)
        var got = _div6(x)
        var want = x / V(6.0)
        for l in range(4):
            assert_equal(
                bitcast[DType.uint64](got[l]), bitcast[DType.uint64](want[l])
            )


def three_triangles(
    extra_dims: Int, shift: Float64
) raises -> SimplicialComplex:
    var r3 = sqrt(3.0)
    var base: List[List[Float64]] = [
        [0, 0],
        [1, 0],
        [0.5, r3 / 2],
        [1.5, r3 / 2],
        [2, 0],
    ]
    var V = real_array(5, 2 + extra_dims)
    for i in range(5):
        for j in range(2 + extra_dims):
            var x = base[i][j] if j < 2 else 0.0
            fset(V, i, j, x + shift * Float64(j + 1))
    return SimplicialComplex(V, I("[[0,1,2],[1,3,2],[1,4,3]]"))


def test_hodge_star_multi() raises:
    var sc = SimplicialComplex(R("[[0],[1],[2],[3]]"), I("[[0,1],[1,2],[2,3]]"))
    check_star(sc, [0], 0.5)
    check_star(sc, [1], 1.0)
    check_star(sc, [3], 0.5)
    check_star(sc, [2, 3], 1.0)
    var r3 = sqrt(3.0)
    for c in range(2):
        sc = three_triangles(0, 0.0) if c == 0 else three_triangles(4, 0.37)
        check_star(sc, [0], r3 / 12)
        check_star(sc, [1], 3 * r3 / 12)
        check_star(sc, [2], 2 * r3 / 12)
        check_star(sc, [3], 2 * r3 / 12)
        check_star(sc, [4], r3 / 12)
        check_star(sc, [0, 1], r3 / 6)
        check_star(sc, [1, 2], 2 * r3 / 6)
        check_star(sc, [1, 3], 2 * r3 / 6)
        check_star(sc, [3, 4], r3 / 6)
        check_star(sc, [2, 3], r3 / 6)
        check_star(sc, [0, 1, 2], 4 / r3)
        check_star(sc, [1, 4, 3], 4 / r3)


def test_star_star_inv_sign() raises:
    var sc = three_triangles(0, 0.0)
    var n = sc.complex_dimension()
    for k in range(n + 1):
        var P = sc[k].star @ sc[k].star_inv
        var expected: Float64 = -1.0 if (k * (n - k)) % 2 == 1 else 1.0
        for i in range(len(P)):
            assert_almost_equal(P[i], expected, atol=1e-15, rtol=0)


def test_boundary_faces() raises:
    var sc = SimplicialComplex(
        R("[[0,0],[1,0],[0,1],[1,1]]"), I("[[0,1,2],[1,3,2]]")
    )
    var b = sc.boundary()
    assert_equal(len(b), 4)
    assert_true(Simplex([1, 2]) not in b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
