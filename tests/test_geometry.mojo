from std.math import sqrt

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_equal,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import (
    fget,
    int_array,
    real_array_from_rows,
    real_vector,
)
from mojodec.routines.geometry import (
    unsigned_volume,
    signed_volume,
    circumcenter,
    circumcenter_barycentric,
    is_wellcentered,
    barycentric_gradients,
)
from mojodec.routines.combinatorial import (
    combinations,
    permutations,
    permutation_parity,
    relative_parity,
)


def A(s: String) raises -> NDArray[f64]:
    return nm.fromstring[f64](s)


def test_unsigned_volume() raises:
    assert_equal(unsigned_volume(A("[[1]]")), 1.0)
    assert_almost_equal(
        unsigned_volume(A("[[1],[10]]")), 9.0, atol=1e-14, rtol=0
    )
    # sqrt, not 2.0**0.5, which is off by ~3e-13 (see AGENTS.md)
    assert_almost_equal(
        unsigned_volume(A("[[0,0],[1,1]]")), sqrt(2.0), atol=1e-15, rtol=0
    )
    assert_almost_equal(
        unsigned_volume(A("[[0,0],[0,1],[1,0]]")), 0.5, atol=1e-15, rtol=0
    )
    assert_almost_equal(
        unsigned_volume(A("[[5,5],[5,6],[6,5]]")), 0.5, atol=1e-15, rtol=0
    )
    assert_almost_equal(
        unsigned_volume(A("[[0,0,0],[0,0,1],[0,1,0]]")),
        0.5,
        atol=1e-15,
        rtol=0,
    )


def test_unsigned_volume_of_degenerate_simplices() raises:
    # Coplanar points have zero volume in every vertex order; the Gram
    # determinant formula used to give ~7e-9 for the second order.
    assert_true(
        unsigned_volume(A("[[0,0,-2],[0,0,0],[0,1,0],[0,0,1]]")) < 1e-12
    )
    assert_true(
        unsigned_volume(A("[[0,0,0],[0,1,0],[0,0,1],[0,0,-2]]")) < 1e-12
    )
    # The same holds for a flat simplex embedded in more dimensions; the
    # Gram determinant formula gave 3.7e-9 for this flat tetrahedron in 4-D.
    assert_true(
        unsigned_volume(A("[[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,3,0]]")) < 1e-12
    )
    assert_true(unsigned_volume(A("[[1,2,3],[2,4,6],[4,8,12]]")) < 1e-12)


def test_unsigned_volume_of_embedded_simplices() raises:
    # a right triangle with legs 3 and 4, and a unit tetrahedron, tilted
    # out of their coordinate planes
    assert_almost_equal(
        unsigned_volume(A("[[0,0,0],[3,0,0],[0,4,0]]")), 6.0, atol=1e-15, rtol=0
    )
    assert_almost_equal(
        unsigned_volume(A("[[1,1,1],[1,4,1],[1,1,5]]")), 6.0, atol=1e-15, rtol=0
    )
    assert_almost_equal(
        unsigned_volume(A("[[0,0,0],[1,1,0],[0,1,1]]")),
        0.5 * sqrt(3.0),
        atol=1e-15,
        rtol=0,
    )
    assert_almost_equal(
        unsigned_volume(A("[[0,0,0,7],[1,0,0,7],[0,1,0,7],[0,0,1,7]]")),
        1.0 / 6.0,
        atol=1e-15,
        rtol=0,
    )


def test_signed_volume() raises:
    comptime tol = 1e-15  # small integer coordinates: rounding only
    assert_almost_equal(signed_volume(A("[[1],[2]]")), 1.0, atol=tol, rtol=0)
    assert_almost_equal(
        signed_volume(A("[[5.5],[-10]]")), -15.5, atol=16 * tol, rtol=0
    )
    assert_almost_equal(
        signed_volume(A("[[0,0],[1,1],[1,0]]")), -0.5, atol=tol, rtol=0
    )
    assert_almost_equal(
        signed_volume(A("[[0,0],[1,0],[1,1]]")), 0.5, atol=tol, rtol=0
    )
    assert_almost_equal(
        signed_volume(A("[[0,0],[0,1],[1,0]]")), -0.5, atol=tol, rtol=0
    )
    assert_almost_equal(
        signed_volume(A("[[5,5],[5,6],[6,5]]")), -0.5, atol=tol, rtol=0
    )
    assert_almost_equal(
        signed_volume(A("[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]")),
        1.0 / 6.0,
        atol=tol,
        rtol=0,
    )


def test_volumes_agree() raises:
    nm.random.seed(0)
    for N in range(1, 10):
        var pts = nm.random.rand[f64](Shape(N + 1, N))
        assert_almost_equal(
            abs(signed_volume(pts)), unsigned_volume(pts), atol=1e-15, rtol=0
        )


def test_is_wellcentered() raises:
    assert_true(is_wellcentered(A("[[1]]")))
    assert_true(is_wellcentered(A("[[0,1]]")))
    assert_true(is_wellcentered(A("[[0],[1]]")))
    assert_true(is_wellcentered(A("[[0,1],[1,0]]")))
    assert_true(is_wellcentered(A("[[0,0],[1,0],[0.5,0.8660254037844386]]")))


def test_circumcenter_examples() raises:
    var b = circumcenter_barycentric(A("[[0,0],[4,0],[0,4]]"))
    assert_almost_equal(b.unsafe_get(0), 0.0, atol=1e-15, rtol=0)
    assert_almost_equal(b.unsafe_get(1), 0.5, atol=1e-15, rtol=0)
    assert_almost_equal(b.unsafe_get(2), 0.5, atol=1e-15, rtol=0)
    var cr = circumcenter(A("[[0,0],[1,0],[0,1]]"))
    assert_almost_equal(cr.center.unsafe_get(0), 0.5, atol=1e-15, rtol=0)
    assert_almost_equal(cr.center.unsafe_get(1), 0.5, atol=1e-15, rtol=0)
    assert_almost_equal(cr.radius, sqrt(0.5), atol=1e-15, rtol=0)


def test_circumcenters_far_from_the_origin() raises:
    # Barycentric circumcenters don't change under translation. Solving
    # PyDEC's system with absolute positions lost about |p|^2 / edge^2 of
    # precision (an error of 6e-8 for this tetrahedron of size 0.1 at
    # distance 1000, 3e-5 at distance 3333 and size 0.01); positions
    # relative to the first vertex keep it to the rounding of the input
    # coordinates (~1e-12 at this size and distance).
    var near: List[List[Float64]] = [
        [0.0, 0.03, -0.02],
        [0.1, 0.01, 0.0],
        [0.02, 0.1, 0.005],
        [0.0, 0.03, 0.1],
    ]
    var far = List[List[Float64]]()
    for p in near:
        far.append([p[0] + 1000, p[1] + 1000, p[2] + 1000])
    var b0 = circumcenter_barycentric(real_array_from_rows(near))
    var b1 = circumcenter_barycentric(real_array_from_rows(far))
    var c0 = circumcenter(real_array_from_rows(near))
    var c1 = circumcenter(real_array_from_rows(far))
    var sc = SimplicialComplex(
        real_array_from_rows(far), nm.fromstring[DType.int]("[[0,1,2,3]]")
    )
    for i in range(4):
        assert_almost_equal(
            b1.unsafe_get(i), b0.unsafe_get(i), atol=2e-12, rtol=0
        )
        assert_almost_equal(
            fget(sc[3].bary_circumcenter, 0, i),
            b0.unsafe_get(i),
            atol=2e-12,
            rtol=0,
        )
    for k in range(3):
        var x = c0.center.unsafe_get(k) + 1000
        assert_almost_equal(c1.center.unsafe_get(k), x, atol=1e-11, rtol=0)
        assert_almost_equal(
            fget(sc[3].circumcenter, 0, k), x, atol=1e-11, rtol=0
        )


def test_circumcenter_random() raises:
    nm.random.seed(0)
    for N in range(1, 10):
        for M in range(1, N + 1):
            var pts = nm.random.rand[f64](Shape(M, N))
            var cr = circumcenter(pts)
            var lo: Float64 = 1e300
            var hi: Float64 = -1e300
            for i in range(M):
                var d2: Float64 = 0
                for k in range(N):
                    d2 += (fget(pts, i, k) - cr.center.unsafe_get(k)) ** 2
                lo = min(lo, d2**0.5)
                hi = max(hi, d2**0.5)
            assert_almost_equal(lo, hi, atol=1e-13, rtol=0)


def test_combinatorial() raises:
    var c = combinations([1, 2, 3, 4], 2)
    assert_equal(len(c), 6)
    assert_true(c[0] == [1, 2] and c[5] == [3, 4])
    assert_equal(len(combinations([1, 2, 3, 4], 0)), 1)
    assert_equal(len(combinations([1, 2, 3, 4], 5)), 1)
    var p = permutations([1, 2, 3])
    assert_true(p[0] == [1, 2, 3] and p[1] == [2, 1, 3] and p[2] == [2, 3, 1])
    assert_true(p[3] == [1, 3, 2] and p[4] == [3, 1, 2] and p[5] == [3, 2, 1])
    assert_equal(permutation_parity([0, 1, 2]), 0)
    assert_equal(permutation_parity([0, 2, 1]), 1)
    assert_equal(permutation_parity([1, 0, 2]), 1)
    assert_equal(permutation_parity([1, 2, 0]), 0)
    assert_equal(permutation_parity([2, 0, 1]), 0)
    assert_equal(permutation_parity([0, 1, 3, 2]), 1)
    assert_equal(relative_parity([0, 1], [1, 0]), 1)
    assert_equal(relative_parity([0, 1, 2], [0, 2, 1]), 1)


def test_negative_combination_sizes() raises:
    # n < 0 once indexed past the end of L
    assert_equal(len(combinations([1, 2], -1)), 1)
    assert_equal(len(combinations([1, 2], -1)[0]), 0)


def test_thin_and_sliver_triangles() raises:
    # a triangle 1e-9 high in 3-D: the complex's own volumes once used the
    # Gram determinant and got 0
    var V = real_array_from_rows(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.000000001, 0.0]]
    )
    var T = int_array(1, 3)
    for j in range(3):
        T.unsafe_set(j, Scalar[DType.int](j))
    var sc = SimplicialComplex(V, T)
    assert_almost_equal(
        sc[2].primal_volume.unsafe_get(0), 5e-10, atol=0, rtol=1e-6
    )
    # an obtuse sliver 1e-8 high: its circumcenter weights' sum cancelled to
    # 0, so it was rejected as degenerate
    var W = real_array_from_rows([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.00000001]])
    var flat = SimplicialComplex(W, T)
    ref c = flat[2].circumcenter
    var r0 = sqrt(fget(c, 0, 0) ** 2 + fget(c, 0, 1) ** 2)
    for i in range(1, 3):
        var dx = fget(c, 0, 0) - fget(W, i, 0)
        var dy = fget(c, 0, 1) - fget(W, i, 1)
        assert_almost_equal(sqrt(dx * dx + dy * dy), r0, atol=0, rtol=1e-6)


def test_thin_triangle_gradients() raises:
    # the Gram-matrix solve gave grad . edge = 3.6 instead of 1 at h = 1e-8
    var h = 1e-8
    var pts: List[List[Float64]] = [
        [0.1, 0.2, 0.3],
        [1.3, 0.7, -0.4],
        [0.7 + 0.3 * h, 0.45 + 0.8 * h, -0.05 + 0.5 * h],
    ]
    var G = barycentric_gradients(real_array_from_rows(pts))
    for i in range(1, 3):
        for j in range(1, 3):
            var s = 0.0
            for k in range(3):
                s += fget(G, i, k) * (pts[j][k] - pts[0][k])
            var want = 1.0 if i == j else 0.0
            assert_true(abs(s - want) < 1e-7, String(i, " ", j, ": ", s))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
