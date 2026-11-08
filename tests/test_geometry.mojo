from std.math import sqrt

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_true,
    assert_equal,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.utils import fget, real_vector
from mojodec.routines.geometry import (
    unsigned_volume,
    signed_volume,
    circumcenter,
    circumcenter_barycentric,
    is_wellcentered,
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
    assert_almost_equal(unsigned_volume(A("[[1]]")), 1.0)
    assert_almost_equal(unsigned_volume(A("[[1],[10]]")), 9.0)
    assert_almost_equal(unsigned_volume(A("[[0,0],[1,1]]")), 2.0**0.5)
    assert_almost_equal(unsigned_volume(A("[[0,0],[0,1],[1,0]]")), 0.5)
    assert_almost_equal(unsigned_volume(A("[[5,5],[5,6],[6,5]]")), 0.5)
    assert_almost_equal(unsigned_volume(A("[[0,0,0],[0,0,1],[0,1,0]]")), 0.5)


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
        unsigned_volume(A("[[0,0,0],[3,0,0],[0,4,0]]")), 6.0, atol=1e-15
    )
    assert_almost_equal(
        unsigned_volume(A("[[1,1,1],[1,4,1],[1,1,5]]")), 6.0, atol=1e-15
    )
    assert_almost_equal(
        unsigned_volume(A("[[0,0,0],[1,1,0],[0,1,1]]")),
        0.5 * sqrt(3.0),
        atol=1e-15,
    )
    assert_almost_equal(
        unsigned_volume(A("[[0,0,0,7],[1,0,0,7],[0,1,0,7],[0,0,1,7]]")),
        1.0 / 6.0,
        atol=1e-15,
    )


def test_signed_volume() raises:
    assert_almost_equal(signed_volume(A("[[1],[2]]")), 1.0)
    assert_almost_equal(signed_volume(A("[[5.5],[-10]]")), -15.5)
    assert_almost_equal(signed_volume(A("[[0,0],[1,1],[1,0]]")), -0.5)
    assert_almost_equal(signed_volume(A("[[0,0],[1,0],[1,1]]")), 0.5)
    assert_almost_equal(signed_volume(A("[[0,0],[0,1],[1,0]]")), -0.5)
    assert_almost_equal(signed_volume(A("[[5,5],[5,6],[6,5]]")), -0.5)
    assert_almost_equal(
        signed_volume(A("[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]")), 1.0 / 6.0
    )


def test_volumes_agree() raises:
    nm.random.seed(0)
    for N in range(1, 10):
        var pts = nm.random.rand[f64](Shape(N + 1, N))
        assert_almost_equal(abs(signed_volume(pts)), unsigned_volume(pts))


def test_is_wellcentered() raises:
    assert_true(is_wellcentered(A("[[1]]")))
    assert_true(is_wellcentered(A("[[0,1]]")))
    assert_true(is_wellcentered(A("[[0],[1]]")))
    assert_true(is_wellcentered(A("[[0,1],[1,0]]")))
    assert_true(is_wellcentered(A("[[0,0],[1,0],[0.5,0.8660254037844386]]")))


def test_circumcenter_examples() raises:
    var b = circumcenter_barycentric(A("[[0,0],[4,0],[0,4]]"))
    assert_almost_equal(b.unsafe_get(0), 0.0)
    assert_almost_equal(b.unsafe_get(1), 0.5)
    assert_almost_equal(b.unsafe_get(2), 0.5)
    var cr = circumcenter(A("[[0,0],[1,0],[0,1]]"))
    assert_almost_equal(cr.center.unsafe_get(0), 0.5)
    assert_almost_equal(cr.center.unsafe_get(1), 0.5)
    assert_almost_equal(cr.radius, 0.5**0.5)


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
            assert_almost_equal(lo, hi, atol=1e-6)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
