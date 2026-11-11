from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from std.math import sqrt
import numojo as nm
from numojo.prelude import *

from mojodec import (
    SimplicialComplex,
    grundmann_moeller,
    integrate_simplices,
    kuhn_grid,
)
from mojodec.core.utils import fget, fset, real_array

comptime _EPS = 2.220446049250313e-16


def _monomial_integral(n: Int, alpha: List[Int]) -> Float64:
    """`n! prod alpha_j! / (n + |alpha|)!`: the mean of `prod lambda_j^alpha_j`
    over the n-simplex, as an alternating product of small integers."""
    var up = List[Float64]()
    var down = List[Float64]()
    var total = 0
    for j in range(2, n + 1):
        up.append(Float64(j))
    for a in alpha:
        total += a
        for j in range(2, a + 1):
            up.append(Float64(j))
    for j in range(2, n + total + 1):
        down.append(Float64(j))
    var r: Float64 = 1
    var i = 0
    var k = 0
    while i < len(up) or k < len(down):
        if k < len(down) and (r >= 1 or i == len(up)):
            r /= down[k]
            k += 1
        else:
            r *= up[i]
            i += 1
    return r


def _next(mut alpha: List[Int], top: Int) -> Bool:
    """The next exponent vector with entries in 0..top (odometer)."""
    for j in range(len(alpha)):
        alpha[j] += 1
        if alpha[j] <= top:
            return True
        alpha[j] = 0
    return False


def test_rules() raises:
    for n in range(0, 5):
        for degree in range(0, 22):
            var r = grundmann_moeller(n, degree)
            # odd degrees only: an even request gets the next one
            assert_equal(r.degree, degree + 1 - degree % 2)
            assert_equal(r.dim, n)
            var s = degree // 2
            var count = 0
            for i in range(s + 1):
                var b = 1
                for j in range(n):
                    b = b * (s - i + n - j) // (j + 1)
                count += b
            assert_equal(len(r), count)
            var total: Float64 = 0
            for q in range(len(r)):
                total += r.weights.unsafe_get(q)
                var row: Float64 = 0
                for j in range(n + 1):
                    var x = r.points.unsafe_get(q * (n + 1) + j)
                    assert_true(x > 0)  # interior nodes
                    row += x
                assert_almost_equal(row, 1.0, atol=1e-15, rtol=0)
            # the weights' own rounding, amplified by cancellation
            assert_almost_equal(
                total, 1.0, atol=64 * _EPS * r.abs_weight_sum(), rtol=0
            )
    # the midpoint rule, then negative weights
    var mid = grundmann_moeller(2, 1)
    assert_equal(len(mid), 1)
    assert_almost_equal(mid.points.unsafe_get(0), 1.0 / 3.0, atol=1e-16, rtol=0)
    assert_true(grundmann_moeller(2, 3).abs_weight_sum() > 1)


def test_exactness() raises:
    # every monomial in the barycentric coordinates up to the degree is
    # integrated exactly (to rounding); degree + 1 is not
    for n in range(1, 5):
        for degree in [1, 3, 5, 7, 9]:
            var r = grundmann_moeller(n, degree)
            var tol = 16 * _EPS * r.abs_weight_sum()
            var alpha = List[Int](length=n + 1, fill=0)
            while True:
                var total = 0
                for a in alpha:
                    total += a
                if total <= degree + 1:
                    var q: Float64 = 0
                    for k in range(len(r)):
                        var p: Float64 = 1
                        for j in range(n + 1):
                            for _ in range(alpha[j]):
                                p *= r.points.unsafe_get(k * (n + 1) + j)
                        q += r.weights.unsafe_get(k) * p
                    var exact = _monomial_integral(n, alpha)
                    if total <= degree:
                        assert_almost_equal(q, exact, atol=tol * exact, rtol=0)
                    elif alpha[0] == total:
                        # lambda_0^(d+1): off by at least 5e-5 relative
                        assert_true(abs(q - exact) > 5e-5 * exact)
                if not _next(alpha, degree + 1):
                    break


def _x2y(x: List[Float64]) -> Float64:
    return x[0] * x[0] * x[1]


def _one(x: List[Float64]) -> Float64:
    return 1.0


def test_integrate_on_a_simplex() raises:
    var tri = real_array(3, 2)
    fset(tri, 1, 0, 1.0)
    fset(tri, 2, 1, 1.0)
    # int x^2 y over the unit right triangle = 2! 1! / 5! = 1/60
    var r = grundmann_moeller(2, 3)
    assert_almost_equal(r.integrate(tri, _x2y), 1.0 / 60.0, atol=1e-16, rtol=0)
    # an embedded triangle: the area, from the unsigned volume
    var t3 = real_array(3, 3)
    fset(t3, 1, 0, 2.0)
    fset(t3, 2, 1, 1.0)
    fset(t3, 2, 2, 1.0)
    var area = grundmann_moeller(2, 1).integrate(t3, _one)
    # edges (2, 0, 0) and (0, 1, 1): area |cross| / 2 = sqrt(2)
    assert_almost_equal(area, sqrt(2.0), atol=1e-15, rtol=0)
    # a closure, capturing a coefficient
    var c = 3.0

    def scaled(x: List[Float64]) raises {c} -> Float64:
        return c * x[0]

    # int x over the triangle = 1/6
    assert_almost_equal(
        grundmann_moeller(2, 1).integrate(tri, scaled), 0.5, atol=1e-16, rtol=0
    )
    with assert_raises(contains="expected 3 vertices"):
        _ = r.integrate(real_array(4, 2), _x2y)


def _xyz(x: List[Float64]) -> Float64:
    return x[0] * x[0] * x[1] * x[2] * x[2] * x[2]


def test_integrate_simplices() raises:
    var sc = SimplicialComplex(kuhn_grid([3, 3, 3]))
    # int_[0,1]^3 x^2 y z^3 = 1/3 * 1/2 * 1/4, by a rule of degree 7
    var cells = integrate_simplices(sc, 3, 6, _xyz)
    var total: Float64 = 0
    for i in range(cells.size):
        total += cells.unsafe_get(i)
    assert_almost_equal(total, 1.0 / 24.0, atol=1e-15, rtol=0)
    # 1 over edges: their lengths; 0-simplices: the values
    var lengths = integrate_simplices(sc, 1, 0, _one)
    for e in range(lengths.size):
        assert_almost_equal(
            lengths.unsafe_get(e),
            sc[1].primal_volume.unsafe_get(e),
            atol=1e-15,
            rtol=0,
        )
    var values = integrate_simplices(sc, 0, 5, _xyz)
    for v in range(values.size):
        var x = fget(sc.vertices(), v, 0)
        var y = fget(sc.vertices(), v, 1)
        var z = fget(sc.vertices(), v, 2)
        assert_almost_equal(
            values.unsafe_get(v), x * x * y * z * z * z, atol=1e-15, rtol=0
        )


def test_errors() raises:
    with assert_raises(contains="negative"):
        _ = grundmann_moeller(-1, 3)
    with assert_raises(contains="outside 0..21"):
        _ = grundmann_moeller(2, 22)
    with assert_raises(contains="outside 0..21"):
        _ = grundmann_moeller(2, -1)
    var sc = SimplicialComplex(kuhn_grid([2, 2]))
    with assert_raises(contains="k = 3 is outside 0..2"):
        _ = integrate_simplices(sc, 3, 1, _one)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
