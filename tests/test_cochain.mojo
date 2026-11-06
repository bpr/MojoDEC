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
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.cochain import (
    d,
    star,
    delta,
    laplace_beltrami,
    laplace_derham,
)
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset


def I(s: String) raises -> NDArray[DType.int]:
    return nm.fromstring[DType.int](s)


def R(s: String) raises -> NDArray[f64]:
    return nm.fromstring[f64](s)


def all_cases() raises -> List[SimplicialComplex]:
    var out = List[SimplicialComplex]()
    out.append(SimplicialComplex(R("[[0],[1]]"), I("[[0,1]]")))
    out.append(SimplicialComplex(R("[[1],[0]]"), I("[[1,0]]")))
    out.append(
        SimplicialComplex(
            R("[[0,0],[1,0],[0.5,0.8660254037844386]]"), I("[[0,1,2]]")
        )
    )
    out.append(
        SimplicialComplex(
            R("[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]"), I("[[0,1,2,3]]")
        )
    )
    out.append(
        SimplicialComplex(
            R("[[0,0],[1,0],[1,1],[0,1],[0.5,0.5]]"),
            I("[[0,1,4],[1,2,4],[2,3,4],[0,4,3]]"),
        )
    )
    out.append(SimplicialComplex(R("[[0],[1],[2]]"), I("[[0,1],[1,2]]")))
    out.append(
        SimplicialComplex(R("[[0],[1],[2],[3]]"), I("[[0,1],[1,2],[2,3]]"))
    )
    return out^


def test_three_edges() raises:
    var sc = SimplicialComplex(R("[[0],[1],[2],[3]]"), I("[[0,1],[1,2],[2,3]]"))
    var f0 = sc.get_cochain(0)
    assert_true(f0.complex.is_same(sc))
    assert_equal(f0.k, 0)
    assert_true(f0.is_primal)
    for i in range(4):
        assert_equal(f0[i], 0.0)
        f0[i] = 10.0
    var f1 = d(f0)
    assert_equal(f1.k, 1)
    assert_true(f1.is_primal)
    assert_equal(f1.v.size, 3)
    for i in range(3):
        assert_equal(f1[i], 0.0)


def test_get_set() raises:
    var sc = SimplicialComplex(
        R("[[0,0],[1,0],[0.5,0.8660254037844386]]"), I("[[0,1,2]]")
    )
    var c0 = sc.get_cochain(0)
    c0[Simplex([0], 0)] = 10.0
    c0[Simplex([1], 0)] = 20.0
    c0[Simplex([2], 1)] = 30.0
    assert_equal(c0[Simplex([0], 0)], 10.0)
    assert_equal(c0[Simplex([1], 1)], -20.0)
    assert_equal(c0[Simplex([2], 1)], 30.0)
    assert_equal(c0[0], 10.0)
    assert_equal(c0[1], 20.0)
    assert_equal(c0[2], -30.0)


def test_kinds_must_match() raises:
    """Arithmetic and indexing reject cochains and simplices of another
    kind, instead of reading past the end of the values."""
    var sc = SimplicialComplex(
        R("[[0,0],[1,0],[1,1],[0,1]]"), I("[[0,1,3],[1,2,3]]")
    )
    var p0 = sc.get_cochain(0)  # 4 vertices
    var d0 = sc.get_cochain(0, False)  # 2 triangles
    with assert_raises(contains="primal/dual"):
        _ = p0 + d0
    with assert_raises(contains="primal/dual"):
        _ = d0 - p0
    # the same size (5 edges each way) is still a different kind
    with assert_raises(contains="primal/dual"):
        _ = sc.get_cochain(1) + sc.get_cochain(1, False)
    with assert_raises(contains="primal/dual"):
        p0.overwrite(Subset.full(sc, 0), d0)
    with assert_raises(contains="primal/dual"):
        _ = sc.get_cochain_basis(1) + sc.get_cochain_basis(1, False)
    # a dual 0-cochain is indexed by triangles, a primal one by vertices
    d0[Simplex([0, 1, 3])] = 2.0
    assert_equal(d0[0], 2.0)
    assert_equal(d0[Simplex([0, 1, 3])], 2.0)
    with assert_raises(contains="expected a 2-simplex"):
        _ = d0[Simplex([0])]
    with assert_raises(contains="expected a 0-simplex"):
        p0[Simplex([0, 1])] = 1.0
    with assert_raises(contains="expected a 0-simplex"):
        _ = p0[Simplex([0, 1, 3])]


def test_d() raises:
    for ref sc in all_cases():
        for p in range(sc.complex_dimension()):
            var df = d(sc.get_cochain_basis(p))
            assert_equal((df.v - sc[p].d).max_abs(), 0.0)
            assert_equal(d(df).v.max_abs(), 0.0)


def test_dual_d() raises:
    for ref sc in all_cases():
        var N = sc.complex_dimension()
        for p in range(N):
            var df = d(sc.get_cochain_basis(p, is_primal=False))
            var sign: Float64 = -1.0 if p % 2 == 1 else 1.0
            assert_equal((df.v - sc[N - p].boundary * sign).max_abs(), 0.0)
            assert_equal(d(df).v.max_abs(), 0.0)


def test_star() raises:
    for ref sc in all_cases():
        var N = sc.complex_dimension()
        for p in range(N):
            var wellcentered = True
            for i in range(sc[p].num_simplices):
                if sc[p].dual_volume.unsafe_get(i) <= 0:
                    wellcentered = False
            if not wellcentered:
                continue
            var result = star(star(sc.get_cochain_basis(p))).v.copy()
            var sign: Float64 = -1.0 if (p * (N - p)) % 2 == 1 else 1.0
            var diff = result - CSRMatrix.identity(sc[p].num_simplices) * sign
            assert_almost_equal(diff.max_abs(), 0.0)


def test_laplacian() raises:
    for ref sc in all_cases():
        var f = sc.get_cochain_basis(0)
        assert_almost_equal(
            (laplace_beltrami(f) - laplace_derham(f)).v.max_abs(), 0.0
        )


def test_basis_matches_vector() raises:
    var sc = all_cases()[3]
    var L = laplace_derham(sc.get_cochain_basis(1)).v.copy()
    var c = sc.get_cochain(1)
    for i in range(c.v.size):
        c[i] = Float64(i + 1)
    var Lc = laplace_derham(c)
    var Lv = L @ c.v
    for i in range(c.v.size):
        assert_almost_equal(Lc[i], Lv.unsafe_get(i))
    var h = delta(c)
    assert_equal(h.k, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
