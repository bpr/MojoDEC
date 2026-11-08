from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_almost_equal,
    assert_raises,
)
from std.collections import Set

import numojo as nm
from numojo.prelude import *

from mojodec.core.abstract_simplicial_complex import AbstractSimplicialComplex
from mojodec.core.kd_tree import KDTree
from mojodec.core.rips_complex import (
    RipsComplex,
    rips_simplices,
    rips_chain_complex,
)
from mojodec.core.chain_complex import ChainComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    iget,
    rows_of,
    cols_of,
    real_row_list,
    fget,
    int_array,
    lexsort_rows,
)


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


def asc_of(var arrays: List[IntArray]) raises -> AbstractSimplicialComplex:
    return AbstractSimplicialComplex(arrays)


def test_asc_1d() raises:
    var asc = asc_of([I("[[0,1],[2,3]]")])
    assert_int_array(asc.simplices[0], "[[0],[1],[2],[3]]")
    assert_int_array(asc.simplices[1], "[[0,1],[2,3]]")
    assert_equal(asc.chain_complex().num_cells(0), 4)
    assert_dense(asc.chain_complex().boundary(1), "[[-1,0],[1,0],[0,-1],[0,1]]")
    asc = asc_of([I("[[0],[2]]"), I("[[0,1],[2,3]]")])
    assert_int_array(asc.simplices[0], "[[0],[1],[2],[3]]")
    assert_dense(asc.chain_complex().boundary(1), "[[-1,0],[1,0],[0,-1],[0,1]]")
    asc = asc_of([I("[[2],[5]]"), I("[[0,1],[3,4]]")])
    assert_int_array(asc.simplices[0], "[[0],[1],[2],[3],[4],[5]]")
    assert_equal(asc.chain_complex().num_cells(0), 6)
    assert_dense(
        asc.chain_complex().boundary(1),
        "[[-1,0],[1,0],[0,0],[0,-1],[0,1],[0,0]]",
    )


def test_asc_2d() raises:
    var asc = asc_of([I("[[0,1,2]]")])
    assert_int_array(asc.simplices[1], "[[0,1],[0,2],[1,2]]")
    assert_dense(
        asc.chain_complex().boundary(1), "[[-1,-1,0],[1,0,-1],[0,1,1]]"
    )
    assert_dense(asc.chain_complex().boundary(2), "[[1],[-1],[1]]")
    asc = asc_of([I("[[0,2]]"), I("[[0,1,2]]")])
    assert_int_array(asc.simplices[1], "[[0,1],[0,2],[1,2]]")
    assert_dense(asc.chain_complex().boundary(2), "[[1],[-1],[1]]")
    asc = asc_of([I("[[0,3],[4,5]]"), I("[[0,1,2]]")])
    assert_int_array(asc.simplices[0], "[[0],[1],[2],[3],[4],[5]]")
    assert_int_array(asc.simplices[1], "[[0,1],[0,2],[0,3],[1,2],[4,5]]")
    assert_dense(
        asc.chain_complex().boundary(1),
        "[[-1,-1,-1,0,0],[1,0,0,-1,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,-1],[0,0,0,0,1]]",
    )
    assert_dense(asc.chain_complex().boundary(2), "[[1],[-1],[0],[1],[0]]")


def test_asc_rejects_duplicate_simplices() raises:
    with assert_raises(contains="have the same vertices"):
        _ = asc_of([I("[[0,1,2],[2,1,0]]")])
    with assert_raises(contains="repeats vertex 0"):
        _ = asc_of([I("[[0,0]]")])
    # repeated lower-dimensional simplices are merged, as before
    var asc = asc_of([I("[[0,1,2]]"), I("[[0,1],[1,0]]")])
    assert_equal(rows_of(asc.simplices[1]), 3)


def test_rips_simplices() raises:
    var r = rips_simplices(3, I("[[0,1],[0,2],[1,2]]"), 2)
    assert_int_array(r[1], "[[0,1],[0,2],[1,2]]")
    assert_int_array(r[2], "[[0,1,2]]")
    r = rips_simplices(
        6, I("[[0,1],[0,2],[0,3],[0,4],[0,5],[1,2],[1,5],[2,5],[3,4]]"), 3
    )
    assert_int_array(r[0], "[[0],[1],[2],[3],[4],[5]]")
    assert_int_array(r[2], "[[0,1,2],[0,1,5],[0,2,5],[0,3,4],[1,2,5]]")
    assert_int_array(r[3], "[[0,1,2,5]]")


def test_rips_chain_complex() raises:
    var r = rips_chain_complex([I("[[0],[1],[2]]"), I("[[0,2]]")])
    assert_equal(r.num_cells(0), 3)
    assert_dense(r.boundary(1), "[[-1],[0],[1]]")
    r = rips_chain_complex(
        [
            I("[[0],[1],[2],[3],[4]]"),
            I("[[0,1],[0,2],[0,3],[1,2],[2,3],[2,4]]"),
            I("[[0,1,2],[0,2,3]]"),
        ]
    )
    assert_dense(
        r.boundary(1),
        "[[-1,-1,-1,0,0,0],[1,0,0,-1,0,0],[0,1,0,1,-1,-1],[0,0,1,0,1,0],[0,0,0,0,0,1]]",
    )
    assert_dense(r.boundary(2), "[[1,0],[-1,1],[0,-1],[1,0],[0,1],[0,0]]")


def test_rips_complex_example() raises:
    var rc = RipsComplex(
        nm.fromstring[f64]("[[0,0],[2,0],[2,2],[0,2],[1,1]]"), 2.1
    )
    assert_int_array(
        rc.simplices[1], "[[0,1],[0,3],[0,4],[1,2],[1,4],[2,3],[2,4],[3,4]]"
    )
    assert_int_array(rc.simplices[2], "[[0,1,4],[0,3,4],[1,2,4],[2,3,4]]")


def test_rips_without_edges() raises:
    # points too far apart for any edge: a 0-dimensional complex
    var rc = RipsComplex(nm.fromstring[f64]("[[0,0],[5,0],[0,5]]"), 1.0)
    assert_equal(len(rc.simplices), 1)
    assert_equal(rc.complex_dimension(), 0)
    ref cc = rc.chain_complex()
    assert_equal(len(cc), 1)
    assert_equal(cc.num_cells(0), 3)
    assert_true(rc.boundary_cells().is_empty())
    assert_equal(len(rc.interior_cells()[0]), 3)


def check_exactness(cc: ChainComplex) raises:
    var n = cc.complex_dimension()
    for k in range(n):
        assert_equal((cc.coboundary(k).T() - cc.boundary(k + 1)).max_abs(), 0.0)
    for k in range(n - 1):
        assert_equal((cc.coboundary(k + 1) @ cc.coboundary(k)).max_abs(), 0.0)
        assert_equal((cc.boundary(k + 1) @ cc.boundary(k + 2)).max_abs(), 0.0)


def test_rips_random() raises:
    nm.random.seed(0)
    for dim in range(2, 4):
        var N = 200
        var R = 0.2 if dim == 2 else 0.25
        var pts = nm.random.rand[f64](Shape(N, dim))
        var rc = RipsComplex(pts, R)
        var edges = Set[Int]()
        for r in range(rows_of(rc.simplices[1])):
            edges.add(
                iget(rc.simplices[1], r, 0) * N + iget(rc.simplices[1], r, 1)
            )
        for i in range(N):
            for j in range(i + 1, N):
                var d2: Float64 = 0
                for c in range(dim):
                    d2 += (fget(pts, i, c) - fget(pts, j, c)) ** 2
                assert_equal(d2**0.5 < R, (i * N + j) in edges)
        for lvl in range(2, len(rc.simplices)):
            ref s = rc.simplices[lvl]
            for r in range(rows_of(s)):
                for a in range(cols_of(s)):
                    for b in range(a + 1, cols_of(s)):
                        assert_true(
                            (iget(s, r, a) * N + iget(s, r, b)) in edges
                        )
        check_exactness(rc.chain_complex())


def test_kd_tree() raises:
    var kdt = KDTree(nm.fromstring[f64]("[[0,0],[1,0],[1,1],[0,1],[0.5,0.5]]"))
    assert_equal(kdt.nearest([2.0, 2.0]).value(), 2)
    assert_equal(kdt.nearest([0.1, 0.2]).value(), 0)
    assert_equal(kdt.nearest([0.9, 0.0]).value(), 1)
    assert_equal(kdt.nearest([0.4, 0.6]).value(), 4)
    nm.random.seed(0)
    for dim in range(1, 5):
        var sizes: List[Int] = [1, 2, 5, 10, 50]
        for num_pts in sizes:
            var pts = nm.random.rand[f64](Shape(num_pts, dim)) * 20.0 - 10.0
            var tree = KDTree(pts)
            for _ in range(5):
                var sample = List[Float64]()
                for _ in range(dim):
                    sample.append(
                        Float64(nm.random.rand[f64](Shape(1)).unsafe_get(0))
                        * 20
                        - 10
                    )
                var dist = List[Float64]()
                for i in range(num_pts):
                    var d2: Float64 = 0
                    for c in range(dim):
                        d2 += (fget(pts, i, c) - sample[c]) ** 2
                    dist.append(d2)
                var order = List[Int]()
                for i in range(num_pts):
                    order.append(i)
                for i in range(1, num_pts):
                    var j = i
                    while j > 0 and dist[order[j]] < dist[order[j - 1]]:
                        order.swap_elements(j, j - 1)
                        j -= 1
                assert_equal(tree.nearest(sample).value(), order[0])
                var ns: List[Int] = [1, 2, 5]
                for n in ns:
                    var got = tree.nearest_n(sample, n)
                    for t in range(min(n, num_pts)):
                        assert_equal(got[t], order[t])


def test_kd_tree_and_sort_edge_cases() raises:
    var tree = KDTree(nm.array[f64]("[[0], [0.1], [0.2]]"))
    assert_equal(len(tree.in_sphere([0.05], 1.0, 0)), 0)  # once crashed
    assert_equal(len(tree.nearest_n([0.05], -1)), 0)
    with assert_raises(contains="coordinate"):
        _ = KDTree(nm.zeros[DType.float64](Shape(3, 0)))
    # a query point of the wrong length was read past its end
    with assert_raises(contains="query point has 2 coordinates"):
        _ = tree.nearest([0.0, 1.0])
    with assert_raises(contains="query point has 0 coordinates"):
        _ = tree.in_sphere(List[Float64](), 1.0)
    # an abstract complex needs at least one top simplex
    with assert_raises(contains="no rows"):
        _ = asc_of([int_array(0, 3)])
    # entries 1.2e19 apart once overflowed the radix sort's span
    var a = int_array(3, 1)
    a.unsafe_set(0, Scalar[DType.int](-6_000_000_000_000_000_000))
    a.unsafe_set(1, Scalar[DType.int](6_000_000_000_000_000_000))
    var perm = lexsort_rows(a)
    assert_equal(perm, [0, 2, 1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
