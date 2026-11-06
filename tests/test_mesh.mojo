from std.testing import TestSuite, assert_equal, assert_raises, assert_true

import numojo as nm
from numojo.prelude import *

from mojodec.core.simplex import Simplex
from mojodec.core.mesh import SimplicialMesh, RegularCubeMesh
from mojodec.core.utils import iget, rows_of
from mojodec.routines.simplex_array import (
    simplex_array_searchsorted,
    simplex_array_parity,
    simplex_array_boundary,
)
from mojodec.routines.combinatorial import relative_parity


def I(s: String) raises -> NDArray[DType.int]:
    return nm.fromstring[DType.int](s)


def R(s: String) raises -> NDArray[f64]:
    return nm.fromstring[f64](s)


def check_same(
    got: List[Simplex], var expected: List[Simplex], with_parity: Bool = True
) raises:
    assert_equal(len(got), len(expected))
    for ref g in got:
        var found = False
        for ref e in expected:
            if g == e:
                found = True
                if with_parity:
                    assert_equal(g.parity, e.parity)
        assert_true(found)


def test_searchsorted() raises:
    var r = simplex_array_searchsorted(
        I("[[0],[1],[4],[6],[7],[10]]"), I("[[6],[10],[0]]")
    )
    assert_equal(Int(r.unsafe_get(0)), 3)
    assert_equal(Int(r.unsafe_get(1)), 5)
    assert_equal(Int(r.unsafe_get(2)), 0)
    r = simplex_array_searchsorted(
        I("[[0,1],[0,2],[1,2],[1,3],[1,4],[3,4]]"), I("[[1,2],[0,2],[3,4]]")
    )
    assert_equal(Int(r.unsafe_get(0)), 2)
    assert_equal(Int(r.unsafe_get(1)), 1)
    assert_equal(Int(r.unsafe_get(2)), 5)
    # labels far beyond the row count, and negative labels
    r = simplex_array_searchsorted(
        I("[[-3,5],[2,1000000000],[1000000000,1000000001]]"),
        I("[[1000000000,1000000001],[-3,5]]"),
    )
    assert_equal(Int(r.unsafe_get(0)), 2)
    assert_equal(Int(r.unsafe_get(1)), 0)
    for ref missing in ["[[0,3]]", "[[2,2]]", "[[5,0]]", "[[-1,0]]"]:
        with assert_raises(contains="not found"):
            _ = simplex_array_searchsorted(
                I("[[0,1],[0,2],[2,4],[3,4]]"), I(missing)
            )
    with assert_raises(contains="not found"):
        _ = simplex_array_searchsorted(
            nm.zeros[DType.int](Shape(0, 2)), I("[[0,1]]")
        )


def test_parity() raises:
    var p = simplex_array_parity(I("[[0,1,4],[0,4,1],[4,0,1]]"))
    assert_equal(Int(p.unsafe_get(0)), 0)
    assert_equal(Int(p.unsafe_get(1)), 1)
    assert_equal(Int(p.unsafe_get(2)), 0)
    assert_equal(Int(simplex_array_parity(I("[[53,2]]")).unsafe_get(0)), 1)
    assert_equal(Int(simplex_array_parity(I("[[117,235]]")).unsafe_get(0)), 0)
    nm.random.seed(0)
    for n_col in range(1, 7):
        for n_row in range(1, 20):
            var A = nm.random.randint[DType.int](Shape(n_row, n_col), 0, 1000)
            var sp = simplex_array_parity(A)
            for i in range(n_row):
                var row = List[Int]()
                var distinct = True
                for j in range(n_col):
                    var x = iget(A, i, j)
                    if x in row:
                        distinct = False
                    row.append(x)
                if not distinct:
                    continue
                var srt = row.copy()
                sort(srt)
                assert_equal(Int(sp.unsafe_get(i)), relative_parity(row, srt))


def test_boundary() raises:
    var fb = simplex_array_boundary(I("[[0,1,2]]"), I("[1]"))
    assert_equal(rows_of(fb.faces), 3)
    assert_equal(iget(fb.faces, 1, 0), 0)
    assert_equal(iget(fb.faces, 1, 1), 2)
    assert_equal(fb.boundary[0, 0], -1.0)
    assert_equal(fb.boundary[1, 0], 1.0)
    assert_equal(fb.boundary[2, 0], -1.0)
    fb = simplex_array_boundary(I("[[2,9]]"), I("[0]"))
    assert_equal(fb.boundary[0, 0], -1.0)
    assert_equal(fb.boundary[1, 0], 1.0)
    # the coboundary is the boundary's transpose, built in canonical form
    fb = simplex_array_boundary(
        I("[[0,1,2],[1,2,3],[0,2,3],[3,4,5]]"), I("[0,1,0,1]")
    )
    var Bt = fb.boundary.transpose()
    ref D = fb.coboundary
    assert_equal(D.nrows, 4)
    assert_true(D.indptr == Bt.indptr)
    assert_true(D.indices == Bt.indices)
    assert_true(D.data == Bt.data)


def test_simplex() raises:
    var s = Simplex([2, 0, 1])
    assert_true(s.vertices == [0, 1, 2])
    assert_equal(s.parity, 0)
    assert_equal(Simplex([1, 0]).parity, 1)
    assert_equal(Simplex([1, 0], 1).parity, 0)
    assert_true(Simplex([1, 0]) == Simplex([0, 1]))


def test_mesh_boundary() raises:
    check_same(
        SimplicialMesh(R("[[0],[1]]"), I("[[0,1]]")).boundary(),
        [Simplex([0], 1), Simplex([1], 0)],
    )
    check_same(
        SimplicialMesh(R("[[0],[1],[2]]"), I("[[0,1],[1,2]]")).boundary(),
        [Simplex([0], 1), Simplex([2], 0)],
    )
    check_same(
        SimplicialMesh(
            R("[[0,0],[1,0],[0,1]]"), I("[[0,1],[1,2],[2,0]]")
        ).boundary(),
        [],
    )
    check_same(
        SimplicialMesh(R("[[0,0],[1,0],[0,1]]"), I("[[0,1,2]]")).boundary(),
        [Simplex([0, 1]), Simplex([1, 2]), Simplex([2, 0])],
    )
    check_same(
        SimplicialMesh(
            R("[[0,0],[1,0],[0,1],[1,1]]"), I("[[0,1,2],[1,3,2]]")
        ).boundary(),
        [Simplex([0, 1]), Simplex([1, 3]), Simplex([3, 2]), Simplex([2, 0])],
    )
    check_same(
        SimplicialMesh(
            R("[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]"), I("[[0,1,2,3]]")
        ).boundary(),
        [
            Simplex([1, 0, 2]),
            Simplex([0, 1, 3]),
            Simplex([2, 0, 3]),
            Simplex([1, 2, 3]),
        ],
    )
    check_same(
        SimplicialMesh(
            R("[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]]"),
            I("[[0,1,2,3],[1,2,3,4]]"),
        ).boundary(),
        [
            Simplex([1, 0, 2]),
            Simplex([0, 1, 3]),
            Simplex([2, 0, 3]),
            Simplex([2, 3, 4]),
            Simplex([1, 4, 3]),
            Simplex([1, 2, 4]),
        ],
    )


def test_mesh_skeleton() raises:
    check_same(
        SimplicialMesh(R("[[0],[1]]"), I("[[0,1]]")).skeleton(0),
        [Simplex([0]), Simplex([1])],
        False,
    )
    check_same(
        SimplicialMesh(R("[[0],[1],[2]]"), I("[[0,1],[1,2]]")).skeleton(1),
        [Simplex([0, 1]), Simplex([1, 2])],
        False,
    )
    check_same(
        SimplicialMesh(R("[[0,0],[1,0],[0,1]]"), I("[[0,1,2]]")).skeleton(1),
        [Simplex([0, 1]), Simplex([1, 2]), Simplex([2, 0])],
        False,
    )


def test_orient() raises:
    var m = SimplicialMesh(
        R("[[0,0],[1,0],[0,1],[1,1]]"), I("[[0,2,1],[1,3,2]]")
    )
    assert_equal(m.orient(), 1)
    assert_equal(iget(m.elements, 0, 0), 2)


def test_cube_mesh() raises:
    var bitmap = nm.ones[DType.int](Shape(2, 2))
    bitmap[Item(0, 1)] = 0
    var ca = RegularCubeMesh(bitmap).cube_array()
    assert_equal(rows_of(ca), 3)
    assert_equal(iget(ca, 1, 0), 1)
    assert_equal(iget(ca, 1, 1), 0)
    assert_equal(iget(ca, 1, 3), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
