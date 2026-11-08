# Property-based tests of MojoDEC's robustness: readers given damaged files,
# raw CSR input, the kd-tree and the dense eigensolver against brute force
# or their defining laws, and round trips of every array dtype. They use
# proptest (see tests/test_properties.mojo for the setup). To run just this
# file:
#
#   pixi run mojo run -I . -I build -I ../proptest tests/test_properties_robust.mojo
#
# A reader that crashes, rather than raising, kills the whole run: that is
# the failure the damaged-file properties look for.

from std.math import sqrt
from std.testing import TestSuite, assert_equal, assert_true

import numojo as nm
from numojo.prelude import *

from proptest import (
    ConjectureData,
    Settings,
    booleans,
    composite,
    fail,
    floats32,
    given,
    integers,
    lists,
)

from mojodec.core.kd_tree import KDTree
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    RealArray,
    fget,
    fset,
    int_array,
    lexsort_rows,
    real_array,
    real_array_from_rows,
    real_vector,
)
from mojodec.routines import dense
from mojodec.routines.io.arrayio import (
    ArrayFormat,
    SparseFormat,
    read_array,
    read_sparse,
    write_array,
    write_sparse,
)
from mojodec.routines.io.gmsh import read_gmsh
from mojodec.routines.io.ply import PlyFormat, read_ply, write_ply
from mojodec.routines.io.text import loadtxt
from mojodec.routines.linalg import eigh_generalized
from mojodec.core.diagonal import DiagonalOperator


comptime TMP = "/tmp/mojodec_properties_robust"


def quick() -> Settings:
    return Settings(max_examples=200, deadline=None)


# ===----------------------------------------------------------------------=== #
# Readers given damaged files
# ===----------------------------------------------------------------------=== #


comptime NUM_FILES = 13


def _gmsh_fixture(i: Int) -> String:
    if i == 0:
        return "tests/meshes/gmsh/square_v22.msh"
    if i == 1:
        return "tests/meshes/gmsh/square_v22_bin.msh"
    if i == 2:
        return "tests/meshes/gmsh/square_v41.msh"
    if i == 3:
        return "tests/meshes/gmsh/square_v41_bin.msh"
    if i == 4:
        return "tests/meshes/gmsh/cube_v22_bin.msh"
    return "tests/meshes/gmsh/cube_v41_bin.msh"


def _read_bytes(path: String) raises -> List[Byte]:
    with open(path, "r") as f:
        return f.read_bytes()


def _write_bytes(path: String, bytes: List[Byte]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))


def _valid_file(which: Int) raises -> List[Byte]:
    """The bytes of a small valid file of kind `which`: arrays in three
    formats, sparse matrices in two, PLY in two, Gmsh fixtures."""
    var path = String(TMP, ".valid")
    var A = real_array_from_rows([[1.5, -2.0], [0.25, 3.0], [-7.0, 0.5]])
    if which < 3:
        write_array(path, A, format=ArrayFormat(UInt8(which)))
    elif which < 5:
        var S = CSRMatrix.from_coo(3, 4, [0, 2, 2], [1, 0, 3], [1.0, 2.0, 3.0])
        write_sparse(
            path, S, format=SparseFormat.BINARY if which == 3 else .ASCII
        )
    elif which < 7:
        var V = real_array_from_rows(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        )
        var T = int_array(2, 3)
        for i in range(6):
            T.unsafe_set(i, Scalar[DType.int]([0, 1, 2, 1, 3, 2][i]))
        write_ply(path, V, T, format=PlyFormat.ASCII if which == 5 else .BINARY)
    else:
        return _read_bytes(_gmsh_fixture(which - 7))
    return _read_bytes(path)


def _read_any(which: Int, path: String) raises:
    if which < 3:
        _ = read_array(path)
    elif which < 5:
        _ = read_sparse(path)
    elif which < 7:
        _ = read_ply(path)
    else:
        _ = read_gmsh(path)


def test_readers_raise_on_truncated_files() raises:
    """Every prefix of a valid file reads or raises; none crashes."""

    def prop(which: Int, cut: Int) raises:
        var bytes = _valid_file(which)
        var n = cut % (len(bytes) + 1)
        var prefix = List[Byte](capacity=n)
        for i in range(n):
            prefix.append(bytes[i])
        var path = String(TMP, ".damaged")
        _write_bytes(path, prefix)
        try:
            _read_any(which, path)
        except:
            pass

    given(
        integers(0, NUM_FILES - 1),
        integers(0, 1 << 20),
        prop,
        Settings(max_examples=100, deadline=None),
    )


def test_readers_raise_on_corrupted_bytes() raises:
    """A valid file with a few bytes overwritten (counts, tags, sizes, header
    words) reads or raises; none crashes or runs out of memory."""

    def prop(which: Int, edits: List[Int]) raises:
        var bytes = _valid_file(which)
        # each edit is a position (high bits) and a byte (low 8 bits)
        for e in edits:
            bytes[(e >> 8) % len(bytes)] = Byte(e & 255)
        var path = String(TMP, ".damaged")
        _write_bytes(path, bytes)
        try:
            _read_any(which, path)
        except:
            pass

    given(
        integers(0, NUM_FILES - 1),
        lists(integers(0, 1 << 30), min_size=1, max_size=4),
        prop,
        Settings(max_examples=200, deadline=None),
    )


# ===----------------------------------------------------------------------=== #
# Raw CSR input
# ===----------------------------------------------------------------------=== #


def test_raw_csr_is_validated() raises:
    """`CSRMatrix(nrows, ncols, indptr, indices, data)` raises exactly when
    the arrays are not a CSR matrix, and otherwise equals the dense matrix
    that sums its entries."""

    def prop(nrows: Int, ncols: Int, indptr: List[Int], cols: List[Int]) raises:
        var data = List[Float64]()
        for i in range(len(cols)):
            data.append(Float64(i + 1))
        var valid = len(indptr) == nrows + 1 and indptr[0] == 0
        if valid:
            for i in range(nrows):
                valid = valid and indptr[i] <= indptr[i + 1]
            valid = valid and indptr[nrows] == len(cols)
        for c in cols:
            valid = valid and 0 <= c and c < ncols
        var A: CSRMatrix
        try:
            A = CSRMatrix(nrows, ncols, indptr.copy(), cols.copy(), data.copy())
        except:
            if valid:
                fail("a valid CSR matrix was rejected")
            return
        if not valid:
            fail("an invalid CSR matrix was accepted")
        var D = A.todense()
        for r in range(nrows):
            for c in range(ncols):
                var want: Float64 = 0
                for j in range(indptr[r], indptr[r + 1]):
                    if cols[j] == c:
                        want += data[j]
                if fget(D, r, c) != want:
                    fail(String("entry (", r, ", ", c, ")"))

    given(
        integers(0, 3),
        integers(1, 3),
        lists(integers(-1, 4), min_size=1, max_size=5),
        lists(integers(-1, 3), max_size=4),
        prop,
        Settings(max_examples=500),
    )


# ===----------------------------------------------------------------------=== #
# Sorting rows
# ===----------------------------------------------------------------------=== #


def test_lexsort_rows_is_a_stable_sort() raises:
    """`lexsort_rows` returns the stable lexicographic order of the rows,
    whether the entries are small (radix sort) or span all of Int (merge
    sort)."""

    def prop(entries: List[Int], width: Int, wide: Bool) raises:
        var n = len(entries) // width
        var a = int_array(n, width)
        for i in range(n * width):
            var x = entries[i]
            if wide and x % 3 == 0:
                x = Int.MIN + x if x % 2 == 0 else Int.MAX - x
            a.unsafe_set(i, Scalar[DType.int](x))
        var perm = lexsort_rows(a)
        assert_equal(len(perm), n)
        var seen = List[Bool](length=n, fill=False)
        for p in perm:
            assert_true(0 <= p and p < n and not seen[p], "not a permutation")
            seen[p] = True
        for i in range(n - 1):
            var order = 0
            for j in range(width):
                var x = Int(a.unsafe_get(perm[i] * width + j))
                var y = Int(a.unsafe_get(perm[i + 1] * width + j))
                if x != y:
                    order = -1 if x < y else 1
                    break
            if order > 0 or (order == 0 and perm[i] > perm[i + 1]):
                fail(String("rows ", perm[i], " and ", perm[i + 1]))

    given(
        lists(integers(-50, 50), max_size=24),
        integers(1, 3),
        booleans(),
        prop,
        quick(),
    )


# ===----------------------------------------------------------------------=== #
# kd-tree
# ===----------------------------------------------------------------------=== #


def _distance(p: List[Float64], q: List[Float64]) -> Float64:
    var s: Float64 = 0
    for j in range(len(p)):
        s += (p[j] - q[j]) * (p[j] - q[j])
    return sqrt(s)


def test_kd_tree_matches_brute_force() raises:
    """On lattice points (with duplicates and ties), `in_sphere` returns
    exactly the points strictly inside the radius, nearest first, and
    `nearest_n` the n nearest."""

    def prop(coords: List[Int], query: List[Int], m: Int, n: Int) raises:
        var N = len(coords) // 2
        var P = real_array(N, 2)
        var points = List[List[Float64]]()
        for i in range(N):
            var p: List[Float64] = [
                Float64(coords[2 * i]),
                Float64(coords[2 * i + 1]),
            ]
            fset(P, i, 0, p[0])
            fset(P, i, 1, p[1])
            points.append(p^)
        var tree = KDTree(P)
        var q: List[Float64] = [Float64(query[0]), Float64(query[1])]
        # a radius equal to some lattice distance, so ties are common
        var r = sqrt(Float64(m))
        var found = tree.in_sphere(q, r)
        var want = 0
        for p in points:
            if _distance(p, q) < r:
                want += 1
        assert_equal(len(found), want, "in_sphere count")
        var last: Float64 = 0
        for i in found:
            var t = _distance(points[i], q)
            assert_true(t < r, "in_sphere returned a point outside")
            assert_true(t >= last, "in_sphere not nearest first")
            last = t
        # nearest_n: the n smallest distances, in order
        var near = tree.nearest_n(q, n)
        assert_equal(len(near), max(0, min(n, N)), "nearest_n count")
        var all = List[Float64]()
        for p in points:
            all.append(_distance(p, q))
        sort(all)
        for i in range(len(near)):
            assert_equal(_distance(points[near[i]], q), all[i], "nearest_n")

    given(
        lists(integers(0, 4), max_size=40),
        lists(integers(-1, 5), min_size=2, max_size=2),
        integers(0, 20),
        integers(-1, 25),
        prop,
        quick(),
    )


# ===----------------------------------------------------------------------=== #
# Dense symmetric eigenproblems
# ===----------------------------------------------------------------------=== #


def _symmetric(entries: List[Int], n: Int, diagonal: Bool) raises -> RealArray:
    """An n x n symmetric matrix from small integers, or a diagonal one with
    repeated entries."""
    var A = real_array(n, n)
    var k = 0
    for i in range(n):
        for j in range(i, n):
            var x = Float64(entries[k % len(entries)])
            k += 1
            if diagonal and i != j:
                x = 0
            fset(A, i, j, x)
            fset(A, j, i, x)
    return A^


def test_eigh_satisfies_its_definition() raises:
    """For symmetric A of size 0..8: A V = V diag(w) and V^T V = I to
    rounding, with w ascending."""

    def prop(entries: List[Int], n: Int, diagonal: Bool) raises:
        var A = _symmetric(entries, n, diagonal)
        var r = dense.eigh(A)
        ref w = r.values
        ref V = r.vectors
        assert_equal(w.size, n)
        var scale: Float64 = 1
        for i in range(n * n):
            scale = max(scale, abs(A.unsafe_get(i)))
        var tol = 64 * Float64(n + 1) * 2.220446049250313e-16
        for i in range(n - 1):
            assert_true(w.unsafe_get(i) <= w.unsafe_get(i + 1), "not ascending")
        for i in range(n):
            for j in range(n):
                var av: Float64 = 0
                var vv: Float64 = 0
                for k in range(n):
                    av += fget(A, i, k) * fget(V, k, j)
                    vv += fget(V, k, i) * fget(V, k, j)
                var res = av - fget(V, i, j) * w.unsafe_get(j)
                if abs(res) > tol * Float64(n) * scale:
                    fail(String("residual ", res, " at (", i, ", ", j, ")"))
                var id = 1.0 if i == j else 0.0
                if abs(vv - id) > tol:
                    fail(String("V^T V off by ", vv - id))

    given(
        lists(integers(-3, 3), min_size=1, max_size=36),
        integers(0, 8),
        booleans(),
        prop,
        quick(),
    )


def test_generalized_eigh_dense_and_diagonal_agree() raises:
    """With a positive diagonal mass matrix, the dense and diagonal overloads
    of `eigh_generalized` give the same eigenvalues; a mass matrix of the
    wrong size is rejected."""

    def prop(entries: List[Int], masses: List[Int], n: Int) raises:
        var K = _symmetric(entries, n, False)
        var m = List[Float64]()
        var M = real_array(n, n)
        for i in range(n):
            m.append(Float64(masses[i % len(masses)]))
            fset(M, i, i, m[i])
        var a = eigh_generalized(K, M)
        var b = eigh_generalized(K, DiagonalOperator(m.copy()))
        var scale: Float64 = 1
        for i in range(n * n):
            scale = max(scale, abs(K.unsafe_get(i)))
        for i in range(n):
            var x = a.values.unsafe_get(i)
            var y = b.values.unsafe_get(i)
            if abs(x - y) > 1e-12 * Float64(n) * scale:
                fail(String("eigenvalue ", i, ": ", x, " vs ", y))
        var wrong = real_array(n + 1, n + 1)
        for i in range(n + 1):
            fset(wrong, i, i, 1.0)
        try:
            _ = eigh_generalized(K, wrong)
        except:
            return
        fail("a mass matrix of the wrong size was accepted")

    given(
        lists(integers(-3, 3), min_size=1, max_size=36),
        lists(integers(1, 5), min_size=1, max_size=8),
        integers(0, 8),
        prop,
        quick(),
    )


# ===----------------------------------------------------------------------=== #
# Round trips of every dtype
# ===----------------------------------------------------------------------=== #


def test_float32_arrays_round_trip() raises:
    """Float32 arrays come back bit for bit in every format."""

    def prop(values: List[Float32], fmt: Int) raises:
        var A = nm.zeros[DType.float32](Shape(len(values), 1))
        for i in range(len(values)):
            A.unsafe_set(i, values[i])
        var path = String(TMP, ".f32")
        write_array(path, A, format=ArrayFormat(UInt8(fmt)))
        var B = read_array[DType.float32](path)
        assert_equal(B.size, A.size)
        for i in range(A.size):
            var x = A.unsafe_get(i)
            var y = B.unsafe_get(i)
            if x.to_bits() != y.to_bits():
                fail(String("wrote ", x, ", read ", y, " (", fmt, ")"))

    given(
        lists(
            floats32(allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=6,
        ),
        integers(0, 2),
        prop,
        quick(),
    )


def test_int64_arrays_round_trip() raises:
    """64-bit integer arrays, through every array format and `loadtxt`."""

    def prop(values: List[Int], fmt: Int) raises:
        var A = nm.zeros[DType.int](Shape(len(values), 1))
        for i in range(len(values)):
            A.unsafe_set(i, Scalar[DType.int](values[i]))
        var path = String(TMP, ".i64")
        write_array(path, A, format=ArrayFormat(UInt8(fmt)))
        var B = read_array[DType.int](path)
        assert_equal(B.size, A.size)
        for i in range(A.size):
            assert_equal(Int(B.unsafe_get(i)), values[i], "array file")
        var text = String()
        for v in values:
            text.write(v, "\n")
        with open(String(TMP, ".txt"), "w") as f:
            f.write(text)
        var C = loadtxt[DType.int](String(TMP, ".txt"))
        for i in range(C.size):
            assert_equal(Int(C.unsafe_get(i)), values[i], "loadtxt")

    given(
        lists(integers(), min_size=1, max_size=6),
        integers(0, 2),
        prop,
        quick(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
