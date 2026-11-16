# Property-based tests of MojoDEC's robustness: readers given damaged files,
# raw CSR input, the kd-tree and the dense eigensolver against brute force
# or their defining laws, the Krylov solvers' options (starting guesses,
# null spaces, Ritz values), the implicit time steppers, and the sparse
# LDL^T and Cholesky factorizations against dense solves, and round trips
# of every array dtype. They use
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
    assume,
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
from mojodec.routines.linalg import (
    NullSpace,
    StopReason,
    cg,
    eigenvalue_range,
    eigh_generalized,
    jacobi,
    minres,
)
from mojodec.routines.time import (
    CrankNicolson,
    ImplicitMidpoint,
    StepSolver,
    _Guess,
)
from mojodec.routines.amg import Relaxation, SmoothedAggregation
from mojodec.routines.lanczos import eigsh
from mojodec.routines.ordering import amd_ordering
from mojodec.routines.sparse_ldlt import (
    FillOrdering,
    SparseCholesky,
    SparseLDLT,
)
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
# Krylov solvers and implicit time steppers against dense solves
# ===----------------------------------------------------------------------=== #


def _graph(n: Int, ints: List[Int], shift: Bool) raises -> RealArray:
    """A weighted graph Laplacian on n vertices from small integers: a path
    (so connected) plus extra edges, weights 1..4; with `shift`, plus a
    diagonal of 1..3, which makes it positive definite. Dense, n x n."""
    var A = real_array(n, n)
    var m = len(ints)

    def add(mut A: RealArray, i: Int, j: Int, w: Float64) raises:
        fset(A, i, i, fget(A, i, i) + w)
        fset(A, j, j, fget(A, j, j) + w)
        fset(A, i, j, fget(A, i, j) - w)
        fset(A, j, i, fget(A, j, i) - w)

    for i in range(n - 1):
        add(A, i, i + 1, Float64(1 + abs(ints[i % m]) % 4))
    for k in range(m // 3):
        var i = abs(ints[3 * k]) % n
        var j = abs(ints[3 * k + 1]) % n
        if i != j:
            add(A, i, j, Float64(1 + abs(ints[3 * k + 2]) % 4))
    if shift:
        for i in range(n):
            fset(
                A, i, i, fget(A, i, i) + Float64(1 + abs(ints[(i + 1) % m]) % 3)
            )
    return A^


def _csr(A: RealArray) raises -> CSRMatrix:
    var n = A.shape[0]
    var r = List[Int]()
    var c = List[Int]()
    var v = List[Float64]()
    for i in range(n):
        for j in range(A.shape[1]):
            var x = fget(A, i, j)
            if x != 0:
                r.append(i)
                c.append(j)
                v.append(x)
    return CSRMatrix.from_coo(n, A.shape[1], r, c, v)


def _vector(n: Int, ints: List[Int], offset: Int) raises -> RealArray:
    var x = real_vector(n)
    for i in range(n):
        x.unsafe_set(i, Float64(ints[(i + offset) % len(ints)]))
    return x^


def _max_abs(x: RealArray) -> Float64:
    var m: Float64 = 0
    for i in range(x.size):
        m = max(m, abs(x.unsafe_get(i)))
    return m


def _close(got: RealArray, want: RealArray, tol: Float64, what: String) raises:
    var scale = 1 + _max_abs(want)
    for i in range(want.size):
        var e = abs(got.unsafe_get(i) - want.unsafe_get(i))
        if not (e <= tol * scale):
            fail(String(what, ": entry ", i, " off by ", e))


def test_krylov_starting_guesses_reach_the_solution() raises:
    """For SPD A (n <= 10, cond <= ~100), cg and minres from any guess, with
    or without Jacobi, reach the dense solution; the dense solution as the
    guess takes no step."""

    def prop(ints: List[Int], n: Int, pre: Bool) raises:
        var D = _graph(n, ints, True)
        var A = _csr(D)
        var b = _vector(n, ints, 1)
        var x0 = _vector(n, ints, 2)
        var exact = dense.solve(D, b)
        var J = jacobi(A)
        var c = cg(A, b, M=J, rtol=1e-12, x0=x0.copy()) if pre else cg(
            A, b, rtol=1e-12, x0=x0.copy()
        )
        if not c.converged():
            fail(String("cg stopped: ", c.stop))
        _close(c.x, exact, 1e-8, "cg")
        var m = minres(A, b, M=J, rtol=1e-12, x0=x0.copy()) if pre else minres(
            A, b, rtol=1e-12, x0=x0.copy()
        )
        if not m.converged():
            fail(String("minres stopped: ", m.stop))
        _close(m.x, exact, 1e-8, "minres")
        if cg(A, b, rtol=1e-12, x0=exact.copy()).iterations != 0:
            fail("cg stepped from the exact solution")
        if minres(A, b, M=J, rtol=1e-12, x0=exact.copy()).iterations != 0:
            fail("minres stepped from the exact solution")

    given(
        lists(integers(-5, 5), min_size=1, max_size=40),
        integers(1, 10),
        booleans(),
        prop,
        quick(),
    )


def test_krylov_null_space_gives_the_least_norm_solution() raises:
    """For a connected graph Laplacian A and any b and guess, cg and minres
    with the constants as null space give the least-squares solution of
    least norm, which solves (A + 1 1^T / n) x = P b."""

    def prop(ints: List[Int], n: Int, pre: Bool) raises:
        var D = _graph(n, ints, False)
        var A = _csr(D)
        var b = _vector(n, ints, 1)
        var x0 = _vector(n, ints, 3)
        var ns = NullSpace.constants(n)
        var Pb = b.copy()
        ns.remove_from(Pb)
        var R = D.copy()
        for i in range(n):
            for j in range(n):
                fset(R, i, j, fget(R, i, j) + 1.0 / Float64(n))
        var exact = dense.solve(R, Pb)
        var J = jacobi(A) if n > 1 else DiagonalOperator([1.0])
        var c = cg(
            A, b, M=J, rtol=1e-12, x0=x0.copy(), nullspace=ns
        ) if pre else cg(A, b, rtol=1e-12, x0=x0.copy(), nullspace=ns)
        if not c.converged():
            fail(String("cg stopped: ", c.stop))
        _close(c.x, exact, 1e-8, "cg")
        var m = minres(
            A, b, M=J, rtol=1e-12, x0=x0.copy(), nullspace=ns
        ) if pre else minres(A, b, rtol=1e-12, x0=x0.copy(), nullspace=ns)
        if not m.converged():
            fail(String("minres stopped: ", m.stop))
        _close(m.x, exact, 1e-8, "minres")

    given(
        lists(integers(-5, 5), min_size=1, max_size=40),
        integers(1, 10),
        booleans(),
        prop,
        quick(),
    )


def test_ritz_values_lie_in_the_spectrum() raises:
    """The Ritz values of cg and minres, and eigenvalue_range's ends, lie in
    the spectrum of M A (M = I or Jacobi), from the dense eigensolver."""

    def prop(ints: List[Int], n: Int, pre: Bool) raises:
        var D = _graph(n, ints, True)
        var A = _csr(D)
        var b = _vector(n, ints, 1)
        assume(_max_abs(b) > 0)
        var diag = List[Float64]()
        for i in range(n):
            diag.append(fget(D, i, i) if pre else 1.0)
        var w = eigh_generalized(D, DiagonalOperator(diag^)).values.copy()
        var lo = w.unsafe_get(0)
        var hi = w.unsafe_get(n - 1)
        var tol = 1e-9 * hi
        var J = jacobi(A)
        var c = cg(A, b, M=J, rtol=1e-12, ritz=True) if pre else cg(
            A, b, rtol=1e-12, ritz=True
        )
        var m = minres(A, b, M=J, rtol=1e-12, ritz=True) if pre else minres(
            A, b, rtol=1e-12, ritz=True
        )
        for ref t in [
            c.ritz_values.value().copy(),
            m.ritz_values.value().copy(),
        ]:
            for i in range(t.size):
                var x = t.unsafe_get(i)
                if not (x >= lo - tol and x <= hi + tol):
                    fail(
                        String(
                            "Ritz value ", x, " outside [", lo, ", ", hi, "]"
                        )
                    )
        var r = eigenvalue_range(A, M=J) if pre else eigenvalue_range(A)
        if not (r.lowest >= lo - tol and r.highest <= hi + tol):
            fail(
                String(
                    "range [",
                    r.lowest,
                    ", ",
                    r.highest,
                    "] leaves the spectrum",
                )
            )
        if not (r.lowest <= r.highest):
            fail("range is reversed")

    given(
        lists(integers(-5, 5), min_size=1, max_size=40),
        integers(1, 10),
        booleans(),
        prop,
        quick(),
    )


def test_fischer_guess_is_never_worse_than_zero() raises:
    """For SPD S and any sequence of right-hand sides, the time steppers'
    guess is at least as close to each solution as zero in the S norm, and
    exact for a right-hand side it has seen."""

    def prop(ints: List[Int], n: Int, count: Int) raises:
        var D = _graph(n, ints, True)
        var S = _csr(D)
        var guess = _Guess()
        for k in range(count):
            var b = _vector(n, ints, 5 * k + 1)
            var x = dense.solve(D, b)
            var g = guess.for_rhs(b)
            if g:
                var e = x.copy()
                axpy_into(-1.0, g.value(), e)
                var err = _dot(e, S @ e)
                var zero = _dot(x, S @ x)
                if not (err <= zero * (1 + 1e-10) + 1e-20):
                    fail(String("guess error ", err, " > ", zero, " at ", k))
            guess.keep(S, x)
            var again = guess.for_rhs(b)
            if again:
                _close(again.value(), x, 1e-9, "a seen right-hand side")
            elif _max_abs(x) != 0:  # only x = 0 adds nothing to an empty basis
                fail("no guess after a nonzero solution")

    given(
        lists(integers(-5, 5), min_size=1, max_size=40),
        integers(1, 10),
        integers(1, 40),
        prop,
        quick(),
    )


def _dot(x: RealArray, y: RealArray) -> Float64:
    var s: Float64 = 0
    for i in range(x.size):
        s += x.unsafe_get(i) * y.unsafe_get(i)
    return s


def axpy_into(a: Float64, x: RealArray, mut y: RealArray):
    for i in range(x.size):
        y.unsafe_set(i, y.unsafe_get(i) + a * x.unsafe_get(i))


def test_implicit_steppers_match_dense_steps() raises:
    """Crank-Nicolson and the implicit midpoint rule, factored once or by
    CG started from the previous steps' guess, follow the dense step to
    rounding over many steps, at small and large dt."""

    def prop(
        ints: List[Int], n: Int, steps: Int, big: Bool, factored: Bool
    ) raises:
        var dt = 0.7 if big else 0.03
        var solver = StepSolver.CHOLESKY if factored else StepSolver.CG
        # Crank-Nicolson: M SPD, K a PSD Laplacian, a source
        var Md = _graph(n, ints, True)
        var rotated: List[Int] = [ints[len(ints) - 1]]
        rotated.extend(ints.copy())
        var Kd = _graph(n, rotated, False)
        var cn = CrankNicolson(_csr(Md), _csr(Kd), dt, solver=solver)
        var u = _vector(n, ints, 2)
        var f = _vector(n, ints, 4)
        var ref_u = u.copy()
        var L = real_array(n, n)
        var Rm = real_array(n, n)
        for i in range(n):
            for j in range(n):
                fset(L, i, j, fget(Md, i, j) + 0.5 * dt * fget(Kd, i, j))
                fset(Rm, i, j, fget(Md, i, j) - 0.5 * dt * fget(Kd, i, j))
        for _ in range(steps):
            cn.step(u, f)
            var rhs = real_vector(n)
            for i in range(n):
                var s = dt * f.unsafe_get(i)
                for j in range(n):
                    s += fget(Rm, i, j) * ref_u.unsafe_get(j)
                rhs.unsafe_set(i, s)
            ref_u = dense.solve(L, rhs)
        _close(u, ref_u, 1e-8, "Crank-Nicolson")
        # implicit midpoint: q of size n, p of size n2, coupling K (n x n2)
        var n2 = 1 + abs(ints[0]) % 6
        var Kc = real_array(n, n2)
        for i in range(n):
            for j in range(n2):
                fset(Kc, i, j, Float64(ints[(3 * i + 5 * j) % len(ints)]))
        var mp = List[Float64]()
        for j in range(n2):
            mp.append(Float64(1 + abs(ints[(j + 2) % len(ints)]) % 3))
        var im = ImplicitMidpoint(
            _csr(Md), DiagonalOperator(mp.copy()), _csr(Kc), dt, solver=solver
        )
        var q = _vector(n, ints, 1)
        var p = _vector(n2, ints, 3)
        var E0 = im.energy(q, p)
        # dense: [Mq 0; 0 Mp] z' = [0 K; -K^T 0] z, midpoint rule
        var N = n + n2
        var Lz = real_array(N, N)
        var Rz = real_array(N, N)
        for i in range(N):
            for j in range(N):
                var m: Float64 = 0
                if i < n and j < n:
                    m = fget(Md, i, j)
                elif i >= n and j >= n and i == j:
                    m = mp[i - n]
                var a: Float64 = 0
                if i < n and j >= n:
                    a = fget(Kc, i, j - n)
                elif i >= n and j < n:
                    a = -fget(Kc, j, i - n)
                fset(Lz, i, j, m - 0.5 * dt * a)
                fset(Rz, i, j, m + 0.5 * dt * a)
        var z = real_vector(N)
        for i in range(n):
            z.unsafe_set(i, q.unsafe_get(i))
        for j in range(n2):
            z.unsafe_set(n + j, p.unsafe_get(j))
        for _ in range(steps):
            im.step(q, p)
            var rhs = real_vector(N)
            for i in range(N):
                var s: Float64 = 0
                for j in range(N):
                    s += fget(Rz, i, j) * z.unsafe_get(j)
                rhs.unsafe_set(i, s)
            z = dense.solve(Lz, rhs)
        var zq = real_vector(n)
        var zp = real_vector(n2)
        for i in range(n):
            zq.unsafe_set(i, z.unsafe_get(i))
        for j in range(n2):
            zp.unsafe_set(j, z.unsafe_get(n + j))
        _close(q, zq, 1e-8, "midpoint q")
        _close(p, zp, 1e-8, "midpoint p")
        var E = im.energy(q, p)
        if not (abs(E - E0) <= 1e-9 * (1 + E0)):
            fail(String("energy drifted from ", E0, " to ", E))

    given(
        lists(integers(-5, 5), min_size=1, max_size=40),
        integers(1, 8),
        integers(1, 30),
        booleans(),
        booleans(),
        prop,
        quick(),
    )


def test_sparse_cholesky_matches_dense_solves() raises:
    """For SPD A (n <= 12), SparseCholesky with AMD and with the natural
    order solves A x = b as the dense solver does; all pivots positive."""

    def prop(ints: List[Int], n: Int) raises:
        var D = _graph(n, ints, True)
        var A = _csr(D)
        var b = _vector(n, ints, 1)
        var exact = dense.solve(D, b)
        for which in range(2):
            var F = SparseCholesky(
                A, FillOrdering.AMD if which == 0 else FillOrdering.NATURAL
            )
            _close(F.solve(b), exact, 1e-10, "cholesky")
            if F.factor.inertia().positive != n:
                fail("a pivot is not positive")
        var p = amd_ordering(A)
        var seen = List[Bool](length=n, fill=False)
        for k in range(n):
            if seen[p[k]]:
                fail("amd_ordering is not a permutation")
            seen[p[k]] = True

    given(
        lists(integers(-5, 5), min_size=1, max_size=40),
        integers(1, 12),
        prop,
        quick(),
    )


def test_sparse_ldlt_solves_saddle_points() raises:
    """For [[-M, B^T], [B, 0]] with M SPD (m <= 10) and B (c x m, c <= m) of
    full rank, in either block order, SparseLDLT with AMD needs no
    regularization, puts the constraints last, solves as the dense solver
    does, and finds m negative and c positive pivots."""

    def prop(ints: List[Int], m: Int, c_in: Int, first: Bool) raises:
        var c = 1 + c_in % m
        var Md = _graph(m, ints, True)
        var N = m + c
        var Ad = real_array(N, N)
        var om = c if first else 0  # where the M block starts
        var oc = 0 if first else m  # where the constraints start
        for i in range(m):
            for j in range(m):
                fset(Ad, om + i, om + j, -fget(Md, i, j))
        for i in range(c):
            for j in range(m):
                # full rank: a nonzero diagonal in the first c columns
                var x = Float64(ints[(7 * i + 3 * j) % len(ints)] % 2)
                if j == i:
                    x = Float64(1 + abs(ints[i % len(ints)]) % 3)
                elif j < c:
                    x = 0
                fset(Ad, oc + i, om + j, x)
                fset(Ad, om + j, oc + i, x)
        var A = _csr(Ad)
        var b = _vector(N, ints, 2)
        var F = SparseLDLT(A)
        if F.num_regularized != 0:
            fail("regularized")
        _close(F.solve(b), dense.solve(Ad, b), 1e-9, "saddle point")
        var inertia = F.inertia()
        if inertia.negative != m or inertia.positive != c:
            fail(String("inertia ", inertia))
        for k in range(N - c, N):
            var row = F.perm[k]
            if row < oc or row >= oc + c:
                fail(String("constraint not last at ", k))

    given(
        lists(integers(-5, 5), min_size=1, max_size=40),
        integers(1, 10),
        integers(0, 9),
        booleans(),
        prop,
        quick(),
    )


# ===----------------------------------------------------------------------=== #
# Round trips of every dtype
# ===----------------------------------------------------------------------=== #


def test_eigsh_finds_every_copy_of_multiple_eigenvalues() raises:
    """K and M are r disjoint copies of a small graph pencil, so every
    eigenvalue has multiplicity exactly r, the case single-vector Lanczos
    misses copies of. eigsh's nev eigenvalues are the nev nearest sigma
    (compared by distance, so ties at the edge of the set don't matter),
    with small backward errors, for any shift and bases small enough to
    force restarts."""

    def prop(
        ints: List[Int], n0: Int, r: Int, nev_pick: Int, extra: Int, s: Int
    ) raises:
        var G = _graph(n0, ints, False)
        var n = n0 * r
        var K = real_array(n, n)
        var M = real_array(n, n)
        for c in range(r):
            for i in range(n0):
                fset(
                    M,
                    c * n0 + i,
                    c * n0 + i,
                    Float64(1 + abs(ints[i % len(ints)]) % 3),
                )
                for j in range(n0):
                    fset(K, c * n0 + i, c * n0 + j, fget(G, i, j))
        var sigma = Float64(s) / 4 - 0.3
        var lam = eigh_generalized(K, M).values.copy()
        for i in range(lam.size):
            assume(abs(lam.unsafe_get(i) - sigma) > 1e-3)
        var nev = 1 + nev_pick % n
        # 2 nev + 1 and up, as the default: smaller bases converge, but
        # linearly with the ratio of neighboring |theta|, which can need
        # thousands of restarts
        var ncv = min(n, 2 * nev + 1 + extra)
        var eig = eigsh(_csr(K), _csr(M), nev, sigma=sigma, ncv=ncv)
        var want = List[Float64]()
        for i in range(lam.size):
            want.append(abs(lam.unsafe_get(i) - sigma))
        sort(want)
        var got = List[Float64]()
        for i in range(nev):
            got.append(abs(eig.values.unsafe_get(i) - sigma))
        sort(got)
        for i in range(nev):
            if not (abs(got[i] - want[i]) <= 1e-9 * (1 + want[i])):
                fail(
                    String(
                        "distance ",
                        i,
                        ": ",
                        got[i],
                        " vs ",
                        want[i],
                        " (n ",
                        n,
                        ", nev ",
                        nev,
                        ", ncv ",
                        ncv,
                        ")",
                    )
                )
        # tol = 1e-12 bounds the residual relative to K - sigma M, so the
        # backward error grows with a shift far from the spectrum
        var eta = eig.residuals(_csr(K), _csr(M))
        for i in range(nev):
            if not (eta.unsafe_get(i) < 4e-12 * (1 + abs(sigma))):
                fail(String("backward error ", eta.unsafe_get(i)))

    given(
        lists(integers(-5, 5), min_size=1, max_size=20),
        integers(2, 6),
        integers(1, 4),
        integers(0, 30),
        integers(0, 6),
        integers(-4, 60),
        prop,
        quick(),
    )


def test_amg_cycle_is_a_valid_preconditioner() raises:
    """For SPD graph matrices (n <= 60) coarsened down to a few unknowns,
    the V-cycle is symmetric and positive definite with either smoother
    (what cg relies on), and cg with it reaches the dense solution."""

    def prop(ints: List[Int], n: Int, gs: Bool, coarse: Int) raises:
        var D = _graph(n, ints, True)
        var A = _csr(D)
        var relax = Relaxation.GAUSS_SEIDEL if gs else Relaxation.CHEBYSHEV
        var amg = SmoothedAggregation(A, relaxation=relax, coarse_size=coarse)
        var x = _vector(n, ints, 1)
        var y = _vector(n, ints, 2)
        assume(_dot(x, x) > 0 and _dot(y, y) > 0)
        var Bx = real_vector(n)
        var By = real_vector(n)
        amg.apply_into(x, Bx)
        amg.apply_into(y, By)
        var a = _dot(y, Bx)
        var c = _dot(x, By)
        var scale = sqrt(_dot(x, Bx) * _dot(y, By)) + 1e-300
        if not (abs(a - c) <= 1e-10 * scale):
            fail(String("not symmetric: ", a, " vs ", c))
        if not (_dot(x, Bx) > 0):
            fail(String("not positive: x^T B x = ", _dot(x, Bx)))
        var b = _vector(n, ints, 3)
        var res = cg(A, b, M=amg, rtol=1e-12, maxiter=500)
        if not res.converged():
            fail(String("cg stopped: ", res.stop))
        _close(res^.take_x(), dense.solve(D, b), 1e-8, "amg-cg")

    given(
        lists(integers(-5, 5), min_size=3, max_size=60),
        integers(2, 60),
        booleans(),
        integers(1, 8),
        prop,
        quick(),
    )


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
