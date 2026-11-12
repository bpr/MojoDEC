# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Array helpers (`mojodec.core.utils`).
=====================================
Small conveniences over NuMojo's `NDArray` used throughout MojoDEC.

MojoDEC stores integer arrays (simplex arrays, cube arrays, index lists) as
`NDArray[DType.int]` and real arrays (vertices, volumes, cochain values) as
`NDArray[DType.float64]`. All arrays built here are C-contiguous, so element
`(i, j)` of an `r x c` array lives at flat offset `i * c + j`.
"""

from std.math import sqrt
import numojo as nm
from numojo.prelude import *

from mojodec.core.parallel import chunk_scratch, num_chunks, parallel_ranges

comptime IntArray = NDArray[DType.int]
"""A C-contiguous array of machine integers."""
comptime RealArray = NDArray[DType.float64]
"""A C-contiguous array of 64-bit floats."""


def int_array(rows: Int, cols: Int) raises -> IntArray:
    """A zero-filled `rows x cols` integer array.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        A `rows x cols` array of zeros.

    Raises:
        If NuMojo cannot allocate the array.
    """
    return nm.zeros[DType.int](Shape(rows, cols))


def int_vector(n: Int) raises -> IntArray:
    """A zero-filled 1-D integer array of length `n`.

    Args:
        n: The length.

    Returns:
        A length-`n` array of zeros.

    Raises:
        If NuMojo cannot allocate the array.
    """
    return nm.zeros[DType.int](Shape(n))


def real_array(rows: Int, cols: Int) raises -> RealArray:
    """A zero-filled `rows x cols` real array.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        A `rows x cols` array of zeros.

    Raises:
        If NuMojo cannot allocate the array.
    """
    return nm.zeros[DType.float64](Shape(rows, cols))


def real_vector(n: Int) raises -> RealArray:
    """A zero-filled 1-D real array of length `n`.

    Args:
        n: The length.

    Returns:
        A length-`n` array of zeros.

    Raises:
        If NuMojo cannot allocate the array.
    """
    return nm.zeros[DType.float64](Shape(n))


def int_array_from_rows(rows: List[List[Int]], cols: Int) raises -> IntArray:
    """Build an `len(rows) x cols` integer array from nested lists.

    Args:
        rows: The rows, each of length `cols`.
        cols: Number of columns.

    Returns:
        A `len(rows) x cols` integer array.

    Raises:
        If a row does not have `cols` entries, or NuMojo cannot allocate
            the array.
    """
    var a = int_array(len(rows), cols)
    for i in range(len(rows)):
        if len(rows[i]) != cols:
            raise Error("int_array_from_rows: ragged rows")
        for j in range(cols):
            a.unsafe_set(i * cols + j, Scalar[DType.int](rows[i][j]))
    return a^


def real_array_from_rows(rows: List[List[Float64]]) raises -> RealArray:
    """Build a real array from nested lists of equal length.

    Args:
        rows: The rows, all of the same length.

    Returns:
        A `len(rows) x len(rows[0])` real array (`0 x 0` when `rows` is
            empty).

    Raises:
        If the rows have different lengths, or NuMojo cannot allocate
            the array.
    """
    var cols = len(rows[0]) if len(rows) > 0 else 0
    var a = real_array(len(rows), cols)
    for i in range(len(rows)):
        if len(rows[i]) != cols:
            raise Error("real_array_from_rows: ragged rows")
        for j in range(cols):
            a.unsafe_set(i * cols + j, rows[i][j])
    return a^


def probe_vector(n: Int, seed: UInt64) raises -> RealArray:
    """A fixed, irregular vector with entries in [-1, 1), for checks and
    estimates that need a start vector with no special structure (the
    symmetry check of `cg`, the norm estimates of `EighResult.residuals`).

    Args:
        n: The length.
        seed: Selects the vector; the same seed gives the same vector.

    Returns:
        The vector, from a linear congruential generator.

    Raises:
        If NuMojo cannot allocate the array.
    """
    var x = real_vector(n)
    var s: UInt64 = 0x9E3779B97F4A7C15 ^ seed
    for i in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        x.unsafe_set(i, Float64((s >> 11) & 0xFFFFF) / 524288.0 - 1.0)
    return x^


def real_vector_from_list(values: List[Float64]) raises -> RealArray:
    """Build a 1-D real array from a list.

    Args:
        values: The entries.

    Returns:
        A 1-D array of length `len(values)`.

    Raises:
        If NuMojo cannot allocate the array.
    """
    var a = real_vector(len(values))
    for i in range(len(values)):
        a.unsafe_set(i, values[i])
    return a^


def int_vector_from_list(values: List[Int]) raises -> IntArray:
    """Build a 1-D integer array from a list.

    Args:
        values: The entries.

    Returns:
        A 1-D array of length `len(values)`.

    Raises:
        If NuMojo cannot allocate the array.
    """
    var a = int_vector(len(values))
    for i in range(len(values)):
        a.unsafe_set(i, Scalar[DType.int](values[i]))
    return a^


@always_inline
def rows_of[dt: DType](a: NDArray[dt]) -> Int:
    """Length of the first axis (0 for a 0-d array).

    Parameters:
        dt: The element type.

    Args:
        a: The array.

    Returns:
        The number of rows.
    """
    return Int(a.shape.unsafe_get(0)) if a.ndim >= 1 else 0


@always_inline
def cols_of[dt: DType](a: NDArray[dt]) -> Int:
    """Length of the second axis (1 for a 1-D array).

    Parameters:
        dt: The element type.

    Args:
        a: The array.

    Returns:
        The number of columns.
    """
    return Int(a.shape.unsafe_get(1)) if a.ndim >= 2 else 1


@always_inline
def iget(a: IntArray, i: Int, j: Int) -> Int:
    """Element `(i, j)` of a C-contiguous 2-D integer array.

    Args:
        a: The array.
        i: The row (not bounds-checked).
        j: The column (not bounds-checked).

    Returns:
        The element as an `Int`.
    """
    return Int(a.unsafe_get(i * cols_of(a) + j))


@always_inline
def iset(mut a: IntArray, i: Int, j: Int, value: Int):
    """Set element `(i, j)` of a C-contiguous 2-D integer array.

    Args:
        a: The array to modify.
        i: The row (not bounds-checked).
        j: The column (not bounds-checked).
        value: The new value.
    """
    a.unsafe_set(i * cols_of(a) + j, Scalar[DType.int](value))


@always_inline
def fget(a: RealArray, i: Int, j: Int) -> Float64:
    """Element `(i, j)` of a C-contiguous 2-D real array.

    Args:
        a: The array.
        i: The row (not bounds-checked).
        j: The column (not bounds-checked).

    Returns:
        The element.
    """
    return a.unsafe_get(i * cols_of(a) + j)


@always_inline
def fset(mut a: RealArray, i: Int, j: Int, value: Float64):
    """Set element `(i, j)` of a C-contiguous 2-D real array.

    Args:
        a: The array to modify.
        i: The row (not bounds-checked).
        j: The column (not bounds-checked).
        value: The new value.
    """
    a.unsafe_set(i * cols_of(a) + j, value)


def row_list(a: IntArray, i: Int) -> List[Int]:
    """Row `i` of a 2-D integer array as a list.

    Args:
        a: The array.
        i: The row (not bounds-checked).

    Returns:
        The `cols_of(a)` entries of row `i`.
    """
    var c = cols_of(a)
    var out = List[Int](capacity=c)
    for j in range(c):
        out.append(Int(a.unsafe_get(i * c + j)))
    return out^


def real_row_list(a: RealArray, i: Int) -> List[Float64]:
    """Row `i` of a 2-D real array as a list.

    Args:
        a: The array.
        i: The row (not bounds-checked).

    Returns:
        The `cols_of(a)` entries of row `i`.
    """
    var c = cols_of(a)
    var out = List[Float64](capacity=c)
    for j in range(c):
        out.append(a.unsafe_get(i * c + j))
    return out^


def take_rows(a: RealArray, rows: List[Int]) raises -> RealArray:
    """The rows of a 2-D real array selected by `rows`, in order.

    Args:
        a: The array.
        rows: Row indices (not bounds-checked); may repeat.

    Returns:
        A `len(rows) x cols_of(a)` array.

    Raises:
        If NuMojo cannot allocate the array.
    """
    var c = cols_of(a)
    var out = real_array(len(rows), c)
    for i in range(len(rows)):
        for j in range(c):
            out.unsafe_set(i * c + j, a.unsafe_get(rows[i] * c + j))
    return out^


def take_int_rows(a: IntArray, rows: List[Int]) raises -> IntArray:
    """The rows of a 2-D integer array selected by `rows`, in order.

    Args:
        a: The array.
        rows: Row indices (not bounds-checked); may repeat.

    Returns:
        A `len(rows) x cols_of(a)` array.

    Raises:
        If NuMojo cannot allocate the array.
    """
    var c = cols_of(a)
    var out = int_array(len(rows), c)
    for i in range(len(rows)):
        for j in range(c):
            out.unsafe_set(i * c + j, a.unsafe_get(rows[i] * c + j))
    return out^


def compare_rows(a: IntArray, i: Int, b: IntArray, j: Int) -> Int:
    """Lexicographic comparison of row `i` of `a` with row `j` of `b`.

    Returns -1, 0 or 1. Both arrays must have the same number of columns.

    Args:
        a: The first array.
        i: A row of `a`.
        b: The second array.
        j: A row of `b`.

    Returns:
        -1, 0 or 1 as row `i` of `a` is less than, equal to, or
            greater than row `j` of `b`.
    """
    var c = cols_of(a)
    for k in range(c):
        var x = Int(a.unsafe_get(i * c + k))
        var y = Int(b.unsafe_get(j * c + k))
        if x < y:
            return -1
        if x > y:
            return 1
    return 0


def lexsort_rows(a: IntArray) -> List[Int]:
    """Stable permutation that sorts the rows of `a` lexicographically.

    Equivalent to NumPy's `lexsort(a.T[::-1])`: the first column is the
    primary key. When the entries span a range not much larger than the
    number of rows, as vertex ids in a simplex array do, this is a radix
    sort (a counting sort per column, last column first); otherwise a merge
    sort.

    Args:
        a: A 2-D integer array.

    Returns:
        The row indices of `a` in sorted order.
    """
    var n = rows_of(a)
    var perm = List[Int](capacity=n)
    for i in range(n):
        perm.append(i)
    if n < 2 or a.size == 0:
        return perm^
    var lo = Int(a.unsafe_get(0))
    var hi = lo
    for i in range(1, a.size):
        var x = Int(a.unsafe_get(i))
        lo = min(lo, x)
        hi = max(hi, x)
    # hi - lo can overflow when lo < 0; lo + span cannot
    var span = 4 * n + 1024
    var narrow = hi - lo <= span if lo >= 0 else hi <= lo + span
    if narrow:
        _radix_sort_rows(a, perm, lo, hi - lo + 1)
    else:
        var scratch = List[Int](length=n, fill=0)
        _merge_sort_rows(a, perm, scratch, 0, n)
    return perm^


def sort_each_row(mut a: IntArray):
    """Sort the entries of every row of a 2-D integer array in place.

    Args:
        a: The array to sort.
    """
    var r = rows_of(a)
    var c = cols_of(a)
    var ap = a.unsafe_ptr()

    def rows(chunk: Int, lo: Int, hi: Int) {ap, c}:
        for i in range(lo, hi):
            # insertion sort: rows are short (simplex vertex lists)
            for j in range(1, c):
                var key = ap[unsafe_offset=i * c + j]
                var k = j - 1
                while k >= 0 and ap[unsafe_offset=i * c + k] > key:
                    ap[unsafe_offset=i * c + k + 1] = ap[
                        unsafe_offset=i * c + k
                    ]
                    k -= 1
                ap[unsafe_offset=i * c + k + 1] = key

    parallel_ranges(r, 50_000, rows)


def require_distinct_simplices(sorted_rows: IntArray, who: String) raises:
    """Raise unless every row (simplex) has distinct vertices and no two rows
    hold the same vertices.

    Args:
        sorted_rows: Simplices with each row sorted (`sort_each_row`).
        who: The caller, named in the error message.

    Raises:
        If a simplex repeats a vertex, or two simplices have the same
        vertices (in any order).
    """
    var r = rows_of(sorted_rows)
    var c = cols_of(sorted_rows)
    # one pass, in parallel chunks: the first row of each chunk that
    # repeats a vertex, and whether the rows are in strictly increasing
    # order, which makes them distinct without sorting them (generated
    # meshes usually are)
    comptime SERIAL_BELOW = 50_000
    var chunks = num_chunks(r, SERIAL_BELOW)
    var repeats = List[Int](length=chunks, fill=-1)
    var unordered = List[Bool](length=chunks, fill=False)
    var sp = sorted_rows.unsafe_ptr()
    var rp = repeats.unsafe_ptr()
    var up = unordered.unsafe_ptr()

    def rows(chunk: Int, lo: Int, hi: Int) {sp, rp, up, c}:
        var ordered = True
        for i in range(lo, hi):
            for j in range(1, c):
                if (
                    sp[unsafe_offset=i * c + j]
                    == sp[unsafe_offset=i * c + j - 1]
                ):
                    rp[unsafe_offset=chunk] = i
                    return
            if ordered and i > 0:
                for j in range(c):
                    var x = sp[unsafe_offset=(i - 1) * c + j]
                    var y = sp[unsafe_offset=i * c + j]
                    if x != y:
                        ordered = x < y
                        break
                    if j == c - 1:
                        ordered = False
        up[unsafe_offset=chunk] = not ordered

    parallel_ranges(r, SERIAL_BELOW, rows)
    var sorted = True
    for k in range(chunks):
        var i = repeats[k]
        if i >= 0:
            for j in range(1, c):
                if iget(sorted_rows, i, j) == iget(sorted_rows, i, j - 1):
                    raise Error(
                        String(who, ": simplex ", i, " repeats vertex ")
                        + String(iget(sorted_rows, i, j))
                    )
        if unordered[k]:
            sorted = False
    if sorted:
        return
    var order = lexsort_rows(sorted_rows)
    for t in range(1, r):
        var a = order[t - 1]
        var b = order[t]
        var same = True
        for j in range(c):
            if iget(sorted_rows, a, j) != iget(sorted_rows, b, j):
                same = False
                break
        if same:
            raise Error(
                String(who, ": simplices ", min(a, b), " and ", max(a, b))
                + " have the same vertices"
            )


def unique_sorted_rows(a: IntArray) raises -> IntArray:
    """Remove consecutive duplicate rows of an array whose rows are sorted.

    Args:
        a: A 2-D integer array with its rows in lexicographic order.

    Returns:
        The distinct rows of `a`, in order.

    Raises:
        If NuMojo cannot allocate the array.
    """
    var keep = List[Int]()
    for i in range(rows_of(a)):
        if i == 0 or compare_rows(a, i, a, i - 1) != 0:
            keep.append(i)
    return take_int_rows(a, keep)


def max_entry(a: IntArray) -> Int:
    """Largest entry of a non-empty integer array.

    Args:
        a: The array (not checked to be non-empty).

    Returns:
        The largest entry.
    """
    var m = Int(a.unsafe_get(0))
    for i in range(1, a.size):
        m = max(m, Int(a.unsafe_get(i)))
    return m


def min_entry(a: IntArray) -> Int:
    """Smallest entry of a non-empty integer array.

    Args:
        a: The array (not checked to be non-empty).

    Returns:
        The smallest entry.
    """
    var m = Int(a.unsafe_get(0))
    for i in range(1, a.size):
        m = min(m, Int(a.unsafe_get(i)))
    return m


def vstack_int(a: IntArray, b: IntArray) raises -> IntArray:
    """Stack two integer arrays with equal column counts vertically.

    Args:
        a: The top rows.
        b: The bottom rows.

    Returns:
        A `(rows_of(a) + rows_of(b)) x cols_of(a)` array.

    Raises:
        If the column counts differ, or NuMojo cannot allocate the array.
    """
    if cols_of(a) != cols_of(b):
        raise Error("vstack_int: column counts differ")
    var c = cols_of(a)
    var out = int_array(rows_of(a) + rows_of(b), c)
    for i in range(a.size):
        out.unsafe_set(i, a.unsafe_get(i))
    for i in range(b.size):
        out.unsafe_set(a.size + i, b.unsafe_get(i))
    return out^


comptime _W = 8
"""SIMD width of the vector kernels: 8 doubles, two AVX2 registers."""


def dot(x: RealArray, y: RealArray) -> Float64:
    """Inner product of two 1-D real arrays of equal length.

    Args:
        x: The first vector.
        y: The second vector (length not checked).

    Returns:
        The sum of `x[i] * y[i]`.
    """
    var n = x.size
    var acc = SIMD[DType.float64, _W](0)
    var i = 0
    while i + _W <= n:
        acc += x.unsafe_load[_W](i) * y.unsafe_load[_W](i)
        i += _W
    var s = acc.reduce_add()
    while i < n:
        s += x.unsafe_get(i) * y.unsafe_get(i)
        i += 1
    return s


def norm2(x: RealArray) -> Float64:
    """Euclidean norm of a real array.

    Args:
        x: The array.

    Returns:
        The square root of the sum of squared entries.
    """
    return sqrt(dot(x, x))


def axpy(alpha: Float64, x: RealArray, mut y: RealArray):
    """`y += alpha * x` for 1-D real arrays.

    Args:
        alpha: The scale factor.
        x: The vector to add.
        y: The vector to update (length not checked).
    """
    var n = x.size
    var i = 0
    while i + _W <= n:
        y.unsafe_store[_W](
            i, y.unsafe_load[_W](i) + alpha * x.unsafe_load[_W](i)
        )
        i += _W
    while i < n:
        y.unsafe_set(i, y.unsafe_get(i) + alpha * x.unsafe_get(i))
        i += 1


def scaled(x: RealArray, alpha: Float64) raises -> RealArray:
    """A copy of `x` multiplied by `alpha`.

    Args:
        x: The array.
        alpha: The scale factor.

    Returns:
        A 1-D array of length `x.size` holding `alpha * x`.

    Raises:
        If NuMojo cannot allocate the array.
    """
    var y = real_vector(x.size)
    var n = x.size
    var i = 0
    while i + _W <= n:
        y.unsafe_store[_W](i, alpha * x.unsafe_load[_W](i))
        i += _W
    while i < n:
        y.unsafe_set(i, alpha * x.unsafe_get(i))
        i += 1
    return y^


def factorial(n: Int) -> Float64:
    """`n!` as a float.

    Args:
        n: A non-negative integer (negative gives 1).

    Returns:
        `n!`.
    """
    var f: Float64 = 1
    for i in range(2, n + 1):
        f *= Float64(i)
    return f


def parallelotope_volume(mut a: List[Float64], m: Int, n: Int) -> Float64:
    """The m-volume of the parallelotope spanned by `m` vectors in n-space.

    It is `prod |R_ii|` from a Householder QR factorization `A^T = Q R` of
    the `m x n` matrix `A` whose rows are the vectors, which equals
    `sqrt(det(A A^T))` without the Gram form's loss of precision near
    degeneracy (a rounding error of `eps` in `det(A A^T)` becomes
    `sqrt(eps)` in the volume).

    Args:
        a: The vectors, row after row (`m * n` entries); overwritten.
        m: Number of vectors, at most `n`.
        n: Their dimension.

    Returns:
        The volume (1 when `m == 0`).
    """
    var diag = List[Float64](length=m, fill=0)
    return parallelotope_volume(a, m, n, diag)


def parallelotope_volume(
    mut a: List[Float64], m: Int, n: Int, mut diag: List[Float64]
) -> Float64:
    """`parallelotope_volume(a, m, n)` with caller-provided scratch, for
    loops that must not allocate per call.

    Args:
        a: The vectors, row after row (`m * n` entries); overwritten.
        m: Number of vectors, at most `n`.
        n: Their dimension.
        diag: Scratch of at least `m` entries; overwritten.

    Returns:
        The volume (1 when `m == 0`).
    """
    if not _householder_rows(a, m, n, diag):
        return 0
    var product: Float64 = 1
    for j in range(m):
        product *= abs(diag[j])
    return product


def dual_basis(
    mut a: List[Float64], m: Int, n: Int, mut work: List[Float64]
) -> Bool:
    """Replace `m` independent vectors in n-space by their dual basis in
    their span.

    The rows `g_i` of the result satisfy `g_i . a_j = delta_ij` and lie in
    the span of the `a_j`: they are the rows of `(A A^T)^-1 A`, the
    gradients of the barycentric coordinates when the `a_j` are the edges
    `x_j - x_0` of a simplex. They are computed from a Householder QR
    factorization `A^T = Q R` as the rows of `R^-1 Q^T`, so for a simplex
    of width `h` and size `L` the error is about `eps L / h` relative; the
    Gram matrix `A A^T` would square that.

    Args:
        a: The vectors, row after row (`m * n` entries); overwritten with
            the dual basis.
        m: Number of vectors, at most `n`.
        n: Their dimension.
        work: Scratch space, resized to `m * (n + 1)` entries if smaller.

    Returns:
        False, leaving `a` undefined, if the vectors are linearly dependent
        (a zero diagonal entry of `R`).
    """
    if len(work) < m * (n + 1):
        work.resize(m * (n + 1), 0)
    var diag = List[Float64](length=m, fill=0)
    if not _householder_rows(a, m, n, diag):
        return False
    # g_c is column c of Q [R^-T; 0]: solve R^T y = e_c, pad y to n
    # entries, and apply H_(m-1), ..., H_0 (Q = H_0 ... H_(m-1))
    for c in range(m):
        var w = c * n
        for i in range(n):
            work[w + i] = 0
        for i in range(c, m):
            var t: Float64 = 1 if i == c else 0
            for k in range(c, i):
                t -= a[i * n + k] * work[w + k]  # R[k, i] is in row i
            work[w + i] = t / diag[i]
        for j in range(m - 1, -1, -1):
            var vv: Float64 = 0
            var vw: Float64 = 0
            for k in range(j, n):
                vv += a[j * n + k] * a[j * n + k]
                vw += a[j * n + k] * work[w + k]
            var f = 2 * vw / vv
            for k in range(j, n):
                work[w + k] -= f * a[j * n + k]
    for i in range(m * n):
        a[i] = work[i]
    return True


def frame_dual_basis(
    mut a: List[Float64], m: Int, n: Int, mut out: List[Float64]
) -> Bool:
    """The dual basis of `m` independent vectors in n-space, as coordinates
    in an orthonormal frame of their span.

    With `A^T = Q R` (Householder QR of the `m x n` matrix `A` whose rows
    are the vectors), the dual basis of `dual_basis` is the rows of
    `R^-1 Q^T`; this returns `R^-1`, its coordinates in the frame `Q`.
    Inner products and Gram determinants of the dual vectors are the same
    in either form, and the triangular `R^-1` keeps their minors free of
    the cancellation a Gram matrix would bring.

    Args:
        a: The vectors, row after row (`m * n` entries); overwritten.
        m: Number of vectors, at most `n`.
        n: Their dimension.
        out: Set to the `m x m` row-major `R^-1` (resized if smaller).

    Returns:
        False, leaving `out` undefined, if the vectors are linearly
        dependent.
    """
    var diag = List[Float64](length=m, fill=0)
    return frame_dual_basis(a, m, n, out, diag)


def frame_dual_basis(
    mut a: List[Float64],
    m: Int,
    n: Int,
    mut out: List[Float64],
    mut diag: List[Float64],
) -> Bool:
    """`frame_dual_basis(a, m, n, out)` with caller-provided scratch, for
    loops that must not allocate per call.

    Args:
        a: The vectors, row after row (`m * n` entries); overwritten.
        m: Number of vectors, at most `n`.
        n: Their dimension.
        out: Set to the `m x m` row-major `R^-1` (resized if smaller).
        diag: Scratch of at least `m` entries; overwritten.

    Returns:
        False, leaving `out` undefined, if the vectors are linearly
        dependent.
    """
    if len(out) < m * m:
        out.resize(m * m, 0)
    if not _householder_rows(a, m, n, diag):
        return False
    # column c of R^-1 solves R x = e_c; R[i, k] (k > i) is in row k
    for c in range(m):
        for i in range(m - 1, -1, -1):
            var t: Float64 = 1 if i == c else 0
            for k in range(i + 1, c + 1):
                t -= a[k * n + i] * out[k * m + c]
            out[i * m + c] = t / diag[i] if i <= c else 0
    return True


def _householder_rows(
    mut a: List[Float64], m: Int, n: Int, mut diag: List[Float64]
) -> Bool:
    """Householder QR of `A^T`, for the `m x n` row-major `A` in `a`.

    Afterwards row `j` of `a` holds `R[k, j]` in columns `k < j` and the
    Householder vector `v_j` in columns `j..n-1` (`H_j = I - 2 v v^T / v^T
    v`), and `diag[j] = R[j, j]`. Returns False if a column of `A^T` is
    zero after the earlier reflections (dependent rows).
    """
    for j in range(m):
        var r = j * n
        var norm2: Float64 = 0
        for k in range(j, n):
            norm2 += a.unsafe_get(r + k) * a.unsafe_get(r + k)
        var norm = sqrt(norm2)
        if norm == 0:
            return False
        # reflect row j onto -sign(x0) |x| e_j, and every later row with
        # it; row j itself becomes the Householder vector
        var x0 = a.unsafe_get(r + j)
        var alpha = -norm if x0 >= 0 else norm
        diag[j] = alpha
        a.unsafe_set(r + j, x0 - alpha)
        var vv: Float64 = 0
        for k in range(j, n):
            vv += a.unsafe_get(r + k) * a.unsafe_get(r + k)
        for l in range(j + 1, m):
            var dot: Float64 = 0
            for k in range(j, n):
                dot += a.unsafe_get(r + k) * a.unsafe_get(l * n + k)
            var f = 2 * dot / vv
            for k in range(j, n):
                a.unsafe_set(
                    l * n + k, a.unsafe_get(l * n + k) - f * a.unsafe_get(r + k)
                )
    return True


# ===----------------------------------------------------------------------=== #
# Private helpers
# ===----------------------------------------------------------------------=== #


def _radix_sort_rows(a: IntArray, mut perm: List[Int], lo: Int, span: Int):
    """Stable LSD radix sort of `perm` by the rows of `a`, whose entries lie
    in `[lo, lo + span)`."""
    var n = rows_of(a)
    var c = cols_of(a)
    var tmp = List[Int](length=n, fill=0)
    var count = List[Int](length=span + 1, fill=0)
    var cp = count.unsafe_ptr()
    for col in range(c - 1, -1, -1):
        for v in range(span + 1):
            cp[unsafe_offset=v] = 0
        for i in range(n):
            cp[unsafe_offset=Int(a.unsafe_get(i * c + col)) - lo + 1] += 1
        for v in range(span):
            cp[unsafe_offset=v + 1] += cp[unsafe_offset=v]
        var pp = perm.unsafe_ptr()
        var tp = tmp.unsafe_ptr()
        for i in range(n):
            var r = pp[unsafe_offset=i]
            var key = Int(a.unsafe_get(r * c + col)) - lo
            tp[unsafe_offset=cp[unsafe_offset=key]] = r
            cp[unsafe_offset=key] += 1
        swap(perm, tmp)


def _merge_sort_rows(
    a: IntArray, mut perm: List[Int], mut scratch: List[Int], lo: Int, hi: Int
):
    if hi - lo < 2:
        return
    if hi - lo <= 16:
        for i in range(lo + 1, hi):
            var key = perm[i]
            var j = i - 1
            while j >= lo and compare_rows(a, perm[j], a, key) > 0:
                perm[j + 1] = perm[j]
                j -= 1
            perm[j + 1] = key
        return
    var mid = (lo + hi) // 2
    _merge_sort_rows(a, perm, scratch, lo, mid)
    _merge_sort_rows(a, perm, scratch, mid, hi)
    var i = lo
    var j = mid
    var k = lo
    while i < mid and j < hi:
        if compare_rows(a, perm[j], a, perm[i]) < 0:
            scratch[k] = perm[j]
            j += 1
        else:
            scratch[k] = perm[i]
            i += 1
        k += 1
    while i < mid:
        scratch[k] = perm[i]
        i += 1
        k += 1
    while j < hi:
        scratch[k] = perm[j]
        j += 1
        k += 1
    for t in range(lo, hi):
        perm[t] = scratch[t]
