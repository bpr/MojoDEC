# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
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

import numojo as nm
from numojo.prelude import *

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
    primary key.

    Args:
        a: A 2-D integer array.

    Returns:
        The row indices of `a` in sorted order.
    """
    var n = rows_of(a)
    var perm = List[Int](capacity=n)
    for i in range(n):
        perm.append(i)
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
    for i in range(r):
        # insertion sort: rows are short (simplex vertex lists)
        for j in range(1, c):
            var key = a.unsafe_get(i * c + j)
            var k = j - 1
            while k >= 0 and a.unsafe_get(i * c + k) > key:
                a.unsafe_set(i * c + k + 1, a.unsafe_get(i * c + k))
                k -= 1
            a.unsafe_set(i * c + k + 1, key)


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


def dot(x: RealArray, y: RealArray) -> Float64:
    """Inner product of two 1-D real arrays of equal length.

    Args:
        x: The first vector.
        y: The second vector (length not checked).

    Returns:
        The sum of `x[i] * y[i]`.
    """
    var s: Float64 = 0
    for i in range(x.size):
        s += x.unsafe_get(i) * y.unsafe_get(i)
    return s


def norm2(x: RealArray) -> Float64:
    """Euclidean norm of a real array.

    Args:
        x: The array.

    Returns:
        The square root of the sum of squared entries.
    """
    return dot(x, x) ** 0.5


def axpy(alpha: Float64, x: RealArray, mut y: RealArray):
    """`y += alpha * x` for 1-D real arrays.

    Args:
        alpha: The scale factor.
        x: The vector to add.
        y: The vector to update (length not checked).
    """
    for i in range(x.size):
        y.unsafe_set(i, y.unsafe_get(i) + alpha * x.unsafe_get(i))


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
    for i in range(x.size):
        y.unsafe_set(i, alpha * x.unsafe_get(i))
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


# ===----------------------------------------------------------------------=== #
# Private helpers
# ===----------------------------------------------------------------------=== #


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
