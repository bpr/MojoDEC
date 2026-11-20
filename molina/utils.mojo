# ===----------------------------------------------------------------------=== #
# molina: linear algebra in Mojo (sparse and dense matrices, direct and
# iterative solvers, eigensolvers, multigrid), split from MojoDEC.
# ===----------------------------------------------------------------------=== #
"""
Vector helpers (`molina.utils`).
================================
Builders and level-1 kernels for `RealArray`: zero arrays, arrays from
lists, a fixed probe vector, `(i, j)` access to matrices, and `dot`,
`norm2`, `axpy`, `scaled`.

```mojo
from molina.utils import axpy, dot, real_vector_from_list

var x = real_vector_from_list([1.0, 2.0, 3.0])
var y = real_vector_from_list([1.0, 1.0, 1.0])
axpy(2.0, x, y)                         # y = [3, 5, 7]
print(dot(x, y))                        # 34.0
```
"""

from std.math import sqrt

from molina.real_array import RealArray


def real_array(rows: Int, cols: Int) raises -> RealArray:
    """A zero-filled `rows x cols` real array.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        A `rows x cols` array of zeros.

    Raises:
        If `rows` or `cols` is negative.
    """
    return RealArray.zeros(rows, cols)


def real_vector(n: Int) raises -> RealArray:
    """A zero-filled 1-D real array of length `n`.

    Args:
        n: The length.

    Returns:
        A length-`n` array of zeros.

    Raises:
        If `n` is negative.
    """
    return RealArray.zeros(n)


def real_array_from_rows(rows: List[List[Float64]]) raises -> RealArray:
    """Build a real array from nested lists of equal length.

    Args:
        rows: The rows, all of the same length.

    Returns:
        A `len(rows) x len(rows[0])` real array (`0 x 0` when `rows` is
            empty).

    Raises:
        If the rows have different lengths.
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
        If `n` is negative.
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
        Never; `raises` for compatibility.
    """
    return RealArray(values=values.copy())


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
    return a.unsafe_get(i * a.cols() + j)


@always_inline
def fset(mut a: RealArray, i: Int, j: Int, value: Float64):
    """Set element `(i, j)` of a C-contiguous 2-D real array.

    Args:
        a: The array to modify.
        i: The row (not bounds-checked).
        j: The column (not bounds-checked).
        value: The new value.
    """
    a.unsafe_set(i * a.cols() + j, value)


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
        Never; `raises` for compatibility.
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
