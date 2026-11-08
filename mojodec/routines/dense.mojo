# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Small dense linear algebra (`mojodec.routines.dense`).
======================================================
LU factorization with row partial pivoting on NuMojo arrays.

NuMojo 0.12's `linalg.solve` factors without row exchanges, which breaks on
the saddle-point systems DEC builds (the circumcenter system has a zero
diagonal block, and a vertex at the origin gives a zero first pivot). These
routines pivot during elimination, as LAPACK's `getrf` does.
"""

from std.math import sqrt
import numojo as nm
from numojo.prelude import *

from mojodec.core.utils import (
    RealArray,
    real_array,
    real_vector,
    rows_of,
    cols_of,
)


comptime _W = 8
"""SIMD width of the inner loops: 8 doubles."""


struct LUFactors(Movable):
    """The factors `P A = L U` packed in one array, with the row permutation."""

    var lu: RealArray
    """`L` (unit diagonal, strictly below) and `U` (on and above) packed."""
    var perm: List[Int]
    """Row `i` of `P A` is row `perm[i]` of `A`."""
    var sign: Float64
    """Sign of the permutation (+1 or -1)."""

    def __init__(
        out self, var lu: RealArray, var perm: List[Int], sign: Float64
    ):
        """Wrap packed factors and their permutation.

        Args:
            lu: `L` and `U` packed in one `n x n` array.
            perm: The row permutation.
            sign: The permutation's sign.
        """
        self.lu = lu^
        self.perm = perm^
        self.sign = sign


struct EighResult(Copyable, Movable):
    """The eigenvalues and eigenvectors of a symmetric (or symmetric-definite)
    eigenproblem."""

    var values: RealArray
    """The `n` eigenvalues, in ascending order."""
    var vectors: RealArray
    """The `n x n` eigenvectors: column `i` belongs to `values[i]`."""

    def __init__(out self, var values: RealArray, var vectors: RealArray):
        """A result from its parts.

        Args:
            values: The eigenvalues, ascending.
            vectors: The eigenvectors, as columns.
        """
        self.values = values^
        self.vectors = vectors^


def lu_factor(A: RealArray) raises -> LUFactors:
    """Factor a square matrix as `P A = L U` with partial pivoting.

    Args:
        A: An `n x n` matrix.

    Returns:
        The packed factors and row permutation.

    Raises:
        If `A` is not square, or is singular (an exactly zero pivot column;
            nearly singular matrices factor without error).
    """
    if A.ndim != 2 or rows_of(A) != cols_of(A):
        raise Error("lu_factor: expected a square matrix")
    var n = rows_of(A)
    var lu = real_array(n, n)
    for i in range(n * n):
        lu.unsafe_set(i, A.unsafe_get(i))
    var perm = List[Int](capacity=n)
    for i in range(n):
        perm.append(i)
    var sign: Float64 = 1
    for k in range(n):
        var p = k
        var best = abs(lu.unsafe_get(k * n + k))
        for i in range(k + 1, n):
            var v = abs(lu.unsafe_get(i * n + k))
            if v > best:
                best = v
                p = i
        if best == 0:
            raise Error("lu_factor: matrix is singular")
        if p != k:
            for j in range(n):
                var t = lu.unsafe_get(k * n + j)
                lu.unsafe_set(k * n + j, lu.unsafe_get(p * n + j))
                lu.unsafe_set(p * n + j, t)
            perm.swap_elements(k, p)
            sign = -sign
        var pivot = lu.unsafe_get(k * n + k)
        for i in range(k + 1, n):
            var f = lu.unsafe_get(i * n + k) / pivot
            lu.unsafe_set(i * n + k, f)
            if f != 0:
                _row_sub(lu, i * n + k + 1, k * n + k + 1, f, n - k - 1)
    return LUFactors(lu^, perm^, sign)


def lu_solve(F: LUFactors, B: RealArray) raises -> RealArray:
    """Solve `A X = B` from the factors of `A`; `B` is 1-D or 2-D.

    Args:
        F: The factors of an `n x n` matrix `A`, from `lu_factor`.
        B: A length-`n` vector or `n x m` matrix.

    Returns:
        `X`, with the same shape as `B`.

    Raises:
        If `B` does not have `n` rows.
    """
    var n = rows_of(F.lu)
    var m = 1 if B.ndim == 1 else cols_of(B)
    if rows_of(B) != n:
        raise Error("lu_solve: dimension mismatch")
    var X = real_vector(n) if B.ndim == 1 else real_array(n, m)
    if m == 1:
        for i in range(n):
            var s = B.unsafe_get(F.perm[i])
            X.unsafe_set(i, s - _dot_at(F.lu, i * n, X, 0, i))
        for i in range(n - 1, -1, -1):
            var s = X.unsafe_get(i) - _dot_at(
                F.lu, i * n + i + 1, X, i + 1, n - i - 1
            )
            X.unsafe_set(i, s / F.lu.unsafe_get(i * n + i))
        return X^
    # several right-hand sides: update whole rows of X, which are contiguous
    for i in range(n):
        for c in range(m):
            X.unsafe_set(i * m + c, B.unsafe_get(F.perm[i] * m + c))
        for j in range(i):
            _row_sub(X, i * m, j * m, F.lu.unsafe_get(i * n + j), m)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            _row_sub(X, i * m, j * m, F.lu.unsafe_get(i * n + j), m)
        var pivot = F.lu.unsafe_get(i * n + i)
        for c in range(m):
            X.unsafe_set(i * m + c, X.unsafe_get(i * m + c) / pivot)
    return X^


def solve(A: RealArray, B: RealArray) raises -> RealArray:
    """Solve `A X = B` for a square non-singular `A`; `B` is 1-D or 2-D.

    Args:
        A: An `n x n` matrix.
        B: A length-`n` vector or `n x m` matrix.

    Returns:
        `X`, with the same shape as `B`.

    Raises:
        If `A` is not square or is singular, or `B` does not have `n` rows.
    """
    return lu_solve(lu_factor(A), B)


def det(A: RealArray) raises -> Float64:
    """Determinant of a square matrix (0 if singular).

    Args:
        A: An `n x n` matrix.

    Returns:
        `det(A)`.

    Raises:
        If `A` is not a square 2-D array.
    """
    if A.ndim != 2 or rows_of(A) != cols_of(A):
        raise Error("det: expected a square matrix")
    var n = rows_of(A)
    if n == 1:
        return A.unsafe_get(0)
    if n == 2:
        return A.unsafe_get(0) * A.unsafe_get(3) - A.unsafe_get(
            1
        ) * A.unsafe_get(2)
    try:
        var F = lu_factor(A)
        var d = F.sign
        for i in range(n):
            d *= F.lu.unsafe_get(i * n + i)
        return d
    except:
        return 0.0


def inv(A: RealArray) raises -> RealArray:
    """Inverse of a square non-singular matrix.

    Args:
        A: An `n x n` matrix.

    Returns:
        `A^-1`.

    Raises:
        If `A` is not square or is singular.
    """
    var n = rows_of(A)
    return solve(A, nm.identity[DType.float64](n))


def cholesky(A: RealArray) raises -> RealArray:
    """Lower-triangular `L` with `A = L L^T` for symmetric positive definite `A`.

    Only the lower triangle of `A` is read; squareness is not checked.

    Args:
        A: An `n x n` symmetric positive definite matrix.

    Returns:
        The `n x n` Cholesky factor `L`.

    Raises:
        If `A` is not square or not positive definite (a non-positive
            pivot).
    """
    var n = rows_of(A)
    if n != cols_of(A):
        raise Error("cholesky: expected a square matrix")
    var L = real_array(n, n)
    for j in range(n):
        var s = A.unsafe_get(j * n + j) - _dot_at(L, j * n, L, j * n, j)
        if s <= 0:
            raise Error("cholesky: matrix is not positive definite")
        var d = sqrt(s)
        L.unsafe_set(j * n + j, d)
        for i in range(j + 1, n):
            var t = A.unsafe_get(i * n + j) - _dot_at(L, i * n, L, j * n, j)
            L.unsafe_set(i * n + j, t / d)
    return L^


def eigh(A: RealArray) raises -> EighResult:
    """Eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix.

    Householder reduction to tridiagonal form followed by the implicit QL
    method (EISPACK's `tred2`/`tql2`), O(n^3) with a small constant. NuMojo
    0.12's `linalg.eigh` uses cyclic Jacobi sweeps, which are far slower for
    the few-hundred-unknown problems of the examples.

    Args:
        A: An `n x n` symmetric matrix.

    Returns:
        The `n` eigenvalues in ascending order, and the `n x n` matrix
        whose column `i` is the unit eigenvector for eigenvalue `i`.

    ```mojo
    var eig = dense.eigh(A)
    print(eig.values.unsafe_get(0))          # the smallest eigenvalue
    ```

    Raises:
        If `A` is not square, or the QL iteration does not converge within
            100 sweeps for some eigenvalue.
    """
    var n = rows_of(A)
    if n != cols_of(A):
        raise Error("eigh: expected a square matrix")
    if n == 0:  # _tred2 and _tql2 assume n >= 1
        return EighResult(real_vector(0), real_array(0, 0))
    # V holds the matrix, then the transposed accumulated transformations:
    # row i of V is eigenvector i, so the inner loops run over contiguous
    # memory. A is symmetric, so it is its own transpose to start.
    var V = List[Float64](length=n * n, fill=0)
    for i in range(n * n):
        V.unsafe_set(i, A.unsafe_get(i))
    var d = List[Float64](length=n, fill=0)
    var e = List[Float64](length=n, fill=0)
    _tred2(n, V, d, e)
    _tql2(n, V, d, e)
    var w = real_vector(n)
    var Z = real_array(n, n)
    for i in range(n):
        w.unsafe_set(i, d.unsafe_get(i))
    for i in range(n):
        for j in range(n):
            Z.unsafe_set(i * n + j, V.unsafe_get(j * n + i))
    return EighResult(w^, Z^)


def _tred2(
    n: Int, mut V: List[Float64], mut d: List[Float64], mut e: List[Float64]
):
    for j in range(n):
        d.unsafe_set(j, V.unsafe_get(j * n + (n - 1)))
    for i in range(n - 1, 0, -1):
        var scale: Float64 = 0
        var h: Float64 = 0
        for k in range(i):
            scale += abs(d.unsafe_get(k))
        if scale == 0:
            e.unsafe_set(i, d.unsafe_get(i - 1))
            for j in range(i):
                d.unsafe_set(j, V.unsafe_get(j * n + (i - 1)))
                V.unsafe_set(j * n + i, 0)
                V.unsafe_set(i * n + j, 0)
        else:
            for k in range(i):
                d.unsafe_set(k, d.unsafe_get(k) / scale)
                h += d.unsafe_get(k) * d.unsafe_get(k)
            var f = d.unsafe_get(i - 1)
            var g = sqrt(h)
            if f > 0:
                g = -g
            e.unsafe_set(i, scale * g)
            h = h - f * g
            d.unsafe_set(i - 1, f - g)
            for j in range(i):
                e.unsafe_set(j, 0)
            for j in range(i):
                f = d.unsafe_get(j)
                V.unsafe_set(i * n + j, f)
                g = e.unsafe_get(j) + V.unsafe_get(j * n + j) * f
                for k in range(j + 1, i):
                    g += V.unsafe_get(j * n + k) * d.unsafe_get(k)
                    e.unsafe_set(
                        k, e.unsafe_get(k) + (V.unsafe_get(j * n + k) * f)
                    )
                e.unsafe_set(j, g)
            f = 0
            for j in range(i):
                e.unsafe_set(j, e.unsafe_get(j) / h)
                f += e.unsafe_get(j) * d.unsafe_get(j)
            var hh = f / (h + h)
            for j in range(i):
                e.unsafe_set(j, e.unsafe_get(j) - (hh * d.unsafe_get(j)))
            for j in range(i):
                f = d.unsafe_get(j)
                g = e.unsafe_get(j)
                for k in range(j, i):
                    V.unsafe_set(
                        j * n + k,
                        V.unsafe_get(j * n + k)
                        - (f * e.unsafe_get(k) + g * d.unsafe_get(k)),
                    )
                d.unsafe_set(j, V.unsafe_get(j * n + (i - 1)))
                V.unsafe_set(j * n + i, 0)
        d.unsafe_set(i, h)
    for i in range(n - 1):
        V.unsafe_set(i * n + (n - 1), V.unsafe_get(i * n + i))
        V.unsafe_set(i * n + i, 1)
        var h = d.unsafe_get(i + 1)
        if h != 0:
            for k in range(i + 1):
                d.unsafe_set(k, V.unsafe_get((i + 1) * n + k) / h)
            for j in range(i + 1):
                var g: Float64 = 0
                for k in range(i + 1):
                    g += V.unsafe_get((i + 1) * n + k) * V.unsafe_get(j * n + k)
                for k in range(i + 1):
                    V.unsafe_set(
                        j * n + k,
                        V.unsafe_get(j * n + k) - (g * d.unsafe_get(k)),
                    )
        for k in range(i + 1):
            V.unsafe_set((i + 1) * n + k, 0)
    for j in range(n):
        d.unsafe_set(j, V.unsafe_get(j * n + (n - 1)))
        V.unsafe_set(j * n + (n - 1), 0)
    V.unsafe_set((n - 1) * n + (n - 1), 1)
    e.unsafe_set(0, 0)


def _tql2(
    n: Int, mut V: List[Float64], mut d: List[Float64], mut e: List[Float64]
) raises:
    for i in range(1, n):
        e.unsafe_set(i - 1, e.unsafe_get(i))
    e.unsafe_set(n - 1, 0)
    var f: Float64 = 0
    var tst1: Float64 = 0
    var eps: Float64 = 2.0**-52
    for l in range(n):
        tst1 = max(tst1, abs(d.unsafe_get(l)) + abs(e.unsafe_get(l)))
        var m = l
        while m < n:
            if abs(e.unsafe_get(m)) <= eps * tst1:
                break
            m += 1
        if m > l:
            var iters = 0
            while True:
                iters += 1
                if iters > 100:
                    raise Error("eigh: QL iteration did not converge")
                var g = d.unsafe_get(l)
                var p = (d.unsafe_get(l + 1) - g) / (2 * e.unsafe_get(l))
                var r = _hypot(p, 1.0)
                if p < 0:
                    r = -r
                d.unsafe_set(l, e.unsafe_get(l) / (p + r))
                d.unsafe_set(l + 1, e.unsafe_get(l) * (p + r))
                var dl1 = d.unsafe_get(l + 1)
                var h = g - d.unsafe_get(l)
                for i in range(l + 2, n):
                    d.unsafe_set(i, d.unsafe_get(i) - h)
                f += h
                p = d.unsafe_get(m)
                var c: Float64 = 1
                var c2 = c
                var c3 = c
                var el1 = e.unsafe_get(l + 1)
                var s: Float64 = 0
                var s2: Float64 = 0
                for i in range(m - 1, l - 1, -1):
                    c3 = c2
                    c2 = c
                    s2 = s
                    g = c * e.unsafe_get(i)
                    h = c * p
                    r = _hypot(p, e.unsafe_get(i))
                    e.unsafe_set(i + 1, s * r)
                    s = e.unsafe_get(i) / r
                    c = p / r
                    p = c * d.unsafe_get(i) - s * g
                    d.unsafe_set(i + 1, h + s * (c * g + s * d.unsafe_get(i)))
                    _rotate_rows(V, n, i, c, s)
                p = -s * s2 * c3 * el1 * e.unsafe_get(l) / dl1
                e.unsafe_set(l, s * p)
                d.unsafe_set(l, c * p)
                if abs(e.unsafe_get(l)) <= eps * tst1:
                    break
        d.unsafe_set(l, d.unsafe_get(l) + f)
        e.unsafe_set(l, 0)
    # sort eigenvalues ascending, permuting eigenvector columns
    for i in range(n - 1):
        var k = i
        var p = d.unsafe_get(i)
        for j in range(i + 1, n):
            if d.unsafe_get(j) < p:
                k = j
                p = d.unsafe_get(j)
        if k != i:
            d.unsafe_set(k, d.unsafe_get(i))
            d.unsafe_set(i, p)
            for j in range(n):
                var t = V.unsafe_get(i * n + j)
                V.unsafe_set(i * n + j, V.unsafe_get(k * n + j))
                V.unsafe_set(k * n + j, t)


def _dot_at(a: RealArray, i: Int, b: RealArray, j: Int, n: Int) -> Float64:
    """`sum(a[i + t] * b[j + t] for t < n)`, on the flat buffers."""
    var acc = SIMD[DType.float64, _W](0)
    var t = 0
    while t + _W <= n:
        acc += a.unsafe_load[_W](i + t) * b.unsafe_load[_W](j + t)
        t += _W
    var s = acc.reduce_add()
    while t < n:
        s += a.unsafe_get(i + t) * b.unsafe_get(j + t)
        t += 1
    return s


def _row_sub(mut a: RealArray, dst: Int, src: Int, f: Float64, n: Int):
    """`a[dst + t] -= f * a[src + t]` for `t < n`; the ranges must not
    overlap."""
    var t = 0
    while t + _W <= n:
        a.unsafe_store[_W](
            dst + t, a.unsafe_load[_W](dst + t) - f * a.unsafe_load[_W](src + t)
        )
        t += _W
    while t < n:
        a.unsafe_set(dst + t, a.unsafe_get(dst + t) - f * a.unsafe_get(src + t))
        t += 1


def _rotate_rows(mut V: List[Float64], n: Int, i: Int, c: Float64, s: Float64):
    """Apply a Givens rotation to rows `i` and `i + 1` of the `n x n` matrix
    `V`: `(v_i, v_i+1) <- (c v_i - s v_i+1, s v_i + c v_i+1)`."""
    var p = V.unsafe_ptr()
    var a = i * n
    var b = a + n
    var k = 0
    while k + _W <= n:
        var x = p.unsafe_load[width=_W](a + k)
        var y = p.unsafe_load[width=_W](b + k)
        p.unsafe_store(b + k, s * x + c * y)
        p.unsafe_store(a + k, c * x - s * y)
        k += _W
    while k < n:
        var x = p[unsafe_offset=a + k]
        var y = p[unsafe_offset=b + k]
        p[unsafe_offset=b + k] = s * x + c * y
        p[unsafe_offset=a + k] = c * x - s * y
        k += 1


def _hypot(a: Float64, b: Float64) -> Float64:
    var x = abs(a)
    var y = abs(b)
    if x < y:
        var t = x
        x = y
        y = t
    if x == 0:
        return 0
    var r = y / x
    return x * sqrt(1 + r * r)
