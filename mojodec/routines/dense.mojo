# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
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

import numojo as nm
from numojo.prelude import *

from mojodec.core.utils import (
    RealArray,
    real_array,
    real_vector,
    rows_of,
    cols_of,
)


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
                for j in range(k + 1, n):
                    lu.unsafe_set(
                        i * n + j,
                        lu.unsafe_get(i * n + j) - f * lu.unsafe_get(k * n + j),
                    )
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
    for c in range(m):
        for i in range(n):
            var s = B.unsafe_get(F.perm[i] * m + c)
            for j in range(i):
                s -= F.lu.unsafe_get(i * n + j) * X.unsafe_get(j * m + c)
            X.unsafe_set(i * m + c, s)
        for i in range(n - 1, -1, -1):
            var s = X.unsafe_get(i * m + c)
            for j in range(i + 1, n):
                s -= F.lu.unsafe_get(i * n + j) * X.unsafe_get(j * m + c)
            X.unsafe_set(i * m + c, s / F.lu.unsafe_get(i * n + i))
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
        If `A` is not positive definite (a non-positive pivot).
    """
    var n = rows_of(A)
    var L = real_array(n, n)
    for j in range(n):
        var s = A.unsafe_get(j * n + j)
        for k in range(j):
            s -= L.unsafe_get(j * n + k) ** 2
        if s <= 0:
            raise Error("cholesky: matrix is not positive definite")
        var d = s**0.5
        L.unsafe_set(j * n + j, d)
        for i in range(j + 1, n):
            var t = A.unsafe_get(i * n + j)
            for k in range(j):
                t -= L.unsafe_get(i * n + k) * L.unsafe_get(j * n + k)
            L.unsafe_set(i * n + j, t / d)
    return L^


def eigh(A: RealArray) raises -> Tuple[RealArray, RealArray]:
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

    Raises:
        If `A` is not square, or the QL iteration does not converge within
            100 sweeps for some eigenvalue.
    """
    var n = rows_of(A)
    if n != cols_of(A):
        raise Error("eigh: expected a square matrix")
    # V holds the matrix, then the accumulated transformations (row-major)
    var V = List[Float64](length=n * n, fill=0)
    for i in range(n * n):
        V[i] = A.unsafe_get(i)
    var d = List[Float64](length=n, fill=0)
    var e = List[Float64](length=n, fill=0)
    _tred2(n, V, d, e)
    _tql2(n, V, d, e)
    var w = real_vector(n)
    var Z = real_array(n, n)
    for i in range(n):
        w.unsafe_set(i, d[i])
    for i in range(n * n):
        Z.unsafe_set(i, V[i])
    return (w^, Z^)


def _tred2(
    n: Int, mut V: List[Float64], mut d: List[Float64], mut e: List[Float64]
):
    for j in range(n):
        d[j] = V[(n - 1) * n + j]
    for i in range(n - 1, 0, -1):
        var scale: Float64 = 0
        var h: Float64 = 0
        for k in range(i):
            scale += abs(d[k])
        if scale == 0:
            e[i] = d[i - 1]
            for j in range(i):
                d[j] = V[(i - 1) * n + j]
                V[i * n + j] = 0
                V[j * n + i] = 0
        else:
            for k in range(i):
                d[k] /= scale
                h += d[k] * d[k]
            var f = d[i - 1]
            var g = h**0.5
            if f > 0:
                g = -g
            e[i] = scale * g
            h = h - f * g
            d[i - 1] = f - g
            for j in range(i):
                e[j] = 0
            for j in range(i):
                f = d[j]
                V[j * n + i] = f
                g = e[j] + V[j * n + j] * f
                for k in range(j + 1, i):
                    g += V[k * n + j] * d[k]
                    e[k] += V[k * n + j] * f
                e[j] = g
            f = 0
            for j in range(i):
                e[j] /= h
                f += e[j] * d[j]
            var hh = f / (h + h)
            for j in range(i):
                e[j] -= hh * d[j]
            for j in range(i):
                f = d[j]
                g = e[j]
                for k in range(j, i):
                    V[k * n + j] -= f * e[k] + g * d[k]
                d[j] = V[(i - 1) * n + j]
                V[i * n + j] = 0
        d[i] = h
    for i in range(n - 1):
        V[(n - 1) * n + i] = V[i * n + i]
        V[i * n + i] = 1
        var h = d[i + 1]
        if h != 0:
            for k in range(i + 1):
                d[k] = V[k * n + i + 1] / h
            for j in range(i + 1):
                var g: Float64 = 0
                for k in range(i + 1):
                    g += V[k * n + i + 1] * V[k * n + j]
                for k in range(i + 1):
                    V[k * n + j] -= g * d[k]
        for k in range(i + 1):
            V[k * n + i + 1] = 0
    for j in range(n):
        d[j] = V[(n - 1) * n + j]
        V[(n - 1) * n + j] = 0
    V[(n - 1) * n + n - 1] = 1
    e[0] = 0


def _tql2(
    n: Int, mut V: List[Float64], mut d: List[Float64], mut e: List[Float64]
) raises:
    for i in range(1, n):
        e[i - 1] = e[i]
    e[n - 1] = 0
    var f: Float64 = 0
    var tst1: Float64 = 0
    var eps: Float64 = 2.0**-52
    for l in range(n):
        tst1 = max(tst1, abs(d[l]) + abs(e[l]))
        var m = l
        while m < n:
            if abs(e[m]) <= eps * tst1:
                break
            m += 1
        if m > l:
            var iters = 0
            while True:
                iters += 1
                if iters > 100:
                    raise Error("eigh: QL iteration did not converge")
                var g = d[l]
                var p = (d[l + 1] - g) / (2 * e[l])
                var r = _hypot(p, 1.0)
                if p < 0:
                    r = -r
                d[l] = e[l] / (p + r)
                d[l + 1] = e[l] * (p + r)
                var dl1 = d[l + 1]
                var h = g - d[l]
                for i in range(l + 2, n):
                    d[i] -= h
                f += h
                p = d[m]
                var c: Float64 = 1
                var c2 = c
                var c3 = c
                var el1 = e[l + 1]
                var s: Float64 = 0
                var s2: Float64 = 0
                for i in range(m - 1, l - 1, -1):
                    c3 = c2
                    c2 = c
                    s2 = s
                    g = c * e[i]
                    h = c * p
                    r = _hypot(p, e[i])
                    e[i + 1] = s * r
                    s = e[i] / r
                    c = p / r
                    p = c * d[i] - s * g
                    d[i + 1] = h + s * (c * g + s * d[i])
                    for k in range(n):
                        h = V[k * n + i + 1]
                        V[k * n + i + 1] = s * V[k * n + i] + c * h
                        V[k * n + i] = c * V[k * n + i] - s * h
                p = -s * s2 * c3 * el1 * e[l] / dl1
                e[l] = s * p
                d[l] = c * p
                if abs(e[l]) <= eps * tst1:
                    break
        d[l] = d[l] + f
        e[l] = 0
    # sort eigenvalues ascending, permuting eigenvector columns
    for i in range(n - 1):
        var k = i
        var p = d[i]
        for j in range(i + 1, n):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            for j in range(n):
                var t = V[j * n + i]
                V[j * n + i] = V[j * n + k]
                V[j * n + k] = t


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
    return x * (1 + r * r) ** 0.5
