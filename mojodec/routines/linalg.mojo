# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Sparse linear solvers (`mojodec.routines.linalg`).
==================================================
The parts of `scipy.sparse.linalg` and `scipy.linalg` that PyDEC's examples
use, over `CSRMatrix` and NuMojo arrays.

- `cg`: conjugate gradients for symmetric positive (semi)definite systems.
- `lsqr`: Paige and Saunders' LSQR for least-squares problems.
- `spsolve`: a direct solve (dense pivoted LU; intended for moderate sizes).
- `eigh_generalized`: all eigenpairs of `K x = lambda M x` for symmetric `K`
  and symmetric positive definite `M` (dense).
"""

import numojo as nm
from numojo.prelude import *

from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    RealArray,
    axpy,
    dot,
    fget,
    fset,
    norm2,
    real_array,
    real_vector,
    rows_of,
    cols_of,
)
from mojodec.routines import dense


@fieldwise_init
struct SolveResult(Movable):
    """The result of an iterative solve."""

    var x: RealArray
    """The computed solution."""
    var info: Int
    """0 on convergence, else the number of iterations performed."""
    var iterations: Int
    """Iterations performed."""
    var residual_norm: Float64
    """Norm of the final residual `b - A x` (of `A^T r` for `lsqr`)."""


def cg(
    A: CSRMatrix,
    b: RealArray,
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Int = -1,
) raises -> SolveResult:
    """Conjugate gradients for `A x = b`, `A` symmetric positive semidefinite.

    Stops when `||b - A x|| <= max(rtol * ||b||, atol)`, as SciPy's `cg`
    does. `maxiter` defaults to `10 * n`. Singular but consistent systems
    (e.g. Laplacians with constant kernels) converge to a solution. The
    iteration starts from zero and stops early, unconverged, if
    `p^T A p <= 0`.

    Args:
        A: An `n x n` symmetric positive semidefinite matrix.
        b: The length-`n` right-hand side.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; non-positive means `10 * n`.

    Returns:
        The solution with its convergence flag, iteration count and
        residual norm.

    Raises:
        If `A` is not square or `b` does not have length `n`.
    """
    var n = A.nrows
    if A.ncols != n or b.size != n:
        raise Error("cg: dimension mismatch")
    var limit = maxiter if maxiter > 0 else 10 * n
    var x = real_vector(n)
    var r = b.copy()
    var bnorm = norm2(b)
    var tol = max(rtol * bnorm, atol)
    var rr = dot(r, r)
    if rr**0.5 <= tol:
        return SolveResult(x^, 0, 0, rr**0.5)
    var p = r.copy()
    var it = 0
    while it < limit:
        it += 1
        var Ap = A @ p
        var pAp = dot(p, Ap)
        if pAp <= 0:
            break
        var alpha = rr / pAp
        axpy(alpha, p, x)
        axpy(-alpha, Ap, r)
        var rr_new = dot(r, r)
        if rr_new**0.5 <= tol:
            return SolveResult(x^, 0, it, rr_new**0.5)
        var beta = rr_new / rr
        for i in range(n):
            p.unsafe_set(i, r.unsafe_get(i) + beta * p.unsafe_get(i))
        rr = rr_new
    return SolveResult(x^, it, it, rr**0.5)


def lsqr(
    A: CSRMatrix,
    b: RealArray,
    atol: Float64 = 1e-8,
    btol: Float64 = 1e-8,
    iter_lim: Int = -1,
) raises -> SolveResult:
    """LSQR: the minimum-norm least-squares solution of `A x ~= b`.

    Follows Paige and Saunders (ACM TOMS 8, 1982) without damping; stops on
    the same tests as SciPy's `lsqr` (`atol`, `btol`). `iter_lim` defaults
    to `2 * ncols`. `residual_norm` is `||A^T r||`.

    Args:
        A: An `m x n` matrix.
        b: The length-`m` right-hand side.
        atol: Tolerance on the relative size of `A^T r`, and of `A`'s
            contribution to the residual.
        btol: Tolerance on the relative size of the residual `r`.
        iter_lim: Iteration limit; non-positive means `2 * n`.

    Returns:
        The solution with its convergence flag, iteration count and
        `||A^T r||`.

    Raises:
        If `b` does not have length `m`.
    """
    var m = A.nrows
    var n = A.ncols
    if b.size != m:
        raise Error("lsqr: dimension mismatch")
    var limit = iter_lim if iter_lim > 0 else 2 * n
    var At = A.transpose()
    var x = real_vector(n)
    var u = b.copy()
    var beta = norm2(u)
    var bnorm = beta
    if beta == 0:
        return SolveResult(x^, 0, 0, 0.0)
    _scale(u, 1.0 / beta)
    var v = At @ u
    var alpha = norm2(v)
    if alpha == 0:
        return SolveResult(x^, 0, 0, 0.0)
    _scale(v, 1.0 / alpha)
    var w = v.copy()
    var phibar = beta
    var rhobar = alpha
    var anorm: Float64 = 0
    var arnorm = alpha * beta
    var it = 0
    while it < limit:
        it += 1
        var Av = A @ v
        for i in range(m):
            u.unsafe_set(i, Av.unsafe_get(i) - alpha * u.unsafe_get(i))
        beta = norm2(u)
        if beta > 0:
            _scale(u, 1.0 / beta)
        anorm = (anorm * anorm + alpha * alpha + beta * beta) ** 0.5
        var Atu = At @ u
        for i in range(n):
            v.unsafe_set(i, Atu.unsafe_get(i) - beta * v.unsafe_get(i))
        alpha = norm2(v)
        if alpha > 0:
            _scale(v, 1.0 / alpha)
        var rho = (rhobar * rhobar + beta * beta) ** 0.5
        var c = rhobar / rho
        var s = beta / rho
        var theta = s * alpha
        rhobar = -c * alpha
        var phi = c * phibar
        phibar = s * phibar
        axpy(phi / rho, w, x)
        for i in range(n):
            w.unsafe_set(i, v.unsafe_get(i) - (theta / rho) * w.unsafe_get(i))
        var rnorm = phibar
        arnorm = alpha * abs(s * phi)
        var xnorm = norm2(x)
        var test1 = rnorm / bnorm
        var test2 = arnorm / (anorm * rnorm) if rnorm > 0 else 0.0
        var rtol = btol + atol * anorm * xnorm / bnorm
        if test1 <= rtol or test2 <= atol or alpha == 0:
            return SolveResult(x^, 0, it, arnorm)
    return SolveResult(x^, it, it, arnorm)


def spsolve(A: CSRMatrix, b: RealArray) raises -> RealArray:
    """Solve the square non-singular system `A x = b` directly.

    Factors a dense copy of `A` with partial pivoting, so it suits the
    moderate sizes of the examples (a few thousand unknowns).

    Args:
        A: An `n x n` non-singular matrix.
        b: The length-`n` right-hand side.

    Returns:
        The length-`n` solution `x`.

    Raises:
        If `A` is not square, `b` does not have length `n`, or `A` is
            singular.
    """
    if A.nrows != A.ncols or b.size != A.nrows:
        raise Error("spsolve: dimension mismatch")
    return dense.solve(A.todense(), b)


def eigh_generalized(
    K: RealArray, M: RealArray
) raises -> Tuple[RealArray, RealArray]:
    """Eigenpairs of `K x = lambda M x`, `K` symmetric, `M` symmetric positive definite.

    Reduces to the standard problem `L^-1 K L^-T y = lambda y` with
    `M = L L^T` and solves it with `dense.eigh`. Returns the eigenvalues
    in ascending order and the eigenvectors `x = L^-T y` as columns,
    `M`-orthonormal.

    Args:
        K: An `n x n` symmetric matrix.
        M: An `n x n` symmetric positive definite matrix.

    Returns:
        The `n` eigenvalues in ascending order, and the `n x n` matrix of
        eigenvectors as columns.

    Raises:
        If `M` is not positive definite, or the eigenvalue iteration does
            not converge.
    """
    var L = dense.cholesky(M)
    # C = L^-1 K L^-T, by two triangular solves
    var Y = _forward(L, K)
    var C = _forward(L, _transpose(Y))
    var wv = dense.eigh(C)
    var X = _backward_transposed(L, wv[1])
    return (wv[0].copy(), X^)


def _scale(mut x: RealArray, s: Float64):
    for i in range(x.size):
        x.unsafe_set(i, s * x.unsafe_get(i))


def _transpose(A: RealArray) raises -> RealArray:
    var r = rows_of(A)
    var c = cols_of(A)
    var out = real_array(c, r)
    for i in range(r):
        for j in range(c):
            fset(out, j, i, fget(A, i, j))
    return out^


def _forward(L: RealArray, B: RealArray) raises -> RealArray:
    """Solve `L X = B` for lower-triangular `L`."""
    var n = rows_of(L)
    var m = cols_of(B)
    var X = real_array(n, m)
    for c in range(m):
        for i in range(n):
            var s = fget(B, i, c)
            for j in range(i):
                s -= fget(L, i, j) * fget(X, j, c)
            fset(X, i, c, s / fget(L, i, i))
    return X^


def _backward_transposed(L: RealArray, B: RealArray) raises -> RealArray:
    """Solve `L^T X = B` for lower-triangular `L`."""
    var n = rows_of(L)
    var m = cols_of(B)
    var X = real_array(n, m)
    for c in range(m):
        for i in range(n - 1, -1, -1):
            var s = fget(B, i, c)
            for j in range(i + 1, n):
                s -= fget(L, j, i) * fget(X, j, c)
            fset(X, i, c, s / fget(L, i, i))
    return X^
