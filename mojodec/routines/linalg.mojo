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
  and symmetric positive definite `M` (dense, or a `DiagonalOperator` such
  as a Hodge star).
"""

from std.math import sqrt
import numojo as nm
from numojo.prelude import *

from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    RealArray,
    axpy,
    dot,
    norm2,
    real_array,
    real_vector,
    rows_of,
    cols_of,
)
from mojodec.routines import dense
from mojodec.routines.dense import EighResult


struct StopReason(Equatable, ImplicitlyCopyable, Writable):
    """Why an iterative solver stopped."""

    var _code: UInt8

    comptime CONVERGED = StopReason(0)
    """The stopping test passed."""
    comptime MAX_ITER = StopReason(1)
    """The iteration limit was reached first."""
    comptime BREAKDOWN = StopReason(2)
    """The iteration could not continue (`cg`: `p^T A p <= 0`, so `A` is
    not positive semidefinite on the Krylov space)."""

    def __init__(out self, code: UInt8):
        """A reason from its code; use the named members instead.

        Args:
            code: 0, 1, or 2.
        """
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        """Whether two reasons are the same.

        Args:
            other: The other reason.

        Returns:
            True if they are equal.
        """
        return self._code == other._code

    def write_to(self, mut writer: Some[Writer]):
        """Write the reason's name.

        Args:
            writer: The destination.
        """
        if self == Self.CONVERGED:
            writer.write("converged")
        elif self == Self.MAX_ITER:
            writer.write("iteration limit reached")
        else:
            writer.write("breakdown")


struct SolveResult(Movable):
    """The result of an iterative solve.

    ```mojo
    var res = cg(A, b, rtol=1e-8)
    if not res.converged():
        print("cg stopped:", res.stop, "after", res.iterations, "iterations")
    var x = res^.take_x()
    ```
    """

    var x: RealArray
    """The computed solution."""
    var stop: StopReason
    """Why the iteration stopped."""
    var iterations: Int
    """Iterations performed."""
    var residual_norm: Float64
    """`||b - A x||` (LSQR's running estimate of it, for `lsqr`)."""
    var normal_residual_norm: Optional[Float64]
    """`||A^T (b - A x)||` for `lsqr`, the quantity least squares drives to
    zero; `None` for `cg`."""

    def __init__(
        out self,
        var x: RealArray,
        stop: StopReason,
        iterations: Int,
        residual_norm: Float64,
        normal_residual_norm: Optional[Float64] = None,
    ):
        """A result from its parts; built by the solvers.

        Args:
            x: The solution.
            stop: Why the iteration stopped.
            iterations: Iterations performed.
            residual_norm: `||b - A x||`.
            normal_residual_norm: `||A^T (b - A x)||`, if computed.
        """
        self.x = x^
        self.stop = stop
        self.iterations = iterations
        self.residual_norm = residual_norm
        self.normal_residual_norm = normal_residual_norm

    def converged(self) -> Bool:
        """Whether the stopping test passed.

        Returns:
            `stop == StopReason.CONVERGED`.
        """
        return self.stop == StopReason.CONVERGED

    def take_x(deinit self) -> RealArray:
        """The solution, moved out of the result.

        Returns:
            `x`, without a copy.
        """
        return self.x^


def cg(
    A: CSRMatrix,
    b: RealArray,
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
) raises -> SolveResult:
    """Conjugate gradients for `A x = b`, `A` symmetric positive semidefinite.

    Stops when `||b - A x|| <= max(rtol * ||b||, atol)`, as SciPy's `cg`
    does. Singular but consistent systems (e.g. Laplacians with constant
    kernels) converge to a solution. The iteration starts from zero and
    stops early with `StopReason.BREAKDOWN` if `p^T A p <= 0`.

    Args:
        A: An `n x n` symmetric positive semidefinite matrix.
        b: The length-`n` right-hand side.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means `10 * n`.

    Returns:
        The solution with why it stopped, the iteration count and the
        residual norm.

    Raises:
        If `A` is not square, `b` does not have length `n`, or `maxiter`
        is negative.
    """
    var n = A.nrows
    if A.ncols != n or b.size != n:
        raise Error("cg: dimension mismatch")
    var limit = maxiter.value() if maxiter else 10 * n
    if limit < 0:
        raise Error("cg: maxiter must be >= 0")
    var x = real_vector(n)
    var r = b.copy()
    var bnorm = norm2(b)
    var tol = max(rtol * bnorm, atol)
    var rr = dot(r, r)
    if sqrt(rr) <= tol:
        return SolveResult(x^, StopReason.CONVERGED, 0, sqrt(rr))
    var p = r.copy()
    var Ap = real_vector(n)
    var it = 0
    while it < limit:
        it += 1
        A.matvec_into(p, Ap)
        var pAp = dot(p, Ap)
        if pAp <= 0:
            return SolveResult(x^, StopReason.BREAKDOWN, it, sqrt(rr))
        var alpha = rr / pAp
        axpy(alpha, p, x)
        axpy(-alpha, Ap, r)
        var rr_new = dot(r, r)
        if sqrt(rr_new) <= tol:
            return SolveResult(x^, StopReason.CONVERGED, it, sqrt(rr_new))
        _xpby(r, rr_new / rr, p)
        rr = rr_new
    return SolveResult(x^, StopReason.MAX_ITER, it, sqrt(rr))


def lsqr(
    A: CSRMatrix,
    b: RealArray,
    atol: Float64 = 1e-8,
    btol: Float64 = 1e-8,
    iter_lim: Optional[Int] = None,
) raises -> SolveResult:
    """LSQR: the minimum-norm least-squares solution of `A x ~= b`.

    Follows Paige and Saunders (ACM TOMS 8, 1982) without damping; stops on
    the same tests as SciPy's `lsqr` (`atol`, `btol`). The result's
    `residual_norm` is `||r||` and `normal_residual_norm` is `||A^T r||`,
    for `r = b - A x`.

    Args:
        A: An `m x n` matrix.
        b: The length-`m` right-hand side.
        atol: Tolerance on the relative size of `A^T r`, and of `A`'s
            contribution to the residual.
        btol: Tolerance on the relative size of the residual `r`.
        iter_lim: Iteration limit; `None` means `2 * n`.

    Returns:
        The solution with why it stopped, the iteration count, `||r||` and
        `||A^T r||`.

    Raises:
        If `b` does not have length `m`, or `iter_lim` is negative.
    """
    var m = A.nrows
    var n = A.ncols
    if b.size != m:
        raise Error("lsqr: dimension mismatch")
    var limit = iter_lim.value() if iter_lim else 2 * n
    if limit < 0:
        raise Error("lsqr: iter_lim must be >= 0")
    var At = A.transpose()
    var x = real_vector(n)
    var u = b.copy()
    var beta = norm2(u)
    var bnorm = beta
    if beta == 0:
        return SolveResult(x^, StopReason.CONVERGED, 0, 0.0, 0.0)
    _scale(u, 1.0 / beta)
    var v = At @ u
    var alpha = norm2(v)
    if alpha == 0:
        # b is orthogonal to the range of A: x = 0 is the solution
        return SolveResult(x^, StopReason.CONVERGED, 0, beta, 0.0)
    _scale(v, 1.0 / alpha)
    var w = v.copy()
    var Av = real_vector(m)
    var Atu = real_vector(n)
    var phibar = beta
    var rhobar = alpha
    var anorm: Float64 = 0
    var arnorm = alpha * beta
    var it = 0
    while it < limit:
        it += 1
        A.matvec_into(v, Av)
        _xpby(Av, -alpha, u)
        beta = norm2(u)
        if beta > 0:
            _scale(u, 1.0 / beta)
        anorm = sqrt(anorm * anorm + alpha * alpha + beta * beta)
        At.matvec_into(u, Atu)
        _xpby(Atu, -beta, v)
        alpha = norm2(v)
        if alpha > 0:
            _scale(v, 1.0 / alpha)
        var rho = sqrt(rhobar * rhobar + beta * beta)
        var c = rhobar / rho
        var s = beta / rho
        var theta = s * alpha
        rhobar = -c * alpha
        var phi = c * phibar
        phibar = s * phibar
        axpy(phi / rho, w, x)
        _xpby(v, -theta / rho, w)
        var rnorm = phibar
        arnorm = alpha * abs(s * phi)
        var xnorm = norm2(x)
        var test1 = rnorm / bnorm
        var test2 = arnorm / (anorm * rnorm) if rnorm > 0 else 0.0
        var rtol = btol + atol * anorm * xnorm / bnorm
        if test1 <= rtol or test2 <= atol or alpha == 0:
            return SolveResult(x^, StopReason.CONVERGED, it, rnorm, arnorm)
    return SolveResult(x^, StopReason.MAX_ITER, it, phibar, arnorm)


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


def eigh_generalized(K: RealArray, M: RealArray) raises -> EighResult:
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
    var eig = dense.eigh(C)
    eig.vectors = _backward_transposed(L, eig.vectors)
    return eig^


def eigh_generalized(K: RealArray, M: DiagonalOperator) raises -> EighResult:
    """Eigenpairs of `K x = lambda M x`, `K` symmetric, `M` diagonal and positive.

    With `M = D^2` the problem is the standard one `D^-1 K D^-1 y = lambda y`,
    `x = D^-1 y`: a scaling of `K` in O(n^2) in place of the Cholesky
    factorization and triangular solves of the dense overload. This is the
    case of a Hodge star as the mass matrix, e.g.
    `eigh_generalized((sc[k].star @ L).todense(), sc[k].star)`.

    Args:
        K: An `n x n` symmetric matrix.
        M: An `n x n` diagonal matrix with positive diagonal.

    Returns:
        The `n` eigenvalues in ascending order, and the `n x n` matrix of
        eigenvectors as columns, `M`-orthonormal.

    Raises:
        If `K` is not `n x n`, a diagonal entry of `M` is not positive, or the
            eigenvalue iteration does not converge.
    """
    var n = len(M)
    if rows_of(K) != n or cols_of(K) != n:
        raise Error("eigh_generalized: K and M differ in shape")
    var scale = List[Float64](capacity=n)
    for i in range(n):
        var m = M.values[i]
        if not m > 0:
            raise Error(
                String(
                    "eigh_generalized: M is not positive definite (M[",
                    i,
                    "] = ",
                    m,
                    ")",
                )
            )
        scale.append(1 / sqrt(m))
    var C = real_array(n, n)
    for i in range(n):
        var si = scale[i]
        for j in range(n):
            C.unsafe_set(i * n + j, K.unsafe_get(i * n + j) * si * scale[j])
    var eig = dense.eigh(C)
    for i in range(n):
        var si = scale[i]
        for j in range(n):
            eig.vectors.unsafe_set(
                i * n + j, eig.vectors.unsafe_get(i * n + j) * si
            )
    return eig^


comptime _W = 8


def _scale(mut x: RealArray, s: Float64):
    var n = x.size
    var i = 0
    while i + _W <= n:
        x.unsafe_store[_W](i, s * x.unsafe_load[_W](i))
        i += _W
    while i < n:
        x.unsafe_set(i, s * x.unsafe_get(i))
        i += 1


def _xpby(x: RealArray, beta: Float64, mut y: RealArray):
    """`y = x + beta * y`, for vectors of equal length."""
    var n = x.size
    var i = 0
    while i + _W <= n:
        y.unsafe_store[_W](
            i, x.unsafe_load[_W](i) + beta * y.unsafe_load[_W](i)
        )
        i += _W
    while i < n:
        y.unsafe_set(i, x.unsafe_get(i) + beta * y.unsafe_get(i))
        i += 1


def _transpose(A: RealArray) raises -> RealArray:
    var r = rows_of(A)
    var c = cols_of(A)
    var out = real_array(c, r)
    for i in range(r):
        for j in range(c):
            out.unsafe_set(j * r + i, A.unsafe_get(i * c + j))
    return out^


def _forward(L: RealArray, B: RealArray) raises -> RealArray:
    """Solve `L X = B` for lower-triangular `L`, a row of `X` at a time."""
    var n = rows_of(L)
    var m = cols_of(B)
    var X = B.copy()
    for i in range(n):
        for j in range(i):
            var l = L.unsafe_get(i * n + j)
            if l != 0:
                dense._row_sub(X, i * m, j * m, l, m)
        _scale_row(X, i * m, m, 1 / L.unsafe_get(i * n + i))
    return X^


def _backward_transposed(L: RealArray, B: RealArray) raises -> RealArray:
    """Solve `L^T X = B` for lower-triangular `L`, a row of `X` at a time."""
    var n = rows_of(L)
    var m = cols_of(B)
    var X = B.copy()
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            var l = L.unsafe_get(j * n + i)
            if l != 0:
                dense._row_sub(X, i * m, j * m, l, m)
        _scale_row(X, i * m, m, 1 / L.unsafe_get(i * n + i))
    return X^


def _scale_row(mut x: RealArray, start: Int, n: Int, s: Float64):
    """`x[start + t] *= s` for `t < n`."""
    var t = 0
    while t + _W <= n:
        x.unsafe_store[_W](start + t, s * x.unsafe_load[_W](start + t))
        t += _W
    while t < n:
        x.unsafe_set(start + t, s * x.unsafe_get(start + t))
        t += 1
