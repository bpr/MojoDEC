# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
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

Each has a typed overload on `MatrixOp`s and `TypedCochain`s, and `spsolve`
one on a `BlockOp` and `CochainPair`s, with some unknowns optionally given. An operator
from one kind of cochain to another, such as the stiffness matrix
`tc.d[0, PRIMAL]().T() @ tc.star[1, PRIMAL]() @ tc.d[0, PRIMAL]()` from primal
0-cochains to dual n-cochains, takes a right-hand side of the output kind
and gives a solution of the input kind:

```mojo
var d0 = tc.d[0, PRIMAL]()
var A = d0.T() @ tc.star[1, PRIMAL]() @ d0    # primal 0 -> dual 2
var b = tc.cochain[2, DUAL]()              # one value per dual cell
var u = cg(A, b, rtol=1e-10).take_x()      # TypedCochain[2, 0, PRIMAL]
```
"""

from std.math import sqrt
import numojo as nm
from numojo.prelude import *

from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.typed import DiagonalOp, MatrixOp, TypedCochain, TypedSubset
from mojodec.core.typed_block import BlockOp, CochainPair
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


struct SolveResult[X: Movable & Deinitable = RealArray](Movable):
    """The result of an iterative solve.

    ```mojo
    var res = cg(A, b, rtol=1e-8)
    if not res.converged():
        print("cg stopped:", res.stop, "after", res.iterations, "iterations")
    var x = res^.take_x()
    ```

    Parameters:
        X: The type of the solution: `RealArray`, or a `TypedCochain` from
            the typed overloads.
    """

    var x: Self.X
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
        var x: Self.X,
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

    def take_x(deinit self) -> Self.X:
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
) raises -> SolveResult[RealArray]:
    """Conjugate gradients for `A x = b`, `A` symmetric positive semidefinite.

    Stops when `||b - A x|| <= max(rtol * ||b||, atol)`, as SciPy's `cg`
    does. Singular but consistent systems (e.g. Laplacians with constant
    kernels) converge to a solution. The iteration starts from zero and
    stops early with `StopReason.BREAKDOWN` if `p^T A p` is not positive (or is NaN).

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
        if not pAp > 0:  # also NaN, from a NaN or infinite entry
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
) raises -> SolveResult[RealArray]:
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
        If `K` and `M` are not both `n x n`, `M` is not positive definite,
            or the eigenvalue iteration does not converge.
    """
    var n = rows_of(M)
    if cols_of(M) != n or rows_of(K) != n or cols_of(K) != n:
        raise Error("eigh_generalized: K and M must both be n x n")
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


# ===----------------------------------------------------------------------=== #
# Typed overloads
# ===----------------------------------------------------------------------=== #


struct TypedEighResult[n: Int, k: Int, primal: Bool](Movable, Sized):
    """The eigenpairs of a typed generalized eigenproblem, the eigenvectors
    being k-cochains.

    ```mojo
    var eig = eigh_generalized(K, tc.star[1, PRIMAL]())
    var mode = eig.mode(0)        # TypedCochain[n, 1, PRIMAL]
    ```

    Parameters:
        n: The complex dimension.
        k: The degree of the eigenvectors.
        primal: Whether the eigenvectors are primal.
    """

    var values: RealArray
    """The eigenvalues, in ascending order."""
    var vectors: RealArray
    """The eigenvectors as columns, as in `EighResult`."""
    var complex: SimplicialComplex
    """The complex the eigenvectors live on."""

    def __init__(
        out self, var eig: EighResult, complex: SimplicialComplex
    ) raises:
        """Type an untyped result, taking its arrays.

        Args:
            eig: The eigenvalues and eigenvectors.
            complex: The complex the eigenvectors live on.

        Raises:
            Only if allocating an empty array fails.
        """
        self.values = real_vector(0)
        self.vectors = real_vector(0)
        swap(self.values, eig.values)
        swap(self.vectors, eig.vectors)
        self.complex = complex

    def __len__(self) -> Int:
        """The number of eigenpairs.

        Returns:
            The number of eigenvalues.
        """
        return self.values.size

    def mode(self, i: Int) raises -> TypedCochain[Self.n, Self.k, Self.primal]:
        """The `i`-th eigenvector, a copy of column `i` of `vectors`.

        Args:
            i: The index of the eigenpair, `0 <= i < len(self)`.

        Returns:
            The k-cochain of `values[i]`.

        Raises:
            If `i` is out of range.
        """
        var m = self.values.size
        if i < 0 or i >= m:
            raise Error("TypedEighResult.mode: index out of range")
        var v = real_vector(m)
        for r in range(m):
            v.unsafe_set(r, self.vectors.unsafe_get(r * m + i))
        return TypedCochain[Self.n, Self.k, Self.primal](self.complex, v^)


def _typed_complex(
    op: Optional[SimplicialComplex], b: SimplicialComplex, caller: String
) raises:
    """Raise unless an operator built on a complex is applied on it."""
    if op and not op.value().is_same(b):
        raise Error(String(caller, ": operands on different complexes"))


def _typed_result[
    n: Int, k: Int, primal: Bool
](var res: SolveResult[RealArray], complex: SimplicialComplex) -> SolveResult[
    TypedCochain[n, k, primal]
]:
    """Wrap the solution of an untyped solve."""
    var stop = res.stop
    var iterations = res.iterations
    var residual_norm = res.residual_norm
    var normal_residual_norm = res.normal_residual_norm
    return SolveResult[TypedCochain[n, k, primal]](
        TypedCochain[n, k, primal](complex, res^.take_x()),
        stop,
        iterations,
        residual_norm,
        normal_residual_norm,
    )


def cg[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    A: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    b: TypedCochain[n, out_k, out_primal],
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
) raises -> SolveResult[TypedCochain[n, in_k, in_primal]]:
    """Conjugate gradients for `A x = b` on cochains; see the untyped overload.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the solution.
        in_primal: Whether the solution is primal.
        out_k: The degree of the right-hand side.
        out_primal: Whether the right-hand side is primal.

    Args:
        A: A symmetric positive semidefinite operator, e.g. a stiffness
            matrix from primal k-cochains to dual (n-k)-cochains.
        b: The right-hand side, of `A`'s output kind.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means 10 times the size.

    Returns:
        The solution, of `A`'s input kind, with why the iteration stopped.

    Raises:
        If `A` and `b` are on different complexes, or as the untyped `cg`.
    """
    _typed_complex(A.complex, b.complex, "cg")
    return _typed_result[n, in_k, in_primal](
        cg(A.mat, b.v, rtol, atol, maxiter), b.complex
    )


def lsqr[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    A: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    b: TypedCochain[n, out_k, out_primal],
    atol: Float64 = 1e-8,
    btol: Float64 = 1e-8,
    iter_lim: Optional[Int] = None,
) raises -> SolveResult[TypedCochain[n, in_k, in_primal]]:
    """LSQR for `A x ~= b` on cochains; see the untyped overload.

    For example, the least-squares potential of a 1-cochain `w` is
    `lsqr(tc.d[0, PRIMAL](), w)`.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the solution.
        in_primal: Whether the solution is primal.
        out_k: The degree of the right-hand side.
        out_primal: Whether the right-hand side is primal.

    Args:
        A: An operator.
        b: The right-hand side, of `A`'s output kind.
        atol: Tolerance on the relative size of `A^T r`.
        btol: Tolerance on the relative size of the residual `r`.
        iter_lim: Iteration limit; `None` means twice the solution size.

    Returns:
        The minimum-norm least-squares solution, of `A`'s input kind.

    Raises:
        If `A` and `b` are on different complexes, or as the untyped
        `lsqr`.
    """
    _typed_complex(A.complex, b.complex, "lsqr")
    return _typed_result[n, in_k, in_primal](
        lsqr(A.mat, b.v, atol, btol, iter_lim), b.complex
    )


def spsolve[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    A: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    b: TypedCochain[n, out_k, out_primal],
) raises -> TypedCochain[n, in_k, in_primal]:
    """Solve `A x = b` on cochains directly; see the untyped overload.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the solution.
        in_primal: Whether the solution is primal.
        out_k: The degree of the right-hand side.
        out_primal: Whether the right-hand side is primal.

    Args:
        A: A non-singular operator with as many output as input cells.
        b: The right-hand side, of `A`'s output kind.

    Returns:
        The solution, of `A`'s input kind.

    Raises:
        If `A` and `b` are on different complexes, `A` is not square, or
        `A` is singular.
    """
    _typed_complex(A.complex, b.complex, "spsolve")
    return TypedCochain[n, in_k, in_primal](b.complex, spsolve(A.mat, b.v))


def eigh_generalized[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    K: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    M: MatrixOp[n, in_k, in_primal, out_k, out_primal],
) raises -> TypedEighResult[n, in_k, in_primal]:
    """Eigenpairs of `K x = lambda M x` with a sparse mass matrix, such as
    a Whitney inner product; see the untyped overload.

    `K` and `M` take the same cochains to the same cochains, e.g. primal
    k-cochains to dual (n-k)-cochains for a stiffness matrix and
    `whitney_innerproduct[k](tc)`.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the eigenvectors.
        in_primal: Whether the eigenvectors are primal.
        out_k: The degree of the cochains `K` and `M` give.
        out_primal: Whether the cochains `K` and `M` give are primal.

    Args:
        K: A symmetric operator.
        M: A symmetric positive definite operator.

    Returns:
        The eigenvalues, ascending, and the eigenvectors as cochains of the
        input kind.

    Raises:
        If either operator was built without a complex, or they are on
        different complexes; or as the untyped `eigh_generalized`.
    """
    if not K.complex or not M.complex:
        raise Error("eigh_generalized: the operators carry no complex")
    var complex = K.complex.value()
    _typed_complex(M.complex, complex, "eigh_generalized")
    var eig = eigh_generalized(K.mat.todense(), M.mat.todense())
    return TypedEighResult[n, in_k, in_primal](eig^, complex)


def eigh_generalized[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    K: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    M: DiagonalOp[n, in_k, in_primal, out_k, out_primal],
) raises -> TypedEighResult[n, in_k, in_primal]:
    """Eigenpairs of `K x = lambda M x` with a diagonal mass matrix, such as
    a Hodge star; see the untyped overload.

    ```mojo
    var S = tc.star[1, PRIMAL]()
    var eig = eigh_generalized(S @ tc.laplacian[1, PRIMAL](), S)
    ```

    Parameters:
        n: The complex dimension.
        in_k: The degree of the eigenvectors.
        in_primal: Whether the eigenvectors are primal.
        out_k: The degree of the cochains `K` and `M` give.
        out_primal: Whether the cochains `K` and `M` give are primal.

    Args:
        K: A symmetric operator.
        M: A diagonal operator with positive entries.

    Returns:
        The eigenvalues, ascending, and the eigenvectors as cochains of the
        input kind.

    Raises:
        If either operator was built without a complex, or they are on
        different complexes; or as the untyped `eigh_generalized`.
    """
    if not K.complex or not M.complex:
        raise Error("eigh_generalized: the operators carry no complex")
    var complex = K.complex.value()
    _typed_complex(M.complex, complex, "eigh_generalized")
    var eig = eigh_generalized(K.mat.todense(), M.diag)
    return TypedEighResult[n, in_k, in_primal](eig^, complex)


def _split[
    n: Int, k1: Int, p1: Bool, k2: Int, p2: Bool
](x: RealArray, n1: Int, complex: SimplicialComplex) raises -> CochainPair[
    n, k1, p1, k2, p2
]:
    """A stacked vector cut back into its two cochains."""
    var first = real_vector(n1)
    var second = real_vector(x.size - n1)
    for i in range(n1):
        first.unsafe_set(i, x.unsafe_get(i))
    for i in range(x.size - n1):
        second.unsafe_set(i, x.unsafe_get(n1 + i))
    return CochainPair(
        TypedCochain[n, k1, p1](complex, first^),
        TypedCochain[n, k2, p2](complex, second^),
    )


def spsolve[
    n: Int,
    in1_k: Int,
    in1_p: Bool,
    in2_k: Int,
    in2_p: Bool,
    out1_k: Int,
    out1_p: Bool,
    out2_k: Int,
    out2_p: Bool,
](
    A: BlockOp[n, in1_k, in1_p, in2_k, in2_p, out1_k, out1_p, out2_k, out2_p],
    b: CochainPair[n, out1_k, out1_p, out2_k, out2_p],
) raises -> CochainPair[n, in1_k, in1_p, in2_k, in2_p]:
    """Solve the block system `A x = b` directly; see the untyped overload.

    Parameters:
        n: The complex dimension.
        in1_k: The degree of the first unknown.
        in1_p: Whether the first unknown is primal.
        in2_k: The degree of the second unknown.
        in2_p: Whether the second unknown is primal.
        out1_k: The degree of the first equation.
        out1_p: Whether the first equation is primal.
        out2_k: The degree of the second equation.
        out2_p: Whether the second equation is primal.

    Args:
        A: A non-singular block operator with as many equations as
            unknowns.
        b: The right-hand side, of `A`'s output kinds.

    Returns:
        The unknowns, of `A`'s input kinds.

    Raises:
        If `A` and `b` are on different complexes, `A` is not square, or
        `A` is singular.
    """
    var complex = b.first.complex
    _typed_complex(A.complex, complex, "spsolve")
    var x = spsolve(A.to_csr(), b.stacked())
    return _split[n, in1_k, in1_p, in2_k, in2_p](x, A.a11.mat.ncols, complex)


def spsolve[
    n: Int,
    in1_k: Int,
    in1_p: Bool,
    in2_k: Int,
    in2_p: Bool,
    out1_k: Int,
    out1_p: Bool,
    out2_k: Int,
    out2_p: Bool,
](
    A: BlockOp[n, in1_k, in1_p, in2_k, in2_p, out1_k, out1_p, out2_k, out2_p],
    b: CochainPair[n, out1_k, out1_p, out2_k, out2_p],
    *,
    values: CochainPair[n, in1_k, in1_p, in2_k, in2_p],
    fixed_first: TypedSubset[n, in1_k, in1_p],
    fixed_second: TypedSubset[n, in2_k, in2_p],
) raises -> CochainPair[n, in1_k, in1_p, in2_k, in2_p]:
    """Solve the block system `A x = b` with some unknowns given.

    The unknowns on the cells of `fixed_first` and `fixed_second` take
    their `values`; the equations on the same cells are dropped, and the
    rest are solved for the other unknowns, with the given ones moved to
    the right-hand side. This needs each equation to live on the cells of
    its unknown: the first equation of the dual kind of the first unknown
    (primal k-cochains and dual (n-k)-cochains), and the same for the
    second, as in a mixed formulation built from Hodge stars, `d`, and its
    transpose. The other kinds do not compile.

    ```mojo
    # Darcy flow: fluxes given on the boundary edges, one pressure pinned
    var x = spsolve(
        A, b, values=known, fixed_first=boundary_edges, fixed_second=pinned
    )
    ```

    Parameters:
        n: The complex dimension.
        in1_k: The degree of the first unknown.
        in1_p: Whether the first unknown is primal.
        in2_k: The degree of the second unknown.
        in2_p: Whether the second unknown is primal.
        out1_k: The degree of the first equation, `n - in1_k`.
        out1_p: Whether the first equation is primal, `not in1_p`.
        out2_k: The degree of the second equation, `n - in2_k`.
        out2_p: Whether the second equation is primal, `not in2_p`.

    Args:
        A: A block operator, non-singular on the unknowns left free.
        b: The right-hand side, of `A`'s output kinds.
        values: The unknowns, read only on the fixed cells.
        fixed_first: The cells where the first unknown is given.
        fixed_second: The cells where the second unknown is given.

    Returns:
        The unknowns: `values` on the fixed cells, the solution elsewhere.

    Raises:
        If the operands are on different complexes, or the reduced system
        is singular.
    """
    comptime assert (
        out1_k == n - in1_k
        and out1_p == (not in1_p)
        and out2_k == n - in2_k
        and out2_p == (not in2_p)
    ), "spsolve: each equation must be of the dual kind of its unknown"
    var complex = b.first.complex
    _typed_complex(A.complex, complex, "spsolve")
    _typed_complex(Optional(values.first.complex), complex, "spsolve")
    _typed_complex(Optional(fixed_first.complex), complex, "spsolve")
    _typed_complex(Optional(fixed_second.complex), complex, "spsolve")
    var M = A.to_csr()
    var x = values.stacked()
    var n1 = A.a11.mat.ncols
    var free = List[Int]()
    for i in range(n1):
        if i not in fixed_first:
            free.append(i)
    for i in range(x.size - n1):
        if i not in fixed_second:
            free.append(n1 + i)
    # only the given values move to the right-hand side
    for i in free:
        x.unsafe_set(i, 0.0)
    var r = b.stacked() - M @ x
    var rf = real_vector(len(free))
    for i in range(len(free)):
        rf.unsafe_set(i, r.unsafe_get(free[i]))
    var u = spsolve(M.select(free, free), rf)
    for i in range(len(free)):
        x.unsafe_set(free[i], u.unsafe_get(i))
    return _split[n, in1_k, in1_p, in2_k, in2_p](x, n1, complex)


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
