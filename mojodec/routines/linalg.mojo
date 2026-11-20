# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Sparse linear solvers (`mojodec.routines.linalg`).
==================================================
The parts of `scipy.sparse.linalg` and `scipy.linalg` that PyDEC's examples
use, over `CSRMatrix` and `RealArray`.

- `cg`: (preconditioned) conjugate gradients for symmetric positive
  (semi)definite systems.
- `minres`: Paige and Saunders' MINRES for symmetric indefinite systems,
  such as the saddle-point `BlockOp`s of mixed formulations; not in PyDEC's
  examples.
- `NullSpace`: the kernel of a singular operator, for `cg` and `minres`
  to keep out of their iterates.
- `eigenvalue_range`: the extreme eigenvalues of a preconditioned
  operator, estimated from the Ritz values of a few CG steps.
- `jacobi`: the diagonal (Jacobi) preconditioner of a matrix.
- `lsqr`: Paige and Saunders' LSQR for least-squares problems.
- `spsolve`: a direct solve (dense pivoted LU; intended for moderate sizes).
- `eigh_generalized`: all eigenpairs of `K x = lambda M x` for symmetric `K`
  and symmetric positive definite `M` (dense, or a `DiagonalOperator` such
  as a Hodge star).

`cg` and `minres` take any `LinearOperator` (`molina.operator`) and,
as `M`, any `Preconditioner`, and optionally a starting guess (`x0`), a
`NullSpace`, and `ritz=True` for the Ritz values. Each solver has a typed overload on
`MatrixOp`s and `TypedCochain`s, and `minres` and `spsolve` one on a
`BlockOp` and `CochainPair`s (`spsolve` with some unknowns optionally
given). An operator
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

The untyped solvers are molina's (`molina.linalg`), re-exported here. `cg`, `minres`, `jacobi`, `lsqr`, `spsolve`, and
`eigh_generalized` are defined here as well, forwarding to it, because
Mojo does not merge overloads of one name imported from two modules.
"""


from std.math import isfinite, sqrt
from std.memory import ArcPointer

from molina.diagonal import DiagonalOperator
from molina.operator import Identity, LinearOperator, Preconditioner
from mojodec.core.simplicial_complex import SimplicialComplex
from molina.sparse import CSRMatrix
from mojodec.core.typed import DiagonalOp, MatrixOp, TypedCochain, TypedSubset
from mojodec.core.typed_block import BlockOp, CochainPair
from mojodec.core.utils import (
    RealArray,
    axpy,
    dot,
    norm2,
    probe_vector,
    real_array,
    real_vector,
    rows_of,
    cols_of,
)
from molina import dense
from molina.dense import EighResult
from mojodec.routines.sparse_ldlt import SparseLDLT
from molina import linalg as molina_linalg
from molina.linalg import (
    DenseCholesky,
    DirectMethod,
    EigenvalueRange,
    NullSpace,
    SolveResult,
    StopReason,
    eigenvalue_range,
)

# ===----------------------------------------------------------------------=== #
# The untyped solvers, forwarded to `molina.linalg`
# ===----------------------------------------------------------------------=== #


def cg[
    Op: LinearOperator
](
    A: Op,
    b: RealArray,
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
    *,
    x0: Optional[RealArray] = None,
    nullspace: Optional[NullSpace] = None,
    ritz: Bool = False,
) raises -> SolveResult[RealArray]:
    """Conjugate gradients for `A x = b`: `molina.linalg.cg`, which
    documents it.

    Parameters:
        Op: The operator type.

    Args:
        A: An `n x n` symmetric positive semidefinite operator.
        b: The length-`n` right-hand side.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means `10 * n`.
        x0: The starting guess; `None` means zero.
        nullspace: The kernel of `A`.
        ritz: Whether to return the Ritz values.

    Returns:
        The solution with why the iteration stopped.

    Raises:
        As `molina.linalg.cg`.
    """
    return molina_linalg.cg(
        A,
        b,
        rtol,
        atol,
        maxiter,
        x0=x0,
        nullspace=nullspace,
        ritz=ritz,
    )


def cg[
    Op: LinearOperator, P: Preconditioner
](
    A: Op,
    b: RealArray,
    *,
    M: P,
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
    x0: Optional[RealArray] = None,
    nullspace: Optional[NullSpace] = None,
    ritz: Bool = False,
) raises -> SolveResult[RealArray]:
    """Preconditioned conjugate gradients: `molina.linalg.cg`, which
    documents it.

    Parameters:
        Op: The operator type.
        P: The preconditioner type.

    Args:
        A: An `n x n` symmetric positive semidefinite operator.
        b: The length-`n` right-hand side.
        M: A symmetric positive definite `n x n` preconditioner.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means `10 * n`.
        x0: The starting guess; `None` means zero.
        nullspace: The kernel of `A`.
        ritz: Whether to return the Ritz values.

    Returns:
        The solution with why the iteration stopped.

    Raises:
        As `molina.linalg.cg`.
    """
    return molina_linalg.cg(
        A,
        b,
        M=M,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        x0=x0,
        nullspace=nullspace,
        ritz=ritz,
    )


def minres[
    Op: LinearOperator
](
    A: Op,
    b: RealArray,
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
    *,
    x0: Optional[RealArray] = None,
    nullspace: Optional[NullSpace] = None,
    ritz: Bool = False,
) raises -> SolveResult[RealArray]:
    """MINRES for symmetric `A x = b`: `molina.linalg.minres`, which
    documents it.

    Parameters:
        Op: The operator type.

    Args:
        A: An `n x n` symmetric operator.
        b: The length-`n` right-hand side.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means `10 * n`.
        x0: The starting guess; `None` means zero.
        nullspace: The kernel of `A`.
        ritz: Whether to return the Ritz values.

    Returns:
        The solution with why the iteration stopped.

    Raises:
        As `molina.linalg.minres`.
    """
    return molina_linalg.minres(
        A,
        b,
        rtol,
        atol,
        maxiter,
        x0=x0,
        nullspace=nullspace,
        ritz=ritz,
    )


def minres[
    Op: LinearOperator, P: Preconditioner
](
    A: Op,
    b: RealArray,
    *,
    M: P,
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
    x0: Optional[RealArray] = None,
    nullspace: Optional[NullSpace] = None,
    ritz: Bool = False,
) raises -> SolveResult[RealArray]:
    """Preconditioned MINRES: `molina.linalg.minres`, which documents it.

    Parameters:
        Op: The operator type.
        P: The preconditioner type.

    Args:
        A: An `n x n` symmetric operator.
        b: The length-`n` right-hand side.
        M: A symmetric positive definite `n x n` preconditioner.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means `10 * n`.
        x0: The starting guess; `None` means zero.
        nullspace: The kernel of `A`.
        ritz: Whether to return the Ritz values.

    Returns:
        The solution with why the iteration stopped.

    Raises:
        As `molina.linalg.minres`.
    """
    return molina_linalg.minres(
        A,
        b,
        M=M,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        x0=x0,
        nullspace=nullspace,
        ritz=ritz,
    )


def jacobi(A: CSRMatrix) raises -> DiagonalOperator:
    """The Jacobi preconditioner of `A`: `molina.linalg.jacobi`, which
    documents it.

    Args:
        A: A square matrix with a positive diagonal.

    Returns:
        The inverse of `A`'s diagonal.

    Raises:
        As `molina.linalg.jacobi`.
    """
    return molina_linalg.jacobi(A)


def lsqr(
    A: CSRMatrix,
    b: RealArray,
    atol: Float64 = 1e-8,
    btol: Float64 = 1e-8,
    iter_lim: Optional[Int] = None,
) raises -> SolveResult[RealArray]:
    """LSQR for `min ||A x - b||`: `molina.linalg.lsqr`, which documents it.

    Args:
        A: An `m x n` matrix.
        b: The length-`m` right-hand side.
        atol: Tolerance on `A`'s relative error.
        btol: Tolerance on `b`'s relative error.
        iter_lim: Iteration limit; `None` means `2 * n`.

    Returns:
        The least-squares solution with why the iteration stopped.

    Raises:
        As `molina.linalg.lsqr`.
    """
    return molina_linalg.lsqr(A, b, atol, btol, iter_lim)


def spsolve(
    A: CSRMatrix,
    b: RealArray,
    *,
    method: DirectMethod = DirectMethod.DENSE_LU,
) raises -> RealArray:
    """A direct solve of `A x = b`: `molina.linalg.spsolve`, which documents
    it.

    Args:
        A: A square nonsingular matrix.
        b: The right-hand side.
        method: Dense LU, or `SparseLDLT` for symmetric `A`.

    Returns:
        The solution.

    Raises:
        As `molina.linalg.spsolve`.
    """
    return molina_linalg.spsolve(A, b, method=method)


def eigh_generalized(K: RealArray, M: RealArray) raises -> EighResult:
    """All eigenpairs of `K x = lambda M x`, dense:
    `molina.linalg.eigh_generalized`, which documents it.

    Args:
        K: A symmetric `n x n` array.
        M: A symmetric positive definite `n x n` array.

    Returns:
        The eigenvalues ascending, and `M`-orthonormal eigenvectors.

    Raises:
        As `molina.linalg.eigh_generalized`.
    """
    return molina_linalg.eigh_generalized(K, M)


def eigh_generalized(K: RealArray, M: DiagonalOperator) raises -> EighResult:
    """All eigenpairs of `K x = lambda M x` with a diagonal `M`:
    `molina.linalg.eigh_generalized`, which documents it.

    Args:
        K: A symmetric `n x n` array.
        M: A positive diagonal operator, such as a Hodge star.

    Returns:
        The eigenvalues ascending, and `M`-orthonormal eigenvectors.

    Raises:
        As `molina.linalg.eigh_generalized`.
    """
    return molina_linalg.eigh_generalized(K, M)


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
        var rows = self.vectors.size // m  # n x m: all pairs, or a few
        var v = real_vector(rows)
        for r in range(rows):
            v.unsafe_set(r, self.vectors.unsafe_get(r * m + i))
        return TypedCochain[Self.n, Self.k, Self.primal](self.complex, v^)

    def residuals[
        out_k: Int, out_primal: Bool
    ](
        self,
        K: MatrixOp[Self.n, Self.k, Self.primal, out_k, out_primal],
        M: MatrixOp[Self.n, Self.k, Self.primal, out_k, out_primal],
    ) raises -> RealArray:
        """The backward error of each eigenpair, with a sparse mass matrix;
        see `EighResult.residuals`.

        Parameters:
            out_k: The degree of the cochains `K` and `M` give.
            out_primal: Whether the cochains `K` and `M` give are primal.

        Args:
            K: The operator of the problem.
            M: The mass matrix of the problem.

        Returns:
            One backward error per eigenpair.

        Raises:
            If `K` or `M` is on another complex than the eigenvectors, or
            not square of the right size.
        """
        _typed_complex(K.complex, self.complex, "residuals")
        _typed_complex(M.complex, self.complex, "residuals")
        return dense._residuals(self.values, self.vectors, K, M)

    def residuals[
        out_k: Int, out_primal: Bool
    ](
        self,
        K: MatrixOp[Self.n, Self.k, Self.primal, out_k, out_primal],
        M: DiagonalOp[Self.n, Self.k, Self.primal, out_k, out_primal],
    ) raises -> RealArray:
        """The backward error of each eigenpair, with a diagonal mass
        matrix such as a Hodge star; see `EighResult.residuals`.

        ```mojo
        var eig = eigh_generalized(K, tc.star[1, PRIMAL]())
        var eta = eig.residuals(K, tc.star[1, PRIMAL]())
        ```

        Parameters:
            out_k: The degree of the cochains `K` and `M` give.
            out_primal: Whether the cochains `K` and `M` give are primal.

        Args:
            K: The operator of the problem.
            M: The mass matrix of the problem.

        Returns:
            One backward error per eigenpair.

        Raises:
            If `K` or `M` is on another complex than the eigenvectors, or
            not square of the right size.
        """
        _typed_complex(K.complex, self.complex, "residuals")
        _typed_complex(M.complex, self.complex, "residuals")
        return dense._residuals(self.values, self.vectors, K, M)


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
    var ritz_values = res.ritz_values.copy()
    var typed = SolveResult[TypedCochain[n, k, primal]](
        TypedCochain[n, k, primal](complex, res^.take_x()),
        stop,
        iterations,
        residual_norm,
        normal_residual_norm,
    )
    typed.ritz_values = ritz_values^
    return typed^


def cg[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    A: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    b: TypedCochain[n, out_k, out_primal],
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
    *,
    x0: Optional[TypedCochain[n, in_k, in_primal]] = None,
    nullspace: Optional[NullSpace] = None,
    ritz: Bool = False,
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
        x0: The starting guess, of `A`'s input kind; `None` means zero.
        nullspace: The kernel of `A` (vectors of its input kind).
        ritz: Whether to return the Ritz values.

    Returns:
        The solution, of `A`'s input kind, with why the iteration stopped.

    Raises:
        If `A` and `b` are on different complexes, or as the untyped `cg`.
    """
    _typed_complex(A.complex, b.complex, "cg")
    var guess: Optional[RealArray] = None
    if x0:
        if not x0.value().complex.is_same(b.complex):
            raise Error("cg: x0 is on a different complex")
        guess = x0.value().v.copy()
    return _typed_result[n, in_k, in_primal](
        cg(
            A.mat,
            b.v,
            rtol,
            atol,
            maxiter,
            x0=guess,
            nullspace=nullspace,
            ritz=ritz,
        ),
        b.complex,
    )


def cg[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    A: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    b: TypedCochain[n, out_k, out_primal],
    *,
    M: DiagonalOp[n, out_k, out_primal, in_k, in_primal],
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
    x0: Optional[TypedCochain[n, in_k, in_primal]] = None,
    nullspace: Optional[NullSpace] = None,
    ritz: Bool = False,
) raises -> SolveResult[TypedCochain[n, in_k, in_primal]]:
    """Preconditioned conjugate gradients on cochains; see the untyped
    overload.

    The preconditioner approximates `A^-1`, so it maps `A`'s output kind
    back to its input kind: for a stiffness matrix from primal 0-cochains
    to dual n-cochains, `jacobi(A)` or an inverse Hodge star.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the solution.
        in_primal: Whether the solution is primal.
        out_k: The degree of the right-hand side.
        out_primal: Whether the right-hand side is primal.

    Args:
        A: A symmetric positive semidefinite operator.
        b: The right-hand side, of `A`'s output kind.
        M: A positive diagonal preconditioner, from `A`'s output kind to
            its input kind.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means 10 times the size.
        x0: The starting guess, of `A`'s input kind; `None` means zero.
        nullspace: The kernel of `A` (vectors of its input kind).
        ritz: Whether to return the Ritz values.

    Returns:
        The solution, of `A`'s input kind, with why the iteration stopped.

    Raises:
        If the operands are on different complexes, or as the untyped `cg`.
    """
    _typed_complex(A.complex, b.complex, "cg")
    _typed_complex(M.complex, b.complex, "cg")
    var guess: Optional[RealArray] = None
    if x0:
        if not x0.value().complex.is_same(b.complex):
            raise Error("cg: x0 is on a different complex")
        guess = x0.value().v.copy()
    return _typed_result[n, in_k, in_primal](
        cg(
            A.mat,
            b.v,
            M=M.diag,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            x0=guess,
            nullspace=nullspace,
            ritz=ritz,
        ),
        b.complex,
    )


def minres[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    A: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    b: TypedCochain[n, out_k, out_primal],
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
    *,
    x0: Optional[TypedCochain[n, in_k, in_primal]] = None,
    nullspace: Optional[NullSpace] = None,
    ritz: Bool = False,
) raises -> SolveResult[TypedCochain[n, in_k, in_primal]]:
    """MINRES for `A x = b` on cochains; see the untyped overload.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the solution.
        in_primal: Whether the solution is primal.
        out_k: The degree of the right-hand side.
        out_primal: Whether the right-hand side is primal.

    Args:
        A: A symmetric operator, possibly indefinite or singular.
        b: The right-hand side, of `A`'s output kind.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means 10 times the size.
        x0: The starting guess, of `A`'s input kind; `None` means zero.
        nullspace: The kernel of `A` (vectors of its input kind).
        ritz: Whether to return the Ritz values.

    Returns:
        The solution, of `A`'s input kind, with why the iteration stopped.

    Raises:
        If `A` and `b` are on different complexes, or as the untyped
        `minres`.
    """
    _typed_complex(A.complex, b.complex, "minres")
    var guess: Optional[RealArray] = None
    if x0:
        if not x0.value().complex.is_same(b.complex):
            raise Error("minres: x0 is on a different complex")
        guess = x0.value().v.copy()
    return _typed_result[n, in_k, in_primal](
        minres(
            A.mat,
            b.v,
            rtol,
            atol,
            maxiter,
            x0=guess,
            nullspace=nullspace,
            ritz=ritz,
        ),
        b.complex,
    )


def minres[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    A: MatrixOp[n, in_k, in_primal, out_k, out_primal],
    b: TypedCochain[n, out_k, out_primal],
    *,
    M: DiagonalOp[n, out_k, out_primal, in_k, in_primal],
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
    x0: Optional[TypedCochain[n, in_k, in_primal]] = None,
    nullspace: Optional[NullSpace] = None,
    ritz: Bool = False,
) raises -> SolveResult[TypedCochain[n, in_k, in_primal]]:
    """Preconditioned MINRES on cochains; see the untyped overload.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the solution.
        in_primal: Whether the solution is primal.
        out_k: The degree of the right-hand side.
        out_primal: Whether the right-hand side is primal.

    Args:
        A: A symmetric operator.
        b: The right-hand side, of `A`'s output kind.
        M: A positive diagonal preconditioner, from `A`'s output kind to
            its input kind.
        rtol: Relative tolerance on the residual in the `M` norm.
        atol: Absolute tolerance on the residual in the `M` norm.
        maxiter: Iteration limit; `None` means 10 times the size.
        x0: The starting guess, of `A`'s input kind; `None` means zero.
        nullspace: The kernel of `A` (vectors of its input kind).
        ritz: Whether to return the Ritz values.

    Returns:
        The solution, of `A`'s input kind, with why the iteration stopped.

    Raises:
        If the operands are on different complexes, or as the untyped
        `minres`.
    """
    _typed_complex(A.complex, b.complex, "minres")
    _typed_complex(M.complex, b.complex, "minres")
    var guess: Optional[RealArray] = None
    if x0:
        if not x0.value().complex.is_same(b.complex):
            raise Error("minres: x0 is on a different complex")
        guess = x0.value().v.copy()
    return _typed_result[n, in_k, in_primal](
        minres(
            A.mat,
            b.v,
            M=M.diag,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            x0=guess,
            nullspace=nullspace,
            ritz=ritz,
        ),
        b.complex,
    )


def minres[
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
    rtol: Float64 = 1e-5,
    atol: Float64 = 0.0,
    maxiter: Optional[Int] = None,
) raises -> SolveResult[CochainPair[n, in1_k, in1_p, in2_k, in2_p]]:
    """MINRES for a block system, such as the saddle-point system of a
    mixed formulation; see the untyped overload.

    The blocks are applied one by one, never assembled. `A` must be
    symmetric as a whole: `a11` and `a22` symmetric, `a21` the transpose
    of `a12`, as in `[[-star1, d1^T], [d1, 0]]`.

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
        A: A symmetric block operator.
        b: The right-hand side, of `A`'s output kinds.
        rtol: Relative tolerance on the residual norm.
        atol: Absolute tolerance on the residual norm.
        maxiter: Iteration limit; `None` means 10 times the size.

    Returns:
        The unknowns, of `A`'s input kinds, with why the iteration stopped.

    Raises:
        If `A` and `b` are on different complexes, or as the untyped
        `minres`.
    """
    var complex = b.first.complex
    _typed_complex(A.complex, complex, "minres")
    var res = minres(A, b.stacked(), rtol, atol, maxiter)
    var stop = res.stop
    var iterations = res.iterations
    var residual_norm = res.residual_norm
    var x = _split[n, in1_k, in1_p, in2_k, in2_p](
        res^.take_x(), A.a11.mat.ncols, complex
    )
    return SolveResult[CochainPair[n, in1_k, in1_p, in2_k, in2_p]](
        x^, stop, iterations, residual_norm
    )


def jacobi[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](A: MatrixOp[n, in_k, in_primal, out_k, out_primal]) raises -> DiagonalOp[
    n, out_k, out_primal, in_k, in_primal
]:
    """The Jacobi preconditioner of a typed operator; see the untyped
    overload.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the cochains `A` takes.
        in_primal: Whether the cochains `A` takes are primal.
        out_k: The degree of the cochains `A` gives.
        out_primal: Whether the cochains `A` gives are primal.

    Args:
        A: An operator with as many output as input cells and a nonzero
            diagonal.

    Returns:
        `diag(1 / |a_ii|)`, from `A`'s output kind to its input kind.

    Raises:
        As the untyped `jacobi`.
    """
    return DiagonalOp[n, out_k, out_primal, in_k, in_primal](
        jacobi(A.mat), A.complex
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
    *,
    method: DirectMethod = DirectMethod.DENSE_LU,
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
        method: `DirectMethod.DENSE_LU` or `DirectMethod.SPARSE_LDLT`.

    Returns:
        The solution, of `A`'s input kind.

    Raises:
        If `A` and `b` are on different complexes, `A` is not square, or
        `A` is singular.
    """
    _typed_complex(A.complex, b.complex, "spsolve")
    return TypedCochain[n, in_k, in_primal](
        b.complex, spsolve(A.mat, b.v, method=method)
    )


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
    *,
    method: DirectMethod = DirectMethod.DENSE_LU,
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
        method: `DirectMethod.DENSE_LU` or `DirectMethod.SPARSE_LDLT`.

    Returns:
        The unknowns, of `A`'s input kinds.

    Raises:
        If `A` and `b` are on different complexes, `A` is not square, or
        `A` is singular.
    """
    var complex = b.first.complex
    _typed_complex(A.complex, complex, "spsolve")
    var x = spsolve(A.to_csr(), b.stacked(), method=method)
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
    method: DirectMethod = DirectMethod.DENSE_LU,
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
        method: How to factor the reduced system: `DirectMethod.DENSE_LU`
            or, for a symmetric `A` (a mixed formulation), the sparse
            `DirectMethod.SPARSE_LDLT`; the reduced matrix keeps the same
            rows as columns, so it stays symmetric.

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
    var u = spsolve(M.select(free, free), rf, method=method)
    for i in range(len(free)):
        x.unsafe_set(free[i], u.unsafe_get(i))
    return _split[n, in1_k, in1_p, in2_k, in2_p](x, n1, complex)
