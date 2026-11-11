# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Linear operators (`mojodec.core.operator`).
===========================================
What an iterative solver needs of a matrix: its shape and its product with
a vector. `cg` and `minres` in `mojodec.routines.linalg` take any
`LinearOperator` and, as a preconditioner, any `Preconditioner`.

- `LinearOperator`: `num_rows`, `num_cols`, and `apply_into(x, y)`,
  `y = A x`. `CSRMatrix`, `DiagonalOperator`, `Identity`, and the typed
  `MatrixOp`, `DiagonalOp`, and `BlockOp` implement it; a `BlockOp`
  applies its blocks without assembling them.
- `Preconditioner`: a `LinearOperator` that promises to be symmetric
  positive definite, which `cg` and `minres` need of it. It has no methods
  of its own; the promise is what a type declares by conforming.
  `DiagonalOperator` (such as `jacobi(A)` or an inverse Hodge star),
  `DiagonalOp`, and `Identity` do. A general `CSRMatrix` does not, so a
  one-sided sweep or a non-symmetric matrix cannot be passed by mistake.

```mojo
var res = cg(A, b, M=jacobi(A), rtol=1e-10)    # preconditioned CG
var y = real_vector(A.nrows)
A.apply_into(x, y)                             # any LinearOperator
```
"""

from mojodec.core.utils import RealArray


trait LinearOperator:
    """A linear map `R^num_cols -> R^num_rows` applied to vectors."""

    def num_rows(self) -> Int:
        """The length of the output.

        Returns:
            The number of rows.
        """
        ...

    def num_cols(self) -> Int:
        """The length of the input.

        Returns:
            The number of columns.
        """
        ...

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """The product `A x`, written into `y` without allocating.

        Args:
            x: A vector of length `num_cols()`.
            y: A vector of length `num_rows()`, overwritten.

        Raises:
            If a length does not match.
        """
        ...


trait Preconditioner(LinearOperator):
    """A square `LinearOperator` that is symmetric positive definite.

    Conforming is a promise the solvers rely on: CG and MINRES build
    their iterates in the inner product the preconditioner defines. A
    solver still stops with `StopReason.BREAKDOWN` if `r^T M r` comes out
    negative, which an indefinite preconditioner shows sooner or later.
    """

    pass


struct Identity(Copyable, Movable, Preconditioner):
    """The `n x n` identity, the preconditioner that does nothing."""

    var n: Int
    """The dimension."""

    def __init__(out self, n: Int):
        """The identity on `R^n`.

        Args:
            n: The dimension.
        """
        self.n = n

    def num_rows(self) -> Int:
        """The dimension.

        Returns:
            `n`.
        """
        return self.n

    def num_cols(self) -> Int:
        """The dimension.

        Returns:
            `n`.
        """
        return self.n

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """Copy `x` into `y`.

        Args:
            x: A vector of length `n`.
            y: A vector of length `n`, overwritten with `x`.

        Raises:
            If a length is not `n`.
        """
        if x.size != self.n or y.size != self.n:
            raise Error("Identity.apply_into: dimension mismatch")
        for i in range(self.n):
            y.unsafe_set(i, x.unsafe_get(i))
