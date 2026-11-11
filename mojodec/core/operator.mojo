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
- `BlockDiagonal(P1, P2)`: two preconditioners side by side, for a 2 x 2
  block system such as a mixed formulation.

```mojo
var res = cg(A, b, M=jacobi(A), rtol=1e-10)    # preconditioned CG
var y = real_vector(A.nrows)
A.apply_into(x, y)                             # any LinearOperator
```
"""

from mojodec.core.utils import RealArray, real_vector


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


struct BlockDiagonal[
    P1: Preconditioner & Movable & Deinitable,
    P2: Preconditioner & Movable & Deinitable,
](Movable, Preconditioner):
    """`diag(P1, P2)`: two preconditioners applied to the two halves of a
    vector, for a 2 x 2 block system.

    For a symmetric saddle-point system such as the mixed Hodge Laplacian
    `[[-M, B^T], [B, A]]`, MINRES needs a positive definite preconditioner
    even though the system is indefinite; a block-diagonal one whose
    blocks are spectrally equivalent to the inverses of the blocks'
    natural norms (Arnold, Falk, and Winther) gives iteration counts that
    do not grow as the mesh is refined.

    ```mojo
    var P = BlockDiagonal(DenseCholesky(S1), DenseCholesky(S2))
    var res = minres(K, b, M=P, rtol=1e-10)
    ```

    Parameters:
        P1: The type of the first block.
        P2: The type of the second block.
    """

    var first: Self.P1
    """The preconditioner of the first `first.num_rows()` entries."""
    var second: Self.P2
    """The preconditioner of the rest."""

    def __init__(out self, var first: Self.P1, var second: Self.P2) raises:
        """Combine two preconditioners.

        Args:
            first: The first block, square.
            second: The second block, square.

        Raises:
            If a block is not square.
        """
        if (
            first.num_rows() != first.num_cols()
            or second.num_rows() != second.num_cols()
        ):
            raise Error("BlockDiagonal: the blocks must be square")
        self.first = first^
        self.second = second^

    def num_rows(self) -> Int:
        """The total size.

        Returns:
            The sum of the blocks' sizes.
        """
        return self.first.num_rows() + self.second.num_rows()

    def num_cols(self) -> Int:
        """The total size.

        Returns:
            The sum of the blocks' sizes.
        """
        return self.num_rows()

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """Apply each block to its part of `x`.

        Args:
            x: A vector of the total size.
            y: A vector of the total size, overwritten.

        Raises:
            If a length does not match, or as a block's `apply_into`.
        """
        var n1 = self.first.num_rows()
        var n = self.num_rows()
        if x.size != n or y.size != n:
            raise Error("BlockDiagonal.apply_into: dimension mismatch")
        var x1 = real_vector(n1)
        var x2 = real_vector(n - n1)
        for i in range(n1):
            x1.unsafe_set(i, x.unsafe_get(i))
        for i in range(n - n1):
            x2.unsafe_set(i, x.unsafe_get(n1 + i))
        var y1 = real_vector(n1)
        var y2 = real_vector(n - n1)
        self.first.apply_into(x1, y1)
        self.second.apply_into(x2, y2)
        for i in range(n1):
            y.unsafe_set(i, y1.unsafe_get(i))
        for i in range(n - n1):
            y.unsafe_set(n1 + i, y2.unsafe_get(i))
