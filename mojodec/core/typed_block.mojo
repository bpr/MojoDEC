# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Typed block operators (`mojodec.core.typed_block`).
===================================================
Mixed formulations solve for two cochains at once, such as the flux and the
pressure of Darcy flow. `CochainPair` holds the two, and `BlockOp` is a
2 x 2 block of typed operators from one pair of cochain kinds to another.

```
[ a11  a12 ] [ x1 ]   [ y1 ]      a11: in1 -> out1    a12: in2 -> out1
[ a21  a22 ] [ x2 ] = [ y2 ]      a21: in1 -> out2    a22: in2 -> out2
```

Each block's kinds are in its type, so the blocks must agree on the kinds
of the unknowns (columns) and of the equations (rows):

```mojo
var d1 = tc.d[1, PRIMAL]()
var A = BlockOp(
    tc.star[1, PRIMAL]().to_matrix_op() * -1.0,   # primal 1 -> dual 1
    d1.T(),                                       # dual 0 -> dual 1
    d1,                                           # primal 1 -> primal 2
    tc.zero[0, DUAL, 2, PRIMAL](),                # dual 0 -> primal 2
)
var y = A @ CochainPair(flux, pressure)
```

`spsolve` in `mojodec.routines.linalg` solves `A x = b` for a `BlockOp`,
optionally with some unknowns of each kind given.
"""

from mojodec.core.operator import LinearOperator
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.typed import MatrixOp, TypedCochain
from mojodec.core.utils import RealArray, real_vector


def _add_product(
    A: CSRMatrix, x: RealArray, x_off: Int, mut y: RealArray, y_off: Int
):
    """`y[y_off:] += A x[x_off:]`, the sizes checked by the caller."""
    for i in range(A.nrows):
        var s: Float64 = 0
        for k in range(A.indptr[i], A.indptr[i + 1]):
            s += A.data[k] * x.unsafe_get(x_off + A.indices[k])
        y.unsafe_set(y_off + i, y.unsafe_get(y_off + i) + s)


def _same_complex(
    a: Optional[SimplicialComplex], b: Optional[SimplicialComplex]
) raises -> Optional[SimplicialComplex]:
    """The complex of a block built from operators on `a` and `b`."""
    if not a:
        return b
    if b and not a.value().is_same(b.value()):
        raise Error("BlockOp: blocks on different complexes")
    return a


struct CochainPair[n: Int, k1: Int, p1: Bool, k2: Int, p2: Bool](
    Copyable, Movable, Writable
):
    """Two cochains on one complex, the unknowns or equations of a mixed
    formulation.

    Parameters:
        n: The dimension of the complex.
        k1: The degree of the first cochain.
        p1: Whether the first cochain is primal.
        k2: The degree of the second cochain.
        p2: Whether the second cochain is primal.
    """

    var first: TypedCochain[Self.n, Self.k1, Self.p1]
    """The first cochain."""
    var second: TypedCochain[Self.n, Self.k2, Self.p2]
    """The second cochain."""

    def __init__(
        out self,
        var first: TypedCochain[Self.n, Self.k1, Self.p1],
        var second: TypedCochain[Self.n, Self.k2, Self.p2],
    ) raises:
        """Pair two cochains.

        Args:
            first: The first cochain.
            second: The second cochain, on the same complex.

        Raises:
            If they live on different complexes.
        """
        if not first.complex.is_same(second.complex):
            raise Error("CochainPair: cochains on different complexes")
        self.first = first^
        self.second = second^

    def stacked(self) raises -> RealArray:
        """The values of both cochains in one vector, first then second.

        Returns:
            A vector of `len(first) + len(second)` values.

        Raises:
            Only if allocating the vector fails.
        """
        var n1 = len(self.first)
        var n2 = len(self.second)
        var v = real_vector(n1 + n2)
        for i in range(n1):
            v.unsafe_set(i, self.first[i])
        for i in range(n2):
            v.unsafe_set(n1 + i, self.second[i])
        return v^

    def __add__(self, other: Self) raises -> Self:
        """The sum, cochain by cochain.

        Args:
            other: A pair of the same kinds on the same complex.

        Returns:
            `self + other`.

        Raises:
            If `other` lives on another complex.
        """
        return Self(self.first + other.first, self.second + other.second)

    def __sub__(self, other: Self) raises -> Self:
        """The difference, cochain by cochain.

        Args:
            other: A pair of the same kinds on the same complex.

        Returns:
            `self - other`.

        Raises:
            If `other` lives on another complex.
        """
        return Self(self.first - other.first, self.second - other.second)

    def write_to(self, mut writer: Some[Writer]):
        """Write both cochains.

        Args:
            writer: The destination.
        """
        writer.write("CochainPair(", self.first, ", ", self.second, ")")


struct BlockOp[
    n: Int,
    in1_k: Int,
    in1_p: Bool,
    in2_k: Int,
    in2_p: Bool,
    out1_k: Int,
    out1_p: Bool,
    out2_k: Int,
    out2_p: Bool,
](Copyable, LinearOperator, Movable, Writable):
    """A 2 x 2 block of typed operators, from a pair of cochains of kinds
    `in1`, `in2` to a pair of kinds `out1`, `out2`.

    As a `LinearOperator` it applies to the stacked values (first then
    second) block by block, without assembling `to_csr()`.

    Parameters:
        n: The dimension of the complex.
        in1_k: The degree of the first unknown.
        in1_p: Whether the first unknown is primal.
        in2_k: The degree of the second unknown.
        in2_p: Whether the second unknown is primal.
        out1_k: The degree of the first equation.
        out1_p: Whether the first equation is primal.
        out2_k: The degree of the second equation.
        out2_p: Whether the second equation is primal.
    """

    var a11: MatrixOp[Self.n, Self.in1_k, Self.in1_p, Self.out1_k, Self.out1_p]
    """The block from the first unknown to the first equation."""
    var a12: MatrixOp[Self.n, Self.in2_k, Self.in2_p, Self.out1_k, Self.out1_p]
    """The block from the second unknown to the first equation."""
    var a21: MatrixOp[Self.n, Self.in1_k, Self.in1_p, Self.out2_k, Self.out2_p]
    """The block from the first unknown to the second equation."""
    var a22: MatrixOp[Self.n, Self.in2_k, Self.in2_p, Self.out2_k, Self.out2_p]
    """The block from the second unknown to the second equation."""
    var complex: Optional[SimplicialComplex]
    """The complex of the blocks, or `None` if all were wrapped from bare
    matrices."""

    def __init__(
        out self,
        a11: MatrixOp[Self.n, Self.in1_k, Self.in1_p, Self.out1_k, Self.out1_p],
        a12: MatrixOp[Self.n, Self.in2_k, Self.in2_p, Self.out1_k, Self.out1_p],
        a21: MatrixOp[Self.n, Self.in1_k, Self.in1_p, Self.out2_k, Self.out2_p],
        a22: MatrixOp[Self.n, Self.in2_k, Self.in2_p, Self.out2_k, Self.out2_p],
    ) raises:
        """The block operator `[[a11, a12], [a21, a22]]`. A zero block is
        `tc.zero[...]()`.

        Args:
            a11: First unknown to first equation.
            a12: Second unknown to first equation.
            a21: First unknown to second equation.
            a22: Second unknown to second equation.

        Raises:
            If the blocks are on different complexes, or their shapes do not
            line up (blocks in a row with as many rows, in a column with as
            many columns).
        """
        if (
            a11.mat.nrows != a12.mat.nrows
            or a21.mat.nrows != a22.mat.nrows
            or a11.mat.ncols != a21.mat.ncols
            or a12.mat.ncols != a22.mat.ncols
        ):
            raise Error("BlockOp: block shapes do not line up")
        self.complex = _same_complex(
            _same_complex(a11.complex, a12.complex),
            _same_complex(a21.complex, a22.complex),
        )
        self.a11 = a11.copy()
        self.a12 = a12.copy()
        self.a21 = a21.copy()
        self.a22 = a22.copy()

    def num_unknowns(self) -> Int:
        """The number of values of both unknowns together.

        Returns:
            The number of columns.
        """
        return self.a11.mat.ncols + self.a12.mat.ncols

    def num_equations(self) -> Int:
        """The number of values of both equations together.

        Returns:
            The number of rows.
        """
        return self.a11.mat.nrows + self.a21.mat.nrows

    def num_rows(self) -> Int:
        """The number of equations, for `LinearOperator`.

        Returns:
            `num_equations()`.
        """
        return self.num_equations()

    def num_cols(self) -> Int:
        """The number of unknowns, for `LinearOperator`.

        Returns:
            `num_unknowns()`.
        """
        return self.num_unknowns()

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """The operator applied to stacked unknowns, block by block.

        Args:
            x: The unknowns stacked, first then second
                (`CochainPair.stacked()`).
            y: The equations stacked, overwritten.

        Raises:
            If a length does not match.
        """
        if x.size != self.num_unknowns() or y.size != self.num_equations():
            raise Error("BlockOp.apply_into: dimension mismatch")
        var c1 = self.a11.mat.ncols
        var r1 = self.a11.mat.nrows
        for i in range(y.size):
            y.unsafe_set(i, 0.0)
        _add_product(self.a11.mat, x, 0, y, 0)
        _add_product(self.a12.mat, x, c1, y, 0)
        _add_product(self.a21.mat, x, 0, y, r1)
        _add_product(self.a22.mat, x, c1, y, r1)

    def to_csr(self) raises -> CSRMatrix:
        """The whole operator as one sparse matrix, unknowns and equations
        each stacked first then second.

        Returns:
            The assembled `CSRMatrix`.

        Raises:
            Only if the assembly fails, which the shape check rules out.
        """
        var blocks = List[List[Optional[CSRMatrix]]]()
        blocks.append(
            [Optional(self.a11.mat.copy()), Optional(self.a12.mat.copy())]
        )
        blocks.append(
            [Optional(self.a21.mat.copy()), Optional(self.a22.mat.copy())]
        )
        return CSRMatrix.bmat(blocks)

    def __matmul__(
        self,
        x: CochainPair[Self.n, Self.in1_k, Self.in1_p, Self.in2_k, Self.in2_p],
    ) raises -> CochainPair[
        Self.n, Self.out1_k, Self.out1_p, Self.out2_k, Self.out2_p
    ]:
        """Apply the operator to a pair of unknowns.

        Args:
            x: The unknowns.

        Returns:
            `(a11 x1 + a12 x2, a21 x1 + a22 x2)`.

        Raises:
            If `x` is on another complex than the operator, or has the wrong
            sizes.
        """
        return CochainPair(
            self.a11 @ x.first + self.a12 @ x.second,
            self.a21 @ x.first + self.a22 @ x.second,
        )

    def write_to(self, mut writer: Some[Writer]):
        """Write the kinds and the size, e.g.
        `BlockOp[2]: (primal 1, dual 0) -> (dual 1, primal 2), 862 x 862`.

        Args:
            writer: The destination.
        """
        writer.write(
            "BlockOp[",
            Self.n,
            "]: (",
            "primal " if Self.in1_p else "dual ",
            Self.in1_k,
            ", ",
            "primal " if Self.in2_p else "dual ",
            Self.in2_k,
            ") -> (",
            "primal " if Self.out1_p else "dual ",
            Self.out1_k,
            ", ",
            "primal " if Self.out2_p else "dual ",
            Self.out2_k,
            "), ",
            self.num_equations(),
            " x ",
            self.num_unknowns(),
        )
