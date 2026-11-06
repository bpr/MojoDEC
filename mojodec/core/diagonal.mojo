# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Diagonal operators (`mojodec.core.diagonal`).
=============================================
A square diagonal matrix stored as its diagonal. DEC's Hodge stars
(`sc[k].star`, `sc[k].star_inv`) are diagonal, and keeping them so makes the
common products cheap: `D @ x` is elementwise, `D @ A` scales the rows of a
`CSRMatrix` and `A @ D` its columns, and `D @ E` stays diagonal, all without
a general sparse product. `to_csr()` gives the `CSRMatrix` when one is
needed, e.g. as a block of `CSRMatrix.bmat`.

```mojo
var L = sc[0].star_inv @ sc[0].d.T() @ sc[1].star @ sc[0].d   # a CSRMatrix
var energy = sc[1].star.inner_product(omega.v, omega.v)       # omega^T S omega
```
"""

from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset
from mojodec.core.utils import RealArray, real_array, real_vector


struct DiagonalOperator(Copyable, Movable, Sized, Writable):
    """A square diagonal matrix, stored as its diagonal."""

    var values: List[Float64]
    """The diagonal entries."""

    # ===------------------------------------------------------------------=== #
    # Construction
    # ===------------------------------------------------------------------=== #

    def __init__(out self, var values: List[Float64]):
        """The diagonal matrix with the given diagonal.

        Args:
            values: The diagonal entries.
        """
        self.values = values^

    def __init__(out self, values: RealArray):
        """The diagonal matrix with the diagonal held in a 1-D array.

        Args:
            values: The diagonal entries.
        """
        self.values = List[Float64](capacity=values.size)
        for i in range(values.size):
            self.values.append(values.unsafe_get(i))

    @staticmethod
    def identity(n: Int) -> Self:
        """The `n x n` identity.

        Args:
            n: The dimension.

        Returns:
            The identity operator.
        """
        return Self(List[Float64](length=n, fill=1.0))

    # ===------------------------------------------------------------------=== #
    # Access
    # ===------------------------------------------------------------------=== #

    def __len__(self) -> Int:
        """The dimension n of the `n x n` matrix.

        Returns:
            The number of diagonal entries.
        """
        return len(self.values)

    def shape(self) -> Tuple[Int, Int]:
        """The shape.

        Returns:
            `(n, n)`.
        """
        return (len(self.values), len(self.values))

    def __getitem__(self, i: Int) raises -> Float64:
        """The diagonal entry `i`.

        Args:
            i: Index into the diagonal, `0 <= i < len(self)`.

        Returns:
            Entry `(i, i)`.

        Raises:
            If `i` is out of range.
        """
        self._check_index(i)
        return self.values[i]

    def __getitem__(self, i: Int, j: Int) raises -> Float64:
        """The matrix entry `(i, j)`: zero off the diagonal.

        Args:
            i: Row index.
            j: Column index.

        Returns:
            The entry.

        Raises:
            If `i` or `j` is out of range.
        """
        self._check_index(i)
        self._check_index(j)
        return self.values[i] if i == j else 0.0

    def __setitem__(mut self, i: Int, value: Float64) raises:
        """Set the diagonal entry `i`.

        Args:
            i: Index into the diagonal, `0 <= i < len(self)`.
            value: The new entry.

        Raises:
            If `i` is out of range.
        """
        self._check_index(i)
        self.values[i] = value

    def diagonal(self) raises -> RealArray:
        """The diagonal as a 1-D array.

        Returns:
            A new array of length n.

        Raises:
            If NuMojo cannot allocate the array.
        """
        var out = real_vector(len(self.values))
        for i in range(len(self.values)):
            out.unsafe_set(i, self.values[i])
        return out^

    def to_csr(self) raises -> CSRMatrix:
        """The same matrix as a `CSRMatrix`.

        Returns:
            The `n x n` sparse diagonal matrix, zeros stored explicitly.

        Raises:
            Does not raise in practice.
        """
        return CSRMatrix.diag(self.values)

    def todense(self) raises -> RealArray:
        """The dense `n x n` array.

        Returns:
            A new 2-D array.

        Raises:
            If NuMojo cannot allocate the array.
        """
        var n = len(self.values)
        var out = real_array(n, n)
        for i in range(n):
            out.unsafe_set(i * n + i, self.values[i])
        return out^

    def write_to(self, mut writer: Some[Writer]):
        """Write a header line, then the diagonal entries, one
        `(i, i)  value` per line.

        Args:
            writer: The destination.
        """
        writer.write(
            "<",
            len(self.values),
            "x",
            len(self.values),
            " DiagonalOperator>",
        )
        for i in range(len(self.values)):
            writer.write("\n  (", i, ", ", i, ")\t", self.values[i])

    # ===------------------------------------------------------------------=== #
    # Products
    # ===------------------------------------------------------------------=== #

    def __matmul__(self, other: Self) raises -> Self:
        """The product of two diagonal matrices, itself diagonal.

        Args:
            other: The right factor.

        Returns:
            The entrywise product of the diagonals.

        Raises:
            If the dimensions differ.
        """
        self._check(len(other), "DiagonalOperator @ DiagonalOperator")
        var out = List[Float64](capacity=len(self.values))
        for i in range(len(self.values)):
            out.append(self.values[i] * other.values[i])
        return Self(out^)

    def __matmul__(self, A: CSRMatrix) raises -> CSRMatrix:
        """`self @ A`: row `i` of `A` scaled by entry `i`.

        Args:
            A: A matrix with n rows.

        Returns:
            The row-scaled copy of `A`, with the same sparsity pattern.

        Raises:
            If `A.nrows` differs from n.
        """
        self._check(A.nrows, "DiagonalOperator @ CSRMatrix")
        var out = A.copy()
        for i in range(A.nrows):
            var s = self.values[i]
            for k in range(A.indptr[i], A.indptr[i + 1]):
                out.data[k] *= s
        return out^

    def __rmatmul__(self, A: CSRMatrix) raises -> CSRMatrix:
        """`A @ self`: column `j` of `A` scaled by entry `j`.

        Args:
            A: A matrix with n columns.

        Returns:
            The column-scaled copy of `A`, with the same sparsity pattern.

        Raises:
            If `A.ncols` differs from n.
        """
        self._check(A.ncols, "CSRMatrix @ DiagonalOperator")
        var out = A.copy()
        for k in range(len(A.data)):
            out.data[k] *= self.values[A.indices[k]]
        return out^

    def __matmul__(self, x: RealArray) raises -> RealArray:
        """The product with a 1-D array: elementwise.

        Args:
            x: A vector of length n.

        Returns:
            The product, of length n.

        Raises:
            If `x.size` differs from n.
        """
        self._check(x.size, "DiagonalOperator @ vector")
        var y = real_vector(x.size)
        for i in range(x.size):
            y.unsafe_set(i, self.values[i] * x.unsafe_get(i))
        return y^

    def inner_product(self, a: RealArray, b: RealArray) raises -> Float64:
        """`a^T D b`, the inner product this operator defines when its
        entries are positive (not checked).

        Args:
            a: A vector of length n.
            b: A vector of length n.

        Returns:
            The weighted sum of `a[i] * b[i]`.

        Raises:
            If `a` or `b` does not have length n.
        """
        self._check(a.size, "DiagonalOperator.inner_product")
        self._check(b.size, "DiagonalOperator.inner_product")
        var s: Float64 = 0
        for i in range(len(self.values)):
            s += a.unsafe_get(i) * self.values[i] * b.unsafe_get(i)
        return s

    def norm(self, x: RealArray) raises -> Float64:
        """`sqrt(x^T D x)`, the norm this operator defines when its entries
        are positive (not checked).

        Args:
            x: A vector of length n.

        Returns:
            The norm (NaN if `x^T D x` is negative).

        Raises:
            If `x` does not have length n.
        """
        return self.inner_product(x, x) ** 0.5

    def inverse(self) raises -> Self:
        """The inverse: the reciprocals of the entries.

        Note `sc[k].star_inv` is not `sc[k].star.inverse()` in general: it
        carries the sign `(-1)^(k(n-k))`.

        Returns:
            The inverse operator.

        Raises:
            If an entry is zero.
        """
        var out = List[Float64](capacity=len(self.values))
        for v in self.values:
            if v == 0:
                raise Error("DiagonalOperator.inverse: zero diagonal entry")
            out.append(1.0 / v)
        return Self(out^)

    def exclude_subset(self, subset: Subset) raises -> Self:
        """The operator with the entries of `subset` set to zero.

        Args:
            subset: The cells indexing the diagonal.

        Returns:
            A copy with those entries zeroed.

        Raises:
            If `subset.size` differs from n.
        """
        return self._keep(subset, False)

    def restrict_to_subset(self, subset: Subset) raises -> Self:
        """The operator with the entries outside `subset` set to zero.

        Args:
            subset: The cells indexing the diagonal.

        Returns:
            A copy with the other entries zeroed.

        Raises:
            If `subset.size` differs from n.
        """
        return self._keep(subset, True)

    def _keep(self, subset: Subset, inside: Bool) raises -> Self:
        self._check(subset.size, "DiagonalOperator subset")
        var out = self.values.copy()
        for i in range(len(out)):
            if (i in subset) != inside:
                out[i] = 0
        return Self(out^)

    # ===------------------------------------------------------------------=== #
    # Arithmetic
    # ===------------------------------------------------------------------=== #

    def __mul__(self, s: Float64) -> Self:
        """Every entry multiplied by `s`.

        Args:
            s: The scale factor.

        Returns:
            The scaled operator.
        """
        var out = self.values.copy()
        for i in range(len(out)):
            out[i] *= s
        return Self(out^)

    def __rmul__(self, s: Float64) -> Self:
        """Every entry multiplied by `s`.

        Args:
            s: The scale factor.

        Returns:
            The scaled operator.
        """
        return self * s

    def __truediv__(self, s: Float64) -> Self:
        """Every entry divided by `s`.

        Args:
            s: The divisor.

        Returns:
            The scaled operator.
        """
        return self * (1.0 / s)

    def __neg__(self) -> Self:
        """The negation.

        Returns:
            The operator times -1.
        """
        return self * -1.0

    def __add__(self, other: Self) raises -> Self:
        """The entrywise sum.

        Args:
            other: The operator to add.

        Returns:
            The sum, itself diagonal.

        Raises:
            If the dimensions differ.
        """
        self._check(len(other), "DiagonalOperator +")
        var out = self.values.copy()
        for i in range(len(out)):
            out[i] += other.values[i]
        return Self(out^)

    def __sub__(self, other: Self) raises -> Self:
        """The entrywise difference.

        Args:
            other: The operator to subtract.

        Returns:
            The difference, itself diagonal.

        Raises:
            If the dimensions differ.
        """
        self._check(len(other), "DiagonalOperator -")
        var out = self.values.copy()
        for i in range(len(out)):
            out[i] -= other.values[i]
        return Self(out^)

    def _check_index(self, i: Int) raises:
        if i < 0 or i >= len(self.values):
            raise Error(
                String(
                    "DiagonalOperator: index ",
                    i,
                    " out of range for size ",
                    len(self.values),
                )
            )

    def _check(self, n: Int, what: String) raises:
        if n != len(self.values):
            raise Error(String(what, ": dimension mismatch"))
