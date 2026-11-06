# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Sparse matrices (`mojodec.core.sparse`).
========================================
A compressed-sparse-row matrix of `Float64`, standing in for
`scipy.sparse.csr_matrix`. NuMojo, Linamo, and SciJo provide only dense
arrays, while every DEC operator (boundary, coboundary, Hodge star, mass
matrix) is sparse.

Conventions:
- Matrices are kept canonical: column indices within a row are strictly
  increasing and duplicates are summed. Explicit zeros may remain until
  `eliminate_zeros()` is called, as in SciPy.
- `A @ B` is the matrix product (SciPy's `A * B`), `A @ x` the product with a
  1-D NuMojo array, and `A * s` scales by a scalar.
"""

import numojo as nm
from numojo.prelude import *

from mojodec.core.subset import Subset
from mojodec.core.utils import (
    RealArray,
    real_vector,
    real_array,
    rows_of,
    cols_of,
)


struct COOEntries(Copyable, Movable):
    """A matrix's stored entries as coordinate (COO) triplets."""

    var rows: List[Int]
    """The row of each entry."""
    var cols: List[Int]
    """The column of each entry."""
    var values: List[Float64]
    """The value of each entry."""

    def __init__(
        out self,
        var rows: List[Int],
        var cols: List[Int],
        var values: List[Float64],
    ):
        """Triplets from their three lists.

        Args:
            rows: The rows.
            cols: The columns.
            values: The values.
        """
        self.rows = rows^
        self.cols = cols^
        self.values = values^

    def __len__(self) -> Int:
        """The number of entries.

        Returns:
            `len(rows)`.
        """
        return len(self.rows)


struct CSRMatrix(Copyable, Movable, Writable):
    """A sparse matrix in compressed sparse row format."""

    var nrows: Int
    """Number of rows."""
    var ncols: Int
    """Number of columns."""
    var indptr: List[Int]
    """Row pointers: row `i` occupies `indices[indptr[i]:indptr[i+1]]`."""
    var indices: List[Int]
    """Column index of each stored entry."""
    var data: List[Float64]
    """Value of each stored entry."""

    # ===------------------------------------------------------------------=== #
    # Construction
    # ===------------------------------------------------------------------=== #

    def __init__(out self, nrows: Int, ncols: Int):
        """An all-zero `nrows x ncols` matrix with no stored entries.

        Args:
            nrows: Number of rows.
            ncols: Number of columns.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.indptr = List[Int](length=nrows + 1, fill=0)
        self.indices = List[Int]()
        self.data = List[Float64]()

    def __init__(
        out self,
        nrows: Int,
        ncols: Int,
        var indptr: List[Int],
        var indices: List[Int],
        var data: List[Float64],
    ) raises:
        """A matrix from raw CSR arrays, canonicalized on construction.

        Rows may hold unsorted and duplicate column indices; they are sorted
        and summed.

        Args:
            nrows: Number of rows.
            ncols: Number of columns.
            indptr: Row pointers, `nrows + 1` entries.
            indices: Column index of each entry.
            data: Value of each entry.

        Raises:
            If `indptr` does not have `nrows + 1` entries, `indices` and `data`
            differ in length or from `indptr[nrows]`, or a column index is
            outside `[0, ncols)`.
        """
        if len(indptr) != nrows + 1:
            raise Error("CSRMatrix: indptr must have nrows + 1 entries")
        if len(indices) != len(data) or indptr[nrows] != len(data):
            raise Error("CSRMatrix: inconsistent indices/data lengths")
        for c in indices:
            if c < 0 or c >= ncols:
                raise Error("CSRMatrix: column index out of range")
        self.nrows = nrows
        self.ncols = ncols
        self.indptr = indptr^
        self.indices = indices^
        self.data = data^
        self._canonicalize()

    @staticmethod
    def from_coo(
        nrows: Int,
        ncols: Int,
        rows: List[Int],
        cols: List[Int],
        values: List[Float64],
    ) raises -> Self:
        """A matrix from coordinate (COO) triplets; duplicates are summed.

        Args:
            nrows: Number of rows.
            ncols: Number of columns.
            rows: Row index of each entry.
            cols: Column index of each entry.
            values: Value of each entry.

        Returns:
            The canonical `nrows x ncols` matrix.

        Raises:
            If the three lists differ in length, or a row or column index is
            out of range.
        """
        if len(rows) != len(cols) or len(rows) != len(values):
            raise Error("CSRMatrix.from_coo: triplet lengths differ")
        var indptr = List[Int](length=nrows + 1, fill=0)
        for r in rows:
            if r < 0 or r >= nrows:
                raise Error("CSRMatrix.from_coo: row index out of range")
            indptr[r + 1] += 1
        for i in range(nrows):
            indptr[i + 1] += indptr[i]
        var nnz = len(rows)
        var indices = List[Int](length=nnz, fill=0)
        var data = List[Float64](length=nnz, fill=0)
        var next = indptr.copy()
        # every row index was checked above
        var rp = rows.unsafe_ptr()
        var cp = cols.unsafe_ptr()
        var vp = values.unsafe_ptr()
        var np = next.unsafe_ptr()
        var ixp = indices.unsafe_ptr()
        var dp = data.unsafe_ptr()
        for k in range(nnz):
            var r = rp[unsafe_offset=k]
            var at = np[unsafe_offset=r]
            ixp[unsafe_offset=at] = cp[unsafe_offset=k]
            dp[unsafe_offset=at] = vp[unsafe_offset=k]
            np[unsafe_offset=r] = at + 1
        return Self(nrows, ncols, indptr^, indices^, data^)

    @staticmethod
    def from_csc(
        nrows: Int,
        ncols: Int,
        indptr: List[Int],
        indices: List[Int],
        data: List[Float64],
    ) raises -> Self:
        """A matrix from compressed sparse column arrays.

        Args:
            nrows: Number of rows.
            ncols: Number of columns.
            indptr: Column pointers, `ncols + 1` entries.
            indices: Row index of each entry.
            data: Value of each entry.

        Returns:
            The same matrix in CSR format.

        Raises:
            If the arrays are inconsistent, as for the raw CSR constructor
            with rows and columns swapped.
        """
        var transposed = Self(
            ncols, nrows, indptr.copy(), indices.copy(), data.copy()
        )
        return transposed.transpose()

    @staticmethod
    def identity(n: Int) raises -> Self:
        """The `n x n` identity.

        Args:
            n: The dimension.

        Returns:
            The identity as a `CSRMatrix`.

        Raises:
            Does not raise in practice.
        """
        var ones = List[Float64](length=n, fill=1.0)
        return Self.diag(ones)

    @staticmethod
    def diag(values: List[Float64]) raises -> Self:
        """A square diagonal matrix with the given diagonal.

        Args:
            values: The diagonal entries.

        Returns:
            The `len(values) x len(values)` diagonal matrix.

        Raises:
            Does not raise in practice.
        """
        var n = len(values)
        var indptr = List[Int](capacity=n + 1)
        var indices = List[Int](capacity=n)
        for i in range(n + 1):
            indptr.append(i)
        for i in range(n):
            indices.append(i)
        return Self(n, n, indptr^, indices^, values.copy())

    @staticmethod
    def diag(values: RealArray) raises -> Self:
        """A square diagonal matrix with the diagonal held in a 1-D array.

        Args:
            values: The diagonal entries.

        Returns:
            The `values.size x values.size` diagonal matrix.

        Raises:
            Does not raise in practice.
        """
        var vals = List[Float64](capacity=values.size)
        for i in range(values.size):
            vals.append(values.unsafe_get(i))
        return Self.diag(vals)

    @staticmethod
    def from_dense(a: RealArray) raises -> Self:
        """The sparse form of a dense 2-D array (zeros are not stored).

        Args:
            a: A 2-D array (a 1-D array is read as a column).

        Returns:
            The matrix with the nonzero entries of `a`.

        Raises:
            Does not raise in practice.
        """
        var r = rows_of(a)
        var c = cols_of(a)
        var rows = List[Int]()
        var cols = List[Int]()
        var vals = List[Float64]()
        for i in range(r):
            for j in range(c):
                var v = a.unsafe_get(i * c + j)
                if v != 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(v)
        return Self.from_coo(r, c, rows, cols, vals)

    @staticmethod
    def bmat(blocks: List[List[Optional[CSRMatrix]]]) raises -> Self:
        """Assemble a block matrix; `None` blocks are zero.

        Every block row needs at least one block that fixes its height and
        every block column one that fixes its width, as in `scipy.sparse.bmat`.

        ```mojo
        var blocks = List[List[Optional[CSRMatrix]]]()
        blocks.append([Optional(M.copy()), Optional(B.T())])
        blocks.append([Optional(B.copy()), Optional[CSRMatrix](None)])
        var K = CSRMatrix.bmat(blocks)   # [[M, B^T], [B, 0]]
        ```

        Args:
            blocks: Block rows, each a list of the same length.

        Returns:
            The assembled matrix.

        Raises:
            If the block rows differ in length, two blocks in a block row
            (column) differ in height (width), or a block row or column has
            no blocks.
        """
        var nbr = len(blocks)
        var nbc = len(blocks[0])
        var heights = List[Int](length=nbr, fill=-1)
        var widths = List[Int](length=nbc, fill=-1)
        for i in range(nbr):
            if len(blocks[i]) != nbc:
                raise Error("bmat: ragged block rows")
            for j in range(nbc):
                if blocks[i][j]:
                    ref b = blocks[i][j].value()
                    if heights[i] not in (-1, b.nrows) or widths[j] not in (
                        -1,
                        b.ncols,
                    ):
                        raise Error("bmat: incompatible block shapes")
                    heights[i] = b.nrows
                    widths[j] = b.ncols
        for h in heights:
            if h < 0:
                raise Error("bmat: a block row has no blocks")
        for w in widths:
            if w < 0:
                raise Error("bmat: a block column has no blocks")
        var rows = List[Int]()
        var cols = List[Int]()
        var vals = List[Float64]()
        var row_off = 0
        for i in range(nbr):
            var col_off = 0
            for j in range(nbc):
                if blocks[i][j]:
                    ref b = blocks[i][j].value()
                    for r in range(b.nrows):
                        for k in range(b.indptr[r], b.indptr[r + 1]):
                            rows.append(row_off + r)
                            cols.append(col_off + b.indices[k])
                            vals.append(b.data[k])
                col_off += widths[j]
            row_off += heights[i]
        return Self.from_coo(row_off, _sum(widths), rows, cols, vals)

    # ===------------------------------------------------------------------=== #
    # Inspection
    # ===------------------------------------------------------------------=== #

    def shape(self) -> Tuple[Int, Int]:
        """The shape.

        Returns:
            `(nrows, ncols)`.
        """
        return (self.nrows, self.ncols)

    def nnz(self) -> Int:
        """Number of stored entries (including explicit zeros).

        Returns:
            The length of `data`.
        """
        return len(self.data)

    def __getitem__(self, i: Int, j: Int) raises -> Float64:
        """Entry `(i, j)`, found by binary search in row `i`.

        Args:
            i: Row index.
            j: Column index.

        Returns:
            The stored value, or 0 if none is stored.

        Raises:
            If `i` or `j` is out of range.
        """
        if i < 0 or i >= self.nrows or j < 0 or j >= self.ncols:
            raise Error("CSRMatrix: index out of range")
        var lo = self.indptr[i]
        var hi = self.indptr[i + 1]
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.indices[mid] < j:
                lo = mid + 1
            else:
                hi = mid
        if lo < self.indptr[i + 1] and self.indices[lo] == j:
            return self.data[lo]
        return 0

    def diagonal(self) raises -> RealArray:
        """The main diagonal as a 1-D array.

        Returns:
            The `min(nrows, ncols)` diagonal entries.

        Raises:
            If NuMojo cannot allocate the array.
        """
        var n = min(self.nrows, self.ncols)
        var out = real_vector(n)
        for i in range(n):
            out.unsafe_set(i, self[i, i])
        return out^

    def todense(self) raises -> RealArray:
        """The dense `nrows x ncols` array.

        Returns:
            A new 2-D array.

        Raises:
            If NuMojo cannot allocate the array.
        """
        var out = real_array(self.nrows, self.ncols)
        for i in range(self.nrows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                var off = i * self.ncols + self.indices[k]
                out.unsafe_set(off, out.unsafe_get(off) + self.data[k])
        return out^

    def max_abs(self) -> Float64:
        """Largest absolute value of a stored entry (0 if none).

        Returns:
            The maximum of `abs(v)` over `data`.
        """
        var m: Float64 = 0
        for v in self.data:
            m = max(m, abs(v))
        return m

    def write_to(self, mut writer: Some[Writer]):
        """Write a header line, then the stored entries, one
        `(row, col)  value` per line, as SciPy prints them.

        Args:
            writer: The destination.
        """
        writer.write(
            "<",
            self.nrows,
            "x",
            self.ncols,
            " CSRMatrix with ",
            self.nnz(),
            " stored elements>",
        )
        for i in range(self.nrows):
            for k in range(self.indptr[i], self.indptr[i + 1]):
                writer.write(
                    "\n  (", i, ", ", self.indices[k], ")\t", self.data[k]
                )

    # ===------------------------------------------------------------------=== #
    # Structural operations
    # ===------------------------------------------------------------------=== #

    def transpose(self) raises -> Self:
        """The transpose, in CSR format.

        Returns:
            The `ncols x nrows` transpose.

        Raises:
            Does not raise in practice.
        """
        var indptr = List[Int](length=self.ncols + 1, fill=0)
        var nnz = self.nnz()
        var ip = self.indptr.unsafe_ptr()
        var ix = self.indices.unsafe_ptr()
        var dv = self.data.unsafe_ptr()
        var tp = indptr.unsafe_ptr()
        for k in range(nnz):
            tp[unsafe_offset=ix[unsafe_offset=k] + 1] += 1
        for j in range(self.ncols):
            tp[unsafe_offset=j + 1] += tp[unsafe_offset=j]
        var indices = List[Int](length=nnz, fill=0)
        var data = List[Float64](length=nnz, fill=0)
        var next = indptr.copy()
        var np = next.unsafe_ptr()
        var tix = indices.unsafe_ptr()
        var tdv = data.unsafe_ptr()
        for i in range(self.nrows):
            for k in range(ip[unsafe_offset=i], ip[unsafe_offset=i + 1]):
                var c = ix[unsafe_offset=k]
                var at = np[unsafe_offset=c]
                tix[unsafe_offset=at] = i
                tdv[unsafe_offset=at] = dv[unsafe_offset=k]
                np[unsafe_offset=c] = at + 1
        var out = Self(self.ncols, self.nrows)
        out.indptr = indptr^
        out.indices = indices^
        out.data = data^
        return out^

    def T(self) raises -> Self:
        """The transpose (alias of `transpose`).

        Returns:
            The `ncols x nrows` transpose.

        Raises:
            Does not raise in practice.
        """
        return self.transpose()

    def eliminate_zeros(mut self):
        """Drop stored entries equal to zero."""
        var w = 0
        var start = 0
        for i in range(self.nrows):
            var end = self.indptr[i + 1]
            for k in range(start, end):
                if self.data[k] != 0:
                    self.indices[w] = self.indices[k]
                    self.data[w] = self.data[k]
                    w += 1
            start = end
            self.indptr[i + 1] = w
        self.indices.resize(w, 0)
        self.data.resize(w, 0)

    def exclude_subset(self, subset: Subset) raises -> Self:
        """The matrix with the rows of `subset` emptied, so its products are
        zero on those cells (e.g. Dirichlet boundary values held fixed).

        Args:
            subset: The cells indexing the rows.

        Returns:
            A copy with those rows empty.

        Raises:
            If `subset.size` differs from `nrows`.
        """
        return self._keep_rows(subset, False)

    def restrict_to_subset(self, subset: Subset) raises -> Self:
        """The matrix with only the rows of `subset` kept, so its products
        are zero off those cells.

        Args:
            subset: The cells indexing the rows.

        Returns:
            A copy with the other rows empty.

        Raises:
            If `subset.size` differs from `nrows`.
        """
        return self._keep_rows(subset, True)

    def _keep_rows(self, subset: Subset, inside: Bool) raises -> Self:
        if subset.size != self.nrows:
            raise Error("CSRMatrix: subset size differs from the row count")
        var out = Self(self.nrows, self.ncols)
        out.indices = List[Int](capacity=len(self.indices))
        out.data = List[Float64](capacity=len(self.data))
        for i in range(self.nrows):
            if (i in subset) == inside:
                for k in range(self.indptr[i], self.indptr[i + 1]):
                    out.indices.append(self.indices[k])
                    out.data.append(self.data[k])
            out.indptr[i + 1] = len(out.indices)
        return out^

    def select(self, rows: List[Int], cols: List[Int]) raises -> Self:
        """The submatrix `A[rows][:, cols]` (fancy indexing, in given order).

        Rows may repeat; columns may not.

        Args:
            rows: Row indices to take, `0 <= i < nrows`.
            cols: Distinct column indices to take, `0 <= j < ncols`.

        Returns:
            The `len(rows) x len(cols)` submatrix.

        Raises:
            If an index is out of range, or `cols` repeats one.
        """
        for i in rows:
            if i < 0 or i >= self.nrows:
                raise Error(
                    String("CSRMatrix.select: row ", i, " out of range")
                )
        var colmap = List[Int](length=self.ncols, fill=-1)
        for j in range(len(cols)):
            if cols[j] < 0 or cols[j] >= self.ncols:
                raise Error(
                    String(
                        "CSRMatrix.select: column ", cols[j], " out of range"
                    )
                )
            if colmap[cols[j]] != -1:
                raise Error("CSRMatrix.select: repeated column index")
            colmap[cols[j]] = j
        var r = List[Int]()
        var c = List[Int]()
        var v = List[Float64]()
        for i in range(len(rows)):
            var src = rows[i]
            for k in range(self.indptr[src], self.indptr[src + 1]):
                var j = colmap[self.indices[k]]
                if j >= 0:
                    r.append(i)
                    c.append(j)
                    v.append(self.data[k])
        return Self.from_coo(len(rows), len(cols), r, c, v)

    def select_rows(self, rows: List[Int]) raises -> Self:
        """The rows `A[rows]`, in given order.

        Args:
            rows: Row indices to take, `0 <= i < nrows`.

        Returns:
            The `len(rows) x ncols` submatrix.

        Raises:
            If a row index is out of range.
        """
        return self.select(rows, _range(self.ncols))

    def select_cols(self, cols: List[Int]) raises -> Self:
        """The columns `A[:, cols]`, in given order.

        Args:
            cols: Distinct column indices to take, `0 <= j < ncols`.

        Returns:
            The `nrows x len(cols)` submatrix.

        Raises:
            If a column index is out of range, or `cols` repeats one.
        """
        return self.select(_range(self.nrows), cols)

    def to_coo(self) -> COOEntries:
        """The stored entries as COO triplets.

        Returns:
            The rows, columns and values, in row-major order.
        """
        var rows = List[Int](capacity=self.nnz())
        for i in range(self.nrows):
            for _ in range(self.indptr[i], self.indptr[i + 1]):
                rows.append(i)
        return COOEntries(rows^, self.indices.copy(), self.data.copy())

    # ===------------------------------------------------------------------=== #
    # Arithmetic
    # ===------------------------------------------------------------------=== #

    def __matmul__(self, other: Self) raises -> Self:
        """The sparse matrix product `self @ other`.

        Args:
            other: The right factor.

        Returns:
            The `nrows x other.ncols` product.

        Raises:
            If `ncols` differs from `other.nrows`.
        """
        if self.ncols != other.nrows:
            raise Error("CSRMatrix @: dimension mismatch")
        var indptr = List[Int](capacity=self.nrows + 1)
        indptr.append(0)
        var indices = List[Int]()
        var data = List[Float64]()
        var acc = List[Float64](length=other.ncols, fill=0)
        var marker = List[Int](length=other.ncols, fill=-1)
        var touched = List[Int]()
        var aip = self.indptr.unsafe_ptr()
        var aix = self.indices.unsafe_ptr()
        var adv = self.data.unsafe_ptr()
        var bip = other.indptr.unsafe_ptr()
        var bix = other.indices.unsafe_ptr()
        var bdv = other.data.unsafe_ptr()
        var accp = acc.unsafe_ptr()
        var markp = marker.unsafe_ptr()
        for i in range(self.nrows):
            touched.clear()
            for ka in range(aip[unsafe_offset=i], aip[unsafe_offset=i + 1]):
                var j = aix[unsafe_offset=ka]
                var a = adv[unsafe_offset=ka]
                for kb in range(bip[unsafe_offset=j], bip[unsafe_offset=j + 1]):
                    var c = bix[unsafe_offset=kb]
                    if markp[unsafe_offset=c] != i:
                        markp[unsafe_offset=c] = i
                        accp[unsafe_offset=c] = 0
                        touched.append(c)
                    accp[unsafe_offset=c] += a * bdv[unsafe_offset=kb]
            sort(Span(touched))
            for c in touched:
                indices.append(c)
                data.append(accp[unsafe_offset=c])
            indptr.append(len(indices))
        var out = Self(self.nrows, other.ncols)
        out.indptr = indptr^
        out.indices = indices^
        out.data = data^
        return out^

    def __matmul__(self, x: RealArray) raises -> RealArray:
        """The matrix-vector product with a 1-D array.

        Args:
            x: A vector of length `ncols`.

        Returns:
            The product, of length `nrows`.

        Raises:
            If `x.size` differs from `ncols`.
        """
        if x.size != self.ncols:
            raise Error("CSRMatrix @ vector: dimension mismatch")
        var y = real_vector(self.nrows)
        self.matvec_into(x, y)
        return y^

    def matvec_into(self, x: RealArray, mut y: RealArray) raises:
        """The matrix-vector product `self @ x`, written into `y`.

        Iterative solvers call this once per iteration with the same `y`,
        so nothing is allocated.

        Args:
            x: A vector of length `ncols`.
            y: A vector of length `nrows`, overwritten with the product.

        Raises:
            If `x.size` differs from `ncols` or `y.size` from `nrows`.
        """
        if x.size != self.ncols or y.size != self.nrows:
            raise Error("CSRMatrix.matvec_into: dimension mismatch")
        var ip = self.indptr.unsafe_ptr()
        var ix = self.indices.unsafe_ptr()
        var dv = self.data.unsafe_ptr()
        var xp = x.unsafe_ptr()
        var yp = y.unsafe_ptr()
        for i in range(self.nrows):
            var s: Float64 = 0
            for k in range(ip[unsafe_offset=i], ip[unsafe_offset=i + 1]):
                s += dv[unsafe_offset=k] * xp[unsafe_offset=ix[unsafe_offset=k]]
            yp[unsafe_offset=i] = s

    def __mul__(self, s: Float64) -> Self:
        """Every entry multiplied by `s`.

        Args:
            s: The scale factor.

        Returns:
            The scaled matrix.
        """
        var out = self.copy()
        for k in range(len(out.data)):
            out.data[k] *= s
        return out^

    def __rmul__(self, s: Float64) -> Self:
        """Every entry multiplied by `s`.

        Args:
            s: The scale factor.

        Returns:
            The scaled matrix.
        """
        return self * s

    def __imul__(mut self, s: Float64):
        """Scale in place by `s`.

        Args:
            s: The scale factor.
        """
        for k in range(len(self.data)):
            self.data[k] *= s

    def __truediv__(self, s: Float64) -> Self:
        """Every entry divided by `s`.

        Args:
            s: The divisor.

        Returns:
            The scaled matrix.
        """
        return self * (1.0 / s)

    def __neg__(self) -> Self:
        """The negation.

        Returns:
            The matrix times -1.
        """
        return self * -1.0

    def __add__(self, other: Self) raises -> Self:
        """The sum of two matrices of the same shape.

        Args:
            other: The matrix to add.

        Returns:
            The entrywise sum.

        Raises:
            If the shapes differ.
        """
        return self._combine(other, 1.0)

    def __sub__(self, other: Self) raises -> Self:
        """The difference of two matrices of the same shape.

        Args:
            other: The matrix to subtract.

        Returns:
            The entrywise difference.

        Raises:
            If the shapes differ.
        """
        return self._combine(other, -1.0)

    # ===------------------------------------------------------------------=== #
    # Private helpers
    # ===------------------------------------------------------------------=== #

    def _combine(self, other: Self, sign: Float64) raises -> Self:
        if self.nrows != other.nrows or self.ncols != other.ncols:
            raise Error("CSRMatrix +/-: shape mismatch")
        var indptr = List[Int](capacity=self.nrows + 1)
        indptr.append(0)
        var indices = List[Int]()
        var data = List[Float64]()
        for i in range(self.nrows):
            var a = self.indptr[i]
            var ae = self.indptr[i + 1]
            var b = other.indptr[i]
            var be = other.indptr[i + 1]
            while a < ae or b < be:
                if b >= be or (a < ae and self.indices[a] < other.indices[b]):
                    indices.append(self.indices[a])
                    data.append(self.data[a])
                    a += 1
                elif a >= ae or other.indices[b] < self.indices[a]:
                    indices.append(other.indices[b])
                    data.append(sign * other.data[b])
                    b += 1
                else:
                    indices.append(self.indices[a])
                    data.append(self.data[a] + sign * other.data[b])
                    a += 1
                    b += 1
            indptr.append(len(indices))
        var out = Self(self.nrows, self.ncols)
        out.indptr = indptr^
        out.indices = indices^
        out.data = data^
        return out^

    def _canonicalize(mut self):
        """Sort column indices within each row and sum duplicates."""
        var ip = self.indptr.unsafe_ptr()
        var ix = self.indices.unsafe_ptr()
        var canonical = True
        for i in range(self.nrows):
            for k in range(ip[unsafe_offset=i] + 1, ip[unsafe_offset=i + 1]):
                if ix[unsafe_offset=k - 1] >= ix[unsafe_offset=k]:
                    canonical = False
                    break
            if not canonical:
                break
        if canonical:
            return
        var new_indices = List[Int](capacity=len(self.indices))
        var new_data = List[Float64](capacity=len(self.data))
        var new_indptr = List[Int](capacity=self.nrows + 1)
        new_indptr.append(0)
        var perm = List[Int]()
        for i in range(self.nrows):
            var lo = ip[unsafe_offset=i]
            var hi = ip[unsafe_offset=i + 1]
            perm.clear()
            for k in range(lo, hi):
                perm.append(k)
            # insertion sort by column: rows of DEC operators are short
            for a in range(1, len(perm)):
                var key = perm.unsafe_get(a)
                var b = a - 1
                while (
                    b >= 0
                    and ix[unsafe_offset=perm.unsafe_get(b)]
                    > ix[unsafe_offset=key]
                ):
                    perm.unsafe_set(b + 1, perm.unsafe_get(b))
                    b -= 1
                perm.unsafe_set(b + 1, key)
            for p in perm:
                var c = ix[unsafe_offset=p]
                if (
                    len(new_indices) > new_indptr[i]
                    and new_indices[len(new_indices) - 1] == c
                ):
                    new_data[len(new_data) - 1] += self.data[p]
                else:
                    new_indices.append(c)
                    new_data.append(self.data[p])
            new_indptr.append(len(new_indices))
        self.indptr = new_indptr^
        self.indices = new_indices^
        self.data = new_data^


def _range(n: Int) -> List[Int]:
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(i)
    return out^


def _sum(xs: List[Int]) -> Int:
    var s = 0
    for x in xs:
        s += x
    return s
