# ===----------------------------------------------------------------------=== #
# molina: linear algebra in Mojo (sparse and dense matrices, direct and
# iterative solvers, eigensolvers, multigrid), split from MojoDEC.
# ===----------------------------------------------------------------------=== #
"""
Real arrays (`molina.real_array`).
==================================
`RealArray`: a C-contiguous vector or matrix of `Float64`, the vectors
and dense matrices of the linear algebra (`CSRMatrix` products, the
solvers, `dense`). It has the part of NuMojo's `NDArray[DType.float64]`
that MojoDEC used, under the same names: flat element access
(`unsafe_get`, `unsafe_set`, SIMD `unsafe_load`/`unsafe_store`,
`unsafe_ptr`), `size`, `ndim`, and the shape, so molina does not depend
on NuMojo.

Element `(i, j)` of an `r x c` array is at flat offset `i * c + j`; a
vector has `ndim == 1` and one column.

```mojo
var x = RealArray.zeros(3)              # a vector
var A = RealArray.zeros(2, 3)           # a 2 x 3 matrix
A.unsafe_set(1 * 3 + 2, 5.0)            # A[1, 2] = 5
print(A[1, 2], A.rows(), A.cols())      # 5.0 2 3
```
"""

from std.memory import unsafe_memcpy, unsafe_memset_zero
from std.python import Python, PythonObject


struct RealArray(Copyable, Movable, Sized, Writable):
    """A C-contiguous vector (`ndim == 1`) or matrix (`ndim == 2`) of
    `Float64`."""

    var _data: List[Float64]
    """The entries, row by row."""
    var ndim: Int
    """1 for a vector, 2 for a matrix."""
    var size: Int
    """The number of entries."""
    var _rows: Int
    var _cols: Int

    def __init__(out self, *, var values: List[Float64]):
        """A vector holding `values`.

        Args:
            values: The entries, taken.
        """
        self.size = len(values)
        self.ndim = 1
        self._rows = self.size
        self._cols = 1
        self._data = values^

    def __init__(
        out self, rows: Int, cols: Int, *, var values: List[Float64]
    ) raises:
        """A `rows x cols` matrix holding `values` row by row.

        Args:
            rows: The number of rows.
            cols: The number of columns.
            values: The `rows * cols` entries, row by row, taken.

        Raises:
            If `rows` or `cols` is negative, or `values` has the wrong
            length.
        """
        if rows < 0 or cols < 0:
            raise Error("RealArray: negative dimension")
        if len(values) != rows * cols:
            raise Error(
                String(
                    "RealArray: ",
                    len(values),
                    " values for a ",
                    rows,
                    " x ",
                    cols,
                    " matrix",
                )
            )
        self.size = rows * cols
        self.ndim = 2
        self._rows = rows
        self._cols = cols
        self._data = values^

    @staticmethod
    def zeros(n: Int) raises -> Self:
        """A vector of `n` zeros.

        Args:
            n: The length.

        Returns:
            The zero vector.

        Raises:
            If `n` is negative.
        """
        if n < 0:
            raise Error("RealArray: negative length")
        return Self(values=_zeros(n))

    @staticmethod
    def zeros(rows: Int, cols: Int) raises -> Self:
        """A `rows x cols` matrix of zeros.

        Args:
            rows: The number of rows.
            cols: The number of columns.

        Returns:
            The zero matrix.

        Raises:
            If `rows` or `cols` is negative.
        """
        if rows < 0 or cols < 0:
            raise Error("RealArray: negative dimension")
        return Self(rows, cols, values=_zeros(rows * cols))

    @staticmethod
    def ones(n: Int) raises -> Self:
        """A vector of `n` ones.

        Args:
            n: The length.

        Returns:
            The vector of ones.

        Raises:
            If `n` is negative.
        """
        if n < 0:
            raise Error("RealArray: negative length")
        return Self(values=List[Float64](length=n, fill=1.0))

    @staticmethod
    def identity(n: Int) raises -> Self:
        """The `n x n` identity matrix.

        Args:
            n: The order.

        Returns:
            The identity.

        Raises:
            If `n` is negative.
        """
        var out = Self.zeros(n, n)
        for i in range(n):
            out.unsafe_set(i * n + i, 1.0)
        return out^

    def rows(self) -> Int:
        """The number of rows (a vector's length).

        Returns:
            The first dimension.
        """
        return self._rows

    def cols(self) -> Int:
        """The number of columns (1 for a vector).

        Returns:
            The second dimension.
        """
        return self._cols

    def __len__(self) -> Int:
        """The number of rows, as NumPy's `len`.

        Returns:
            The first dimension.
        """
        return self._rows

    @always_inline
    def unsafe_get(self, index: Int) -> Float64:
        """The entry at flat `index`, unchecked.

        Args:
            index: The flat index, `0 <= index < size`.

        Returns:
            The entry.
        """
        return self._data.unsafe_ptr()[unsafe_offset=index]

    @always_inline
    def unsafe_set(mut self, index: Int, value: Float64):
        """Set the entry at flat `index`, unchecked.

        Args:
            index: The flat index, `0 <= index < size`.
            value: The new entry.
        """
        self._data.unsafe_ptr()[unsafe_offset=index] = value

    @always_inline
    def unsafe_load[
        width: Int = 1
    ](self, index: Int) -> SIMD[DType.float64, width]:
        """`width` consecutive entries from flat `index`, unchecked.

        Parameters:
            width: The number of entries.

        Args:
            index: The flat index of the first.

        Returns:
            The entries as a vector.
        """
        return self._data.unsafe_ptr().unsafe_load[width=width](index)

    @always_inline
    def unsafe_store[
        width: Int = 1
    ](mut self, index: Int, value: SIMD[DType.float64, width]):
        """Store `width` consecutive entries from flat `index`, unchecked.

        Parameters:
            width: The number of entries.

        Args:
            index: The flat index of the first.
            value: The entries.
        """
        self._data.unsafe_ptr().unsafe_store(index, value)

    def unsafe_ptr[
        mutable: Bool,
        //,
        org: Origin[mut=mutable],
    ](ref[org] self) -> Pointer[Float64, org]:
        """A pointer to the first entry.

        Parameters:
            mutable: Whether the pointer may write.
            org: The origin of the array the pointer borrows.

        Returns:
            A pointer whose origin is tied to this array.
        """
        return (
            self._data.unsafe_ptr()
            .unsafe_mut_cast[mutable]()
            .unsafe_origin_cast[org]()
        )

    def __getitem__(self, i: Int) raises -> Float64:
        """Entry `i` of a vector, bounds-checked.

        Args:
            i: The index, `0 <= i < size`.

        Returns:
            The entry.

        Raises:
            If `i` is out of range or the array is a matrix.
        """
        if self.ndim != 1:
            raise Error("RealArray: one index into a matrix")
        if i < 0 or i >= self.size:
            raise Error(String("RealArray: index ", i, " out of range"))
        return self.unsafe_get(i)

    def __getitem__(self, i: Int, j: Int) raises -> Float64:
        """Entry `(i, j)` of a matrix, bounds-checked.

        Args:
            i: The row.
            j: The column.

        Returns:
            The entry.

        Raises:
            If `(i, j)` is out of range or the array is a vector.
        """
        self._check(i, j)
        return self.unsafe_get(i * self._cols + j)

    def __setitem__(mut self, i: Int, value: Float64) raises:
        """Set entry `i` of a vector, bounds-checked.

        Args:
            i: The index, `0 <= i < size`.
            value: The new entry.

        Raises:
            If `i` is out of range or the array is a matrix.
        """
        if self.ndim != 1:
            raise Error("RealArray: one index into a matrix")
        if i < 0 or i >= self.size:
            raise Error(String("RealArray: index ", i, " out of range"))
        self.unsafe_set(i, value)

    def __setitem__(mut self, i: Int, j: Int, value: Float64) raises:
        """Set entry `(i, j)` of a matrix, bounds-checked.

        Args:
            i: The row.
            j: The column.
            value: The new entry.

        Raises:
            If `(i, j)` is out of range or the array is a vector.
        """
        self._check(i, j)
        self.unsafe_set(i * self._cols + j, value)

    def _check(self, i: Int, j: Int) raises:
        if self.ndim != 2:
            raise Error("RealArray: two indices into a vector")
        if i < 0 or i >= self._rows or j < 0 or j >= self._cols:
            raise Error(
                String("RealArray: index (", i, ", ", j, ") out of range")
            )

    def _same_shape(self, other: Self, op: String) raises:
        if (
            self.ndim != other.ndim
            or self._rows != other._rows
            or self._cols != other._cols
        ):
            raise Error(String("RealArray ", op, ": shapes differ"))

    def __add__(self, other: Self) raises -> Self:
        """The entrywise sum.

        Args:
            other: An array of the same shape.

        Returns:
            `self + other`.

        Raises:
            If the shapes differ.
        """
        self._same_shape(other, "+")
        var out = self.copy()
        for i in range(self.size):
            out.unsafe_set(i, self.unsafe_get(i) + other.unsafe_get(i))
        return out^

    def __sub__(self, other: Self) raises -> Self:
        """The entrywise difference.

        Args:
            other: An array of the same shape.

        Returns:
            `self - other`.

        Raises:
            If the shapes differ.
        """
        self._same_shape(other, "-")
        var out = self.copy()
        for i in range(self.size):
            out.unsafe_set(i, self.unsafe_get(i) - other.unsafe_get(i))
        return out^

    def __mul__(self, s: Float64) -> Self:
        """The array scaled by `s`.

        Args:
            s: The factor.

        Returns:
            `self * s`, entry by entry.
        """
        var out = self.copy()
        for i in range(self.size):
            out.unsafe_set(i, self.unsafe_get(i) * s)
        return out^

    def __rmul__(self, s: Float64) -> Self:
        """The array scaled by `s`, for `s * a`.

        Args:
            s: The factor.

        Returns:
            `s * self`, entry by entry.
        """
        var out = self.copy()
        for i in range(self.size):
            out.unsafe_set(i, s * self.unsafe_get(i))
        return out^

    def __itruediv__(mut self, s: Float64):
        """Divide every entry by `s` (a division, not a product with
        `1 / s`).

        Args:
            s: The divisor.
        """
        for i in range(self.size):
            self.unsafe_set(i, self.unsafe_get(i) / s)

    def __neg__(self) -> Self:
        """The entrywise negation.

        Returns:
            `-self`.
        """
        var out = self.copy()
        for i in range(self.size):
            out.unsafe_set(i, -self.unsafe_get(i))
        return out^

    def to_numpy(self) raises -> PythonObject:
        """A NumPy copy of the array (`float64`, the same shape).

        Imports NumPy, so it loads Python; nothing else here does.

        Returns:
            The NumPy array.

        Raises:
            If NumPy cannot be imported.
        """
        var np = Python.import_module("numpy")
        np.set_printoptions(4)  # as NuMojo's `to_numpy` did
        var dims = Python.list()
        dims.append(self._rows)
        if self.ndim == 2:
            dims.append(self._cols)
        var out = np.empty(dims, dtype=np.float64, order=PythonObject("C"))
        var dst = out.__array_interface__[PythonObject("data")][
            0
        ].unsafe_get_as_pointer[DType.float64]()
        unsafe_memcpy(dest=dst, src=self._data.unsafe_ptr(), count=self.size)
        return out^

    def write_to(self, mut writer: Some[Writer]):
        """Write the entries, a matrix row by row, in brackets.

        Args:
            writer: The destination.
        """
        writer.write("[")
        if self.ndim == 1:
            for i in range(self.size):
                if i > 0:
                    writer.write(", ")
                writer.write(self.unsafe_get(i))
        else:
            for i in range(self._rows):
                if i > 0:
                    writer.write(",\n ")
                writer.write("[")
                for j in range(self._cols):
                    if j > 0:
                        writer.write(", ")
                    writer.write(self.unsafe_get(i * self._cols + j))
                writer.write("]")
        writer.write("]")


def _zeros(n: Int) -> List[Float64]:
    """`n` zeros, by one memset."""
    var out = List[Float64](unsafe_uninit_length=n)
    if n > 0:
        unsafe_memset_zero(out.unsafe_ptr(), n)
    return out^
