# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Chain complexes (`mojodec.core.chain_complex`).
===============================================
The boundary operators `B_k` (k-chains to (k-1)-chains, `N_{k-1} x N_k`)
of a complex of dimension n, for k = 1 to n, and the coboundary operators
`d_k = B_{k+1}^T` (k-cochains to (k+1)-cochains), for k = 0 to n-1.

PyDEC returns these as two lists padded with zero matrices (a `1 x N_0`
`B_0` and a `1 x N_n` `d_n`); here there is no padding, and each matrix is
stored once and read by reference.

```mojo
var cc = sc.chain_complex()
for k in range(1, cc.complex_dimension()):
    print((cc.boundary(k) @ cc.boundary(k + 1)).max_abs())   # 0
```
"""

from mojodec.core.cell_set import (
    CellComplex,
    cells_in_rows,
    cells_once_in_rows,
)
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset


struct ChainComplex(CellComplex, Copyable, Movable, Sized, Writable):
    """The boundary and coboundary operators of a complex."""

    var _num_cells: List[Int]
    """`_num_cells[k]`: the number of k-cells, `N_k`."""
    var _boundary: List[CSRMatrix]
    """`_boundary[k - 1]` is `B_k`, for k = 1 to n."""
    var _coboundary: List[CSRMatrix]
    """`_coboundary[k]` is `d_k = B_{k+1}^T`, for k = 0 to n-1."""

    def __init__(
        out self, num_vertices: Int, var boundaries: List[CSRMatrix]
    ) raises:
        """A chain complex from its boundary operators; the coboundaries are
        their transposes.

        Args:
            num_vertices: `N_0`, the number of 0-cells.
            boundaries: `B_1, ..., B_n`, with `B_k` of shape `N_{k-1} x N_k`;
                empty for a complex of vertices only.

        Raises:
            If the shapes do not chain (`B_1` must have `num_vertices` rows
            and `B_{k+1}` as many rows as `B_k` has columns), or
            `B_k B_{k+1}` is not zero.
        """
        var coboundaries = List[CSRMatrix](capacity=len(boundaries))
        for ref B in boundaries:
            coboundaries.append(B.transpose())
        self = Self(num_vertices, boundaries^, coboundaries^)

    def __init__(
        out self,
        num_vertices: Int,
        var boundaries: List[CSRMatrix],
        var coboundaries: List[CSRMatrix],
    ) raises:
        """A chain complex from both kinds of operators, which must agree.

        Args:
            num_vertices: `N_0`, the number of 0-cells.
            boundaries: `B_1, ..., B_n`.
            coboundaries: `d_0, ..., d_{n-1}`, with `d_k = B_{k+1}^T`.

        Raises:
            If the lists differ in length, the shapes do not chain, a
            coboundary is not the transpose of its boundary, or
            `B_k B_{k+1}` is not zero (to rounding: entries at most `1e-12`
            times the largest possible product term sum).
        """
        if len(boundaries) != len(coboundaries):
            raise Error("ChainComplex: boundary and coboundary counts differ")
        self._num_cells = [num_vertices]
        for k in range(len(boundaries)):
            ref B = boundaries[k]
            ref d = coboundaries[k]
            if B.nrows != self._num_cells[k]:
                raise Error(
                    String(
                        "ChainComplex: B_",
                        k + 1,
                        " has ",
                        B.nrows,
                        " rows, expected ",
                        self._num_cells[k],
                    )
                )
            if d.nrows != B.ncols or d.ncols != B.nrows:
                raise Error(
                    String(
                        "ChainComplex: d_",
                        k,
                        " is not the shape of B_",
                        k + 1,
                        "^T",
                    )
                )
            self._num_cells.append(B.ncols)
        for k in range(len(boundaries)):
            if (coboundaries[k] - boundaries[k].transpose()).max_abs() != 0:
                raise Error(
                    String("ChainComplex: d_", k, " is not B_", k + 1, "^T")
                )
        for k in range(len(boundaries) - 1):
            ref A = boundaries[k]
            ref B = boundaries[k + 1]
            var width = 0
            for r in range(A.nrows):
                width = max(width, A.indptr[r + 1] - A.indptr[r])
            var tol = 1e-12 * Float64(width) * A.max_abs() * B.max_abs()
            if (A @ B).max_abs() > tol:
                raise Error(
                    String(
                        "ChainComplex: B_",
                        k + 1,
                        " B_",
                        k + 2,
                        " is not zero; the boundary of a boundary must vanish",
                    )
                )
        self._boundary = boundaries^
        self._coboundary = coboundaries^

    def complex_dimension(self) -> Int:
        """The dimension n of the complex.

        Returns:
            The number of boundary operators.
        """
        return len(self._boundary)

    def __len__(self) -> Int:
        """The number of cell dimensions.

        Returns:
            `complex_dimension() + 1`.
        """
        return len(self._num_cells)

    def num_cells(self, k: Int) raises -> Int:
        """The number of k-cells, `N_k`.

        Args:
            k: A dimension in `0..n`.

        Returns:
            The cell count.

        Raises:
            If `k` is outside `0..n`.
        """
        if k < 0 or k >= len(self._num_cells):
            raise Error(String("ChainComplex: no ", k, "-cells"))
        return self._num_cells[k]

    def boundary(self, k: Int) raises -> ref[self._boundary] CSRMatrix:
        """The boundary operator `B_k`, from k-chains to (k-1)-chains.

        Args:
            k: A dimension in `1..n`.

        Returns:
            The `N_{k-1} x N_k` matrix, by reference.

        Raises:
            If `k` is outside `1..n`.
        """
        if k < 1 or k > len(self._boundary):
            raise Error(
                String(
                    "ChainComplex.boundary: k = ",
                    k,
                    " outside 1..",
                    len(self._boundary),
                )
            )
        return self._boundary.unsafe_ptr()[unsafe_offset=k - 1]

    def coboundary(self, k: Int) raises -> ref[self._coboundary] CSRMatrix:
        """The coboundary operator `d_k = B_{k+1}^T`, from k-cochains to
        (k+1)-cochains.

        Args:
            k: A dimension in `0..n-1`.

        Returns:
            The `N_{k+1} x N_k` matrix, by reference.

        Raises:
            If `k` is outside `0..n-1`.
        """
        if k < 0 or k >= len(self._coboundary):
            raise Error(
                String(
                    "ChainComplex.coboundary: k = ",
                    k,
                    " outside 0..",
                    len(self._coboundary) - 1,
                )
            )
        return self._coboundary.unsafe_ptr()[unsafe_offset=k]

    def faces_of(self, cells: Subset) raises -> Subset:
        """The (k-1)-cells that are faces of some k-cell of `cells`.

        Args:
            cells: Primal k-cells, k > 0.

        Returns:
            Primal (k-1)-cells.

        Raises:
            If k = 0 or `cells` does not fit the complex.
        """
        # row i of d_{k-1} lists the faces of k-cell i
        return cells_in_rows(
            self.coboundary(cells.dim - 1), cells, cells.dim - 1
        )

    def cofaces_of(self, cells: Subset) raises -> Subset:
        """The (k+1)-cells that have some k-cell of `cells` as a face.

        Args:
            cells: Primal k-cells, k < n.

        Returns:
            Primal (k+1)-cells.

        Raises:
            If k = n or `cells` does not fit the complex.
        """
        # row i of B_{k+1} lists the cofaces of k-cell i
        return cells_in_rows(self.boundary(cells.dim + 1), cells, cells.dim + 1)

    def boundary_faces_of(self, cells: Subset) raises -> Subset:
        """The (k-1)-cells that are faces of exactly one k-cell of `cells`.

        Args:
            cells: Primal k-cells, k > 0.

        Returns:
            Primal (k-1)-cells.

        Raises:
            If k = 0 or `cells` does not fit the complex.
        """
        return cells_once_in_rows(
            self.coboundary(cells.dim - 1), cells, cells.dim - 1
        )

    def write_to(self, mut writer: Some[Writer]):
        """Write the number of cells of each dimension.

        Args:
            writer: The destination.
        """
        writer.write("ChainComplex(")
        for k in range(len(self._num_cells)):
            if k > 0:
                writer.write(", ")
            writer.write(self._num_cells[k], " ", k, "-cells")
        writer.write(")")
