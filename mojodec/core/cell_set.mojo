# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Sets of cells of every dimension (`mojodec.core.cell_set`).
===========================================================
A `CellSet` holds cells of a complex of every dimension `0..n`, one
`Subset` (a bitset) per dimension; the neighborhood queries of
`mojodec.routines.topology` take and return them.

`CellComplex` is the trait those queries need from a complex: its cell
counts and its face and coface incidence. `SimplicialComplex`,
`RegularCubeComplex`, `AbstractSimplicialComplex`, `RipsComplex`, and
`ChainComplex` implement it.

```mojo
var ring = sc.link(Simplex([v]))         # a CellSet
print(len(ring[0]), "neighbors of", v)
for e in ring[1].indices():              # the ring's edges
    ...
var region = CellSet.from_lists(sc, [[], [], [3, 7]])
var grown = sc.closure(sc.star(region))
```
"""

from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.primality import Primality
from mojodec.core.subset import Subset


trait CellComplex:
    """A complex of cells of dimensions `0..n` with face incidence: what the
    neighborhood queries of `mojodec.routines.topology` need."""

    def complex_dimension(self) -> Int:
        """The highest cell dimension n.

        Returns:
            The dimension n.
        """
        ...

    def num_cells(self, k: Int) raises -> Int:
        """The number of k-cells.

        Args:
            k: A dimension in `0..n`.

        Returns:
            The cell count.

        Raises:
            If `k` is outside `0..n`.
        """
        ...

    def faces_of(self, cells: Subset) raises -> Subset:
        """The (k-1)-cells that are faces of some k-cell of `cells`.

        Args:
            cells: Primal k-cells, k > 0.

        Returns:
            Primal (k-1)-cells.

        Raises:
            If k = 0 or `cells` does not fit the complex.
        """
        ...

    def cofaces_of(self, cells: Subset) raises -> Subset:
        """The (k+1)-cells that have some k-cell of `cells` as a face.

        Args:
            cells: Primal k-cells, k < n.

        Returns:
            Primal (k+1)-cells.

        Raises:
            If k = n or `cells` does not fit the complex.
        """
        ...

    def boundary_faces_of(self, cells: Subset) raises -> Subset:
        """The (k-1)-cells that are faces of exactly one k-cell of `cells`.

        Args:
            cells: Primal k-cells, k > 0.

        Returns:
            Primal (k-1)-cells.

        Raises:
            If k = 0 or `cells` does not fit the complex.
        """
        ...


struct CellSet(Copyable, Equatable, Movable, Sized, Writable):
    """Cells of a complex of every dimension: `self[k]` is a `Subset` of
    its k-cells, for k = 0 to n."""

    var levels: List[Subset]
    """`levels[k]`: the primal k-cells in the set."""

    # ===------------------------------------------------------------------=== #
    # Construction
    # ===------------------------------------------------------------------=== #

    def __init__(out self, var levels: List[Subset]) raises:
        """A cell set from one subset per dimension.

        Args:
            levels: Primal subsets of k-cells for k = 0 to n, in order.

        Raises:
            If `levels` is empty, a subset is dual, or the dimensions do not
            run `0..n`.
        """
        if len(levels) == 0:
            raise Error("CellSet: no levels")
        var n = len(levels) - 1
        for k in range(len(levels)):
            ref s = levels[k]
            if s.primality().is_dual() or s.dim != k or s.n != n:
                raise Error(
                    String(
                        "CellSet: level ", k, " is not a set of ", k, "-cells"
                    )
                )
        self.levels = levels^

    @staticmethod
    def empty[C: CellComplex](cx: C) raises -> Self:
        """The empty set of cells of `cx`.

        Parameters:
            C: The complex type.

        Args:
            cx: The complex.

        Returns:
            A set with no cells.

        Raises:
            If `cx` reports no cell counts.
        """
        var n = cx.complex_dimension()
        var levels = List[Subset](capacity=n + 1)
        for k in range(n + 1):
            levels.append(Subset(k, n, Primality.PRIMAL, cx.num_cells(k)))
        return Self(levels^)

    @staticmethod
    def full[C: CellComplex](cx: C) raises -> Self:
        """Every cell of `cx`.

        Parameters:
            C: The complex type.

        Args:
            cx: The complex.

        Returns:
            A set with every cell.

        Raises:
            If `cx` reports no cell counts.
        """
        var out = Self.empty(cx)
        for k in range(len(out.levels)):
            out.levels[k] = out.levels[k].complement()
        return out^

    @staticmethod
    def top_cells[C: CellComplex](cx: C) raises -> Self:
        """Every n-cell of `cx`, and nothing else.

        Parameters:
            C: The complex type.

        Args:
            cx: The complex.

        Returns:
            The set of top cells.

        Raises:
            If `cx` reports no cell counts.
        """
        var out = Self.empty(cx)
        var n = len(out.levels) - 1
        out.levels[n] = out.levels[n].complement()
        return out^

    @staticmethod
    def from_indices[
        C: CellComplex
    ](cx: C, dim: Int, indices: List[Int]) raises -> Self:
        """The `dim`-cells with the given indices.

        Parameters:
            C: The complex type.

        Args:
            cx: The complex.
            dim: The cell dimension, `0..n`.
            indices: Cell indices (repeats are allowed).

        Returns:
            The set of those cells.

        Raises:
            If `dim` or an index is out of range.
        """
        var out = Self.empty(cx)
        if dim < 0 or dim >= len(out.levels):
            raise Error(
                String(
                    "cells: ",
                    dim,
                    "-cells given, but the complex has dimension ",
                    len(out.levels) - 1,
                )
            )
        for i in indices:
            out._add(dim, i)
        return out^

    @staticmethod
    def from_lists[
        C: CellComplex
    ](cx: C, cells: List[List[Int]]) raises -> Self:
        """The cells listed by dimension: `cells[k]` holds k-cell indices.

        `cells` may stop short of the top dimension, so `[[v]]` is the vertex
        `v` and `[[], [e]]` the edge `e`.

        Parameters:
            C: The complex type.

        Args:
            cx: The complex.
            cells: Cell indices per dimension.

        Returns:
            The set of those cells.

        Raises:
            If `cells` has more dimensions than `cx` or an index is out of
            range.
        """
        var out = Self.empty(cx)
        if len(cells) > len(out.levels):
            raise Error(
                String(
                    "cells: ",
                    len(cells),
                    " dimensions given, but the complex has ",
                    len(out.levels),
                )
            )
        for k in range(len(cells)):
            for i in cells[k]:
                out._add(k, i)
        return out^

    @staticmethod
    def from_subset[C: CellComplex](cx: C, cells: Subset) raises -> Self:
        """A set holding the cells of one primal `Subset`.

        Parameters:
            C: The complex type.

        Args:
            cx: The complex.
            cells: Primal k-cells of `cx`.

        Returns:
            The set with `cells` at dimension k and nothing elsewhere.

        Raises:
            If `cells` is dual or does not fit `cx`.
        """
        var out = Self.empty(cx)
        out._check_level(cells)
        out.levels[cells.dim] = cells.copy()
        return out^

    @staticmethod
    def from_simplex(sc: SimplicialComplex, s: Simplex) raises -> Self:
        """The set holding one simplex of `sc`.

        Args:
            sc: The complex.
            s: A simplex of `sc`; orientation is ignored.

        Returns:
            The set with `s` alone.

        Raises:
            If `s` has the wrong dimension or is not in `sc`.
        """
        var k = len(s) - 1
        if k < 0 or k >= len(sc):
            raise Error("simplex has the wrong dimension for this complex")
        return Self.from_indices(sc, k, [sc[k].simplex_to_index(s)])

    @staticmethod
    def from_simplices(
        sc: SimplicialComplex, simplices: List[Simplex]
    ) raises -> Self:
        """The set holding the given simplices of `sc`, of any dimensions.

        Args:
            sc: The complex.
            simplices: Simplices of `sc`; orientation is ignored.

        Returns:
            The set of those simplices.

        Raises:
            If a simplex has the wrong dimension or is not in `sc`.
        """
        var out = Self.empty(sc)
        for ref s in simplices:
            var k = len(s) - 1
            if k < 0 or k >= len(sc):
                raise Error("simplex has the wrong dimension for this complex")
            out._add(k, sc[k].simplex_to_index(s))
        return out^

    # ===------------------------------------------------------------------=== #
    # Access
    # ===------------------------------------------------------------------=== #

    def __getitem__(ref self, k: Int) raises -> ref[self.levels] Subset:
        """The k-cells in the set.

        Args:
            k: A dimension in `0..n`.

        Returns:
            A reference to the subset of k-cells.

        Raises:
            If `k` is out of range.
        """
        if k < 0 or k >= len(self.levels):
            raise Error(
                String(
                    "CellSet: dimension ",
                    k,
                    " out of range 0..",
                    len(self.levels) - 1,
                )
            )
        return self.levels.unsafe_ptr()[unsafe_offset=k]

    def __len__(self) -> Int:
        """The number of cells in the set, of all dimensions.

        Returns:
            The cell count.
        """
        var c = 0
        for ref s in self.levels:
            c += len(s)
        return c

    def __contains__(self, cell: Tuple[Int, Int]) -> Bool:
        """Whether the set holds a cell, given as `(dimension, index)`.

        Args:
            cell: The cell's dimension and index.

        Returns:
            True if the cell is in the set.
        """
        var k = cell[0]
        if k < 0 or k >= len(self.levels):
            return False
        return cell[1] in self.levels[k]

    def complex_dimension(self) -> Int:
        """The dimension n of the complex the set belongs to.

        Returns:
            The dimension n, one less than the number of levels.
        """
        return len(self.levels) - 1

    def dimension(self) -> Int:
        """The highest dimension of a cell in the set.

        Returns:
            The dimension, or -1 for the empty set.
        """
        for k in range(len(self.levels) - 1, -1, -1):
            if not self.levels[k].is_empty():
                return k
        return -1

    def is_empty(self) -> Bool:
        """Whether the set has no cells.

        Returns:
            True if every level is empty.
        """
        return self.dimension() < 0

    def euler_characteristic(self) -> Int:
        """The alternating sum of the numbers of k-cells in the set.

        Returns:
            `N_0 - N_1 + N_2 - ...`.
        """
        var chi = 0
        for k in range(len(self.levels)):
            var c = len(self.levels[k])
            chi += -c if k % 2 == 1 else c
        return chi

    def indices(self) -> List[List[Int]]:
        """The cell indices of every dimension.

        Returns:
            Entry k lists the k-cells' indices, increasing.
        """
        var out = List[List[Int]](capacity=len(self.levels))
        for ref s in self.levels:
            out.append(s.indices())
        return out^

    def simplices(self, sc: SimplicialComplex) raises -> List[Simplex]:
        """The cells as simplices of `sc`, lowest dimension first.

        Args:
            sc: The complex the set belongs to.

        Returns:
            The simplices, sorted and with parity 0.

        Raises:
            If the set does not belong to `sc`.
        """
        self.check_complex(sc)
        var out = List[Simplex]()
        for k in range(len(self.levels)):
            ref level = sc[k]
            for i in self.levels[k].indices():
                out.append(level.index_to_simplex(i))
        return out^

    def __eq__(self, other: Self) -> Bool:
        """Same complex shape and same cells.

        Args:
            other: Another cell set.

        Returns:
            True if the sets are equal.
        """
        if len(self.levels) != len(other.levels):
            return False
        for k in range(len(self.levels)):
            if self.levels[k] != other.levels[k]:
                return False
        return True

    def write_to(self, mut writer: Some[Writer]):
        """Write the cell indices of each dimension.

        Args:
            writer: The destination.
        """
        writer.write("CellSet(")
        for k in range(len(self.levels)):
            if k > 0:
                writer.write(", ")
            writer.write(k, ": [")
            var first = True
            for i in self.levels[k].indices():
                if not first:
                    writer.write(", ")
                writer.write(i)
                first = False
            writer.write("]")
        writer.write(")")

    # ===------------------------------------------------------------------=== #
    # Set operations
    # ===------------------------------------------------------------------=== #

    def __or__(self, other: Self) raises -> Self:
        """The cells in either set.

        Args:
            other: A cell set of the same complex.

        Returns:
            A new set.

        Raises:
            If the sets belong to differently shaped complexes.
        """
        self._check_same(other)
        var out = self.copy()
        for k in range(len(out.levels)):
            out.levels[k] = out.levels[k] | other.levels[k]
        return out^

    def __and__(self, other: Self) raises -> Self:
        """The cells in both sets.

        Args:
            other: A cell set of the same complex.

        Returns:
            A new set.

        Raises:
            If the sets belong to differently shaped complexes.
        """
        self._check_same(other)
        var out = self.copy()
        for k in range(len(out.levels)):
            out.levels[k] = out.levels[k] & other.levels[k]
        return out^

    def __sub__(self, other: Self) raises -> Self:
        """The cells in this set but not in `other`.

        Args:
            other: A cell set of the same complex.

        Returns:
            A new set.

        Raises:
            If the sets belong to differently shaped complexes.
        """
        self._check_same(other)
        var out = self.copy()
        for k in range(len(out.levels)):
            out.levels[k] = out.levels[k] - other.levels[k]
        return out^

    # ===------------------------------------------------------------------=== #
    # Helpers
    # ===------------------------------------------------------------------=== #

    def check_complex[C: CellComplex](self, cx: C) raises:
        """Raise unless the set has the dimensions and cell counts of `cx`.

        Parameters:
            C: The complex type.

        Args:
            cx: The complex.

        Raises:
            If the set does not fit `cx`.
        """
        var n = cx.complex_dimension()
        if len(self.levels) != n + 1:
            raise Error(
                String(
                    "cells: a set for dimension ",
                    len(self.levels) - 1,
                    ", but the complex has dimension ",
                    n,
                )
            )
        for k in range(n + 1):
            if self.levels[k].size != cx.num_cells(k):
                raise Error(
                    String("cells: ", k, "-cell count differs from the complex")
                )

    def _check_level(self, s: Subset) raises:
        if (
            s.primality().is_dual()
            or s.dim < 0
            or s.dim >= len(self.levels)
            or s.n != len(self.levels) - 1
            or s.size != self.levels[s.dim].size
        ):
            raise Error("CellSet: the subset does not fit the complex")

    def _check_same(self, other: Self) raises:
        if len(self.levels) != len(other.levels):
            raise Error("CellSet: sets of different complexes")

    def _add(mut self, k: Int, i: Int) raises:
        if i < 0 or i >= self.levels[k].size:
            raise Error(String("cells: no ", k, "-cell with index ", i))
        self.levels[k].add(i)


def cells_in_rows(A: CSRMatrix, rows: Subset, dim: Int) raises -> Subset:
    """The columns of `A` with a nonzero in a row of `rows`.

    With `A` an incidence matrix whose rows are k-cells, this gives the
    cells of dimension `dim` incident to the cells of `rows`; the
    `CellComplex` implementations are built on it.

    Args:
        A: An incidence matrix with one row per cell of `rows`'s kind.
        rows: The cells whose rows are read.
        dim: The dimension of the cells the columns stand for.

    Returns:
        A primal subset of `A.ncols` cells of dimension `dim`.

    Raises:
        If `A` does not have one row per cell of `rows`'s kind.
    """
    _check_rows(A, rows)
    var out = Subset(dim, rows.n, Primality.PRIMAL, A.ncols)
    var ip = A.indptr.unsafe_ptr()
    var ix = A.indices.unsafe_ptr()
    var dv = A.data.unsafe_ptr()
    for i in rows.indices():
        for p in range(ip[unsafe_offset=i], ip[unsafe_offset=i + 1]):
            if dv[unsafe_offset=p] != 0:
                out._set(ix[unsafe_offset=p])
    return out^


def cells_once_in_rows(A: CSRMatrix, rows: Subset, dim: Int) raises -> Subset:
    """The columns of `A` with a nonzero in exactly one row of `rows`.

    Args:
        A: An incidence matrix with one row per cell of `rows`'s kind.
        rows: The cells whose rows are read.
        dim: The dimension of the cells the columns stand for.

    Returns:
        A primal subset of `A.ncols` cells of dimension `dim`.

    Raises:
        If `A` does not have one row per cell of `rows`'s kind.
    """
    _check_rows(A, rows)
    var once = Subset(dim, rows.n, Primality.PRIMAL, A.ncols)
    var twice = once.copy()
    var ip = A.indptr.unsafe_ptr()
    var ix = A.indices.unsafe_ptr()
    var dv = A.data.unsafe_ptr()
    for i in rows.indices():
        for p in range(ip[unsafe_offset=i], ip[unsafe_offset=i + 1]):
            if dv[unsafe_offset=p] != 0:
                var j = ix[unsafe_offset=p]
                if j in once:
                    twice._set(j)
                else:
                    once._set(j)
    return once - twice


def _check_rows(A: CSRMatrix, rows: Subset) raises:
    if rows.primality().is_dual() or A.nrows != rows.size:
        raise Error(
            String(
                "cells: ",
                rows.size,
                " ",
                rows.dim,
                "-cells, but the incidence matrix has ",
                A.nrows,
                " rows",
            )
        )
