# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Subsets of cells (`mojodec.core.subset`).
=========================================
A set of primal k-simplices, or of dual k-cells, of one complex, stored as a
bitset over their indices. Subsets mark where boundary conditions or material
parameters apply: operators drop the rows of a subset with
`exclude_subset` (or keep only them with `restrict_to_subset`), and
`Cochain.overwrite` copies values on a subset.

The dual k-cells correspond to the primal (n-k)-simplices, so both kinds of
subset index into `sc[k].simplices` or `sc[n-k].simplices`; `dual()` swaps
the kind without changing the bits.

```mojo
var fixed = sc.boundary_subset(0)             # vertices on the boundary
var L = sc[0].d.T() @ sc[1].star @ sc[0].d
var I = DiagonalOperator.identity(sc[0].num_simplices)
var A = L.exclude_subset(fixed) + I.restrict_to_subset(fixed).to_csr()
f.overwrite(fixed, g)                         # rows of `fixed` read u = g
var u = spsolve(A, f.v)
```
"""

from std.bit import count_trailing_zeros, pop_count

from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix


struct Subset(Copyable, Equatable, Movable, Sized, Writable):
    """A set of primal k-simplices or dual k-cells of a complex."""

    var dim: Int
    """Dimension k of the cells."""
    var n: Int
    """Dimension of the complex."""
    var is_primal: Bool
    """Primal k-simplices, or dual k-cells (primal (n-k)-simplices)."""
    var size: Int
    """Number of k-cells in the complex: indices run over `0..size-1`."""
    var _words: List[UInt64]
    """Bit `i % 64` of word `i // 64` is set when cell `i` is in the set.
    Bits past `size` are always clear."""

    # ===------------------------------------------------------------------=== #
    # Construction
    # ===------------------------------------------------------------------=== #

    def __init__(out self, dim: Int, n: Int, is_primal: Bool, size: Int):
        """The empty set of k-cells, for a complex of dimension `n` with
        `size` such cells.

        `Subset.empty(sc, ...)` reads `n` and `size` from a complex.

        Args:
            dim: The cell dimension k.
            n: The complex dimension.
            is_primal: Primal k-simplices, or dual k-cells.
            size: The number of k-cells.
        """
        self.dim = dim
        self.n = n
        self.is_primal = is_primal
        self.size = size
        self._words = List[UInt64](length=(size + 63) // 64, fill=0)

    @staticmethod
    def empty(
        sc: SimplicialComplex, dim: Int, is_primal: Bool = True
    ) raises -> Self:
        """The empty set of k-cells of `sc`.

        Args:
            sc: The complex.
            dim: The cell dimension k, `0 <= k <= n`.
            is_primal: Primal k-simplices, or dual k-cells.

        Returns:
            A subset with no cells.

        Raises:
            If `dim` is outside `0..n`.
        """
        var n = sc.complex_dimension()
        if dim < 0 or dim > n:
            raise Error(String("Subset: invalid dimension (", dim, ")"))
        var level = dim if is_primal else n - dim
        return Self(dim, n, is_primal, sc[level].num_simplices)

    @staticmethod
    def full(
        sc: SimplicialComplex, dim: Int, is_primal: Bool = True
    ) raises -> Self:
        """Every k-cell of `sc`.

        Args:
            sc: The complex.
            dim: The cell dimension k, `0 <= k <= n`.
            is_primal: Primal k-simplices, or dual k-cells.

        Returns:
            A subset with every cell.

        Raises:
            If `dim` is outside `0..n`.
        """
        return Self.empty(sc, dim, is_primal).complement()

    @staticmethod
    def from_indices(
        sc: SimplicialComplex,
        dim: Int,
        indices: List[Int],
        is_primal: Bool = True,
    ) raises -> Self:
        """The k-cells with the given indices (repeats are allowed).

        Args:
            sc: The complex.
            dim: The cell dimension k, `0 <= k <= n`.
            indices: Cell indices in `0..size-1`.
            is_primal: Primal k-simplices, or dual k-cells.

        Returns:
            The subset of those cells.

        Raises:
            If `dim` is outside `0..n` or an index is out of range.
        """
        var out = Self.empty(sc, dim, is_primal)
        for i in indices:
            out.add(i)
        return out^

    @staticmethod
    def from_mask(
        sc: SimplicialComplex,
        dim: Int,
        mask: List[Bool],
        is_primal: Bool = True,
    ) raises -> Self:
        """The k-cells `i` with `mask[i]` true; `mask` has one entry per cell.

        Args:
            sc: The complex.
            dim: The cell dimension k, `0 <= k <= n`.
            mask: One entry per k-cell.
            is_primal: Primal k-simplices, or dual k-cells.

        Returns:
            The subset of the cells marked true.

        Raises:
            If `dim` is outside `0..n` or `mask` has the wrong length.
        """
        var out = Self.empty(sc, dim, is_primal)
        if len(mask) != out.size:
            raise Error("Subset.from_mask: mask length differs from cell count")
        for i in range(len(mask)):
            if mask[i]:
                out._set(i)
        return out^

    @staticmethod
    def from_simplices(
        sc: SimplicialComplex, simplices: List[Simplex]
    ) raises -> Self:
        """The primal simplices given by their vertices.

        All must have the same dimension and be in `sc`; orientation is
        ignored. An empty list raises, as its dimension is unknown.

        Args:
            sc: The complex.
            simplices: Simplices of `sc`, all of one dimension.

        Returns:
            A primal subset of those simplices.

        Raises:
            If the list is empty, the dimensions are mixed or out of range,
            or a simplex is not in `sc`.
        """
        if len(simplices) == 0:
            raise Error("Subset.from_simplices: empty list")
        var dim = len(simplices[0]) - 1
        var out = Self.empty(sc, dim)
        ref level = sc[dim]
        for ref s in simplices:
            if len(s) - 1 != dim:
                raise Error("Subset.from_simplices: mixed dimensions")
            out._set(level.simplex_to_index(s))
        return out^

    # ===------------------------------------------------------------------=== #
    # Elements
    # ===------------------------------------------------------------------=== #

    def __len__(self) -> Int:
        """The number of cells in the set.

        Returns:
            The cell count.
        """
        var c = 0
        for w in self._words:
            c += Int(pop_count(w))
        return c

    def is_empty(self) -> Bool:
        """Whether the set has no cells.

        Returns:
            True if the set is empty.
        """
        for w in self._words:
            if w != 0:
                return False
        return True

    def __contains__(self, i: Int) -> Bool:
        """Whether cell `i` is in the set (false for `i` out of range).

        Args:
            i: A cell index.

        Returns:
            True if cell `i` is in the set.
        """
        if i < 0 or i >= self.size:
            return False
        return (self._words[i >> 6] >> UInt64(i & 63)) & 1 != 0

    def add(mut self, i: Int) raises:
        """Put cell `i` in the set.

        Args:
            i: A cell index.

        Raises:
            If `i` is outside `0..size-1`.
        """
        self._check_index(i)
        self._set(i)

    def discard(mut self, i: Int) raises:
        """Take cell `i` out of the set, if present.

        Args:
            i: A cell index.

        Raises:
            If `i` is outside `0..size-1`.
        """
        self._check_index(i)
        self._words[i >> 6] &= ~(UInt64(1) << UInt64(i & 63))

    def indices(self) -> List[Int]:
        """The indices of the cells in the set, increasing.

        Returns:
            The sorted indices.
        """
        var out = List[Int](capacity=len(self))
        for wi in range(len(self._words)):
            var w = self._words[wi]
            while w != 0:
                out.append(wi * 64 + Int(count_trailing_zeros(w)))
                w &= w - 1
        return out^

    def mask(self) -> List[Bool]:
        """One entry per k-cell of the complex: true for the cells in the set.

        Returns:
            A list of `size` flags.
        """
        var out = List[Bool](length=self.size, fill=False)
        for i in self.indices():
            out[i] = True
        return out^

    def __eq__(self, other: Self) -> Bool:
        """Same kind of cells and same members.

        Args:
            other: Another subset.

        Returns:
            True if the subsets are equal.
        """
        return self._same_kind(other) and self._words == other._words

    def write_to(self, mut writer: Some[Writer]):
        """Write the kind of cells, the count, and the indices.

        Args:
            writer: The destination.
        """
        writer.write(
            "Subset(",
            "primal " if self.is_primal else "dual ",
            self.dim,
            "-cells, ",
            len(self),
            " of ",
            self.size,
            ": ",
        )
        var first = True
        for i in self.indices():
            if not first:
                writer.write(", ")
            writer.write(i)
            first = False
        writer.write(")")

    # ===------------------------------------------------------------------=== #
    # Set operations
    # ===------------------------------------------------------------------=== #

    def union(self, other: Self) raises -> Self:
        """The cells in either set.

        Args:
            other: A subset of the same kind of cells.

        Returns:
            A new subset.

        Raises:
            If `other` is of a different kind of cells.
        """
        self._check_kind(other, "union")
        var out = self.copy()
        for i in range(len(out._words)):
            out._words[i] |= other._words[i]
        return out^

    def intersection(self, other: Self) raises -> Self:
        """The cells in both sets.

        Args:
            other: A subset of the same kind of cells.

        Returns:
            A new subset.

        Raises:
            If `other` is of a different kind of cells.
        """
        self._check_kind(other, "intersection")
        var out = self.copy()
        for i in range(len(out._words)):
            out._words[i] &= other._words[i]
        return out^

    def difference(self, other: Self) raises -> Self:
        """The cells in this set but not in `other`.

        Args:
            other: A subset of the same kind of cells.

        Returns:
            A new subset.

        Raises:
            If `other` is of a different kind of cells.
        """
        self._check_kind(other, "difference")
        var out = self.copy()
        for i in range(len(out._words)):
            out._words[i] &= ~other._words[i]
        return out^

    def complement(self) -> Self:
        """The k-cells of the complex not in this set.

        Returns:
            A new subset.
        """
        var out = self.copy()
        for i in range(len(out._words)):
            out._words[i] = ~out._words[i]
        out._clear_tail()
        return out^

    def __or__(self, other: Self) raises -> Self:
        """`self.union(other)`.

        Args:
            other: A subset of the same kind of cells.

        Returns:
            A new subset.

        Raises:
            If `other` is of a different kind of cells.
        """
        return self.union(other)

    def __and__(self, other: Self) raises -> Self:
        """`self.intersection(other)`.

        Args:
            other: A subset of the same kind of cells.

        Returns:
            A new subset.

        Raises:
            If `other` is of a different kind of cells.
        """
        return self.intersection(other)

    def __sub__(self, other: Self) raises -> Self:
        """`self.difference(other)`.

        Args:
            other: A subset of the same kind of cells.

        Returns:
            A new subset.

        Raises:
            If `other` is of a different kind of cells.
        """
        return self.difference(other)

    def __invert__(self) -> Self:
        """`self.complement()`.

        Returns:
            A new subset.
        """
        return self.complement()

    def dual(self) -> Self:
        """The same cells seen from the other side: primal k-simplices become
        dual (n-k)-cells and vice versa.

        Returns:
            A subset with the same bits and the other kind.
        """
        var out = self.copy()
        out.dim = self.n - self.dim
        out.is_primal = not self.is_primal
        return out^

    # ===------------------------------------------------------------------=== #
    # Topology
    # ===------------------------------------------------------------------=== #

    def boundary(self, sc: SimplicialComplex) raises -> Self:
        """The (k-1)-cells on the boundary of some cell of the set.

        For dual cells this is the dual of the primal coboundary, so dual
        cells cut off by the mesh boundary lack the faces lying on it, as in
        `d` on dual cochains.

        Args:
            sc: The complex the subset belongs to.

        Returns:
            A subset of (k-1)-cells of the same kind.

        Raises:
            If the subset does not belong to `sc`, or k = 0.
        """
        self._check_complex(sc)
        if self.dim == 0:
            raise Error("Subset.boundary: 0-cells have no boundary")
        return self._neighbors(sc, self.dim - 1, self._step(-1))

    def coboundary(self, sc: SimplicialComplex) raises -> Self:
        """The (k+1)-cells with some cell of the set on their boundary.

        Args:
            sc: The complex the subset belongs to.

        Returns:
            A subset of (k+1)-cells of the same kind.

        Raises:
            If the subset does not belong to `sc`, or k = n.
        """
        self._check_complex(sc)
        if self.dim == self.n:
            raise Error("Subset.coboundary: n-cells have no coboundary")
        return self._neighbors(sc, self.dim + 1, self._step(1))

    def inner_coboundary(self, sc: SimplicialComplex) raises -> Self:
        """The (k+1)-cells with every cell of their boundary in the set.

        Unlike `coboundary`, `inner_coboundary(sc).boundary(sc)` stays inside
        the set.

        Args:
            sc: The complex the subset belongs to.

        Returns:
            A subset of (k+1)-cells of the same kind.

        Raises:
            If the subset does not belong to `sc`, or k = n.
        """
        var cob = self.coboundary(sc)
        var level = self._level() + self._step(1)
        if self.is_primal:
            # sc[m].d: rows (m+1)-simplices, columns their faces
            _keep_rows_within(sc[level - 1].d, self, cob)
        else:
            # sc[m].boundary: rows (m-1)-simplices, columns their cofaces
            _keep_rows_within(sc[level + 1].boundary, self, cob)
        return cob^

    def manifold_boundary(self, sc: SimplicialComplex) raises -> Self:
        """For a set of primal n-simplices, the (n-1)-faces bounding exactly
        one of them: the boundary of the region as a manifold.

        `boundary` instead gives every face of every simplex in the set.

        Args:
            sc: The complex the subset belongs to.

        Returns:
            A primal subset of (n-1)-simplices.

        Raises:
            If the subset does not belong to `sc`, is not of primal
            n-simplices, or n = 0.
        """
        self._check_complex(sc)
        if not self.is_primal or self.dim != self.n or self.n == 0:
            raise Error(
                "Subset.manifold_boundary: needs primal n-simplices, n > 0"
            )
        var out = Self(self.n - 1, self.n, True, sc[self.n - 1].num_simplices)
        # row f of the n-boundary lists the n-simplices that have face f
        ref cob = sc[self.n].boundary
        for f in range(out.size):
            var count = 0
            for k in range(cob.indptr[f], cob.indptr[f + 1]):
                if cob.indices[k] in self:
                    count += 1
            if count == 1:
                out._set(f)
        return out^

    def simplices(self, sc: SimplicialComplex) raises -> List[Simplex]:
        """The primal simplices of the set, or for dual cells the primal
        simplices they are dual to, in index order.

        Args:
            sc: The complex the subset belongs to.

        Returns:
            The simplices, with sorted vertices and parity 0.

        Raises:
            If the subset does not belong to `sc`.
        """
        self._check_complex(sc)
        ref level = sc[self._level()]
        var out = List[Simplex]()
        for i in self.indices():
            out.append(level.index_to_simplex(i))
        return out^

    # ===------------------------------------------------------------------=== #
    # Helpers
    # ===------------------------------------------------------------------=== #

    def _level(self) -> Int:
        """The primal level the cells are indexed by."""
        return self.dim if self.is_primal else self.n - self.dim

    def _step(self, ddim: Int) -> Int:
        """The primal level change for a cell dimension change `ddim`."""
        return ddim if self.is_primal else -ddim

    def _neighbors(
        self, sc: SimplicialComplex, dim: Int, step: Int
    ) raises -> Self:
        """The cells one primal level `step` away from some cell of the set."""
        var level = self._level()
        var out = Self(
            dim, self.n, self.is_primal, sc[level + step].num_simplices
        )
        if step == 1:
            # sc[m+1].boundary: rows m-simplices, columns their cofaces
            _gather_rows(sc[level + 1].boundary, self, out)
        else:
            # sc[m-1].d: rows m-simplices, columns their faces
            _gather_rows(sc[level - 1].d, self, out)
        return out^

    def _set(mut self, i: Int):
        self._words[i >> 6] |= UInt64(1) << UInt64(i & 63)

    def _clear_tail(mut self):
        var r = self.size & 63
        if r != 0:
            self._words[len(self._words) - 1] &= (UInt64(1) << UInt64(r)) - 1

    def _check_index(self, i: Int) raises:
        if i < 0 or i >= self.size:
            raise Error(
                String("Subset: index ", i, " out of range 0..", self.size)
            )

    def _same_kind(self, other: Self) -> Bool:
        return (
            self.dim == other.dim
            and self.n == other.n
            and self.is_primal == other.is_primal
            and self.size == other.size
        )

    def _check_kind(self, other: Self, what: String) raises:
        if not self._same_kind(other):
            raise Error(String("Subset.", what, ": different kinds of cells"))

    def _check_complex(self, sc: SimplicialComplex) raises:
        if (
            sc.complex_dimension() != self.n
            or sc[self._level()].num_simplices != self.size
        ):
            raise Error("Subset: does not belong to this complex")


def _gather_rows(A: CSRMatrix, rows: Subset, mut out: Subset):
    """Add to `out` the columns of `A` in the rows of `rows`."""
    for i in rows.indices():
        for k in range(A.indptr[i], A.indptr[i + 1]):
            out._set(A.indices[k])


def _keep_rows_within(A: CSRMatrix, cols: Subset, mut rows: Subset):
    """Drop from `rows` each row of `A` with a column outside `cols`."""
    for i in rows.indices():
        for k in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[k] not in cols:
                rows._words[i >> 6] &= ~(UInt64(1) << UInt64(i & 63))
                break
