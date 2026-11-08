# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Abstract simplicial complexes (`mojodec.core.abstract_simplicial_complex`).
===========================================================================
Port of `pydec.dec.abstract_simplicial_complex`: a purely combinatorial
complex (no vertex coordinates) built from its maximal simplices of each
dimension, e.g. the graph used for least-squares ranking.
"""

from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    compare_rows,
    int_array,
    int_vector,
    lexsort_rows,
    max_entry,
    require_distinct_simplices,
    rows_of,
    cols_of,
    sort_each_row,
    take_int_rows,
    vstack_int,
)
from mojodec.core.cell_set import CellComplex, CellSet
from mojodec.core.chain_complex import ChainComplex
from mojodec.core.subset import Subset
from mojodec.routines.topology import (
    boundary_of,
    closure_of,
    interior_of,
    link_of,
    star_of,
)
from mojodec.routines.simplex_array import (
    simplex_array_boundary,
    simplex_array_parity,
    simplex_array_searchsorted,
)


struct AbstractSimplicialComplex(CellComplex, Copyable, Movable):
    """A simplicial complex given only by its simplices.

    ```mojo
    # two edges plus an isolated vertex
    var simplices = List[IntArray]()
    simplices.append(nm.fromstring[DType.int]("[[4]]"))
    simplices.append(nm.fromstring[DType.int]("[[0,1],[2,3]]"))
    var asc = AbstractSimplicialComplex(simplices)
    ref B1 = asc.chain_complex().boundary(1)
    ```
    """

    var simplices: List[IntArray]
    """`simplices[k]`: the k-simplices. Lower dimensions hold sorted rows in
    lexicographic order; the top dimension is the array given, unchanged."""
    var _chains: ChainComplex

    def __init__(out self, simplices: List[IntArray]) raises:
        """Build the complex from arrays of (maximal) simplices.

        Each array holds simplices of one dimension (its column count minus
        one); arrays may be given in any order and any dimension may be
        missing. Faces of the top simplices are generated automatically and
        merged with user-given simplices. Vertices are numbered
        `0, ..., max id`, so ids that appear in no simplex become isolated
        vertices.

        Args:
            simplices: Arrays of simplices, one dimension each: an
                `N x (k+1)` array of vertex ids holds k-simplices.

        Raises:
            If `simplices` is empty, the highest-dimensional array has no
            rows, or a top simplex repeats a vertex or has the same vertices
            as another.
        """
        var D = 0
        for ref s in simplices:
            D = max(D, cols_of(s) - 1)
        var given = List[Optional[IntArray]](length=D + 1, fill=None)
        for ref s in simplices:
            given[cols_of(s) - 1] = s.copy()
        if not given[D]:
            raise Error("AbstractSimplicialComplex: no simplices given")
        if rows_of(given[D].value()) == 0:
            raise Error("AbstractSimplicialComplex: the top array has no rows")

        var s = given[D].value().copy()
        var parity = simplex_array_parity(s)
        sort_each_row(s)
        require_distinct_simplices(s, "AbstractSimplicialComplex")

        var out_simplices = List[IntArray](length=D + 1, fill=int_array(0, 1))
        var chain = List[CSRMatrix](length=D + 1, fill=CSRMatrix(0, 0))
        # as in PyDEC, the top simplices keep the order and orientation given
        out_simplices[D] = given[D].value().copy()

        for d in range(D - 1, -1, -1):
            var fb = simplex_array_boundary(s, parity)
            s = fb.faces.copy()
            var B = fb.boundary.copy()
            if given[d]:
                var old_s = s.copy()
                var user = given[d].value().copy()
                sort_each_row(user)
                var merged = vstack_int(s, user)
                merged = take_int_rows(merged, lexsort_rows(merged))
                s = _unique_rows(merged)
                var remap = simplex_array_searchsorted(s, old_s)
                var coo = B.to_coo()
                var rows = List[Int](capacity=len(coo.rows))
                for r in coo.rows:
                    rows.append(Int(remap.unsafe_get(r)))
                B = CSRMatrix.from_coo(
                    rows_of(s), B.ncols, rows, coo.cols, coo.values
                )
            parity = int_vector(rows_of(s))
            out_simplices[d] = s.copy()
            chain[d + 1] = B^

        # vertices are 0..max; renumber B1's rows from positions to vertex ids
        var num_vertices = max_entry(s) + 1 if rows_of(s) > 0 else 0
        if D >= 1:
            var coo = chain[1].to_coo()
            var rows = List[Int](capacity=len(coo.rows))
            for r in coo.rows:
                rows.append(Int(s.unsafe_get(r)))
            chain[1] = CSRMatrix.from_coo(
                num_vertices, chain[1].ncols, rows, coo.cols, coo.values
            )
        var verts = int_array(num_vertices, 1)
        for i in range(num_vertices):
            verts.unsafe_set(i, Scalar[DType.int](i))
        out_simplices[0] = verts^

        _ = chain.pop(0)
        self.simplices = out_simplices^
        self._chains = ChainComplex(num_vertices, chain^)

    def complex_dimension(self) -> Int:
        """Dimension of the top simplices.

        Returns:
            The highest dimension given to the constructor.
        """
        return self._chains.complex_dimension()

    def chain_complex(self) -> ref[self._chains] ChainComplex:
        """The boundary operators `B_k` and coboundary operators `d_k`.

        Returns:
            The complex's `ChainComplex`, by reference.
        """
        return self._chains

    def star(self, cells: CellSet) raises -> CellSet:
        """Every cell that has a cell of `cells` as a face, `cells` included.

        See `mojodec.routines.topology` for the definitions.

        Args:
            cells: Simplices of this complex.

        Returns:
            The star.

        Raises:
            If `cells` does not belong to this complex.
        """
        return star_of(self, cells)

    def closure(self, cells: CellSet) raises -> CellSet:
        """Every face of every cell of `cells`: the subcomplex they generate.

        Args:
            cells: Simplices of this complex.

        Returns:
            The closure.

        Raises:
            If `cells` does not belong to this complex.
        """
        return closure_of(self, cells)

    def link(self, cells: CellSet) raises -> CellSet:
        """The closure of the star minus the star of the closure of `cells`.

        Args:
            cells: Simplices of this complex.

        Returns:
            The link.

        Raises:
            If `cells` does not belong to this complex.
        """
        return link_of(self, cells)

    def boundary(self, cells: CellSet) raises -> CellSet:
        """The closure of the faces bounding exactly one top cell of `cells`.

        Args:
            cells: Simplices of this complex.

        Returns:
            The boundary, a closed set.

        Raises:
            If `cells` does not belong to this complex or is not pure (a
            lower cell is not a face of a top cell).
        """
        return boundary_of(self, cells)

    def interior(self, cells: CellSet) raises -> CellSet:
        """The closure of `cells` minus its boundary.

        Args:
            cells: Simplices of this complex.

        Returns:
            The interior.

        Raises:
            If `cells` does not belong to this complex or is not pure.
        """
        return interior_of(self, cells)

    def boundary_cells(self) raises -> CellSet:
        """The boundary of the whole complex: the faces bounding exactly one
        top cell, and their faces.

        `boundary_cells()[k]` is the `Subset` of boundary k-cells.

        Returns:
            The boundary cells.

        Raises:
            If the complex's operators are inconsistent.
        """
        return boundary_of(self)

    def interior_cells(self) raises -> CellSet:
        """The cells not on the boundary of the whole complex.

        Returns:
            The interior cells.

        Raises:
            If the complex's operators are inconsistent.
        """
        return interior_of(self)

    def num_cells(self, k: Int) raises -> Int:
        """The number of k-cells.

        Args:
            k: A dimension in `0..n`.

        Returns:
            The cell count.

        Raises:
            If `k` is outside `0..n`.
        """
        return self._chains.num_cells(k)

    def faces_of(self, cells: Subset) raises -> Subset:
        """The (k-1)-cells that are faces of some k-cell of `cells`.

        Args:
            cells: Primal k-cells, k > 0.

        Returns:
            Primal (k-1)-cells.

        Raises:
            If k = 0 or `cells` does not fit the complex.
        """
        return self._chains.faces_of(cells)

    def cofaces_of(self, cells: Subset) raises -> Subset:
        """The (k+1)-cells that have some k-cell of `cells` as a face.

        Args:
            cells: Primal k-cells, k < n.

        Returns:
            Primal (k+1)-cells.

        Raises:
            If k = n or `cells` does not fit the complex.
        """
        return self._chains.cofaces_of(cells)

    def boundary_faces_of(self, cells: Subset) raises -> Subset:
        """The (k-1)-cells that are faces of exactly one k-cell of `cells`.

        Args:
            cells: Primal k-cells, k > 0.

        Returns:
            Primal (k-1)-cells.

        Raises:
            If k = 0 or `cells` does not fit the complex.
        """
        return self._chains.boundary_faces_of(cells)

    def complex(self) -> List[IntArray]:
        """The simplex arrays of every dimension.

        Returns:
            A copy of `simplices`.
        """
        return self.simplices.copy()

    def euler_characteristic(self) -> Int:
        """The alternating sum of the numbers of k-simplices.

        Returns:
            `N_0 - N_1 + N_2 - ...`.
        """
        var chi = 0
        for k in range(len(self.simplices)):
            var n = rows_of(self.simplices[k])
            chi += -n if k % 2 == 1 else n
        return chi


def _unique_rows(a: IntArray) raises -> IntArray:
    var keep = List[Int]()
    for i in range(rows_of(a)):
        if i == 0 or compare_rows(a, i, a, i - 1) != 0:
            keep.append(i)
    return take_int_rows(a, keep)
