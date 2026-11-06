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
    rows_of,
    cols_of,
    sort_each_row,
    take_int_rows,
    vstack_int,
)
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


struct AbstractSimplicialComplex(Copyable, Movable):
    """A simplicial complex given only by its simplices.

    ```mojo
    # two edges plus an isolated vertex
    var simplices = List[IntArray]()
    simplices.append(nm.fromstring[DType.int]("[[4]]"))
    simplices.append(nm.fromstring[DType.int]("[[0,1],[2,3]]"))
    var asc = AbstractSimplicialComplex(simplices)
    var B1 = asc.chain_complex()[1].copy()
    ```
    """

    var simplices: List[IntArray]
    """`simplices[k]`: the k-simplices. Lower dimensions hold sorted rows in
    lexicographic order; the top dimension is the array given, unchanged."""
    var _chain_complex: List[CSRMatrix]
    var _cochain_complex: List[CSRMatrix]

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
            If `simplices` is empty.
        """
        var D = 0
        for ref s in simplices:
            D = max(D, cols_of(s) - 1)
        var given = List[Optional[IntArray]](length=D + 1, fill=None)
        for ref s in simplices:
            given[cols_of(s) - 1] = s.copy()
        if not given[D]:
            raise Error("AbstractSimplicialComplex: no simplices given")

        var s = given[D].value().copy()
        var parity = simplex_array_parity(s)
        sort_each_row(s)

        var out_simplices = List[IntArray](length=D + 1, fill=int_array(0, 1))
        var chain = List[CSRMatrix](length=D + 1, fill=CSRMatrix(0, 0))
        # as in PyDEC, the top simplices keep the order and orientation given
        out_simplices[D] = given[D].value().copy()

        for d in range(D - 1, -1, -1):
            var fb = simplex_array_boundary(s, parity)
            s = fb[0].copy()
            var B = fb[1].copy()
            if given[d]:
                var old_s = s.copy()
                var user = given[d].value().copy()
                sort_each_row(user)
                var merged = vstack_int(s, user)
                merged = take_int_rows(merged, lexsort_rows(merged))
                s = _unique_rows(merged)
                var remap = simplex_array_searchsorted(s, old_s)
                var coo = B.to_coo()
                var rows = List[Int](capacity=len(coo[0]))
                for r in coo[0]:
                    rows.append(Int(remap.unsafe_get(r)))
                B = CSRMatrix.from_coo(
                    rows_of(s), B.ncols, rows, coo[1], coo[2]
                )
            parity = int_vector(rows_of(s))
            out_simplices[d] = s.copy()
            chain[d + 1] = B^

        # vertices are 0..max; renumber B1's rows from positions to vertex ids
        var num_vertices = max_entry(s) + 1 if rows_of(s) > 0 else 0
        if D >= 1:
            var coo = chain[1].to_coo()
            var rows = List[Int](capacity=len(coo[0]))
            for r in coo[0]:
                rows.append(Int(s.unsafe_get(r)))
            chain[1] = CSRMatrix.from_coo(
                num_vertices, chain[1].ncols, rows, coo[1], coo[2]
            )
        var verts = int_array(num_vertices, 1)
        for i in range(num_vertices):
            verts.unsafe_set(i, Scalar[DType.int](i))
        out_simplices[0] = verts^
        chain[0] = CSRMatrix(1, num_vertices)

        var cochain = List[CSRMatrix]()
        for k in range(1, len(chain)):
            cochain.append(chain[k].transpose())
        cochain.append(CSRMatrix(1, chain[len(chain) - 1].ncols))

        self.simplices = out_simplices^
        self._chain_complex = chain^
        self._cochain_complex = cochain^

    def complex_dimension(self) -> Int:
        """Dimension of the top simplices.

        Returns:
            The highest dimension given to the constructor.
        """
        return len(self._cochain_complex) - 1

    def complex(self) -> List[IntArray]:
        """The simplex arrays of every dimension.

        Returns:
            A copy of `simplices`.
        """
        return self.simplices.copy()

    def star(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """Every cell that has a cell of `cells` as a face, `cells` included.

        See `mojodec.routines.topology` for the definitions.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-simplices,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-simplices, sorted.

        Raises:
            If `cells` has too many dimensions or an index is out of range.
        """
        return star_of(self.chain_complex(), cells)

    def closure(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """Every face of every cell of `cells`: the subcomplex they generate.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-simplices,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-simplices, sorted.

        Raises:
            If `cells` has too many dimensions or an index is out of range.
        """
        return closure_of(self.chain_complex(), cells)

    def link(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """The closure of the star minus the star of the closure of `cells`.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-simplices,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-simplices, sorted.

        Raises:
            If `cells` has too many dimensions or an index is out of range.
        """
        return link_of(self.chain_complex(), cells)

    def boundary(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """The closure of the faces bounding exactly one top cell of `cells`.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-simplices,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-simplices, sorted.

        Raises:
            If `cells` has too many dimensions, an index is out of range, or
            `cells` is not pure (a lower cell is not a face of a top cell).
        """
        return boundary_of(self.chain_complex(), cells)

    def interior(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """The closure of `cells` minus its boundary.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-simplices,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-simplices, sorted.

        Raises:
            If `cells` has too many dimensions, an index is out of range, or
            `cells` is not pure.
        """
        return interior_of(self.chain_complex(), cells)

    def boundary_cells(self) raises -> List[List[Int]]:
        """The boundary of the whole complex: the faces bounding exactly one
        top cell, and their faces.

        Returns:
            Indices per dimension: entry k lists k-simplices, sorted.

        Raises:
            If the complex's operators are inconsistent.
        """
        return boundary_of(self.chain_complex())

    def interior_cells(self) raises -> List[List[Int]]:
        """The cells not on the boundary of the whole complex.

        Returns:
            Indices per dimension: entry k lists k-simplices, sorted.

        Raises:
            If the complex's operators are inconsistent.
        """
        return interior_of(self.chain_complex())

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

    def chain_complex(self) -> List[CSRMatrix]:
        """The boundary operators `B_k` (k-chains to (k-1)-chains).

        Returns:
            `B_0, ..., B_n`, with `B_k` the `N_{k-1} x N_k` matrix and `B_0`
            the `1 x N_0` zero matrix.
        """
        return self._chain_complex.copy()

    def cochain_complex(self) -> List[CSRMatrix]:
        """The coboundary operators `d_k = B_{k+1}^T`.

        Returns:
            `d_0, ..., d_n`, with `d_n` the `1 x N_n` zero matrix.
        """
        return self._cochain_complex.copy()


def _unique_rows(a: IntArray) raises -> IntArray:
    var keep = List[Int]()
    for i in range(rows_of(a)):
        if i == 0 or compare_rows(a, i, a, i - 1) != 0:
            keep.append(i)
    return take_int_rows(a, keep)
