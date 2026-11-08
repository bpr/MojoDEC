# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Rips complexes (`mojodec.core.rips_complex`).
=============================================
Port of `pydec.dec.rips_complex`. The Rips complex of a point set at radius
`delta` has an edge between every two points closer than `delta`, and a
k-simplex for every (k+1)-clique of that graph, up to the dimension of the
embedding space.
"""

from mojodec.core.kd_tree import KDTree
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    RealArray,
    int_array,
    iget,
    iset,
    lexsort_rows,
    real_row_list,
    rows_of,
    cols_of,
    take_int_rows,
)
from mojodec.routines.simplex_array import simplex_array_searchsorted
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


struct RipsComplex(CellComplex, Copyable, Movable):
    """The Rips complex of a point cloud.

    ```mojo
    var pts = nm.fromstring[f64]("[[0,0],[2,0],[2,2],[0,2],[1,1]]")
    var rc = RipsComplex(pts, 2.1)
    # rc.simplices[1] == [[0,1],[0,3],[0,4],[1,2],[1,4],[2,3],[2,4],[3,4]]
    # rc.simplices[2] == [[0,1,4],[0,3,4],[1,2,4],[2,3,4]]
    ```
    """

    var vertices: RealArray
    """The points, one per row."""
    var simplices: List[IntArray]
    """`simplices[k]`: the k-simplices, sorted rows in lexicographic order."""
    var _chains: ChainComplex

    def __init__(out self, vertices: RealArray, delta: Float64) raises:
        """The Rips complex of the rows of `vertices` at radius `delta`.

        Args:
            vertices: `N x D` point coordinates; `D` caps the complex's
                dimension.
            delta: The radius: points strictly closer than this are joined
                by an edge.

        Raises:
            Never in practice; the helpers raise only on malformed simplex
            arrays, which the constructor does not produce.
        """
        var tree = KDTree(vertices)
        var ei = List[List[Int]]()
        for i in range(rows_of(vertices)):
            for j in tree.in_sphere(real_row_list(vertices, i), delta):
                ei.append([i, j])
        var edges = int_array(len(ei), 2)
        for r in range(len(ei)):
            iset(edges, r, 0, ei[r][0])
            iset(edges, r, 1, ei[r][1])
        self.simplices = rips_simplices(
            rows_of(vertices), edges, cols_of(vertices)
        )
        self._chains = rips_chain_complex(self.simplices)
        self.vertices = vertices.copy()

    def complex_dimension(self) -> Int:
        """Dimension of the highest simplices.

        Returns:
            The top dimension `D` of `simplices` (0 when there are no
            edges).
        """
        return self._chains.complex_dimension()

    def embedding_dimension(self) -> Int:
        """Dimension of the space holding the points.

        Returns:
            The number of columns of `vertices`.
        """
        return cols_of(self.vertices)

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


def rips_chain_complex(simplices: List[IntArray]) raises -> ChainComplex:
    """Boundary operators of a Rips complex from its simplex arrays.

    `simplices[k]` holds the sorted k-simplices in lexicographic order.

    Args:
        simplices: `simplices[k]`: the `N_k x (k+1)` array of k-simplices.

    Returns:
        The chain complex, with `boundary(k)` the `N_{k-1} x N_k` matrix
        `B_k`. With vertices only, it has dimension 0 and no operators.

    Raises:
        If a face of a (k+1)-simplex is missing from `simplices[k]`.
    """
    var Bs = List[CSRMatrix]()
    var nv = rows_of(simplices[0])
    if len(simplices) == 1:
        return ChainComplex(nv, Bs^)

    # B1: column e is -1 at edge[e][0] and +1 at edge[e][1]
    var ne = rows_of(simplices[1])
    var rows = List[Int](capacity=2 * ne)
    var cols = List[Int](capacity=2 * ne)
    var vals = List[Float64](capacity=2 * ne)
    for e in range(ne):
        rows.append(iget(simplices[1], e, 0))
        cols.append(e)
        vals.append(-1.0)
        rows.append(iget(simplices[1], e, 1))
        cols.append(e)
        vals.append(1.0)
    Bs.append(CSRMatrix.from_coo(rows_of(simplices[0]), ne, rows, cols, vals))

    for lvl in range(1, len(simplices) - 1):
        ref f = simplices[lvl]
        ref s = simplices[lvl + 1]
        var k = cols_of(f)
        var ns = rows_of(s)
        var faces = int_array(ns * (k + 1), k)
        for r in range(ns):
            for i in range(k + 1):
                var c = 0
                for j in range(k + 1):
                    if j != i:
                        iset(faces, r * (k + 1) + i, c, iget(s, r, j))
                        c += 1
        var indices = simplex_array_searchsorted(f, faces)
        var frows = List[Int](capacity=ns * (k + 1))
        var fcols = List[Int](capacity=ns * (k + 1))
        var fvals = List[Float64](capacity=ns * (k + 1))
        for r in range(ns):
            for i in range(k + 1):
                frows.append(Int(indices.unsafe_get(r * (k + 1) + i)))
                fcols.append(r)
                fvals.append(1.0 if i % 2 == 0 else -1.0)
        Bs.append(CSRMatrix.from_coo(rows_of(f), ns, frows, fcols, fvals))
    return ChainComplex(nv, Bs^)


def rips_simplices(
    num_vertices: Int, edges: IntArray, k: Int
) raises -> List[IntArray]:
    """The 0- to k-simplices of the clique complex of a graph.

    `edges` is an `N x 2` array; each undirected edge may appear in either
    or both orientations (and self-loops are ignored).

    ```mojo
    # edges [[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]], k = 2
    # -> [[0],[1],[2]], [[0,1],[0,2],[1,2]], [[0,1,2]]
    ```

    Args:
        num_vertices: The number of vertices; vertex ids are
            `0, ..., num_vertices - 1`.
        edges: `N x 2` vertex ids of the graph's edges.
        k: The highest dimension to generate.

    Returns:
        `simplices[j]`: the j-simplices as sorted rows in lexicographic
        order, for j up to `k`; the list stops early at the first empty
        dimension.

    Raises:
        If `edges` is not an `N x 2` array.
    """
    if edges.ndim != 2 or cols_of(edges) != 2:
        raise Error("rips_simplices: expected N by 2 array for argument edges")
    var simplices = List[IntArray]()
    var verts = int_array(num_vertices, 1)
    for i in range(num_vertices):
        iset(verts, i, 0, i)
    simplices.append(verts^)
    if k == 0:
        return simplices^

    var oriented = List[Int]()
    for r in range(rows_of(edges)):
        if iget(edges, r, 0) < iget(edges, r, 1):
            oriented.append(r)
    var E = take_int_rows(edges, oriented)
    if rows_of(E) == 0:
        return simplices^
    E = take_int_rows(E, lexsort_rows(E))
    simplices.append(E.copy())
    if k == 1:
        return simplices^

    # A[i, j] = 1 iff (i, j) is an edge with i < j
    var er = List[Int]()
    var ec = List[Int]()
    var ev = List[Float64]()
    for r in range(rows_of(E)):
        er.append(iget(E, r, 0))
        ec.append(iget(E, r, 1))
        ev.append(1.0)
    var A = CSRMatrix.from_coo(num_vertices, num_vertices, er, ec, ev)

    var k_faces = E^
    for n in range(1, k):
        if rows_of(k_faces) == 0:
            break
        # FA[f, j] == n+1 iff every vertex of face f precedes and neighbors j
        var nf = rows_of(k_faces)
        var fr = List[Int]()
        var fc = List[Int]()
        var fv = List[Float64]()
        for r in range(nf):
            for c in range(n + 1):
                fr.append(r)
                fc.append(iget(k_faces, r, c))
                fv.append(1.0)
        var FA = CSRMatrix.from_coo(nf, num_vertices, fr, fc, fv) @ A
        var new_rows = List[List[Int]]()
        for r in range(nf):
            for p in range(FA.indptr[r], FA.indptr[r + 1]):
                if FA.data[p] == Float64(n + 1):
                    var row = List[Int](capacity=n + 2)
                    for c in range(n + 1):
                        row.append(iget(k_faces, r, c))
                    row.append(FA.indices[p])
                    new_rows.append(row^)
        if len(new_rows) == 0:
            return simplices^
        var next_faces = int_array(len(new_rows), n + 2)
        for r in range(len(new_rows)):
            for c in range(n + 2):
                iset(next_faces, r, c, new_rows[r][c])
        simplices.append(next_faces.copy())
        k_faces = next_faces^
    return simplices^
