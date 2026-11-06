# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Simplicial complexes (`mojodec.core.simplicial_complex`).
=========================================================
Port of `pydec.dec.simplicial_complex`. A `SimplicialComplex` is built from a
`SimplicialMesh`; `sc[k]` is the `SimplexLevel` holding the k-simplices with
their boundary and coboundary operators and, for complexes with geometry,
their circumcenters, primal and dual volumes, and diagonal Hodge stars.

PyDEC computes geometric data lazily on first attribute access. Mojo has no
`__getattr__` hook, so MojoDEC computes it when the complex is built (pass
`compute_geometry=False` to skip it, e.g. for degenerate meshes).

The complex's data lives behind a shared `ArcPointer`: copying a
`SimplicialComplex` is cheap and every copy, like every `Cochain` built from
it, refers to the same data, mirroring Python's reference semantics.
"""

from std.math import sqrt
from std.memory import ArcPointer

import numojo as nm
from numojo.prelude import *

from mojodec.core.cell_set import (
    CellComplex,
    CellSet,
    cells_in_rows,
    cells_once_in_rows,
)
from mojodec.core.chain_complex import ChainComplex
from mojodec.core.cochain import Cochain, CochainBasis
from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.primality import Primality
from mojodec.core.mesh import SimplicialMesh
from mojodec.core.simplex import Simplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset
from mojodec.core.utils import (
    IntArray,
    RealArray,
    compare_rows,
    factorial,
    int_vector,
    lexsort_rows,
    real_array,
    real_vector,
    row_list,
    rows_of,
    cols_of,
    sort_each_row,
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
)


struct SimplexLevel(Copyable, Movable):
    """The k-simplices of a complex and the operators attached to them."""

    var dim: Int
    """The dimension k."""
    var simplices: IntArray
    """`num_simplices x (k+1)` sorted vertex lists, in lexicographic order
    (for the top level, in the order of the mesh's elements)."""
    var num_simplices: Int
    """Number of k-simplices."""
    var simplex_parity: IntArray
    """Orientation of each simplex relative to its sorted vertex list."""
    var d: CSRMatrix
    """Exterior derivative from k-cochains to (k+1)-cochains."""
    var boundary: CSRMatrix
    """Boundary operator from k-chains to (k-1)-chains."""
    var _order: List[Int]
    """Permutation putting `simplices` in lexicographic order, or empty when
    they already are (every level but possibly the top one)."""
    var circumcenter: RealArray
    """Circumcenters, `num_simplices x embedding_dimension`."""
    var bary_circumcenter: RealArray
    """Circumcenters in barycentric coordinates, `num_simplices x (k+1)`."""
    var primal_volume: RealArray
    """Volume of each simplex (signed for full-dimensional top simplices)."""
    var dual_volume: RealArray
    """Volume of each simplex's circumcentric dual cell."""
    var star: DiagonalOperator
    """Diagonal Hodge star from primal k-cochains to dual (n-k)-cochains."""
    var star_inv: DiagonalOperator
    """Inverse Hodge star, signed so `star_inv @ star = (-1)^(k(n-k))`."""

    def __init__(
        out self,
        dim: Int,
        var simplices: IntArray,
        var simplex_parity: IntArray,
        var d: CSRMatrix,
        var boundary: CSRMatrix,
    ) raises:
        """A level without geometry; `compute_geometry` fills that in.

        Args:
            dim: The dimension k.
            simplices: `N x (k+1)` vertex lists, each row sorted.
            simplex_parity: The orientation of each row relative to its
                sorted order (0 or 1).
            d: The exterior derivative on k-cochains, `N_{k+1} x N`.
            boundary: The boundary operator on k-chains, `N_{k-1} x N`.

        Raises:
            If the arrays cannot be read.
        """
        self.dim = dim
        self.num_simplices = rows_of(simplices)
        self._order = List[Int]()
        for i in range(1, self.num_simplices):
            if compare_rows(simplices, i - 1, simplices, i) > 0:
                self._order = lexsort_rows(simplices)
                break
        self.simplices = simplices^
        self.simplex_parity = simplex_parity^
        self.d = d^
        self.boundary = boundary^
        self.circumcenter = real_vector(0)
        self.bary_circumcenter = real_vector(0)
        self.primal_volume = real_vector(0)
        self.dual_volume = real_vector(0)
        self.star = DiagonalOperator(List[Float64]())
        self.star_inv = DiagonalOperator(List[Float64]())

    def find_simplex(self, s: Simplex) -> Int:
        """Index of `s` in `simplices`, or -1 if it is not a simplex here.

        A binary search over the sorted vertex lists; orientation is ignored.

        Args:
            s: The simplex to look up.

        Returns:
            Its row in `simplices`, or -1.
        """
        if len(s) != self.dim + 1:
            return -1
        var w = self.dim + 1
        var lo = 0
        var hi = self.num_simplices
        while lo < hi:
            var mid = (lo + hi) // 2
            var row = self._order[mid] if len(self._order) > 0 else mid
            var c = 0
            for j in range(w):
                var x = Int(self.simplices.unsafe_get(row * w + j))
                if x != s.vertices[j]:
                    c = -1 if x < s.vertices[j] else 1
                    break
            if c == 0:
                return row
            if c < 0:
                lo = mid + 1
            else:
                hi = mid
        return -1

    def __contains__(self, s: Simplex) -> Bool:
        """Whether `s` is one of this level's simplices, in either orientation.

        Args:
            s: The simplex to look up.

        Returns:
            True if `s` is a k-simplex of the complex.
        """
        return self.find_simplex(s) >= 0

    def simplex_to_index(self, s: Simplex) raises -> Int:
        """Index of `s` in `simplices`, ignoring its orientation.

        Args:
            s: The simplex to look up.

        Returns:
            Its row in `simplices`.

        Raises:
            If `s` is not a k-simplex of the complex.
        """
        var i = self.find_simplex(s)
        if i < 0:
            raise Error(String("simplex ", s, " is not in the complex"))
        return i

    def index_to_simplex(self, i: Int) raises -> Simplex:
        """The simplex at index `i`, sorted and with parity 0.

        Args:
            i: A row of `simplices`.

        Returns:
            The simplex with that row's vertices.

        Raises:
            If `i` is out of range.
        """
        if i < 0 or i >= self.num_simplices:
            raise Error(String("no ", self.dim, "-simplex ", i))
        return Simplex(row_list(self.simplices, i))


struct ComplexData(Movable):
    """The shared state of a `SimplicialComplex`."""

    var mesh: SimplicialMesh
    """The mesh the complex was built from."""
    var levels: List[SimplexLevel]
    """One level per dimension, 0 to n."""
    var has_geometry: Bool
    """Whether `compute_geometry` has run."""

    def __init__(out self, var mesh: SimplicialMesh):
        """State for a complex of `mesh`, with no levels yet.

        Args:
            mesh: The mesh.
        """
        self.mesh = mesh^
        self.levels = List[SimplexLevel]()
        self.has_geometry = False


struct SimplicialComplex(
    CellComplex, ImplicitlyCopyable, Movable, Sized, Writable
):
    """A simplicial complex with its chain and cochain operators.

    ```mojo
    var V = nm.fromstring[f64]("[[0,0],[1,0],[1,1],[0,1]]")
    var S = nm.fromstring[DType.int]("[[0,1,3],[1,2,3]]")
    var sc = SimplicialComplex(V, S)
    print(sc[1].num_simplices)   # 5 edges
    var star1 = sc[1].star.copy()
    ```
    """

    var _data: ArcPointer[ComplexData]

    def __init__(
        out self, mesh: SimplicialMesh, compute_geometry: Bool = True
    ) raises:
        """The complex of a mesh: every face of its elements, with the
        boundary and coboundary operators.

        Args:
            mesh: The vertices and top simplices.
            compute_geometry: Also compute circumcenters, volumes, and Hodge
                stars; pass False for meshes with degenerate simplices.

        Raises:
            If the elements are malformed, or (with geometry) a simplex is
            degenerate.
        """
        self._data = ArcPointer(ComplexData(mesh.copy()))
        self._build_complex()
        if compute_geometry:
            self.compute_geometry()

    def __init__(
        out self,
        vertices: RealArray,
        simplices: IntArray,
        compute_geometry: Bool = True,
    ) raises:
        """The complex of the mesh `(vertices, simplices)`.

        Args:
            vertices: `num_vertices x embedding_dimension` coordinates.
            simplices: `num_elements x (n+1)` vertex indices of the top
                simplices; their order gives the orientation.
            compute_geometry: Also compute circumcenters, volumes, and Hodge
                stars; pass False for meshes with degenerate simplices.

        Raises:
            If the arrays do not form a mesh, or (with geometry) a simplex is
            degenerate.
        """
        self = Self(SimplicialMesh(vertices, simplices), compute_geometry)

    # ===------------------------------------------------------------------=== #
    # Access
    # ===------------------------------------------------------------------=== #

    def __getitem__(
        self, k: Int
    ) raises -> ref[self._data[].levels] SimplexLevel:
        """The level of k-simplices, by reference.

        Args:
            k: The dimension, `0 <= k <= complex_dimension()`.

        Returns:
            The `SimplexLevel` holding the k-simplices and their operators.

        Raises:
            If `k` is out of range.
        """
        if k < 0 or k >= len(self._data[].levels):
            raise Error("SimplicialComplex: level index out of range")
        return self._data[].levels.unsafe_ptr()[unsafe_offset=k]

    def __len__(self) -> Int:
        """Number of levels.

        Returns:
            The complex dimension plus one.
        """
        return len(self._data[].levels)

    def mesh(self) -> ref[self._data[].mesh] SimplicialMesh:
        """The mesh the complex was built from.

        Returns:
            A reference to the shared mesh.
        """
        return self._data[].mesh

    def vertices(self) -> ref[self._data[].mesh.vertices] RealArray:
        """Vertex coordinates.

        Returns:
            The `num_vertices x embedding_dimension` array, by reference.
        """
        return self._data[].mesh.vertices

    def simplices(self) -> ref[self._data[].mesh.elements] IntArray:
        """Top simplices in the mesh's original vertex order and orientation.

        Returns:
            The `num_elements x (n+1)` array, by reference.
        """
        return self._data[].mesh.elements

    def complex_dimension(self) -> Int:
        """Dimension n of the top simplices.

        Returns:
            The number of vertices per element minus one.
        """
        return cols_of(self._data[].mesh.elements) - 1

    def embedding_dimension(self) -> Int:
        """Dimension of the space the vertices live in.

        Returns:
            The number of coordinates per vertex.
        """
        return cols_of(self._data[].mesh.vertices)

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

    def star(self, s: Simplex) raises -> CellSet:
        """The star of one simplex: every simplex that has it as a face.

        Args:
            s: A simplex of the complex, in either orientation.

        Returns:
            The cells, by dimension.

        Raises:
            If `s` is not in the complex.
        """
        return star_of(self, CellSet.from_simplex(self, s))

    def closure(self, s: Simplex) raises -> CellSet:
        """The faces of one simplex, itself included.

        Args:
            s: A simplex of the complex, in either orientation.

        Returns:
            The cells, by dimension.

        Raises:
            If `s` is not in the complex.
        """
        return closure_of(self, CellSet.from_simplex(self, s))

    def link(self, s: Simplex) raises -> CellSet:
        """The link of one simplex: the faces of its star that do not meet it.

        ```mojo
        var ring = sc.link(Simplex([v]))   # the ring around vertex v
        print(ring.simplices(sc))
        ```

        Args:
            s: A simplex of the complex, in either orientation.

        Returns:
            The cells, by dimension.

        Raises:
            If `s` is not in the complex.
        """
        return link_of(self, CellSet.from_simplex(self, s))

    def boundary(self, s: Simplex) raises -> CellSet:
        """The proper faces of one simplex.

        Args:
            s: A simplex of the complex, in either orientation.

        Returns:
            The cells, by dimension.

        Raises:
            If `s` is not in the complex.
        """
        return boundary_of(self, CellSet.from_simplex(self, s))

    def interior(self, s: Simplex) raises -> CellSet:
        """One simplex alone: its relative interior.

        Args:
            s: A simplex of the complex, in either orientation.

        Returns:
            The cells, by dimension.

        Raises:
            If `s` is not in the complex.
        """
        return interior_of(self, CellSet.from_simplex(self, s))

    def num_cells(self, k: Int) raises -> Int:
        """The number of k-cells.

        Args:
            k: A dimension in `0..n`.

        Returns:
            The cell count.

        Raises:
            If `k` is outside `0..n`.
        """
        return self[k].num_simplices

    def faces_of(self, cells: Subset) raises -> Subset:
        """The (k-1)-cells that are faces of some k-cell of `cells`.

        Args:
            cells: Primal k-cells, k > 0.

        Returns:
            Primal (k-1)-cells.

        Raises:
            If k = 0 or `cells` does not fit the complex.
        """
        var k = cells.dim
        if k < 1:
            raise Error("faces_of: 0-cells have no faces")
        return cells_in_rows(self[k - 1].d, cells, k - 1)

    def cofaces_of(self, cells: Subset) raises -> Subset:
        """The (k+1)-cells that have some k-cell of `cells` as a face.

        Args:
            cells: Primal k-cells, k < n.

        Returns:
            Primal (k+1)-cells.

        Raises:
            If k = n or `cells` does not fit the complex.
        """
        var k = cells.dim
        if k >= self.complex_dimension():
            raise Error("cofaces_of: n-cells have no cofaces")
        return cells_in_rows(self[k + 1].boundary, cells, k + 1)

    def boundary_faces_of(self, cells: Subset) raises -> Subset:
        """The (k-1)-cells that are faces of exactly one k-cell of `cells`.

        Args:
            cells: Primal k-cells, k > 0.

        Returns:
            Primal (k-1)-cells.

        Raises:
            If k = 0 or `cells` does not fit the complex.
        """
        var k = cells.dim
        if k < 1:
            raise Error("boundary_faces_of: 0-cells have no faces")
        return cells_once_in_rows(self[k - 1].d, cells, k - 1)

    def euler_characteristic(self) -> Int:
        """The alternating sum of the numbers of k-simplices.

        Returns:
            `N_0 - N_1 + N_2 - ...`.
        """
        var chi = 0
        for k in range(len(self)):
            var n = self._data[].levels[k].num_simplices
            chi += -n if k % 2 == 1 else n
        return chi

    def has_geometry(self) -> Bool:
        """Whether circumcenters, volumes, and Hodge stars were computed.

        Returns:
            False for complexes built with `compute_geometry=False`.
        """
        return self._data[].has_geometry

    def is_same(self, other: Self) -> Bool:
        """Whether both values refer to the same complex.

        Copies share their data, so this compares identity, not contents.

        Args:
            other: Another complex.

        Returns:
            True if both share one `ComplexData`.
        """
        return self._data.ptr() == other._data.ptr()

    def chain_complex(self) raises -> ChainComplex:
        """The boundary and coboundary operators of every level, copied.

        `self[k].boundary` and `self[k].d` give them without copying.

        Returns:
            A `ChainComplex` with `boundary(k)` = `self[k].boundary` and
            `coboundary(k)` = `self[k].d`.

        Raises:
            Never in practice; the levels' operators chain.
        """
        var n = self.complex_dimension()
        var bs = List[CSRMatrix](capacity=n)
        var ds = List[CSRMatrix](capacity=n)
        for k in range(n):
            bs.append(self[k + 1].boundary.copy())
            ds.append(self[k].d.copy())
        return ChainComplex(self[0].num_simplices, bs^, ds^)

    def complex(self) -> List[IntArray]:
        """The simplex arrays of every level.

        Returns:
            Copies of `self[k].simplices` for k = 0 to n.
        """
        var out = List[IntArray]()
        for ref lvl in self._data[].levels:
            out.append(lvl.simplices.copy())
        return out^

    def get_cochain(
        self, dimension: Int, primality: Primality = .PRIMAL
    ) raises -> Cochain:
        """A zero cochain of the given dimension.

        Args:
            dimension: The degree k.
            primality: `PRIMAL` for a k-cochain on k-simplices, `DUAL` for
                one on dual k-cells (the duals of the (n-k)-simplices).

        Returns:
            The cochain, with one zero per cell.

        Raises:
            If `dimension` is not in `0..n`.
        """
        var N = self.complex_dimension()
        if dimension < 0 or dimension > N:
            raise Error(String("invalid dimension (", dimension, ")"))
        var level = primality.level(N, dimension)
        return Cochain(
            self, dimension, primality, real_vector(self[level].num_simplices)
        )

    def get_cochain_basis(
        self, dimension: Int, primality: Primality = .PRIMAL
    ) raises -> CochainBasis:
        """The basis of all cochains of a dimension: values form the identity.

        Applying operators to a basis composes their matrices.

        Args:
            dimension: The degree k.
            primality: Primal or dual k-cochains.

        Returns:
            The basis, whose values are a sparse identity matrix.

        Raises:
            If `dimension` is not in `0..n`.
        """
        var N = self.complex_dimension()
        if dimension < 0 or dimension > N:
            raise Error(String("invalid dimension (", dimension, ")"))
        var level = primality.level(N, dimension)
        return CochainBasis(
            self,
            dimension,
            primality,
            CSRMatrix.identity(self[level].num_simplices),
        )

    def boundary(self) raises -> List[Simplex]:
        """The (n-1)-faces of the complex that bound exactly one n-simplex.

        As in PyDEC, the faces are returned in their canonical (sorted, even
        parity) form, in the order of `self[n-1].simplices`. `boundary_cells`
        gives the boundary subcomplex, faces of faces included, as a
        `CellSet`.

        Returns:
            The boundary faces.

        Raises:
            If the complex has dimension 0.
        """
        var n = self.complex_dimension()
        if n <= 0:
            raise Error(
                "SimplicialComplex.boundary: complex dimension must be positive"
            )
        ref faces = self[n - 1]
        # row f of the n-boundary lists the n-simplices that have face f
        ref cob = self[n].boundary
        var out = List[Simplex]()
        for i in range(faces.num_simplices):
            if cob.indptr[i + 1] - cob.indptr[i] == 1:
                out.append(faces.index_to_simplex(i))
        return out^

    def write_to(self, mut writer: Some[Writer]):
        """Write the number of simplices of each dimension, as PyDEC prints
        a complex.

        Args:
            writer: The destination.
        """
        writer.write("simplicial_complex:\n  complex:\n")
        for i in range(len(self) - 1, -1, -1):
            writer.write(
                "   ",
                self._data[].levels[i].num_simplices,
                ": ",
                i,
                "-simplices\n",
            )

    # ===------------------------------------------------------------------=== #
    # Geometry
    # ===------------------------------------------------------------------=== #

    def compute_geometry(self) raises:
        """Compute circumcenters, volumes, and Hodge stars for every level.

        The constructors call this unless given `compute_geometry=False`.

        Raises:
            If a simplex is degenerate.
        """
        for dim in range(len(self)):
            self.compute_circumcenters(dim)
            self.compute_primal_volume(dim)
        self.compute_dual_volume()
        self.construct_hodge()
        self._data[].has_geometry = True

    def compute_circumcenters(self, dim: Int) raises:
        """Circumcenters of the simplices of one dimension, in Cartesian
        (`circumcenter`) and barycentric (`bary_circumcenter`) coordinates.

        Solves PyDEC's system `[[2 P P^T, 1], [1^T, 0]] [b; mu] = [|p_i|^2; 1]`
        once per simplex, in place, with buffers reused across simplices.

        Args:
            dim: The dimension of the simplices.

        Raises:
            If `dim` is out of range or a simplex is degenerate.
        """
        ref data = self[dim]
        ref V = self.vertices()
        var E = self.embedding_dimension()
        var w = dim + 1
        var m = w + 1
        var out = real_array(data.num_simplices, E)
        var bary = real_array(data.num_simplices, w)
        var A = List[Float64](length=m * m, fill=0)
        var b = List[Float64](length=m, fill=0)
        var verts = List[Int](length=w, fill=0)
        for i in range(data.num_simplices):
            for r in range(w):
                verts.unsafe_set(r, Int(data.simplices.unsafe_get(i * w + r)))
            for r in range(w):
                for c in range(w):
                    var s: Float64 = 0
                    for k in range(E):
                        s += V.unsafe_get(
                            verts.unsafe_get(r) * E + k
                        ) * V.unsafe_get(verts.unsafe_get(c) * E + k)
                    A.unsafe_set(r * m + c, 2 * s)
                    if r == c:
                        b.unsafe_set(r, s)
                A.unsafe_set(r * m + w, 1.0)
                A.unsafe_set(w * m + r, 1.0)
            A.unsafe_set(w * m + w, 0.0)
            b.unsafe_set(w, 1.0)
            _solve_in_place(A, b, m)
            for r in range(w):
                bary.unsafe_set(i * w + r, b.unsafe_get(r))
            for k in range(E):
                var s: Float64 = 0
                for r in range(w):
                    s += b.unsafe_get(r) * V.unsafe_get(
                        verts.unsafe_get(r) * E + k
                    )
                out.unsafe_set(i * E + k, s)
        data.circumcenter = out^
        data.bary_circumcenter = bary^

    def compute_primal_volume(self, dim: Int) raises:
        """Volumes of the simplices of one dimension.

        Top simplices of a full-dimensional complex get signed volumes, in
        the orientation of the mesh's elements.

        Args:
            dim: The dimension of the simplices.

        Raises:
            If `dim` is out of range.
        """
        ref data = self[dim]
        ref V = self.vertices()
        var E = self.embedding_dimension()
        var out = real_vector(data.num_simplices)
        if dim == 0:
            for i in range(data.num_simplices):
                out.unsafe_set(i, 1.0)
            data.primal_volume = out^
            return
        # signed: in the mesh's own vertex order, which carries orientation
        if dim == E:
            _simplex_volumes(V, self.simplices(), dim, True, out)
        else:
            _simplex_volumes(V, data.simplices, dim, False, out)
        data.primal_volume = out^

    def compute_dual_volume(self) raises:
        """Circumcentric dual volumes of the simplices of every dimension.

        The dual cell of a simplex is assembled from the simplices spanned by
        chains of circumcenters; each piece is signed by whether the
        circumcenter of a parent lies on the same side of a face as the
        opposite vertex (PyDEC's well-centeredness-free construction).

        Faces are reached through the rows of the `d` matrices, and each
        piece's volume is updated incrementally: the chain's edges, measured
        from the top simplex's circumcenter, are orthonormalized one at a
        time, so a piece's volume is the previous one times the new edge's
        distance from the span of the others, over the new dimension.

        Needs the circumcenters of every level.

        Raises:
            If the circumcenters have not been computed.
        """
        var n = self.complex_dimension()
        var E = self.embedding_dimension()
        for dim in range(len(self)):
            self[dim].dual_volume = real_vector(self[dim].num_simplices)
        ref top = self[n]
        # q[j*E : (j+1)*E] is the j-th orthonormalized edge of the chain
        var q = List[Float64](length=max(n, 1) * E, fill=0)
        for t in range(top.num_simplices):
            # the dual of a top simplex is its circumcenter: unit volume
            top.dual_volume.unsafe_set(t, top.dual_volume.unsafe_get(t) + 1)
            if n > 0:
                self._add_dual_pieces(n, t, t, 1.0, 1.0, q)

    def construct_hodge(self) raises:
        """The diagonal Hodge stars `dual_volume / primal_volume`, and their
        inverses signed by `(-1)^(k(n-k))`.

        Raises:
            If the volumes have not been computed.
        """
        var n = self.complex_dimension()
        for dim in range(len(self)):
            ref data = self[dim]
            var N = data.num_simplices
            var star = List[Float64](capacity=N)
            var star_inv = List[Float64](capacity=N)
            var sign: Float64 = -1.0 if (dim * (n - dim)) % 2 == 1 else 1.0
            for i in range(N):
                var r = data.dual_volume.unsafe_get(
                    i
                ) / data.primal_volume.unsafe_get(i)
                star.append(r)
                star_inv.append(sign / r)
            data.star = DiagonalOperator(star^)
            data.star_inv = DiagonalOperator(star_inv^)

    # ===------------------------------------------------------------------=== #
    # Private helpers
    # ===------------------------------------------------------------------=== #

    def _build_complex(self) raises:
        """Faces and boundary operators of every dimension."""
        ref mesh = self._data[].mesh
        var K = cols_of(mesh.elements)
        var s = mesh.elements.copy()
        var parity = simplex_array_parity(s)
        sort_each_row(s)

        var simplices = List[IntArray]()
        var parities = List[IntArray]()
        var chain = List[CSRMatrix]()
        simplices.append(s.copy())
        parities.append(parity.copy())
        while cols_of(s) > 1:
            var fb = simplex_array_boundary(s, parity)
            s = fb.faces.copy()
            parity = int_vector(rows_of(s))
            simplices.append(s.copy())
            chain.append(fb.boundary.copy())
            parities.append(parity.copy())
        chain.append(CSRMatrix(1, rows_of(s)))

        simplices.reverse()
        chain.reverse()
        parities.reverse()

        # d_k = B_{k+1}^T, and the top-level d is a 1 x N zero matrix
        var cochain = List[CSRMatrix]()
        for k in range(1, len(chain)):
            cochain.append(chain[k].transpose())
        cochain.append(CSRMatrix(1, chain[len(chain) - 1].ncols))

        for k in range(K):
            self._data[].levels.append(
                SimplexLevel(
                    k,
                    simplices[k].copy(),
                    parities[k].copy(),
                    cochain[k].copy(),
                    chain[k].copy(),
                )
            )

    def _add_dual_pieces(
        self,
        dim: Int,
        i: Int,
        root: Int,
        sign: Float64,
        vol: Float64,
        mut q: List[Float64],
    ) raises:
        """Add the pieces of a chain of circumcenters to the dual volumes.

        The chain runs from top simplex `root` down to `dim`-simplex `i`;
        `sign` and `vol` (the chain's volume times its factorial) belong to
        that prefix, and `q` holds its orthonormalized edges.
        """
        var n = self.complex_dimension()
        var E = self.embedding_dimension()
        var m = n - dim
        ref parent = self[dim]
        ref faces = self[dim - 1]
        ref C = self[n].circumcenter
        # row i of d_{dim-1} lists the faces of the dim-simplex i
        ref d = faces.d
        for p in range(d.indptr.unsafe_get(i), d.indptr.unsafe_get(i + 1)):
            var f = d.indices.unsafe_get(p)
            # sign from the parent's barycentric coordinate at the vertex
            # opposite the face
            var ov = 0
            for j in range(dim + 1):
                var v = parent.simplices.unsafe_get(i * (dim + 1) + j)
                var in_face = False
                for jj in range(dim):
                    if faces.simplices.unsafe_get(f * dim + jj) == v:
                        in_face = True
                        break
                if not in_face:
                    ov = j
                    break
            var b = parent.bary_circumcenter.unsafe_get(i * (dim + 1) + ov)
            var s = sign * (
                -1.0 if b < 0 or (b == 0 and _is_negative_zero(b)) else 1.0
            )
            # new edge from the root circumcenter, orthogonalized against q
            for k in range(E):
                q.unsafe_set(
                    m * E + k,
                    faces.circumcenter.unsafe_get(f * E + k)
                    - C.unsafe_get(root * E + k),
                )
            for j in range(m):
                var proj: Float64 = 0
                for k in range(E):
                    proj += q.unsafe_get(j * E + k) * q.unsafe_get(m * E + k)
                for k in range(E):
                    q.unsafe_set(
                        m * E + k,
                        q.unsafe_get(m * E + k)
                        - (proj * q.unsafe_get(j * E + k)),
                    )
            var h2: Float64 = 0
            for k in range(E):
                h2 += q.unsafe_get(m * E + k) * q.unsafe_get(m * E + k)
            var h = sqrt(h2)
            if h > 0:
                for k in range(E):
                    q.unsafe_set(m * E + k, q.unsafe_get(m * E + k) / (h))
            var new_vol = vol * h
            faces.dual_volume.unsafe_set(
                f,
                faces.dual_volume.unsafe_get(f)
                + s * new_vol / factorial(m + 1),
            )
            if dim > 1:
                self._add_dual_pieces(dim - 1, f, root, s, new_vol, q)


def _is_negative_zero(x: Float64) -> Bool:
    return x == 0 and (1.0 / x) < 0


def _solve_in_place(mut A: List[Float64], mut b: List[Float64], n: Int) raises:
    """Solve the `n x n` system `A x = b` by Gaussian elimination with
    partial pivoting, overwriting `b` with `x` (and `A` with scratch).

    Raises if `A` is singular to working precision, like `dense.solve`.
    """
    for k in range(n):
        var p = k
        var best = abs(A.unsafe_get(k * n + k))
        for r in range(k + 1, n):
            var v = abs(A.unsafe_get(r * n + k))
            if v > best:
                best = v
                p = r
        if best == 0:
            raise Error("circumcenter: degenerate simplex (singular system)")
        if p != k:
            for c in range(n):
                var t = A.unsafe_get(k * n + c)
                A.unsafe_set(k * n + c, A.unsafe_get(p * n + c))
                A.unsafe_set(p * n + c, t)
            var t = b.unsafe_get(k)
            b.unsafe_set(k, b.unsafe_get(p))
            b.unsafe_set(p, t)
        var pivot = A.unsafe_get(k * n + k)
        for r in range(k + 1, n):
            var f = A.unsafe_get(r * n + k) / pivot
            if f != 0:
                for c in range(k + 1, n):
                    A.unsafe_set(
                        r * n + c,
                        A.unsafe_get(r * n + c) - (f * A.unsafe_get(k * n + c)),
                    )
                b.unsafe_set(r, b.unsafe_get(r) - (f * b.unsafe_get(k)))
    for k in range(n - 1, -1, -1):
        var s = b.unsafe_get(k)
        for c in range(k + 1, n):
            s -= A.unsafe_get(k * n + c) * b.unsafe_get(c)
        b.unsafe_set(k, s / A.unsafe_get(k * n + k))


def _det_in_place(mut A: List[Float64], n: Int) -> Float64:
    """Determinant of the `n x n` matrix `A`, overwriting `A` with scratch.

    The same arithmetic as `dense.det` (0 if singular), without allocating.
    """
    if n == 1:
        return A.unsafe_get(0)
    if n == 2:
        return A.unsafe_get(0) * A.unsafe_get(3) - A.unsafe_get(
            1
        ) * A.unsafe_get(2)
    var sign: Float64 = 1
    for k in range(n):
        var p = k
        var best = abs(A.unsafe_get(k * n + k))
        for r in range(k + 1, n):
            var v = abs(A.unsafe_get(r * n + k))
            if v > best:
                best = v
                p = r
        if best == 0:
            return 0.0
        if p != k:
            for c in range(n):
                var t = A.unsafe_get(k * n + c)
                A.unsafe_set(k * n + c, A.unsafe_get(p * n + c))
                A.unsafe_set(p * n + c, t)
            sign = -sign
        var pivot = A.unsafe_get(k * n + k)
        for r in range(k + 1, n):
            var f = A.unsafe_get(r * n + k) / pivot
            if f != 0:
                for c in range(k + 1, n):
                    A.unsafe_set(
                        r * n + c,
                        A.unsafe_get(r * n + c) - (f * A.unsafe_get(k * n + c)),
                    )
    var d = sign
    for k in range(n):
        d *= A.unsafe_get(k * n + k)
    return d


def _simplex_volumes(
    V: RealArray, S: IntArray, dim: Int, signed: Bool, mut out: RealArray
):
    """Volumes of the `dim`-simplices in the rows of `S` into `out`.

    Signed: the determinant of the edge vectors over `dim!` (needs
    `dim` equal to the embedding dimension). Unsigned: the square root of
    the determinant of their Gram matrix over `dim!`. Buffers are reused
    across simplices.
    """
    var E = cols_of(V)
    var w = dim + 1
    var edges = List[Float64](length=dim * E, fill=0)
    var M = List[Float64](length=dim * dim, fill=0)
    var scale = factorial(dim)
    for i in range(rows_of(S)):
        var v0 = Int(S.unsafe_get(i * w))
        for r in range(dim):
            var v = Int(S.unsafe_get(i * w + r + 1))
            for k in range(E):
                edges[r * E + k] = V.unsafe_get(v * E + k) - V.unsafe_get(
                    v0 * E + k
                )
        if signed:
            for r in range(dim * dim):
                M[r] = edges[r]
            out.unsafe_set(i, _det_in_place(M, dim) / scale)
        else:
            for r in range(dim):
                for c in range(dim):
                    var g: Float64 = 0
                    for k in range(E):
                        g += edges[r * E + k] * edges[c * E + k]
                    M[r * dim + c] = g
            out.unsafe_set(i, sqrt(abs(_det_in_place(M, dim))) / scale)
