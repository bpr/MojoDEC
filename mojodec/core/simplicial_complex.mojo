# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Simplicial complexes (`mojodec.core.simplicial_complex`).
=========================================================
Largely inspired by `pydec.dec.simplicial_complex`. A `SimplicialComplex` is
built from a `SimplicialMesh`; `sc[k]` is the `SimplexLevel` holding the
k-simplices with their boundary and coboundary operators and, for complexes
with geometry, their circumcenters, primal and dual volumes, and diagonal Hodge
stars.

PyDEC computes geometric data lazily on first attribute access. Mojo has no
`__getattr__` hook, so MojoDEC computes it when the complex is built (pass
`compute_geometry=False` to skip it, e.g. for degenerate meshes).

The complex's data lives behind a shared `ArcPointer`: copying a
`SimplicialComplex` is cheap and every copy, like every `Cochain` built from
it, refers to the same data, mirroring Python's reference semantics.
"""

from std.math import isfinite, sqrt
from std.memory import ArcPointer, bitcast, stack_allocation

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
from mojodec.core.parallel import chunk_scratch, num_chunks, parallel_ranges
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset
from mojodec.core.utils import (
    IntArray,
    RealArray,
    compare_rows,
    factorial,
    parallelotope_volume,
    int_vector,
    lexsort_rows,
    require_distinct_simplices,
    real_array,
    real_vector,
    row_list,
    rows_of,
    cols_of,
    sort_each_row,
)
from mojodec.routines.homology import (
    betti_numbers as _betti_numbers,
    non_manifold_cells as _non_manifold_cells,
    relative_betti_numbers as _relative_betti_numbers,
)
from mojodec.routines.topology import (
    boundary_of,
    closure_of,
    interior_of,
    link_of,
    star_of,
)
from mojodec.routines.simplex_array import (
    require_coherent_orientation,
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
    """Inverse Hodge star, signed so `star_inv @ star = (-1)^(k(n-k))`.
    Infinite where a dual cell has zero volume; `star` and `delta` on dual
    cochains raise there (see `require_finite_star_inv`)."""
    var _infinite_star_inv: Int
    """The first simplex whose `star_inv` is not finite, or -1."""

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
        self._infinite_star_inv = -1

    def require_finite_star_inv(self) raises:
        """Raise if `star_inv` has an infinite entry.

        A simplex whose circumcentric dual cell has zero volume (the
        hypotenuse of a right triangle, an edge whose circumcenter lies on
        the boundary of every triangle around it) has a zero Hodge star, so
        the inverse star, and `delta` on cochains that need it, are
        undefined there. `mojodec.routines.quality.well_centeredness` finds
        such simplices; meshes that avoid them are well-centered.

        Raises:
            If a k-simplex has a zero (or non-finite) Hodge star.
        """
        var i = self._infinite_star_inv
        if i >= 0:
            raise Error(
                String(
                    "star_inv: the dual cell of ",
                    self.dim,
                    "-simplex ",
                    i,
                    " ",
                    Simplex(row_list(self.simplices, i)),
                    " has zero volume, so the inverse Hodge star is",
                    " infinite there",
                )
            )

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
        return self.find_vertices(s.vertices)

    def find_vertices(self, vertices: List[Int]) -> Int:
        """Index of the simplex with these vertices, or -1 if there is none.

        `find_simplex` without building a `Simplex`, for lookups in a loop.

        Args:
            vertices: `k + 1` vertex indices in increasing order.

        Returns:
            The simplex's row in `simplices`, or -1.
        """
        var w = self.dim + 1
        if len(vertices) != w:
            return -1
        var lo = 0
        var hi = self.num_simplices
        var ordered = len(self._order) > 0
        while lo < hi:
            var mid = (lo + hi) // 2
            var row = self._order.unsafe_get(mid) if ordered else mid
            var c = 0
            for j in range(w):
                var x = Int(self.simplices.unsafe_get(row * w + j))
                var y = vertices.unsafe_get(j)
                if x != y:
                    c = -1 if x < y else 1
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
            If the elements are malformed (an element repeats a vertex, or
            two elements have the same vertices), do not form a
            pseudomanifold (an (n-1)-simplex is a face of three or more
            elements), or are not coherently oriented (two elements induce
            the same orientation on a shared face; `orient_coherently`
            fixes an orientable mesh), or (with geometry) a simplex is
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
            If the arrays do not form a mesh, an element repeats a vertex,
            two elements have the same vertices, the elements do not form
            a coherently oriented pseudomanifold, or (with geometry) a
            simplex is degenerate.
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

    def betti_numbers(self) raises -> List[Int]:
        """The Betti numbers over the rationals; see
        `mojodec.routines.homology`.

        Returns:
            `b_0, ..., b_n`: the numbers of connected pieces, independent
            loops, enclosed voids, and so on.

        Raises:
            Never in practice.
        """
        return _betti_numbers(self.chain_complex())

    def betti_numbers(self, cells: CellSet) raises -> List[Int]:
        """The Betti numbers of a subcomplex, such as a link.

        Args:
            cells: A subcomplex: every face of a cell in it is in it.

        Returns:
            `b_0, ..., b_n` of the subcomplex.

        Raises:
            If `cells` does not belong to the complex or is not closed.
        """
        return _betti_numbers(self.chain_complex(), cells)

    def relative_betti_numbers(self) raises -> List[Int]:
        """The Betti numbers relative to the boundary, `b_k(K, ∂K)`.

        Count the harmonic forms that vanish on the boundary; for an
        orientable manifold, `b_k(K, ∂K) = b_{n-k}(K)`.

        Returns:
            `b_0, ..., b_n` of the pair (complex, `boundary_cells()`).

        Raises:
            Never in practice.
        """
        return _relative_betti_numbers(
            self.chain_complex(), self.boundary_cells()
        )

    def relative_betti_numbers(self, sub: CellSet) raises -> List[Int]:
        """The Betti numbers of the pair (complex, `sub`).

        Args:
            sub: A subcomplex: every face of a cell in it is in it.

        Returns:
            `b_0, ..., b_n` of the pair.

        Raises:
            If `sub` does not belong to the complex or is not closed.
        """
        return _relative_betti_numbers(self.chain_complex(), sub)

    def non_manifold_cells(self) raises -> CellSet:
        """The cells around which the complex is not a homology manifold;
        see `mojodec.routines.homology.non_manifold_cells`.

        Returns:
            The cells whose links have the wrong homology; empty for a
            homology manifold.

        Raises:
            Never in practice.
        """
        return _non_manifold_cells(self.chain_complex())

    def is_homology_manifold(self) raises -> Bool:
        """Whether every cell's link has the homology of a sphere (inside)
        or of a point (on the boundary).

        Returns:
            True if `non_manifold_cells()` is empty.

        Raises:
            Never in practice.
        """
        return self.non_manifold_cells().is_empty()

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

        Vertices are their own circumcenters and edges have their midpoints.
        A triangle's barycentric weights are `a (b + c - a)`, `b (c + a - b)`,
        `c (a + b - c)` over `16 area^2`, with `a`, `b`, `c` the squared
        lengths of the edges opposite each vertex; the factors in
        parentheses are computed as dot products of edges and the area by
        Householder QR, since the weights' own sum cancels for slivers. Higher dimensions solve PyDEC's
        system `[[2 P P^T, 1], [1^T, 0]] [b; mu] = [|p_i|^2; 1]` per simplex,
        in place, with `P` the positions relative to the first vertex (PyDEC
        uses absolute positions, which loses precision far from the origin).
        From triangles up, the Cartesian center is `p_0 + sum_r b_r (p_r -
        p_0)` for the same reason. Simplices are handled in parallel chunks.

        Args:
            dim: The dimension of the simplices.

        Raises:
            If `dim` is out of range or a simplex is degenerate.
        """
        ref data = self[dim]
        var E = self.embedding_dimension()
        var w = dim + 1
        var m = w + 1
        var N = data.num_simplices
        var out = real_array(N, E)
        var bary = real_array(N, w)
        var Vp = self.vertices().unsafe_ptr()
        var Sp = data.simplices.unsafe_ptr()
        var op = out.unsafe_ptr()
        var bp = bary.unsafe_ptr()
        # simplices in parallel chunks (below SERIAL_BELOW, serially); a
        # chunk that meets a degenerate simplex flags it, and the error is
        # raised after the loop
        comptime SERIAL_BELOW = 50_000
        var degenerate = List[Bool](
            length=num_chunks(N, SERIAL_BELOW), fill=False
        )
        var badp = degenerate.unsafe_ptr()

        def simplices(
            chunk: Int, lo: Int, hi: Int
        ) {dim, E, w, m, Vp, Sp, op, bp, badp}:
            var A = chunk_scratch[Float64](m * m, 0)
            var b = chunk_scratch[Float64](m, 0)
            var edges = chunk_scratch[Float64](2 * E, 0)
            for i in range(lo, hi):
                if dim == 0:
                    bp[unsafe_offset=i] = 1.0
                elif dim == 1:
                    bp[unsafe_offset=2 * i] = 0.5
                    bp[unsafe_offset=2 * i + 1] = 0.5
                elif dim == 2:
                    var v0 = Int(Sp[unsafe_offset=3 * i]) * E
                    var v1 = Int(Sp[unsafe_offset=3 * i + 1]) * E
                    var v2 = Int(Sp[unsafe_offset=3 * i + 2]) * E
                    # u = p1 - p0, w = p2 - p0, z = p2 - p1; the weight of
                    # each vertex is the squared length of the opposite
                    # edge times the dot product of the edges at the vertex
                    var uu: Float64 = 0
                    var ww: Float64 = 0
                    var zz: Float64 = 0
                    var uw: Float64 = 0
                    var uz: Float64 = 0
                    var wz: Float64 = 0
                    for k in range(E):
                        var x0 = Vp[unsafe_offset=v0 + k]
                        var u = Vp[unsafe_offset=v1 + k] - x0
                        var w = Vp[unsafe_offset=v2 + k] - x0
                        var z = (
                            Vp[unsafe_offset=v2 + k] - Vp[unsafe_offset=v1 + k]
                        )
                        uu += u * u
                        ww += w * w
                        zz += z * z
                        uw += u * w
                        uz += u * z
                        wz += w * z
                        edges.unsafe_set(k, u)
                        edges.unsafe_set(E + k, w)
                    # the weights add up to 2 |u ^ w|^2; summing them
                    # cancels for slivers, so take |u ^ w| from QR
                    var area2 = parallelotope_volume(edges, 2, E)
                    var total = 2 * area2 * area2
                    if total == 0:
                        badp[unsafe_offset=chunk] = True
                        return
                    bp[unsafe_offset=3 * i] = zz * uw / total
                    bp[unsafe_offset=3 * i + 1] = -ww * uz / total
                    bp[unsafe_offset=3 * i + 2] = uu * wz / total
                else:
                    # positions relative to vertex 0, which leaves the
                    # barycentric solution unchanged and keeps the system
                    # well conditioned far from the origin
                    var v0 = Int(Sp[unsafe_offset=i * w]) * E
                    for r in range(w):
                        var vr = Int(Sp[unsafe_offset=i * w + r]) * E
                        for c in range(w):
                            var vc = Int(Sp[unsafe_offset=i * w + c]) * E
                            var s: Float64 = 0
                            for k in range(E):
                                var x0 = Vp[unsafe_offset=v0 + k]
                                s += (Vp[unsafe_offset=vr + k] - x0) * (
                                    Vp[unsafe_offset=vc + k] - x0
                                )
                            A.unsafe_set(r * m + c, 2 * s)
                            if r == c:
                                b.unsafe_set(r, s)
                        A.unsafe_set(r * m + w, 1.0)
                        A.unsafe_set(w * m + r, 1.0)
                    A.unsafe_set(w * m + w, 0.0)
                    b.unsafe_set(w, 1.0)
                    if not _solve_in_place(A, b, m):
                        badp[unsafe_offset=chunk] = True
                        return
                    for r in range(w):
                        bp[unsafe_offset=i * w + r] = b.unsafe_get(r)
                # sum_r b_r p_r; from triangles up, as p_0 + sum_r b_r (p_r -
                # p_0), which loses no precision far from the origin
                var u0 = Int(Sp[unsafe_offset=i * w]) * E
                for k in range(E):
                    var x0: Float64 = 0
                    var first = 0
                    if dim >= 2:
                        x0 = Vp[unsafe_offset=u0 + k]
                        first = 1
                    var x: Float64 = 0
                    for r in range(first, w):
                        x += bp[unsafe_offset=i * w + r] * (
                            Vp[
                                unsafe_offset=Int(Sp[unsafe_offset=i * w + r])
                                * E
                                + k
                            ]
                            - x0
                        )
                    op[unsafe_offset=i * E + k] = x0 + x

        parallel_ranges(N, SERIAL_BELOW, simplices)
        for bad in degenerate:
            if bad:
                raise Error(
                    "circumcenter: degenerate simplex (singular system)"
                )
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

        From 4096 top simplices the walk runs in parallel: each chunk owns
        a block of vertices and the faces whose first vertex is in it, and
        walks the top simplices touching its block, so no two chunks write
        the same entry and every sum is added in the serial order (the
        results are the same as the serial walk's, bit for bit).

        Needs the circumcenters of every level.

        Raises:
            If the circumcenters have not been computed.
        """
        var n = self.complex_dimension()
        for dim in range(len(self)):
            self[dim].dual_volume = real_vector(self[dim].num_simplices)
        var walk = _DualWalk(self)
        var num_top = self[n].num_simplices
        # vertex ids index the mesh's vertices, which may include unused
        # ones, so they can exceed the number of 0-simplices
        var nv = rows_of(self.vertices())
        # Each chunk owns a block of vertices [lo, hi) and, at every level,
        # the faces whose first (smallest) vertex is in it. A top simplex
        # containing a face contains that vertex, so the chunk sees every
        # piece of the faces it owns by walking the top simplices with a
        # vertex in its block; it adds pieces to its own faces only, in
        # the serial order, so the sums are bit for bit the serial ones.
        comptime SERIAL_BELOW = 4096
        var chunks = num_chunks(num_top, SERIAL_BELOW)
        if chunks <= 1:
            walk.run(0, nv)
            return
        var bounds = _vertex_blocks(self[n].simplices, nv, chunks)
        var bp = bounds.unsafe_ptr()

        def blocks(chunk: Int, lo: Int, hi: Int) {walk, bp}:
            for c in range(lo, hi):
                walk.run(bp[unsafe_offset=c], bp[unsafe_offset=c + 1])

        parallel_ranges(chunks, 0, blocks)
        _ = bounds^
        _ = walk^

    def dual_edge_piece_values(self) raises -> RealArray:
        """The pieces `|*e ^ t|` of each edge's dual cell inside each top
        simplex: the terms the dual-volume walk adds to `sc[1].dual_volume`,
        kept apart by top simplex.

        `dual_edge_pieces` (in `mojodec.routines.musical`) pairs them with
        their edges.

        Returns:
            A `num_top_simplices x C(n+1, 2)` array: column `j` is the `j`-th
            vertex pair of the top simplex's row of `self[n].simplices`, in
            `combinations` order.

        Raises:
            If the complex has no edges or its circumcenters have not been
            computed.
        """
        var n = self.complex_dimension()
        if n < 1:
            raise Error("dual_edge_piece_values: the complex has no edges")
        var E = self.embedding_dimension()
        ref top = self[n]
        var pairs = (n + 1) * n // 2
        var pieces = List[Float64](length=top.num_simplices * pairs, fill=0)
        if n == 1:
            # an edge is its own top simplex, whose dual is a point
            for t in range(top.num_simplices):
                pieces.unsafe_set(t, 1.0)
        else:
            var q = List[Float64](length=n * E, fill=0)
            var fact = _factorials(n + 1)
            for t in range(top.num_simplices):
                self._add_dual_pieces(n, t, t, 1.0, 1.0, q, fact, pieces, pairs)
        var out = real_array(top.num_simplices, pairs)
        for i in range(len(pieces)):
            out.unsafe_set(i, pieces.unsafe_get(i))
        return out^

    def construct_hodge(self) raises:
        """The diagonal Hodge stars `dual_volume / primal_volume`, and their
        inverses signed by `(-1)^(k(n-k))` (infinite where a dual volume is
        zero; `require_finite_star_inv` reports the first).

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
            data._infinite_star_inv = -1
            for i in range(N):
                var r = data.dual_volume.unsafe_get(
                    i
                ) / data.primal_volume.unsafe_get(i)
                star.append(r)
                star_inv.append(sign / r)
                if data._infinite_star_inv < 0 and not isfinite(sign / r):
                    data._infinite_star_inv = i
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
        require_distinct_simplices(s, "SimplicialComplex")

        # levels from the top down; each step's faces are the next level
        var simplices = List[IntArray]()
        var parities = List[IntArray]()
        var cochain = List[CSRMatrix]()
        var chain = List[CSRMatrix]()
        # the top-level d is a 1 x N zero matrix
        cochain.append(CSRMatrix(1, rows_of(s)))
        simplices.append(s^)
        parities.append(parity^)
        while cols_of(simplices[len(simplices) - 1]) > 1:
            var top = len(simplices) - 1
            var fb = simplex_array_boundary(simplices[top], parities[top])
            if top == 0:
                require_coherent_orientation(
                    fb.faces, fb.boundary, fb.coboundary, "SimplicialComplex"
                )
            var faces = IntArray(Shape(0))
            var boundary = CSRMatrix(0, 0)
            var d = CSRMatrix(0, 0)
            swap(faces, fb.faces)
            swap(boundary, fb.boundary)
            swap(d, fb.coboundary)
            chain.append(boundary^)
            # d_k = B_{k+1}^T, built with the faces
            cochain.append(d^)
            parities.append(int_vector(rows_of(faces)))
            simplices.append(faces^)
        chain.append(CSRMatrix(1, rows_of(simplices[len(simplices) - 1])))

        for k in range(K):
            var j = K - 1 - k
            var level_s = IntArray(Shape(0))
            var level_p = IntArray(Shape(0))
            var level_d = CSRMatrix(0, 0)
            var level_b = CSRMatrix(0, 0)
            swap(level_s, simplices[j])
            swap(level_p, parities[j])
            swap(level_d, cochain[j])
            swap(level_b, chain[j])
            self._data[].levels.append(
                SimplexLevel(k, level_s^, level_p^, level_d^, level_b^)
            )

    def _add_dual_pieces(
        self,
        dim: Int,
        i: Int,
        root: Int,
        sign: Float64,
        vol: Float64,
        mut q: List[Float64],
        fact: List[Float64],
        mut edge_pieces: List[Float64],
        pairs: Int,
    ) raises:
        """Add the pieces of a chain of circumcenters to the dual volumes.

        The chain runs from top simplex `root` down to `dim`-simplex `i`;
        `sign` and `vol` (the chain's volume times its factorial) belong to
        that prefix, and `q` holds its orthonormalized edges. `fact[j]` is
        `j!`.

        With `pairs > 0`, the pieces of the edges' dual cells are added to
        `edge_pieces[root * pairs + j]` (`j` the edge's vertex pair in the
        root's row) instead, and the walk stops at the edges.
        """
        var n = self.complex_dimension()
        var E = self.embedding_dimension()
        var m = n - dim
        var record = pairs > 0
        ref parent = self[dim]
        ref faces = self[dim - 1]
        ref C = self[n].circumcenter
        var scale = fact.unsafe_get(m + 1)
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
            if not record:
                faces.dual_volume.unsafe_set(
                    f, faces.dual_volume.unsafe_get(f) + s * new_vol / scale
                )
            elif dim == 2:
                var j = self._local_pair(root, f)
                edge_pieces.unsafe_set(
                    root * pairs + j,
                    edge_pieces.unsafe_get(root * pairs + j)
                    + s * new_vol / scale,
                )
                continue
            if dim > 1:
                self._add_dual_pieces(
                    dim - 1, f, root, s, new_vol, q, fact, edge_pieces, pairs
                )

    def _local_pair(self, t: Int, e: Int) raises -> Int:
        """The position of edge `e` among the vertex pairs of top simplex
        `t`'s row, in `combinations` order."""
        var n = self.complex_dimension()
        ref top = self[n].simplices
        ref edges = self[1].simplices
        var a = edges.unsafe_get(2 * e)
        var b = edges.unsafe_get(2 * e + 1)
        var pa = -1
        var pb = -1
        for j in range(n + 1):
            var v = top.unsafe_get(t * (n + 1) + j)
            if v == a or v == b:
                if pa < 0:
                    pa = j
                else:
                    pb = j
        return pa * (n + 1) - pa * (pa + 1) // 2 + (pb - pa - 1)


comptime _IntPtr = Pointer[Scalar[DType.int], MutUntrackedOrigin]
comptime _FloatPtr = Pointer[Float64, MutUntrackedOrigin]


struct _DualWalk(Copyable, Movable):
    """The dual-volume walk of `compute_dual_volume` over raw pointers into
    the levels of a complex, so that it can run in a parallel loop (it
    cannot raise). The pointers are only valid while the complex lives.
    """

    var n: Int
    var E: Int
    var num_top: Int
    var simplices: List[_IntPtr]
    """Per level: the sorted vertex rows."""
    var indptr: List[Pointer[Int, MutUntrackedOrigin]]
    """Per level k: row pointers of `d_k` (rows: (k+1)-simplices)."""
    var indices: List[Pointer[Int, MutUntrackedOrigin]]
    """Per level k: column indices of `d_k`."""
    var bary: List[_FloatPtr]
    """Per level: barycentric circumcenters."""
    var center: List[_FloatPtr]
    """Per level: Cartesian circumcenters."""
    var dual: List[_FloatPtr]
    """Per level: the dual volumes being summed."""
    var fact: List[Float64]
    var vertex_index: List[Int]
    """The level-0 index of each vertex id; empty when every vertex is
    used, as the ids are then the indices."""

    def __init__(out self, sc: SimplicialComplex) raises:
        self.n = sc.complex_dimension()
        self.E = sc.embedding_dimension()
        self.num_top = sc[self.n].num_simplices
        self.simplices = List[_IntPtr]()
        self.indptr = List[Pointer[Int, MutUntrackedOrigin]]()
        self.indices = List[Pointer[Int, MutUntrackedOrigin]]()
        self.bary = List[_FloatPtr]()
        self.center = List[_FloatPtr]()
        self.dual = List[_FloatPtr]()
        for k in range(self.n + 1):
            ref level = sc[k]
            self.simplices.append(
                level.simplices.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ]()
            )
            self.indptr.append(
                level.d.indptr.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ]()
            )
            self.indices.append(
                level.d.indices.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ]()
            )
            self.bary.append(
                level.bary_circumcenter.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ]()
            )
            self.center.append(
                level.circumcenter.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ]()
            )
            self.dual.append(
                level.dual_volume.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ]()
            )
        self.fact = _factorials(self.n + 1)
        self.vertex_index = List[Int]()
        ref vertices = sc[0].simplices
        var nv = rows_of(sc.vertices())
        if rows_of(vertices) != nv:
            self.vertex_index = List[Int](length=nv, fill=-1)
            for i in range(rows_of(vertices)):
                self.vertex_index[Int(vertices.unsafe_get(i))] = i

    def run(self, lo: Int, hi: Int):
        """Add every piece whose face has its first vertex in `[lo, hi)`;
        `run(0, num_vertices)` is the whole serial walk.

        Each top simplex's faces are named by bit masks of its vertex
        positions: the walk first gathers every face's index, circumcenter,
        and barycentric circumcenter into small local tables (independent
        loads, each face once), then enumerates the chains locally. Row
        `M` of `d` lists the faces of `M` in increasing index, which for a
        sorted simplex means entry `p` omits the vertex of rank `dim - p`;
        the chains are visited in that order, as `_add_dual_pieces` does,
        so every sum is added in the same order with the same arithmetic.
        """
        var n = self.n
        var w = n + 1
        var E = self.E
        var size = 1 << w
        var top = self.simplices[n]
        var q = chunk_scratch[Float64](max(n, 1) * E, 0)
        var gidx = chunk_scratch[Int](size, 0)
        var cen = chunk_scratch[Float64](size * E, 0)
        var lbary = chunk_scratch[Float64](size * w, 0)
        # bitpos[M * w + r]: the position of the r-th set bit of M
        var bitpos = chunk_scratch[Int](size * w, 0)
        var pop = chunk_scratch[Int](size, 0)
        for M in range(size):
            var r = 0
            for j in range(w):
                if (M >> j) & 1 == 1:
                    bitpos[M * w + r] = j
                    r += 1
            pop[M] = r
        var tab = _LocalTables(
            q.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            gidx.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            cen.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            lbary.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            bitpos.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        )
        for t in range(self.num_top):
            # skip top simplices without a vertex in the block: none of
            # their faces is owned
            var touches = False
            for j in range(w):
                var v = Int(top[unsafe_offset=t * w + j])
                if v >= lo and v < hi:
                    touches = True
                    break
            if not touches:
                continue
            # the dual of a top simplex is its circumcenter: unit volume
            var first = Int(top[unsafe_offset=t * w])
            if first >= lo and first < hi:
                var dp = self.dual[n]
                dp[unsafe_offset=t] = dp[unsafe_offset=t] + 1
            if n == 0:
                continue
            if n == 3 and E == 3:
                self._gather_fixed[3, 3](t, tab)
                self._walk_soa[3, 3](tab, t, lo, hi)
            elif n == 2 and E == 2:
                self._gather_fixed[2, 2](t, tab)
                self._walk_soa[2, 2](tab, t, lo, hi)
            elif n == 2 and E == 3:
                self._gather_fixed[2, 3](t, tab)
                self._walk_soa[2, 3](tab, t, lo, hi)
            else:
                self._gather(t, size, pop, tab)
                self._walk(size - 1, n, 1.0, 1.0, tab, t, lo, hi)
        _ = q^
        _ = gidx^
        _ = cen^
        _ = lbary^
        _ = bitpos^

    @always_inline
    def _gather_fixed[n: Int, E: Int](self, t: Int, tab: _LocalTables):
        """`_gather` for a complex of dimension `n` in `E` dimensions known
        at compile time: the masks of each dimension are unrolled."""
        comptime w = n + 1
        comptime size = 1 << w
        var top = self.simplices[n]
        tab.gidx[unsafe_offset=size - 1] = t
        comptime for j in range(w):
            var v = Int(top[unsafe_offset=t * w + j])
            if self.vertex_index:
                v = self.vertex_index.unsafe_get(v)
            tab.gidx[unsafe_offset=1 << j] = v
        comptime for i in range(n - 1):
            comptime L = n - i
            var ip = self.indptr[L - 1]
            var ix = self.indices[L - 1]
            comptime for M in range(size):
                comptime if _popcount(M) == L + 1:
                    var base = ip[unsafe_offset=tab.gidx[unsafe_offset=M]]
                    comptime for p in range(L + 1):
                        comptime face = M ^ (1 << _set_bit(M, L - p))
                        tab.gidx[unsafe_offset=face] = ix[
                            unsafe_offset=base + p
                        ]
        comptime for M in range(1, size):
            comptime L = _popcount(M) - 1
            var g = tab.gidx[unsafe_offset=M]
            var c = self.center[L]
            comptime for k in range(E):
                tab.cen[unsafe_offset=M * E + k] = c[unsafe_offset=g * E + k]
            comptime if L >= 2:
                var b = self.bary[L]
                comptime for r in range(L + 1):
                    tab.bary[unsafe_offset=M * w + r] = b[
                        unsafe_offset=g * (L + 1) + r
                    ]

    def _gather(self, t: Int, size: Int, pop: List[Int], tab: _LocalTables):
        """Fill the local tables of top simplex `t`: every face's global
        index, circumcenter, and (from triangles up) barycentric
        circumcenter, by mask."""
        var n = self.n
        var w = n + 1
        var E = self.E
        var top = self.simplices[n]
        tab.gidx[unsafe_offset=size - 1] = t
        for j in range(w):
            tab.gidx[unsafe_offset=1 << j] = Int(top[unsafe_offset=t * w + j])
        if self.vertex_index:
            for j in range(w):
                tab.gidx[unsafe_offset=1 << j] = self.vertex_index.unsafe_get(
                    tab.gidx[unsafe_offset=1 << j]
                )
        # faces from the top down: the row of an L-simplex in d_{L-1}
        # names its (L-1)-faces; edges' faces are the vertices above
        for L in range(n, 1, -1):
            var ip = self.indptr[L - 1]
            var ix = self.indices[L - 1]
            for M in range(size):
                if pop[M] != L + 1:
                    continue
                var base = ip[unsafe_offset=tab.gidx[unsafe_offset=M]]
                for p in range(L + 1):
                    var bit = tab.bitpos[unsafe_offset=M * w + (L - p)]
                    tab.gidx[unsafe_offset=M ^ (1 << bit)] = ix[
                        unsafe_offset=base + p
                    ]
        for M in range(1, size):
            var L = pop[M] - 1
            var g = tab.gidx[unsafe_offset=M]
            var c = self.center[L]
            for k in range(E):
                tab.cen[unsafe_offset=M * E + k] = c[unsafe_offset=g * E + k]
            if L >= 2:
                var b = self.bary[L]
                for r in range(L + 1):
                    tab.bary[unsafe_offset=M * w + r] = b[
                        unsafe_offset=g * (L + 1) + r
                    ]

    @always_inline
    def _walk_soa[
        n: Int, E: Int
    ](self, tab: _LocalTables, t: Int, lo: Int, hi: Int):
        """`_walk` from top simplex `t`, breadth first and four chains at a
        time, for a complex of dimension `n` in `E` dimensions known at
        compile time.

        Level `m` holds the chains of `m + 1` faces below the top simplex,
        numbered in the order `_walk` visits them (chain `c`'s parent is
        chain `c // (n + 1 - m)` of level `m - 1`). Each level is computed
        four chains per SIMD vector, each lane with `_walk`'s arithmetic,
        operation for operation (the multiply-adds that the compiler fuses
        in `_walk` are explicit fused multiply-adds here); the chains of a
        level are independent, so their square roots and divisions
        overlap, where `_walk` waits on each in turn. The pieces are then
        added to the dual volumes level by level in `_walk`'s order, so
        every sum is `_walk`'s, bit for bit.
        """
        comptime w = n + 1
        comptime full = (1 << w) - 1
        comptime L = 4 * ((_chains(n, n - 1) + 3) // 4)
        # stack slots of L lanes: Q[m][k] edge k-components, then VOL[m],
        # SGN[m], and PIECE[m]
        comptime VOL0 = n * E * L
        comptime SGN0 = VOL0 + n * L
        comptime PC0 = SGN0 + n * L
        var st = stack_allocation[PC0 + n * L, DType.float64]()
        comptime V4 = SIMD[DType.float64, 4]
        var rootv = SIMD[DType.float64, 4](0)
        comptime for k in range(E):
            rootv[k] = tab.cen[unsafe_offset=full * E + k]
        comptime for m in range(n):
            comptime cnt = _chains(n, m)
            comptime pdim = n - m
            comptime for g in range((cnt + 3) // 4):
                comptime c0 = min(4 * g, cnt - 1)
                comptime c1 = min(4 * g + 1, cnt - 1)
                comptime c2 = min(4 * g + 2, cnt - 1)
                comptime c3 = min(4 * g + 3, cnt - 1)
                comptime M0 = _chain_mask(n, m, c0)
                comptime M1 = _chain_mask(n, m, c1)
                comptime M2 = _chain_mask(n, m, c2)
                comptime M3 = _chain_mask(n, m, c3)
                # new edges from the root circumcenter
                comptime for k in range(E):
                    var e = V4(
                        tab.cen[unsafe_offset=M0 * E + k],
                        tab.cen[unsafe_offset=M1 * E + k],
                        tab.cen[unsafe_offset=M2 * E + k],
                        tab.cen[unsafe_offset=M3 * E + k],
                    ) - V4(rootv[k])
                    st.unsafe_store((m * E + k) * L + 4 * g, e)
                # orthogonalized against the ancestors' edges
                comptime for j in range(m):
                    comptime D = _chain_div(n, j, m)
                    var proj = V4(0)
                    comptime for k in range(E):
                        comptime o = (j * E + k) * L
                        var qj = V4(
                            st[unsafe_offset=o + c0 // D],
                            st[unsafe_offset=o + c1 // D],
                            st[unsafe_offset=o + c2 // D],
                            st[unsafe_offset=o + c3 // D],
                        )
                        var e = st.unsafe_load[width=4]((m * E + k) * L + 4 * g)
                        proj = qj.fma(e, proj)
                    comptime for k in range(E):
                        comptime o = (j * E + k) * L
                        var qj = V4(
                            st[unsafe_offset=o + c0 // D],
                            st[unsafe_offset=o + c1 // D],
                            st[unsafe_offset=o + c2 // D],
                            st[unsafe_offset=o + c3 // D],
                        )
                        comptime at = (m * E + k) * L + 4 * g
                        st.unsafe_store(
                            at, (-proj).fma(qj, st.unsafe_load[width=4](at))
                        )
                var h2 = V4(0)
                comptime for k in range(E):
                    var e = st.unsafe_load[width=4]((m * E + k) * L + 4 * g)
                    h2 = e.fma(e, h2)
                var h = sqrt(h2)
                # the last level's edges are not used again
                comptime if m < n - 1:
                    var pos = h.gt(0)
                    comptime for k in range(E):
                        comptime at = (m * E + k) * L + 4 * g
                        var e = st.unsafe_load[width=4](at)
                        st.unsafe_store(at, pos.select(e / h, e))
                # parent's volume and sign
                var pvol = V4(1.0)
                var psgn = V4(1.0)
                comptime if m > 0:
                    comptime P = n + 1 - m
                    comptime ov = VOL0 + (m - 1) * L
                    comptime os = SGN0 + (m - 1) * L
                    pvol = V4(
                        st[unsafe_offset=ov + c0 // P],
                        st[unsafe_offset=ov + c1 // P],
                        st[unsafe_offset=ov + c2 // P],
                        st[unsafe_offset=ov + c3 // P],
                    )
                    psgn = V4(
                        st[unsafe_offset=os + c0 // P],
                        st[unsafe_offset=os + c1 // P],
                        st[unsafe_offset=os + c2 // P],
                        st[unsafe_offset=os + c3 // P],
                    )
                var vol = pvol * h
                var sgn = psgn
                comptime if pdim >= 2:
                    # flip where the parent's barycentric coordinate at
                    # the omitted vertex is < 0 or -0 (an edge's are 1/2)
                    var b = V4(
                        tab.bary[unsafe_offset=_chain_bary(n, m, c0)],
                        tab.bary[unsafe_offset=_chain_bary(n, m, c1)],
                        tab.bary[unsafe_offset=_chain_bary(n, m, c2)],
                        tab.bary[unsafe_offset=_chain_bary(n, m, c3)],
                    )
                    var neg = (bitcast[DType.uint64, 4](b) >> 63).eq(1) & b.eq(
                        b
                    )
                    sgn = neg.select(-psgn, psgn)
                st.unsafe_store(VOL0 + m * L + 4 * g, vol)
                st.unsafe_store(SGN0 + m * L + 4 * g, sgn)
                var piece = sgn * vol
                comptime if m >= 1:
                    piece = piece / V4(self.fact.unsafe_get(m + 1))
                st.unsafe_store(PC0 + m * L + 4 * g, piece)
        # add the pieces in _walk's order, to the faces the block owns
        var top = self.simplices[n]
        comptime for m in range(n):
            var fdual = self.dual[n - 1 - m]
            comptime for c in range(_chains(n, m)):
                comptime Mc = _chain_mask(n, m, c)
                var first = Int(top[unsafe_offset=t * w + _set_bit(Mc, 0)])
                if first >= lo and first < hi:
                    var f = tab.gidx[unsafe_offset=Mc]
                    fdual[unsafe_offset=f] = (
                        fdual[unsafe_offset=f]
                        + st[unsafe_offset=PC0 + m * L + c]
                    )

    def _walk(
        self,
        M: Int,
        dim: Int,
        sign: Float64,
        vol: Float64,
        tab: _LocalTables,
        t: Int,
        lo: Int,
        hi: Int,
    ):
        """`_add_dual_pieces` on the local tables, restricted to the faces
        owned by `[lo, hi)`; the arithmetic is the same, term for term.

        The multiply-adds of the dot products and of the projection's
        update are explicit fused multiply-adds (as the compiler fuses
        them anyway), so that `_walk_soa`, which writes them the same way,
        gives the same sums bit for bit.
        """
        var n = self.n
        var w = n + 1
        var E = self.E
        var m = n - dim
        var top = self.simplices[n]
        var fdual = self.dual[dim - 1]
        var q = tab.q
        var full = (1 << w) - 1
        var scale = self.fact.unsafe_get(m + 1)
        for p in range(dim + 1):
            # entry p of the row omits the vertex of rank dim - p
            var r = dim - p
            var child = M ^ (1 << tab.bitpos[unsafe_offset=M * w + r])
            var f = tab.gidx[unsafe_offset=child]
            # faces of f start at or after f's first vertex: past the
            # block, nothing below is owned
            var first = Int(
                top[unsafe_offset=t * w + tab.bitpos[unsafe_offset=child * w]]
            )
            if first >= hi:
                continue
            # sign from the parent's barycentric coordinate at the vertex
            # opposite the face (an edge's are both 1/2)
            var b: Float64 = 0.5
            if dim >= 2:
                b = tab.bary[unsafe_offset=M * w + r]
            var s = sign * (
                -1.0 if b < 0 or (b == 0 and _is_negative_zero(b)) else 1.0
            )
            # new edge from the root circumcenter, orthogonalized against q
            for k in range(E):
                q[unsafe_offset=m * E + k] = (
                    tab.cen[unsafe_offset=child * E + k]
                    - tab.cen[unsafe_offset=full * E + k]
                )
            for j in range(m):
                var proj: Float64 = 0
                for k in range(E):
                    proj = q[unsafe_offset=j * E + k].fma(
                        q[unsafe_offset=m * E + k], proj
                    )
                for k in range(E):
                    q[unsafe_offset=m * E + k] = (-proj).fma(
                        q[unsafe_offset=j * E + k], q[unsafe_offset=m * E + k]
                    )
            var h2: Float64 = 0
            for k in range(E):
                h2 = q[unsafe_offset=m * E + k].fma(
                    q[unsafe_offset=m * E + k], h2
                )
            var h = sqrt(h2)
            if h > 0:
                for k in range(E):
                    q[unsafe_offset=m * E + k] = q[unsafe_offset=m * E + k] / (
                        h
                    )
            var new_vol = vol * h
            if first >= lo:
                fdual[unsafe_offset=f] = (
                    fdual[unsafe_offset=f] + s * new_vol / scale
                )
            if dim > 1:
                self._walk(child, dim - 1, s, new_vol, tab, t, lo, hi)


@fieldwise_init
struct _LocalTables(Copyable, Movable):
    """Scratch tables of one top simplex for `_DualWalk`, indexed by face
    mask (see `_DualWalk.run`)."""

    var q: _FloatPtr
    """The chain's orthonormalized edges, `n x E`."""
    var gidx: Pointer[Int, MutUntrackedOrigin]
    """Global index of each face."""
    var cen: _FloatPtr
    """Circumcenter of each face, `E` per mask."""
    var bary: _FloatPtr
    """Barycentric circumcenter of each face, `n + 1` slots per mask."""
    var bitpos: Pointer[Int, MutUntrackedOrigin]
    """Position of the r-th set bit of each mask, `n + 1` slots per mask."""


def _vertex_blocks(top: IntArray, nv: Int, chunks: Int) -> List[Int]:
    """Boundaries `0 = b_0 <= ... <= b_chunks = nv` of vertex blocks holding
    about equal numbers of (top simplex, vertex) incidences."""
    var w = cols_of(top)
    var N = rows_of(top)
    var degree = List[Int](length=nv, fill=0)
    var tp = top.unsafe_ptr()
    for i in range(N * w):
        var v = Int(tp[unsafe_offset=i])
        degree.unsafe_set(v, degree.unsafe_get(v) + 1)
    var bounds = List[Int](length=chunks + 1, fill=nv)
    bounds.unsafe_set(0, 0)
    var total = N * w
    var acc = 0
    var c = 1
    for v in range(nv):
        while c < chunks and acc * chunks >= c * total:
            bounds.unsafe_set(c, v)
            c += 1
        acc += degree.unsafe_get(v)
    return bounds^


def _set_bit(M: Int, r: Int) -> Int:
    """The position of the `r`-th lowest set bit of `M`."""
    var c = 0
    for j in range(64):
        if (M >> j) & 1 == 1:
            if c == r:
                return j
            c += 1
    return -1


def _popcount(M: Int) -> Int:
    """The number of set bits of `M`."""
    var c = 0
    for j in range(64):
        c += (M >> j) & 1
    return c


def _chains(n: Int, m: Int) -> Int:
    """The number of chains at level `m` of an `n`-simplex's dual walk:
    `(n + 1) n ... (n + 1 - m)`."""
    var c = 1
    for l in range(m + 1):
        c *= n + 1 - l
    return c


def _chain_div(n: Int, j: Int, m: Int) -> Int:
    """Chain `c` of level `m` descends from chain `c // _chain_div(n, j, m)`
    of level `j`."""
    var d = 1
    for l in range(j + 1, m + 1):
        d *= n + 1 - l
    return d


def _chain_mask(n: Int, m: Int, c: Int) -> Int:
    """The face mask of chain `c` at level `m`: from the top simplex, step
    `l` drops the vertex of rank `(n - l) - p_l` of the face before, where
    `p_l` is digit `l` of `c` (as `_walk` drops entry `p` of a `d` row)."""
    var M = (1 << (n + 1)) - 1
    for l in range(m + 1):
        var p = (c // _chain_div(n, l, m)) % (n + 1 - l)
        M ^= 1 << _set_bit(M, (n - l) - p)
    return M


def _chain_bary(n: Int, m: Int, c: Int) -> Int:
    """The barycentric-table offset of the coordinate that signs chain `c`
    of level `m`: its parent's, at the vertex the chain drops."""
    var P = (1 << (n + 1)) - 1
    if m > 0:
        P = _chain_mask(n, m - 1, c // (n + 1 - m))
    var p = c % (n + 1 - m)
    return P * (n + 1) + (n - m) - p


def _factorials(n: Int) -> List[Float64]:
    """`j!` for `j = 0..n`."""
    var out = List[Float64](capacity=n + 1)
    var f: Float64 = 1
    out.append(1)
    for j in range(1, n + 1):
        f *= Float64(j)
        out.append(f)
    return out^


def _is_negative_zero(x: Float64) -> Bool:
    return x == 0 and (1.0 / x) < 0


def _solve_in_place(mut A: List[Float64], mut b: List[Float64], n: Int) -> Bool:
    """Solve the `n x n` system `A x = b` by Gaussian elimination with
    partial pivoting, overwriting `b` with `x` (and `A` with scratch).

    Returns False, leaving `b` undefined, if `A` is singular to working
    precision (an exactly zero pivot column, as `dense.solve` raises on).
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
            return False
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
    return True


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
    `dim` equal to the embedding dimension). Unsigned: the volume of the
    parallelotope they span (by Householder QR) over `dim!`. Runs in parallel
    chunks, each with its own buffers.
    """
    var E = cols_of(V)
    var w = dim + 1
    var scale = factorial(dim)
    var Vp = V.unsafe_ptr()
    var Sp = S.unsafe_ptr()
    var op = out.unsafe_ptr()

    def simplices(
        chunk: Int, lo: Int, hi: Int
    ) {dim, E, w, signed, scale, Vp, Sp, op}:
        var edges = chunk_scratch[Float64](dim * E, 0)
        var M = chunk_scratch[Float64](dim * dim, 0)
        var diag = chunk_scratch[Float64](dim, 0)
        for i in range(lo, hi):
            var v0 = Int(Sp[unsafe_offset=i * w]) * E
            for r in range(dim):
                var v = Int(Sp[unsafe_offset=i * w + r + 1]) * E
                for k in range(E):
                    edges.unsafe_set(
                        r * E + k,
                        Vp[unsafe_offset=v + k] - Vp[unsafe_offset=v0 + k],
                    )
            if signed:
                for r in range(dim * dim):
                    M.unsafe_set(r, edges.unsafe_get(r))
                op[unsafe_offset=i] = _det_in_place(M, dim) / scale
            else:
                op[unsafe_offset=i] = (
                    parallelotope_volume(edges, dim, E, diag) / scale
                )

    parallel_ranges(rows_of(S), 50_000, simplices)
