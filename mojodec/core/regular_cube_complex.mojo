# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Regular cube complexes (`mojodec.core.regular_cube_complex`).
=============================================================
Port of `pydec.dec.regular_cube_complex`: the cubical complex of a
`RegularCubeMesh`, with the cube arrays and boundary operators of every
dimension.
"""

from mojodec.core.cell_set import (
    CellComplex,
    CellSet,
    cells_in_rows,
    cells_once_in_rows,
)
from mojodec.core.chain_complex import ChainComplex
from mojodec.core.mesh import RegularCubeMesh
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset
from mojodec.core.utils import (
    IntArray,
    RealArray,
    int_array,
    real_array,
    rows_of,
    cols_of,
)
from mojodec.routines.cube_array import cube_array_boundary
from mojodec.routines.topology import (
    boundary_of,
    closure_of,
    interior_of,
    link_of,
    star_of,
)


struct CubeLevel(Copyable, Movable):
    """The k-cubes of a regular cube complex."""

    var cube_array: IntArray
    """The k-cubes, one per row (corner then spanned axes)."""
    var boundary: CSRMatrix
    """Boundary operator from k-chains to (k-1)-chains."""
    var d: CSRMatrix
    """Coboundary operator from k-cochains to (k+1)-cochains, the transpose
    of the next level's `boundary`."""

    def __init__(
        out self,
        var cube_array: IntArray,
        var boundary: CSRMatrix,
        var d: CSRMatrix,
    ):
        """A level from its cubes and operators.

        Args:
            cube_array: `N_k x (N + k)` cube rows (corner, then axes).
            boundary: The `N_{k-1} x N_k` boundary operator (`1 x N_0` zero
                for k = 0).
            d: The `N_{k+1} x N_k` coboundary operator (`1 x N_n` zero for
                the top level).
        """
        self.cube_array = cube_array^
        self.boundary = boundary^
        self.d = d^

    def num_cubes(self) -> Int:
        """The number of k-cubes.

        Returns:
            The rows of `cube_array`.
        """
        return rows_of(self.cube_array)


struct RegularCubeComplex(CellComplex, Copyable, Movable, Sized, Writable):
    """The complex of a regular cube mesh.

    ```mojo
    var bitmap = nm.ones[DType.int](Shape(3, 3))
    bitmap[Item(1, 1)] = 0
    var rcc = RegularCubeComplex(RegularCubeMesh(bitmap))
    print(rcc[1].cube_array)     # the edges
    ```
    """

    var mesh: RegularCubeMesh
    """The mesh the complex was built from."""
    var levels: List[CubeLevel]
    """`levels[k]`: the k-cubes."""
    var vertices: RealArray
    """Grid coordinates of the vertices (the 0-cube corners)."""

    def __init__(out self, mesh: RegularCubeMesh) raises:
        """The complex of `mesh`'s cubes and all their faces.

        Args:
            mesh: The cube mesh; its top cubes are the bitmap's set cells.

        Raises:
            Never in practice; the cube array helpers raise only on
            malformed arrays.
        """
        self.mesh = mesh.copy()
        var N = mesh.dimension()
        var arrays = List[IntArray](length=N + 1, fill=int_array(0, 1))
        var bounds = List[CSRMatrix](length=N + 1, fill=CSRMatrix(0, 0))
        arrays[N] = mesh.cube_array()
        for i in range(N - 1, -1, -1):
            var fb = cube_array_boundary(arrays[i + 1], i + 1)
            arrays[i] = fb.faces.copy()
            bounds[i + 1] = fb.boundary.copy()
        bounds[0] = CSRMatrix(1, rows_of(arrays[0]))
        self.levels = List[CubeLevel]()
        for i in range(N + 1):
            var d = bounds[i + 1].transpose() if i < N else CSRMatrix(
                1, rows_of(arrays[i])
            )
            self.levels.append(
                CubeLevel(arrays[i].copy(), bounds[i].copy(), d^)
            )
        ref v = self.levels[0].cube_array
        self.vertices = real_array(rows_of(v), cols_of(v))
        for i in range(v.size):
            self.vertices.unsafe_set(i, Float64(Int(v.unsafe_get(i))))

    def __getitem__(self, k: Int) raises -> ref[self.levels] CubeLevel:
        """The level of k-cubes.

        Args:
            k: The dimension, `0 <= k <= complex_dimension()`.

        Returns:
            A reference to `levels[k]`.

        Raises:
            If `k` is out of range.
        """
        if k < 0 or k >= len(self.levels):
            raise Error("RegularCubeComplex: level index out of range")
        return self.levels.unsafe_ptr()[unsafe_offset=k]

    def __len__(self) -> Int:
        """The number of levels.

        Returns:
            `complex_dimension() + 1`.
        """
        return len(self.levels)

    def complex_dimension(self) -> Int:
        """Dimension of the top cubes.

        Returns:
            The mesh's grid dimension.
        """
        return self.mesh.dimension()

    def embedding_dimension(self) -> Int:
        """Dimension of the space holding the grid.

        Returns:
            The mesh's grid dimension, the same as `complex_dimension()`.
        """
        return self.mesh.dimension()

    def star(self, cells: CellSet) raises -> CellSet:
        """Every cell that has a cell of `cells` as a face, `cells` included.

        See `mojodec.routines.topology` for the definitions.

        Args:
            cells: Cubes of this complex.

        Returns:
            The star.

        Raises:
            If `cells` does not belong to this complex.
        """
        return star_of(self, cells)

    def closure(self, cells: CellSet) raises -> CellSet:
        """Every face of every cell of `cells`: the subcomplex they generate.

        Args:
            cells: Cubes of this complex.

        Returns:
            The closure.

        Raises:
            If `cells` does not belong to this complex.
        """
        return closure_of(self, cells)

    def link(self, cells: CellSet) raises -> CellSet:
        """The closure of the star minus the star of the closure of `cells`.

        Args:
            cells: Cubes of this complex.

        Returns:
            The link.

        Raises:
            If `cells` does not belong to this complex.
        """
        return link_of(self, cells)

    def boundary(self, cells: CellSet) raises -> CellSet:
        """The closure of the faces bounding exactly one top cell of `cells`.

        Args:
            cells: Cubes of this complex.

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
            cells: Cubes of this complex.

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
        return self[k].num_cubes()

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
        """The alternating sum of the numbers of k-cubes.

        Returns:
            `N_0 - N_1 + N_2 - ...`.
        """
        var chi = 0
        for k in range(len(self.levels)):
            var n = rows_of(self.levels[k].cube_array)
            chi += -n if k % 2 == 1 else n
        return chi

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
            bs.append(self.levels[k + 1].boundary.copy())
            ds.append(self.levels[k].d.copy())
        return ChainComplex(self.levels[0].num_cubes(), bs^, ds^)

    def write_to(self, mut writer: Some[Writer]):
        """Write the number of cubes of each dimension, highest first.

        Args:
            writer: The destination.
        """
        writer.write("regular_cube_complex:\n  Complex:\n")
        for i in range(len(self.levels) - 1, -1, -1):
            writer.write(
                "   ", rows_of(self.levels[i].cube_array), ": ", i, "-D cubes\n"
            )
