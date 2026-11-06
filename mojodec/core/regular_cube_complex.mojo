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

from mojodec.core.mesh import RegularCubeMesh
from mojodec.core.sparse import CSRMatrix
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

    def __init__(out self, var cube_array: IntArray, var boundary: CSRMatrix):
        """A level from its cubes and boundary operator.

        Args:
            cube_array: `N_k x (N + k)` cube rows (corner, then axes).
            boundary: The `N_{k-1} x N_k` boundary operator (`1 x N_0` zero
                for k = 0).
        """
        self.cube_array = cube_array^
        self.boundary = boundary^


struct RegularCubeComplex(Copyable, Movable, Sized, Writable):
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
            arrays[i] = fb[0].copy()
            bounds[i + 1] = fb[1].copy()
        bounds[0] = CSRMatrix(1, rows_of(arrays[0]))
        self.levels = List[CubeLevel]()
        for i in range(N + 1):
            self.levels.append(CubeLevel(arrays[i].copy(), bounds[i].copy()))
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

    def star(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """Every cell that has a cell of `cells` as a face, `cells` included.

        See `mojodec.routines.topology` for the definitions.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-cubes,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-cubes, sorted.

        Raises:
            If `cells` has too many dimensions or an index is out of range.
        """
        return star_of(self.chain_complex(), cells)

    def closure(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """Every face of every cell of `cells`: the subcomplex they generate.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-cubes,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-cubes, sorted.

        Raises:
            If `cells` has too many dimensions or an index is out of range.
        """
        return closure_of(self.chain_complex(), cells)

    def link(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """The closure of the star minus the star of the closure of `cells`.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-cubes,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-cubes, sorted.

        Raises:
            If `cells` has too many dimensions or an index is out of range.
        """
        return link_of(self.chain_complex(), cells)

    def boundary(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """The closure of the faces bounding exactly one top cell of `cells`.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-cubes,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-cubes, sorted.

        Raises:
            If `cells` has too many dimensions, an index is out of range, or
            `cells` is not pure (a lower cell is not a face of a top cell).
        """
        return boundary_of(self.chain_complex(), cells)

    def interior(self, cells: List[List[Int]]) raises -> List[List[Int]]:
        """The closure of `cells` minus its boundary.

        Args:
            cells: Indices of cells by dimension: `cells[k]` lists k-cubes,
                and may stop short of the top dimension.

        Returns:
            Indices per dimension: entry k lists k-cubes, sorted.

        Raises:
            If `cells` has too many dimensions, an index is out of range, or
            `cells` is not pure.
        """
        return interior_of(self.chain_complex(), cells)

    def boundary_cells(self) raises -> List[List[Int]]:
        """The boundary of the whole complex: the faces bounding exactly one
        top cell, and their faces.

        Returns:
            Indices per dimension: entry k lists k-cubes, sorted.

        Raises:
            If the complex's operators are inconsistent.
        """
        return boundary_of(self.chain_complex())

    def interior_cells(self) raises -> List[List[Int]]:
        """The cells not on the boundary of the whole complex.

        Returns:
            Indices per dimension: entry k lists k-cubes, sorted.

        Raises:
            If the complex's operators are inconsistent.
        """
        return interior_of(self.chain_complex())

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

    def chain_complex(self) -> List[CSRMatrix]:
        """The boundary operators of every level.

        Returns:
            `B_0, ..., B_n`, with `B_k` the `N_{k-1} x N_k` matrix from
            k-chains to (k-1)-chains and `B_0` the `1 x N_0` zero matrix.
        """
        var out = List[CSRMatrix]()
        for ref lvl in self.levels:
            out.append(lvl.boundary.copy())
        return out^

    def cochain_complex(self) raises -> List[CSRMatrix]:
        """The coboundary operators `d_k = B_{k+1}^T`, then a `1 x N_top` zero.

        Returns:
            `d_0, ..., d_n`, with `d_n` the `1 x N_n` zero matrix.

        Raises:
            Never in practice; `transpose` does not fail on a valid matrix.
        """
        var out = List[CSRMatrix]()
        for k in range(1, len(self.levels)):
            out.append(self.levels[k].boundary.transpose())
        out.append(
            CSRMatrix(1, rows_of(self.levels[len(self.levels) - 1].cube_array))
        )
        return out^

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
