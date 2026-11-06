# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Meshes (`mojodec.core.mesh`).
=============================
Ports of `pydec.mesh.simplex.simplicial_mesh` and
`pydec.mesh.regular_cube.regular_cube_mesh`.
"""

import numojo as nm
from numojo.prelude import *

from mojodec.core.simplex import Simplex
from mojodec.core.utils import (
    IntArray,
    RealArray,
    int_array,
    iget,
    iset,
    max_entry,
    min_entry,
    row_list,
    rows_of,
    cols_of,
    take_rows,
)
from mojodec.routines.combinatorial import combinations
from mojodec.routines.geometry import signed_volume


struct SimplicialMesh(Copyable, Movable, Writable):
    """A simplicial mesh: vertex coordinates plus top-simplex vertex lists.

    ```mojo
    var V = nm.fromstring[f64]("[[0,0],[1,0],[0,1]]")
    var E = nm.fromstring[DType.int]("[[0,1,2]]")
    var mesh = SimplicialMesh(V, E)
    ```
    """

    var vertices: RealArray
    """`num_vertices x embedding_dimension` vertex coordinates."""
    var elements: IntArray
    """`num_elements x (manifold_dimension + 1)` vertex indices."""

    def __init__(out self, vertices: RealArray, elements: IntArray) raises:
        """A mesh on copies of `vertices` and `elements`.

        Args:
            vertices: `num_vertices x embedding_dimension` coordinates.
            elements: `num_elements x (manifold_dimension + 1)` vertex
                indices.

        Raises:
            If either array is not 2-D, or an index is not a row of
            `vertices`.
        """
        if vertices.ndim != 2 or elements.ndim != 2:
            raise Error(
                "SimplicialMesh: index and vertex arrays must have rank 2"
            )
        if elements.size > 0 and (
            min_entry(elements) < 0 or max_entry(elements) >= rows_of(vertices)
        ):
            raise Error("SimplicialMesh: invalid index value")
        self.vertices = vertices.copy()
        self.elements = elements.copy()

    def indices(self) -> IntArray:
        """The element array (PyDEC's `indices` alias of `elements`).

        Returns:
            A copy of `elements`.
        """
        return self.elements.copy()

    def manifold_dimension(self) -> Int:
        """Dimension of the top simplices.

        Returns:
            The number of columns of `elements`, minus 1.
        """
        return cols_of(self.elements) - 1

    def embedding_dimension(self) -> Int:
        """Dimension of the space holding the vertices.

        Returns:
            The number of columns of `vertices`.
        """
        return cols_of(self.vertices)

    def boundary(self) raises -> List[Simplex]:
        """The oriented faces of the top simplices that occur exactly once.

        Returned in lexicographic order of their vertices.

        Returns:
            The boundary faces, each oriented as induced by the one top
            simplex containing it.

        Raises:
            If an element has a repeated vertex.
        """
        var count = Dict[Simplex, Int]()
        var faces = List[Simplex]()
        for i in range(rows_of(self.elements)):
            var s = Simplex(row_list(self.elements, i))
            for b in s.boundary():
                if b in count:
                    count[b] += 1
                else:
                    count[b.copy()] = 1
                    faces.append(b.copy())
        var out = List[Simplex]()
        for ref f in faces:
            if count[f] == 1:
                out.append(f.copy())
        _sort_simplices(out)
        return out^

    def skeleton(self, p: Int) raises -> List[Simplex]:
        """All the p-faces of the mesh, in lexicographic order.

        Args:
            p: The face dimension, `0 <= p <= manifold_dimension()`.

        Returns:
            The distinct p-faces, with parity 0.

        Raises:
            If `p` is out of range, or an element has a repeated vertex.
        """
        if p < 0 or p > self.manifold_dimension():
            raise Error("SimplicialMesh.skeleton: invalid dimension")
        var seen = Dict[Simplex, Int]()
        var out = List[Simplex]()
        for i in range(rows_of(self.elements)):
            for b in combinations(row_list(self.elements, i), p + 1):
                var s = Simplex(b)
                if s not in seen:
                    seen[s.copy()] = 1
                    out.append(s^)
        _sort_simplices(out)
        return out^

    def orient(mut self) raises -> Int:
        """Orient a full-dimensional mesh so all top simplices have positive volume.

        Returns the number of simplices flipped. Only meshes whose manifold
        dimension equals their embedding dimension are supported, as in PyDEC.
        A simplex is flipped by swapping its first two vertices.

        Returns:
            The number of simplices flipped.

        Raises:
            If the manifold dimension is neither 0 nor the embedding
            dimension.
        """
        if self.manifold_dimension() == 0:
            return 0
        if self.manifold_dimension() != self.embedding_dimension():
            raise Error(
                "SimplicialMesh.orient: only full-dimensional meshes are"
                " supported"
            )
        var num_flips = 0
        for i in range(rows_of(self.elements)):
            var pts = take_rows(self.vertices, row_list(self.elements, i))
            if signed_volume(pts) < 0:
                num_flips += 1
                var t = iget(self.elements, i, 0)
                iset(self.elements, i, 0, iget(self.elements, i, 1))
                iset(self.elements, i, 1, t)
        return num_flips

    def write_to(self, mut writer: Some[Writer]):
        """Writes a one-line summary of the dimensions and sizes.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "simplicial_mesh< ",
            self.manifold_dimension(),
            "D manifold, ",
            self.embedding_dimension(),
            "D embedding, ",
            rows_of(self.vertices),
            " vertices, ",
            rows_of(self.elements),
            " elements >",
        )


struct RegularCubeMesh(Copyable, Movable):
    """A regular grid of hypercubes given by a boolean occupancy bitmap.

    The bitmap is an N-dimensional NuMojo array; nonzero entries are cubes.

    ```mojo
    var bitmap = nm.ones[DType.int](Shape(3, 3))
    bitmap[Item(1, 1)] = 0   # a 3x3 grid with a center hole
    var mesh = RegularCubeMesh(bitmap)
    ```
    """

    var bitmap: IntArray
    """Occupancy of each grid cell (nonzero means present)."""

    def __init__(out self, bitmap: IntArray):
        """A mesh on a copy of `bitmap`.

        Args:
            bitmap: An N-dimensional occupancy array.
        """
        self.bitmap = bitmap.copy()

    def dimension(self) -> Int:
        """Dimension of the grid.

        Returns:
            The number of axes of `bitmap`.
        """
        return self.bitmap.ndim

    def cube_array(self) raises -> IntArray:
        """The cube array of the bitmap's cubes.

        Each row is `[i_0, ..., i_{N-1}, 0, 1, ..., N-1]`: the cube's grid
        corner followed by the axes it spans. Rows are in C (row-major)
        order of the bitmap.

        Returns:
            A `num_cubes x 2N` integer array.

        Raises:
            If NuMojo cannot allocate the array.
        """
        var N = self.bitmap.ndim
        var shape = List[Int](capacity=N)
        for d in range(N):
            shape.append(Int(self.bitmap.shape.unsafe_get(d)))
        var corners = List[List[Int]]()
        var idx = List[Int](length=N, fill=0)
        for flat in range(self.bitmap.size):
            if self.bitmap.unsafe_get(flat) != 0:
                corners.append(idx.copy())
            # advance the multi-index in C order
            var d = N - 1
            while d >= 0:
                idx[d] += 1
                if idx[d] < shape[d]:
                    break
                idx[d] = 0
                d -= 1
        var out = int_array(len(corners), 2 * N)
        for i in range(len(corners)):
            for d in range(N):
                iset(out, i, d, corners[i][d])
                iset(out, i, N + d, d)
        return out^


def _sort_simplices(mut xs: List[Simplex]):
    def less(a: Simplex, b: Simplex) -> Bool:
        return _less(a.vertices, b.vertices)

    sort(Span(xs), less)


def _less(a: List[Int], b: List[Int]) -> Bool:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return a[i] < b[i]
    return len(a) < len(b)
