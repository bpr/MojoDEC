# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Mesh generation and subdivision (`mojodec.routines.mesh_generation`).
=====================================================================
Largely inspired by `pydec.mesh.generation` and `pydec.mesh.subdivision`, plus
`icosahedron` and `icosphere`, closed surfaces for testing on curved
domains.
"""

from std.math import sqrt

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import SimplicialMesh
from mojodec.routines.simplex_array import coherent_orientation
from mojodec.core.utils import (
    IntArray,
    RealArray,
    fget,
    fset,
    iget,
    int_array,
    iset,
    real_array,
    rows_of,
    cols_of,
)


struct CubeGrid(Copyable, Movable):
    """A regular grid of N-cubes, as `cube_grid` builds it."""

    var vertices: RealArray
    """`num_vertices x N` integer coordinates, axis 0 varying fastest."""
    var corners: IntArray
    """`(num_cubes, 2, ..., 2)`: entry `[c, b_0, ..., b_{N-1}]` is the
    vertex at corner `b` of cube `c`."""

    def __init__(out self, var vertices: RealArray, var corners: IntArray):
        """A grid from its parts; normally built by `cube_grid`.

        Args:
            vertices: The vertex coordinates.
            corners: The corner vertices of each cube.
        """
        self.vertices = vertices^
        self.corners = corners^


def simplicial_grid_2d(n: Int) raises -> SimplicialMesh:
    """An `n x n` grid of the unit square, each cell split into two triangles.

    ```mojo
    var sc = SimplicialComplex(simplicial_grid_2d(8))
    ```

    Args:
        n: Cells per side, at least 1.

    Returns:
        The mesh: `(n+1)^2` vertices (x varying fastest) and `2 n^2`
        counter-clockwise triangles.

    Raises:
        Does not raise in practice.
    """
    var vertices = real_array((n + 1) * (n + 1), 2)
    for j in range(n + 1):
        for i in range(n + 1):
            var v = j * (n + 1) + i
            fset(vertices, v, 0, Float64(i) / Float64(n))
            fset(vertices, v, 1, Float64(j) / Float64(n))
    var indices = int_array(2 * n * n, 3)
    for j in range(n):
        for i in range(n):
            var base = j * (n + 1) + i
            var r = 2 * n * j + i
            iset(indices, r, 0, base)
            iset(indices, r, 1, base + 1)
            iset(indices, r, 2, base + n + 2)
            iset(indices, r + n, 0, base)
            iset(indices, r + n, 1, base + n + 2)
            iset(indices, r + n, 2, base + n + 1)
    return SimplicialMesh(vertices, indices)


def cube_grid(dims: List[Int]) raises -> CubeGrid:
    """A regular N-cube mesh with the given number of cells per axis.

    Vertex `v` sits at integer coordinates with axis 0 varying fastest.
    `corners` has shape `(num_cubes, 2, ..., 2)`: entry
    `[c, b_0, ..., b_{N-1}]` is the vertex at corner `b` of cube `c`; cubes
    are enumerated with the last axis varying fastest.

    ```mojo
    var grid = cube_grid([2, 1])
    # grid.vertices [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]]
    # grid.corners  [[[0,3],[1,4]], [[1,4],[2,5]]]
    ```

    Args:
        dims: Cells along each of the `N >= 1` axes.

    Returns:
        The grid: `num_vertices x N` vertices and `(num_cubes, 2, ..., 2)`
        corner indices.

    Raises:
        Does not raise in practice.
    """
    var N = len(dims)
    var vert_dims = List[Int](capacity=N)
    for d in dims:
        vert_dims.append(d + 1)
    var cycle = List[Int](length=N, fill=1)
    for i in range(1, N):
        cycle[i] = cycle[i - 1] * vert_dims[i - 1]
    var num_vertices = cycle[N - 1] * vert_dims[N - 1]
    var vertices = real_array(num_vertices, N)
    for v in range(num_vertices):
        var rem = v
        for i in range(N):
            fset(vertices, v, i, Float64(rem % vert_dims[i]))
            rem //= vert_dims[i]
    var num_cubes = 1
    for d in dims:
        num_cubes *= d
    var shape = List[Int](capacity=N + 1)
    shape.append(num_cubes)
    for _ in range(N):
        shape.append(2)
    var indices = nm.zeros[DType.int](NDArrayShape(shape))
    var corners = 1 << N
    var corner = List[Int](length=N, fill=0)
    for c in range(num_cubes):
        var origin = 0
        for i in range(N):
            origin += corner[i] * cycle[i]
        for b in range(corners):
            # b's bits, most significant first, are (b_0, ..., b_{N-1})
            var off = 0
            for i in range(N):
                if (b >> (N - 1 - i)) & 1 == 1:
                    off += cycle[i]
            indices.unsafe_set(c * corners + b, Scalar[DType.int](origin + off))
        # next cube corner, last axis fastest
        var i = N - 1
        while i >= 0:
            corner[i] += 1
            if corner[i] < dims[i]:
                break
            corner[i] = 0
            i -= 1
    return CubeGrid(vertices^, indices^)


def triangulate_ncube(grid: CubeGrid) raises -> SimplicialMesh:
    """Triangulate a `cube_grid` of N-cubes (N <= 2); see the other overload.

    ```mojo
    var mesh = triangulate_ncube(cube_grid([3, 2]))
    ```

    Args:
        grid: The cubes.

    Returns:
        The vertices (with cell centers appended) and the simplices.

    Raises:
        If the cubes have dimension above 2.
    """
    return triangulate_ncube(grid.vertices, grid.corners)


def triangulate_ncube(
    vertices: RealArray, indices: IntArray
) raises -> SimplicialMesh:
    """Triangulate a mesh of N-cubes (N <= 2) by coning each face to the cell center.

    `indices` is laid out as `CubeGrid.corners`. A 1-cube mesh (edges) is
    returned unchanged; 2-cubes become four triangles each, with one new
    vertex per cell (numbered after the original vertices).

    Args:
        vertices: `num_vertices x E` coordinates.
        indices: `(num_cubes, 2, ..., 2)` corner indices.

    Returns:
        The vertices (with cell centers appended) and the simplices.

    Raises:
        If the cubes have dimension above 2.
    """
    var n_dims = indices.ndim - 1
    if n_dims <= 1:
        var flat = int_array(
            rows_of(indices), indices.size // max(rows_of(indices), 1)
        )
        for i in range(indices.size):
            flat.unsafe_set(i, indices.unsafe_get(i))
        return SimplicialMesh(vertices, flat)
    if n_dims > 2:
        raise Error("triangulate_ncube: nCube meshes with n > 2 not supported")
    var n_cubes = rows_of(indices)
    var n_verts = rows_of(vertices)
    var E = cols_of(vertices)
    var new_vertices = real_array(n_verts + n_cubes, E)
    for i in range(vertices.size):
        new_vertices.unsafe_set(i, vertices.unsafe_get(i))
    for c in range(n_cubes):
        for e in range(E):
            var s: Float64 = 0
            for b in range(4):
                s += fget(vertices, Int(indices.unsafe_get(c * 4 + b)), e)
            fset(new_vertices, n_verts + c, e, s / 4)
    # faces in blocks (axis 0 side 0, axis 0 side 1, axis 1 side 0, axis 1 side 1),
    # oriented so each cone triangle is positively oriented
    var tris = int_array(4 * n_cubes, 3)
    for blk in range(4):
        var axis = blk // 2
        var side = blk % 2
        var flip = (side == 0) if axis == 0 else (side == 1)
        for c in range(n_cubes):
            var a: Int
            var b: Int
            if axis == 0:
                a = Int(indices.unsafe_get(c * 4 + side * 2 + 0))
                b = Int(indices.unsafe_get(c * 4 + side * 2 + 1))
            else:
                a = Int(indices.unsafe_get(c * 4 + 0 * 2 + side))
                b = Int(indices.unsafe_get(c * 4 + 1 * 2 + side))
            if flip:
                var t = a
                a = b
                b = t
            var r = blk * n_cubes + c
            iset(tris, r, 0, n_verts + c)
            iset(tris, r, 1, a)
            iset(tris, r, 2, b)
    return SimplicialMesh(new_vertices, tris)


def icosahedron(radius: Float64 = 1.0) raises -> SimplicialMesh:
    """The regular icosahedron inscribed in the sphere of radius `radius`.

    Args:
        radius: The circumscribed sphere's radius.

    Returns:
        The mesh: 12 vertices and 20 triangles, each oriented with its
        normal pointing outward.

    Raises:
        Does not raise in practice.
    """
    var phi = (1 + sqrt(5.0)) / 2
    var raw: List[List[Float64]] = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    var scale = radius / sqrt(1 + phi * phi)
    var vertices = real_array(12, 3)
    for i in range(12):
        for k in range(3):
            fset(vertices, i, k, raw[i][k] * scale)
    var faces: List[List[Int]] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    var triangles = int_array(20, 3)
    for t in range(20):
        for k in range(3):
            iset(triangles, t, k, faces[t][k])
    return SimplicialMesh(vertices, triangles)


def icosphere(
    subdivisions: Int, radius: Float64 = 1.0
) raises -> SimplicialMesh:
    """A triangulated sphere: a subdivided icosahedron projected onto the sphere.

    Each subdivision splits every triangle into four (`loop_subdivision`)
    and moves the new vertices out to the sphere, so the mesh has
    `10 * 4^s + 2` vertices and `20 * 4^s` triangles, all oriented outward
    and nearly equilateral.

    Args:
        subdivisions: The number of subdivision steps `s`, at least 0.
        radius: The sphere's radius.

    Returns:
        The mesh.

    Raises:
        If `subdivisions` is negative.
    """
    if subdivisions < 0:
        raise Error("icosphere: subdivisions must be >= 0")
    var mesh = icosahedron(radius)
    for _ in range(subdivisions):
        mesh = loop_subdivision(mesh)
        ref vertices = mesh.vertices
        for v in range(rows_of(vertices)):
            var r: Float64 = 0
            for k in range(3):
                r += fget(vertices, v, k) ** 2
            r = sqrt(r)
            for k in range(3):
                fset(vertices, v, k, fget(vertices, v, k) * radius / r)
    return mesh^


def loop_subdivision(mesh: SimplicialMesh) raises -> SimplicialMesh:
    """One step of Loop's subdivision connectivity on a triangle mesh.

    Every edge gains a midpoint vertex and every triangle becomes four.
    Midpoint vertices are numbered after the original vertices, in order of
    first appearance of their edges. Vertices are not moved, so this is
    midpoint subdivision; Loop's smoothing weights are not applied.

    Args:
        mesh: A triangle mesh.

    Returns:
        The vertices with edge midpoints appended, and the
        `4 num_triangles x 3` triangles, oriented as their parents.

    Raises:
        If the mesh is not made of triangles.
    """
    ref vertices = mesh.vertices
    ref simplices = mesh.elements
    if cols_of(simplices) != 3:
        raise Error("loop_subdivision: the mesh must be made of triangles")
    var nv = rows_of(vertices)
    var nt = rows_of(simplices)
    var edge_index = Dict[Int, Int]()
    var edge_ends = List[Tuple[Int, Int]]()

    def edge_vertex(
        a: Int, b: Int
    ) {mut edge_index, mut edge_ends, imm nv} -> Int:
        var lo = min(a, b)
        var hi = max(a, b)
        var key = lo * nv + hi
        var found = edge_index.get(key)
        if found:
            return found.value()
        var idx = nv + len(edge_ends)
        edge_index[key] = idx
        edge_ends.append((lo, hi))
        return idx

    var new_tris = int_array(4 * nt, 3)
    for t in range(nt):
        var v0 = iget(simplices, t, 0)
        var v1 = iget(simplices, t, 1)
        var v2 = iget(simplices, t, 2)
        var e01 = edge_vertex(v0, v1)
        var e12 = edge_vertex(v1, v2)
        var e20 = edge_vertex(v2, v0)
        var rows: List[List[Int]] = [
            [v0, e01, e20],
            [v1, e12, e01],
            [v2, e20, e12],
            [e01, e12, e20],
        ]
        for k in range(4):
            for j in range(3):
                iset(new_tris, 4 * t + k, j, rows[k][j])
    var E = cols_of(vertices)
    var new_vertices = real_array(nv + len(edge_ends), E)
    for i in range(vertices.size):
        new_vertices.unsafe_set(i, vertices.unsafe_get(i))
    for k in range(len(edge_ends)):
        for e in range(E):
            var m = 0.5 * (
                fget(vertices, edge_ends[k][0], e)
                + fget(vertices, edge_ends[k][1], e)
            )
            fset(new_vertices, nv + k, e, m)
    return SimplicialMesh(new_vertices, new_tris)


def orient_coherently(mesh: SimplicialMesh) raises -> SimplicialMesh:
    """The mesh with its elements coherently oriented, as
    `SimplicialComplex` requires.

    Neighboring elements are made to induce opposite orientations on their
    shared (n-1)-face (see `coherent_orientation`). The first element of
    each piece connected through faces keeps its orientation; an element
    that must be reversed has its last two vertices swapped. Vertices and
    element order are unchanged.

    Args:
        mesh: A pseudomanifold mesh.

    Returns:
        A copy of `mesh` that `SimplicialComplex` accepts.

    Raises:
        If the elements are malformed, or the mesh is not a pseudomanifold
        or not orientable.
    """
    var signs = coherent_orientation(mesh.elements)
    if not signs:
        raise Error("orient_coherently: the mesh is not orientable")
    var out = mesh.copy()
    var K = cols_of(out.elements)
    ref s = signs.value()
    for t in range(len(s)):
        if s[t] < 0:
            var a = iget(out.elements, t, K - 2)
            iset(out.elements, t, K - 2, iget(out.elements, t, K - 1))
            iset(out.elements, t, K - 1, a)
    return out^
