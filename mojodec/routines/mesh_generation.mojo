# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Mesh generation and subdivision (`mojodec.routines.mesh_generation`).
=====================================================================
Largely inspired by `pydec.mesh.generation` and `pydec.mesh.subdivision`, plus
`icosahedron` and `icosphere`, closed surfaces for testing on curved
domains, and `kuhn_grid` and `flat_torus`, boxes and flat tori of any
dimension.

`kuhn_grid` splits each box of a grid into d! simplices, one per order of
the axes (Kuhn's or Freudenthal's triangulation): from a corner, step
along the axes in that order. The simplices are right-angled, so some
circumcentric dual volumes are zero (the hypotenuses in 2D); use the
Whitney inner products there, not the diagonal Hodge star's inverse.
`flat_torus` glues opposite faces of the same grid and places it in
R^(2d), each axis on a circle sized so the mesh is exactly the flat torus:
a closed manifold with Betti numbers `binomial(d, k)`. `mobius_band` and
`klein_bottle` glue a 2D grid with a flip: non-orientable surfaces, which
`SimplicialComplex` refuses (test meshes for orientation checks).

```mojo
var sc = SimplicialComplex(kuhn_grid([4, 4, 4]))        # 384 tetrahedra
var torus = SimplicialComplex(flat_torus([8, 8]))       # in R^4
print(torus.betti_numbers())                            # [1, 2, 1]
```
"""

from std.math import cos, isfinite, pi, sin, sqrt

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
    lexsort_rows,
    sort_each_row,
    take_rows,
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


struct ReorderedMesh(Copyable, Movable):
    """A mesh renumbered by `reorder_vertices`, with the permutations that
    carry data between the old and new numbering."""

    var mesh: SimplicialMesh
    """The renumbered mesh."""
    var old_vertex: List[Int]
    """`old_vertex[i]` is the old number of new vertex `i`: vertex data
    `x` becomes `x[old_vertex[i]]` (`take_rows(x, old_vertex)`, from
    `mojodec.core.utils`)."""
    var new_vertex: List[Int]
    """`new_vertex[j]` is the new number of old vertex `j`: a list of old
    vertex numbers (boundary vertices, say) maps through it."""
    var old_element: List[Int]
    """`old_element[t]` is the old row of new element `t`: element data
    `y` becomes `y[old_element[t]]`."""

    def __init__(
        out self,
        var mesh: SimplicialMesh,
        var old_vertex: List[Int],
        var new_vertex: List[Int],
        var old_element: List[Int],
    ):
        """Assemble the result; normally built by `reorder_vertices`.

        Args:
            mesh: The renumbered mesh.
            old_vertex: The old number of each new vertex.
            new_vertex: The new number of each old vertex.
            old_element: The old row of each new element.
        """
        self.mesh = mesh^
        self.old_vertex = old_vertex^
        self.new_vertex = new_vertex^
        self.old_element = old_element^


def morton_keys(vertices: RealArray) raises -> List[Int]:
    """The Morton (Z-order) code of each point, from its coordinates.

    Each coordinate is scaled to the points' bounding box and quantized to
    `b = min(21, 63 // E)` bits (`E` the number of columns; an axis on
    which all points agree quantizes to 0), and the code interleaves the
    bits, most significant first and axis 0 lowest within each bit: bit
    `l E + a` of the code is bit `l` of axis `a`. Points close in the code
    are close in space (the converse fails across the curve's jumps).

    Args:
        vertices: `num_points x E` coordinates, `1 <= E <= 63`.

    Returns:
        One code per point, in `[0, 2^(b E))`.

    Raises:
        If `E` is 0 or above 63, or a coordinate is not finite.
    """
    var nv = rows_of(vertices)
    var E = cols_of(vertices)
    if E < 1 or E > 63:
        raise Error(
            String("morton_keys: need 1 to 63 coordinates per point, got ", E)
        )
    var vp = vertices.unsafe_ptr()
    var lo = List[Float64](length=E, fill=0)
    var hi = List[Float64](length=E, fill=0)
    for i in range(nv):
        for a in range(E):
            var x = Float64(vp[unsafe_offset=i * E + a])
            if not isfinite(x):
                raise Error(
                    String(
                        "morton_keys: coordinate ",
                        a,
                        " of point ",
                        i,
                        " is not finite",
                    )
                )
            if i == 0 or x < lo[a]:
                lo[a] = x
            if i == 0 or x > hi[a]:
                hi[a] = x
    var bits = min(21, 63 // E)
    var top = (1 << bits) - 1
    var scale = Float64(1 << bits)
    var q = List[Int](length=E, fill=0)
    var keys = List[Int](length=nv, fill=0)
    for i in range(nv):
        for a in range(E):
            var width = hi[a] - lo[a]
            if width > 0:
                var t = (Float64(vp[unsafe_offset=i * E + a]) - lo[a]) / width
                q[a] = min(Int(t * scale), top)
            else:
                q[a] = 0
        var key = 0
        for l in range(bits):
            for a in range(E):
                key |= ((q[a] >> l) & 1) << (l * E + a)
        keys[i] = key
    return keys^


def reorder_vertices(mesh: SimplicialMesh) raises -> ReorderedMesh:
    """The mesh with its vertices renumbered in Morton (Z-curve) order and
    its elements sorted to match: the same mesh, numbered so that nearby
    vertices and elements get nearby numbers.

    Faces are numbered by their sorted vertex numbers, and the loops that
    build a `SimplicialComplex` and its Whitney matrices run in vertex,
    face, or element order, so a mesh whose numbering jumps around in space
    (as files often have) is several times slower to build than a
    spatially numbered one; the parallel dual-volume walk also splits its
    work by blocks of vertex numbers. See the guide's vertex numbering
    page.

    Vertices are ordered by `morton_keys` of their coordinates, equal keys
    in their old order (so unused vertices are kept, and reordering twice
    changes nothing). Elements are ordered by their vertex numbers sorted
    (lexicographically, with ties in their old order); each element keeps
    its vertices in the same positions, so its orientation is unchanged.

    ```mojo
    from mojodec.core.utils import take_rows

    var r = reorder_vertices(mesh)
    var sc = SimplicialComplex(r.mesh)
    var u = take_rows(u_old, r.old_vertex)    # vertex data, new order
    ```

    Args:
        mesh: Any simplicial mesh.

    Returns:
        The renumbered mesh and the permutations between the numberings.

    Raises:
        If a coordinate is not finite, the embedding dimension is above 63,
        or an element names a vertex that does not exist.
    """
    var nv = rows_of(mesh.vertices)
    var ne = rows_of(mesh.elements)
    var K = cols_of(mesh.elements)
    var keys = morton_keys(mesh.vertices)
    var key_col = int_array(nv, 1)
    for i in range(nv):
        key_col.unsafe_set(i, Scalar[DType.int](keys[i]))
    var old_vertex = lexsort_rows(key_col)
    var new_vertex = List[Int](length=nv, fill=0)
    for i in range(nv):
        new_vertex[old_vertex[i]] = i
    # every element in the new numbering, and a copy with sorted rows
    var ep = mesh.elements.unsafe_ptr()
    var renamed = int_array(ne, K)
    for i in range(ne * K):
        var v = Int(ep[unsafe_offset=i])
        if v < 0 or v >= nv:
            raise Error(
                String(
                    "reorder_vertices: element ",
                    i // K,
                    " names vertex ",
                    v,
                    " of ",
                    nv,
                )
            )
        renamed.unsafe_set(i, Scalar[DType.int](new_vertex[v]))
    var sorted_rows = renamed.copy()
    sort_each_row(sorted_rows)
    var old_element = lexsort_rows(sorted_rows)
    var elements = int_array(ne, K)
    for t in range(ne):
        var r = old_element[t]
        for j in range(K):
            elements.unsafe_set(t * K + j, renamed.unsafe_get(r * K + j))
    var vertices = take_rows(mesh.vertices, old_vertex)
    return ReorderedMesh(
        SimplicialMesh(vertices, elements),
        old_vertex^,
        new_vertex^,
        old_element^,
    )


def kuhn_grid(dims: List[Int]) raises -> SimplicialMesh:
    """`kuhn_grid(dims, lengths)` with the unit cube `[0, 1]^d`.

    Args:
        dims: Cells along each axis.

    Returns:
        The mesh.

    Raises:
        As the overload with `lengths`.
    """
    return kuhn_grid(dims, List[Float64](length=len(dims), fill=1.0))


def kuhn_grid(dims: List[Int], lengths: List[Float64]) raises -> SimplicialMesh:
    """A box split into simplices by Kuhn's triangulation, in any dimension.

    The box `[0, L_0] x ... x [0, L_{d-1}]` is cut into `dims[a]` cells
    along axis `a`, and each cell into `d!` simplices, one per permutation
    `p` of the axes: from the cell's lowest corner, step along axis `p[0]`,
    then `p[1]`, and so on to the opposite corner. Neighboring cells cut
    their shared faces the same way, so the simplices form a conforming
    mesh, every one positively oriented (odd permutations are listed with
    their last two vertices swapped). In 2D every square is cut along the
    same diagonal, as in `simplicial_grid_2d`.

    The simplices are right-angled: in 2D the circumcenter of each
    triangle is the midpoint of its hypotenuse, so the hypotenuse's dual
    edge has length zero, and similar zero dual volumes occur in every
    dimension. The diagonal Hodge star has zeros there and its inverse is
    infinite; use `whitney_innerproduct`. With it, the 0-form stiffness
    `d0^T W_1 d0` is exactly the finite-difference Laplacian (the
    `2d + 1`-point stencil) times the cell volume.

    ```mojo
    var mesh = kuhn_grid([4, 2], [2.0, 1.0])    # 16 triangles on [0,2] x [0,1]
    ```

    Args:
        dims: Cells along each axis, `d = len(dims) >= 1`, each at least 1.
        lengths: The side lengths `L_a` of the box.

    Returns:
        The mesh: `prod(dims[a] + 1)` vertices with axis 0 varying fastest
        (as `cube_grid`), and `d! prod(dims)` elements, cell by cell.

    Raises:
        If `dims` is empty or has an entry below 1, or `lengths` has the
        wrong length or an entry that is not positive and finite.
    """
    var h = _spacing(dims, lengths, 1, "kuhn_grid")
    var d = len(dims)
    var vdims = List[Int](capacity=d)
    for a in range(d):
        vdims.append(dims[a] + 1)
    var nv = 1
    for a in range(d):
        nv *= vdims[a]
    var vertices = real_array(nv, d)
    for v in range(nv):
        var rem = v
        for a in range(d):
            fset(vertices, v, a, Float64(rem % vdims[a]) * h[a])
            rem //= vdims[a]
    return SimplicialMesh(vertices, _kuhn_elements(dims, vdims, False))


def flat_torus(dims: List[Int]) raises -> SimplicialMesh:
    """`flat_torus(dims, lengths)` with periods 1.

    Args:
        dims: Cells along each axis.

    Returns:
        The mesh.

    Raises:
        As the overload with `lengths`.
    """
    return flat_torus(dims, List[Float64](length=len(dims), fill=1.0))


def flat_torus(
    dims: List[Int], lengths: List[Float64]
) raises -> SimplicialMesh:
    """The flat d-torus `R^d / (L_0 Z x ... x L_{d-1} Z)`, triangulated as a
    periodic `kuhn_grid` and embedded isometrically in `R^(2d)`.

    Opposite faces of the grid are glued, so vertex `i` along axis `a` is
    vertex `i mod dims[a]`. A flat torus has no isometric embedding in
    `R^d`; axis `a` goes on a circle in its own plane (coordinates `2a`,
    `2a + 1`, the Clifford embedding), with radius
    `r_a = h_a / (2 sin(pi / dims[a]))`, so that each step along the axis,
    a chord of the circle, has exactly the grid spacing `h_a = L_a /
    dims[a]`. The axes' planes are orthogonal, so every simplex has the
    edge lengths and angles of its flat counterpart: the mesh is the flat
    torus exactly, not an approximation of it (its total volume is
    `prod(L_a)`).

    It is a closed, orientable manifold with Betti numbers
    `binomial(d, k)`, the constant forms `dx^I` being the harmonic ones,
    and the Laplacian on functions has the eigenvalues
    `(2 pi)^2 sum_a (m_a / L_a)^2`. Like `kuhn_grid`, it has zero dual
    volumes; use the Whitney inner products.

    ```mojo
    var sc = SimplicialComplex(flat_torus([6, 6, 6]))   # a 3-torus in R^6
    print(sc.betti_numbers())                         # [1, 3, 3, 1]
    ```

    Args:
        dims: Cells along each axis, `d = len(dims) >= 1`, each at least 3
            (with 2, the two neighbors of a vertex along an axis would be
            the same vertex).
        lengths: The periods `L_a`.

    Returns:
        The mesh: `prod(dims)` vertices in `R^(2d)` (vertex numbering as in
        `kuhn_grid`, without the last layer of each axis) and
        `d! prod(dims)` elements.

    Raises:
        If `dims` is empty or has an entry below 3, or `lengths` has the
        wrong length or an entry that is not positive and finite.
    """
    var h = _spacing(dims, lengths, 3, "flat_torus")
    var d = len(dims)
    var nv = 1
    for a in range(d):
        nv *= dims[a]
    var vertices = real_array(nv, 2 * d)
    for v in range(nv):
        var rem = v
        for a in range(d):
            var i = rem % dims[a]
            rem //= dims[a]
            var n = Float64(dims[a])
            var r = h[a] / (2 * sin(pi / n))
            var t = 2 * pi * Float64(i) / n
            fset(vertices, v, 2 * a, r * cos(t))
            fset(vertices, v, 2 * a + 1, r * sin(t))
    return SimplicialMesh(vertices, _kuhn_elements(dims, dims, True))


def mobius_band(
    segments: Int, across: Int = 1, width: Float64 = 1.0
) raises -> SimplicialMesh:
    """A Möbius band in `R^3`: a strip of Kuhn triangles glued to itself
    with a half twist. It is not orientable.

    The strip's `(u, w)` grid, `u` around in `segments` steps and `w`
    across in `across` steps, is glued at `u = 2 pi` with `w -> -w`, and
    placed at `((1 + w cos(u/2)) cos u, (1 + w cos(u/2)) sin u,
    w sin(u/2))`. Every triangle is oriented as in the flat strip, so two
    triangles across the seam induce the same orientation on their shared
    edge: `SimplicialComplex` refuses the mesh, `coherent_orientation`
    returns `None`, and `orient_coherently` raises. It is a homology
    2-manifold with boundary (one circle of `2 segments` edges) and Betti
    numbers `[1, 1, 0]`; use it through `AbstractSimplicialComplex`.

    ```mojo
    var m = mobius_band(12, 2)
    var asc = AbstractSimplicialComplex([m.elements])
    print(asc.betti_numbers())          # [1, 1, 0]
    ```

    Args:
        segments: Steps around, at least 3.
        across: Steps across, at least 1.
        width: The width of the strip, positive and below 2.

    Returns:
        `segments (across + 1)` vertices and `2 segments across` triangles.

    Raises:
        If `segments < 3`, `across < 1`, or `width` is not in `(0, 2)`.
    """
    if segments < 3 or across < 1:
        raise Error(
            String(
                "mobius_band: need segments >= 3 and across >= 1, got ",
                segments,
                " and ",
                across,
            )
        )
    if not (width > 0 and width < 2):
        raise Error(String("mobius_band: width must be in (0, 2), got ", width))
    var rows = across + 1
    var vertices = real_array(segments * rows, 3)
    for i in range(segments):
        var u = 2 * pi * Float64(i) / Float64(segments)
        for j in range(rows):
            var w = width * (Float64(j) / Float64(across) - 0.5)
            var r = 1 + w * cos(u / 2)
            var v = i * rows + j
            fset(vertices, v, 0, r * cos(u))
            fset(vertices, v, 1, r * sin(u))
            fset(vertices, v, 2, w * sin(u / 2))

    def vid(i: Int, j: Int) {imm segments, imm rows, imm across} -> Int:
        if i == segments:  # the seam: w -> -w
            return across - j
        return i * rows + j

    return SimplicialMesh(vertices, _strip_triangles(segments, across, vid))


def klein_bottle(dims: List[Int]) raises -> SimplicialMesh:
    """A Klein bottle in `R^4`: a periodic Kuhn grid with one pair of
    sides glued with a flip. It is not orientable.

    The `(u, v)` grid is glued at `v = 2 pi` as a torus and at `u = 2 pi`
    with `v -> -v`, and placed at `((2 + cos v) cos u, (2 + cos v) sin u,
    sin v cos(u/2), sin v sin(u/2))`, which is well defined on the
    quotient and has no self-intersections. As for `mobius_band`, the
    triangles keep the grid's orientation, so they disagree across the
    flipped seam and `SimplicialComplex` refuses the mesh. It is a closed
    homology 2-manifold with Euler characteristic 0 and rational Betti
    numbers `[1, 1, 0]` (over Z/2, `[1, 2, 1]`).

    Args:
        dims: Cells along `u` and `v`, `[n_u, n_v]`, each at least 3.

    Returns:
        `n_u n_v` vertices and `2 n_u n_v` triangles.

    Raises:
        If `dims` does not have two entries of at least 3.
    """
    if len(dims) != 2 or dims[0] < 3 or dims[1] < 3:
        raise Error("klein_bottle: dims must be [n_u, n_v], each at least 3")
    var nu = dims[0]
    var nv = dims[1]
    var vertices = real_array(nu * nv, 4)
    for i in range(nu):
        var u = 2 * pi * Float64(i) / Float64(nu)
        for j in range(nv):
            var v = 2 * pi * Float64(j) / Float64(nv)
            var k = i * nv + j
            fset(vertices, k, 0, (2 + cos(v)) * cos(u))
            fset(vertices, k, 1, (2 + cos(v)) * sin(u))
            fset(vertices, k, 2, sin(v) * cos(u / 2))
            fset(vertices, k, 3, sin(v) * sin(u / 2))

    def vid(i: Int, j: Int) {imm nu, imm nv} -> Int:
        var jj = j % nv
        if i == nu:  # the flipped seam: v -> -v
            return (nv - jj) % nv
        return i * nv + jj

    return SimplicialMesh(vertices, _strip_triangles(nu, nv, vid))


def _strip_triangles[
    F: def(Int, Int) -> Int
](n: Int, m: Int, vid: F) raises -> IntArray:
    """The two Kuhn triangles of each cell `(i, j)` of an `n x m` grid,
    positively oriented in the grid's coordinates, with vertex numbers
    from `vid(i, j)` (`i` up to `n`, `j` up to `m`), which does the
    gluing."""
    var out = int_array(2 * n * m, 3)
    var row = 0
    for i in range(n):
        for j in range(m):
            var a = vid(i, j)
            var b = vid(i + 1, j)
            var c = vid(i + 1, j + 1)
            var e = vid(i, j + 1)
            iset(out, row, 0, a)
            iset(out, row, 1, b)
            iset(out, row, 2, c)
            iset(out, row + 1, 0, a)
            iset(out, row + 1, 1, c)
            iset(out, row + 1, 2, e)
            row += 2
    return out^


def _spacing(
    dims: List[Int], lengths: List[Float64], min_cells: Int, who: String
) raises -> List[Float64]:
    """Check the arguments of `kuhn_grid`/`flat_torus`; the cell sizes."""
    var d = len(dims)
    if d == 0:
        raise Error(who, ": dims is empty")
    for a in range(d):
        if dims[a] < min_cells:
            raise Error(
                String(
                    who,
                    ": dims[",
                    a,
                    "] = ",
                    dims[a],
                    ", need at least ",
                    min_cells,
                    " cells per axis",
                )
            )
    if len(lengths) != d:
        raise Error(
            String(who, ": ", len(lengths), " lengths for ", d, " axes")
        )
    var h = List[Float64](capacity=d)
    for a in range(d):
        if not (lengths[a] > 0) or not isfinite(lengths[a]):
            raise Error(
                String(
                    who, ": lengths[", a, "] = ", lengths[a], " is not positive"
                )
            )
        h.append(lengths[a] / Float64(dims[a]))
    return h^


def _kuhn_elements(
    dims: List[Int], vdims: List[Int], periodic: Bool
) raises -> IntArray:
    """The Kuhn simplices of every cell; vertex `(i_0, ...)` is
    `sum i_a * prod_{b<a} vdims[b]`, coordinates taken modulo `dims` when
    `periodic`."""
    var d = len(dims)
    var stride = List[Int](length=d, fill=1)
    for a in range(1, d):
        stride[a] = stride[a - 1] * vdims[a - 1]
    var perms = List[List[Int]]()
    var odd = List[Bool]()
    _permutations(d, perms, odd)
    var ncells = 1
    for a in range(d):
        ncells *= dims[a]
    var elements = int_array(ncells * len(perms), d + 1)
    var corner = List[Int](length=d, fill=0)
    var pos = List[Int](length=d, fill=0)
    var row = 0
    for _ in range(ncells):
        for q in range(len(perms)):
            for a in range(d):
                pos[a] = corner[a]
            var ids = List[Int](capacity=d + 1)
            for k in range(d + 1):
                if k > 0:
                    pos[perms[q][k - 1]] += 1
                var v = 0
                for a in range(d):
                    var i = pos[a] % dims[a] if periodic else pos[a]
                    v += i * stride[a]
                ids.append(v)
            if odd[q]:  # det of the edge vectors is the permutation's sign
                ids.swap_elements(d - 1, d)
            for k in range(d + 1):
                iset(elements, row, k, ids[k])
            row += 1
        # the next cell, axis 0 fastest
        for a in range(d):
            corner[a] += 1
            if corner[a] < dims[a]:
                break
            corner[a] = 0
    return elements^


def _permutations(d: Int, mut perms: List[List[Int]], mut odd: List[Bool]):
    """All permutations of `0..d-1` in lexicographic order, with parities."""
    var p = List[Int](capacity=d)
    for i in range(d):
        p.append(i)
    while True:
        perms.append(p.copy())
        var inversions = 0
        for i in range(d):
            for j in range(i + 1, d):
                if p[i] > p[j]:
                    inversions += 1
        odd.append(inversions % 2 == 1)
        # next permutation
        var i = d - 2
        while i >= 0 and p[i] >= p[i + 1]:
            i -= 1
        if i < 0:
            return
        var j = d - 1
        while p[j] <= p[i]:
            j -= 1
        p.swap_elements(i, j)
        var lo = i + 1
        var hi = d - 1
        while lo < hi:
            p.swap_elements(lo, hi)
            lo += 1
            hi -= 1
