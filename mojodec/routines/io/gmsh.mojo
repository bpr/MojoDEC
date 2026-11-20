# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Gmsh meshes (`mojodec.routines.io.gmsh`).
=========================================
Reader for the `.msh` files of the Gmsh mesh generator (https://gmsh.info),
in format 4.1 (Gmsh's default) and the older 2.2, ASCII or binary. Not in
PyDEC.

The mesh is made of the highest-dimensional elements in the file: triangles
or tetrahedra (or line segments). Gmsh's physical groups, which name parts
of the geometry ("inlet", "walls", a material region), are kept as
`GmshGroup`s; `GmshMesh.subset` turns one into a `Subset` of the complex,
at its own dimension or any lower one (the vertices of a tagged curve, say).

```mojo
var msh = read_gmsh("channel.msh", dim=2)
var sc = SimplicialComplex(msh.vertices, msh.elements)
var inlet = msh.subset(sc, "inlet", 1)        # edges of the inlet curve
var fixed = msh.subset(sc, "walls", 0)        # vertices of the walls
```

With a `TypedComplex`, the dimension is a parameter and the result typed:

```mojo
var tc = TypedComplex[2](msh.mesh())
var walls = msh.subset[0](tc, "walls")        # TypedSubset[2, 0, PRIMAL]
```

Gmsh saves only the elements of physical groups once any group is
defined, so a `.geo` file that tags boundary curves must also tag the
surface (`Physical Surface("domain") = {1};`) or set `Mesh.SaveAll = 1;`,
or the triangles are missing. Passing the expected `dim` turns that into an
error instead of a mesh of line segments.
"""

from std.bit import pop_count
from std.memory import unsafe_memcpy
from std.sys import stderr

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import SimplicialMesh
from molina.parallel import num_chunks, parallel_ranges
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.subset import Subset
from mojodec.core.typed import PRIMAL, TypedComplex, TypedSubset
from mojodec.core.utils import (
    IntArray,
    RealArray,
    _lexsort_rows,
    cols_of,
    int_array,
    lexsort_rows,
    real_array,
    rows_of,
    sort_each_row,
)
from mojodec.routines.io.scan import ByteReader, NumberTokens, parse_numbers
from mojodec.routines.mesh_generation import (
    _numbering_spread,
    reorder_vertices as _reorder_mesh,
)

comptime _PARALLEL_BYTES = 1 << 18
"""ASCII sections of this many bytes or more are parsed in parallel."""

comptime _DENSE = 4
"""Node tags are looked up in a table when they span less than this many
times their number."""
comptime _PART_BITS = 9
"""`_unique_rows` splits the rows into parts by their smallest vertex
position, `2^_PART_BITS` positions per part, so that each part's rows fit
in the L2 cache."""
comptime _WARN_MIN_ELEMENTS = 50_000
"""`read_gmsh` checks the numbering of meshes with this many elements or
more; smaller ones build in a few ms whatever their numbering."""
comptime _WARN_SPREAD = 1.9
"""`read_gmsh` warns above this `numbering_spread`: Morton order gives
1.0–1.4, Gmsh's numbering 2.1–3.6, and a 21³ `kuhn_grid` in its own
order 1.75."""


struct GmshGroup(Copyable, Movable, Writable):
    """A Gmsh physical group: the cells of one dimension it tags."""

    var name: String
    """The group's name, or "" for a group known only by its tag."""
    var dim: Int
    """Dimension of the tagged cells."""
    var tag: Int
    """The physical tag, unique among the groups of one dimension."""
    var cells: IntArray
    """The cells, one row of `dim + 1` vertex indices each, in file order."""

    def __init__(
        out self, name: String, dim: Int, tag: Int, var cells: IntArray
    ):
        """A group; normally built by `read_gmsh`.

        Args:
            name: The group's name, or "" if it has none.
            dim: Dimension of the tagged cells.
            tag: The physical tag.
            cells: The cells, `num_cells x (dim + 1)` vertex indices.
        """
        self.name = name
        self.dim = dim
        self.tag = tag
        self.cells = cells^

    def num_cells(self) -> Int:
        """The number of cells in the group.

        Returns:
            The number of rows of `cells`.
        """
        return self.cells.size // (self.dim + 1)

    def simplices(self) raises -> List[Simplex]:
        """The cells as `Simplex`es.

        Returns:
            One `Simplex` per cell, in file order.

        Raises:
            If a cell repeats a vertex.
        """
        var w = self.dim + 1
        var out = List[Simplex](capacity=self.num_cells())
        for c in range(self.num_cells()):
            var v = List[Int](capacity=w)
            for j in range(w):
                v.append(Int(self.cells.unsafe_get(c * w + j)))
            out.append(Simplex(v))
        return out^

    def write_to(self, mut writer: Some[Writer]):
        """Write the name, dimension, tag, and cell count.

        Args:
            writer: The destination.
        """
        writer.write(
            "GmshGroup(",
            '"' + self.name + '"' if self.name else String("(unnamed)"),
            ", dim ",
            self.dim,
            ", tag ",
            self.tag,
            ", ",
            self.num_cells(),
            " cells)",
        )


struct GmshMesh(Movable, Writable):
    """A mesh read from a Gmsh `.msh` file, with its physical groups."""

    var vertices: RealArray
    """Vertex coordinates, `num_vertices x 3` (or `x 2`, see `read_gmsh`)."""
    var elements: IntArray
    """The top-dimensional simplices as rows of vertex indices, oriented as
    in the file."""
    var node_tags: List[Int]
    """The Gmsh node tag of each vertex."""
    var groups: List[GmshGroup]
    """The physical groups, ordered by dimension and tag."""

    def __init__(
        out self,
        var vertices: RealArray,
        var elements: IntArray,
        var node_tags: List[Int],
        var groups: List[GmshGroup],
    ):
        """Assemble a mesh from its parts; normally built by `read_gmsh`.

        Args:
            vertices: Vertex coordinates, one row per vertex.
            elements: The top-dimensional simplices as rows of vertex
                indices.
            node_tags: The Gmsh node tag of each vertex.
            groups: The physical groups.
        """
        self.vertices = vertices^
        self.elements = elements^
        self.node_tags = node_tags^
        self.groups = groups^

    def mesh(self) raises -> SimplicialMesh:
        """The mesh, for `SimplicialComplex(msh.mesh())`.

        Returns:
            A `SimplicialMesh` of `vertices` and `elements`.

        Raises:
            If `SimplicialMesh` rejects the arrays.
        """
        return SimplicialMesh(self.vertices, self.elements)

    def reorder_vertices(self) raises -> GmshMesh:
        """This mesh with its vertices renumbered in Morton order and its
        elements sorted to match (`reorder_vertices` of `mesh()`), and the
        node tags and physical groups carried along.

        Gmsh numbers nodes by the geometric entity they lie on, which can
        scatter neighbors; renumbering makes building the complex and its
        Whitney matrices faster (see the guide's vertex numbering page).
        Each vertex keeps its node tag, so `node_tags` still names it in
        the file, and each group keeps its cells in file order.

        ```mojo
        var msh = read_gmsh("cube.msh").reorder_vertices()
        var sc = SimplicialComplex(msh.mesh())
        var walls = msh.subset(sc, "walls", 2)
        ```

        Returns:
            The renumbered mesh.

        Raises:
            As `reorder_vertices`.
        """
        var r = _reorder_mesh(self.mesh())
        var tags = List[Int](capacity=len(r.old_vertex))
        for i in range(len(r.old_vertex)):
            tags.append(self.node_tags[r.old_vertex[i]])
        var groups = List[GmshGroup](capacity=len(self.groups))
        for ref g in self.groups:
            var cells = g.cells.copy()
            for i in range(cells.size):
                cells.unsafe_set(
                    i,
                    Scalar[DType.int](r.new_vertex[Int(cells.unsafe_get(i))]),
                )
            groups.append(GmshGroup(g.name, g.dim, g.tag, cells^))
        return GmshMesh(
            r.mesh.vertices.copy(), r.mesh.elements.copy(), tags^, groups^
        )

    def group_names(self) -> List[String]:
        """The names of the named groups.

        Returns:
            The non-empty names, ordered by dimension and tag; a name used
            in several dimensions appears once per dimension.
        """
        var out = List[String]()
        for ref g in self.groups:
            if g.name:
                out.append(g.name)
        return out^

    def group(self, name: String) raises -> GmshGroup:
        """The group with this name (the lowest-dimensional one if several
        dimensions share it).

        Args:
            name: A physical group's name.

        Returns:
            A copy of the group.

        Raises:
            If no group has that name; the message lists the names.
        """
        for ref g in self.groups:
            if g.name == name:
                return g.copy()
        raise Error(self._unknown(name))

    def subset(
        self, sc: SimplicialComplex, name: String, dim: Int
    ) raises -> Subset:
        """The primal `dim`-simplices of the named group: its cells if they
        have dimension `dim`, else all their `dim`-dimensional faces.

        Groups of this name in several dimensions contribute together; those
        of dimension below `dim` are skipped.

        Args:
            sc: The complex built from `self.mesh()` (or from `vertices`
                and `elements`).
            name: A physical group's name.
            dim: The dimension of the simplices to return.

        Returns:
            A primal `Subset` of `sc`'s `dim`-simplices.

        Raises:
            If no group has that name, none of that name has cells of
            dimension >= `dim`, `dim` is negative, or a cell is not a
            simplex of `sc`.
        """
        var out = Subset.empty(sc, dim)
        var found = False
        for ref g in self.groups:
            if g.name == name and g.dim >= dim:
                out = out | _group_subset(sc, g, dim)
                found = True
        if not found:
            for ref g in self.groups:
                if g.name == name:
                    raise Error(
                        String(
                            "GmshMesh.subset: group '",
                            name,
                            "' has no cells of dimension >= ",
                            dim,
                        )
                    )
            raise Error(self._unknown(name))
        return out^

    def subset[
        n: Int, //, k: Int
    ](self, tc: TypedComplex[n], name: String) raises -> TypedSubset[
        n, k, PRIMAL
    ]:
        """The primal k-simplices of the named group, typed; see the
        untyped overload.

        Parameters:
            n: The complex dimension (inferred).
            k: The dimension of the simplices to return, `0..n`.

        Args:
            tc: The complex built from `self.mesh()`.
            name: A physical group's name.

        Returns:
            The k-simplices of the group, or the k-faces of its cells.

        Raises:
            As the untyped overload.
        """
        comptime assert 0 <= k and k <= n, "GmshMesh.subset: k out of range"
        return TypedSubset[n, k, PRIMAL](tc.sc, self.subset(tc.sc, name, k))

    def subset[
        n: Int, //, k: Int
    ](
        self, tc: TypedComplex[n], *, group_dim: Int, tag: Int
    ) raises -> TypedSubset[n, k, PRIMAL]:
        """The primal k-simplices of the group with this dimension and
        physical tag, typed; see the untyped overload.

        Parameters:
            n: The complex dimension (inferred).
            k: The dimension of the simplices to return, `0..n`.

        Args:
            tc: The complex built from `self.mesh()`.
            group_dim: The group's dimension.
            tag: The group's physical tag.

        Returns:
            The k-simplices of the group, or the k-faces of its cells.

        Raises:
            As the untyped overload.
        """
        comptime assert 0 <= k and k <= n, "GmshMesh.subset: k out of range"
        return TypedSubset[n, k, PRIMAL](
            tc.sc, self.subset(tc.sc, group_dim=group_dim, tag=tag, dim=k)
        )

    def subset(
        self, sc: SimplicialComplex, *, group_dim: Int, tag: Int, dim: Int
    ) raises -> Subset:
        """The primal `dim`-simplices of the group with this dimension and
        physical tag (for unnamed groups).

        ```mojo
        var g = msh.subset(sc, group_dim=1, tag=42, dim=0)
        ```

        Args:
            sc: The complex built from `self.mesh()`.
            group_dim: The group's dimension.
            tag: The group's physical tag.
            dim: The dimension of the simplices to return, at most
                `group_dim`.

        Returns:
            A primal `Subset` of `sc`'s `dim`-simplices: the group's cells
            or their `dim`-faces.

        Raises:
            If there is no such group, `dim` is not in `0..group_dim`, or a
            cell is not a simplex of `sc`.
        """
        for ref g in self.groups:
            if g.dim == group_dim and g.tag == tag:
                return _group_subset(sc, g, dim)
        raise Error(
            String(
                "GmshMesh.subset: no physical group of dimension ",
                group_dim,
                " with tag ",
                tag,
            )
        )

    def write_to(self, mut writer: Some[Writer]):
        """Write the vertex and element counts, then one line per group.

        Args:
            writer: The destination.
        """
        writer.write(
            "GmshMesh(",
            len(self.node_tags),
            " vertices, ",
            rows_of(self.elements),
            " elements of dimension ",
            cols_of(self.elements) - 1,
        )
        for ref g in self.groups:
            writer.write("\n  ", g)
        writer.write(")")

    def _unknown(self, name: String) -> String:
        var s = String("GmshMesh: no physical group named '", name, "'; have")
        var names = self.group_names()
        if len(names) == 0:
            s += " none"
        for i in range(len(names)):
            s += String(" '" if i == 0 else ", '", names[i], "'")
        return s


def _group_subset(
    sc: SimplicialComplex, g: GmshGroup, dim: Int
) raises -> Subset:
    if dim < 0 or dim > g.dim:
        raise Error(
            String(
                "GmshMesh.subset: dimension ",
                dim,
                " is not in 0..",
                g.dim,
                " for this group",
            )
        )
    var s = Subset.empty(sc, g.dim)
    if g.num_cells() > 0:
        s = Subset.from_simplices(sc, g.simplices())
    for _ in range(g.dim - dim):
        s = s.boundary(sc)
    return s^


def read_gmsh(
    path: String,
    dim: Optional[Int] = None,
    drop_z: Bool = True,
    warn: Bool = True,
) raises -> GmshMesh:
    """Read a Gmsh `.msh` file (format 4.1 or 2.2, ASCII or binary).

    Only vertices of the top-dimensional simplices are kept, numbered in
    the order of their node tags (`node_tags` maps back). Points, lines,
    triangles, and tetrahedra are read; other element types (quadrangles,
    hexahedra, second-order elements) raise. Elements repeated because they
    belong to several physical groups (format 2.2) are kept once.

    Gmsh numbers nodes in an order that scatters neighbors, and a mesh
    numbered that way builds 1.3–2.2× slower than after
    `GmshMesh.reorder_vertices` (see the guide's vertex numbering page).
    So for a mesh of 50,000 elements or more whose `numbering_spread` is
    above 1.9, `read_gmsh` prints a warning to stderr that suggests it.

    ```mojo
    var msh = read_gmsh("square.msh", dim=2)
    var sc = SimplicialComplex(msh.mesh())
    ```

    Args:
        path: The file.
        dim: The expected dimension of the mesh (`None`, the default,
            accepts any). Pass it: once a `.geo` file defines physical groups, Gmsh
            saves only their elements, so a file that tags only boundary
            curves holds no triangles and would otherwise read as a mesh of
            line segments (tag the domain too, or set `Mesh.SaveAll = 1`).
        drop_z: Return two coordinate columns when the mesh has dimension at
            most 2 and every z is zero (Gmsh always writes three).
        warn: Warn on stderr when a large mesh is badly numbered (pass
            `False` when the mesh will be reordered or speed does not
            matter).

    Returns:
        The mesh and its physical groups.

    Raises:
        If the file cannot be opened; has no `$MeshFormat` section, or a
        version other than 4.1 or 2.x; is binary 4.1 with a `size_t` other
        than 8 bytes or a bad byte-order marker; is malformed (truncated,
        a missing `$End` line, an element type that does not match its
        entity); has no elements or an unsupported element type; has top
        elements of a dimension other than `dim`; or has an element or a
        physical group that uses a node that is undefined or not on a
        top-dimensional simplex.
    """
    var raw = _parse(path)
    # the top dimension
    var top = -1
    for ref b in raw.blocks:
        top = max(top, b.dim)
    if top < 0:
        raise Error("read_gmsh: no elements in " + path)
    if dim and top != dim.value():
        raise Error(
            String(
                "read_gmsh: expected a ",
                dim.value(),
                "-dimensional mesh but the highest elements in ",
                path,
                " have dimension ",
                top,
                " (when a .geo file defines physical groups, Gmsh saves",
                " only their elements: tag the domain too, or set",
                " Mesh.SaveAll = 1)",
            )
        )
    var w = top + 1
    var nodes = _TagIndex(raw.node_tags)
    var span = nodes.span
    # the top blocks, and where each starts among their nodes
    var tops = List[Int]()
    var starts = List[Int]()
    var N = 0
    for k in range(len(raw.blocks)):
        if raw.blocks[k].dim == top:
            tops.append(k)
            starts.append(N)
            N += len(raw.blocks[k].nodes)
    starts.append(N)
    # each node of the top elements as its position; per chunk, a bitmap
    # of the positions it uses and its first unknown tag
    comptime SERIAL_BELOW = 50_000
    var C = num_chunks(N, SERIAL_BELOW)
    var words = (span + 63) // 64
    var stride = words + 16  # no two chunks write one cache line
    var marks = List[UInt64](length=C * stride, fill=0)
    var bad = List[Int](length=C, fill=-1)
    var ats = List[Int](unsafe_uninit_length=N)
    var bp = raw.blocks.unsafe_ptr()
    var kp = tops.unsafe_ptr()
    var sp = starts.unsafe_ptr()
    var ap = ats.unsafe_ptr()
    var mp = marks.unsafe_ptr()
    var xp = bad.unsafe_ptr()
    var ip = Pointer(to=nodes)
    var T = len(tops)

    def lookup(
        chunk: Int, lo: Int, hi: Int
    ) {bp, kp, sp, ap, mp, xp, ip, T, stride}:
        # the top block holding entry lo: the last one starting at or
        # before it
        var a = 0
        var b = T
        while b - a > 1:
            var mid = (a + b) // 2
            if sp[unsafe_offset=mid] <= lo:
                a = mid
            else:
                b = mid
        var base = chunk * stride
        var i = lo
        while i < hi:
            var off = sp[unsafe_offset=a]
            var end = min(hi, sp[unsafe_offset=a + 1])
            var tp = bp[unsafe_offset=kp[unsafe_offset=a]].nodes.unsafe_ptr()
            for g in range(i, end):
                var p = ip[].find(tp[unsafe_offset=g - off])
                if p < 0:
                    xp[unsafe_offset=chunk] = g
                    return
                ap[unsafe_offset=g] = p
                mp[unsafe_offset=base + (p >> 6)] |= UInt64(1) << UInt64(p & 63)
            i = end
            a += 1

    parallel_ranges(N, SERIAL_BELOW, lookup)
    for k in range(C):
        var g = bad[k]
        if g >= 0:
            var a = 0
            while starts[a + 1] <= g:
                a += 1
            var t = raw.blocks[tops[a]].nodes[g - starts[a]]
            raise Error(String("read_gmsh: element uses unknown node ", t))
    _ = tops^
    _ = starts^
    # without the repeats that format 2.2 writes for elements in several
    # physical groups (equal positions are equal tags)
    var rows = _unique_rows(ats, w, span)
    # the positions used (the repeats use the same ones), and how many
    # come before each word of the bitmap
    var used = List[UInt64](length=words, fill=0)
    var rank = List[Int](length=words, fill=0)
    var nv = 0
    for wd in range(words):
        var bits = UInt64(0)
        for k in range(C):
            bits |= marks.unsafe_get(k * stride + wd)
        used.unsafe_set(wd, bits)
        rank.unsafe_set(wd, nv)
        nv += Int(pop_count(bits))
    _ = marks^
    # number them in tag order
    var vertex = List[Int](unsafe_uninit_length=span)
    var node_tags = List[Int](unsafe_uninit_length=nv)
    var slots = List[Int](unsafe_uninit_length=nv)
    var up = used.unsafe_ptr()
    var rp = rank.unsafe_ptr()
    var vp = vertex.unsafe_ptr()
    var np = node_tags.unsafe_ptr()
    var lp = slots.unsafe_ptr()

    def number(chunk: Int, lo: Int, hi: Int) {up, rp, vp, np, lp, ip, span}:
        for wd in range(lo, hi):
            var v = rp[unsafe_offset=wd]
            var bits = up[unsafe_offset=wd]
            for p in range(64 * wd, min(64 * wd + 64, span)):
                if (bits >> UInt64(p & 63)) & 1:
                    vp[unsafe_offset=p] = v
                    np[unsafe_offset=v] = ip[].tag(p)
                    lp[unsafe_offset=v] = ip[].first.unsafe_get(p)
                    v += 1
                else:
                    vp[unsafe_offset=p] = -1

    parallel_ranges(words, SERIAL_BELOW // 64, number)
    _ = used^
    _ = rank^
    var ne = len(rows)
    var elements = int_array(ne, w)
    var ep = elements.unsafe_ptr()
    var op = rows.unsafe_ptr()

    def fill(chunk: Int, lo: Int, hi: Int) {ep, op, ap, vp, w}:
        for i in range(lo, hi):
            var r = op[unsafe_offset=i]
            for j in range(w):
                ep[unsafe_offset=i * w + j] = Scalar[DType.int](
                    vp[unsafe_offset=ap[unsafe_offset=r * w + j]]
                )

    parallel_ranges(ne, SERIAL_BELOW, fill)
    _ = rows^
    _ = ats^
    # coordinates, from the first node with each tag
    var cp = raw.coords.unsafe_ptr()
    var lifted = List[Bool](length=num_chunks(nv, SERIAL_BELOW), fill=False)
    var fp = lifted.unsafe_ptr()

    def any_z(chunk: Int, lo: Int, hi: Int) {cp, lp, fp}:
        for i in range(lo, hi):
            if cp[unsafe_offset=3 * lp[unsafe_offset=i] + 2] != 0:
                fp[unsafe_offset=chunk] = True
                return

    parallel_ranges(nv, SERIAL_BELOW, any_z)
    var flat = True
    for z in lifted:
        if z:
            flat = False
    var e = 2 if drop_z and top <= 2 and flat else 3
    var vertices = real_array(nv, e)
    var cv = vertices.unsafe_ptr()

    def copy(chunk: Int, lo: Int, hi: Int) {cp, lp, cv, e}:
        for i in range(lo, hi):
            var at = 3 * lp[unsafe_offset=i]
            for c in range(e):
                cv[unsafe_offset=i * e + c] = cp[unsafe_offset=at + c]

    parallel_ranges(nv, SERIAL_BELOW, copy)
    _ = slots^
    # physical groups, ordered by dimension and tag
    var groups = List[GmshGroup]()
    for d in range(top + 1):
        var tags = List[Int]()
        for ref b in raw.blocks:
            if b.dim == d:
                for p in b.physical:
                    if p not in tags:
                        tags.append(p)
        for ref n in raw.names:
            if n.dim == d and n.tag not in tags:
                tags.append(n.tag)
        sort(tags)
        for p in tags:
            var cells = List[Int]()
            for ref b in raw.blocks:
                if b.dim == d and p in b.physical:
                    _reserve_more(cells, len(b.nodes))
                    for t in b.nodes:
                        var at = nodes.find(t)
                        var v = vertex.unsafe_get(at) if at >= 0 else -1
                        if v < 0:
                            raise Error(
                                String(
                                    "read_gmsh: physical group ",
                                    p,
                                    " uses node ",
                                    t,
                                    ", which no ",
                                    top,
                                    "-simplex has",
                                )
                            )
                        cells.append(v)
            var array = int_array(len(cells) // (d + 1), d + 1)
            for i in range(len(cells)):
                array.unsafe_set(i, Scalar[DType.int](cells.unsafe_get(i)))
            groups.append(GmshGroup(raw.name_of(d, p), d, p, array^))
    if warn:
        var message = _numbering_warning(path, elements, nv)
        if message:
            print(message, file=stderr)
    return GmshMesh(vertices^, elements^, node_tags^, groups^)


def _numbering_warning(
    path: String, elements: IntArray, nv: Int
) raises -> String:
    """`read_gmsh`'s warning about a badly numbered mesh, or an empty
    string when the mesh is small or well numbered."""
    if rows_of(elements) < _WARN_MIN_ELEMENTS:
        return ""
    var spread = _numbering_spread(elements, nv)
    if spread <= _WARN_SPREAD:
        return ""
    return String(
        "read_gmsh: warning: the vertex numbering of ",
        path,
        " is scattered (numbering_spread ",
        round(spread, 2),
        "), so building a complex on it can take about twice as long;",
        " call .reorder_vertices() on the result, or pass warn=False",
    )


def _unique_rows(nodes: List[Int], w: Int, span: Int) raises -> List[Int]:
    """Indices of the rows of `nodes` (width `w`, entries in `[0, span)`)
    to keep: the first of each set of rows with the same vertices, in file
    order.

    Equal rows have the same smallest entry and the same others. Each row
    is written once, as its other entries sorted and packed into one
    integer, into the part of its smallest entry (`_PART_BITS`); then each
    part, small enough for the L2 cache, is searched for equal rows with a
    hash table, in parallel over the parts. Sorting the rows instead
    (`_unique_rows_sorted`) took 3-4 times as long: Gmsh scatters vertex
    numbers, so each thread of its counting sort touches every bucket. Only
    format 2.2 repeats elements (those in several physical groups)."""
    var n = len(nodes) // w
    var bits = 0
    while (1 << bits) < span:
        bits += 1
    if w > 4 or (w - 1) * bits > 63:
        return _unique_rows_sorted(nodes, w)
    comptime B = _PART_BITS
    comptime W = 1 << B
    var P = ((span - 1) >> B) + 1 if span > 0 else 1
    comptime SERIAL_BELOW = 50_000
    var C = num_chunks(n, SERIAL_BELOW)
    # rows per (chunk, part), then where each chunk writes its rows of a
    # part: parts in order, a part's rows in file order
    var counts = List[Int](length=C * P, fill=0)
    var np = nodes.unsafe_ptr()
    var cp = counts.unsafe_ptr()

    def count(chunk: Int, lo: Int, hi: Int) {np, cp, w, P}:
        var base = chunk * P
        for r in range(lo, hi):
            var v = np[unsafe_offset=r * w]
            for j in range(1, w):
                v = min(v, np[unsafe_offset=r * w + j])
            cp[unsafe_offset=base + (v >> B)] += 1

    parallel_ranges(n, SERIAL_BELOW, count)
    var part = List[Int](length=P + 1, fill=0)
    var total = 0
    for p in range(P):
        part[p] = total
        for k in range(C):
            var x = counts[k * P + p]
            counts[k * P + p] = total
            total += x
    part[P] = total
    # per row: its other entries packed, and its index and smallest entry
    # within the part
    var tails = List[Int](unsafe_uninit_length=n)
    var ids = List[Int](unsafe_uninit_length=n)
    var tp = tails.unsafe_ptr()
    var ip = ids.unsafe_ptr()

    def scatter(chunk: Int, lo: Int, hi: Int) {np, cp, tp, ip, w, bits, P}:
        var base = chunk * P
        for r in range(lo, hi):
            # a sorting network on four entries (the missing ones last)
            var x0 = np[unsafe_offset=r * w]
            var x1 = np[unsafe_offset=r * w + 1] if w > 1 else Int.MAX
            var x2 = np[unsafe_offset=r * w + 2] if w > 2 else Int.MAX
            var x3 = np[unsafe_offset=r * w + 3] if w > 3 else Int.MAX
            _order(x0, x1)
            _order(x2, x3)
            _order(x0, x2)
            _order(x1, x3)
            _order(x1, x2)
            var tail = 0
            if w > 1:
                tail = x1
            if w > 2:
                tail = (tail << bits) | x2
            if w > 3:
                tail = (tail << bits) | x3
            var at = cp[unsafe_offset=base + (x0 >> B)]
            cp[unsafe_offset=base + (x0 >> B)] = at + 1
            tp[unsafe_offset=at] = tail
            ip[unsafe_offset=at] = (r << B) | (x0 & (W - 1))

    parallel_ranges(n, SERIAL_BELOW, scatter)
    _ = counts^
    # threads take runs of parts with about equal numbers of rows: the
    # smallest of a row's scattered entries is usually small, so the first
    # parts hold several times the average
    var bound = List[Int](length=C + 1, fill=P)
    var p = 0
    for k in range(C):
        while part[p] < k * n // C:
            p += 1
        bound[k] = p
    # in each part, a hash table of the distinct rows so far, chained; a
    # row equal to one on its chain is dropped
    var keep = List[Bool](length=n, fill=True)
    var dropped = List[Int](length=C, fill=0)
    var pp = part.unsafe_ptr()
    var bp = bound.unsafe_ptr()
    var ep = keep.unsafe_ptr()
    var dp = dropped.unsafe_ptr()

    def search(chunk: Int, k0: Int, k1: Int) {tp, ip, pp, bp, ep, dp}:
        var head = List[Int32]()
        var next = List[Int32]()
        var drops = 0
        for p in range(bp[unsafe_offset=k0], bp[unsafe_offset=k1]):
            var b0 = pp[unsafe_offset=p]
            var b1 = pp[unsafe_offset=p + 1]
            var m = b1 - b0
            if m > len(next):
                next.resize(m, 0)
            # at least twice as many slots as rows
            var shift = 60
            while (1 << (64 - shift)) < 2 * m:
                shift -= 1
            var slots = 1 << (64 - shift)
            if slots > len(head):
                head.resize(slots, 0)
            for i in range(slots):
                head.unsafe_set(i, -1)
            for e in range(b0, b1):
                var id = ip[unsafe_offset=e]
                var tail = tp[unsafe_offset=e]
                var h = (
                    UInt64(tail) * 0x9E3779B97F4A7C15 + UInt64(id & (W - 1))
                ) * 0xBF58476D1CE4E5B9
                var slot = Int(h >> UInt64(shift))
                var f = Int(head.unsafe_get(slot))
                var same = False
                while f >= 0:
                    if (
                        tp[unsafe_offset=b0 + f] == tail
                        and (ip[unsafe_offset=b0 + f] ^ id) & (W - 1) == 0
                    ):
                        same = True
                        break
                    f = Int(next.unsafe_get(f))
                if same:
                    ep[unsafe_offset=id >> B] = False
                    drops += 1
                else:
                    next.unsafe_set(e - b0, head.unsafe_get(slot))
                    head.unsafe_set(slot, Int32(e - b0))
        dp[unsafe_offset=chunk] = drops

    parallel_ranges(C, 1, search)
    var drops = 0
    for k in range(C):
        drops += dropped[k]
    var out = List[Int](capacity=n - drops)
    for r in range(n):
        if drops == 0 or keep.unsafe_get(r):
            out.append(r)
    return out^


@always_inline
def _order(mut x: Int, mut y: Int):
    """Order `x <= y`, by min and max."""
    var lo = min(x, y)
    y = max(x, y)
    x = lo


def _unique_rows_sorted(nodes: List[Int], w: Int) raises -> List[Int]:
    """`_unique_rows` by sorting: the rows, each sorted, are sorted
    lexicographically, so that equal ones are neighbors; when the sort
    found none, every row is kept without looking at the order."""
    var n = len(nodes) // w
    var keys = int_array(n, w)
    var np = nodes.unsafe_ptr().unsafe_bitcast[Scalar[DType.int]]()
    var kp = keys.unsafe_ptr()
    comptime SERIAL_BELOW = 50_000

    def copy(chunk: Int, lo: Int, hi: Int) {np, kp, w}:
        unsafe_memcpy(
            dest=kp.unsafe_offset(lo * w),
            src=np.unsafe_offset(lo * w),
            count=(hi - lo) * w,
        )

    parallel_ranges(n, SERIAL_BELOW, copy, memory_bound=True)
    sort_each_row(keys)
    var duplicate = 0
    var order = _lexsort_rows(keys, duplicate)
    var out = List[Int](capacity=n)
    if duplicate == -1:
        for r in range(n):
            out.append(r)
        return out^
    # each run of equal rows in sorted order keeps its first, which is its
    # first in file order (the sort is stable); a chunk takes the runs that
    # start in it
    var keep = List[Bool](length=n, fill=True)
    var op = order.unsafe_ptr()
    var ep = keep.unsafe_ptr()

    @always_inline
    def same(s: Int, t: Int) {kp, op, w} -> Bool:
        var a = op[unsafe_offset=s] * w
        var b = op[unsafe_offset=t] * w
        for j in range(w):
            if kp[unsafe_offset=a + j] != kp[unsafe_offset=b + j]:
                return False
        return True

    def runs(chunk: Int, lo: Int, hi: Int) {op, ep, n, same}:
        var i = lo
        while i > 0 and i < hi and same(i - 1, i):
            i += 1
        while i < hi:
            var j = i + 1
            while j < n and same(i, j):
                ep[unsafe_offset=op[unsafe_offset=j]] = False
                j += 1
            i = j

    parallel_ranges(n, SERIAL_BELOW, runs)
    for r in range(n):
        if keep.unsafe_get(r):
            out.append(r)
    return out^


# ===----------------------------------------------------------------------=== #
# Parsing
# ===----------------------------------------------------------------------=== #


struct _Block(Copyable, Movable):
    """Elements of one type and the same physical groups."""

    var dim: Int
    var type: Int
    var physical: List[Int]
    var nodes: List[Int]
    """Node tags, `nodes_per_element(type)` per element."""

    def __init__(out self, type: Int, var physical: List[Int]) raises:
        self.type = type
        self.dim = _element_dim(type)
        self.physical = physical^
        self.nodes = List[Int]()


struct _Name(Copyable, Movable):
    var dim: Int
    var tag: Int
    var name: String

    def __init__(out self, dim: Int, tag: Int, name: String):
        self.dim = dim
        self.tag = tag
        self.name = name


struct _TagIndex(Movable):
    """Each node tag of a file as a position in `[0, span)`, increasing with
    the tag. Dense tags (Gmsh numbers nodes 1 to n) are their own
    positions, less the smallest; others are numbered in sorted order and
    looked up in a hash table (bisection took 120 ns a lookup, 160 ms for
    a 438k-tetrahedron file)."""

    var dense: Bool
    """Whether the tags span less than `_DENSE` times their number, so
    that a table of the span is bounded by the file's size."""
    var lo: Int
    """The smallest tag."""
    var span: Int
    """The number of positions."""
    var first: List[Int]
    """Per position: the index in the file of the first node with its
    tag, or -1 (a gap in dense tags)."""
    var tags: List[Int]
    """Sparse tags: the tag at each position (the distinct tags, sorted);
    empty when dense."""
    var hash: List[Int]
    """Sparse tags: open addressing with linear probing, a (tag,
    position) pair per slot, tag -1 when empty; at least twice as many
    slots as tags."""
    var shift: Int
    """Sparse tags: a tag's first slot is the top `64 - shift` bits of
    its product with the golden ratio constant."""

    def __init__(out self, file_tags: List[Int]) raises:
        var n = len(file_tags)
        self.lo = Int.MAX if n > 0 else 0
        var hi = -1
        for i in range(n):
            var t = file_tags.unsafe_get(i)
            if t < 0:
                raise Error(String("read_gmsh: negative node tag ", t))
            self.lo = min(self.lo, t)
            hi = max(hi, t)
        self.dense = hi - self.lo < _DENSE * n
        self.tags = List[Int]()
        self.hash = List[Int]()
        self.shift = 63
        if self.dense:
            self.span = hi - self.lo + 1
            self.first = List[Int](length=self.span, fill=-1)
            for i in range(n - 1, -1, -1):
                self.first.unsafe_set(file_tags.unsafe_get(i) - self.lo, i)
            return
        # sparse: the distinct tags in order (a stable sort, so each
        # tag's first node comes first)
        var column = int_array(n, 1)
        for i in range(n):
            column.unsafe_set(i, Scalar[DType.int](file_tags.unsafe_get(i)))
        var order = lexsort_rows(column)
        self.first = List[Int](capacity=n)
        self.tags = List[Int](capacity=n)
        for i in order:
            var t = file_tags.unsafe_get(i)
            if not self.tags or self.tags[len(self.tags) - 1] != t:
                self.tags.append(t)
                self.first.append(i)
        self.span = len(self.tags)
        while (1 << (64 - self.shift)) < 2 * self.span:
            self.shift -= 1
        var mask = (1 << (64 - self.shift)) - 1
        self.hash = List[Int](length=2 * (mask + 1), fill=-1)
        for p in range(self.span):
            var t = self.tags.unsafe_get(p)
            var h = self._slot(t)
            while self.hash.unsafe_get(2 * h) >= 0:
                h = (h + 1) & mask
            self.hash.unsafe_set(2 * h, t)
            self.hash.unsafe_set(2 * h + 1, p)

    @always_inline
    def _slot(self, t: Int) -> Int:
        return Int((UInt64(t) * 0x9E3779B97F4A7C15) >> UInt64(self.shift))

    @always_inline
    def find(self, t: Int) -> Int:
        """The position of tag `t`, or -1 when no node has it."""
        if t < 0:
            return -1
        if self.dense:
            var p = t - self.lo
            if UInt(p) < UInt(self.span) and self.first.unsafe_get(p) >= 0:
                return p
            return -1
        var mask = (1 << (64 - self.shift)) - 1
        var h = self._slot(t)
        while True:
            var k = self.hash.unsafe_get(2 * h)
            if k == t:
                return self.hash.unsafe_get(2 * h + 1)
            if k < 0:
                return -1
            h = (h + 1) & mask

    @always_inline
    def tag(self, p: Int) -> Int:
        """The tag at position `p`."""
        return self.lo + p if self.dense else self.tags.unsafe_get(p)


struct _Raw(Movable):
    var node_tags: List[Int]
    var coords: List[Float64]
    """Three per node."""
    var blocks: List[_Block]
    var names: List[_Name]

    def __init__(out self):
        self.node_tags = List[Int]()
        self.coords = List[Float64]()
        self.blocks = List[_Block]()
        self.names = List[_Name]()

    def name_of(self, dim: Int, tag: Int) -> String:
        for ref n in self.names:
            if n.dim == dim and n.tag == tag:
                return n.name
        return ""


def _parse(path: String, parallel_bytes: Int = _PARALLEL_BYTES) raises -> _Raw:
    """The file's sections; ASCII `$Nodes` and `$Elements` sections of
    `parallel_bytes` or more are parsed ahead in parallel (tests lower it
    to check that path on small files)."""
    var cur = _Cursor(ByteReader.open(path, "read_gmsh", " in " + path))
    var raw = _Raw()
    var version = 0
    # physical tags of each entity (format 4.1), per entity dimension
    var entity_tags = List[List[Int]]()
    var entity_phys = List[List[List[Int]]]()
    for _ in range(4):
        entity_tags.append(List[Int]())
        entity_phys.append(List[List[Int]]())
    while cur.more():
        var section = cur.section()
        if section == "MeshFormat":
            var v = cur.token()
            var file_type = cur.int_()
            var size = cur.int_()
            if v.startswith("4.1"):
                version = 4
            elif v.startswith("2."):
                version = 2
            else:
                raise cur.err(
                    "format " + v + " is not supported (save as 4.1 or 2.2)"
                )
            _ = cur.line()
            if file_type == 1:
                if version == 4 and size != 8:
                    raise cur.err("binary files need 8-byte size_t")
                cur.start_binary()
        elif section == "PhysicalNames":
            var n = cur.int_()
            for _ in range(n):
                var d = cur.int_()
                var t = cur.int_()
                raw.names.append(_Name(d, t, cur.quoted()))
        elif version == 0:
            raise cur.err("$" + section + " before $MeshFormat")
        elif section == "Entities" and version == 4:
            _read_entities(cur, entity_tags, entity_phys)
        elif section == "Nodes" or section == "Elements":
            _read_section(
                cur,
                raw,
                section,
                version,
                entity_tags,
                entity_phys,
                parallel_bytes,
            )
        else:
            cur.skip_section(section)
            continue
        cur.end(section)
    if version == 0:
        raise cur.err("no $MeshFormat section")
    return raw^


def _read_section(
    mut cur: _Cursor,
    mut raw: _Raw,
    section: String,
    version: Int,
    entity_tags: List[List[Int]],
    entity_phys: List[List[List[Int]]],
    parallel_bytes: Int,
) raises:
    """Read a `$Nodes` or `$Elements` section, a large ASCII one from
    numbers parsed ahead, in parallel. Errors are the serial reader's."""
    var start = cur.r.pos
    var end = -1 if cur.binary else cur.tokenize(parallel_bytes)
    if end < 0:
        _read_section_body(cur, raw, section, version, entity_tags, entity_phys)
        return
    try:
        _read_section_body(cur, raw, section, version, entity_tags, entity_phys)
    except e:
        # the serial reader raises too, naming the token
        cur.untokenize()
        cur.r.pos = start
        _read_section_body(cur, raw, section, version, entity_tags, entity_phys)
        raise e^
    var used = cur.tokens.next
    var left = not cur.tokens.done()
    cur.untokenize()
    if left:
        # numbers the section does not use: the serial reader would stop
        # before them, and the $End line check fails there
        cur.r.pos = start
        cur.r.skip_tokens(used)
    else:
        cur.r.pos = end


def _read_section_body(
    mut cur: _Cursor,
    mut raw: _Raw,
    section: String,
    version: Int,
    entity_tags: List[List[Int]],
    entity_phys: List[List[List[Int]]],
) raises:
    if section == "Nodes":
        if version == 4:
            _read_nodes_v4(cur, raw)
        else:
            _read_nodes_v2(cur, raw)
    elif version == 4:
        _read_elements_v4(cur, raw, entity_tags, entity_phys)
    else:
        _read_elements_v2(cur, raw)


def _read_entities(
    mut cur: _Cursor,
    mut entity_tags: List[List[Int]],
    mut entity_phys: List[List[List[Int]]],
) raises:
    var counts = List[Int]()
    for _ in range(4):
        counts.append(cur.size_t())
    for d in range(4):
        for _ in range(counts[d]):
            var tag = cur.i32()
            for _ in range(3 if d == 0 else 6):
                _ = cur.f64()
            var phys = List[Int]()
            for _ in range(cur.size_t()):
                phys.append(cur.i32())
            if d > 0:
                for _ in range(cur.size_t()):
                    _ = cur.i32()
            entity_tags[d].append(tag)
            entity_phys[d].append(phys^)


def _read_nodes_v4(mut cur: _Cursor, mut raw: _Raw) raises:
    var num_blocks = cur.size_t()
    _ = cur.size_t()
    _ = cur.size_t()
    _ = cur.size_t()
    for _ in range(num_blocks):
        var d = cur.i32()
        _ = cur.i32()
        var parametric = cur.i32()
        var n = cur.size_t()
        if cur.binary:
            # the tags, then the coordinates, as contiguous blocks
            cur.r.require_items(n, 8)
            var tags = nm.zeros[DType.uint64](Shape(n))
            cur.r.binary_into(tags, 0, n)
            for i in range(n):
                raw.node_tags.append(Int(tags.unsafe_get(i)))
            if parametric == 0:
                cur.r.require_items(n, 24)
                var xyz = nm.zeros[DType.float64](Shape(3 * n))
                cur.r.binary_into(xyz, 0, 3 * n)
                _reserve_more(raw.coords, 3 * n)
                for i in range(3 * n):
                    raw.coords.append(xyz.unsafe_get(i))
                continue
        else:
            cur.ints(raw.node_tags, n, 0, 1)
            if parametric == 0:
                cur.floats(raw.coords, 3 * n)
                continue
        for _ in range(n):
            for _ in range(3):
                raw.coords.append(cur.f64())
            if parametric != 0:
                for _ in range(d):
                    _ = cur.f64()


def _read_nodes_v2(mut cur: _Cursor, mut raw: _Raw) raises:
    var n = cur.int_()
    if cur.binary:
        _ = cur.line()
    for _ in range(n):
        raw.node_tags.append(cur.i32())
        for _ in range(3):
            raw.coords.append(cur.f64())


def _read_elements_v4(
    mut cur: _Cursor,
    mut raw: _Raw,
    entity_tags: List[List[Int]],
    entity_phys: List[List[List[Int]]],
) raises:
    var num_blocks = cur.size_t()
    _ = cur.size_t()
    _ = cur.size_t()
    _ = cur.size_t()
    for _ in range(num_blocks):
        var d = cur.i32()
        var entity = cur.i32()
        var type = cur.i32()
        var n = cur.size_t()
        var phys = List[Int]()
        if 0 <= d and d < 4:
            for i in range(len(entity_tags[d])):
                if entity_tags[d][i] == entity:
                    phys = entity_phys[d][i].copy()
        var block = _Block(type, phys^)
        if block.dim != d:
            raise cur.err("element type does not match its entity")
        var nn = _nodes_per_element(type)
        if cur.binary:
            cur.r.require_items(n, 8 * (nn + 1))
        block.nodes.reserve(min(n, len(cur.r.data)) * nn)
        if cur.binary:
            # each element is its tag and its nodes, all size_t
            var rows = nm.zeros[DType.uint64](Shape(n * (nn + 1)))
            cur.r.binary_into(rows, 0, n * (nn + 1))
            for e in range(n):
                for j in range(nn):
                    block.nodes.append(
                        Int(rows.unsafe_get(e * (nn + 1) + 1 + j))
                    )
        else:
            # each element is its tag and its nodes
            cur.ints(block.nodes, n, 1, nn)
        raw.blocks.append(block^)


def _read_elements_v2(mut cur: _Cursor, mut raw: _Raw) raises:
    var n = cur.int_()
    if cur.binary:
        _ = cur.line()
        var done = 0
        while done < n:
            var type = cur.i32()
            var count = cur.i32()
            var num_tags = cur.i32()
            var nn = _nodes_per_element(type)
            # each element is its number, its tags, and its nodes, all int32
            if count < 0 or num_tags < 0:
                raise cur.err("negative element count")
            var width = 1 + num_tags + nn
            cur.r.require_items(count, 4 * width)
            if count < 16:
                for _ in range(count):
                    _ = cur.i32()
                    var phys = 0
                    for t in range(num_tags):
                        var tag = cur.i32()
                        if t == 0:
                            phys = tag
                    var b = _block_v2(raw, type, phys)
                    for _ in range(nn):
                        raw.blocks[b].nodes.append(cur.i32())
            else:
                var rows = nm.zeros[DType.int32](Shape(count * width))
                cur.r.binary_into(rows, 0, count * width)
                for e in range(count):
                    var phys = (
                        Int(rows.unsafe_get(e * width + 1)) if num_tags
                        > 0 else 0
                    )
                    var b = _block_v2(raw, type, phys)
                    for j in range(nn):
                        raw.blocks[b].nodes.append(
                            Int(rows.unsafe_get(e * width + 1 + num_tags + j))
                        )
            done += count
        return
    if cur.tokenized:
        _elements_v2_from_tokens(cur, raw, n)
        return
    for _ in range(n):
        _ = cur.int_()
        var type = cur.int_()
        var num_tags = cur.int_()
        var phys = 0
        for t in range(num_tags):
            var tag = cur.int_()
            if t == 0:
                phys = tag
        var nn = _nodes_per_element(type)
        var b = _block_v2(raw, type, phys)
        for _ in range(nn):
            raw.blocks[b].nodes.append(cur.int_())


def _elements_v2_from_tokens(mut cur: _Cursor, mut raw: _Raw, n: Int) raises:
    """`_read_elements_v2`'s ASCII loop over parsed numbers, a run of
    elements of one type and physical tag at a time."""
    if not cur.tokens.rest_are_ints():
        raise cur.err("not an integer")
    var vp = cur.tokens.values.unsafe_ptr()
    var T = len(cur.tokens.values)
    var t = cur.tokens.next
    var e = 0
    while e < n:
        # each element is its number, type, tag count, tags, and nodes
        if t + 3 > T:
            raise cur.err("file ends early")
        var type = Int(Int64(vp[unsafe_offset=t + 1]))
        var num_tags = Int(Int64(vp[unsafe_offset=t + 2]))
        var tags = max(num_tags, 0)
        var nn = _nodes_per_element(type)
        var w = 3 + tags + nn
        if t + w > T:
            raise cur.err("file ends early")
        var phys = Int(Int64(vp[unsafe_offset=t + 3])) if tags > 0 else 0
        var first = t
        var count = 0
        while e < n and t + w <= T:
            if (
                Int(Int64(vp[unsafe_offset=t + 1])) != type
                or Int(Int64(vp[unsafe_offset=t + 2])) != num_tags
                or (tags > 0 and Int(Int64(vp[unsafe_offset=t + 3])) != phys)
            ):
                break
            t += w
            e += 1
            count += 1
        var b = _block_v2(raw, type, phys)
        ref nodes = raw.blocks[b].nodes
        _reserve_more(nodes, count * nn)
        for r in range(count):
            var at = first + r * w + 3 + tags
            for j in range(nn):
                nodes.append(Int(Int64(vp[unsafe_offset=at + j])))
    cur.tokens.next = t


def _reserve_more[T: Copyable](mut list: List[T], extra: Int):
    """Make room for `extra` more items, at least doubling the capacity
    when it grows. `reserve(len + extra)` grows to exactly that, so calling
    it per block copied the list per block: quadratic when format 2.2
    alternates physical groups (an element in two is written twice in a
    row, so each element starts a block), 335 s for a 438k-element file."""
    var size = len(list) + extra
    if size > list.capacity():
        list.reserve(max(size, 2 * list.capacity()))


def _block_v2(mut raw: _Raw, type: Int, phys: Int) raises -> Int:
    """The block for the next element, of this type and physical tag (0 for
    none): the last block if it matches, else a new one."""
    var last = len(raw.blocks) - 1
    if last >= 0 and raw.blocks[last].type == type:
        ref p = raw.blocks[last].physical
        if (phys == 0 and len(p) == 0) or (
            phys != 0 and len(p) == 1 and p[0] == phys
        ):
            return last
    var physical = List[Int]()
    if phys != 0:
        physical.append(phys)
    raw.blocks.append(_Block(type, physical^))
    return last + 1


def _element_dim(type: Int) raises -> Int:
    """The dimension of a linear simplex element type; raises for others."""
    if type == 15:
        return 0
    if type == 1:
        return 1
    if type == 2:
        return 2
    if type == 4:
        return 3
    var what: String
    if type == 3 or type == 16 or type == 10:
        what = "quadrangles"
    elif type == 5 or type == 6 or type == 7 or (type >= 12 and type <= 14):
        what = "hexahedra, prisms, or pyramids"
    elif type >= 17 and type <= 19:
        what = "hexahedra, prisms, or pyramids"
    elif type == 8 or type == 9 or type == 11:
        what = "second-order elements (mesh with -order 1)"
    else:
        what = "unsupported elements"
    raise Error(
        String(
            "read_gmsh: element type ",
            type,
            " (",
            what,
            "); only points, lines, triangles, and tetrahedra are read",
        )
    )


def _nodes_per_element(type: Int) raises -> Int:
    return _element_dim(type) + 1


struct _Cursor(Movable):
    """A position in the file: ASCII tokens, or binary numbers after
    `start_binary`."""

    var r: ByteReader
    var binary: Bool
    var tokens: NumberTokens
    """A section's numbers, parsed ahead in parallel (see `tokenize`)."""
    var tokenized: Bool
    """Whether numbers come from `tokens` rather than `r`."""

    def __init__(out self, var r: ByteReader):
        self.r = r^
        self.binary = False
        self.tokens = NumberTokens()
        self.tokenized = False

    def tokenize(mut self, parallel_bytes: Int) -> Int:
        """Parse the numbers up to the next `$` in parallel, for the next
        reads to take; returns the offset of the `$`, or -1 (and reads go
        on from the text) when the stretch is small, has no `$`, or holds
        a token for the serial reader to judge."""
        if len(self.r.data) - self.r.pos < parallel_bytes:
            return -1
        var end = _find_byte(self.r.data, self.r.pos, Byte(ord("$")))
        if end < 0:
            return -1
        var t = parse_numbers(self.r, self.r.pos, end, parallel_bytes)
        if not t:
            return -1
        self.tokens = t.take()
        self.tokenized = True
        return end

    def untokenize(mut self):
        """Go back to reading the text, freeing the parsed numbers."""
        self.tokens = NumberTokens()
        self.tokenized = False

    def err(self, message: String) -> Error:
        return self.r.err(message)

    def more(mut self) -> Bool:
        return self.r.more()

    def section(mut self) raises -> String:
        """The name of the section starting here: `$Name` -> `Name`."""
        if not self.r.more():
            raise self.err("file ends early")
        if self.r.data[self.r.pos] != Byte(ord("$")):
            raise self.err("expected a $Section line")
        self.r.pos += 1
        return self.r.line()

    def end(mut self, section: String) raises:
        """Read the `$EndName` line closing a section."""
        var name = self.section()
        if name != "End" + section:
            raise self.err(String("expected $End", section, ", found $", name))

    def skip_section(mut self, section: String) raises:
        var marker = ("$End" + section).as_bytes()
        ref data = self.r.data
        while self.r.pos + len(marker) <= len(data):
            var found = True
            for j in range(len(marker)):
                if data[self.r.pos + j] != marker[j]:
                    found = False
                    break
            if found:
                self.r.pos += len(marker)
                _ = self.r.line()
                return
            self.r.pos += 1
        raise self.err("unterminated $" + section)

    def start_binary(mut self) raises:
        """Read the integer 1 that follows a binary `$MeshFormat` line to
        find the byte order."""
        if self.r.pos + 4 > len(self.r.data):
            raise self.err("file ends early")
        self.r.big_endian = self.r.data[self.r.pos] == 0
        self.binary = True
        if self.i32() != 1:
            raise self.err("bad binary byte-order marker")
        _ = self.r.line()

    def line(mut self) -> String:
        """The rest of the current line, without its line break."""
        return self.r.line()

    def token(mut self) raises -> String:
        return self.r.token()

    def quoted(mut self) raises -> String:
        """A double-quoted string (may contain spaces)."""
        self.r.skip_space()
        ref data = self.r.data
        if self.r.pos >= len(data) or data[self.r.pos] != Byte(ord('"')):
            raise self.err("expected a quoted name")
        var start = self.r.pos + 1
        self.r.pos = start
        while self.r.pos < len(data) and data[self.r.pos] != Byte(ord('"')):
            self.r.pos += 1
        if self.r.pos >= len(data):
            raise self.err("unterminated quoted name")
        self.r.pos += 1
        try:
            return String(from_utf8=Span(data)[start : self.r.pos - 1])
        except:
            raise self.err("a quoted name is not valid UTF-8")

    def ints(
        mut self, mut out: List[Int], rows: Int, skip: Int, keep: Int
    ) raises:
        """Read `rows` rows of `skip + keep` ASCII integers, appending the
        last `keep` of each row to `out`."""
        if self.tokenized:
            self.tokens.ints(out, rows, skip, keep)
            return
        for _ in range(rows):
            for _ in range(skip):
                _ = self.r.int_()
            for _ in range(keep):
                out.append(self.r.int_())

    def floats(mut self, mut out: List[Float64], count: Int) raises:
        """Read `count` ASCII numbers, appending them to `out`."""
        if self.tokenized:
            self.tokens.floats(out, count)
            return
        for _ in range(count):
            out.append(self.r.float_())

    @always_inline
    def int_(mut self) raises -> Int:
        """An ASCII integer, even in a binary file (counts, names)."""
        if self.tokenized:
            return self.tokens.int_()
        return self.r.int_()

    @always_inline
    def i32(mut self) raises -> Int:
        if not self.binary:
            return self.int_()
        return Int(self.r.binary[DType.int32]())

    @always_inline
    def size_t(mut self) raises -> Int:
        if not self.binary:
            return self.int_()
        return Int(self.r.binary[DType.uint64]())

    @always_inline
    def f64(mut self) raises -> Float64:
        if not self.binary:
            if self.tokenized:
                return self.tokens.float_()
            return self.r.float_()
        return self.r.binary[DType.float64]()


def _find_byte(data: List[Byte], start: Int, byte: Byte) -> Int:
    """The first offset `>= start` of `byte` in `data`, or -1; compares
    bytes, so the data need not be UTF-8."""
    comptime W = 32
    var p = data.unsafe_ptr()
    var n = len(data)
    var i = start
    var target = SIMD[DType.uint8, W](byte)
    while i + W <= n:
        var hit = p.unsafe_offset(i).unsafe_load[width=W]().eq(target)
        if hit.reduce_or():
            break
        i += W
    while i < n:
        if p[unsafe_offset=i] == byte:
            return i
        i += 1
    return -1
