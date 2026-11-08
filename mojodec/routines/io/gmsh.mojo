# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
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

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import SimplicialMesh
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.subset import Subset
from mojodec.core.typed import PRIMAL, TypedComplex, TypedSubset
from mojodec.core.utils import (
    IntArray,
    RealArray,
    cols_of,
    compare_rows,
    int_array,
    lexsort_rows,
    real_array,
    rows_of,
    sort_each_row,
)
from mojodec.routines.io.scan import ByteReader


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
    path: String, dim: Optional[Int] = None, drop_z: Bool = True
) raises -> GmshMesh:
    """Read a Gmsh `.msh` file (format 4.1 or 2.2, ASCII or binary).

    Only vertices of the top-dimensional simplices are kept, numbered in
    the order of their node tags (`node_tags` maps back). Points, lines,
    triangles, and tetrahedra are read; other element types (quadrangles,
    hexahedra, second-order elements) raise. Elements repeated because they
    belong to several physical groups (format 2.2) are kept once.

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
    # the top elements, as node tags, without the repeats that format 2.2
    # writes for elements in several physical groups
    var tagged = List[Int]()
    for ref b in raw.blocks:
        if b.dim == top:
            tagged.extend(Span(b.nodes))
    var rows = _unique_rows(tagged, w)
    # number the nodes of the top elements in tag order; tags can be large
    # and sparse, so they are sorted and searched, not used as indices
    var nodes = _TagIndex(raw.node_tags)
    for r in rows:
        for j in range(w):
            var t = tagged.unsafe_get(r * w + j)
            var at = nodes.find(t)
            if at < 0:
                raise Error(String("read_gmsh: element uses unknown node ", t))
            nodes.vertex.unsafe_set(at, 0)
    var node_tags = List[Int]()
    for at in range(len(nodes.tags)):
        if nodes.vertex.unsafe_get(at) == 0:
            nodes.vertex.unsafe_set(at, len(node_tags))
            node_tags.append(nodes.tags.unsafe_get(at))
    var elements = int_array(len(rows), w)
    for i in range(len(rows)):
        var r = rows.unsafe_get(i)
        for j in range(w):
            var at = nodes.find(tagged.unsafe_get(r * w + j))
            elements.unsafe_set(
                i * w + j, Scalar[DType.int](nodes.vertex.unsafe_get(at))
            )
    # coordinates, from the first node with each tag
    var slots = List[Int](capacity=len(node_tags))
    for t in node_tags:
        slots.append(nodes.slot.unsafe_get(nodes.find(t)))
    var flat = True
    for at in slots:
        if raw.coords.unsafe_get(3 * at + 2) != 0:
            flat = False
            break
    var e = 2 if drop_z and top <= 2 and flat else 3
    var vertices = real_array(len(node_tags), e)
    for i in range(len(node_tags)):
        var at = 3 * slots.unsafe_get(i)
        for c in range(e):
            vertices.unsafe_set(i * e + c, raw.coords.unsafe_get(at + c))
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
                    cells.reserve(len(cells) + len(b.nodes))
                    for t in b.nodes:
                        var at = nodes.find(t)
                        var v = nodes.vertex.unsafe_get(at) if at >= 0 else -1
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
    return GmshMesh(vertices^, elements^, node_tags^, groups^)


def _unique_rows(nodes: List[Int], w: Int) raises -> List[Int]:
    """Indices of the rows of `nodes` (width `w`) to keep: the first of
    each set of rows with the same vertices, in file order."""
    var n = len(nodes) // w
    var keys = int_array(n, w)
    for i in range(len(nodes)):
        keys.unsafe_set(i, Scalar[DType.int](nodes[i]))
    sort_each_row(keys)
    var perm = lexsort_rows(keys)
    var keep = List[Bool](length=n, fill=True)
    var i = 0
    while i < n:
        var j = i + 1
        var first = perm[i]
        while j < n and compare_rows(keys, perm[i], keys, perm[j]) == 0:
            first = min(first, perm[j])
            j += 1
        for r in range(i, j):
            if perm[r] != first:
                keep[perm[r]] = False
        i = j
    var out = List[Int](capacity=n)
    for r in range(n):
        if keep[r]:
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
    """The node tags of a file in increasing order, searched by bisection."""

    var tags: List[Int]
    """The tags, sorted (stably, so a repeated tag finds its first node)."""
    var slot: List[Int]
    """The position of each tag's node in the file."""
    var vertex: List[Int]
    """Per tag: its vertex number, 0 while marked used, or -1."""

    def __init__(out self, file_tags: List[Int]) raises:
        var n = len(file_tags)
        var column = int_array(n, 1)
        for i in range(n):
            var t = file_tags.unsafe_get(i)
            if t < 0:
                raise Error(String("read_gmsh: negative node tag ", t))
            column.unsafe_set(i, Scalar[DType.int](t))
        self.slot = lexsort_rows(column)
        self.tags = List[Int](capacity=n)
        for i in self.slot:
            self.tags.append(file_tags.unsafe_get(i))
        self.vertex = List[Int](length=n, fill=-1)

    def find(self, t: Int) -> Int:
        """The first position of tag `t`, or -1."""
        var lo = 0
        var hi = len(self.tags)
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.tags.unsafe_get(mid) < t:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.tags) and self.tags.unsafe_get(lo) == t:
            return lo
        return -1


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


def _parse(path: String) raises -> _Raw:
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
        elif section == "Nodes":
            if version == 4:
                _read_nodes_v4(cur, raw)
            else:
                _read_nodes_v2(cur, raw)
        elif section == "Elements":
            if version == 4:
                _read_elements_v4(cur, raw, entity_tags, entity_phys)
            else:
                _read_elements_v2(cur, raw)
        else:
            cur.skip_section(section)
            continue
        cur.end(section)
    if version == 0:
        raise cur.err("no $MeshFormat section")
    return raw^


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
                raw.coords.reserve(len(raw.coords) + 3 * n)
                for i in range(3 * n):
                    raw.coords.append(xyz.unsafe_get(i))
                continue
        else:
            for _ in range(n):
                raw.node_tags.append(cur.size_t())
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
            for _ in range(n):
                _ = cur.size_t()
                for _ in range(nn):
                    block.nodes.append(cur.size_t())
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

    def __init__(out self, var r: ByteReader):
        self.r = r^
        self.binary = False

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
        return String(unsafe_from_utf8=Span(data)[start : self.r.pos - 1])

    def int_(mut self) raises -> Int:
        """An ASCII integer, even in a binary file (counts, names)."""
        return self.r.int_()

    def i32(mut self) raises -> Int:
        if not self.binary:
            return self.r.int_()
        return Int(self.r.binary[DType.int32]())

    def size_t(mut self) raises -> Int:
        if not self.binary:
            return self.r.int_()
        return Int(self.r.binary[DType.uint64]())

    def f64(mut self) raises -> Float64:
        if not self.binary:
            return self.r.float_()
        return self.r.binary[DType.float64]()
