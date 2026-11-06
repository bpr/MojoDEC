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

Gmsh saves only the elements of physical groups once any group is
defined, so a `.geo` file that tags boundary curves must also tag the
surface (`Physical Surface("domain") = {1};`) or set `Mesh.SaveAll = 1;`,
or the triangles are missing. Passing the expected `dim` turns that into an
error instead of a mesh of line segments.
"""

from std.memory import bitcast

from mojodec.core.mesh import SimplicialMesh
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.subset import Subset
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


struct GmshGroup(Copyable, Movable, Writable):
    """A Gmsh physical group: the cells of one dimension it tags."""

    var name: String
    """The group's name, or "" for a group known only by its tag."""
    var dim: Int
    """Dimension of the tagged cells."""
    var tag: Int
    """The physical tag, unique among the groups of one dimension."""
    var cells: List[Int]
    """Vertex indices of the cells, `dim + 1` per cell, in file order."""

    def __init__(out self, name: String, dim: Int, tag: Int):
        """An empty group; `read_gmsh` appends the cells.

        Args:
            name: The group's name, or "" if it has none.
            dim: Dimension of the tagged cells.
            tag: The physical tag.
        """
        self.name = name
        self.dim = dim
        self.tag = tag
        self.cells = List[Int]()

    def num_cells(self) -> Int:
        """The number of cells in the group.

        Returns:
            `len(cells) // (dim + 1)`.
        """
        return len(self.cells) // (self.dim + 1)

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
                v.append(self.cells[c * w + j])
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

    def subset(
        self, sc: SimplicialComplex, group_dim: Int, tag: Int, dim: Int
    ) raises -> Subset:
        """The primal `dim`-simplices of the group with this dimension and
        physical tag (for unnamed groups).

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
    path: String, dim: Int = -1, drop_z: Bool = True
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
        dim: The expected dimension of the mesh (-1, the default, accepts
            any). Pass it: once a `.geo` file defines physical groups, Gmsh
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
    if dim >= 0 and top != dim:
        raise Error(
            String(
                "read_gmsh: expected a ",
                dim,
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
            tagged.extend(b.nodes.copy())
    var rows = _unique_rows(tagged, w)
    # number the nodes of the top elements in tag order
    var max_tag = 0
    for t in raw.node_tags:
        max_tag = max(max_tag, t)
    var slot = List[Int](length=max_tag + 1, fill=-1)
    for i in range(len(raw.node_tags)):
        slot[raw.node_tags[i]] = i
    var index = List[Int](length=max_tag + 1, fill=-1)
    for r in rows:
        for j in range(w):
            var t = tagged[r * w + j]
            if t < 0 or t > max_tag or slot[t] < 0:
                raise Error(String("read_gmsh: element uses unknown node ", t))
            index[t] = 0
    var node_tags = List[Int]()
    for t in range(max_tag + 1):
        if index[t] == 0:
            index[t] = len(node_tags)
            node_tags.append(t)
    var elements = int_array(len(rows), w)
    for i in range(len(rows)):
        for j in range(w):
            elements.unsafe_set(
                i * w + j, Scalar[DType.int](index[tagged[rows[i] * w + j]])
            )
    # coordinates
    var flat = True
    for t in node_tags:
        if raw.coords[3 * slot[t] + 2] != 0:
            flat = False
    var e = 2 if drop_z and top <= 2 and flat else 3
    var vertices = real_array(len(node_tags), e)
    for i in range(len(node_tags)):
        for c in range(e):
            vertices.unsafe_set(
                i * e + c, raw.coords[3 * slot[node_tags[i]] + c]
            )
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
            var g = GmshGroup(raw.name_of(d, p), d, p)
            for ref b in raw.blocks:
                if b.dim == d and p in b.physical:
                    for t in b.nodes:
                        if t < 0 or t > max_tag or index[t] < 0:
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
                        g.cells.append(index[t])
            groups.append(g^)
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
    var data: List[Byte]
    with open(path, "r") as f:
        data = f.read_bytes()
    var cur = _Cursor(data^, path)
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
            for _ in range(count):
                _ = cur.i32()
                var phys = List[Int]()
                for t in range(num_tags):
                    var tag = cur.i32()
                    if t == 0 and tag != 0:
                        phys.append(tag)
                var nodes = List[Int](capacity=nn)
                for _ in range(nn):
                    nodes.append(cur.i32())
                _add_v2(raw, type, phys^, nodes)
            done += count
        return
    for _ in range(n):
        _ = cur.int_()
        var type = cur.int_()
        var num_tags = cur.int_()
        var phys = List[Int]()
        for t in range(num_tags):
            var tag = cur.int_()
            if t == 0 and tag != 0:
                phys.append(tag)
        var nn = _nodes_per_element(type)
        var nodes = List[Int](capacity=nn)
        for _ in range(nn):
            nodes.append(cur.int_())
        _add_v2(raw, type, phys^, nodes)


def _add_v2(
    mut raw: _Raw, type: Int, var phys: List[Int], nodes: List[Int]
) raises:
    """Append an element, extending the last block when it matches."""
    var last = len(raw.blocks) - 1
    if (
        last < 0
        or raw.blocks[last].type != type
        or raw.blocks[last].physical != phys
    ):
        raw.blocks.append(_Block(type, phys^))
        last += 1
    raw.blocks[last].nodes.extend(nodes.copy())


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

    var data: List[Byte]
    var pos: Int
    var binary: Bool
    var swap: Bool
    """Big-endian binary data."""
    var path: String

    def __init__(out self, var data: List[Byte], path: String):
        self.data = data^
        self.pos = 0
        self.binary = False
        self.swap = False
        self.path = path

    def err(self, message: String) -> Error:
        return Error(String("read_gmsh: ", message, " in ", self.path))

    def more(mut self) -> Bool:
        self._skip_space()
        return self.pos < len(self.data)

    def section(mut self) raises -> String:
        """The name of the section starting here: `$Name` -> `Name`."""
        self._skip_space()
        if self.pos >= len(self.data):
            raise self.err("file ends early")
        if self.data[self.pos] != Byte(ord("$")):
            raise self.err("expected a $Section line")
        self.pos += 1
        return self.line()

    def end(mut self, section: String) raises:
        """Read the `$EndName` line closing a section."""
        var name = self.section()
        if name != "End" + section:
            raise self.err(String("expected $End", section, ", found $", name))

    def skip_section(mut self, section: String) raises:
        var marker = ("$End" + section).as_bytes()
        while self.pos + len(marker) <= len(self.data):
            var found = True
            for j in range(len(marker)):
                if self.data[self.pos + j] != marker[j]:
                    found = False
                    break
            if found:
                self.pos += len(marker)
                _ = self.line()
                return
            self.pos += 1
        raise self.err("unterminated $" + section)

    def start_binary(mut self) raises:
        """Read the integer 1 that follows a binary `$MeshFormat` line to
        find the byte order."""
        if self.pos + 4 > len(self.data):
            raise self.err("file ends early")
        self.swap = self.data[self.pos] == 0
        self.binary = True
        if self.i32() != 1:
            raise self.err("bad binary byte-order marker")
        _ = self.line()

    def line(mut self) -> String:
        """The rest of the current line, without its line break."""
        var start = self.pos
        while self.pos < len(self.data) and self.data[self.pos] != Byte(
            ord("\n")
        ):
            self.pos += 1
        var end = self.pos
        if self.pos < len(self.data):
            self.pos += 1
        if end > start and self.data[end - 1] == Byte(ord("\r")):
            end -= 1
        return String(unsafe_from_utf8=Span(self.data)[start:end])

    def token(mut self) raises -> String:
        self._skip_space()
        var start = self.pos
        while self.pos < len(self.data) and not _is_space(self.data[self.pos]):
            self.pos += 1
        if start == self.pos:
            raise self.err("file ends early")
        return String(unsafe_from_utf8=Span(self.data)[start : self.pos])

    def quoted(mut self) raises -> String:
        """A double-quoted string (may contain spaces)."""
        self._skip_space()
        if self.pos >= len(self.data) or self.data[self.pos] != Byte(ord('"')):
            raise self.err("expected a quoted name")
        var start = self.pos + 1
        self.pos = start
        while self.pos < len(self.data) and self.data[self.pos] != Byte(
            ord('"')
        ):
            self.pos += 1
        if self.pos >= len(self.data):
            raise self.err("unterminated quoted name")
        self.pos += 1
        return String(unsafe_from_utf8=Span(self.data)[start : self.pos - 1])

    def int_(mut self) raises -> Int:
        """An ASCII integer, even in a binary file (counts, names)."""
        return atol(self.token())

    def i32(mut self) raises -> Int:
        if not self.binary:
            return self.int_()
        return Int(bitcast[DType.int32](UInt32(self._bytes(4))))

    def size_t(mut self) raises -> Int:
        if not self.binary:
            return self.int_()
        return Int(self._bytes(8))

    def f64(mut self) raises -> Float64:
        if not self.binary:
            return atof(self.token())
        return bitcast[DType.float64](self._bytes(8))

    def _bytes(mut self, n: Int) raises -> UInt64:
        if self.pos + n > len(self.data):
            raise self.err("file ends early")
        var u: UInt64 = 0
        for b in range(n):
            var shift = 8 * (n - 1 - b) if self.swap else 8 * b
            u |= UInt64(self.data[self.pos + b]) << UInt64(shift)
        self.pos += n
        return u

    def _skip_space(mut self):
        while self.pos < len(self.data) and _is_space(self.data[self.pos]):
            self.pos += 1


def _is_space(b: Byte) -> Bool:
    return (
        b == Byte(ord(" "))
        or b == Byte(ord("\n"))
        or b == Byte(ord("\r"))
        or b == Byte(ord("\t"))
    )
