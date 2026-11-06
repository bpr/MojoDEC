# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
PLY files (`mojodec.routines.io.ply`).
======================================
Reader and writer for triangle meshes in the PLY ("Stanford triangle")
format, which MeshLab, Blender, and most scanners read and write. Not in
PyDEC.

`read_ply` handles the `ascii`, `binary_little_endian`, and
`binary_big_endian` formats. It takes the `x`, `y`, and `z` properties of
the `vertex` element and the `vertex_indices` (or `vertex_index`) list of
the `face` element, skipping every other property and element. Polygons
with more than three vertices are split into fans of triangles.
"""

from std.memory import bitcast

from mojodec.core.mesh import SimplicialMesh
from mojodec.core.utils import (
    IntArray,
    RealArray,
    cols_of,
    fget,
    iget,
    int_array,
    real_array,
    rows_of,
)


def read_ply(path: String) raises -> SimplicialMesh:
    """Read a triangle mesh from a PLY file.

    The vertices have one column per coordinate property present, in the
    order `x`, `y`, `z`; a 2D mesh written with `z = 0` reads back with
    three columns.

    ```mojo
    var mesh = read_ply("bunny.ply")
    var sc = SimplicialComplex(mesh.vertices, mesh.elements)
    ```

    Args:
        path: The PLY file.

    Returns:
        The mesh: vertices with 2 or 3 columns, and the faces as triangles.

    Raises:
        If the file cannot be opened; does not start with `ply` or has no
        `end_header`; has an unsupported format or property type, a
        `property` line before any `element`, or truncated data; has no
        vertices, or vertices without `x` and `y`; has no `face` element,
        faces without a `vertex_indices` list, or a face with fewer than 3
        vertices or an out-of-range index; or `SimplicialMesh` rejects
        the result.
    """
    var data: List[Byte]
    with open(path, "r") as f:
        data = f.read_bytes()
    var header = _Header(data)
    var body = _Body(data, header.data_start, header.format)
    var vertices = real_array(0, 1)
    var triangles = List[Int]()
    var have_faces = False
    for ref element in header.elements:
        if element.name == "vertex":
            vertices = _read_vertices(body, element)
        elif element.name == "face":
            _read_faces(body, element, triangles)
            have_faces = True
        else:
            for _ in range(element.count):
                for ref prop in element.properties:
                    _ = body.property(prop)
    if rows_of(vertices) == 0:
        raise Error("read_ply: no vertex element in " + path)
    if not have_faces:
        raise Error("read_ply: no face element in " + path)
    var elements = int_array(len(triangles) // 3, 3)
    for i in range(len(triangles)):
        var v = triangles[i]
        if v < 0 or v >= rows_of(vertices):
            raise Error("read_ply: face vertex index out of range")
        elements.unsafe_set(i, Scalar[DType.int](v))
    return SimplicialMesh(vertices, elements)


def write_ply(
    path: String,
    vertices: RealArray,
    triangles: IntArray,
    format: String = "ascii",
) raises:
    """Write a triangle mesh as a PLY file, `format` `"ascii"` or `"binary"`.

    Coordinates are written as doubles; 2D vertices get `z = 0`, which
    most programs expect. `"binary"` is `binary_little_endian`. Faces are
    a `uchar` count and `int` (32-bit) indices.

    Args:
        path: The file to write; an existing file is replaced.
        vertices: Vertex coordinates, `N x 2` or `N x 3`.
        triangles: The triangles, `M x 3` vertex indices.
        format: `ascii` or `binary`.

    Raises:
        If `format` is unknown, `vertices` is not `N x 2` or `N x 3`,
        `triangles` is not `M x 3`, or the file cannot be written.
    """
    if format != "ascii" and format != "binary":
        raise Error("write_ply: unknown format [" + format + "]")
    if vertices.ndim != 2 or cols_of(vertices) < 2 or cols_of(vertices) > 3:
        raise Error("write_ply: vertices must be N x 2 or N x 3")
    if triangles.ndim != 2 or cols_of(triangles) != 3:
        raise Error("write_ply: triangles must be N x 3")
    var nv = rows_of(vertices)
    var nt = rows_of(triangles)
    var E = cols_of(vertices)
    var header = String(
        "ply\nformat ",
        "ascii" if format == "ascii" else "binary_little_endian",
        " 1.0\ncomment written by MojoDEC\nelement vertex ",
        nv,
        "\nproperty double x\nproperty double y\nproperty double z\n",
        "element face ",
        nt,
        "\nproperty list uchar int vertex_indices\nend_header\n",
    )
    with open(path, "w") as f:
        f.write(header)
        if format == "ascii":
            var text = String()
            for v in range(nv):
                for k in range(3):
                    var x: Float64 = fget(vertices, v, k) if k < E else 0.0
                    text.write(x, "\n" if k == 2 else " ")
            for t in range(nt):
                text.write(
                    "3 ",
                    iget(triangles, t, 0),
                    " ",
                    iget(triangles, t, 1),
                    " ",
                    iget(triangles, t, 2),
                    "\n",
                )
            f.write(text)
        else:
            var bytes = List[Byte](capacity=24 * nv + 13 * nt)
            for v in range(nv):
                for k in range(3):
                    var x: Float64 = fget(vertices, v, k) if k < E else 0.0
                    _put_le(bytes, bitcast[DType.uint64](x), 8)
            for t in range(nt):
                bytes.append(3)
                for k in range(3):
                    _put_le(
                        bytes,
                        UInt64(
                            bitcast[DType.uint32](Int32(iget(triangles, t, k)))
                        ),
                        4,
                    )
            f.write_bytes(Span(bytes))


struct _Property(Copyable, Movable):
    var name: String
    var type: String
    var is_list: Bool
    var count_type: String

    def __init__(
        out self,
        name: String,
        type: String,
        is_list: Bool = False,
        count_type: String = "",
    ):
        self.name = name
        self.type = type
        self.is_list = is_list
        self.count_type = count_type


struct _Element(Copyable, Movable):
    var name: String
    var count: Int
    var properties: List[_Property]

    def __init__(out self, name: String, count: Int):
        self.name = name
        self.count = count
        self.properties = List[_Property]()


struct _Header(Movable):
    var format: String
    var elements: List[_Element]
    var data_start: Int

    def __init__(out self, data: List[Byte]) raises:
        var end = _find(data, "end_header")
        if not _starts_with(data, "ply") or end < 0:
            raise Error("read_ply: not a PLY file")
        # the data starts after the end_header line's newline
        var start = end
        while start < len(data) and data[start] != Byte(ord("\n")):
            start += 1
        self.data_start = start + 1
        var text = String(unsafe_from_utf8=Span(data)[:end])
        self.format = ""
        self.elements = List[_Element]()
        for line in text.splitlines():
            var words = line.split()
            if len(words) == 0:
                continue
            var key = words[0]
            if key == "format":
                self.format = String(words[1])
            elif key == "element":
                self.elements.append(_Element(String(words[1]), atol(words[2])))
            elif key == "property":
                if len(self.elements) == 0:
                    raise Error("read_ply: property before any element")
                if words[1] == "list":
                    self.elements[len(self.elements) - 1].properties.append(
                        _Property(
                            String(words[4]),
                            _type_name(String(words[3])),
                            True,
                            _type_name(String(words[2])),
                        )
                    )
                else:
                    self.elements[len(self.elements) - 1].properties.append(
                        _Property(
                            String(words[2]), _type_name(String(words[1]))
                        )
                    )
        if (
            self.format != "ascii"
            and self.format != "binary_little_endian"
            and self.format != "binary_big_endian"
        ):
            raise Error("read_ply: unsupported format [" + self.format + "]")


struct _Body(Movable):
    """A cursor over the data: whitespace-separated tokens or raw bytes."""

    var data: List[Byte]
    var pos: Int
    var ascii: Bool
    var big_endian: Bool
    var tokens: List[String]

    def __init__(out self, data: List[Byte], start: Int, format: String):
        self.data = data.copy()
        self.ascii = format == "ascii"
        self.big_endian = format == "binary_big_endian"
        self.tokens = List[String]()
        self.pos = start
        if self.ascii:
            var text = String(unsafe_from_utf8=Span(data)[start:])
            for word in text.split():
                self.tokens.append(String(word))
            self.pos = 0

    def scalar(mut self, type: String) raises -> Float64:
        if self.ascii:
            if self.pos >= len(self.tokens):
                raise Error("read_ply: file ends early")
            self.pos += 1
            return atof(self.tokens[self.pos - 1])
        var size = _type_size(type)
        if self.pos + size > len(self.data):
            raise Error("read_ply: file ends early")
        var u: UInt64 = 0
        for b in range(size):
            var shift = 8 * (size - 1 - b) if self.big_endian else 8 * b
            u |= UInt64(self.data[self.pos + b]) << UInt64(shift)
        self.pos += size
        return _decode(type, u)

    def property(mut self, prop: _Property) raises -> List[Float64]:
        """One property's value, or all of a list property's items."""
        var out = List[Float64]()
        if prop.is_list:
            var n = Int(self.scalar(prop.count_type))
            for _ in range(n):
                out.append(self.scalar(prop.type))
        else:
            out.append(self.scalar(prop.type))
        return out^


def _read_vertices(mut body: _Body, element: _Element) raises -> RealArray:
    var axes = List[Int]()  # property position of x, y, z
    for name in ["x", "y", "z"]:
        for i in range(len(element.properties)):
            if element.properties[i].name == name:
                axes.append(i)
    if len(axes) < 2:
        raise Error("read_ply: vertices need at least x and y properties")
    var vertices = real_array(element.count, len(axes))
    var values = List[Float64](length=len(element.properties), fill=0.0)
    for v in range(element.count):
        for i in range(len(element.properties)):
            var x = body.property(element.properties[i])
            values[i] = x[0] if len(x) > 0 else 0.0
        for k in range(len(axes)):
            vertices.unsafe_set(v * len(axes) + k, values[axes[k]])
    return vertices^


def _read_faces(
    mut body: _Body, element: _Element, mut triangles: List[Int]
) raises:
    var index = -1
    for i in range(len(element.properties)):
        ref prop = element.properties[i]
        if prop.is_list and (
            prop.name == "vertex_indices" or prop.name == "vertex_index"
        ):
            index = i
    if index < 0:
        raise Error("read_ply: faces need a vertex_indices list property")
    for _ in range(element.count):
        for i in range(len(element.properties)):
            var x = body.property(element.properties[i])
            if i != index:
                continue
            if len(x) < 3:
                raise Error("read_ply: a face has fewer than 3 vertices")
            # a fan of triangles around the first vertex
            for j in range(1, len(x) - 1):
                triangles.append(Int(x[0]))
                triangles.append(Int(x[j]))
                triangles.append(Int(x[j + 1]))


def _type_name(t: String) raises -> String:
    """The canonical name of a PLY scalar type."""
    if t == "char" or t == "int8":
        return "int8"
    if t == "uchar" or t == "uint8":
        return "uint8"
    if t == "short" or t == "int16":
        return "int16"
    if t == "ushort" or t == "uint16":
        return "uint16"
    if t == "int" or t == "int32":
        return "int32"
    if t == "uint" or t == "uint32":
        return "uint32"
    if t == "float" or t == "float32":
        return "float32"
    if t == "double" or t == "float64":
        return "float64"
    raise Error("read_ply: unknown property type [" + t + "]")


def _type_size(t: String) -> Int:
    if t == "float64":
        return 8
    if t == "int32" or t == "uint32" or t == "float32":
        return 4
    if t == "int16" or t == "uint16":
        return 2
    return 1


def _decode(t: String, u: UInt64) -> Float64:
    if t == "float64":
        return bitcast[DType.float64](u)
    if t == "float32":
        return Float64(bitcast[DType.float32](UInt32(u)))
    if t == "int32":
        return Float64(bitcast[DType.int32](UInt32(u)))
    if t == "int16":
        return Float64(bitcast[DType.int16](UInt16(u)))
    if t == "int8":
        return Float64(bitcast[DType.int8](UInt8(u)))
    return Float64(u)


def _put_le(mut out: List[Byte], u: UInt64, size: Int):
    for b in range(size):
        out.append(Byte((u >> UInt64(8 * b)) & 0xFF))


def _find(data: List[Byte], word: String) -> Int:
    """Offset of the first occurrence of `word` in `data`, or -1."""
    var w = word.as_bytes()
    for i in range(len(data) - len(w) + 1):
        var found = True
        for j in range(len(w)):
            if data[i + j] != w[j]:
                found = False
                break
        if found:
            return i
    return -1


def _starts_with(data: List[Byte], word: String) -> Bool:
    var w = word.as_bytes()
    if len(data) < len(w):
        return False
    for j in range(len(w)):
        if data[j] != w[j]:
            return False
    return True
