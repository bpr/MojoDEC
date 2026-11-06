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

from std.sys import size_of

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import SimplicialMesh
from mojodec.core.utils import (
    IntArray,
    RealArray,
    cols_of,
    int_array,
    real_array,
    rows_of,
)
from mojodec.routines.io.scan import (
    ByteReader,
    append_le,
    native_little_endian,
)


struct PlyFormat(Equatable, ImplicitlyCopyable, Writable):
    """How `write_ply` stores the mesh."""

    var _code: UInt8

    comptime ASCII = PlyFormat(0)
    """Readable text."""
    comptime BINARY = PlyFormat(1)
    """`binary_little_endian`."""

    def __init__(out self, code: UInt8):
        """A format from its code; use the named members instead.

        Args:
            code: 0 or 1.
        """
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        """Whether two formats are the same.

        Args:
            other: The other format.

        Returns:
            True if they are equal.
        """
        return self._code == other._code

    def write_to(self, mut writer: Some[Writer]):
        """Write the name used in the PLY header.

        Args:
            writer: The destination.
        """
        writer.write("ascii" if self == Self.ASCII else "binary_little_endian")


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
    var reader = ByteReader.open(path, "read_ply")
    var header = _Header(reader.data)
    reader.pos = header.data_start
    reader.big_endian = header.format == _BIG_ENDIAN
    var ascii = header.format == _ASCII
    var vertices = real_array(0, 1)
    var triangles = List[Int]()
    var have_faces = False
    for ref element in header.elements:
        if element.name == "vertex":
            vertices = _read_vertices(reader, ascii, element)
        elif element.name == "face":
            _read_faces(reader, ascii, element, triangles)
            have_faces = True
        else:
            for _ in range(element.count):
                for ref prop in element.properties:
                    _skip(reader, ascii, prop)
    if rows_of(vertices) == 0:
        raise Error("read_ply: no vertex element in " + path)
    if not have_faces:
        raise Error("read_ply: no face element in " + path)
    var elements = int_array(len(triangles) // 3, 3)
    var nv = rows_of(vertices)
    for i in range(len(triangles)):
        var v = triangles.unsafe_get(i)
        if v < 0 or v >= nv:
            raise Error("read_ply: face vertex index out of range")
        elements.unsafe_set(i, Scalar[DType.int](v))
    return SimplicialMesh(vertices, elements)


def write_ply(
    path: String,
    vertices: RealArray,
    triangles: IntArray,
    format: PlyFormat = .ASCII,
) raises:
    """Write a triangle mesh as a PLY file.

    Coordinates are written as doubles; 2D vertices get `z = 0`, which
    most programs expect. Faces are a `uchar` count and `int` (32-bit)
    indices.

    Args:
        path: The file to write; an existing file is replaced.
        vertices: Vertex coordinates, `N x 2` or `N x 3`.
        triangles: The triangles, `M x 3` vertex indices.
        format: `ASCII` or `BINARY` (`binary_little_endian`).

    Raises:
        If `vertices` is not `N x 2` or `N x 3`, `triangles` is not
        `M x 3`, or the file cannot be written.
    """
    if vertices.ndim != 2 or cols_of(vertices) < 2 or cols_of(vertices) > 3:
        raise Error("write_ply: vertices must be N x 2 or N x 3")
    if triangles.ndim != 2 or cols_of(triangles) != 3:
        raise Error("write_ply: triangles must be N x 3")
    var nv = rows_of(vertices)
    var nt = rows_of(triangles)
    var E = cols_of(vertices)
    var header = String(
        "ply\nformat ",
        format,
        " 1.0\ncomment written by MojoDEC\nelement vertex ",
        nv,
        "\nproperty double x\nproperty double y\nproperty double z\n",
        "element face ",
        nt,
        "\nproperty list uchar int vertex_indices\nend_header\n",
    )
    with open(path, "w") as f:
        f.write(header)
        if format == PlyFormat.ASCII:
            var text = String()
            for v in range(nv):
                for k in range(3):
                    var x: Float64 = (
                        vertices.unsafe_get(v * E + k) if k < E else 0.0
                    )
                    text.write(x, "\n" if k == 2 else " ")
            for t in range(nt):
                text.write(
                    "3 ",
                    triangles.unsafe_get(3 * t),
                    " ",
                    triangles.unsafe_get(3 * t + 1),
                    " ",
                    triangles.unsafe_get(3 * t + 2),
                    "\n",
                )
            f.write(text)
        else:
            var bytes = List[Byte](capacity=24 * nv + 13 * nt)
            for v in range(nv):
                for k in range(3):
                    var x: Float64 = (
                        vertices.unsafe_get(v * E + k) if k < E else 0.0
                    )
                    append_le(bytes, x)
            for t in range(nt):
                bytes.append(3)
                for k in range(3):
                    append_le(bytes, Int32(triangles.unsafe_get(3 * t + k)))
            f.write_bytes(Span(bytes))


# ===----------------------------------------------------------------------=== #
# Header
# ===----------------------------------------------------------------------=== #

comptime _ASCII = 0
comptime _LITTLE_ENDIAN = 1
comptime _BIG_ENDIAN = 2

# scalar property types
comptime _INT8 = 0
comptime _UINT8 = 1
comptime _INT16 = 2
comptime _UINT16 = 3
comptime _INT32 = 4
comptime _UINT32 = 5
comptime _FLOAT32 = 6
comptime _FLOAT64 = 7


struct _Property(Copyable, Movable):
    var name: String
    var type: Int
    """A scalar type code; the item type of a list."""
    var is_list: Bool
    var count_type: Int

    def __init__(
        out self,
        name: String,
        type: Int,
        is_list: Bool = False,
        count_type: Int = _UINT8,
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
    var format: Int
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
        var format = String()
        self.elements = List[_Element]()
        for line in text.splitlines():
            var words = line.split()
            if len(words) == 0:
                continue
            var key = words[0]
            if key == "format":
                format = String(words[1])
            elif key == "element":
                self.elements.append(_Element(String(words[1]), atol(words[2])))
            elif key == "property":
                if len(self.elements) == 0:
                    raise Error("read_ply: property before any element")
                if words[1] == "list":
                    self.elements[len(self.elements) - 1].properties.append(
                        _Property(
                            String(words[4]),
                            _type_code(String(words[3])),
                            True,
                            _type_code(String(words[2])),
                        )
                    )
                else:
                    self.elements[len(self.elements) - 1].properties.append(
                        _Property(
                            String(words[2]), _type_code(String(words[1]))
                        )
                    )
        if format == "ascii":
            self.format = _ASCII
        elif format == "binary_little_endian":
            self.format = _LITTLE_ENDIAN
        elif format == "binary_big_endian":
            self.format = _BIG_ENDIAN
        else:
            raise Error("read_ply: unsupported format [" + format + "]")


def _type_code(t: String) raises -> Int:
    """The code of a PLY scalar type name."""
    if t == "char" or t == "int8":
        return _INT8
    if t == "uchar" or t == "uint8":
        return _UINT8
    if t == "short" or t == "int16":
        return _INT16
    if t == "ushort" or t == "uint16":
        return _UINT16
    if t == "int" or t == "int32":
        return _INT32
    if t == "uint" or t == "uint32":
        return _UINT32
    if t == "float" or t == "float32":
        return _FLOAT32
    if t == "double" or t == "float64":
        return _FLOAT64
    raise Error("read_ply: unknown property type [" + t + "]")


def _type_size(t: Int) -> Int:
    if t == _FLOAT64:
        return 8
    if t >= _INT32:
        return 4
    if t >= _INT16:
        return 2
    return 1


# ===----------------------------------------------------------------------=== #
# Data
# ===----------------------------------------------------------------------=== #


def _real(mut r: ByteReader, ascii: Bool, t: Int) raises -> Float64:
    """One scalar of type `t`."""
    if ascii:
        return r.float_()
    if t == _FLOAT64:
        return r.binary[DType.float64]()
    if t == _FLOAT32:
        return Float64(r.binary[DType.float32]())
    return Float64(_int(r, ascii, t))


def _int(mut r: ByteReader, ascii: Bool, t: Int) raises -> Int:
    """One scalar of integer type `t` (floats are truncated)."""
    if ascii:
        return Int(r.number[DType.int]())
    if t == _UINT8:
        return Int(r.binary[DType.uint8]())
    if t == _INT32:
        return Int(r.binary[DType.int32]())
    if t == _UINT32:
        return Int(r.binary[DType.uint32]())
    if t == _INT8:
        return Int(r.binary[DType.int8]())
    if t == _INT16:
        return Int(r.binary[DType.int16]())
    if t == _UINT16:
        return Int(r.binary[DType.uint16]())
    return Int(_real(r, ascii, t))


def _skip(mut r: ByteReader, ascii: Bool, prop: _Property) raises:
    """Move past one property's value, or all of a list's items."""
    var n = _int(r, ascii, prop.count_type) if prop.is_list else 1
    if ascii:
        for _ in range(n):
            _ = r.float_()
    else:
        r.skip(n * _type_size(prop.type))


def _read_vertices(
    mut r: ByteReader, ascii: Bool, element: _Element
) raises -> RealArray:
    var axes = List[Int]()  # property position of x, y, z
    for name in ["x", "y", "z"]:
        for i in range(len(element.properties)):
            if element.properties[i].name == name:
                axes.append(i)
    if len(axes) < 2:
        raise Error("read_ply: vertices need at least x and y properties")
    var n = element.count
    var cols = len(axes)
    # a vertex takes at least one byte: check before allocating
    r.require(n)
    # the common binary layouts: exactly x, y, z of one type, in the
    # machine's byte order
    var plain = not ascii and len(element.properties) == cols
    for k in range(cols):
        ref prop = element.properties[k]
        plain = plain and axes[k] == k and not prop.is_list
        plain = plain and prop.type == element.properties[0].type
    plain = plain and r.big_endian != native_little_endian()
    if plain and element.properties[0].type == _FLOAT64:
        var vertices = real_array(n, cols)
        r.binary_into(vertices, 0, n * cols)
        return vertices^
    if plain and element.properties[0].type == _FLOAT32:
        var block = nm.zeros[DType.float32](Shape(n * cols))
        r.binary_into(block, 0, n * cols)
        var vertices = real_array(n, cols)
        for i in range(n * cols):
            vertices.unsafe_set(i, Float64(block.unsafe_get(i)))
        return vertices^
    # the column of each property, or -1
    var column = List[Int](length=len(element.properties), fill=-1)
    for k in range(cols):
        column[axes[k]] = k
    var vertices = real_array(n, cols)
    for v in range(n):
        for i in range(len(element.properties)):
            ref prop = element.properties[i]
            if prop.is_list or column[i] < 0:
                _skip(r, ascii, prop)
            else:
                vertices.unsafe_set(
                    v * cols + column[i], _real(r, ascii, prop.type)
                )
    return vertices^


def _read_faces(
    mut r: ByteReader,
    ascii: Bool,
    element: _Element,
    mut triangles: List[Int],
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
    r.require(element.count)
    triangles.reserve(len(triangles) + 3 * element.count)
    for _ in range(element.count):
        for i in range(len(element.properties)):
            ref prop = element.properties[i]
            if i != index:
                _skip(r, ascii, prop)
                continue
            var n = _int(r, ascii, prop.count_type)
            if n < 3:
                raise Error("read_ply: a face has fewer than 3 vertices")
            # a fan of triangles around the first vertex
            var first = _int(r, ascii, prop.type)
            var prev = _int(r, ascii, prop.type)
            for _ in range(2, n):
                var v = _int(r, ascii, prop.type)
                triangles.append(first)
                triangles.append(prev)
                triangles.append(v)
                prev = v


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
