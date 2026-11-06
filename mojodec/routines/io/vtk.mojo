# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
VTK files (`mojodec.routines.io.vtk`).
======================================
Writers for VTK's XML unstructured-grid format (`.vtu`), which ParaView and
VisIt open. Not in PyDEC.

`format` (a `VTKFormat`) chooses how the data arrays are stored:

- `.ASCII` (the default): readable text.
- `.BINARY`: base64 inside the XML, so the file is still valid XML.
- `.APPENDED`: raw bytes after the XML; the smallest and fastest to read.

Binary values are little-endian Float64, Int64, and UInt8, each array
preceded by its byte count as a UInt64, as VTK's `header_type="UInt64"`
specifies.

Cochains are written as data on the cells or points they live on: a
k-cochain is cell data on the k-simplices, and a vector field sampled by
`simplex_quivers` is cell data on the top simplices. 2D points and vectors
are padded to 3D with zeros.

```mojo
from mojodec import write_vtu, VTKField

# a 1-cochain h on the edges, and the vertices' x coordinates
write_vtu(
    "h.vtu",
    sc.vertices(),
    sc[1].simplices,
    point_data=[VTKField("x", x)],
    cell_data=[VTKField("h", h)],
    format=.APPENDED,
)
```
"""

from std.base64 import b64encode
from std.sys import size_of

from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.utils import IntArray, RealArray, cols_of, int_array, rows_of
from mojodec.routines.cube_array import cube_array_search
from mojodec.routines.io.scan import append_le


struct VTKFormat(Equatable, ImplicitlyCopyable, Writable):
    """How `write_vtu` and `write_vtu_cubes` store the data arrays."""

    var _code: UInt8

    comptime ASCII = VTKFormat(0)
    """Readable text."""
    comptime BINARY = VTKFormat(1)
    """Base64 inside the XML."""
    comptime APPENDED = VTKFormat(2)
    """Raw bytes after the XML."""

    def __init__(out self, code: UInt8):
        """A format from its code; use the named members instead.

        Args:
            code: 0, 1, or 2.
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
        """Write the name VTK uses in a `DataArray`'s `format` attribute.

        Args:
            writer: The destination.
        """
        if self == Self.ASCII:
            writer.write("ascii")
        elif self == Self.BINARY:
            writer.write("binary")
        else:
            writer.write("appended")


struct VTKField(Copyable, Movable):
    """A named data array: one value or vector per point or cell."""

    var name: String
    """The array's name, as ParaView lists it."""
    var values: RealArray
    """A vector of `n` scalars, or an `n x c` array of `c <= 3` components."""

    def __init__(out self, name: String, values: RealArray):
        """A field named `name` holding a copy of `values`.

        Args:
            name: The array's name.
            values: A vector of `n` scalars, or an `n x c` array of
                `c <= 3` components (vectors are padded to 3).
        """
        self.name = name
        self.values = values.copy()


def write_vtu(
    path: String,
    vertices: RealArray,
    cells: IntArray,
    point_data: List[VTKField] = [],
    cell_data: List[VTKField] = [],
    format: VTKFormat = .ASCII,
) raises:
    """Write a simplicial mesh, with optional data, as a VTK `.vtu` file.

    `vertices` is `num_vertices x E` with `E <= 3`. Each row of `cells`
    lists the vertices of one k-simplex, `0 <= k <= 3`: vertices, edges,
    triangles, or tetrahedra.

    Args:
        path: The file to write; an existing file is replaced.
        vertices: Vertex coordinates, `num_vertices x E` with `E <= 3`.
        cells: The simplices, one row of `k + 1` vertex indices each.
        point_data: Fields with one row per vertex.
        cell_data: Fields with one row per cell.
        format: `ASCII`, `BINARY` (base64), or `APPENDED` (raw bytes).

    Raises:
        If `cells` is not 2-D, has more than 4 columns, or uses a vertex
        index out of range; `vertices` is not 2-D with at most 3 columns;
        a field has more than 3 columns or the wrong
        number of rows; or the file cannot be written.
    """
    if cells.ndim != 2:
        raise Error("write_vtu: cells must be a 2-D array")
    var n = cols_of(cells)
    if n < 1 or n > 4:
        raise Error("write_vtu: cells must be k-simplices with 0 <= k <= 3")
    var connectivity = List[Int](capacity=cells.size)
    for i in range(cells.size):
        var v = Int(cells.unsafe_get(i))
        if v < 0 or v >= rows_of(vertices):
            raise Error("write_vtu: cell vertex index out of range")
        connectivity.append(v)
    var simplex_types: List[Int] = [_VERTEX, _LINE, _TRIANGLE, _TETRA]
    _write(
        path,
        vertices,
        connectivity,
        n,
        simplex_types[n - 1],
        point_data,
        cell_data,
        format,
    )


def write_vtu_cubes(
    path: String,
    cmplx: RegularCubeComplex,
    k: Int,
    point_data: List[VTKField] = [],
    cell_data: List[VTKField] = [],
    format: VTKFormat = .ASCII,
) raises:
    """Write the k-cubes of a regular cube complex as a VTK `.vtu` file.

    The points are the complex's vertices (`cmplx.vertices`, in the order of
    the 0-cubes) and the cells are its k-cubes in `cmplx[k].cube_array`
    order, so `point_data` holds 0-cochains and `cell_data` k-cochains.
    Cells are VTK lines, pixels, or voxels.

    Args:
        path: The file to write; an existing file is replaced.
        cmplx: The cube complex, of dimension at most 3.
        k: The dimension of the cubes to write, `0 <= k <= min(N, 3)`.
        point_data: Fields with one row per vertex.
        cell_data: Fields with one row per k-cube.
        format: `ASCII`, `BINARY` (base64), or `APPENDED` (raw bytes).

    Raises:
        If `k` is out of range; the complex has dimension above 3 (its
        vertices have too many coordinates); a field
        has more than 3 columns or the wrong number of rows; or the file
        cannot be written.
    """
    var N = cmplx.complex_dimension()
    if k < 0 or k > min(N, 3):
        raise Error("write_vtu_cubes: need 0 <= k <= min(dimension, 3)")
    ref cubes = cmplx[k].cube_array
    var num_cubes = rows_of(cubes)
    var corners = 1 << k
    # the corners of every cube, as 0-cubes, in VTK's x-fastest order
    var points = int_array(num_cubes * corners, N)
    for c in range(num_cubes):
        for b in range(corners):
            var r = c * corners + b
            for i in range(N):
                points.unsafe_set(r * N + i, cubes.unsafe_get(c * (N + k) + i))
            for j in range(k):
                if (b >> j) & 1 == 1:
                    var axis = Int(cubes.unsafe_get(c * (N + k) + N + j))
                    points.unsafe_set(
                        r * N + axis, points.unsafe_get(r * N + axis) + 1
                    )
    var index = cube_array_search(cmplx[0].cube_array, points)
    var connectivity = List[Int](capacity=index.size)
    for i in range(index.size):
        connectivity.append(Int(index.unsafe_get(i)))
    var cube_types: List[Int] = [_VERTEX, _LINE, _PIXEL, _VOXEL]
    _write(
        path,
        cmplx.vertices,
        connectivity,
        corners,
        cube_types[k],
        point_data,
        cell_data,
        format,
    )


comptime _VERTEX = 1
comptime _LINE = 3
comptime _TRIANGLE = 5
comptime _PIXEL = 8
comptime _TETRA = 10
comptime _VOXEL = 11


def _write(
    path: String,
    vertices: RealArray,
    connectivity: List[Int],
    verts_per_cell: Int,
    cell_type: Int,
    point_data: List[VTKField],
    cell_data: List[VTKField],
    format: VTKFormat,
) raises:
    if vertices.ndim != 2 or cols_of(vertices) > 3:
        raise Error("write_vtu: vertices must be N x E with E <= 3")
    var num_points = rows_of(vertices)
    var num_cells = len(connectivity) // verts_per_cell
    var enc = _Encoder(format)
    enc.xml.write(
        '<?xml version="1.0"?>\n',
        '<VTKFile type="UnstructuredGrid" version="1.0"',
        ' byte_order="LittleEndian" header_type="UInt64">\n',
        "<UnstructuredGrid>\n",
        '<Piece NumberOfPoints="',
        num_points,
        '" NumberOfCells="',
        num_cells,
        '">\n',
    )
    _write_fields(enc, "PointData", point_data, num_points)
    _write_fields(enc, "CellData", cell_data, num_cells)
    enc.xml += "<Points>\n"
    enc.reals("", vertices, padded=True)
    enc.xml += "</Points>\n<Cells>\n"
    enc.ints[DType.int64]("connectivity", connectivity, verts_per_cell)
    var offsets = List[Int](capacity=num_cells)
    for c in range(num_cells):
        offsets.append((c + 1) * verts_per_cell)
    enc.ints[DType.int64]("offsets", offsets, 1)
    enc.ints[DType.uint8](
        "types", List[Int](length=num_cells, fill=cell_type), 1
    )
    enc.xml += "</Cells>\n</Piece>\n</UnstructuredGrid>\n"
    with open(path, "w") as f:
        f.write(enc.xml)
        if format == VTKFormat.APPENDED:
            f.write('<AppendedData encoding="raw">\n_')
            f.write_bytes(Span(enc.appended))
            f.write("\n</AppendedData>\n")
        f.write("</VTKFile>\n")


struct _Encoder(Movable):
    """Writes `DataArray` elements in one of the three formats.

    `appended` collects the raw blocks that follow the XML in the
    `"appended"` format; each array's `offset` points into it.
    """

    var xml: String
    var format: VTKFormat
    var appended: List[Byte]

    def __init__(out self, format: VTKFormat):
        self.xml = String()
        self.format = format
        self.appended = List[Byte]()

    def reals(mut self, name: String, values: RealArray, padded: Bool) raises:
        """Scalars, or rows padded to 3 components if `padded`."""
        var rows = rows_of(values)
        var cols = 1 if values.ndim == 1 else cols_of(values)
        var components = 3 if padded else cols
        self._open("Float64", name, components)
        var block = _block(8 * rows * components)
        for r in range(rows):
            for c in range(components):
                var x: Float64 = 0
                if c < cols:
                    x = values.unsafe_get(r * cols + c)
                if self.format == VTKFormat.ASCII:
                    self.xml.write(x, "\n" if c == components - 1 else " ")
                else:
                    append_le(block, x)
        self._close(block^)

    def ints[
        dtype: DType
    ](mut self, name: String, values: List[Int], per_line: Int) raises:
        """Integers as `dtype`, `per_line` to a line in ASCII."""
        comptime size = size_of[dtype]()
        self._open(_vtk_type[dtype](), name, 1)
        var block = _block(size * len(values))
        for i in range(len(values)):
            if self.format == VTKFormat.ASCII:
                self.xml.write(
                    values[i], "\n" if (i + 1) % per_line == 0 else " "
                )
            else:
                append_le(block, Scalar[dtype](values[i]))
        self._close(block^)

    def _open(mut self, type_name: String, name: String, components: Int):
        self.xml += '<DataArray type="' + type_name + '"'
        if name:
            self.xml += ' Name="' + _escape(name) + '"'
        self.xml.write(
            ' NumberOfComponents="', components, '" format="', self.format, '"'
        )
        if self.format == VTKFormat.APPENDED:
            self.xml.write(' offset="', len(self.appended), '"')
        self.xml += ">\n"

    def _close(mut self, var block: List[Byte]):
        if self.format != VTKFormat.ASCII:
            # the UInt64 byte count that heads every block
            var count = UInt64(len(block) - 8)
            for b in range(8):
                block[b] = Byte((count >> UInt64(8 * b)) & 0xFF)
            if self.format == VTKFormat.BINARY:
                # b64encode overwrites its result, so encode separately
                var encoded = String()
                b64encode(Span(block), encoded)
                self.xml += encoded + "\n"
            else:
                self.appended.extend(Span(block))
        self.xml += "</DataArray>\n"


def _write_fields(
    mut enc: _Encoder, tag: String, fields: List[VTKField], count: Int
) raises:
    if len(fields) == 0:
        return
    enc.xml += "<" + tag + ">\n"
    for ref field in fields:
        ref v = field.values
        if v.ndim > 2 or (v.ndim == 2 and cols_of(v) > 3):
            raise Error(
                "write_vtu: field '"
                + field.name
                + "' must be a vector or have at most 3 columns"
            )
        if rows_of(v) != count:
            raise Error(
                "write_vtu: field '"
                + field.name
                + "' has "
                + String(rows_of(v))
                + " rows, expected "
                + String(count)
            )
        enc.reals(field.name, v, padded=v.ndim == 2 and cols_of(v) > 1)
    enc.xml += "</" + tag + ">\n"


def _block(data_bytes: Int) -> List[Byte]:
    """An empty binary block, with room for its header and `data_bytes`."""
    var block = List[Byte](capacity=8 + data_bytes)
    for _ in range(8):
        block.append(0)
    return block^


def _vtk_type[dtype: DType]() -> String:
    comptime if dtype == DType.int64:
        return "Int64"
    else:
        comptime assert dtype == DType.uint8, "unsupported VTK integer type"
        return "UInt8"


def _escape(s: String) -> String:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
