from std.base64 import b64decode
from std.memory import bitcast
from std.utils.numerics import inf, nan
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import RegularCubeMesh
from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.utils import real_vector
from mojodec.routines.io.vtk import (
    VTKField,
    VTKFormat,
    write_vtu,
    write_vtu_cubes,
)

comptime TMP = "/tmp/mojodec_vtk_test.vtu"


def read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def block(xml: String, name: String) raises -> String:
    """The text of the DataArray named `name`."""
    var start = xml.find('Name="' + name + '"')
    assert_true(start >= 0, "no DataArray " + name)
    var body = xml.find(">", start) + 1
    return String(xml[byte = body : xml.find("</DataArray>", body)].strip())


def test_triangles() raises:
    var V = nm.fromstring[f64]("[[0,0],[1,0],[1,1],[0,1]]")
    var S = nm.fromstring[DType.int]("[[0,1,3],[1,2,3]]")
    var p = real_vector(2)
    p.unsafe_set(0, 0.5)
    p.unsafe_set(1, -2.0)
    var u = nm.fromstring[f64]("[[1,0],[0,1]]")
    write_vtu(
        TMP,
        V,
        S,
        point_data=[VTKField("x", nm.fromstring[f64]("[0,1,1,0]"))],
        cell_data=[VTKField("p", p), VTKField("u<&>", u)],
    )
    var xml = read(TMP)
    assert_true('NumberOfPoints="4" NumberOfCells="2"' in xml)
    assert_equal(block(xml, "connectivity"), "0 1 3\n1 2 3")
    assert_equal(block(xml, "offsets"), "3\n6")
    assert_equal(block(xml, "types"), "5\n5")
    assert_equal(block(xml, "p"), "0.5\n-2.0")
    # 2D vectors are padded to 3 components, and names are escaped
    assert_true('Name="u&lt;&amp;&gt;" NumberOfComponents="3"' in xml)
    assert_equal(block(xml, "u&lt;&amp;&gt;"), "1.0 0.0 0.0\n0.0 1.0 0.0")
    assert_true("<PointData>" in xml and "<CellData>" in xml)


def triangle_mesh(format: VTKFormat) raises:
    var p = real_vector(2)
    p.unsafe_set(0, 0.5)
    p.unsafe_set(1, -2.0)
    write_vtu(
        TMP,
        nm.fromstring[f64]("[[0,0],[1,0],[1,1],[0,1]]"),
        nm.fromstring[DType.int]("[[0,1,3],[1,2,3]]"),
        cell_data=[VTKField("p", p)],
        format=format,
    )


def le(bytes: List[Byte], start: Int, size: Int) -> UInt64:
    var u: UInt64 = 0
    for b in range(size):
        u |= UInt64(bytes[start + b]) << UInt64(8 * b)
    return u


def check_block(
    bytes: List[Byte], start: Int, size: Int, expected: List[UInt64]
) raises:
    """A UInt64 byte count, then `expected` as `size`-byte values."""
    assert_equal(le(bytes, start, 8), UInt64(size * len(expected)))
    for i in range(len(expected)):
        assert_equal(le(bytes, start + 8 + size * i, size), expected[i])


def test_binary() raises:
    triangle_mesh(.BINARY)
    var xml = read(TMP)
    assert_true('Name="p" NumberOfComponents="1" format="binary"' in xml)
    var p = b64decode(block(xml, "p"))
    check_block(
        p, 0, 8, [bitcast[DType.uint64](0.5), bitcast[DType.uint64](-2.0)]
    )
    var conn = b64decode(block(xml, "connectivity"))
    check_block(conn, 0, 8, [0, 1, 3, 1, 2, 3])
    var types = b64decode(block(xml, "types"))
    check_block(types, 0, 1, [5, 5])


def test_appended() raises:
    triangle_mesh(.APPENDED)
    var bytes: List[Byte]
    with open(TMP, "r") as f:
        bytes = f.read_bytes()
    var text = String(unsafe_from_utf8=bytes)
    var marker = '<AppendedData encoding="raw">\n_'
    var at = text.find(marker)
    assert_true(at >= 0)
    var base = at + marker.byte_length()

    def offset(name: String) raises {imm text} -> Int:
        var tag = text.find('Name="' + name + '"')
        var o = text.find('offset="', tag) + 8
        return atol(text[byte = o : text.find('"', o)])

    check_block(
        bytes,
        base + offset("p"),
        8,
        [bitcast[DType.uint64](0.5), bitcast[DType.uint64](-2.0)],
    )
    check_block(bytes, base + offset("connectivity"), 8, [0, 1, 3, 1, 2, 3])
    check_block(bytes, base + offset("offsets"), 8, [3, 6])
    check_block(bytes, base + offset("types"), 1, [5, 5])
    assert_true(text.endswith("\n</AppendedData>\n</VTKFile>\n"))


def test_edges_and_tets() raises:
    var V = nm.fromstring[f64]("[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]")
    write_vtu(TMP, V, nm.fromstring[DType.int]("[[0,1],[2,3]]"))
    var xml = read(TMP)
    assert_equal(block(xml, "types"), "3\n3")
    assert_true("<PointData>" not in xml)
    write_vtu(TMP, V, nm.fromstring[DType.int]("[[0,1,2,3]]"))
    assert_equal(block(read(TMP), "types"), "10")


def test_cubes() raises:
    # a 2x1 grid of squares
    var rcc = RegularCubeComplex(
        RegularCubeMesh(nm.ones[DType.int](Shape(2, 1)))
    )
    write_vtu_cubes(TMP, rcc, 2)
    var xml = read(TMP)
    assert_true('NumberOfPoints="6" NumberOfCells="2"' in xml)
    assert_equal(block(xml, "types"), "8\n8")
    # pixel corners in x-fastest order, as indices into rcc.vertices
    var conn = block(xml, "connectivity").split("\n")
    for c in range(2):
        var ids = conn[c].strip().split(" ")
        var corner = List[List[Float64]]()
        for ref s in ids:
            var v = atol(s)
            corner.append(
                [
                    rcc.vertices.unsafe_get(2 * v),
                    rcc.vertices.unsafe_get(2 * v + 1),
                ]
            )
        assert_equal(corner[1][0] - corner[0][0], 1.0)
        assert_equal(corner[1][1], corner[0][1])
        assert_equal(corner[2][0], corner[0][0])
        assert_equal(corner[2][1] - corner[0][1], 1.0)
        assert_equal(corner[3][0], corner[1][0])
        assert_equal(corner[3][1], corner[2][1])
    write_vtu_cubes(TMP, rcc, 1)
    xml = read(TMP)
    assert_true('NumberOfCells="7"' in xml)
    assert_equal(block(xml, "types").count("3"), 7)


def test_bad_inputs() raises:
    var V = nm.fromstring[f64]("[[0,0],[1,0],[0,1]]")
    var S = nm.fromstring[DType.int]("[[0,1,2]]")
    with assert_raises(contains="expected 1"):
        write_vtu(TMP, V, S, cell_data=[VTKField("p", real_vector(2))])
    with assert_raises(contains="out of range"):
        write_vtu(TMP, V, nm.fromstring[DType.int]("[[0,1,3]]"))
    with assert_raises(contains="E <= 3"):
        write_vtu(TMP, nm.zeros[f64](Shape(3, 4)), S)
    var rcc = RegularCubeComplex(
        RegularCubeMesh(nm.ones[DType.int](Shape(1, 1)))
    )
    with assert_raises(contains="0 <= k"):
        write_vtu_cubes(TMP, rcc, 3)


def test_non_finite_values() raises:
    # VTK's ASCII reader cannot parse nan or inf, which were once written
    var V = nm.fromstring[f64]("[[0,0],[1,0],[1,1],[0,1]]")
    var S = nm.fromstring[DType.int]("[[0,1,3],[1,2,3]]")
    var p = real_vector(2)
    p.unsafe_set(1, nan[DType.float64]())
    with assert_raises(contains="field p holds nan"):
        write_vtu(TMP, V, S, cell_data=[VTKField("p", p)])
    var W = V.copy()
    W.unsafe_set(3, inf[DType.float64]())
    with assert_raises(contains="a point holds inf"):
        write_vtu(TMP, W, S)
    # the binary formats carry them exactly
    var formats: List[VTKFormat] = [.BINARY, .APPENDED]
    for fmt in formats:
        write_vtu(TMP, W, S, cell_data=[VTKField("p", p)], format=fmt)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
