from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import SimplicialMesh
from mojodec.core.utils import cols_of, fget, iget, rows_of
from mojodec.routines.io.ply import PlyFormat, read_ply, write_ply
from mojodec.routines.io.scan import append_le
from mojodec.routines.io.text import loadtxt
from mojodec.routines.mesh_generation import icosphere, simplicial_grid_2d

comptime DIR = "tests/meshes/ply/"
comptime TMP = "/tmp/mojodec_ply_test.ply"


def assert_mesh(
    mesh: SimplicialMesh, V: nm.NDArray[f64], T: nm.NDArray[DType.int]
) raises:
    assert_equal(rows_of(mesh.vertices), rows_of(V))
    assert_equal(cols_of(mesh.vertices), cols_of(V))
    assert_equal(rows_of(mesh.elements), rows_of(T))
    for i in range(V.size):
        assert_equal(mesh.vertices.unsafe_get(i), V.unsafe_get(i))
    for i in range(T.size):
        assert_equal(Int(mesh.elements.unsafe_get(i)), Int(T.unsafe_get(i)))


def test_vtk_files_in_every_encoding() raises:
    # written by VTK, with normals and colors to skip
    var V = loadtxt(DIR + "sphere_points.txt")
    var T = loadtxt[DType.int](DIR + "sphere_triangles.txt")
    for name in ["ascii", "binary_le", "binary_be"]:
        assert_mesh(read_ply(DIR + "sphere_" + name + ".ply"), V, T)


def test_round_trip() raises:
    var vt = icosphere(1, 2.0)
    var formats: List[PlyFormat] = [.ASCII, .BINARY]
    for format in formats:
        write_ply(TMP, vt.vertices, vt.elements, format=format)
        assert_mesh(read_ply(TMP), vt.vertices, vt.elements)
    # 2D vertices come back with z = 0
    var grid = simplicial_grid_2d(2)
    write_ply(TMP, grid.vertices, grid.elements)
    var mesh = read_ply(TMP)
    assert_equal(cols_of(mesh.vertices), 3)
    for v in range(rows_of(grid.vertices)):
        for k in range(2):
            assert_equal(fget(mesh.vertices, v, k), fget(grid.vertices, v, k))
        assert_equal(fget(mesh.vertices, v, 2), 0.0)


def test_float_vertices_and_other_index_types() raises:
    # float32 x, y (no z) and uint32 indices with a ushort count, little-
    # and big-endian, plus an extra element to skip
    for big in [False, True]:
        var header = String(
            "ply\nformat ",
            "binary_big_endian" if big else "binary_little_endian",
            " 1.0\nelement vertex 4\nproperty float x\nproperty float y\n",
            "element face 1\nproperty list ushort uint vertex_indices\n",
            "element edge 1\nproperty int a\nproperty list uchar int b\n",
            "end_header\n",
        )
        var body = List[Byte]()

        def put[dtype: DType](x: Scalar[dtype]) {mut body, big}:
            var le = List[Byte]()
            append_le(le, x)
            if big:
                le.reverse()
            body.extend(Span(le))

        var xy: List[Float32] = [0, 0, 1, 0, 1, 1, 0, 1.5]
        for x in xy:
            put(x)
        put(UInt16(4))
        for v in range(4):
            put(UInt32(v))
        put(Int32(7))
        put(UInt8(2))
        put(Int32(1))
        put(Int32(2))
        with open(TMP, "w") as f:
            f.write(header)
            f.write_bytes(Span(body))
        var mesh = read_ply(TMP)
        assert_equal(cols_of(mesh.vertices), 2)
        assert_equal(fget(mesh.vertices, 3, 1), 1.5)
        assert_equal(rows_of(mesh.elements), 2)
        assert_equal(iget(mesh.elements, 1, 0), 0)
        assert_equal(iget(mesh.elements, 1, 1), 2)
        assert_equal(iget(mesh.elements, 1, 2), 3)


def test_polygons_become_fans() raises:
    with open(TMP, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\ncomment a quad and a pentagon\n",
            "element vertex 7\nproperty double x\nproperty double y\n",
            "element face 2\nproperty list uchar int vertex_index\n",
            "property int label\nelement edge 1\nproperty int vertex1\n",
            "property int vertex2\nend_header\n",
            "0 0\n1 0\n1 1\n0 1\n2 0\n2 1\n1.5 2\n",
            "4 0 1 2 3 7\n5 1 4 5 6 2 8\n1 2\n",
        )
    var mesh = read_ply(TMP)
    assert_equal(cols_of(mesh.vertices), 2)
    var expected: List[List[Int]] = [
        [0, 1, 2],
        [0, 2, 3],
        [1, 4, 5],
        [1, 5, 6],
        [1, 6, 2],
    ]
    assert_equal(rows_of(mesh.elements), len(expected))
    for t in range(len(expected)):
        for k in range(3):
            assert_equal(iget(mesh.elements, t, k), expected[t][k])


def test_bad_files() raises:
    with open(TMP, "w") as f:
        f.write("solid ascii stl\n")
    with assert_raises(contains="not a PLY file"):
        _ = read_ply(TMP)
    with open(TMP, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n",
            "property float y\nelement face 1\n",
            "property list uchar int vertex_indices\nend_header\n",
            "0 0\n1 0\n0 1\n3 0 1 5\n",
        )
    with assert_raises(contains="out of range"):
        _ = read_ply(TMP)
    with open(TMP, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n",
            "property float y\nelement face 1\n",
            "property list uchar int vertex_indices\nend_header\n",
            "0 0\n1 0\n0 1\n3 0 1\n",
        )
    with assert_raises(contains="ends early"):
        _ = read_ply(TMP)


def test_hostile_headers_and_counts() raises:
    # a header line without its count
    with open(TMP, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex\nend_header\n")
    with assert_raises(contains="malformed header"):
        _ = read_ply(TMP)
    # an element without properties and an enormous count: skipped at
    # once (it once looped 1e18 times)
    with open(TMP, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement junk 1000000000000000000\n",
            "element vertex 3\nproperty float x\nproperty float y\n",
            "element face 1\nproperty list uchar int vertex_indices\n",
            "end_header\n0 0\n1 0\n0 1\n3 0 1 2\n",
        )
    assert_equal(rows_of(read_ply(TMP).elements), 1)
    # a negative list count in a skipped binary element once moved the
    # reader before the start of the file
    var bytes = List[Byte](
        String(
            "ply\nformat binary_little_endian 1.0\nelement junk 1\n",
            "property list int int stuff\nelement vertex 3\n",
            "property double x\nproperty double y\nelement face 1\n",
            "property list uchar int vertex_indices\nend_header\n",
        ).as_bytes()
    )
    append_le(bytes, Int32(-100000000))
    with open(TMP, "w") as f:
        f.write_bytes(Span(bytes))
    with assert_raises(contains="ends early"):
        _ = read_ply(TMP)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
