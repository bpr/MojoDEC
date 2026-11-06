from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.mesh import SimplicialMesh
from mojodec.core.utils import cols_of, fget, iget, rows_of
from mojodec.routines.io.ply import read_ply, write_ply
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
        assert_almost_equal(
            mesh.vertices.unsafe_get(i), V.unsafe_get(i), atol=0
        )
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
    for format in ["ascii", "binary"]:
        write_ply(TMP, vt[0], vt[1], format=format)
        assert_mesh(read_ply(TMP), vt[0], vt[1])
    # 2D vertices come back with z = 0
    var grid = simplicial_grid_2d(2)
    write_ply(TMP, grid[0], grid[1])
    var mesh = read_ply(TMP)
    assert_equal(cols_of(mesh.vertices), 3)
    for v in range(rows_of(grid[0])):
        for k in range(2):
            assert_equal(fget(mesh.vertices, v, k), fget(grid[0], v, k))
        assert_equal(fget(mesh.vertices, v, 2), 0.0)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
