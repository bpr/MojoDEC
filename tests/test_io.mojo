from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_raises,
    assert_true,
)

import numojo as nm
from numojo.prelude import *

from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import iget, row_list, rows_of, cols_of
from mojodec.routines.io.arrayio import (
    read_array,
    read_sparse,
    write_array,
    write_sparse,
)
from mojodec.routines.io.meshio import read_mesh, write_mesh


def meshes() -> List[String]:
    return [
        "tests/meshes/cube.xml",
        "tests/meshes/sphere3.xml",
        "tests/meshes/torus_tet.xml",
    ]


comptime TMP = "/tmp/mojodec_arrayio_testfile.dat"


def test_read_meshes() raises:
    var cube = read_mesh("tests/meshes/cube.xml")
    assert_equal(rows_of(cube.vertices), 8)
    assert_equal(cols_of(cube.vertices), 3)
    assert_equal(rows_of(cube.elements), 12)
    var torus = read_mesh("tests/meshes/torus_tet.xml")
    assert_equal(rows_of(torus.vertices), 204)
    assert_equal(rows_of(torus.elements), 592)
    assert_equal(cols_of(torus.elements), 4)


def test_read_mesh_fields() raises:
    # a <vertices> node without a <file> must not take the elements' file
    with open("/tmp/mojodec_bad_mesh.xml", "w") as f:
        f.write(
            "<mesh>\n  <vertices/>\n  <elements>\n"
            "    <file name='cube.elements'/>\n  </elements>\n</mesh>\n"
        )
    with assert_raises(contains="missing <file> node for vertices"):
        _ = read_mesh("/tmp/mojodec_bad_mesh.xml")
    with open("/tmp/mojodec_bad_mesh.xml", "w") as f:
        f.write(
            "<mesh>\n  <vertices>\n  </vertices>\n  <elements>\n"
            "    <file name='cube.elements'/>\n  </elements>\n</mesh>\n"
        )
    with assert_raises(contains="missing <file> node for vertices"):
        _ = read_mesh("/tmp/mojodec_bad_mesh.xml")
    # `name` is matched as a whole attribute, not the end of `filename`
    with open("/tmp/mojodec_bad_mesh.xml", "w") as f:
        f.write(
            "<mesh>\n  <vertices>\n    <file filename='x' name='nope.v'/>\n"
            "  </vertices>\n  <elements>\n    <file name='nope.e'/>\n"
            "  </elements>\n</mesh>\n"
        )
    with assert_raises(contains="nope.v"):
        _ = read_mesh("/tmp/mojodec_bad_mesh.xml")


def test_exactness_faces_boundary() raises:
    for ref path in meshes():
        var sc = SimplicialComplex(read_mesh(path), compute_geometry=False)
        var cmplx = sc.chain_complex()
        for k in range(len(cmplx) - 1):
            assert_equal((cmplx[k] @ cmplx[k + 1]).max_abs(), 0.0)
        assert_equal(sc[0].boundary.nrows, 1)
        assert_equal(sc[0].boundary.nnz(), 0)
        for k in range(len(sc) - 1):
            ref face_data = sc[k]
            ref cell_data = sc[k + 1]
            # faces of the cells are exactly the stored faces
            var seen = List[Bool](length=face_data.num_simplices, fill=False)
            ref B = cell_data.boundary
            assert_equal(B.nrows, face_data.num_simplices)
            assert_equal(B.ncols, cell_data.num_simplices)
            assert_equal(B.nnz(), cell_data.num_simplices * (cell_data.dim + 1))
            for r in range(cell_data.num_simplices):
                var s = Simplex(row_list(cell_data.simplices, r))
                var s_index = cell_data.simplex_to_index(s)
                var parity = Int(cell_data.simplex_parity.unsafe_get(r))
                for f in s.boundary():
                    var f_index = face_data.simplex_to_index(f)
                    seen[f_index] = True
                    var expected: Float64 = (
                        -1.0 if (f.parity ^ parity) == 1 else 1.0
                    )
                    assert_equal(B[f_index, s_index], expected)
            for x in seen:
                assert_true(x)


def test_geometry_on_meshes() raises:
    for ref path in meshes():
        var sc = SimplicialComplex(read_mesh(path))
        var n = sc.complex_dimension()
        # dual volumes of vertices partition the total (n-)volume
        var total_primal: Float64 = 0
        var total_dual: Float64 = 0
        for j in range(sc[n].num_simplices):
            total_primal += abs(sc[n].primal_volume.unsafe_get(j))
        for j in range(sc[0].num_simplices):
            total_dual += sc[0].dual_volume.unsafe_get(j)
        assert_almost_equal(total_primal, total_dual, rtol=1e-7)


def test_dense_roundtrip() raises:
    nm.random.seed(0)
    var shapes: List[List[Int]] = [
        [2, 2],
        [3, 3],
        [5, 1],
        [1, 5],
        [2, 2, 2],
        [4, 3, 2],
    ]
    var formats: List[String] = ["binary", "ascii", "basic"]
    for ref sh in shapes:
        var A = nm.random.rand[f64](NDArrayShape(sh))
        var Ai = nm.zeros[DType.int32](NDArrayShape(sh))
        for j in range(Ai.size):
            Ai.unsafe_set(j, Int32(j * 7 - 3))
        for ref fmt in formats:
            if fmt == "basic" and len(sh) > 2:
                continue
            write_array(TMP, A, fmt)
            var B = read_array(TMP)
            for j in range(A.size):
                assert_almost_equal(
                    A.unsafe_get(j), B.unsafe_get(j), atol=1e-12
                )
            write_array(TMP, Ai, fmt)
            var Bi = read_array[DType.int32](TMP)
            for j in range(Ai.size):
                assert_equal(Ai.unsafe_get(j), Bi.unsafe_get(j))


def test_sparse_roundtrip() raises:
    var A = CSRMatrix.from_coo(3, 4, [0, 2, 2], [1, 0, 3], [1.5, -2.0, 3.25])
    var formats: List[String] = ["binary", "ascii"]
    for ref fmt in formats:
        write_sparse(TMP, A, fmt)
        var B = read_sparse(TMP)
        assert_equal(B.nrows, 3)
        assert_equal(B.ncols, 4)
        assert_equal((A - B).max_abs(), 0.0)


def test_mesh_roundtrip() raises:
    var mesh = read_mesh("tests/meshes/sphere3.xml")
    write_mesh("/tmp/mojodec_mesh_test.xml", mesh, "basic")
    var back = read_mesh("/tmp/mojodec_mesh_test.xml")
    for j in range(mesh.elements.size):
        assert_equal(mesh.elements.unsafe_get(j), back.elements.unsafe_get(j))
    for j in range(mesh.vertices.size):
        assert_equal(mesh.vertices.unsafe_get(j), back.vertices.unsafe_get(j))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
