from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from std.memory import bitcast
import numojo as nm
from numojo.prelude import *

from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import iget, row_list, rows_of, cols_of
from mojodec.routines.io.arrayio import (
    ArrayFormat,
    SparseFormat,
    read_array,
    read_sparse,
    write_array,
    write_sparse,
)
from mojodec.routines.io.gmsh import read_gmsh
from mojodec.routines.io.meshio import read_mesh, write_mesh
from mojodec.routines.io.ply import read_ply
from mojodec.routines.io.scan import ByteReader, parse_numbers
from mojodec.routines.io.text import loadtxt
from mojodec.core.utils import real_array, real_vector


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
        for k in range(1, cmplx.complex_dimension()):
            assert_equal(
                (cmplx.boundary(k) @ cmplx.boundary(k + 1)).max_abs(), 0.0
            )
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
        assert_almost_equal(total_primal, total_dual, atol=1e-13, rtol=0)


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
    var formats: List[ArrayFormat] = [.BINARY, .ASCII, .BASIC]
    for ref sh in shapes:
        # NuMojo arrays, of any rank (RealArray is rank 1 or 2)
        var A = nm.random.rand[f64](NDArrayShape(sh))
        var Ai = nm.zeros[DType.int32](NDArrayShape(sh))
        for j in range(Ai.size):
            Ai.unsafe_set(j, Int32(j * 7 - 3))
        for ref fmt in formats:
            if fmt == ArrayFormat.BASIC and len(sh) > 2:
                continue
            write_array(TMP, A, fmt)
            var B = read_array[DType.float64](TMP)
            for j in range(A.size):
                # every format round-trips doubles exactly
                assert_equal(A.unsafe_get(j), B.unsafe_get(j))
            write_array(TMP, Ai, fmt)
            var Bi = read_array[DType.int32](TMP)
            for j in range(Ai.size):
                assert_equal(Ai.unsafe_get(j), Bi.unsafe_get(j))


def test_int64_precision() raises:
    # integers above 2^53 do not survive a trip through Float64
    var A = nm.zeros[DType.int64](Shape(3))
    A.unsafe_set(0, Int64(1) << 62)
    A.unsafe_set(1, (Int64(1) << 53) + 1)
    A.unsafe_set(2, -((Int64(1) << 60) + 7))
    var formats: List[ArrayFormat] = [.BINARY, .ASCII, .BASIC]
    for fmt in formats:
        write_array(TMP, A, fmt)
        var B = read_array[DType.int64](TMP)
        var C = read_array[DType.int](TMP)
        for j in range(3):
            assert_equal(B.unsafe_get(j), A.unsafe_get(j))
            assert_equal(Int64(C.unsafe_get(j)), A.unsafe_get(j))


def test_binary_conversions() raises:
    # stored float32 and int32 read as float64 and int; float64 as int32
    var F = nm.zeros[DType.float32](Shape(2, 2))
    var I = nm.zeros[DType.int32](Shape(4))
    for j in range(4):
        F.unsafe_set(j, Float32(j) + 0.5)
        I.unsafe_set(j, Int32(-j * 1000))
    write_array(TMP, F)
    var Fd = read_array(TMP)
    assert_equal(Fd.unsafe_get(3), 3.5)
    write_array(TMP, I)
    var Ii = read_array[DType.int](TMP)
    assert_equal(Int(Ii.unsafe_get(3)), -3000)
    var Ifl = read_array(TMP)
    assert_equal(Ifl.unsafe_get(2), -2000.0)


def test_truncated_files() raises:
    var A = real_array(4, 3)
    var formats: List[ArrayFormat] = [.BINARY, .ASCII, .BASIC]
    for fmt in formats:
        write_array(TMP, A, fmt)
        var bytes: List[Byte]
        with open(TMP, "r") as f:
            bytes = f.read_bytes()
        with open(TMP, "w") as f:
            f.write_bytes(Span(bytes)[: len(bytes) - 8])
        with assert_raises():
            _ = read_array(TMP)
    # a header claiming far more data than the file holds
    with open(TMP, "w") as f:
        f.write(
            "5\ntype=ndarray\nformat=binary\ndtype=float64\nrank=2\n",
            "dims=100000000,100000000\n",
        )
    with assert_raises(contains="too little data"):
        _ = read_array(TMP)


def test_sparse_roundtrip() raises:
    var A = CSRMatrix.from_coo(3, 4, [0, 2, 2], [1, 0, 3], [1.5, -2.0, 3.25])
    var formats: List[SparseFormat] = [.BINARY, .ASCII]
    for fmt in formats:
        write_sparse(TMP, A, fmt)
        var B = read_sparse(TMP)
        assert_equal(B.nrows, 3)
        assert_equal(B.ncols, 4)
        assert_equal((A - B).max_abs(), 0.0)


def test_empty_sparse_roundtrip() raises:
    # an ASCII array with no values has no data line; the reader used to
    # skip a line anyway and swallow the next array's header
    var A = CSRMatrix(2, 3)
    var formats: List[SparseFormat] = [.BINARY, .ASCII]
    for fmt in formats:
        write_sparse(TMP, A, fmt)
        var B = read_sparse(TMP)
        assert_equal(B.nrows, 2)
        assert_equal(B.ncols, 3)
        assert_equal(B.nnz(), 0)


def test_text_floats_roundtrip_exactly() raises:
    # Mojo prints these integral doubles with a digit too few (the first as
    # 6.793834373838986e+16, which is the double before it)
    var values: List[Float64] = [
        67938343738389864.0,
        -30000000000000004.0,
        1.000000000000064e19 + 2048.0,
        0.1,
        -2.5e-300,
    ]
    var a = real_vector(len(values))
    for i in range(len(values)):
        a.unsafe_set(i, values[i])
    var formats: List[ArrayFormat] = [.ASCII, .BASIC]
    for fmt in formats:
        write_array(TMP, a, fmt)
        var b = read_array(TMP)
        for i in range(len(values)):
            assert_equal(b.unsafe_get(i), values[i])


def _next(mut state: UInt64) -> Int:
    state = state * 6364136223846793005 + 1442695040888963407
    return Int(state >> 33)


def test_scan_numbers_match_atof() raises:
    # tokens on both sides of the one-pass float path's limits (19
    # digits, a significand of 2^53, powers 10^±22), parsed by ByteReader
    # and by atof, bit for bit
    var tokens: List[String] = [
        "9007199254740992",
        "9007199254740993",
        "9007199254740993e-5",
        "1e22",
        "1e23",
        "1e-22",
        "1e-23",
        "123456789012345678e-3",
        "1234567890123456789",
        "12345678901234567890",
        "0.00000000000000000000000001",
        "000000000000000000000012.5",
        "-0.0",
        "-0",
        "+.5e1",
        "1.",
        ".5",
        "1e+5",
        "0.3913043478260859",
        "-9.999999994736442e-08",
        "1e0005",
        "1e400",
        "4.9e-324",
        "-0e-400",
        "0.000e999",
    ]
    var state: UInt64 = 7
    for _ in range(20000):
        var digits = 1 + _next(state) % 20
        var t = String("-") if _next(state) % 2 == 1 else String()
        var point = _next(state) % (digits + 1)
        for k in range(digits):
            if k == point:
                t += "."
            t += String(_next(state) % 10)
        if _next(state) % 2 == 1:
            t += String("e", _next(state) % 61 - 30)
        tokens.append(t)
    var text = String()
    for t in tokens:
        text += t + "\n"
    var r = ByteReader(List[Byte](text.as_bytes()), "test")
    for t in tokens:
        var x = r.float_()
        var y = atof(t)
        assert_equal(
            bitcast[DType.uint64](x), bitcast[DType.uint64](y), String(t)
        )
    # integers: signs, 15 digits (one pass), 18 and 19 digits
    var ints: List[Int] = [0, -0, 7, -42, 999999999999999999, -(10**18) - 3]
    var itext = String("+5 ")
    for v in ints:
        itext += String(v, " ")
    var ri = ByteReader(List[Byte](itext.as_bytes()), "test")
    assert_equal(ri.int_(), 5)
    for v in ints:
        assert_equal(ri.int_(), v)
    for bad in ["5e", "1.5x", "-", "12abc"]:
        var rb = ByteReader(List[Byte](String(bad).as_bytes()), "test")
        with assert_raises(contains="not a"):
            _ = rb.float_() if bad != "12abc" else Float64(rb.int_())


def _tokens_text(tokens: List[String], mut state: UInt64) -> List[Byte]:
    """The tokens, separated by varied white space, then `$End`."""
    var seps: List[String] = [" ", "\n", "\t", "\r\n", "  \n"]
    var text = String()
    for t in tokens:
        text += t + seps[_next(state) % len(seps)]
    text += "$End"
    return List[Byte](text.as_bytes())


def test_parse_numbers_match_reader() raises:
    # the parallel parse gives ByteReader's values bit for bit, however
    # the text is split into chunks (here every core's chunk is a few
    # bytes for the short prefixes)
    var tokens: List[String] = [
        "-0",
        "+0",
        "-0.0",
        "+5",
        "-42",
        "1.",
        ".5",
        "1e+5",
        "1E5",
        "1e400",
        "-0e-400",
        "0.9123456789012345",
        "123456789012345",
        "9007199254740992",
        "-9007199254740992",
        "1234567890123456789e-3",
        "1e0005",
    ]
    var state: UInt64 = 11
    for _ in range(5000):
        var digits = 1 + _next(state) % 20
        var t = String("-") if _next(state) % 2 == 1 else String()
        var point = _next(state) % (digits + 2)
        for k in range(digits):
            if k == point:
                t += "."
            t += String(_next(state) % 10)
        if _next(state) % 3 == 1 or (point >= digits and digits > 15):
            # (integers past 2^53 are left to the reader: tested below)
            t += String("e", _next(state) % 61 - 30)
        tokens.append(t)
    for count in [1, 2, 3, 5, 8, 13, 40, len(tokens)]:
        var some = List[String](tokens[:count])
        var data = _tokens_text(some, state)
        var end = len(data) - 4
        var o = parse_numbers(ByteReader(data.copy(), "test"), 0, end, 0)
        assert_true(Bool(o), String(count))
        var parsed = o.take()
        assert_equal(len(parsed.values), count)
        var r = ByteReader(data^, "test")
        for t in some:
            assert_equal(
                bitcast[DType.uint64](parsed.float_()),
                bitcast[DType.uint64](r.float_()),
                t,
            )
    # integers, keeping the last 2 of each row of 3
    var ints: List[String] = ["7", "-0", "+5", "-42", "999999999999999", "0"]
    var data = _tokens_text(ints, state)
    var o = parse_numbers(ByteReader(data.copy(), "test"), 0, len(data) - 4, 0)
    var parsed = o.take()
    var kept = List[Int]()
    parsed.ints(kept, 2, 1, 2)
    assert_true(kept == [-0, 5, 999999999999999, 0])
    # tokens left to the reader, a token running into the end, and
    # stretches below the threshold give None
    # (subnormals too: the pool's threads flush them to zero)
    for bad in [
        "1 2 abc 3 ",
        "1 4.9e-324 2 ",
        "1 99999999999999999999 2 ",
        "1 9007199254740993 2 ",
        "1 2 0x10 ",
        "1 2 3",
    ]:
        var b = List[Byte]((String(bad) + "$End").as_bytes())
        var n = len(b) - 4
        assert_false(Bool(parse_numbers(ByteReader(b^, "test"), 0, n, 0)), bad)
    var b = List[Byte](String("1 2 3 $End").as_bytes())
    assert_false(Bool(parse_numbers(ByteReader(b^, "test"), 0, 6, 7)))


def test_mesh_roundtrip() raises:
    var mesh = read_mesh("tests/meshes/sphere3.xml")
    write_mesh("/tmp/mojodec_mesh_test.xml", mesh, .BASIC)
    var back = read_mesh("/tmp/mojodec_mesh_test.xml")
    for j in range(mesh.elements.size):
        assert_equal(mesh.elements.unsafe_get(j), back.elements.unsafe_get(j))
    for j in range(mesh.vertices.size):
        assert_equal(mesh.vertices.unsafe_get(j), back.vertices.unsafe_get(j))


def test_hostile_dims_and_narrow_types() raises:
    # dims whose product overflows Int (to 0) once passed the size check
    with open(TMP, "w") as f:
        f.write(
            "5\ntype=ndarray\nformat=binary\ndtype=float64\nrank=3\n",
            "dims=4294967296,4294967296,1\n",
        )
    with assert_raises(contains="too little data"):
        _ = read_array(TMP)
    with open(TMP, "w") as f:
        f.write(
            "5\ntype=ndarray\nformat=binary\ndtype=float64\nrank=2\n",
            "dims=-1,2\n",
        )
    with assert_raises(contains="negative dims"):
        _ = read_array(TMP)
    # a float32 that Mojo prints with a digit too few
    var A = nm.zeros[DType.float32](Shape(1, 1))
    A.unsafe_set(0, bitcast[DType.float32](UInt32(4274181095)))
    var formats: List[ArrayFormat] = [.ASCII, .BASIC]
    for fmt in formats:
        write_array(TMP, A, fmt)
        var B = read_array[DType.float32](TMP)
        assert_equal(bitcast[DType.uint32](B.unsafe_get(0)), UInt32(4274181095))
    # integers past 2^53 through loadtxt
    with open(TMP, "w") as f:
        f.write("9007199254740993\n-9007199254740995\n")
    var C = loadtxt[DType.int](TMP)
    assert_equal(Int(C.unsafe_get(0)), 9007199254740993)
    assert_equal(Int(C.unsafe_get(1)), -9007199254740995)


def test_sparse_dims_are_bounded() raises:
    # a coo file with one entry and 1e14 rows once aborted on allocating
    # the row pointers
    def one_value(dtype: String, value: String) -> String:
        return String(
            "6\ndims=1\ndtype=",
            dtype,
            "\ntype=ndarray\nrank=1\nformat=ascii\nversion=1.0\n",
            value,
            "\n",
        )

    def coo(dims: String) -> String:
        return (
            "5\ntype=sparse\nsptype=coo\ndims="
            + dims
            + "\nformat=ascii\nversion=1.0\n"
            + one_value("float64", "5.0")
            + one_value("int32", "0")
            + one_value("int32", "1")
        )

    with open(TMP, "w") as f:
        f.write(coo("100000000000000,3"))
    with assert_raises(contains="raise max_dim"):
        _ = read_sparse(TMP)
    with open(TMP, "w") as f:
        f.write(coo("2,-3"))
    with assert_raises(contains="outside 0.."):
        _ = read_sparse(TMP)
    with open(TMP, "w") as f:
        f.write(coo("2,3"))
    var A = read_sparse(TMP)
    assert_equal(A.nrows, 2)
    with assert_raises(contains="raise max_dim"):
        _ = read_sparse(TMP, max_dim=2)


def _write_with_bytes(
    path: String, before: String, bad: List[Byte], after: String
) raises:
    """A text file with `bad` (not UTF-8) between two strings."""
    var data = List[Byte]()
    for b in before.as_bytes():
        data.append(b)
    data.extend(bad.copy())
    for b in after.as_bytes():
        data.append(b)
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def test_readers_refuse_invalid_utf8() raises:
    # bytes that are not UTF-8 where text or numbers belong raise reader
    # errors (they used to trip string assertions, found by
    # test_properties_robust's corrupted files)
    var bad: List[Byte] = [0xFF, 0xC3, 0x80, 0xFE]
    var path = "/tmp/mojodec_invalid_utf8.dat"
    _write_with_bytes(path, "1\n", bad, "=3\n")
    with assert_raises():
        _ = read_array(path)
    _write_with_bytes(path, "2\nformat=ascii\n", bad, "\n")
    with assert_raises():
        _ = read_array(path)
    var gmsh = "/tmp/mojodec_invalid_utf8.msh"
    _write_with_bytes(
        gmsh, "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 ", bad, " 1 1\n"
    )
    with assert_raises():
        _ = read_gmsh(gmsh)
    _write_with_bytes(
        gmsh,
        '$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n1\n2 1 "',
        bad,
        '"\n$EndPhysicalNames\n',
    )
    with assert_raises(contains="not valid UTF-8"):
        _ = read_gmsh(gmsh)
    var ply = "/tmp/mojodec_invalid_utf8.ply"
    _write_with_bytes(
        ply, "ply\nformat ascii 1.0\nelement ", bad, " 3\nend_header\n"
    )
    with assert_raises():
        _ = read_ply(ply)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
