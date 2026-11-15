from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from mojodec import (
    GmshMesh,
    PRIMAL,
    SimplicialComplex,
    Subset,
    TypedComplex,
    TypedSubset,
    kuhn_grid,
    read_gmsh,
)
from mojodec.core.utils import cols_of, fget, iget, row_list, rows_of
from std.memory import bitcast

from mojodec.routines.io.gmsh import _Raw, _numbering_warning, _parse

# Fixtures: tests/meshes/gmsh/*.geo, meshed by Gmsh 4.15 in each format.
comptime DIR = "tests/meshes/gmsh/"
comptime TMP = "/tmp/mojodec_gmsh_test.msh"


def _square(fmt: String) raises -> GmshMesh:
    return read_gmsh(DIR + "square_" + fmt + ".msh", dim=2)


def test_formats_agree() raises:
    for name in ["square", "cube"]:
        var ref_ = read_gmsh(DIR + name + "_v41.msh")
        for fmt in ["v41_bin", "v22", "v22_bin"]:
            var m = read_gmsh(DIR + name + "_" + fmt + ".msh")
            assert_equal(rows_of(m.vertices), rows_of(ref_.vertices))
            assert_equal(cols_of(m.vertices), cols_of(ref_.vertices))
            for i in range(m.vertices.size):
                assert_almost_equal(
                    m.vertices.unsafe_get(i),
                    ref_.vertices.unsafe_get(i),
                    atol=1e-15,
                    rtol=0,
                )
            assert_equal(m.elements.size, ref_.elements.size)
            for i in range(m.elements.size):
                assert_equal(
                    m.elements.unsafe_get(i), ref_.elements.unsafe_get(i)
                )
            assert_equal(m.node_tags, ref_.node_tags)
            assert_equal(len(m.groups), len(ref_.groups))
            for g in range(len(m.groups)):
                assert_equal(m.groups[g].name, ref_.groups[g].name)
                assert_equal(m.groups[g].dim, ref_.groups[g].dim)
                assert_equal(m.groups[g].tag, ref_.groups[g].tag)
                ref a = m.groups[g].cells
                ref b = ref_.groups[g].cells
                assert_equal(a.size, b.size)
                assert_equal(cols_of(a), m.groups[g].dim + 1)
                for i in range(a.size):
                    assert_equal(a.unsafe_get(i), b.unsafe_get(i))


def _all_vertices(
    sc: SimplicialComplex, s: Subset, coord: Int, value: Float64
) raises -> Bool:
    """Whether every vertex of the cells of `s` has the given coordinate."""
    var v = s.copy()
    while v.dim > 0:
        v = v.boundary(sc)
    for i in v.indices():
        if abs(fget(sc.vertices(), i, coord) - value) > 1e-12:
            return False
    return True


def test_square_groups() raises:
    var msh = _square("v41")
    assert_equal(cols_of(msh.vertices), 2)  # z dropped
    var sc = SimplicialComplex(msh.mesh())
    assert_equal(sc.complex_dimension(), 2)
    assert_equal(
        msh.group_names(),
        ["corner", "inlet", "outlet", "walls", "interface", "left", "right"],
    )
    # boundary pieces sit where the .geo file puts them
    var inlet = msh.subset(sc, "inlet", 1)
    var outlet = msh.subset(sc, "outlet", 1)
    var walls = msh.subset(sc, "walls", 1)
    assert_equal(len(inlet), 4)
    assert_true(_all_vertices(sc, inlet, 0, 0.0))
    assert_true(_all_vertices(sc, outlet, 0, 1.0))
    assert_equal(len(msh.subset(sc, "inlet", 0)), 5)
    # together they make up the whole boundary; the interface is inside
    assert_true((inlet | outlet | walls) == sc.boundary_cells()[1])
    var interface = msh.subset(sc, "interface", 1)
    assert_true((interface & sc.boundary_cells()[1]).is_empty())
    assert_true(_all_vertices(sc, interface, 0, 0.5))
    # the two regions split the triangles, and meet at the interface
    var left = msh.subset(sc, "left", 2)
    var right = msh.subset(sc, "right", 2)
    assert_true((left & right).is_empty())
    assert_true((left | right) == Subset.full(sc, 2))
    for t in left.indices():
        assert_true(fget(sc[2].circumcenter, t, 0) < 0.5)
    assert_true(
        (left.manifold_boundary(sc) & right.manifold_boundary(sc)) == interface
    )
    # a region's subset in a lower dimension holds the faces of its cells
    assert_true(msh.subset(sc, "left", 1) == left.boundary(sc))
    # the corner point and the unnamed group 42 (the lower-left wall)
    var corner = msh.subset(sc, "corner", 0)
    assert_equal(len(corner), 1)
    var c = corner.indices()[0]
    assert_equal(fget(sc.vertices(), c, 0), 0.0)
    assert_equal(fget(sc.vertices(), c, 1), 0.0)
    var g42 = msh.subset(sc, group_dim=1, tag=42, dim=1)
    assert_equal(len(g42), 2)
    assert_true((g42 - walls).is_empty())
    assert_true(_all_vertices(sc, g42, 1, 0.0))
    assert_equal(msh.group("walls").num_cells(), 8)


def test_typed_subsets() raises:
    var msh = _square("v41")
    var tc = TypedComplex[2](msh.mesh())
    var inlet: TypedSubset[2, 0, PRIMAL] = msh.subset[0](tc, "inlet")
    assert_true(inlet.untyped() == msh.subset(tc.sc, "inlet", 0))
    var left: TypedSubset[2, 2, PRIMAL] = msh.subset[2](tc, "left")
    assert_true(left.untyped() == msh.subset(tc.sc, "left", 2))
    var g42 = msh.subset[1](tc, group_dim=1, tag=42)
    assert_equal(len(g42), 2)
    with assert_raises(contains="no physical group"):
        _ = msh.subset[1](tc, "nowhere")
    # msh.subset[3](tc, "left") does not compile: k exceeds n = 2


def test_square_errors() raises:
    var msh = _square("v41")
    var sc = SimplicialComplex(msh.mesh())
    with assert_raises(contains="no physical group named 'nope'"):
        _ = msh.subset(sc, "nope", 1)
    with assert_raises(contains="no cells of dimension >= 2"):
        _ = msh.subset(sc, "inlet", 2)
    with assert_raises():
        _ = msh.subset(sc, group_dim=1, tag=43, dim=1)
    with assert_raises():
        _ = msh.group("nope")
    # tagging only boundary curves leaves out the triangles
    with assert_raises(contains="Mesh.SaveAll"):
        _ = read_gmsh(DIR + "square_edges_only.msh", dim=2)
    var edges = read_gmsh(DIR + "square_edges_only.msh")
    assert_equal(cols_of(edges.elements), 2)


def test_cube() raises:
    var msh = read_gmsh(DIR + "cube_v41_bin.msh", dim=3)
    assert_equal(cols_of(msh.vertices), 3)
    var sc = SimplicialComplex(msh.mesh())
    var total: Float64 = 0
    for t in range(sc[3].num_simplices):
        total += sc[3].primal_volume.unsafe_get(t)
    assert_almost_equal(abs(total), 1.0, atol=1e-14, rtol=0)
    for pair in [("bottom", 0.0), ("top", 1.0)]:
        var faces = msh.subset(sc, pair[0], 2)
        assert_true(_all_vertices(sc, faces, 2, pair[1]))
        assert_true((faces - sc.boundary_cells()[2]).is_empty())
        var area: Float64 = 0
        for f in faces.indices():
            area += sc[2].primal_volume.unsafe_get(f)
        assert_almost_equal(area, 1.0, atol=1e-14, rtol=0)
    assert_true(msh.subset(sc, "solid", 3) == Subset.full(sc, 3))


def _circumcenter_sum(
    sc: SimplicialComplex, cells: Subset, dim: Int
) raises -> List[Float64]:
    var out = List[Float64](length=cols_of(sc.vertices()), fill=0)
    for i in cells.indices():
        for a in range(len(out)):
            out[a] += fget(sc[dim].circumcenter, i, a)
    return out^


def test_reorder_vertices() raises:
    for path in [DIR + "square_v41.msh", DIR + "cube_v41_bin.msh"]:
        var msh = read_gmsh(path)
        var r = msh.reorder_vertices()
        var nv = rows_of(msh.vertices)
        assert_equal(rows_of(r.vertices), nv)
        assert_equal(len(r.node_tags), nv)
        # each vertex keeps its node tag
        for i in range(nv):
            var found = False
            for j in range(nv):
                if msh.node_tags[j] == r.node_tags[i]:
                    for a in range(cols_of(msh.vertices)):
                        assert_equal(
                            fget(r.vertices, i, a), fget(msh.vertices, j, a)
                        )
                    found = True
            assert_true(found)
        # every group names the same cells
        var sc = SimplicialComplex(msh.mesh())
        var sr = SimplicialComplex(r.mesh())
        var n = sc.complex_dimension()
        assert_equal(len(r.groups), len(msh.groups))
        for gi in range(len(msh.groups)):
            ref g = msh.groups[gi]
            assert_equal(r.groups[gi].name, g.name)
            assert_equal(r.groups[gi].num_cells(), g.num_cells())
            for dim in range(min(g.dim, n) + 1):
                var a = msh.subset(sc, group_dim=g.dim, tag=g.tag, dim=dim)
                var b = r.subset(sr, group_dim=g.dim, tag=g.tag, dim=dim)
                assert_equal(len(a), len(b))
                var ca = _circumcenter_sum(sc, a, dim)
                var cb = _circumcenter_sum(sr, b, dim)
                for x in range(len(ca)):
                    # sums in another order
                    assert_almost_equal(ca[x], cb[x], atol=1e-12, rtol=1e-13)


def test_numbering_warning() raises:
    # 82,944 tetrahedra in grid order, then numbered by v -> 7919 v mod nv
    # (a bijection: nv = 25^3), which scatters neighbors
    var mesh = kuhn_grid([24, 24, 24])
    var nv = rows_of(mesh.vertices)
    assert_equal(_numbering_warning("grid.msh", mesh.elements, nv), "")
    var scrambled = mesh.elements.copy()
    for i in range(scrambled.size):
        var v = Int(scrambled.unsafe_get(i))
        scrambled.unsafe_set(i, Scalar[DType.int](v * 7919 % nv))
    var message = _numbering_warning("grid.msh", scrambled, nv)
    assert_true("vertex numbering of grid.msh is scattered" in message)
    assert_true("reorder_vertices()" in message)
    # small meshes are not checked
    var small = kuhn_grid([4, 4, 4])
    var few = small.elements.copy()
    var n = rows_of(small.vertices)
    for i in range(few.size):
        few.unsafe_set(i, Scalar[DType.int](Int(few.unsafe_get(i)) * 7 % n))
    assert_equal(_numbering_warning("small.msh", few, n), "")
    # warn=False reads the same mesh
    var a = read_gmsh(DIR + "cube_v41.msh")
    var b = read_gmsh(DIR + "cube_v41.msh", warn=False)
    assert_equal(rows_of(a.elements), rows_of(b.elements))


def _write(text: String) raises:
    with open(TMP, "w") as f:
        f.write(text)


def test_malformed() raises:
    _write("$MeshFormat\n4.0 0 8\n$EndMeshFormat\n")
    with assert_raises(contains="not supported"):
        _ = read_gmsh(TMP)
    # a quadrangle
    _write(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n1 0 0 0\n2 1 0 0\n"
        "3 1 1 0\n4 0 1 0\n$EndNodes\n$Elements\n1\n1 3 2 0 1 1 2 3 4\n"
        "$EndElements\n"
    )
    with assert_raises(contains="quadrangles"):
        _ = read_gmsh(TMP)
    # an element on a missing node
    _write(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n2\n1 0 0 0\n2 1 0 0\n"
        "$EndNodes\n$Elements\n1\n1 2 2 0 1 1 2 3\n$EndElements\n"
    )
    with assert_raises(contains="unknown node 3"):
        _ = read_gmsh(TMP)
    # a section left open
    _write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1\n1 0 0 0\n")
    with assert_raises():
        _ = read_gmsh(TMP)
    # sections it does not know are skipped; z is kept when not all zero
    _write(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Comments\nhello\n$EndComments\n"
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 1\n$EndNodes\n$Elements\n1\n"
        "1 2 0 3 1 2\n$EndElements\n"
    )
    var m = read_gmsh(TMP)
    assert_equal(cols_of(m.vertices), 3)
    assert_equal(iget(m.elements, 0, 0), 2)  # node 3, in file order
    assert_equal(len(m.groups), 0)


def _two_triangles(tags: List[Int], corner: Int) -> String:
    """Format 2.2: nodes tagged `tags` (at (0, 0), (1, 0), (0, 1), (1, 1)),
    listed out of order, and two triangles; the first names `corner`
    where it would name `tags[0]`."""
    var t = String("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n")
    t += String(tags[3], " 1 1 0\n", tags[0], " 0 0 0\n")
    t += String(tags[2], " 0 1 0\n", tags[1], " 1 0 0\n")
    t += "$EndNodes\n$Elements\n2\n"
    t += String("1 2 0 ", corner, " ", tags[1], " ", tags[2], "\n")
    t += String("2 2 0 ", tags[1], " ", tags[3], " ", tags[2], "\n")
    return t + "$EndElements\n"


def test_node_tag_gaps() raises:
    # tags spanning less than 4x their number are looked up in a table
    # (here with gaps), others by bisection; both number vertices in tag
    # order and reject tags no node has
    var cases: List[List[Int]] = [[2, 5, 7, 9], [2, 5000000, 7000000, 9000000]]
    for ref tags in cases:
        _write(_two_triangles(tags, tags[0]))
        var m = read_gmsh(TMP)
        assert_true(m.node_tags == tags)
        assert_true(row_list(m.elements, 0) == [0, 1, 2])
        assert_true(row_list(m.elements, 1) == [1, 3, 2])
        assert_equal(fget(m.vertices, 3, 0), 1.0)
        assert_equal(fget(m.vertices, 3, 1), 1.0)
        for missing in [tags[1] - 1, 1, tags[3] + 1, 0]:
            _write(_two_triangles(tags, missing))
            with assert_raises(
                contains=String("element uses unknown node ", missing)
            ):
                _ = read_gmsh(TMP)


def test_large_node_tags() raises:
    # tags are sorted and searched: tags near 1e14 once allocated tables of
    # that many entries
    _write(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n3\n"
        "100000000000002 0 1 0\n100000000000000 0 0 0\n"
        "100000000000001 1 0 0\n$EndNodes\n$Elements\n1\n"
        "1 2 0 100000000000000 100000000000001 100000000000002\n"
        "$EndElements\n"
    )
    var m = read_gmsh(TMP)
    assert_equal(len(m.node_tags), 3)
    assert_equal(m.node_tags[0], 100000000000000)
    assert_equal(fget(m.vertices, 2, 1), 1.0)


def _digest(raw: _Raw) -> List[Int]:
    """Everything `_parse` read, as integers (coordinates by their bits)."""
    var out = List[Int]()
    out.append(len(raw.node_tags))
    out.extend(raw.node_tags.copy())
    for x in raw.coords:
        out.append(Int(bitcast[DType.int64](x)))
    for ref b in raw.blocks:
        out.append(b.type)
        out.append(len(b.physical))
        out.extend(b.physical.copy())
        out.append(len(b.nodes))
        out.extend(b.nodes.copy())
    return out^


def _parse_both(path: String) raises -> List[String]:
    """`_parse` serially and from numbers parsed ahead (with the
    threshold at 0, in chunks of a few bytes): each one's digest, or its
    error."""
    var out = List[String]()
    for parallel_bytes in [Int.MAX, 0]:
        try:
            out.append(String(_digest(_parse(path, parallel_bytes))))
        except e:
            out.append(String("error: ", e))
    return out^


def test_parallel_parse_agrees() raises:
    # large ASCII sections are parsed ahead, in parallel; the result, and
    # any error, must be the serial reader's
    for name in [
        "square_v41",
        "square_v22",
        "cube_v41",
        "cube_v22",
        "square_edges_only",
        "square_v41_bin",
        "cube_v22_bin",
    ]:
        var r = _parse_both(DIR + name + ".msh")
        assert_equal(r[0], r[1], name)
        assert_false(r[0].startswith("error"), name)
    var tags: List[Int] = [2, 5, 7, 9]
    var good = _two_triangles(tags, 2)
    # signs, -0, exponents, tabs and CRLF, a tag past 2^53 (left to the
    # serial reader); then malformed numbers: a fraction where an integer
    # goes, text, a number missing, one too many, a section left open
    var texts: List[String] = [
        good,
        good.replace(" 0 0 0\n", " -0 0.0e0 -0.\n").replace("\n5 ", "\n+5 "),
        good.replace(" 1 1 0\n", " 1.0e+00\t1E0 0\r\n"),
        good.replace("9 1 1 0", "9007199254740993 1 1 0"),
        good.replace("2 2 0 5 9 7", "2 2 0 5.0 9 7"),
        good.replace("2 2 0 5 9 7", "2 2 0 5 9 seven"),
        good.replace("2 2 0 5 9 7", "2 2 0 5 9"),
        good.replace("2 2 0 5 9 7", "2 2 0 5 9 7 7"),
        good.replace("\n7 0 1 0", "\n7 0 1 0 1"),
        good.replace("$EndElements\n", ""),
    ]
    for i in range(len(texts)):
        assert_true(i == 0 or texts[i] != good, String("text ", i))
        _write(texts[i])
        var r = _parse_both(TMP)
        assert_equal(r[0], r[1], String("text ", i))
        assert_equal(r[0].startswith("error"), i >= 4, String("text ", i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
