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
    read_gmsh,
)
from mojodec.core.utils import cols_of, fget, iget, rows_of

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
    assert_almost_equal(abs(total), 1.0, atol=1e-12)
    for pair in [("bottom", 0.0), ("top", 1.0)]:
        var faces = msh.subset(sc, pair[0], 2)
        assert_true(_all_vertices(sc, faces, 2, pair[1]))
        assert_true((faces - sc.boundary_cells()[2]).is_empty())
        var area: Float64 = 0
        for f in faces.indices():
            area += sc[2].primal_volume.unsafe_get(f)
        assert_almost_equal(area, 1.0, atol=1e-12)
    assert_true(msh.subset(sc, "solid", 3) == Subset.full(sc, 3))


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
