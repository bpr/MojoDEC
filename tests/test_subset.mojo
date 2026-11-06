from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from mojodec import (
    CSRMatrix,
    DiagonalOperator,
    SimplicialComplex,
    Simplex,
    Subset,
    simplicial_grid_2d,
    spsolve,
)
from mojodec.core.utils import fget, real_vector
from mojodec.routines.io.meshio import read_mesh


def _grid(n: Int) raises -> SimplicialComplex:
    var m = simplicial_grid_2d(n)
    return SimplicialComplex(m[0], m[1])


def _cube() raises -> SimplicialComplex:
    """The surface of a cube: closed, so without boundary."""
    return SimplicialComplex(read_mesh("tests/meshes/cube.xml"))


def _torus() raises -> SimplicialComplex:
    """A solid torus of tetrahedra."""
    return SimplicialComplex(read_mesh("tests/meshes/torus_tet.xml"))


def test_set_operations() raises:
    var sc = _grid(6)  # 72 triangles: more than one 64-bit word
    var a = Subset.from_indices(sc, 2, [0, 5, 64, 71, 5])
    var b = Subset.from_indices(sc, 2, [5, 6, 70])
    assert_equal(len(a), 4)
    assert_equal(a.size, 72)
    assert_true(64 in a)
    assert_false(63 in a)
    assert_false(72 in a)
    assert_equal(a.indices(), [0, 5, 64, 71])
    assert_equal((a | b).indices(), [0, 5, 6, 64, 70, 71])
    assert_equal((a & b).indices(), [5])
    assert_equal((a - b).indices(), [0, 64, 71])
    # the complement stays within the 72 triangles
    var c = ~a
    assert_equal(len(c), 68)
    assert_true(len(~Subset.empty(sc, 2)) == 72)
    assert_true((c | a) == Subset.full(sc, 2))
    assert_true((c & a).is_empty())
    var m = a.mask()
    assert_equal(len(m), 72)
    assert_true(m[71] and not m[70])
    assert_true(Subset.from_mask(sc, 2, m) == a)
    a.discard(0)
    a.add(1)
    assert_equal(a.indices(), [1, 5, 64, 71])
    # dual 0-cells are the triangles: same bits, other kind
    var ad = a.dual()
    assert_equal(ad.dim, 0)
    assert_false(ad.is_primal)
    assert_true(ad.dual() == a)
    assert_false(ad.indices() != a.indices())
    with assert_raises():
        _ = a | ad
    with assert_raises():
        a.add(72)
    with assert_raises():
        _ = Subset.from_indices(sc, 3, [0])


def _brute_boundary(
    sc: SimplicialComplex, k: Int, cells: List[Int]
) raises -> List[Bool]:
    """Faces of the given k-simplices, found through their vertices."""
    var out = List[Bool](length=sc[k - 1].num_simplices, fill=False)
    for i in cells:
        var s = sc[k].index_to_simplex(i)
        for j in range(len(s)):
            var face = List[Int]()
            for l in range(len(s)):
                if l != j:
                    face.append(s[l])
            out[sc[k - 1].simplex_to_index(Simplex(face))] = True
    return out^


def test_boundary_coboundary() raises:
    for sc in [_grid(4), _cube(), _torus()]:
        var n = sc.complex_dimension()
        for k in range(1, n + 1):
            var cells: List[Int] = [0, 2, sc[k].num_simplices - 1]
            var s = Subset.from_indices(sc, k, cells)
            var bd = s.boundary(sc)
            assert_equal(bd.dim, k - 1)
            assert_true(bd.mask() == _brute_boundary(sc, k, cells))
            # coboundary is the transpose relation: t in cob(f) iff f in bd(t)
            var f = Subset.from_indices(sc, k - 1, [1, 3])
            var cob = f.coboundary(sc)
            for t in range(sc[k].num_simplices):
                var faces = Subset.from_indices(sc, k, [t]).boundary(sc)
                assert_equal(t in cob, not (faces & f).is_empty())
            # inner coboundary: cells whose whole boundary lies in the set
            var inner = bd.inner_coboundary(sc)
            assert_true((inner.boundary(sc) - bd).is_empty())
            assert_true((s - inner).is_empty())
            # dual cells: boundary is the primal coboundary, and vice versa
            assert_true(f.dual().boundary(sc) == cob.dual())
            assert_true(s.dual().coboundary(sc) == bd.dual())
        with assert_raises():
            _ = Subset.full(sc, 0).boundary(sc)
        with assert_raises():
            _ = Subset.full(sc, n).coboundary(sc)


def test_mesh_boundary() raises:
    for sc in [_grid(5), _torus()]:
        var n = sc.complex_dimension()
        var cells = sc.boundary_cells()
        for k in range(n):
            var bd = sc.boundary_subset(k)
            var expected = cells[k].copy()
            sort(expected)
            assert_equal(bd.indices(), expected)
        assert_true(sc.boundary_subset(n).is_empty())
        # the boundary faces are those `sc.boundary()` lists
        var faces = Subset.from_simplices(sc, sc.boundary())
        assert_true(faces == sc.boundary_subset(n - 1))
    var closed = _cube()
    for k in range(3):
        assert_true(closed.boundary_subset(k).is_empty())
    # manifold boundary of a region: the grid's lower-left 2x2 block
    var sc = _grid(4)
    var region = Subset.empty(sc, 2)
    for t in range(sc[2].num_simplices):
        ref c = sc[2].circumcenter
        if fget(c, t, 0) < 0.5 and fget(c, t, 1) < 0.5:
            region.add(t)
    assert_equal(len(region), 8)
    var rim = region.manifold_boundary(sc)
    assert_equal(len(rim), 8)  # 4 unit-length sides of 2 edges each
    assert_equal(len(region.boundary(sc)), 16)  # 8 rim + 8 inside edges
    with assert_raises():
        _ = rim.manifold_boundary(sc)


def test_exclude_restrict() raises:
    var sc = _grid(3)
    var bd = sc.boundary_subset(0)
    var L = sc[0].d.T() @ sc[1].star @ sc[0].d
    var Le = L.exclude_subset(bd)
    var Lr = L.restrict_to_subset(bd)
    for i in range(L.nrows):
        for j in range(L.ncols):
            if i in bd:
                assert_equal(Le[i, j], 0.0)
                assert_equal(Lr[i, j], L[i, j])
            else:
                assert_equal(Le[i, j], L[i, j])
                assert_equal(Lr[i, j], 0.0)
    assert_equal(Le.nnz() + Lr.nnz(), L.nnz())
    var S = sc[0].star.exclude_subset(bd)
    var Sr = sc[0].star.restrict_to_subset(bd)
    for i in range(len(S)):
        assert_equal(S[i] + Sr[i], sc[0].star[i])
        assert_equal(S[i] == 0.0, i in bd)
    with assert_raises():
        _ = L.exclude_subset(sc.boundary_subset(1))
    with assert_raises():
        _ = sc[1].star.restrict_to_subset(bd)


def test_dirichlet_poisson() raises:
    """The DEC Laplacian reproduces a linear function from its boundary
    values: L u = 0 inside, u = g on the boundary."""
    var sc = _grid(8)
    var fixed = sc.boundary_subset(0)
    var N = sc[0].num_simplices
    var L = sc[0].d.T() @ sc[1].star @ sc[0].d
    var A = (
        L.exclude_subset(fixed)
        + DiagonalOperator.identity(N).restrict_to_subset(fixed).to_csr()
    )
    var g = sc.get_cochain(0)
    for i in range(N):
        g[i] = 1.0 + fget(sc.vertices(), i, 0) - 2.0 * fget(sc.vertices(), i, 1)
    var b = sc.get_cochain(0)
    b.overwrite(fixed, g)
    var u = spsolve(A, b.v)
    for i in range(N):
        assert_almost_equal(u.unsafe_get(i), g[i], atol=1e-12)
    # overwrite checks the kinds
    with assert_raises():
        b.overwrite(sc.boundary_subset(1), g)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
