from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    SimplicialComplex,
    derham_map,
    flat_torus,
    icosphere,
    kuhn_grid,
    read_mesh,
    refine,
    whitney_innerproduct,
)
from mojodec.core.mesh import SimplicialMesh
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    RealArray,
    cols_of,
    fget,
    int_array,
    real_array,
    real_array_from_rows,
    rows_of,
)
from mojodec.routines.geometry import signed_volume
from mojodec.routines.refinement import _template


def _simplex(n: Int) raises -> SimplicialComplex:
    """One irregular n-simplex in R^n, n = 1..4."""
    var rows: List[List[Float64]] = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0, 0.2],
        [0.3, 0.9, 0.2, 0.1],
        [0.2, 0.3, 1.1, 0.3],
        [0.1, 0.2, 0.4, 0.8],
    ]
    var pts = real_array(n + 1, n)
    for i in range(n + 1):
        for a in range(n):
            pts.unsafe_set(i * n + a, rows[i][a])
    var T = int_array(1, n + 1)
    # an odd order, so the element is oriented against its sorted list
    for i in range(n + 1):
        T.unsafe_set(i, Scalar[DType.int](n - i))
    return SimplicialComplex(SimplicialMesh(pts, T))


def _element(sc: SimplicialComplex, t: Int) raises -> RealArray:
    """The vertices of element t, in the mesh's order."""
    var n = sc.complex_dimension()
    ref V = sc.vertices()
    ref T = sc.simplices()
    var N = cols_of(V)
    var out = real_array(n + 1, N)
    for i in range(n + 1):
        var v = Int(T.unsafe_get(t * (n + 1) + i))
        for a in range(N):
            out.unsafe_set(i * N + a, fget(V, v, a))
    return out^


def test_children_split_volume_and_keep_orientation() raises:
    # 2^n children, each with 2^-n of its parent's signed volume
    for n in range(5):
        assert_equal(len(_template(n).children), 1 << n)
    for n in range(1, 5):
        var sc = _simplex(n)
        var r = refine(sc)
        var vol = signed_volume(_element(sc, 0))
        assert_true(vol != 0)
        assert_equal(rows_of(r.fine.simplices()), 1 << n)
        var total: Float64 = 0
        for t in range(1 << n):
            var v = signed_volume(_element(r.fine, t))
            assert_almost_equal(
                v, vol / Float64(1 << n), atol=1e-15 * abs(vol), rtol=0
            )
            assert_equal(r.parent[t], 0)
            total += v
        assert_almost_equal(total, vol, atol=1e-14 * abs(vol), rtol=0)


def _shape_classes(sc: SimplicialComplex) raises -> Int:
    """The number of distinct elements up to similarity, by their sorted
    edge lengths relative to the longest (rounded to 1e-6)."""
    var n = sc.complex_dimension()
    var seen = List[List[Int]]()
    for t in range(rows_of(sc.simplices())):
        var P = _element(sc, t)
        var L = List[Float64]()
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                var s: Float64 = 0
                for a in range(cols_of(P)):
                    var x = fget(P, i, a) - fget(P, j, a)
                    s += x * x
                L.append(sqrt(s))
        sort(L)
        var key = List[Int]()
        for ref l in L:
            key.append(Int(l / L[len(L) - 1] * 1e6 + 0.5))
        if key not in seen:
            seen.append(key^)
    return len(seen)


def test_shapes_stay_bounded() raises:
    # the vertex numbering makes each child split as its parent was: a
    # tetrahedron's descendants have 3 shapes at every level (Bey), a
    # triangle's 1
    for n in [2, 3]:
        var sc = _simplex(n)
        for _ in range(3):
            sc = refine(sc).fine
            assert_equal(_shape_classes(sc), 1 if n == 2 else 3)


def _check_topology(sc: SimplicialComplex) raises:
    var n = sc.complex_dimension()
    var r = refine(sc)
    ref F = r.fine
    assert_equal(F[0].num_simplices, sc[0].num_simplices + sc[1].num_simplices)
    assert_equal(F[n].num_simplices, sc[n].num_simplices << n)
    var b = sc.betti_numbers()
    var bf = F.betti_numbers()
    for k in range(n + 1):
        assert_equal(bf[k], b[k])
    # conforming: the boundary faces split 2^(n-1) ways and nothing more
    var bc = sc.boundary_cells()[n - 1].copy()
    var bfine = F.boundary_cells()[n - 1].copy()
    assert_equal(len(bfine), len(bc) << (n - 1))
    for t in range(F[n].num_simplices):
        assert_equal(r.parent[t], t >> n)


def test_counts_topology_and_conformity() raises:
    _check_topology(SimplicialComplex(kuhn_grid([3, 2])))
    _check_topology(SimplicialComplex(kuhn_grid([2, 2, 1])))
    _check_topology(SimplicialComplex(flat_torus([3, 3])))
    _check_topology(SimplicialComplex(icosphere(1)))
    _check_topology(SimplicialComplex(read_mesh("tests/meshes/torus_tet.xml")))


def _max_abs(A: CSRMatrix) -> Float64:
    var m: Float64 = 0
    for v in A.data:
        m = max(m, abs(v))
    return m


def _check_prolongation(sc: SimplicialComplex) raises:
    var n = sc.complex_dimension()
    var r = refine(sc)
    ref F = r.fine
    for k in range(n + 1):
        var P = r.prolongation(k)
        assert_equal(P.nrows, F[k].num_simplices)
        assert_equal(P.ncols, sc[k].num_simplices)
        # the coarse Whitney space is inside the fine one
        var Mc = whitney_innerproduct(sc, k)
        var G = P.T() @ whitney_innerproduct(F, k) @ P - Mc
        assert_true(_max_abs(G) <= 1e-14 * _max_abs(Mc))
        # P commutes with d, exactly: the entries are dyadic
        if k < n:
            var D = r.prolongation(k + 1) @ sc[k].d - F[k].d @ P
            assert_equal(_max_abs(D), 0.0)
    # vertices: old ones copied, midpoints averaged
    var P0 = r.prolongation(0)
    for i in range(P0.nrows):
        var lo = P0.indptr[i]
        var hi = P0.indptr[i + 1]
        if hi - lo == 1:
            assert_equal(P0.data[lo], 1.0)
        else:
            assert_equal(hi - lo, 2)
            assert_equal(P0.data[lo], 0.5)
            assert_equal(P0.data[lo + 1], 0.5)


def test_prolongation() raises:
    _check_prolongation(_simplex(1))
    _check_prolongation(_simplex(4))
    _check_prolongation(SimplicialComplex(kuhn_grid([3, 2])))
    _check_prolongation(SimplicialComplex(kuhn_grid([2, 1, 2])))
    _check_prolongation(SimplicialComplex(icosphere(1)))
    _check_prolongation(SimplicialComplex(flat_torus([3, 3])))


def test_prolongation_reproduces_constant_forms() raises:
    # W R omega = omega for a constant form, so P R_coarse = R_fine there
    def omega(x: List[Float64]) -> List[Float64]:
        return [0.3, -1.2, 0.7]  # dx^dy, dx^dz, dy^dz for k = 2; dx.. k = 1

    var sc = SimplicialComplex(kuhn_grid([2, 2, 2], [1.0, 1.5, 0.5]))
    var r = refine(sc)
    for k in [1, 2]:
        var Pc = r.prolongation(k) @ derham_map(sc, k, 1, omega).v
        var f = derham_map(r.fine, k, 1, omega).v.copy()
        for i in range(f.size):
            assert_almost_equal(
                Pc.unsafe_get(i), f.unsafe_get(i), atol=1e-15, rtol=0
            )


def test_errors() raises:
    var r = refine(SimplicialComplex(kuhn_grid([2, 2])))
    with assert_raises(contains="outside 0..2"):
        _ = r.prolongation(3)
    with assert_raises(contains="outside 0..2"):
        _ = r.prolongation(-1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
