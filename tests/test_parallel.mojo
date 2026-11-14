from std.math import sqrt
from std.os import setenv, unsetenv
from std.sys import num_logical_cores, num_physical_cores
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

import numojo as nm
from numojo.prelude import *
from mojodec import (
    CSRMatrix,
    SimplicialComplex,
    SimplicialMesh,
    kuhn_grid,
    simplicial_grid_2d,
    whitney_innerproduct,
)
from mojodec.core.parallel import num_chunks, parallel_ranges
from mojodec.core.sparse import PARALLEL_NNZ
from mojodec.core.utils import (
    _lexsort_rows,
    _merge_sort_rows,
    compare_rows,
    fget,
    fset,
    iget,
    int_array,
    iset,
    int_vector,
    lexsort_rows,
    real_vector,
    rows_of,
    sort_each_row,
)
from mojodec.routines.simplex_array import (
    _boundary_by_buckets,
    _boundary_by_lexsort,
    _vertex_bound,
    simplex_array_parity,
)

# The meshes here are large enough for the parallel paths: 80k triangles
# for circumcenters and volumes (50k), 5000 for Whitney (4096), and a
# Laplacian with over 1M stored entries for matvec.


def _lcg(mut state: UInt64) -> Float64:
    """The next value in [-1, 1) of a 64-bit linear congruential generator."""
    state = state * 6364136223846793005 + 1442695040888963407
    return Float64(state >> 11) / Float64(1 << 53) * 2.0 - 1.0


def _jittered(dims: List[Int], shuffle: Bool) raises -> SimplicialMesh:
    """A Kuhn grid with interior vertices moved up to 5% of a cell (so it is
    not well-centered and some dual pieces are negative); with `shuffle`,
    the vertex numbers are permuted, so vertex blocks are scattered."""
    var mesh = kuhn_grid(dims)
    var d = len(dims)
    var nv = mesh.vertices.shape[0]
    var state: UInt64 = 12345
    for i in range(nv):
        var interior = True
        for a in range(d):
            var x = fget(mesh.vertices, i, a)
            var h = 1.0 / Float64(dims[a])
            if x < 0.5 * h or x > 1.0 - 0.5 * h:
                interior = False
        if interior:
            for a in range(d):
                var h = 1.0 / Float64(dims[a])
                fset(
                    mesh.vertices,
                    i,
                    a,
                    fget(mesh.vertices, i, a) + 0.05 * h * _lcg(state),
                )
    if not shuffle:
        return mesh^
    # Fisher-Yates permutation p; vertex i becomes p[i]
    var p = List[Int](length=nv, fill=0)
    for i in range(nv):
        p[i] = i
    for i in range(nv - 1, 0, -1):
        var j = Int((_lcg(state) + 1.0) * 0.5 * Float64(i + 1)) % (i + 1)
        var t = p[i]
        p[i] = p[j]
        p[j] = t
    var V = mesh.vertices.copy()
    for i in range(nv):
        for a in range(d):
            fset(V, p[i], a, fget(mesh.vertices, i, a))
    var E = mesh.elements.copy()
    for e in range(E.shape[0]):
        for j in range(d + 1):
            iset(E, e, j, p[iget(mesh.elements, e, j)])
    return SimplicialMesh(V, E)


def _dual_volumes(mesh: SimplicialMesh) raises -> List[List[Float64]]:
    var sc = SimplicialComplex(mesh.vertices, mesh.elements)
    var out = List[List[Float64]]()
    for k in range(len(sc)):
        var level = List[Float64]()
        for i in range(sc[k].num_simplices):
            level.append(sc[k].dual_volume.unsafe_get(i))
        out.append(level^)
    return out^


def _check_dual_volumes_match_serial(mesh: SimplicialMesh) raises:
    _ = setenv("MOJODEC_NUM_THREADS", "1")
    var serial = _dual_volumes(mesh)
    _ = unsetenv("MOJODEC_NUM_THREADS")
    var parallel = _dual_volumes(mesh)
    assert_equal(len(serial), len(parallel))
    for k in range(len(serial)):
        assert_equal(len(serial[k]), len(parallel[k]))
        for i in range(len(serial[k])):
            # bit for bit: each face's pieces are added in the serial order
            assert_equal(serial[k][i], parallel[k][i])


def test_parallel_dual_volumes_match_serial() raises:
    # past the 4096 top simplices where the walk goes parallel
    _check_dual_volumes_match_serial(_jittered([60, 60], False))
    _check_dual_volumes_match_serial(_jittered([12, 12, 12], False))
    # scattered vertex numbers: every block touches most of the mesh
    _check_dual_volumes_match_serial(_jittered([60, 60], True))
    _check_dual_volumes_match_serial(_jittered([12, 12, 12], True))


def _with_unused_vertices(
    mesh: SimplicialMesh, extra: Int
) raises -> SimplicialMesh:
    """The same mesh behind `extra` unused vertices: vertex i becomes
    i + extra, which keeps every sorted face in the same order."""
    var nv = mesh.vertices.shape[0]
    var d = mesh.vertices.shape[1]
    var V = nm.zeros[DType.float64](Shape(nv + extra, d))
    for i in range(nv):
        for a in range(d):
            fset(V, i + extra, a, fget(mesh.vertices, i, a))
    var E = mesh.elements.copy()
    for e in range(E.shape[0]):
        for j in range(E.shape[1]):
            iset(E, e, j, iget(mesh.elements, e, j) + extra)
    return SimplicialMesh(V, E)


def _check_unused_vertices(mesh: SimplicialMesh) raises:
    # dual volumes and Whitney matrices behind unused vertices: the dual
    # walk's vertex blocks are by vertex id, which unused vertices push
    # past the number of 0-simplices, and Whitney's k = 0 rows are level-0
    # indices, not vertex ids
    var shifted = _with_unused_vertices(mesh, 3)
    for threads in ["1", ""]:
        if threads:
            _ = setenv("MOJODEC_NUM_THREADS", threads)
        var want = _dual_volumes(mesh)
        var got = _dual_volumes(shifted)
        _ = unsetenv("MOJODEC_NUM_THREADS")
        for k in range(len(want)):
            assert_equal(len(want[k]), len(got[k]))
            for i in range(len(want[k])):
                assert_equal(want[k][i], got[k][i])
        # Whitney matrices too: their k = 0 rows are level-0 indices
        var sc = SimplicialComplex(mesh.vertices, mesh.elements)
        var shifted_sc = SimplicialComplex(shifted.vertices, shifted.elements)
        if threads:
            _ = setenv("MOJODEC_NUM_THREADS", threads)
        for k in range(len(sc)):
            var A = whitney_innerproduct(sc, k)
            var B = whitney_innerproduct(shifted_sc, k)
            assert_true(A.indptr == B.indptr)
            assert_true(A.indices == B.indices)
            assert_true(A.data == B.data)
        _ = unsetenv("MOJODEC_NUM_THREADS")


def test_unused_vertices() raises:
    _check_unused_vertices(_jittered([60, 60], False))
    _check_unused_vertices(_jittered([12, 12, 12], True))
    _check_unused_vertices(_jittered([4, 4], False))


def _same_csr(A: CSRMatrix, B: CSRMatrix) raises:
    assert_equal(A.nrows, B.nrows)
    assert_equal(A.ncols, B.ncols)
    assert_true(A.indptr == B.indptr)
    assert_true(A.indices == B.indices)
    assert_true(A.data == B.data)


def _check_boundary_paths(mesh: SimplicialMesh) raises:
    # every level's faces and matrices by buckets (parallel) and by one
    # lexicographic sort (serial), entry for entry
    var s = mesh.elements.copy()
    var parity = simplex_array_parity(s)
    sort_each_row(s)
    while s.shape[1] > 1:
        var y = _boundary_by_lexsort(s, parity)
        # packed integer keys, and the comparator they replace
        for pack in [True, False]:
            var x = _boundary_by_buckets(s, parity, _vertex_bound(s), pack)
            assert_equal(x.faces.size, y.faces.size)
            for i in range(x.faces.size):
                assert_equal(x.faces.unsafe_get(i), y.faces.unsafe_get(i))
            _same_csr(x.coboundary, y.coboundary)
            _same_csr(x.boundary, y.boundary)
        s = y.faces.copy()
        parity = int_vector(rows_of(s))


def _fan(n: Int, K: Int) raises -> SimplicialMesh:
    # n simplices around the K - 1 lowest vertices, each with two of the
    # n + 1 others, numbered at random: buckets of n and more occurrences
    var E = int_array(n, K)
    var perm = List[Int](length=n + 1, fill=0)
    for i in range(n + 1):
        perm[i] = i * 7919 % (n + 1)  # n + 1 prime to 7919
    for t in range(n):
        for j in range(K - 2):
            E.unsafe_set(t * K + j, Scalar[DType.int](j))
        E.unsafe_set(t * K + K - 2, Scalar[DType.int](K - 2 + perm[t]))
        E.unsafe_set(t * K + K - 1, Scalar[DType.int](K - 2 + perm[t + 1]))
    var V = nm.zeros[DType.float64](Shape(n + K - 1, K - 1))
    return SimplicialMesh(V, E)


def test_boundary_by_buckets_matches_lexsort() raises:
    _check_boundary_paths(_jittered([60, 60], False))
    _check_boundary_paths(_jittered([60, 60], True))
    _check_boundary_paths(_jittered([12, 12, 12], True))
    _check_boundary_paths(
        _with_unused_vertices(_jittered([12, 12, 12], True), 5)
    )
    # buckets of 65 to _RANK_SORT_TO, sorted by rank
    _check_boundary_paths(_fan(40, 3))
    _check_boundary_paths(_fan(30, 4))
    # buckets past _RANK_SORT_TO, sorted by merging
    _check_boundary_paths(_fan(5000, 3))
    _check_boundary_paths(_fan(3000, 4))


def test_large_meshes_check_distinct_simplices() raises:
    # past the 50k rows where the checks of the elements go parallel
    var mesh = kuhn_grid([160, 160])
    var n = rows_of(mesh.elements)
    _ = SimplicialComplex(mesh.vertices, mesh.elements, compute_geometry=False)
    # row 70 again, reordered, at the end
    var E = nm.zeros[DType.int](Shape(n + 1, 3))
    for i in range(n):
        for j in range(3):
            iset(E, i, j, iget(mesh.elements, i, j))
    for j in range(3):
        iset(E, n, j, iget(mesh.elements, 70, 2 - j))
    with assert_raises(
        contains=String("simplices 70 and ", n, " have the same vertices")
    ):
        _ = SimplicialComplex(mesh.vertices, E, compute_geometry=False)
    # repeated vertices in two rows of different chunks: the first is named
    var R = mesh.elements.copy()
    iset(R, 45_000, 2, iget(R, 45_000, 0))
    iset(R, 40_000, 2, iget(R, 40_000, 1))
    with assert_raises(contains="simplex 40000 repeats vertex"):
        _ = SimplicialComplex(mesh.vertices, R, compute_geometry=False)


def test_parallel_ranges_covers_once() raises:
    var n = 100_003
    var hits = List[Int](length=n, fill=0)
    var hp = hits.unsafe_ptr()
    var chunks = num_chunks(n, 0)
    var seen = List[Int](length=chunks, fill=0)
    var sp = seen.unsafe_ptr()

    def body(chunk: Int, lo: Int, hi: Int) {hp, sp}:
        sp[unsafe_offset=chunk] += 1
        for i in range(lo, hi):
            hp[unsafe_offset=i] += 1

    parallel_ranges(n, 0, body)
    for h in hits:
        assert_equal(h, 1)
    for c in seen:
        assert_equal(c, 1)
    # below the threshold: one chunk on the calling thread
    assert_equal(num_chunks(10, 100), 1)
    assert_equal(num_chunks(0, 0), 0)


def test_large_lexsort_rows() raises:
    # past the 16384 first entries where the runs are sorted in parallel:
    # the merge sort's order, with a run of 100 equal first entries (sorted
    # by merging) and one pair of equal rows
    var n = 40_000
    var a = int_array(n, 3)
    var state: UInt64 = 777
    for i in range(n):
        for j in range(3):
            var x = Int((_lcg(state) + 1.0) * 30_000.0) % 60_000
            iset(a, i, j, 0 if j == 0 and i % 400 == 0 else x)
    for j in range(3):
        iset(a, 30_000, j, iget(a, 1234, j))
    var expected = List[Int](length=n, fill=0)
    for i in range(n):
        expected[i] = i
    var scratch = List[Int](length=n, fill=0)
    _merge_sort_rows(a, expected, scratch, 0, n)
    assert_true(lexsort_rows(a) == expected)
    var duplicate = 0
    _ = _lexsort_rows(a, duplicate)
    var first = -1
    for t in range(1, n):
        if compare_rows(a, expected[t - 1], a, expected[t]) == 0:
            first = t
            break
    assert_true(first > 0)
    assert_equal(duplicate, first)


def _complex_error(mesh: SimplicialMesh, threads: String) raises -> String:
    _ = setenv("MOJODEC_NUM_THREADS", threads)
    var message = String()
    try:
        _ = SimplicialComplex(
            mesh.vertices, mesh.elements, compute_geometry=False
        )
    except e:
        message = String(e)
    _ = unsetenv("MOJODEC_NUM_THREADS")
    return message


def test_large_meshes_check_orientation() raises:
    # past the 50k faces where the orientation check goes parallel: the
    # same error as the serial check, for the first bad face
    var mesh = kuhn_grid([160, 160])
    var E = mesh.elements.copy()
    for t in [30_000, 20_000]:
        var x = iget(E, t, 0)
        iset(E, t, 0, iget(E, t, 1))
        iset(E, t, 1, x)
    var flipped = SimplicialMesh(mesh.vertices, E)
    var serial = _complex_error(flipped, "1")
    assert_true("induce the same orientation" in serial, serial)
    assert_equal(_complex_error(flipped, "16"), serial)
    # a third triangle on an edge of element 40000
    var n = rows_of(mesh.elements)
    var nv = rows_of(mesh.vertices)
    var T = nm.zeros[DType.int](Shape(n + 1, 3))
    for i in range(n):
        for j in range(3):
            iset(T, i, j, iget(mesh.elements, i, j))
    iset(T, n, 0, iget(mesh.elements, 40_000, 0))
    iset(T, n, 1, iget(mesh.elements, 40_000, 1))
    iset(T, n, 2, nv)
    var V = nm.zeros[DType.float64](Shape(nv + 1, 2))
    for i in range(nv):
        for j in range(2):
            fset(V, i, j, fget(mesh.vertices, i, j))
    var crowded = SimplicialMesh(V, T)
    serial = _complex_error(crowded, "1")
    assert_true("not a pseudomanifold" in serial, serial)
    assert_equal(_complex_error(crowded, "16"), serial)


def test_large_mesh_checks_indices() raises:
    # past the 2^18 entries where the copy and range check go parallel
    var mesh = kuhn_grid([210, 210])
    var nv = rows_of(mesh.vertices)
    _ = SimplicialMesh(mesh.vertices, mesh.elements)
    for bad in [nv, -1]:
        var E = mesh.elements.copy()
        iset(E, 80_000, 1, bad)
        with assert_raises(contains="invalid index value"):
            _ = SimplicialMesh(mesh.vertices, E)


def test_num_threads_env_caps_chunks() raises:
    var cores = num_logical_cores()
    _ = setenv("MOJODEC_NUM_THREADS", "1")
    assert_equal(num_chunks(100_000, 0), 1)
    _ = setenv("MOJODEC_NUM_THREADS", "2")
    assert_equal(num_chunks(100_000, 0), min(2, cores))
    # not a positive integer: ignored
    _ = setenv("MOJODEC_NUM_THREADS", "zero")
    assert_equal(num_chunks(100_000, 0), cores)
    _ = setenv("MOJODEC_NUM_THREADS", "0")
    assert_equal(num_chunks(100_000, 0), cores)
    _ = unsetenv("MOJODEC_NUM_THREADS")
    assert_equal(num_chunks(100_000, 0), cores)
    # memory-bound loops: one chunk per physical core, capped the same way
    var physical = num_physical_cores()
    assert_equal(num_chunks(100_000, 0, memory_bound=True), physical)
    _ = setenv("MOJODEC_NUM_THREADS", "2")
    assert_equal(num_chunks(100_000, 0, memory_bound=True), min(2, physical))
    _ = unsetenv("MOJODEC_NUM_THREADS")


def test_large_geometry() raises:
    var m = simplicial_grid_2d(200)
    var sc = SimplicialComplex(m.vertices, m.elements)
    ref V = sc.vertices()
    # every circumcenter is equidistant from its triangle's vertices
    ref tri = sc[2]
    for t in range(tri.num_simplices):
        var r0: Float64 = -1
        for j in range(3):
            var v = iget(tri.simplices, t, j)
            var dx = fget(tri.circumcenter, t, 0) - fget(V, v, 0)
            var dy = fget(tri.circumcenter, t, 1) - fget(V, v, 1)
            var r = sqrt(dx * dx + dy * dy)
            if r0 < 0:
                r0 = r
            # one ulp of a unit-square coordinate is 1.1e-16
            assert_almost_equal(r, r0, atol=2e-15, rtol=0)
    # edges have their midpoints
    for e in range(sc[1].num_simplices):
        var a = iget(sc[1].simplices, e, 0)
        var b = iget(sc[1].simplices, e, 1)
        assert_equal(
            fget(sc[1].circumcenter, e, 0),
            0.5 * fget(V, a, 0) + 0.5 * fget(V, b, 0),
        )
    # the triangles' areas fill the unit square
    var area: Float64 = 0
    for t in range(tri.num_simplices):
        assert_almost_equal(
            abs(tri.primal_volume.unsafe_get(t)),
            1.0 / 80_000,
            atol=2.5e-17,
            rtol=0,
        )
        area += abs(tri.primal_volume.unsafe_get(t))
    # a sum of 80000 terms
    assert_almost_equal(area, 1.0, atol=1e-11, rtol=0)


def test_degenerate_triangle_in_large_mesh() raises:
    var m = simplicial_grid_2d(200)
    var V = m.vertices.copy()
    # vertex 1 onto vertex 0: the triangles holding both collapse
    fset(V, 1, 0, fget(V, 0, 0))
    fset(V, 1, 1, fget(V, 0, 1))
    with assert_raises(contains="degenerate"):
        _ = SimplicialComplex(V, m.elements)


def test_large_whitney() raises:
    var m = simplicial_grid_2d(50)  # 5000 triangles
    var sc = SimplicialComplex(m.vertices, m.elements)
    ref V = sc.vertices()
    # Whitney forms reproduce linear functions, constant 1-forms, and
    # constant 2-forms, so these integrals over the unit square are exact
    var x = real_vector(sc[0].num_simplices)
    for i in range(x.size):
        x.unsafe_set(i, fget(V, i, 0))
    var M0 = whitney_innerproduct(sc, 0)
    var Mx = M0 @ x
    var xx: Float64 = 0
    for i in range(x.size):
        xx += x.unsafe_get(i) * Mx.unsafe_get(i)
    assert_almost_equal(xx, 1.0 / 3.0, atol=1e-14, rtol=0)  # integral of x^2
    var dx = real_vector(sc[1].num_simplices)
    for e in range(dx.size):
        var a = iget(sc[1].simplices, e, 0)
        var b = iget(sc[1].simplices, e, 1)
        dx.unsafe_set(e, fget(V, b, 0) - fget(V, a, 0))
    var M1 = whitney_innerproduct(sc, 1)
    var Mdx = M1 @ dx
    var dxdx: Float64 = 0
    for e in range(dx.size):
        dxdx += dx.unsafe_get(e) * Mdx.unsafe_get(e)
    assert_almost_equal(dxdx, 1.0, atol=1e-12, rtol=0)  # integral of |dx|^2
    var vol = sc[2].primal_volume.copy()
    var M2 = whitney_innerproduct(sc, 2)
    var Mv = M2 @ vol
    var one: Float64 = 0
    for t in range(vol.size):
        one += vol.unsafe_get(t) * Mv.unsafe_get(t)
    assert_almost_equal(one, 1.0, atol=1e-12, rtol=0)  # integral of 1


def test_large_matvec() raises:
    var m = simplicial_grid_2d(400)
    var sc = SimplicialComplex(m.vertices, m.elements, compute_geometry=False)
    var A = sc[0].d.T() @ sc[0].d
    assert_true(A.nnz() >= PARALLEL_NNZ)
    var x = real_vector(A.ncols)
    for i in range(x.size):
        x.unsafe_set(i, Float64((i * 7) % 13) - 6.0)
    var y = A @ x
    # the same sums as a serial loop over the rows
    for i in range(A.nrows):
        var s: Float64 = 0
        for k in range(A.indptr[i], A.indptr[i + 1]):
            s += A.data[k] * x.unsafe_get(A.indices[k])
        assert_equal(y.unsafe_get(i), s)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
