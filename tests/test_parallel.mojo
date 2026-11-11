from std.math import sqrt
from std.os import setenv, unsetenv
from std.sys import num_physical_cores
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    SimplicialComplex,
    simplicial_grid_2d,
    whitney_innerproduct,
)
from mojodec.core.parallel import num_chunks, parallel_ranges
from mojodec.core.sparse import PARALLEL_NNZ
from mojodec.core.utils import fget, fset, iget, real_vector

# The meshes here are large enough for the parallel paths: 80k triangles
# for circumcenters and volumes (50k), 5000 for Whitney (4096), and a
# Laplacian with over 1M stored entries for matvec.


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


def test_num_threads_env_caps_chunks() raises:
    var cores = num_physical_cores()
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
