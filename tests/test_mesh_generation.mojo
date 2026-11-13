from std.testing import (
    TestSuite,
    assert_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from std.math import cos, pi

import numojo as nm
from numojo.prelude import *

from mojodec.core.ncube import NCube
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.routines.innerproduct import whitney_innerproduct
from mojodec.routines.linalg import eigh_generalized
from mojodec.core.utils import (
    compare_rows,
    fget,
    fset,
    iget,
    int_array_from_rows,
    iset,
    real_array,
    real_array_from_rows,
    row_list,
    rows_of,
    cols_of,
    sort_each_row,
    take_rows,
)
from mojodec.core.mesh import SimplicialMesh
from mojodec.routines.geometry import signed_volume, unsigned_volume
from mojodec.routines.mesh_generation import (
    simplicial_grid_2d,
    cube_grid,
    triangulate_ncube,
    loop_subdivision,
    kuhn_grid,
    flat_torus,
    klein_bottle,
    mobius_band,
    orient_coherently,
    morton_keys,
    reorder_vertices,
)
from mojodec.core.abstract_simplicial_complex import AbstractSimplicialComplex
from mojodec.routines.simplex_array import coherent_orientation


def test_simplicial_grid_2d() raises:
    var vi = simplicial_grid_2d(4)
    assert_equal(rows_of(vi.vertices), 25)
    assert_equal(rows_of(vi.elements), 32)
    var total: Float64 = 0
    for t in range(rows_of(vi.elements)):
        var v = signed_volume(take_rows(vi.vertices, row_list(vi.elements, t)))
        assert_true(v > 0)
        total += v
    assert_almost_equal(total, 1.0, atol=1e-14, rtol=0)
    var sc = SimplicialComplex(vi)
    assert_equal(sc[1].num_simplices, 25 + 32 - 1)


def test_cube_grid() raises:
    var vi = cube_grid([2, 1])
    var V = nm.fromstring[f64]("[[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]]")
    for i in range(V.size):
        assert_equal(vi.vertices.unsafe_get(i), V.unsafe_get(i))
    var expected: List[Int] = [0, 3, 1, 4, 1, 4, 2, 5]
    assert_equal(vi.corners.ndim, 3)
    for i in range(8):
        assert_equal(Int(vi.corners.unsafe_get(i)), expected[i])


def test_triangulate_ncube() raises:
    var vi = cube_grid([3, 2])
    var tv = triangulate_ncube(vi)
    assert_equal(rows_of(tv.elements), 4 * 6)
    var total: Float64 = 0
    for t in range(rows_of(tv.elements)):
        var v = signed_volume(take_rows(tv.vertices, row_list(tv.elements, t)))
        assert_true(v > 0)
        total += v
    assert_almost_equal(total, 6.0, atol=1e-14, rtol=0)


def test_loop_subdivision() raises:
    var vi = simplicial_grid_2d(2)
    var sub = loop_subdivision(vi)
    assert_equal(rows_of(sub.elements), 4 * rows_of(vi.elements))
    assert_equal(rows_of(sub.vertices), 9 + 16)
    var total: Float64 = 0
    for t in range(rows_of(sub.elements)):
        total += signed_volume(
            take_rows(sub.vertices, row_list(sub.elements, t))
        )
    assert_almost_equal(total, 1.0, atol=1e-14, rtol=0)


def check_orientation(b: String, expected: Bool) raises:
    var B = nm.fromstring[DType.int](b)
    var A = nm.zeros[DType.int](B.shape)
    for i in range(A.size):
        A.unsafe_set(i, Scalar[DType.int](i))
    var a_cube = NCube(A)
    var b_cube = NCube(B)
    assert_false(a_cube.relative_orientation(a_cube))
    assert_false(b_cube.relative_orientation(b_cube))
    assert_equal(a_cube.relative_orientation(b_cube), expected)
    assert_equal(b_cube.relative_orientation(a_cube), expected)


def test_ncube() raises:
    check_orientation("[0,1]", False)
    check_orientation("[1,0]", True)
    check_orientation("[[0,1],[2,3]]", False)
    check_orientation("[[1,3],[0,2]]", False)
    check_orientation("[[3,2],[1,0]]", False)
    check_orientation("[[2,0],[3,1]]", False)
    check_orientation("[[2,3],[0,1]]", True)
    check_orientation("[[0,2],[1,3]]", True)
    check_orientation("[[1,0],[3,2]]", True)
    check_orientation("[[3,1],[2,0]]", True)
    check_orientation("[[[0,1],[2,3]],[[4,5],[6,7]]]", False)
    check_orientation("[[[1,5],[3,7]],[[0,4],[2,6]]]", False)
    check_orientation("[[[5,4],[7,6]],[[1,0],[3,2]]]", False)
    check_orientation("[[[7,3],[5,1]],[[6,2],[4,0]]]", False)
    check_orientation("[[[4,5],[6,7]],[[0,1],[2,3]]]", True)
    check_orientation("[[[0,4],[2,6]],[[1,5],[3,7]]]", True)
    check_orientation("[[[5,1],[7,3]],[[4,0],[6,2]]]", True)


def test_kuhn_grid() raises:
    # d! positively oriented simplices per cell, filling the box
    for d in range(1, 5):
        var dims = List[Int]()
        var lengths = List[Float64]()
        var cells = 1
        var fact = 1
        for a in range(d):
            dims.append(a + 2)
            lengths.append(0.5 + Float64(a))
            cells *= a + 2
            fact *= a + 1
        var mesh = kuhn_grid(dims, lengths)
        assert_equal(rows_of(mesh.elements), fact * cells)
        var total: Float64 = 0
        var box: Float64 = 1
        for a in range(d):
            box *= lengths[a]
        for t in range(rows_of(mesh.elements)):
            var v = signed_volume(
                take_rows(mesh.vertices, row_list(mesh.elements, t))
            )
            assert_true(v > 0)
            total += v
        # one rounding per simplex volume
        assert_almost_equal(total, box, atol=1e-13 * box, rtol=0)
        var sc = SimplicialComplex(mesh)  # coherent, a pseudomanifold
        if d <= 3:
            assert_true(sc.is_homology_manifold())
        var b = sc.betti_numbers()
        assert_equal(b[0], 1)
        for k in range(1, d + 1):
            assert_equal(b[k], 0)
    # in 2D, the same triangles as simplicial_grid_2d
    var g = simplicial_grid_2d(3)
    var kg = kuhn_grid([3, 3])
    var a = List[String]()
    var c = List[String]()
    for t in range(rows_of(g.elements)):
        a.append(String(row_list(g.elements, t)))
        c.append(String(row_list(kg.elements, t)))
    sort(a)
    sort(c)
    for t in range(len(a)):
        assert_equal(a[t], c[t])
    with assert_raises(contains="dims is empty"):
        _ = kuhn_grid(List[Int]())
    with assert_raises(contains="need at least 1"):
        _ = kuhn_grid([2, 0])
    with assert_raises(contains="2 lengths for 3 axes"):
        _ = kuhn_grid([2, 2, 2], [1.0, 1.0])
    with assert_raises(contains="is not positive"):
        _ = kuhn_grid([2, 2], [1.0, -1.0])


def test_flat_torus() raises:
    for d in range(1, 5):
        var dims = List[Int]()
        var lengths = List[Float64]()
        var vol: Float64 = 1
        for a in range(d):
            dims.append(3 + a)
            lengths.append(1.0 + 0.5 * Float64(a))
            vol *= lengths[a]
        var mesh = flat_torus(dims, lengths)
        assert_equal(cols_of(mesh.vertices), 2 * d)
        var sc = SimplicialComplex(mesh)
        # the flat torus exactly: its volume, and every step along an axis
        # of the grid spacing
        var total: Float64 = 0
        for t in range(sc[d].num_simplices):
            total += sc[d].primal_volume.unsafe_get(t)
        # thousands of simplices with cos/sin coordinates: 1.6e-13 in 4D
        assert_almost_equal(total, vol, atol=1e-12 * vol, rtol=0)
        assert_almost_equal(
            sc[1].primal_volume.unsafe_get(0),
            lengths[0] / Float64(dims[0]),
            atol=1e-15,
            rtol=0,
        )
        # a closed manifold with Betti numbers binomial(d, k); the link of
        # every cell of the 4-torus takes about a second, so up to 3D
        if d <= 3:
            assert_true(sc.is_homology_manifold())
        assert_equal(len(sc.boundary_cells().indices()[d - 1]), 0)
        var b = sc.betti_numbers()
        var binom = 1
        for k in range(d + 1):
            assert_equal(b[k], binom)
            binom = binom * (d - k) // (k + 1)
    with assert_raises(contains="need at least 3"):
        _ = flat_torus([4, 2])


def _grid_index(v: Int, vdims: List[Int]) -> List[Int]:
    var out = List[Int]()
    var rem = v
    for a in range(len(vdims)):
        out.append(rem % vdims[a])
        rem //= vdims[a]
    return out^


def _check_fdm(
    sc: SimplicialComplex,
    vdims: List[Int],
    h: List[Float64],
    periodic: Bool,
) raises:
    """Rows of d0^T W1 d0 at interior vertices (all, on a torus) are the
    finite-difference Laplacian times the cell volume."""
    var d = len(h)
    var V: Float64 = 1
    for a in range(d):
        V *= h[a]
    var K = sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d
    var diag: Float64 = 0
    for a in range(d):
        diag += 2 * V / (h[a] * h[a])
    var checked = 0
    for v in range(K.nrows):
        var iv = _grid_index(v, vdims)
        var interior = True
        if not periodic:
            for a in range(d):
                if iv[a] == 0 or iv[a] == vdims[a] - 1:
                    interior = False
        if not interior:
            continue
        checked += 1
        var nonzero = 0
        for q in range(K.indptr[v], K.indptr[v + 1]):
            var u = K.indices[q]
            var iu = _grid_index(u, vdims)
            var want: Float64 = 0
            if u == v:
                want = diag
            else:
                var axis = -1
                var steps = 0
                for a in range(d):
                    var s = iu[a] - iv[a]
                    if periodic:
                        s = (s + vdims[a] + 1) % vdims[a] - 1
                    if s != 0:
                        steps += 1
                        axis = a if abs(s) == 1 else -2
                if steps == 1 and axis >= 0:
                    want = -V / (h[axis] * h[axis])
            # Whitney entries from QR minors: rounding relative to diag
            assert_almost_equal(K.data[q], want, atol=1e-13 * diag, rtol=0)
            if abs(K.data[q]) > 1e-13 * diag:
                nonzero += 1
        assert_equal(nonzero, 2 * d + 1)
    assert_true(checked > 0)


def test_whitney_stiffness_is_finite_differences() raises:
    for d in range(1, 4):
        var dims = List[Int]()
        var lengths = List[Float64]()
        var h = List[Float64]()
        var vdims = List[Int]()
        var tdims = List[Int]()
        for a in range(d):
            dims.append(3 + a)
            lengths.append(1.0 + 0.75 * Float64(a))
            h.append(lengths[a] / Float64(dims[a]))
            vdims.append(dims[a] + 1)
            tdims.append(dims[a])
        _check_fdm(SimplicialComplex(kuhn_grid(dims, lengths)), vdims, h, False)
        _check_fdm(SimplicialComplex(flat_torus(dims, lengths)), tdims, h, True)


def test_flat_torus_spectrum_and_harmonic_forms() raises:
    # the Laplacian on the torus R^2 / (Z x 2Z): eigenvalues
    # (2 pi)^2 (m^2 + (n/2)^2); the first nonzero one, pi^2, twice
    var sc = SimplicialComplex(flat_torus([16, 16], [1.0, 2.0]))
    var K = sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d
    var M = whitney_innerproduct(sc, 0)
    var eig = eigh_generalized(K.todense(), M.todense())
    var eta = eig.residuals(K, M)
    for i in range(eta.size):
        assert_true(eta.unsafe_get(i) < 1e-13)
    ref lam = eig.values
    assert_almost_equal(lam.unsafe_get(0), 0.0, atol=1e-12, rtol=0)
    # the mode cos(pi y) is constant along x, and on the Kuhn grid the P1
    # stiffness and mass stencils reduce to the 1-D ones: with h = 2/16 and
    # theta = pi/8, lambda = 6 (1 - cos theta) / (h^2 (2 + cos theta)),
    # 1.3% above pi^2 (the O(h^2) error of P1 elements)
    var hy = 2.0 / 16.0
    var c = cos(pi / 8.0)
    var exact_discrete = 6 * (1 - c) / (hy * hy * (2 + c))
    for i in [1, 2]:
        assert_almost_equal(
            lam.unsafe_get(i), exact_discrete, atol=1e-11, rtol=0
        )
    assert_almost_equal(exact_discrete, pi * pi, atol=0.015 * pi * pi, rtol=0)
    assert_almost_equal(
        lam.unsafe_get(1), lam.unsafe_get(2), atol=1e-10, rtol=0
    )
    assert_true(lam.unsafe_get(3) > 3.9 * pi * pi)  # next: (2 pi)^2, 4 times
    # the constant 1-forms dx_a are harmonic: closed, and coclosed in the
    # Whitney inner product, exactly (up to rounding)
    var dims: List[Int] = [3, 4, 5]
    var t3 = SimplicialComplex(flat_torus(dims, [1.0, 1.5, 2.0]))
    var W1 = whitney_innerproduct(t3, 1)
    var E = t3[1].num_simplices
    for a in range(3):
        var w = nm.zeros[f64](Shape(E))
        var h = (1.0 + 0.5 * Float64(a)) / Float64(dims[a])
        for e in range(E):
            var iu = _grid_index(Int(t3[1].simplices.unsafe_get(2 * e)), dims)
            var iv = _grid_index(
                Int(t3[1].simplices.unsafe_get(2 * e + 1)), dims
            )
            var s = (iv[a] - iu[a] + dims[a] + 1) % dims[a] - 1
            w.unsafe_set(e, Float64(s) * h)
        var dw = t3[1].d @ w
        for i in range(dw.size):
            assert_almost_equal(dw.unsafe_get(i), 0.0, atol=1e-15, rtol=0)
        var co = t3[0].d.transpose() @ (W1 @ w)
        for i in range(co.size):
            assert_almost_equal(co.unsafe_get(i), 0.0, atol=1e-14, rtol=0)


def test_non_orientable_surfaces() raises:
    # the Möbius band and the Klein bottle: homology manifolds that no
    # orientation makes coherent, so SimplicialComplex refuses them
    var band = mobius_band(12, 2)
    var klein = klein_bottle([6, 5])
    assert_equal(rows_of(band.vertices), 36)
    assert_equal(rows_of(band.elements), 48)
    assert_equal(rows_of(klein.vertices), 30)
    assert_equal(rows_of(klein.elements), 60)
    for which in range(2):
        var m = band.copy() if which == 0 else klein.copy()
        assert_true(not coherent_orientation(m.elements))
        with assert_raises(
            contains="orient_coherently: the mesh is not orientable"
        ):
            _ = orient_coherently(m)
        with assert_raises(contains="the mesh is not orientable"):
            _ = SimplicialComplex(m)
        # orientation does not matter to rational homology or to the
        # manifold check
        var asc = AbstractSimplicialComplex([m.elements.copy()])
        var b = asc.betti_numbers()
        assert_equal(b[0], 1)
        assert_equal(b[1], 1)
        assert_equal(b[2], 0)
        assert_equal(asc.euler_characteristic(), 0)
        assert_true(asc.is_homology_manifold())
    # the band's boundary is one circle twice as long as the band
    var bd = AbstractSimplicialComplex([band.elements.copy()]).boundary_cells()
    assert_equal(len(bd[1].indices()), 24)
    assert_equal(len(bd[0].indices()), 24)
    assert_equal(
        len(
            AbstractSimplicialComplex([klein.elements.copy()])
            .boundary_cells()[1]
            .indices()
        ),
        0,
    )
    # the flat strip and torus glued without a flip are orientable
    _ = SimplicialComplex(flat_torus([6, 5]))
    with assert_raises(contains="segments >= 3"):
        _ = mobius_band(2)
    with assert_raises(contains="width must be in (0, 2)"):
        _ = mobius_band(5, 1, 2.0)
    with assert_raises(contains="each at least 3"):
        _ = klein_bottle([6, 2])


def test_morton_keys() raises:
    # the corners of a square, in Z order: axis 0 is the lower bit
    var V = real_array_from_rows(
        [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    var keys = morton_keys(V)
    # 21 bits per axis: (1, 0) sets every even bit, (0, 1) every odd one
    var even = ((1 << 42) - 1) // 3
    assert_equal(keys, [(1 << 42) - 1, 0, even, 2 * even])
    # an axis on which all points agree contributes nothing
    var flat = real_array_from_rows([[0.0, 5.0], [1.0, 5.0], [0.5, 5.0]])
    var fk = morton_keys(flat)
    assert_equal(fk[0], 0)
    assert_equal(fk[2] & 0xAAAAAAAAAAA, 0)
    assert_true(fk[0] < fk[2] and fk[2] < fk[1])
    # one bit per axis at 63 coordinates, and no more
    var wide = real_array(2, 63)
    fset(wide, 1, 62, 1.0)
    assert_equal(morton_keys(wide), [0, 1 << 62])
    with assert_raises(contains="1 to 63 coordinates"):
        _ = morton_keys(real_array(2, 64))
    var bad = real_array_from_rows([[0.0, 0.0], [1.0, Float64.MAX * 2]])
    with assert_raises(contains="coordinate 1 of point 1 is not finite"):
        _ = morton_keys(bad)


def _lcg(mut state: UInt64) -> Int:
    state = state * 6364136223846793005 + 1442695040888963407
    return Int(state >> 33)


def _jittered_cube(n: Int, shuffle: Bool) raises -> SimplicialMesh:
    """A Kuhn grid of the unit cube with interior vertices moved up to 5%
    of a cell; with `shuffle`, its vertex numbers and element order
    randomly permuted."""
    var mesh = kuhn_grid([n, n, n])
    var nv = rows_of(mesh.vertices)
    var ne = rows_of(mesh.elements)
    var state: UInt64 = 2024
    var h = 1.0 / Float64(n)
    for i in range(nv):
        var interior = True
        for a in range(3):
            var x = fget(mesh.vertices, i, a)
            if x < 0.5 * h or x > 1.0 - 0.5 * h:
                interior = False
        if interior:
            for a in range(3):
                var j = Float64(_lcg(state) % 2001 - 1000) / 1000.0
                fset(
                    mesh.vertices,
                    i,
                    a,
                    fget(mesh.vertices, i, a) + 0.05 * h * j,
                )
    if not shuffle:
        return mesh^
    var p = List[Int](length=nv, fill=0)
    for i in range(nv):
        p[i] = i
    for i in range(nv - 1, 0, -1):
        var j = _lcg(state) % (i + 1)
        var t = p[i]
        p[i] = p[j]
        p[j] = t
    var q = List[Int](length=ne, fill=0)
    for i in range(ne):
        q[i] = i
    for i in range(ne - 1, 0, -1):
        var j = _lcg(state) % (i + 1)
        var t = q[i]
        q[i] = q[j]
        q[j] = t
    var V = mesh.vertices.copy()
    for i in range(nv):
        for a in range(3):
            fset(V, p[i], a, fget(mesh.vertices, i, a))
    var E = mesh.elements.copy()
    for e in range(ne):
        for j in range(4):
            iset(E, q[e], j, p[iget(mesh.elements, e, j)])
    return SimplicialMesh(V, E)


def test_reorder_vertices_small() raises:
    var V = real_array_from_rows(
        [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    var E = int_array_from_rows([[1, 2, 0], [1, 0, 3]], 3)
    var r = reorder_vertices(SimplicialMesh(V, E))
    assert_equal(r.old_vertex, [1, 2, 3, 0])
    assert_equal(r.new_vertex, [3, 0, 1, 2])
    assert_equal(r.old_element, [0, 1])
    # each element keeps its vertices in place: [1, 2, 0] -> [0, 1, 3]
    assert_equal(row_list(r.mesh.elements, 0), [0, 1, 3])
    assert_equal(row_list(r.mesh.elements, 1), [0, 3, 2])
    assert_equal(fget(r.mesh.vertices, 3, 0), 1.0)
    assert_equal(fget(r.mesh.vertices, 3, 1), 1.0)
    # elements whose sorted rows are out of order are swapped
    var E2 = int_array_from_rows([[0, 1, 3], [1, 2, 0]], 3)
    var r2 = reorder_vertices(SimplicialMesh(V, E2))
    assert_equal(r2.old_element, [1, 0])
    # an element naming a missing vertex
    var bad = SimplicialMesh(V, E)
    iset(bad.elements, 1, 2, 7)
    with assert_raises(contains="element 1 names vertex 7 of 4"):
        _ = reorder_vertices(bad)


def test_reorder_vertices() raises:
    var mesh = _jittered_cube(5, shuffle=True)
    # an unused vertex, kept and placed by its coordinates
    var nv0 = rows_of(mesh.vertices)
    var V = real_array(nv0 + 1, 3)
    for i in range(nv0):
        for a in range(3):
            fset(V, i, a, fget(mesh.vertices, i, a))
    fset(V, nv0, 0, 0.3)
    fset(V, nv0, 1, 0.6)
    fset(V, nv0, 2, 0.1)
    mesh = SimplicialMesh(V, mesh.elements)
    var r = reorder_vertices(mesh)
    var nv = rows_of(mesh.vertices)
    var ne = rows_of(mesh.elements)
    assert_equal(len(r.old_vertex), nv)
    assert_equal(len(r.old_element), ne)
    var seen = List[Bool](length=ne, fill=False)
    for t in range(ne):
        assert_false(seen[r.old_element[t]])
        seen[r.old_element[t]] = True
    for i in range(nv):
        assert_equal(r.new_vertex[r.old_vertex[i]], i)
        for a in range(3):
            assert_equal(
                fget(r.mesh.vertices, i, a),
                fget(mesh.vertices, r.old_vertex[i], a),
            )
    for t in range(ne):
        for j in range(4):
            assert_equal(
                iget(r.mesh.elements, t, j),
                r.new_vertex[iget(mesh.elements, r.old_element[t], j)],
            )
    # vertices in Morton order, elements in order of their sorted rows
    var keys = morton_keys(r.mesh.vertices)
    for i in range(1, nv):
        assert_true(keys[i - 1] <= keys[i])
    var s = r.mesh.elements.copy()
    sort_each_row(s)
    for t in range(1, ne):
        assert_equal(compare_rows(s, t - 1, s, t), -1)
    # orientations are unchanged, so the complex builds; the same complex
    var a = SimplicialComplex(mesh)
    var b = SimplicialComplex(r.mesh)
    for k in range(4):
        assert_equal(a[k].num_simplices, b[k].num_simplices)
    for t in range(ne):
        assert_almost_equal(
            b[3].primal_volume.unsafe_get(t),
            a[3].primal_volume.unsafe_get(r.old_element[t]),
            atol=1e-17,
            rtol=1e-12,
        )
    # 0-simplices are the used vertices, so not every vertex has one
    for i in range(b[0].num_simplices):
        var v = iget(b[0].simplices, i, 0)
        var j = a[0].find_vertices([r.old_vertex[v]])
        assert_true(j >= 0)
        assert_almost_equal(
            b[0].dual_volume.unsafe_get(i),
            a[0].dual_volume.unsafe_get(j),
            atol=1e-17,
            rtol=1e-12,
        )
    for k in range(4):
        var na: Float64 = 0
        var nb: Float64 = 0
        for x in whitney_innerproduct(a, k).data:
            na += x * x
        for x in whitney_innerproduct(b, k).data:
            nb += x * x
        assert_almost_equal(na, nb, atol=0, rtol=1e-12)
    # the result does not depend on the numbering it starts from (the
    # Morton keys are distinct here)
    var direct = reorder_vertices(_jittered_cube(5, shuffle=False))
    var shuffled = reorder_vertices(_jittered_cube(5, shuffle=True))
    for i in range(direct.mesh.vertices.size):
        assert_equal(
            direct.mesh.vertices.unsafe_get(i),
            shuffled.mesh.vertices.unsafe_get(i),
        )
    for i in range(direct.mesh.elements.size):
        assert_equal(
            direct.mesh.elements.unsafe_get(i),
            shuffled.mesh.elements.unsafe_get(i),
        )
    # reordering again changes nothing
    var again = reorder_vertices(r.mesh)
    for i in range(nv):
        assert_equal(again.old_vertex[i], i)
    for t in range(ne):
        assert_equal(again.old_element[t], t)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
