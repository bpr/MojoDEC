from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    SimplicialComplex,
    TypedComplex,
    cohomology_basis,
    d,
    delta,
    flat_torus,
    harmonic_basis,
    kuhn_grid,
    read_mesh,
    whitney_innerproduct,
)
from mojodec.core.cochain import Cochain
from mojodec.core.primality import Primality
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    RealArray,
    cols_of,
    dot,
    fget,
    norm2,
    real_vector,
    rows_of,
    take_int_rows,
)
from mojodec.routines.cohomology import _Sparse, _reduce


def _remove_cells(
    dims: List[Int], boxes: List[List[Float64]]
) raises -> SimplicialComplex:
    """A Kuhn grid of the unit square without the triangles whose centroid
    lies in one of the boxes `[x0, x1, y0, y1]`."""
    var m = kuhn_grid(dims)
    var keep = List[Int]()
    for t in range(rows_of(m.elements)):
        var x: Float64 = 0
        var y: Float64 = 0
        for j in range(3):
            var v = Int(m.elements.unsafe_get(3 * t + j))
            x += fget(m.vertices, v, 0) / 3
            y += fget(m.vertices, v, 1) / 3
        var inside = False
        for ref b in boxes:
            if b[0] < x and x < b[1] and b[2] < y and y < b[3]:
                inside = True
        if not inside:
            keep.append(t)
    return SimplicialComplex(m.vertices, take_int_rows(m.elements, keep))


def _complexes() raises -> List[SimplicialComplex]:
    var out = List[SimplicialComplex]()
    out.append(SimplicialComplex(flat_torus([3, 4], [1.0, 2.0])))
    out.append(SimplicialComplex(flat_torus([3, 3, 3])))
    out.append(SimplicialComplex(read_mesh("tests/meshes/torus_tet.xml")))
    out.append(SimplicialComplex(read_mesh("tests/meshes/sphere3.xml")))
    # a square with two holes, and two squares (the middle column removed)
    out.append(
        _remove_cells([5, 5], [[0.2, 0.4, 0.2, 0.4], [0.6, 0.8, 0.6, 0.8]])
    )
    out.append(_remove_cells([5, 2], [[0.4, 0.6, 0.0, 1.0]]))
    return out^


def test_cocycles_and_cycles() raises:
    # counts equal the Betti numbers; cocycles and cycles are integer, closed
    # and boundaryless exactly, and dual to each other exactly
    var expected: List[List[Int]] = [
        [1, 2, 1],
        [1, 3, 3, 1],
        [1, 1, 0, 0],
        [1, 0, 1],
        [1, 2, 0],
        [2, 0, 0],
    ]
    var complexes = _complexes()
    for c in range(len(complexes)):
        ref sc = complexes[c]
        var n = sc.complex_dimension()
        var betti = sc.betti_numbers()
        for k in range(n + 1):
            assert_equal(betti[k], expected[c][k])
            var B = cohomology_basis(sc, k)
            assert_equal(len(B), betti[k])
            assert_equal(B.k, k)
            for i in range(len(B)):
                ref z = B.cocycles[i].v
                ref cyc = B.cycles[i]
                for t in range(z.size):
                    var a = z.unsafe_get(t)
                    var b = cyc.unsafe_get(t)
                    assert_true(a == Float64(Int(a)) and abs(a) <= 1)
                    assert_true(b == Float64(Int(b)) and abs(b) <= 1)
                if k < n:
                    var dz = sc[k].d @ z
                    for t in range(dz.size):
                        assert_equal(dz.unsafe_get(t), 0.0)
                if k > 0:
                    var bc = sc[k].boundary @ cyc
                    for t in range(bc.size):
                        assert_equal(bc.unsafe_get(t), 0.0)
                var p = B.periods(B.cocycles[i])
                for j in range(len(B)):
                    assert_equal(p[j], 1.0 if i == j else 0.0)


def _fit_residual(h: RealArray, basis: List[RealArray]) raises -> Float64:
    """`|h - P h| / |h|`, P the least-squares projection onto the span of
    `basis` (at most three vectors, by Gram-Schmidt)."""
    var q = List[RealArray]()
    for ref b in basis:
        var v = b.copy()
        for ref u in q:
            var c = dot(u, v)
            for t in range(v.size):
                v.unsafe_set(t, v.unsafe_get(t) - c * u.unsafe_get(t))
        var s = norm2(v)
        for t in range(v.size):
            v.unsafe_set(t, v.unsafe_get(t) / s)
        q.append(v^)
    var r = h.copy()
    for ref u in q:
        var c = dot(u, r)
        for t in range(r.size):
            r.unsafe_set(t, r.unsafe_get(t) - c * u.unsafe_get(t))
    return norm2(r) / norm2(h)


def _grid_index(v: Int, dims: List[Int]) -> List[Int]:
    var out = List[Int]()
    var rem = v
    for a in range(len(dims)):
        out.append(rem % dims[a])
        rem //= dims[a]
    return out^


def test_constant_forms_on_flat_torus() raises:
    # on a flat torus the harmonic 1-forms are the constant dx_a, whose
    # cochains are h_a times the edge's step along axis a; both the
    # Whitney and the diagonal-star harmonic bases span exactly those
    var dims: List[Int] = [3, 4, 5]
    var lengths: List[Float64] = [1.0, 1.5, 2.0]
    var sc = SimplicialComplex(flat_torus(dims, lengths))
    var E = sc[1].num_simplices
    var constant = List[RealArray]()
    for a in range(3):
        var w = real_vector(E)
        for e in range(E):
            var iu = _grid_index(Int(sc[1].simplices.unsafe_get(2 * e)), dims)
            var iv = _grid_index(
                Int(sc[1].simplices.unsafe_get(2 * e + 1)), dims
            )
            var s = (iv[a] - iu[a] + dims[a] + 1) % dims[a] - 1
            w.unsafe_set(e, Float64(s) * lengths[a] / Float64(dims[a]))
        constant.append(w^)
    var W1 = whitney_innerproduct(sc, 1)
    var whitney = harmonic_basis(sc, 1, W1)
    var star = harmonic_basis(sc, 1)
    assert_equal(len(whitney), 3)
    assert_equal(len(star), 3)
    var B = cohomology_basis(sc, 1)
    for i in range(3):
        assert_true(_fit_residual(whitney[i].v, constant) < 1e-12)
        assert_true(_fit_residual(star[i].v, constant) < 1e-12)
        # a constant form is harmonic in both inner products, and the
        # periods fix the combination: the two bases agree
        for t in range(E):
            assert_almost_equal(
                whitney[i].v.unsafe_get(t),
                star[i].v.unsafe_get(t),
                atol=1e-12,
                rtol=0,
            )
        var p = B.periods(whitney[i])
        for j in range(3):
            assert_almost_equal(
                p[j], 1.0 if i == j else 0.0, atol=1e-13, rtol=0
            )
    # 2-forms: closed, and coclosed in the Whitney inner product
    var W2 = whitney_innerproduct(sc, 2)
    var H2 = harmonic_basis(sc, 2, W2)
    var B2 = cohomology_basis(sc, 2)
    assert_equal(len(H2), 3)
    for i in range(3):
        var dh = sc[2].d @ H2[i].v
        assert_true(norm2(dh) < 1e-14)
        var co = sc[2].boundary @ (W2 @ H2[i].v)
        # MINRES stops at 1e-12 relative in its preconditioned norm
        assert_true(norm2(co) < 1e-10 * norm2(H2[i].v))
        var p = B2.periods(H2[i])
        for j in range(3):
            assert_almost_equal(
                p[j], 1.0 if i == j else 0.0, atol=1e-13, rtol=0
            )


def test_top_degree_is_the_volume_form() raises:
    # with the diagonal star (1 / area), the harmonic 2-cochain on a closed
    # surface is the area form over the total area: its period over the
    # fundamental cycle is 1
    var sc = SimplicialComplex(flat_torus([3, 4], [1.0, 2.0]))
    var H = harmonic_basis(sc, 2)
    assert_equal(len(H), 1)
    var area: Float64 = 0
    for t in range(sc[2].num_simplices):
        area += sc[2].primal_volume.unsafe_get(t)
    var sign = 1.0 if H[0].v.unsafe_get(0) > 0 else -1.0
    for t in range(sc[2].num_simplices):
        assert_almost_equal(
            sign * H[0].v.unsafe_get(t) * area,
            sc[2].primal_volume.unsafe_get(t),
            atol=1e-14,
            rtol=0,
        )
    # 0-forms: the indicators of the connected pieces
    var two = _remove_cells([5, 2], [[0.4, 0.6, 0.0, 1.0]])
    var H0 = harmonic_basis(two, 0)
    assert_equal(len(H0), 2)
    var total = real_vector(two[0].num_simplices)
    for ref h in H0:
        for t in range(h.v.size):
            var x = h.v.unsafe_get(t)
            assert_true(x == 0.0 or x == 1.0)
            total.unsafe_set(t, total.unsafe_get(t) + x)
        assert_true(norm2(d(h).v) == 0)
    for t in range(total.size):
        assert_equal(total.unsafe_get(t), 1.0)


def test_square_with_holes() raises:
    # one harmonic 1-form per hole, circulating once around it
    var sc = _remove_cells([5, 5], [[0.2, 0.4, 0.2, 0.4], [0.6, 0.8, 0.6, 0.8]])
    var B = cohomology_basis(sc, 1)
    var M = whitney_innerproduct(sc, 1)
    var star = harmonic_basis(sc, 1)
    var whitney = harmonic_basis(sc, 1, M)
    assert_equal(len(star), 2)
    for i in range(2):
        assert_true(norm2(d(star[i]).v) < 1e-14)
        assert_true(norm2(delta(star[i]).v) < 1e-10 * norm2(star[i].v))
        assert_true(norm2(d(whitney[i]).v) < 1e-14)
        var co = sc[1].boundary @ (M @ whitney[i].v)
        assert_true(norm2(co) < 1e-12 * norm2(whitney[i].v))
        var ps = B.periods(star[i])
        var pw = B.periods(whitney[i])
        for j in range(2):
            var want = 1.0 if i == j else 0.0
            assert_almost_equal(ps[j], want, atol=1e-13, rtol=0)
            assert_almost_equal(pw[j], want, atol=1e-13, rtol=0)
    # a closed cochain is its periods' combination plus an exact part: d of
    # a vertex function changes no period
    var f = real_vector(sc[0].num_simplices)
    for v in range(f.size):
        f.unsafe_set(v, Float64((7 * v) % 11))
    var z = B.cocycles[0].v.copy()
    var df = sc[0].d @ f
    for t in range(z.size):
        z.unsafe_set(t, 3 * z.unsafe_get(t) + df.unsafe_get(t))
    var p = B.periods(Cochain(sc, 1, Primality.PRIMAL, z^))
    assert_almost_equal(p[0], 3.0, atol=1e-12, rtol=0)
    assert_almost_equal(p[1], 0.0, atol=1e-12, rtol=0)


def test_typed() raises:
    var sc = _remove_cells([4, 4], [[0.25, 0.5, 0.25, 0.5]])
    var tc = TypedComplex[2](sc)
    var H = harmonic_basis[1](tc)
    var Hw = harmonic_basis(tc, whitney_innerproduct[1](tc))
    var U = harmonic_basis(sc, 1)
    var Uw = harmonic_basis(sc, 1, whitney_innerproduct(sc, 1))
    assert_equal(len(H), 1)
    assert_equal(len(Hw), 1)
    for t in range(U[0].v.size):
        assert_equal(H[0].v.unsafe_get(t), U[0].v.unsafe_get(t))
        assert_equal(Hw[0].v.unsafe_get(t), Uw[0].v.unsafe_get(t))
    var other = TypedComplex[2](_remove_cells([4, 4], [[0.25, 0.5, 0.25, 0.5]]))
    with assert_raises(contains="different complex"):
        _ = harmonic_basis(tc, whitney_innerproduct[1](other))


def test_errors() raises:
    var sc = SimplicialComplex(flat_torus([3, 3]))
    with assert_raises(contains="outside 0..2"):
        _ = cohomology_basis(sc, 3)
    with assert_raises(contains="outside 0..2"):
        _ = cohomology_basis(sc, -1)
    with assert_raises(contains="outside 0..2"):
        _ = harmonic_basis(sc, 3)
    with assert_raises(contains="M is not"):
        _ = harmonic_basis(sc, 1, whitney_innerproduct(sc, 0))
    var B = cohomology_basis(sc, 1)
    with assert_raises(contains="primal 1-cochain"):
        _ = B.periods(sc.get_cochain(2))
    # an elimination that needs the multiple 3/2 raises
    var a = _Sparse()
    a.idx.append(0)
    a.val.append(2)
    var b = _Sparse()
    b.idx.append(0)
    b.val.append(3)
    var cols = List[_Sparse]()
    cols.append(a^)
    cols.append(b^)
    with assert_raises(contains="non-integer multiple 3/2"):
        _ = _reduce(cols, 1, List[Bool](), "test")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
