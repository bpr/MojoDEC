from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
)

from mojodec import (
    PRIMAL,
    SimplicialComplex,
    TypedComplex,
    d,
    derham_map,
    form_index_sets,
    icosphere,
    kuhn_grid,
    l2_error,
    l2_projection,
    whitney_form,
    whitney_innerproduct,
    whitney_load,
)
from std.math import sqrt

from mojodec.core.cochain import Cochain
from mojodec.core.primality import Primality
from mojodec.core.utils import cols_of, fget, fset, real_array, real_vector
from mojodec.routines.geometry import barycentric_gradients
from mojodec.routines.quadrature import grundmann_moeller


def _box() raises -> SimplicialComplex:
    """A 3D Kuhn grid with its interior vertices moved, so no simplex is
    special."""
    var m = kuhn_grid([2, 2, 2], [1.0, 1.5, 2.0])
    for v in range(13, 14):  # the one interior vertex
        fset(m.vertices, v, 0, fget(m.vertices, v, 0) + 0.07)
        fset(m.vertices, v, 2, fget(m.vertices, v, 2) - 0.11)
    return SimplicialComplex(m)


# a 0-, 1-, and 2-form on R^3 and their exterior derivatives
def _f(x: List[Float64]) -> List[Float64]:
    return [x[0] * x[0] * x[1] + x[2]]


def _df(x: List[Float64]) -> List[Float64]:
    return [2 * x[0] * x[1], x[0] * x[0], 1.0]


def _w(x: List[Float64]) -> List[Float64]:
    return [x[1] * x[2] * x[2], x[0] * x[0], x[0] * x[1] * x[2]]


def _dw(x: List[Float64]) -> List[Float64]:
    # (dw)_ab = d_a w_b - d_b w_a for ab = xy, xz, yz
    return [2 * x[0] - x[2] * x[2], -x[1] * x[2], x[0] * x[2]]


def _eta(x: List[Float64]) -> List[Float64]:
    return [x[0] * x[1], x[2] * x[2], x[0] * x[2]]


def _deta(x: List[Float64]) -> List[Float64]:
    # d_z eta_xy - d_y eta_xz + d_x eta_yz
    return [x[2]]


def _max_diff(a: Cochain, b: Cochain) -> Float64:
    var m: Float64 = 0
    for i in range(a.v.size):
        m = max(m, abs(a.v.unsafe_get(i) - b.v.unsafe_get(i)))
    return m


def test_form_index_sets() raises:
    var s = form_index_sets(3, 2)
    assert_equal(len(s), 3)
    assert_equal(String(s[0]), "[0, 1]")
    assert_equal(String(s[1]), "[0, 2]")
    assert_equal(String(s[2]), "[1, 2]")
    assert_equal(len(form_index_sets(4, 0)), 1)
    assert_equal(len(form_index_sets(4, 0)[0]), 0)
    assert_equal(len(form_index_sets(5, 2)), 10)
    with assert_raises(contains="k = 4 is outside 0..3"):
        _ = form_index_sets(3, 4)


def test_stokes() raises:
    # R(d omega) = d R(omega), exactly for polynomial forms: Stokes'
    # theorem on every simplex
    var sc = _box()
    assert_true(
        _max_diff(derham_map(sc, 1, 3, _df), d(derham_map(sc, 0, 0, _f)))
        < 1e-14
    )
    assert_true(
        _max_diff(derham_map(sc, 2, 3, _dw), d(derham_map(sc, 1, 3, _w)))
        < 1e-14
    )
    assert_true(
        _max_diff(derham_map(sc, 3, 1, _deta), d(derham_map(sc, 2, 3, _eta)))
        < 1e-14
    )
    # on a surface in R^3: the forms are restricted to the triangles
    var s2 = SimplicialComplex(icosphere(2))
    assert_true(
        _max_diff(derham_map(s2, 1, 3, _df), d(derham_map(s2, 0, 0, _f)))
        < 1e-14
    )
    assert_true(
        _max_diff(derham_map(s2, 2, 3, _dw), d(derham_map(s2, 1, 3, _w)))
        < 1e-14
    )

    # the integral of dz over each edge is its rise, of dx^dy^dz over
    # each tetrahedron its volume: kuhn_grid orients them all positively,
    # and the top-level cochains follow the mesh's orientation
    def dz(x: List[Float64]) -> List[Float64]:
        return [0.0, 0.0, 1.0]

    var rise = derham_map(sc, 1, 0, dz)
    for e in range(sc[1].num_simplices):
        var a = Int(sc[1].simplices.unsafe_get(2 * e))
        var b = Int(sc[1].simplices.unsafe_get(2 * e + 1))
        assert_almost_equal(
            rise.v.unsafe_get(e),
            fget(sc.vertices(), b, 2) - fget(sc.vertices(), a, 2),
            atol=1e-15,
            rtol=0,
        )

    def vol(x: List[Float64]) -> List[Float64]:
        return [1.0]

    var V = derham_map(sc, 3, 0, vol)
    var total: Float64 = 0
    for t in range(sc[3].num_simplices):
        assert_true(V.v.unsafe_get(t) > 0)
        total += V.v.unsafe_get(t)
    assert_almost_equal(total, 3.0, atol=1e-14, rtol=0)


def _faces_with_elements(
    sc: SimplicialComplex, k: Int
) raises -> List[List[Int]]:
    """For each k-simplex, the elements containing it."""
    var n = sc.complex_dimension()
    var out = List[List[Int]]()
    for _ in range(sc[k].num_simplices):
        out.append(List[Int]())
    var local = List[Int]()
    for t in range(sc[n].num_simplices):
        # every (k+1)-subset of the element's vertices
        for mask in range(1 << (n + 1)):
            var face = List[Int]()
            for j in range(n + 1):
                if (mask >> j) & 1 == 1:
                    face.append(
                        Int(sc[n].simplices.unsafe_get(t * (n + 1) + j))
                    )
            if len(face) == k + 1:
                out[sc[k].find_vertices(face)].append(t)
    return out^


def _integrate_whitney(c: Cochain, t: Int, face: List[Int]) raises -> Float64:
    """int over the k-face `face` (global vertices, sorted) of W c, as
    evaluated in element `t`: the midpoint rule, exact for degree 1."""
    ref sc = c.complex
    var n = sc.complex_dimension()
    var k = len(face) - 1
    var N = cols_of(sc.vertices())
    var bary = List[Float64](length=n + 1, fill=0.0)
    for j in range(n + 1):
        var v = Int(sc[n].simplices.unsafe_get(t * (n + 1) + j))
        for f in face:
            if f == v:
                bary[j] = 1.0 / Float64(k + 1)
    var w = whitney_form(c, t, bary)
    # contract with the face's edge vectors
    var sets = form_index_sets(N, k)
    var kfact: Float64 = 1
    for j in range(2, k + 1):
        kfact *= Float64(j)
    var total: Float64 = 0
    for I in range(len(sets)):
        var M = real_array(max(k, 1), max(k, 1))
        for r in range(k):
            for col in range(k):
                fset(
                    M,
                    r,
                    col,
                    fget(sc.vertices(), face[r + 1], sets[I][col])
                    - fget(sc.vertices(), face[0], sets[I][col]),
                )
        var det: Float64 = 1
        if k == 1:
            det = fget(M, 0, 0)
        elif k == 2:
            det = fget(M, 0, 0) * fget(M, 1, 1) - fget(M, 0, 1) * fget(M, 1, 0)
        elif k == 3:
            det = (
                fget(M, 0, 0)
                * (
                    fget(M, 1, 1) * fget(M, 2, 2)
                    - fget(M, 1, 2) * fget(M, 2, 1)
                )
                - fget(M, 0, 1)
                * (
                    fget(M, 1, 0) * fget(M, 2, 2)
                    - fget(M, 1, 2) * fget(M, 2, 0)
                )
                + fget(M, 0, 2)
                * (
                    fget(M, 1, 0) * fget(M, 2, 1)
                    - fget(M, 1, 1) * fget(M, 2, 0)
                )
            )
        total += w[I] * det
    # the elements' cochain values follow the mesh's orientation
    if sc[k].simplex_parity.unsafe_get(sc[k].find_vertices(face)) == 1:
        total = -total
    return total / kfact


def test_derham_of_whitney_is_identity() raises:
    # R W c = c: each Whitney form integrates to 1 on its own simplex and
    # 0 on the others; the tangential trace on a face is the same from
    # every element around it
    var sc = _box()
    for k in range(4):
        var c = sc.get_cochain(k)
        for i in range(c.v.size):
            c.v.unsafe_set(i, Float64((i * 7) % 5) - 2.0 + 0.25 * Float64(k))
        var elems = _faces_with_elements(sc, k)
        for s in range(sc[k].num_simplices):
            var face = List[Int]()
            for j in range(k + 1):
                face.append(Int(sc[k].simplices.unsafe_get(s * (k + 1) + j)))
            for t in elems[s]:
                assert_almost_equal(
                    _integrate_whitney(c, t, face),
                    c.v.unsafe_get(s),
                    atol=1e-13,
                    rtol=0,
                )


def test_whitney_properties() raises:
    var sc = _box()
    var n = 3
    var bary: List[Float64] = [0.1, 0.2, 0.3, 0.4]
    # partition of unity: the Whitney 0-forms sum to 1
    var ones = sc.get_cochain(0)
    for i in range(ones.v.size):
        ones.v.unsafe_set(i, 1.0)
    for t in [0, 17, 47]:
        assert_almost_equal(
            whitney_form(ones, t, bary)[0], 1.0, atol=1e-15, rtol=0
        )
    # d W = W d. W c is linear in the barycentric coordinates, so its
    # derivative along x_a is sum_j (d lambda_j / d x_a) W c(vertex j)
    for k in range(3):
        var c = sc.get_cochain(k)
        for i in range(c.v.size):
            c.v.unsafe_set(i, Float64((i * 5) % 7) - 3.0)
        var dc = d(c)
        var sets_k = form_index_sets(3, k)
        var sets_k1 = form_index_sets(3, k + 1)
        for t in [0, 17, 47]:
            var pts = real_array(4, 3)
            for j in range(4):
                var v = Int(sc[n].simplices.unsafe_get(t * 4 + j))
                for col in range(3):
                    fset(pts, j, col, fget(sc.vertices(), v, col))
            var g = barycentric_gradients(pts)
            var at_vertex = List[List[Float64]]()
            for j in range(4):
                var e = List[Float64](length=4, fill=0.0)
                e[j] = 1.0
                at_vertex.append(whitney_form(c, t, e))
            var lhs = whitney_form(dc, t, bary)
            for J in range(len(sets_k1)):
                # (d w)_J = sum_p (-1)^p d_{J_p} w_{J without J_p}
                var want: Float64 = 0
                for p in range(k + 1):
                    var rest = List[Int]()
                    for q in range(k + 1):
                        if q != p:
                            rest.append(sets_k1[J][q])
                    var I = 0
                    while String(sets_k[I]) != String(rest):
                        I += 1
                    var deriv: Float64 = 0
                    for j in range(4):
                        deriv += fget(g, j, sets_k1[J][p]) * at_vertex[j][I]
                    want += (-1.0 if p % 2 == 1 else 1.0) * deriv
                assert_almost_equal(lhs[J], want, atol=1e-12, rtol=0)

    # W R reproduces constant forms
    def c1(x: List[Float64]) -> List[Float64]:
        return [0.5, -1.0, 2.0]

    def c2(x: List[Float64]) -> List[Float64]:
        return [1.0, 0.25, -3.0]

    var w1 = derham_map(sc, 1, 0, c1)
    var w2 = derham_map(sc, 2, 0, c2)
    for t in [0, 17, 47]:
        var a = whitney_form(w1, t, bary)
        var b = whitney_form(w2, t, bary)
        var ea: List[Float64] = [0.5, -1.0, 2.0]
        var eb: List[Float64] = [1.0, 0.25, -3.0]
        for i in range(3):
            assert_almost_equal(a[i], ea[i], atol=1e-13, rtol=0)
            assert_almost_equal(b[i], eb[i], atol=1e-13, rtol=0)


def test_mass_matrices_match_whitney_innerproduct() raises:
    # int <W_s, W_t> over the mesh, by a degree-2 rule (W is linear in
    # each element), against whitney_innerproduct's closed-form minors
    var m = kuhn_grid([2, 2], [1.0, 1.5])
    fset(m.vertices, 4, 0, fget(m.vertices, 4, 0) + 0.1)  # the middle vertex
    fset(m.vertices, 4, 1, fget(m.vertices, 4, 1) - 0.2)
    var sc = SimplicialComplex(m)
    var n = 2
    var rule = grundmann_moeller(n, 2)
    for k in range(n + 1):
        var M = whitney_innerproduct(sc, k).todense()
        var N = sc[k].num_simplices
        var ours = real_array(N, N)
        for t in range(sc[n].num_simplices):
            var vol = abs(sc[n].primal_volume.unsafe_get(t))
            # the element's k-faces
            var faces = List[Int]()
            var elems = _faces_with_elements(sc, k)
            for s in range(N):
                for e in elems[s]:
                    if e == t:
                        faces.append(s)
            for q in range(len(rule)):
                var bary = List[Float64]()
                for j in range(n + 1):
                    bary.append(rule.points.unsafe_get(q * (n + 1) + j))
                var vals = List[List[Float64]]()
                for s in faces:
                    var unit = sc.get_cochain(k)
                    unit.v.unsafe_set(s, 1.0)
                    vals.append(whitney_form(unit, t, bary))
                for a in range(len(faces)):
                    for b in range(len(faces)):
                        var ip: Float64 = 0
                        for I in range(len(vals[a])):
                            ip += vals[a][I] * vals[b][I]
                        var i = faces[a]
                        var j = faces[b]
                        fset(
                            ours,
                            i,
                            j,
                            fget(ours, i, j)
                            + vol * rule.weights.unsafe_get(q) * ip,
                        )
        for i in range(N):
            for j in range(N):
                assert_almost_equal(
                    fget(ours, i, j), fget(M, i, j), atol=1e-13, rtol=0
                )


def test_load_projection_and_error() raises:
    # constant forms are Whitney forms (W R omega = omega), so their load
    # vector is M R omega, their projection R omega, and both errors vanish
    var sc = _box()
    for k in range(4):
        var deg = k

        def const(x: List[Float64]) raises {deg} -> List[Float64]:
            if deg == 1:
                return [0.5, -1.0, 2.0]
            if deg == 2:
                return [1.0, 0.25, -3.0]
            return [1.0]

        var r = derham_map(sc, k, 0, const)
        var b = whitney_load(sc, k, 1, const)
        var Mr = whitney_innerproduct(sc, k) @ r.v
        for i in range(b.size):
            assert_almost_equal(
                b.unsafe_get(i), Mr.unsafe_get(i), atol=1e-14, rtol=0
            )
        var p = l2_projection(sc, k, 1, const)
        for i in range(p.v.size):
            # CG to rtol 1e-12; the mass matrix is well conditioned
            assert_almost_equal(
                p.v.unsafe_get(i), r.v.unsafe_get(i), atol=1e-11, rtol=0
            )
        assert_true(l2_error(r, 2, const) < 1e-13)

    def one(x: List[Float64]) -> List[Float64]:
        return [1.0]

    def c1(x: List[Float64]) -> List[Float64]:
        return [0.5, -1.0, 2.0]

    # the error of the zero cochain is the form's norm: int_box 1 = 3
    assert_almost_equal(
        l2_error(sc.get_cochain(0), 0, one), sqrt(3.0), atol=1e-14, rtol=0
    )
    with assert_raises(contains="omega gave 1 components, expected 3"):
        _ = whitney_load(sc, 1, 1, one)
    with assert_raises(contains="omega gave 3 components, expected 1"):
        _ = l2_error(sc.get_cochain(0), 1, c1)


def test_typed_and_errors() raises:
    var m = kuhn_grid([3, 3], [1.0, 2.0])
    var tc = TypedComplex[2](m.vertices, m.elements)

    def g(x: List[Float64]) -> List[Float64]:
        return [x[1], x[0] * x[0]]

    var typed = derham_map[1](tc, 2, g)
    var plain = derham_map(tc.sc, 1, 2, g)
    for i in range(len(typed)):
        assert_equal(typed[i], plain.v.unsafe_get(i))
    var bary: List[Float64] = [0.2, 0.3, 0.5]
    var a = whitney_form(typed, 4, bary)
    var b = whitney_form(plain, 4, bary)
    assert_equal(a[0], b[0])
    assert_equal(a[1], b[1])
    # the load vector is a dual (n-k)-cochain, what the typed mass matrix
    # gives; the projection a primal k-cochain
    var load = whitney_load[1](tc, 3, g)  # TypedCochain[2, 1, DUAL]
    var raw_load = whitney_load(tc.sc, 1, 3, g)
    for i in range(len(load)):
        assert_equal(load[i], raw_load.unsafe_get(i))
    var proj = l2_projection[1](tc, 3, g)  # TypedCochain[2, 1, PRIMAL]
    assert_true(l2_error(proj, 5, g) <= l2_error(typed, 5, g))
    with assert_raises(contains="omega gave 2 components"):
        _ = derham_map(tc.sc, 2, 1, g)
    with assert_raises(contains="k = 3 is outside 0..2"):
        _ = derham_map(tc.sc, 3, 1, g)
    with assert_raises(contains="must be primal"):
        _ = whitney_form(tc.sc.get_cochain(1, Primality.DUAL), 0, bary)
    with assert_raises(contains="element 18 is outside"):
        _ = whitney_form(plain, 18, bary)
    with assert_raises(contains="2 barycentric coordinates"):
        _ = whitney_form(plain, 0, [0.5, 0.5])
    with assert_raises(contains="sum to"):
        _ = whitney_form(plain, 0, [0.5, 0.5, 0.5])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
