# Property-based tests, using proptest (the sibling ../proptest repository,
# a port of Python's Hypothesis). Each test states a law that must hold for
# every input in a family, and proptest searches that family for a
# counterexample, shrinking any it finds to a minimal one.
#
# proptest must be on the include path: tests/run_all.sh passes
# `-I $PROPTEST_PATH` (default ../proptest). To run just this file:
#
#   pixi run mojo run -I . -I build -I ../proptest tests/test_properties.mojo

from std.math import sqrt
from std.testing import TestSuite, assert_equal, assert_true

from proptest import (
    ConjectureData,
    Settings,
    assume,
    booleans,
    composite,
    fail,
    given,
    integers,
    lists,
    permutations,
    tuples,
)

from mojodec.core.cell_set import CellSet
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    _lexsort_rows,
    compare_rows,
    int_array,
    int_vector,
    lexsort_rows,
    rows_of,
    real_array_from_rows,
    real_vector_from_list,
    require_distinct_simplices,
)
from mojodec.routines.combinatorial import (
    combinations,
    permutation_parity,
    relative_parity,
)
from mojodec.routines.geometry import (
    barycentric_gradients,
    circumcenter,
    signed_volume,
    unsigned_volume,
)
from mojodec.routines.innerproduct import whitney_innerproduct
from mojodec.routines.mesh_generation import (
    cube_grid,
    icosphere,
    loop_subdivision,
    triangulate_ncube,
)
from mojodec.routines.simplex_array import (
    _boundary_by_buckets,
    _boundary_by_lexsort,
    _vertex_bound,
)
from mojodec.routines.topology import closure_of, star_of


# ===----------------------------------------------------------------------=== #
# Strategies
# ===----------------------------------------------------------------------=== #


def quick() -> Settings:
    """Settings for cheap properties."""
    return Settings(max_examples=200)


def meshes() -> Settings:
    """Settings for properties that build complexes: fewer examples, and no
    per-example deadline (building and checking a complex can take longer
    than proptest's default 200 ms)."""
    return Settings(max_examples=25, deadline=None)


def _permutation_of_range(mut data: ConjectureData) raises -> List[Int]:
    """A permutation of [0, n) for n in [1, 7]."""
    var n = data.draw(integers(1, 7), "n")
    var base = List[Int]()
    for i in range(n):
        base.append(i)
    return data.draw(permutations(base^), "perm")


def _two_permutations(
    mut data: ConjectureData,
) raises -> Tuple[List[Int], List[Int]]:
    """Two permutations of the same [0, n)."""
    var n = data.draw(integers(1, 7), "n")
    var base = List[Int]()
    for i in range(n):
        base.append(i)
    var p = data.draw(permutations(base.copy()), "p")
    var q = data.draw(permutations(base^), "q")
    return (p^, q^)


def _simplex_points(mut data: ConjectureData) raises -> List[List[Float64]]:
    """The M+1 vertices of an M-simplex in M-space (M = 1, 2 or 3), with
    small integer coordinates so volumes are exact."""
    var m = data.draw(integers(1, 3), "dimension")
    var pts = List[List[Float64]]()
    for _ in range(m + 1):
        var row = List[Float64]()
        for _ in range(m):
            row.append(Float64(data.draw(integers(-5, 5))))
        pts.append(row^)
    return pts^


def _embedded_simplex(
    mut data: ConjectureData,
) raises -> Tuple[List[List[Float64]], List[List[Float64]]]:
    """An M-simplex in M-space (M = 2 or 3) and the same simplex embedded in
    N-space (N = M+1 or M+2): the extra coordinates are constant, and the
    coordinates are then permuted. Half the time the last vertex is an
    integer combination of the others' edges, so the simplex is flat."""
    var m = data.draw(integers(2, 3), "M")
    var pts = List[List[Float64]]()
    for _ in range(m + 1):
        var row = List[Float64]()
        for _ in range(m):
            row.append(Float64(data.draw(integers(-5, 5))))
        pts.append(row^)
    if data.draw(booleans(), "flat"):
        # last = p0 + sum_i c_i (p_i - p0) over i = 1..m-1
        for j in range(m):
            var x = pts[0][j]
            pts[m][j] = x
        for i in range(1, m):
            var c = Float64(data.draw(integers(-3, 3)))
            for j in range(m):
                pts[m][j] += c * (pts[i][j] - pts[0][j])
    var extra = data.draw(integers(1, 2), "extra dimensions")
    var n = m + extra
    var fill = List[Float64]()
    for _ in range(extra):
        fill.append(Float64(data.draw(integers(-5, 5))))
    var base = List[Int]()
    for i in range(n):
        base.append(i)
    var perm = data.draw(permutations(base^), "coordinate order")
    var embedded = List[List[Float64]]()
    for p in pts:
        var full = p.copy()
        for f in fill:
            full.append(f)
        var row = List[Float64]()
        for j in range(n):
            row.append(full[perm[j]])
        embedded.append(row^)
    return (pts^, embedded^)


def _sparse_triplets(
    mut data: ConjectureData,
) raises -> Tuple[Int, Int, List[Int], List[Int], List[Float64]]:
    """A matrix shape and COO triplets (duplicates allowed), with small
    integer values so sums and products are exact."""
    var nrows = data.draw(integers(1, 6), "nrows")
    var ncols = data.draw(integers(1, 6), "ncols")
    var n = data.draw(integers(0, 12), "nnz")
    var rows = List[Int]()
    var cols = List[Int]()
    var vals = List[Float64]()
    for _ in range(n):
        rows.append(data.draw(integers(0, nrows - 1)))
        cols.append(data.draw(integers(0, ncols - 1)))
        vals.append(Float64(data.draw(integers(-9, 9))))
    return (nrows, ncols, rows^, cols^, vals^)


def _simplex_rows(
    mut data: ConjectureData,
) raises -> Tuple[Int, List[Int], List[Int]]:
    """`N` rows of `K` distinct vertex ids from `[0, nv)`, each sorted, and a
    parity per row (`K` in [2, 5], `N` in [1, 40], `nv` in [K, 12]): few
    ids, so rows and faces repeat and the sorts see ties."""
    var K = data.draw(integers(2, 5), "K")
    var nv = data.draw(integers(K, 12), "nv")
    var N = data.draw(integers(1, 40), "N")
    var base = List[Int]()
    for i in range(nv):
        base.append(i)
    var flat = List[Int]()
    var parity = List[Int]()
    for _ in range(N):
        var p = data.draw(permutations(base.copy()))
        var row = List[Int]()
        for j in range(K):
            row.append(p[j])
        sort(row)
        for v in row:
            flat.append(v)
        parity.append(data.draw(integers(0, 1)))
    return (K, flat^, parity^)


def _int_rows(
    mut data: ConjectureData,
) raises -> Tuple[Int, List[Int]]:
    """`N` rows of `K` integers (`K` in [1, 4], `N` in [1, 120]): from a
    handful of values, so that rows repeat and runs of equal first entries
    pass 32, or from a range too wide for a counting sort."""
    var K = data.draw(integers(1, 4), "K")
    var N = data.draw(integers(1, 120), "N")
    var wide = data.draw(booleans(), "wide")
    var hi = 1_000_000 if wide else data.draw(integers(1, 6), "values")
    var flat = List[Int]()
    for _ in range(N * K):
        flat.append(data.draw(integers(0, hi - 1)))
    return (K, flat^)


def _stable_row_order(a: IntArray) -> List[Int]:
    """The rows of `a` in lexicographic order, ties by index (insertion
    sort)."""
    var order = List[Int]()
    for i in range(rows_of(a)):
        order.append(i)
        var t = len(order) - 1
        while t > 0 and compare_rows(a, order[t - 1], a, order[t]) > 0:
            var x = order[t]
            order[t] = order[t - 1]
            order[t - 1] = x
            t -= 1
    return order^


def _grid_dims(mut data: ConjectureData) raises -> List[Int]:
    """Dimensions of a 1- or 2-dimensional cube grid (triangulate_ncube
    supports n <= 2)."""
    return data.draw(lists(integers(1, 4), min_size=1, max_size=2))


def _grid_complex(dims: List[Int]) raises -> SimplicialComplex:
    return SimplicialComplex(triangulate_ncube(cube_grid(dims)))


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _compose(p: List[Int], q: List[Int]) -> List[Int]:
    """(p . q)[i] = p[q[i]]."""
    var out = List[Int]()
    for i in range(len(q)):
        out.append(p[q[i]])
    return out^


def _binomial(n: Int, k: Int) -> Int:
    var r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def _close(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return abs(a - b) <= tol * max(1.0, max(abs(a), abs(b)))


def _dense(
    nrows: Int,
    ncols: Int,
    rows: List[Int],
    cols: List[Int],
    vals: List[Float64],
) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for _ in range(nrows):
        out.append(List[Float64](length=ncols, fill=0.0))
    for i in range(len(rows)):
        out[rows[i]][cols[i]] += vals[i]
    return out^


def _assert_zero(A: CSRMatrix, what: String) raises:
    if A.max_abs() != 0.0:
        fail(String(what, " has an entry of size ", A.max_abs()))


# ===----------------------------------------------------------------------=== #
# Combinatorics
# ===----------------------------------------------------------------------=== #


def test_parity_of_composition_is_xor() raises:
    def prop(pq: Tuple[List[Int], List[Int]]) raises:
        var p = pq[0].copy()
        var q = pq[1].copy()
        assert_equal(
            permutation_parity(_compose(p, q)),
            permutation_parity(p) ^ permutation_parity(q),
        )

    given(composite(_two_permutations), prop, quick())


def test_a_transposition_is_odd() raises:
    def prop(t: Tuple[Int, Int, Int]) raises:
        var n = t[0]
        var i = t[1] % n
        var j = t[2] % n
        assume(i != j)
        var p = List[Int]()
        for k in range(n):
            p.append(k)
        p.swap_elements(i, j)
        assert_equal(permutation_parity(p), 1)

    given(tuples(integers(2, 8), integers(0, 7), integers(0, 7)), prop, quick())


def test_relative_parity_is_parity_of_the_permutation() raises:
    def prop(p: List[Int]) raises:
        # distinct vertex labels, rearranged by p
        var A = List[Int]()
        for i in range(len(p)):
            A.append(3 * i + 1)
        var B = List[Int]()
        for i in range(len(p)):
            B.append(A[p[i]])
        assert_equal(relative_parity(A, B), permutation_parity(p))

    given(composite(_permutation_of_range), prop, quick())


def test_combinations_count_and_are_distinct_sorted_subsets() raises:
    def prop(nk: Tuple[Int, Int]) raises:
        var n = nk[0]
        var k = min(nk[1], n)
        var L = List[Int]()
        for i in range(n):
            L.append(i)
        var cs = combinations(L, k)
        assert_equal(len(cs), _binomial(n, k))
        for c in cs:
            assert_equal(len(c), k)
            for i in range(1, len(c)):
                assert_true(c[i - 1] < c[i], "combination is not increasing")

    given(tuples(integers(1, 8), integers(1, 8)), prop, quick())


# ===----------------------------------------------------------------------=== #
# Geometry
# ===----------------------------------------------------------------------=== #


def test_volume_is_invariant_under_translation_and_reordering() raises:
    def prop(pts: List[List[Float64]], shift: Int, rotate: Int) raises:
        # Degenerate simplices are included. This test found that the old
        # sqrt(|det(Gram)|) / M! formula turned a ~1e-16 rounding error into
        # an order-dependent ~1e-8 volume for four coplanar points
        # [[0,0,-2],[0,0,0],[0,1,0],[0,0,1]]; M-simplices in M-space now use
        # |det(edges)| / M! directly.
        var V = unsigned_volume(real_array_from_rows(pts))
        var moved = List[List[Float64]]()
        var n = len(pts)
        for i in range(n):
            # rotate the vertex order and translate every coordinate
            var row = pts[(i + rotate) % n].copy()
            for j in range(len(row)):
                row[j] += Float64(shift)
            moved.append(row^)
        var W = unsigned_volume(real_array_from_rows(moved))
        assert_true(_close(V, W), String(V, " != ", W))

    given(
        composite(_simplex_points),
        integers(-100, 100),
        integers(0, 3),
        prop,
        quick(),
    )


def test_volume_is_invariant_under_embedding() raises:
    # Embedding an M-simplex in more dimensions (constant extra coordinates,
    # coordinates permuted) must not change its M-volume. In N > M dimensions
    # unsigned_volume uses the Gram determinant, so flat simplices are the
    # interesting inputs.
    def prop(t: Tuple[List[List[Float64]], List[List[Float64]]]) raises:
        var V = unsigned_volume(real_array_from_rows(t[0]))
        var W = unsigned_volume(real_array_from_rows(t[1]))
        assert_true(_close(V, W), String(V, " != ", W))

    given(composite(_embedded_simplex), prop, quick())


def test_signed_volume_changes_sign_with_odd_reorderings() raises:
    def prop(pts: List[List[Float64]], seed: List[Int]) raises:
        var n = len(pts)
        # a permutation of [0, n) from the seed's ordering
        var perm = List[Int]()
        for i in range(n):
            perm.append(i)
        for i in range(min(len(seed), n - 1)):
            var j = i + seed[i] % (n - i)
            perm.swap_elements(i, j)
        var permuted = List[List[Float64]]()
        for i in range(n):
            permuted.append(pts[perm[i]].copy())
        var v = signed_volume(real_array_from_rows(pts))
        var w = signed_volume(real_array_from_rows(permuted))
        var sign = -1.0 if permutation_parity(perm) == 1 else 1.0
        assert_true(_close(w, sign * v), String(w, " != ", sign, " * ", v))

    given(
        composite(_simplex_points),
        lists(integers(0, 3), min_size=3, max_size=3),
        prop,
        quick(),
    )


def test_circumcenter_is_equidistant_from_the_vertices() raises:
    def prop(pts: List[List[Float64]]) raises:
        var P = real_array_from_rows(pts)
        assume(unsigned_volume(P) > 1e-6)  # skip degenerate simplices
        var s = circumcenter(P)
        for i in range(len(pts)):
            var d2 = 0.0
            for j in range(len(pts[i])):
                var x = pts[i][j] - s.center.unsafe_get(j)
                d2 += x * x
            assert_true(
                _close(d2, s.radius * s.radius, 1e-6),
                String("vertex ", i, " is at distance^2 ", d2),
            )

    given(composite(_simplex_points), prop, quick())


def test_barycentric_gradients_sum_to_zero() raises:
    def prop(pts: List[List[Float64]]) raises:
        var P = real_array_from_rows(pts)
        assume(unsigned_volume(P) > 1e-6)
        var G = barycentric_gradients(P)
        var n = len(pts)
        var dim = len(pts[0])
        for j in range(dim):
            var s = 0.0
            for i in range(n):
                s += G.unsafe_get(i * dim + j)
            assert_true(abs(s) < 1e-9, String("component ", j, " sums to ", s))

    given(composite(_simplex_points), prop, quick())


def _int_det(a: List[List[Int]]) -> Int:
    """The determinant of a small integer matrix, exactly (cofactors)."""
    var n = len(a)
    if n == 1:
        return a[0][0]
    var d = 0
    for j in range(n):
        var minor = List[List[Int]]()
        for i in range(1, n):
            var row = List[Int]()
            for k in range(n):
                if k != j:
                    row.append(a[i][k])
            minor.append(row^)
        var t = a[0][j] * _int_det(minor)
        d += t if j % 2 == 0 else -t
    return d


def _thin_simplex(mut data: ConjectureData) raises -> List[List[Float64]]:
    """An M-simplex in N-space (1 <= M <= 3, M <= N <= M + 2) whose last
    vertex is an integer combination of the others plus 10^-j times an
    integer offset (0 <= j <= 8), so it may be very thin, but not flat.

    Flat draws (a flat base, or an offset in the base's span) are rejected
    exactly, by the integer Gram determinant of the base's edges and the
    offset: in floating point their volumes are rounding noise, not 0, and
    `_thinness` would be noise over noise (it came out as low as 8.6)."""
    var m = data.draw(integers(1, 3), "M")
    var n = m + data.draw(integers(0, 2), "N - M")
    var pts = List[List[Float64]]()
    for _ in range(m):
        var row = List[Float64]()
        for _ in range(n):
            row.append(Float64(data.draw(integers(-5, 5))))
        pts.append(row^)
    var scale = 1.0
    for _ in range(data.draw(integers(0, 8), "j")):
        scale /= 10
    var last = pts[0].copy()
    for i in range(1, m):
        var c = Float64(data.draw(integers(-2, 2), "coefficient"))
        for k in range(n):
            last[k] += c * (pts[i][k] - pts[0][k])
    var vecs = List[List[Int]]()
    for i in range(1, m):
        var v = List[Int]()
        for k in range(n):
            v.append(Int(pts[i][k] - pts[0][k]))
        vecs.append(v^)
    var offset = List[Int]()
    for k in range(n):
        var o = data.draw(integers(-3, 3), "offset")
        offset.append(o)
        last[k] += scale * Float64(o)
    vecs.append(offset^)
    var gram = List[List[Int]]()
    for i in range(m):
        var row = List[Int]()
        for j in range(m):
            var g = 0
            for k in range(n):
                g += vecs[i][k] * vecs[j][k]
            row.append(g)
        gram.append(row^)
    assume(_int_det(gram) != 0)
    pts.append(last^)
    return pts^


def _thinness(pts: List[List[Float64]]) raises -> Float64:
    """`L / h` for a simplex: its longest edge over the height of its last
    vertex above the others (0 if it is flat)."""
    var m = len(pts) - 1
    var n = len(pts[0])
    var base = List[List[Float64]]()
    for i in range(m):
        base.append(pts[i].copy())
    var vol = unsigned_volume(real_array_from_rows(pts))
    var base_vol = unsigned_volume(real_array_from_rows(base))
    if vol <= 0 or base_vol <= 0:
        return 0
    var h = Float64(m) * vol / base_vol
    var L = 0.0
    for i in range(m + 1):
        for j in range(i):
            var d2 = 0.0
            for k in range(n):
                var t = pts[i][k] - pts[j][k]
                d2 += t * t
            L = max(L, sqrt(d2))
    return L / h


def test_thin_simplex_gradients_are_dual_to_the_edges() raises:
    # grad lambda_i . (x_j - x_0) = delta_ij for i, j >= 1, to within the
    # problem's own conditioning eps L / h; the Gram-matrix solve this
    # replaced was off by eps (L / h)^2 (and by 3.6 at h = 1e-8)
    def prop(pts: List[List[Float64]]) raises:
        var m = len(pts) - 1
        var n = len(pts[0])
        var P = real_array_from_rows(pts)
        var kappa = _thinness(pts)
        assume(kappa > 0)
        var tol = 64 * 2.220446049250313e-16 * kappa
        var G = barycentric_gradients(P)
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                var s = 0.0
                for k in range(n):
                    s += G.unsafe_get(i * n + k) * (pts[j][k] - pts[0][k])
                var want = 1.0 if i == j else 0.0
                if abs(s - want) > tol:
                    fail(
                        String(
                            "L/h=",
                            kappa,
                            ": grad ",
                            i,
                            " . edge ",
                            j,
                            " = ",
                            s,
                        )
                    )

    given(composite(_thin_simplex), prop, quick())


def test_thin_simplex_top_form_mass_is_inverse_volume() raises:
    # The Whitney mass matrix of an M-simplex's top form is 1 / volume, a
    # product of k x k minors of its gradients, to within eps L / h; from
    # the Gram matrix of the gradients these lost (L / h)^2 to cancellation
    def prop(pts: List[List[Float64]]) raises:
        var m = len(pts) - 1
        var P = real_array_from_rows(pts)
        var kappa = _thinness(pts)
        assume(kappa > 0)
        var T = int_array(1, m + 1)
        for i in range(m + 1):
            T.unsafe_set(i, Scalar[DType.int](i))
        var sc: SimplicialComplex
        try:
            sc = SimplicialComplex(P, T)
        except e:
            # the circumcenter of a tetrahedron this thin in 4 or 5
            # dimensions can be beyond the general solver (about 1 in 3000
            # with L / h from 1e8 to 1e10; in 3D the closed form has it);
            # the mass matrix does not need it, but the complex does
            assume("circumcenter: degenerate simplex" not in String(e))
            raise e^
        var M = whitney_innerproduct(sc, m).todense().unsafe_get(0)
        var got = M * abs(sc[m].primal_volume.unsafe_get(0))
        if abs(got - 1) > 64 * 2.220446049250313e-16 * kappa:
            fail(String("L/h=", kappa, ": M * volume = ", got))

    given(composite(_thin_simplex), prop, quick())


# ===----------------------------------------------------------------------=== #
# Sparse matrices, against a dense model
# ===----------------------------------------------------------------------=== #


def test_from_coo_sums_duplicates_like_a_dense_matrix() raises:
    def prop(t: Tuple[Int, Int, List[Int], List[Int], List[Float64]]) raises:
        var A = CSRMatrix.from_coo(t[0], t[1], t[2], t[3], t[4])
        var D = _dense(t[0], t[1], t[2], t[3], t[4])
        for i in range(t[0]):
            for j in range(t[1]):
                assert_equal(A[i, j], D[i][j], String("entry ", i, ", ", j))

    given(composite(_sparse_triplets), prop, quick())


def _same_csr(A: CSRMatrix, B: CSRMatrix, what: String) raises:
    assert_equal(A.nrows, B.nrows, what)
    assert_equal(A.ncols, B.ncols, what)
    assert_true(A.indptr == B.indptr, what + ": indptr")
    assert_true(A.indices == B.indices, what + ": indices")
    assert_true(A.data == B.data, what + ": data")


def test_boundary_by_buckets_matches_lexsort() raises:
    # simplex_array_boundary's parallel path (one bucket per first vertex)
    # gives the serial path's faces and matrices, entry for entry
    def prop(t: Tuple[Int, List[Int], List[Int]]) raises:
        var K = t[0]
        var N = len(t[2])
        var s = int_array(N, K)
        for i in range(N * K):
            s.unsafe_set(i, Scalar[DType.int](t[1][i]))
        var parity = int_vector(N)
        for i in range(N):
            parity.unsafe_set(i, Scalar[DType.int](t[2][i]))
        var y = _boundary_by_lexsort(s, parity)
        # packed integer keys, and the comparator they replace
        for pack in [True, False]:
            var x = _boundary_by_buckets(s, parity, _vertex_bound(s), pack)
            assert_equal(x.faces.size, y.faces.size)
            for i in range(x.faces.size):
                assert_equal(x.faces.unsafe_get(i), y.faces.unsafe_get(i))
            _same_csr(x.coboundary, y.coboundary, "d")
            _same_csr(x.boundary, y.boundary, "boundary")

    given(composite(_simplex_rows), prop, quick())


def test_lexsort_rows_is_stable_lexicographic() raises:
    # the counting sort by the first column, with its runs sorted by
    # insertion or (past 32 rows) by merging, and the merge sort for wide
    # ranges give the stable order; _lexsort_rows also finds the first
    # position whose row equals the one before
    def prop(t: Tuple[Int, List[Int]]) raises:
        var K = t[0]
        var N = len(t[1]) // K
        var a = int_array(N, K)
        for i in range(N * K):
            a.unsafe_set(i, Scalar[DType.int](t[1][i]))
        var expected = _stable_row_order(a)
        assert_true(lexsort_rows(a) == expected)
        var duplicate = 0
        assert_true(_lexsort_rows(a, duplicate) == expected)
        if duplicate == -2:
            return
        var first = -1
        for i in range(1, N):
            if compare_rows(a, expected[i - 1], a, expected[i]) == 0:
                first = i
                break
        assert_equal(duplicate, first)

    given(composite(_int_rows), prop, quick())


def test_require_distinct_simplices_finds_duplicates() raises:
    # raises exactly when two rows are equal, whether or not the rows are
    # in increasing order (the shortcut that skips sorting)
    def prop(t: Tuple[Int, List[Int], List[Int]]) raises:
        var K = t[0]
        var N = len(t[2])
        var s = int_array(N, K)
        for i in range(N * K):
            s.unsafe_set(i, Scalar[DType.int](t[1][i]))
        var duplicate = False
        for a in range(N):
            for b in range(a + 1, N):
                var same = True
                for j in range(K):
                    if t[1][a * K + j] != t[1][b * K + j]:
                        same = False
                if same:
                    duplicate = True
        var raised = False
        var message = String()
        try:
            require_distinct_simplices(s, "prop")
        except e:
            raised = True
            message = String(e)
        assert_equal(raised, duplicate)
        if raised:
            # the first equal neighbors in stable sorted order
            var order = _stable_row_order(s)
            for i in range(1, N):
                if compare_rows(s, order[i - 1], s, order[i]) == 0:
                    var a = min(order[i - 1], order[i])
                    var b = max(order[i - 1], order[i])
                    assert_true(
                        String("simplices ", a, " and ", b) in message,
                        message,
                    )
                    break

    given(composite(_simplex_rows), prop, quick())


def test_transpose_is_an_involution() raises:
    def prop(t: Tuple[Int, Int, List[Int], List[Int], List[Float64]]) raises:
        var A = CSRMatrix.from_coo(t[0], t[1], t[2], t[3], t[4])
        _assert_zero(A.T().T() - A, "A.T().T() - A")

    given(composite(_sparse_triplets), prop, quick())


def test_transpose_of_a_product() raises:
    def prop(
        a: Tuple[Int, Int, List[Int], List[Int], List[Float64]],
        b: Tuple[Int, Int, List[Int], List[Int], List[Float64]],
    ) raises:
        # reshape b so that A @ B is defined (rows of B = columns of A)
        var brows = List[Int]()
        for r in b[2]:
            brows.append(r % a[1])
        var A = CSRMatrix.from_coo(a[0], a[1], a[2], a[3], a[4])
        var B = CSRMatrix.from_coo(a[1], b[1], brows, b[3], b[4])
        _assert_zero((A @ B).T() - B.T() @ A.T(), "(A B)^T - B^T A^T")

    given(
        composite(_sparse_triplets), composite(_sparse_triplets), prop, quick()
    )


def test_matrix_vector_products_associate() raises:
    def prop(
        a: Tuple[Int, Int, List[Int], List[Int], List[Float64]],
        b: Tuple[Int, Int, List[Int], List[Int], List[Float64]],
        xs: List[Int],
    ) raises:
        var brows = List[Int]()
        for r in b[2]:
            brows.append(r % a[1])
        var A = CSRMatrix.from_coo(a[0], a[1], a[2], a[3], a[4])
        var B = CSRMatrix.from_coo(a[1], b[1], brows, b[3], b[4])
        var x = List[Float64]()
        for i in range(b[1]):
            x.append(Float64(xs[i % len(xs)]))
        var xv = real_vector_from_list(x)
        var left = A @ (B @ xv)
        var right = (A @ B) @ xv
        for i in range(a[0]):
            # integer data: the results are exact
            assert_equal(left.unsafe_get(i), right.unsafe_get(i))

    given(
        composite(_sparse_triplets),
        composite(_sparse_triplets),
        lists(integers(-9, 9), min_size=1, max_size=6),
        prop,
        quick(),
    )


comptime Triplets = Tuple[Int, Int, List[Int], List[Int], List[Float64]]


def _csr(t: Triplets) raises -> CSRMatrix:
    return CSRMatrix.from_coo(t[0], t[1], t[2], t[3], t[4])


def _reshaped(t: Triplets, nrows: Int, ncols: Int) raises -> CSRMatrix:
    """The triplets of `t` folded into an `nrows x ncols` matrix."""
    var rows = List[Int]()
    var cols = List[Int]()
    for r in t[2]:
        rows.append(r % nrows)
    for c in t[3]:
        cols.append(c % ncols)
    return CSRMatrix.from_coo(nrows, ncols, rows, cols, t[4])


def _assert_matches(A: CSRMatrix, D: List[List[Float64]], what: String) raises:
    """A equals the dense matrix D (integer data, so exactly)."""
    assert_equal(A.nrows, len(D), String(what, ": rows"))
    for i in range(A.nrows):
        assert_equal(A.ncols, len(D[i]), String(what, ": columns"))
        for j in range(A.ncols):
            if A[i, j] != D[i][j]:
                fail(
                    String(
                        what,
                        "[",
                        i,
                        ", ",
                        j,
                        "] = ",
                        A[i, j],
                        ", not ",
                        D[i][j],
                    )
                )


def _dense_of(A: CSRMatrix) raises -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for i in range(A.nrows):
        var row = List[Float64]()
        for j in range(A.ncols):
            row.append(A[i, j])
        out.append(row^)
    return out^


def test_sums_and_scalings_match_dense() raises:
    def prop(a: Triplets, b: Triplets, s: Int) raises:
        var A = _csr(a)
        var B = _reshaped(b, a[0], a[1])
        var DA = _dense_of(A)
        var DB = _dense_of(B)
        var sum = List[List[Float64]]()
        var diff = List[List[Float64]]()
        var scaled = List[List[Float64]]()
        var neg = List[List[Float64]]()
        for i in range(a[0]):
            sum.append(List[Float64]())
            diff.append(List[Float64]())
            scaled.append(List[Float64]())
            neg.append(List[Float64]())
            for j in range(a[1]):
                sum[i].append(DA[i][j] + DB[i][j])
                diff[i].append(DA[i][j] - DB[i][j])
                scaled[i].append(DA[i][j] * Float64(s))
                neg[i].append(-DA[i][j])
        _assert_matches(A + B, sum, "A + B")
        _assert_matches(A - B, diff, "A - B")
        _assert_matches(A * Float64(s), scaled, "A * s")
        _assert_matches(Float64(s) * A, scaled, "s * A")
        _assert_matches(-A, neg, "-A")

    given(
        composite(_sparse_triplets),
        composite(_sparse_triplets),
        integers(-5, 5),
        prop,
        quick(),
    )


def test_products_match_dense() raises:
    def prop(a: Triplets, b: Triplets, xs: List[Int]) raises:
        var A = _csr(a)
        var B = _reshaped(b, a[1], b[1])
        var DA = _dense_of(A)
        var DB = _dense_of(B)
        var prod = List[List[Float64]]()
        for i in range(a[0]):
            var row = List[Float64](length=b[1], fill=0.0)
            for k in range(a[1]):
                for j in range(b[1]):
                    row[j] += DA[i][k] * DB[k][j]
            prod.append(row^)
        _assert_matches(A @ B, prod, "A @ B")
        var x = List[Float64]()
        for j in range(a[1]):
            x.append(Float64(xs[j % len(xs)]))
        var y = A @ real_vector_from_list(x)
        for i in range(a[0]):
            var s = 0.0
            for j in range(a[1]):
                s += DA[i][j] * x[j]
            assert_equal(y.unsafe_get(i), s, String("(A x)[", i, "]"))

    given(
        composite(_sparse_triplets),
        composite(_sparse_triplets),
        lists(integers(-9, 9), min_size=1, max_size=6),
        prop,
        quick(),
    )


def test_select_matches_dense() raises:
    def prop(a: Triplets, row_picks: List[Int], col_picks: List[Int]) raises:
        var A = _csr(a)
        var D = _dense_of(A)
        # rows may repeat; columns must be distinct
        var rows = List[Int]()
        for r in row_picks:
            rows.append(r % a[0])
        var cols = List[Int]()
        for c in col_picks:
            var j = c % a[1]
            if j not in cols:
                cols.append(j)
        var expected = List[List[Float64]]()
        for i in rows:
            var row = List[Float64]()
            for j in cols:
                row.append(D[i][j])
            expected.append(row^)
        _assert_matches(A.select(rows, cols), expected, "A.select")

    given(
        composite(_sparse_triplets),
        lists(integers(0, 20), max_size=6),
        lists(integers(0, 20), max_size=6),
        prop,
        quick(),
    )


def test_bmat_matches_dense() raises:
    def prop(a: Triplets, b: Triplets, c: Triplets, zero: Int) raises:
        # [[A, B], [C, Z]] with A: r1 x c1, B: r1 x c2, C: r2 x c1, and
        # one of the four blocks (chosen by `zero`) left out
        var r1 = a[0]
        var c1 = a[1]
        var c2 = b[1]
        var r2 = c[0]
        var blocks = List[CSRMatrix]()
        blocks.append(_csr(a))
        blocks.append(_reshaped(b, r1, c2))
        blocks.append(_reshaped(c, r2, c1))
        blocks.append(_reshaped(a, r2, c2))
        var grid = List[List[Optional[CSRMatrix]]]()
        for bi in range(2):
            var row = List[Optional[CSRMatrix]]()
            for bj in range(2):
                var k = 2 * bi + bj
                if k == zero:
                    row.append(None)
                else:
                    row.append(blocks[k].copy())
            grid.append(row^)
        var expected = List[List[Float64]]()
        for i in range(r1 + r2):
            var row = List[Float64]()
            for j in range(c1 + c2):
                var bi = 0 if i < r1 else 1
                var bj = 0 if j < c1 else 1
                var k = 2 * bi + bj
                var ii = i if bi == 0 else i - r1
                var jj = j if bj == 0 else j - c1
                row.append(0.0 if k == zero else blocks[k][ii, jj])
            expected.append(row^)
        _assert_matches(CSRMatrix.bmat(grid), expected, "bmat")

    given(
        composite(_sparse_triplets),
        composite(_sparse_triplets),
        composite(_sparse_triplets),
        integers(0, 3),
        prop,
        quick(),
    )


def test_dense_and_coo_round_trips() raises:
    def prop(a: Triplets) raises:
        var A = _csr(a)
        _assert_zero(
            CSRMatrix.from_dense(A.todense()) - A, "from_dense(todense)"
        )
        var coo = A.to_coo()
        _assert_zero(
            CSRMatrix.from_coo(A.nrows, A.ncols, coo.rows, coo.cols, coo.values)
            - A,
            "from_coo(to_coo)",
        )
        var diag = A.diagonal()
        assert_equal(diag.size, min(A.nrows, A.ncols))
        for i in range(diag.size):
            assert_equal(diag.unsafe_get(i), A[i, i])
        _assert_zero(CSRMatrix.identity(A.nrows) @ A - A, "I @ A - A")

    given(composite(_sparse_triplets), prop, quick())


# ===----------------------------------------------------------------------=== #
# Complexes
# ===----------------------------------------------------------------------=== #


def _check_chain_complex(sc: SimplicialComplex) raises:
    """d d = 0, boundary boundary = 0, and boundary = d^T at every level."""
    var n = sc.complex_dimension()
    for k in range(n):
        _assert_zero(
            sc[k].d.T() - sc[k + 1].boundary,
            String("d", k, "^T - boundary", k + 1),
        )
    for k in range(n - 1):
        _assert_zero(sc[k + 1].d @ sc[k].d, String("d", k + 1, " d", k))
        _assert_zero(
            sc[k + 1].boundary @ sc[k + 2].boundary,
            String("boundary", k + 1, " boundary", k + 2),
        )


def test_grids_form_a_chain_complex() raises:
    def prop(dims: List[Int]) raises:
        _check_chain_complex(_grid_complex(dims))

    given(composite(_grid_dims), prop, meshes())


def test_icospheres_form_a_chain_complex() raises:
    def prop(n: Int) raises:
        _check_chain_complex(SimplicialComplex(icosphere(n)))

    given(integers(0, 2), prop, meshes())


def test_euler_characteristics() raises:
    def grid(dims: List[Int]) raises:
        assert_equal(_grid_complex(dims).euler_characteristic(), 1)

    given(composite(_grid_dims), grid, meshes())

    def sphere(n: Int) raises:
        var mesh = icosphere(n)
        assert_equal(SimplicialComplex(mesh.copy()).euler_characteristic(), 2)
        # midpoint subdivision doesn't change the topology
        assert_equal(
            SimplicialComplex(loop_subdivision(mesh)).euler_characteristic(), 2
        )

    given(integers(0, 2), sphere, meshes())


def test_simplex_index_round_trip() raises:
    def prop(dims: List[Int], picks: List[Int], flip: Bool) raises:
        var sc = _grid_complex(dims)
        for p in picks:
            var k = p % (sc.complex_dimension() + 1)
            var i = p % sc[k].num_simplices
            var s = sc[k].index_to_simplex(i)
            assert_equal(sc[k].simplex_to_index(s), i)
            # the index ignores orientation
            var v = s.vertices.copy()
            if flip and len(v) >= 2:
                v.swap_elements(0, 1)
            assert_equal(sc[k].simplex_to_index(Simplex(v)), i)

    given(
        composite(_grid_dims),
        lists(integers(0, 1000), min_size=1, max_size=5),
        booleans(),
        prop,
        meshes(),
    )


def test_hodge_star_inverse_on_icospheres() raises:
    # `star_inv` is signed so that star_inv @ star = (-1)^(k(n-k)) (as in
    # PyDEC); icospheres are well-centered, so no star entry is zero
    def prop(n: Int) raises:
        var sc = SimplicialComplex(icosphere(n))
        var dim = sc.complex_dimension()
        for k in range(dim + 1):
            var sign = -1.0 if (k * (dim - k)) % 2 == 1 else 1.0
            for i in range(sc[k].num_simplices):
                var p = sc[k].star[i] * sc[k].star_inv[i]
                assert_true(_close(p, sign), String("k=", k, " i=", i, ": ", p))

    given(integers(0, 2), prop, meshes())


def test_closure_and_star_laws() raises:
    def prop(n: Int, picks: List[Int]) raises:
        var sc = SimplicialComplex(icosphere(n))
        var cc = sc.chain_complex()
        # a random set of cells: CellSet.from_lists takes the indices of the
        # k-cells in its k-th list
        var by_dim = List[List[Int]]()
        for _ in range(sc.complex_dimension() + 1):
            by_dim.append(List[Int]())
        for p in picks:
            var k = p % len(by_dim)
            var i = (p // len(by_dim)) % sc[k].num_simplices
            var seen = False
            for j in by_dim[k]:
                if j == i:
                    seen = True
            if not seen:
                by_dim[k].append(i)
        var S = CellSet.from_lists(sc, by_dim)
        var cl = closure_of(cc, S)
        var st = star_of(cc, S)
        assert_true((S - cl).is_empty(), "S is not in its closure")
        assert_true((S - st).is_empty(), "S is not in its star")
        assert_true(closure_of(cc, cl) == cl, "closure is not idempotent")
        assert_true(star_of(cc, st) == st, "star is not idempotent")

    given(
        integers(0, 1),
        lists(integers(0, 10000), min_size=1, max_size=6),
        prop,
        meshes(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
