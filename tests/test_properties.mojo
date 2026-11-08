# Property-based tests, using proptest (the sibling ../proptest repository,
# a port of Python's Hypothesis). Each test states a law that must hold for
# every input in a family, and proptest searches that family for a
# counterexample, shrinking any it finds to a minimal one.
#
# proptest must be on the include path: tests/run_all.sh passes
# `-I $PROPTEST_PATH` (default ../proptest). To run just this file:
#
#   pixi run mojo run -I . -I build -I ../proptest tests/test_properties.mojo

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
from mojodec.core.utils import real_array_from_rows, real_vector_from_list
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
from mojodec.routines.mesh_generation import (
    cube_grid,
    icosphere,
    loop_subdivision,
    triangulate_ncube,
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
