# Property-based tests of MojoDEC's geometry and operators on whole meshes,
# using proptest (see tests/test_properties.mojo for the setup). To run just
# this file:
#
#   pixi run mojo run -I . -I build -I ../proptest tests/test_properties_dec.mojo
#
# The meshes come from `MeshSpec`, which names a family (jittered 1-D and
# 2-D grids, icospheres, the fixtures in tests/meshes) and its parameters, so
# a counterexample prints as a recipe rather than as arrays.

from std.math import cos, pi, sin
from std.testing import TestSuite, assert_equal, assert_true

from proptest import (
    ConjectureData,
    Settings,
    booleans,
    composite,
    fail,
    floats,
    given,
    integers,
    lists,
    permutations,
)

from mojodec.core.cochain import Cochain, d
from mojodec.core.mesh import SimplicialMesh
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    RealArray,
    cols_of,
    fget,
    fset,
    iget,
    iset,
    real_array,
    real_vector,
    rows_of,
)
from mojodec.routines.innerproduct import whitney_innerproduct
from mojodec.routines.io.arrayio import (
    ArrayFormat,
    SparseFormat,
    read_array,
    read_sparse,
    write_array,
    write_sparse,
)
from mojodec.routines.io.meshio import read_mesh, write_mesh
from mojodec.routines.io.ply import PlyFormat, read_ply, write_ply
from mojodec.routines.mesh_generation import (
    cube_grid,
    icosphere,
    triangulate_ncube,
)
from mojodec.routines.musical import flat
from mojodec.routines.wedge import wedge


# ===----------------------------------------------------------------------=== #
# Meshes
# ===----------------------------------------------------------------------=== #


comptime GRID = 0
comptime ICOSPHERE = 1
comptime FIXTURE = 2


def _fixture(i: Int) -> String:
    if i == 0:
        return "tests/meshes/sphere3.xml"  # 128 triangles in 3-D
    if i == 1:
        return "tests/meshes/cube.xml"  # the 12 triangles of a cube's surface
    return "tests/meshes/torus_tet.xml"  # 592 tetrahedra


@fieldwise_init
struct MeshSpec(Copyable, Writable):
    """A recipe for a test mesh."""

    var kind: Int
    """GRID, ICOSPHERE, or FIXTURE."""
    var dims: List[Int]
    """GRID: cells per axis (1 or 2 axes); ICOSPHERE: [subdivisions];
    FIXTURE: [fixture index]."""
    var jitter: Int
    """GRID: seed for moving each vertex by up to 1 per coordinate (the
    cells have side 8, so no triangle degenerates); 0 for none."""

    def write_to(self, mut writer: Some[Writer]):
        if self.kind == GRID:
            writer.write("grid ", self.dims, " jitter ", self.jitter)
        elif self.kind == ICOSPHERE:
            writer.write("icosphere(", self.dims[0], ")")
        else:
            writer.write(_fixture(self.dims[0]))

    def mesh(self) raises -> SimplicialMesh:
        if self.kind == ICOSPHERE:
            return icosphere(self.dims[0])
        if self.kind == FIXTURE:
            return read_mesh(_fixture(self.dims[0]))
        var m = triangulate_ncube(cube_grid(self.dims))
        var V = m.vertices.copy()
        var state = UInt64(self.jitter)
        for i in range(rows_of(V)):
            for j in range(cols_of(V)):
                var offset = 0.0
                if self.jitter != 0:
                    state = state * 6364136223846793005 + 1442695040888963407
                    offset = Float64(Int((state >> 33) % 3)) - 1.0
                fset(V, i, j, 8.0 * fget(V, i, j) + offset)
        return SimplicialMesh(V, m.elements)


def _grid_spec(mut data: ConjectureData) raises -> MeshSpec:
    var dims = data.draw(lists(integers(1, 4), min_size=1, max_size=2), "dims")
    return MeshSpec(GRID, dims^, data.draw(integers(0, 1000), "jitter"))


def _any_mesh(mut data: ConjectureData) raises -> MeshSpec:
    """Any test mesh: a grid, an icosphere, or a fixture."""
    var kind = data.draw(integers(0, 2), "kind")
    if kind == ICOSPHERE:
        return MeshSpec(ICOSPHERE, [data.draw(integers(0, 1))], 0)
    if kind == FIXTURE:
        return MeshSpec(FIXTURE, [data.draw(integers(0, 2))], 0)
    return _grid_spec(data)


def _flat_mesh(mut data: ConjectureData) raises -> MeshSpec:
    """A mesh whose dimension equals its embedding dimension: a grid or the
    tetrahedral torus."""
    if data.draw(integers(0, 3), "torus") == 3:
        return MeshSpec(FIXTURE, [2], 0)
    return _grid_spec(data)


def meshes() -> Settings:
    return Settings(max_examples=25, deadline=None)


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _max_abs(a: RealArray) -> Float64:
    var m = 0.0
    for i in range(a.size):
        m = max(m, abs(a.unsafe_get(i)))
    return m


def _assert_close_arrays(
    a: RealArray, b: RealArray, what: String, tol: Float64 = 1e-9
) raises:
    """Entrywise agreement to `tol` times the largest entry."""
    assert_equal(a.size, b.size, what)
    var scale = max(1e-300, max(_max_abs(a), _max_abs(b)))
    for i in range(a.size):
        var x = a.unsafe_get(i)
        var y = b.unsafe_get(i)
        if abs(x - y) > tol * scale:
            fail(String(what, "[", i, "]: ", x, " != ", y))


def _star_values(sc: SimplicialComplex, k: Int) raises -> RealArray:
    var out = real_vector(sc[k].num_simplices)
    for i in range(out.size):
        out.unsafe_set(i, sc[k].star[i])
    return out^


def _seeded_values(n: Int, seed: Int) raises -> RealArray:
    """n small integers in [-3, 3] from a seed (cochains are too large to
    draw entry by entry)."""
    var out = real_vector(n)
    var state = UInt64(seed) * 2654435761 + 1
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        out.unsafe_set(i, Float64(Int((state >> 33) % 7)) - 3.0)
    return out^


def _cochain(sc: SimplicialComplex, k: Int, seed: Int) raises -> Cochain:
    var c = sc.get_cochain(k)
    c.v = _seeded_values(c.v.size, seed)
    return c^


def _quadratic_form(M: CSRMatrix, x: RealArray) raises -> Float64:
    var Mx = M @ x
    var s = 0.0
    for i in range(x.size):
        s += x.unsafe_get(i) * Mx.unsafe_get(i)
    return s


def _total_volume(sc: SimplicialComplex) raises -> Float64:
    var n = sc.complex_dimension()
    var s = 0.0
    for i in range(sc[n].num_simplices):
        s += abs(sc[n].primal_volume.unsafe_get(i))
    return s


def _binomial(n: Int, k: Int) -> Int:
    var r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def _rotation(E: Int, a: Int, b: Int) -> List[List[Float64]]:
    """A proper rotation of E-space: by `a` degrees in the first coordinate
    plane, then (in 3-D) by `b` degrees in the last."""
    var R = List[List[Float64]]()
    for i in range(E):
        var row = List[Float64](length=E, fill=0.0)
        row[i] = 1.0
        R.append(row^)
    if E < 2:
        return R^
    var t = Float64(a) * pi / 180
    R[0][0] = cos(t)
    R[0][1] = -sin(t)
    R[1][0] = sin(t)
    R[1][1] = cos(t)
    if E == 3:
        var u = Float64(b) * pi / 180
        var c = cos(u)
        var s = sin(u)
        var Q = List[List[Float64]]()
        Q.append([1.0, 0.0, 0.0])
        Q.append([0.0, c, -s])
        Q.append([0.0, s, c])
        var QR = List[List[Float64]]()
        for i in range(3):
            var row = List[Float64](length=3, fill=0.0)
            for j in range(3):
                for k in range(3):
                    row[j] += Q[i][k] * R[k][j]
            QR.append(row^)
        return QR^
    return R^


# ===----------------------------------------------------------------------=== #
# Invariance
# ===----------------------------------------------------------------------=== #


def test_rigid_motions_preserve_volumes_and_stars() raises:
    def prop(spec: MeshSpec, a: Int, b: Int, shift: Int) raises:
        var mesh = spec.mesh()
        var sc = SimplicialComplex(mesh.copy())
        var E = cols_of(mesh.vertices)
        var R = _rotation(E, a, b)
        var V = mesh.vertices.copy()
        for i in range(rows_of(V)):
            for j in range(E):
                var x = Float64(shift)
                for k in range(E):
                    x += R[j][k] * fget(mesh.vertices, i, k)
                fset(V, i, j, x)
        var moved = SimplicialComplex(V, mesh.elements)
        for k in range(sc.complex_dimension() + 1):
            _assert_close_arrays(
                sc[k].primal_volume,
                moved[k].primal_volume,
                String("primal_volume ", k),
            )
            _assert_close_arrays(
                sc[k].dual_volume,
                moved[k].dual_volume,
                String("dual_volume ", k),
            )
            _assert_close_arrays(
                _star_values(sc, k), _star_values(moved, k), String("star ", k)
            )

    given(
        composite(_any_mesh),
        integers(0, 359),
        integers(0, 359),
        integers(-50, 50),
        prop,
        meshes(),
    )


def test_relabelling_vertices_preserves_volumes_and_stars() raises:
    def prop(spec: MeshSpec, seed: Int) raises:
        var mesh = spec.mesh()
        var nv = rows_of(mesh.vertices)
        # a permutation of the vertices from the seed (Fisher-Yates)
        var perm = List[Int]()
        for i in range(nv):
            perm.append(i)
        var state = UInt64(seed) + 1
        for i in range(nv - 1, 0, -1):
            state = state * 6364136223846793005 + 1442695040888963407
            perm.swap_elements(i, Int((state >> 33) % UInt64(i + 1)))
        # vertex v becomes perm[v]
        var V = mesh.vertices.copy()
        for v in range(nv):
            for j in range(cols_of(V)):
                fset(V, perm[v], j, fget(mesh.vertices, v, j))
        var T = mesh.elements.copy()
        for t in range(rows_of(T)):
            for j in range(cols_of(T)):
                iset(T, t, j, perm[iget(mesh.elements, t, j)])
        var sc = SimplicialComplex(mesh.copy())
        var relabelled = SimplicialComplex(V, T)
        assert_equal(
            sc.euler_characteristic(), relabelled.euler_characteristic()
        )
        for k in range(sc.complex_dimension() + 1):
            var n = sc[k].num_simplices
            assert_equal(n, relabelled[k].num_simplices)
            # the same quantities, gathered in the original order
            var primal = real_vector(n)
            var dual = real_vector(n)
            var star = real_vector(n)
            for i in range(n):
                var s = sc[k].index_to_simplex(i)
                var w = List[Int]()
                for v in s.vertices:
                    w.append(perm[v])
                var j = relabelled[k].simplex_to_index(Simplex(w))
                # top simplices may change orientation
                primal.unsafe_set(
                    i, abs(relabelled[k].primal_volume.unsafe_get(j))
                )
                dual.unsafe_set(i, relabelled[k].dual_volume.unsafe_get(j))
                star.unsafe_set(i, relabelled[k].star[j])
            var p0 = real_vector(n)
            for i in range(n):
                p0.unsafe_set(i, abs(sc[k].primal_volume.unsafe_get(i)))
            _assert_close_arrays(p0, primal, String("primal_volume ", k))
            _assert_close_arrays(
                sc[k].dual_volume, dual, String("dual_volume ", k)
            )
            _assert_close_arrays(_star_values(sc, k), star, String("star ", k))

    given(composite(_any_mesh), integers(0, 10000), prop, meshes())


# ===----------------------------------------------------------------------=== #
# Hodge stars
# ===----------------------------------------------------------------------=== #


def test_primal_dual_volume_identity() raises:
    # A k-simplex and its circumcentric dual cell are orthogonal, and the
    # convex hulls of the pairs ("diamonds") tile the mesh (with signs, for
    # meshes that are not well centred). A diamond's volume is
    # |s| |*s| / C(n, k), so the sum of |s| |*s| over the k-simplices is
    # C(n, k) times the volume of the mesh. In terms of the stars, the sum
    # over k-simplices of |s|^2 star_k[s] is C(n, k) |M|.
    def prop(spec: MeshSpec) raises:
        var sc = SimplicialComplex(spec.mesh())
        var n = sc.complex_dimension()
        var total = _total_volume(sc)
        for k in range(n + 1):
            var s = 0.0
            for i in range(sc[k].num_simplices):
                s += abs(sc[k].primal_volume.unsafe_get(i)) * sc[
                    k
                ].dual_volume.unsafe_get(i)
            var expected = Float64(_binomial(n, k)) * total
            if abs(s - expected) > 1e-9 * expected:
                fail(String("k=", k, ": ", s, " != ", expected))

    given(composite(_any_mesh), prop, meshes())


# ===----------------------------------------------------------------------=== #
# Whitney mass matrices
# ===----------------------------------------------------------------------=== #


def test_whitney_mass_matrices_are_symmetric_positive_definite() raises:
    def prop(spec: MeshSpec, seed: Int) raises:
        var sc = SimplicialComplex(spec.mesh())
        for k in range(sc.complex_dimension() + 1):
            var M = whitney_innerproduct(sc, k)
            var asym = (M - M.T()).max_abs()
            if asym > 1e-12 * M.max_abs():
                fail(String("M", k, " is not symmetric: ", asym))
            var x = _seeded_values(M.ncols, seed)
            if _max_abs(x) == 0:
                continue
            var q = _quadratic_form(M, x)
            if not (q > 0):
                fail(String("x^T M", k, " x = ", q))

    given(composite(_any_mesh), integers(0, 10000), prop, meshes())


def test_whitney_forms_integrate_linear_functions_exactly() raises:
    # For f = a.x + c, Whitney 0-forms reproduce f, 1-forms reproduce df,
    # and n-forms the volume form, so these integrals are exact:
    #   f^T M0 f = int f^2, computed per simplex by the closed form
    #       |T| (sum f_i^2 + (sum f_i)^2) / ((n+1)(n+2)),
    #   df^T M1 df = |a|^2 |M|,   vol^T Mn vol = |M|.
    def prop(spec: MeshSpec, a: List[Int], c: Int) raises:
        var sc = SimplicialComplex(spec.mesh())
        var n = sc.complex_dimension()
        ref V = sc.vertices()
        var f = real_vector(sc[0].num_simplices)
        for v in range(f.size):
            var x = Float64(c)
            for j in range(n):
                x += Float64(a[j]) * fget(V, v, j)
            f.unsafe_set(v, x)
        ref top = sc[n]
        var expected = 0.0
        for t in range(top.num_simplices):
            var s = 0.0
            var s2 = 0.0
            for j in range(n + 1):
                var fv = f.unsafe_get(iget(top.simplices, t, j))
                s += fv
                s2 += fv * fv
            expected += (
                abs(top.primal_volume.unsafe_get(t))
                * (s2 + s * s)
                / Float64((n + 1) * (n + 2))
            )
        var got = _quadratic_form(whitney_innerproduct(sc, 0), f)
        if abs(got - expected) > 1e-9 * max(1.0, expected):
            fail(String("int f^2: ", got, " != ", expected))

        var total = _total_volume(sc)
        var a2 = 0.0
        for j in range(n):
            a2 += Float64(a[j] * a[j])
        var df = sc[0].d @ f
        got = _quadratic_form(whitney_innerproduct(sc, 1), df)
        if abs(got - a2 * total) > 1e-9 * max(1.0, a2 * total):
            fail(String("int |df|^2: ", got, " != ", a2 * total))

        var vol = top.primal_volume.copy()
        got = _quadratic_form(whitney_innerproduct(sc, n), vol)
        if abs(got - total) > 1e-9 * total:
            fail(String("int 1: ", got, " != ", total))

    given(
        composite(_flat_mesh),
        lists(integers(-3, 3), min_size=3, max_size=3),
        integers(-5, 5),
        prop,
        meshes(),
    )


# ===----------------------------------------------------------------------=== #
# Wedge and flat
# ===----------------------------------------------------------------------=== #


def test_wedge_is_graded_commutative_and_leibniz() raises:
    def prop(spec: MeshSpec, k_pick: Int, l_pick: Int, seed: Int) raises:
        var sc = SimplicialComplex(spec.mesh())
        var n = sc.complex_dimension()
        var k = k_pick % (n + 1)
        var l = l_pick % (n - k + 1)
        var a = _cochain(sc, k, seed)
        var b = _cochain(sc, l, seed + 1)
        var ab = wedge(a, b)
        var ba = wedge(b, a)
        var sign = -1.0 if (k * l) % 2 == 1 else 1.0
        _assert_close_arrays(
            ab.v,
            ba.scaled(sign).v,
            String("a^b vs b^a, k=", k, " l=", l),
            1e-12,
        )
        if k + l < n:
            var left = d(ab)
            var right = wedge(d(a), b) + wedge(a, d(b)).scaled(
                -1.0 if k % 2 == 1 else 1.0
            )
            _assert_close_arrays(
                left.v, right.v, String("Leibniz, k=", k, " l=", l), 1e-12
            )

    given(
        composite(_any_mesh),
        integers(0, 3),
        integers(0, 3),
        integers(0, 10000),
        prop,
        meshes(),
    )


def test_flat_of_a_constant_field_is_exact() raises:
    # The DPP flat of a constant vector field a is the 1-cochain of a.x:
    # its value on an edge [u, v] is a . (v - u).
    def prop(spec: MeshSpec, a: List[Int]) raises:
        var sc = SimplicialComplex(spec.mesh())
        var n = sc.complex_dimension()
        ref V = sc.vertices()
        var X = real_array(sc[n].num_simplices, n)
        for t in range(rows_of(X)):
            for j in range(n):
                fset(X, t, j, Float64(a[j]))
        var w = flat(sc, X)
        var expected = real_vector(sc[1].num_simplices)
        for e in range(expected.size):
            var u = iget(sc[1].simplices, e, 0)
            var v = iget(sc[1].simplices, e, 1)
            var s = 0.0
            for j in range(n):
                s += Float64(a[j]) * (fget(V, v, j) - fget(V, u, j))
            expected.unsafe_set(e, s)
        _assert_close_arrays(w.v, expected, "flat")

    given(
        composite(_flat_mesh),
        lists(integers(-5, 5), min_size=3, max_size=3),
        prop,
        meshes(),
    )


# ===----------------------------------------------------------------------=== #
# I/O round trips
# ===----------------------------------------------------------------------=== #


comptime TMP = "/tmp/mojodec_properties"


def _float_rows(mut data: ConjectureData) raises -> List[List[Float64]]:
    """A small matrix of arbitrary finite floats."""
    var rows = data.draw(integers(1, 4), "rows")
    var cols = data.draw(integers(1, 4), "cols")
    var out = List[List[Float64]]()
    for _ in range(rows):
        out.append(
            data.draw(
                lists(
                    floats(allow_nan=False, allow_infinity=False),
                    min_size=cols,
                    max_size=cols,
                )
            )
        )
    return out^


def _assert_same_bits(a: RealArray, b: RealArray, what: String) raises:
    assert_equal(a.size, b.size, what)
    for i in range(a.size):
        var x = a.unsafe_get(i)
        var y = b.unsafe_get(i)
        # equal values, and the same sign of zero
        if x != y or (x == 0 and (1.0 / x) != (1.0 / y)):
            fail(String(what, "[", i, "]: wrote ", x, ", read ", y))


def test_arrays_round_trip() raises:
    def prop(rows: List[List[Float64]], fmt: Int) raises:
        var A = real_array(len(rows), len(rows[0]))
        for i in range(len(rows)):
            for j in range(len(rows[0])):
                fset(A, i, j, rows[i][j])
        var path = String(TMP, ".arr")
        write_array(path, A, format=ArrayFormat(UInt8(fmt)))
        var B = read_array(path)
        assert_equal(rows_of(B), rows_of(A))
        assert_equal(cols_of(B), cols_of(A))
        _assert_same_bits(A, B, String("format ", ArrayFormat(UInt8(fmt))))

    given(
        composite(_float_rows), integers(0, 2), prop, Settings(max_examples=200)
    )


def test_sparse_matrices_round_trip() raises:
    def prop(rows: List[List[Float64]], keep: List[Bool], binary: Bool) raises:
        # a dense matrix of arbitrary floats with some entries dropped
        var r = List[Int]()
        var c = List[Int]()
        var v = List[Float64]()
        var m = 0
        for i in range(len(rows)):
            for j in range(len(rows[0])):
                if keep[m % len(keep)]:
                    r.append(i)
                    c.append(j)
                    v.append(rows[i][j])
                m += 1
        var A = CSRMatrix.from_coo(len(rows), len(rows[0]), r, c, v)
        var path = String(TMP, ".sparse")
        write_sparse(
            path,
            A,
            format=SparseFormat.BINARY if binary else SparseFormat.ASCII,
        )
        var B = read_sparse(path)
        assert_equal(B.nrows, A.nrows)
        assert_equal(B.ncols, A.ncols)
        assert_equal(B.indptr, A.indptr)
        assert_equal(B.indices, A.indices)
        for k in range(len(A.data)):
            if A.data[k] != B.data[k]:
                fail(
                    String(
                        "entry ", k, ": wrote ", A.data[k], ", read ", B.data[k]
                    )
                )

    given(
        composite(_float_rows),
        lists(booleans(), min_size=1, max_size=8),
        booleans(),
        prop,
        Settings(max_examples=200),
    )


def test_meshes_round_trip() raises:
    def prop(spec: MeshSpec, fmt: Int) raises:
        var mesh = spec.mesh()
        var path = String(TMP, "_mesh.xml")
        write_mesh(path, mesh, format=ArrayFormat(UInt8(fmt)))
        var back = read_mesh(path)
        _assert_same_bits(mesh.vertices, back.vertices, "vertices")
        assert_equal(rows_of(back.elements), rows_of(mesh.elements))
        assert_equal(cols_of(back.elements), cols_of(mesh.elements))
        for i in range(rows_of(mesh.elements)):
            for j in range(cols_of(mesh.elements)):
                assert_equal(
                    iget(back.elements, i, j), iget(mesh.elements, i, j)
                )

    given(composite(_any_mesh), integers(0, 2), prop, meshes())


def test_triangle_meshes_round_trip_through_ply() raises:
    def prop(spec: MeshSpec, binary: Bool) raises:
        var mesh = spec.mesh()
        if cols_of(mesh.elements) != 3:
            return  # PLY holds triangle meshes
        var path = String(TMP, ".ply")
        write_ply(
            path,
            mesh.vertices,
            mesh.elements,
            format=PlyFormat.BINARY if binary else PlyFormat.ASCII,
        )
        var back = read_ply(path)
        var E = cols_of(mesh.vertices)
        assert_equal(rows_of(back.vertices), rows_of(mesh.vertices))
        # 2-D vertices come back with z = 0
        for i in range(rows_of(mesh.vertices)):
            for j in range(cols_of(back.vertices)):
                var x = fget(mesh.vertices, i, j) if j < E else 0.0
                var y = fget(back.vertices, i, j)
                if x != y:
                    fail(
                        String(
                            "vertex ", i, "[", j, "]: wrote ", x, ", read ", y
                        )
                    )
        for i in range(rows_of(mesh.elements)):
            for j in range(3):
                assert_equal(
                    iget(back.elements, i, j), iget(mesh.elements, i, j)
                )

    given(composite(_any_mesh), booleans(), prop, meshes())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
