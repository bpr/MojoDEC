# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
The de Rham map and Whitney interpolation (`mojodec.routines.derham`).
=====================================================================
The two maps between differential forms and cochains; not in PyDEC.

- `derham_map` (R) takes a k-form to the k-cochain of its integrals over
  the oriented k-simplices, by Grundmann–Möller quadrature. By Stokes'
  theorem `R(d omega) = d R(omega)`.
- `l2_projection` takes a k-form to the cochain whose Whitney form is
  closest to it in L2 (`whitney_load` gives the right-hand side), and
  `l2_error` measures `||W c - omega||`.
- `whitney_form` (W) evaluates the Whitney form of a cochain at a point of
  an element: `W c = sum_sigma c_sigma W_sigma`, with
  `W_sigma = k! sum_i (-1)^i lambda_{s_i} dlambda_{s_0} ^ ... (no s_i) ... ^ dlambda_{s_k}`
  for `sigma = [s_0, ..., s_k]`. It satisfies `R W = id` and `d W = W d`,
  and reproduces constant forms: `W R omega = omega` for constant `omega`.

A k-form on `R^N` is given by its components on the basis
`dx^{i_1} ^ ... ^ dx^{i_k}`, `i_1 < ... < i_k`, in lexicographic order of
the index sets (`form_index_sets(N, k)`): a 1-form in 3D is
`[w_x, w_y, w_z]`, a 2-form `[w_xy, w_xz, w_yz]`. Simplices are oriented
as the cochains of `sc[k]` are: by their sorted vertex lists, except
the elements, which keep the orientation of the mesh (`simplex_parity`).

```mojo
def omega(x: List[Float64]) -> List[Float64]:      # x^2 dx + x y dy
    return [x[0] * x[0], x[0] * x[1]]

var c = derham_map(sc, 1, 3, omega)                # a Cochain on the edges
var w = whitney_form(c, 0, [0.2, 0.3, 0.5])        # components at a point
```
"""

from std.math import sqrt

from mojodec.core.cochain import Cochain
from mojodec.core.primality import Primality
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.typed import DUAL, PRIMAL, TypedCochain, TypedComplex
from mojodec.core.utils import (
    RealArray,
    cols_of,
    fget,
    fset,
    real_array,
    real_vector,
)
from mojodec.routines.geometry import barycentric_gradients
from mojodec.routines.innerproduct import whitney_innerproduct
from mojodec.routines.linalg import cg, jacobi
from mojodec.routines.quadrature import grundmann_moeller


def form_index_sets(N: Int, k: Int) raises -> List[List[Int]]:
    """The index sets of the basis k-forms on `R^N`, in the order of the
    components `derham_map` and `whitney_form` use.

    ```mojo
    form_index_sets(3, 2)    # [[0, 1], [0, 2], [1, 2]]: dx^dy, dx^dz, dy^dz
    ```

    Args:
        N: The dimension of the space.
        k: The degree, `0..N`.

    Returns:
        The `binomial(N, k)` increasing index lists, lexicographically
        ordered (one empty list for `k = 0`).

    Raises:
        If `k` is outside `0..N`.
    """
    if k < 0 or k > N:
        raise Error(String("form_index_sets: k = ", k, " is outside 0..", N))
    var out = List[List[Int]]()
    var idx = List[Int](capacity=k)
    for i in range(k):
        idx.append(i)
    while True:
        out.append(idx.copy())
        var i = k - 1
        while i >= 0 and idx[i] == N - k + i:
            i -= 1
        if i < 0:
            return out^
        idx[i] += 1
        for j in range(i + 1, k):
            idx[j] = idx[j - 1] + 1


def derham_map[
    F: def(List[Float64]) raises -> List[Float64]
](sc: SimplicialComplex, k: Int, degree: Int, omega: F) raises -> Cochain:
    """The de Rham map: the k-cochain of the integrals of a k-form over the
    k-simplices.

    The value on the simplex `[x_0, ..., x_k]` (sorted vertices; for the
    elements, the mesh's orientation, which negates it when
    `simplex_parity` is 1) is

        int_sigma omega = (1/k!) mean_sigma omega(x)(x_1 - x_0, ..., x_k - x_0),

    the mean taken by the Grundmann–Möller rule of the given degree, so it
    is exact when the components of `omega` are polynomials of at most
    that degree. For `k = 0` it samples `omega` at the vertices.

    ```mojo
    def grad_f(x: List[Float64]) -> List[Float64]:     # d(x^2 y)
        return [2 * x[0] * x[1], x[0] * x[0]]

    var c = derham_map(sc, 1, 2, grad_f)               # = d0 of the values of f
    ```

    Parameters:
        F: The form's type: a function or closure.

    Args:
        sc: The complex.
        k: The degree of the form, `0..n`.
        degree: The quadrature degree, `0..21`.
        omega: The form: given a point's `N` Cartesian coordinates, its
            `binomial(N, k)` components in the order of
            `form_index_sets(N, k)`.

    Returns:
        A primal k-cochain on `sc`.

    Raises:
        If `k` or `degree` is out of range, `omega` returns the wrong number
        of components, or `omega` raises.
    """
    var n = sc.complex_dimension()
    if k < 0 or k > n:
        raise Error(String("derham_map: k = ", k, " is outside 0..", n))
    var rule = grundmann_moeller(k, degree)
    ref V = sc.vertices()
    var N = cols_of(V)
    var sets = form_index_sets(N, k)
    ref level = sc[k]
    var m = level.num_simplices
    var out = real_vector(m)
    var kfact: Float64 = 1
    for j in range(2, k + 1):
        kfact *= Float64(j)
    var x = List[Float64](length=N, fill=0.0)
    var edges = List[Float64](length=k * N, fill=0.0)
    var minors = List[Float64](length=len(sets), fill=0.0)
    var verts = List[Int](length=k + 1, fill=0)
    for s in range(m):
        for j in range(k + 1):
            verts[j] = Int(level.simplices.unsafe_get(s * (k + 1) + j))
        for j in range(k):
            for c in range(N):
                edges[j * N + c] = fget(V, verts[j + 1], c) - fget(
                    V, verts[0], c
                )
        for I in range(len(sets)):
            minors[I] = _minor(edges, k, N, sets[I])
        var total: Float64 = 0
        for q in range(len(rule)):
            for c in range(N):
                var p: Float64 = 0
                for j in range(k + 1):
                    p += rule.points.unsafe_get(q * (k + 1) + j) * fget(
                        V, verts[j], c
                    )
                x[c] = p
            var w = omega(x)
            if len(w) != len(sets):
                raise Error(
                    String(
                        "derham_map: omega gave ",
                        len(w),
                        " components; a ",
                        k,
                        "-form on R^",
                        N,
                        " has ",
                        len(sets),
                    )
                )
            var val: Float64 = 0
            for I in range(len(sets)):
                val += w[I] * minors[I]
            total += rule.weights.unsafe_get(q) * val
        # the cochains of sc[k] follow its orientations: sorted vertices,
        # and for the elements the mesh's own order (simplex_parity)
        if level.simplex_parity.unsafe_get(s) == 1:
            total = -total
        out.unsafe_set(s, total / kfact)
    return Cochain(sc, k, Primality.PRIMAL, out^)


def derham_map[
    n: Int, //, k: Int, F: def(List[Float64]) raises -> List[Float64]
](tc: TypedComplex[n], degree: Int, omega: F) raises -> TypedCochain[
    n, k, PRIMAL
]:
    """The de Rham map as a typed primal k-cochain; see the untyped
    overload.

    Parameters:
        n: The complex dimension (inferred).
        k: The degree of the form.
        F: The form's type.

    Args:
        tc: The complex.
        degree: The quadrature degree.
        omega: The form's components at a point.

    Returns:
        The cochain of integrals over the k-simplices.

    Raises:
        As the untyped overload.
    """
    comptime assert 0 <= k and k <= n, "derham_map: k must be in 0..n"
    var c = derham_map(tc.sc, k, degree, omega)
    return TypedCochain[n, k, PRIMAL](tc.sc, c.v.copy())


def whitney_form(
    c: Cochain, element: Int, barycentric: List[Float64]
) raises -> List[Float64]:
    """The Whitney form of a primal k-cochain at a point of an element.

    `W c = sum_sigma c_sigma W_sigma` over the k-faces `sigma` of the
    element, with `W_sigma` built on the sorted vertices (and `c_sigma`
    negated for an element whose mesh orientation is the opposite one,
    `simplex_parity` 1):
    `W_sigma = k! sum_i (-1)^i lambda_{s_i} dlambda_{s_0} ^ ... ^ dlambda_{s_k}`
    (the factor `dlambda_{s_i}` left out). Inside an element it is a
    polynomial of degree 1; across a face its tangential part is
    continuous, so the value on a shared face does not depend on which
    element it is evaluated from. For `k = 0` it is the piecewise linear
    interpolant, for `k = n` the constant `c_t / vol(t)` times the volume
    form.

    ```mojo
    var w = whitney_form(c, t, [0.25, 0.25, 0.5])     # a 1-form: [w_x, w_y]
    ```

    Args:
        c: A primal k-cochain.
        element: The index of an n-simplex (`sc[n]`).
        barycentric: The point's `n + 1` barycentric coordinates in the
            element, in the order of its sorted vertices, summing to 1.

    Returns:
        The form's `binomial(N, k)` components in the order of
        `form_index_sets(N, k)`, `N` the dimension of the vertex
        coordinates.

    Raises:
        If `c` is dual, `element` is out of range, `barycentric` has the
        wrong length or does not sum to 1, or the element is degenerate.
    """
    if c.primality() != Primality.PRIMAL:
        raise Error("whitney_form: the cochain must be primal")
    ref sc = c.complex
    var n = sc.complex_dimension()
    var k = c.k
    ref top = sc[n]
    if element < 0 or element >= top.num_simplices:
        raise Error(
            String(
                "whitney_form: element ",
                element,
                " is outside 0..",
                top.num_simplices - 1,
            )
        )
    if len(barycentric) != n + 1:
        raise Error(
            String(
                "whitney_form: ",
                len(barycentric),
                " barycentric coordinates for a ",
                n,
                "-simplex",
            )
        )
    var total: Float64 = 0
    for b in barycentric:
        total += b
    if abs(total - 1) > 1e-12:
        raise Error(
            String("whitney_form: barycentric coordinates sum to ", total)
        )
    var e = _WhitneyElement(sc, k, element)
    var out = List[Float64](length=e.ncomp, fill=0.0)
    var w = List[Float64](length=e.ncomp, fill=0.0)
    for f in range(len(e.rows)):
        var cs = c.v.unsafe_get(e.rows[f])
        if cs == 0:
            continue
        e.basis(f, barycentric, w)
        for I in range(e.ncomp):
            out[I] += cs * w[I]
    return out^


def whitney_form[
    n: Int, k: Int
](
    c: TypedCochain[n, k, PRIMAL], element: Int, barycentric: List[Float64]
) raises -> List[Float64]:
    """The Whitney form of a typed primal k-cochain at a point; see the
    untyped overload.

    Parameters:
        n: The complex dimension.
        k: The degree.

    Args:
        c: The cochain.
        element: The index of an n-simplex.
        barycentric: The point's barycentric coordinates in it.

    Returns:
        The form's components in the order of `form_index_sets(N, k)`.

    Raises:
        As the untyped overload.
    """
    return whitney_form(
        Cochain(c.complex, k, Primality.PRIMAL, c.v.copy()),
        element,
        barycentric,
    )


def _minor(vecs: List[Float64], k: Int, N: Int, cols: List[Int]) -> Float64:
    """`det` of the `k x k` matrix of the columns `cols` of the `k` row
    vectors `vecs` (`k x N`, row-major): `(v_1 ^ ... ^ v_k)` on
    `dx^cols`. Gaussian elimination with partial pivoting; 1 for `k = 0`."""
    if k == 0:
        return 1
    if k == 1:
        return vecs[cols[0]]
    if k == 2:
        return (
            vecs[cols[0]] * vecs[N + cols[1]]
            - vecs[cols[1]] * vecs[N + cols[0]]
        )
    var a = List[Float64](capacity=k * k)
    for r in range(k):
        for c in range(k):
            a.append(vecs[r * N + cols[c]])
    var det: Float64 = 1
    for p in range(k):
        var best = p
        for r in range(p + 1, k):
            if abs(a[r * k + p]) > abs(a[best * k + p]):
                best = r
        if a[best * k + p] == 0:
            return 0
        if best != p:
            for c in range(k):
                var t = a[p * k + c]
                a[p * k + c] = a[best * k + c]
                a[best * k + c] = t
            det = -det
        det *= a[p * k + p]
        for r in range(p + 1, k):
            var f = a[r * k + p] / a[p * k + p]
            for c in range(p, k):
                a[r * k + c] -= f * a[p * k + c]
    return det


struct _WhitneyElement(Movable):
    """The Whitney k-forms of one element's k-faces, ready to evaluate:
    `W_f = sign_f k! sum_i (-1)^i lambda_{l_i} wedges[f][i]`, the wedge of
    the other gradients being constant in the element."""

    var rows: List[Int]
    """The global index of each local k-face in `sc[k]`."""
    var locals: List[List[Int]]
    """Each face's local vertex numbers in the element."""
    var wedges: List[List[List[Float64]]]
    """`[face][i]`: the components of the wedge of the face's gradients
    other than the i-th, times `sign_f k! (-1)^i`."""
    var ncomp: Int
    """The number of components of a k-form on `R^N`."""
    var verts: List[Int]
    """The element's vertices, sorted."""
    var volume: Float64
    """The element's unsigned volume."""

    def __init__(out self, sc: SimplicialComplex, k: Int, element: Int) raises:
        var n = sc.complex_dimension()
        ref V = sc.vertices()
        var N = cols_of(V)
        ref top = sc[n]
        self.verts = List[Int](capacity=n + 1)
        var pts = real_array(n + 1, N)
        for j in range(n + 1):
            self.verts.append(
                Int(top.simplices.unsafe_get(element * (n + 1) + j))
            )
            for col in range(N):
                fset(pts, j, col, fget(V, self.verts[j], col))
        self.volume = abs(top.primal_volume.unsafe_get(element))
        var grads = barycentric_gradients(pts)
        var sets = form_index_sets(N, k)
        self.ncomp = len(sets)
        var kfact: Float64 = 1
        for j in range(2, k + 1):
            kfact *= Float64(j)
        ref level = sc[k]
        self.rows = List[Int]()
        self.locals = List[List[Int]]()
        self.wedges = List[List[List[Float64]]]()
        var face = List[Int](length=k + 1, fill=0)
        var local = List[Int](capacity=k + 1)
        for i in range(k + 1):
            local.append(i)
        var vecs = List[Float64](length=k * N, fill=0.0)
        while True:
            for j in range(k + 1):
                face[j] = self.verts[local[j]]
            var row = level.find_vertices(face)
            # the cochain's value is on the oppositely oriented simplex
            # where the parity is 1 (elements keep the mesh's orientation)
            var sign: Float64 = (
                -1.0 if level.simplex_parity.unsafe_get(row) == 1 else 1.0
            )
            var per_i = List[List[Float64]]()
            for i in range(k + 1):
                var coef = sign * kfact * (-1.0 if i % 2 == 1 else 1.0)
                var r = 0
                for j in range(k + 1):
                    if j == i:
                        continue
                    for col in range(N):
                        vecs[r * N + col] = fget(grads, local[j], col)
                    r += 1
                var comps = List[Float64](capacity=len(sets))
                for I in range(len(sets)):
                    comps.append(coef * _minor(vecs, k, N, sets[I]))
                per_i.append(comps^)
            self.rows.append(row)
            self.locals.append(local.copy())
            self.wedges.append(per_i^)
            var p = k
            while p >= 0 and local[p] == n - k + p:
                p -= 1
            if p < 0:
                break
            local[p] += 1
            for j in range(p + 1, k + 1):
                local[j] = local[j - 1] + 1

    def basis(self, f: Int, barycentric: List[Float64], mut out: List[Float64]):
        """The components of face `f`'s Whitney form at a point."""
        for I in range(self.ncomp):
            out[I] = 0
        ref loc = self.locals[f]
        for i in range(len(loc)):
            var lam = barycentric[loc[i]]
            ref w = self.wedges[f][i]
            for I in range(self.ncomp):
                out[I] += lam * w[I]


def _point(
    sc: SimplicialComplex,
    verts: List[Int],
    barycentric: List[Float64],
    mut x: List[Float64],
):
    """The Cartesian point with the given barycentric coordinates."""
    ref V = sc.vertices()
    for col in range(len(x)):
        var p: Float64 = 0
        for j in range(len(verts)):
            p += barycentric[j] * fget(V, verts[j], col)
        x[col] = p


def _check_components(got: Int, want: Int, who: String) raises:
    if got != want:
        raise Error(
            String(who, ": omega gave ", got, " components, expected ", want)
        )


def whitney_load[
    F: def(List[Float64]) raises -> List[Float64]
](sc: SimplicialComplex, k: Int, degree: Int, omega: F) raises -> RealArray:
    """The L2 inner products of a k-form with the Whitney k-forms:
    `b_sigma = int <omega, W_sigma>`, over the mesh.

    The right-hand side of a Galerkin problem with Whitney forms, and of
    the L2 projection (`l2_projection`): `M c = b` with `M` the Whitney
    mass matrix `whitney_innerproduct(sc, k)`. Integrated element by
    element with the Grundmann–Möller rule of the given degree, exact
    when the components of `omega` are polynomials of degree at most
    `degree - 1` (the Whitney forms add one).

    Parameters:
        F: The form's type: a function or closure.

    Args:
        sc: The complex.
        k: The degree of the form, `0..n`.
        degree: The quadrature degree, `0..21`.
        omega: The form's components at a point (as for `derham_map`).

    Returns:
        One value per k-simplex, in the order of `sc[k]`.

    Raises:
        If `k` or `degree` is out of range, `omega` gives the wrong number
        of components, an element is degenerate, or `omega` raises.
    """
    var n = sc.complex_dimension()
    if k < 0 or k > n:
        raise Error(String("whitney_load: k = ", k, " is outside 0..", n))
    var rule = grundmann_moeller(n, degree)
    var N = cols_of(sc.vertices())
    var out = real_vector(sc[k].num_simplices)
    var x = List[Float64](length=N, fill=0.0)
    var bary = List[Float64](length=n + 1, fill=0.0)
    var w = List[Float64]()
    for t in range(sc[n].num_simplices):
        var e = _WhitneyElement(sc, k, t)
        if len(w) != e.ncomp:
            w = List[Float64](length=e.ncomp, fill=0.0)
        for q in range(len(rule)):
            for j in range(n + 1):
                bary[j] = rule.points.unsafe_get(q * (n + 1) + j)
            _point(sc, e.verts, bary, x)
            var om = omega(x)
            _check_components(len(om), e.ncomp, "whitney_load")
            var scale = e.volume * rule.weights.unsafe_get(q)
            for f in range(len(e.rows)):
                e.basis(f, bary, w)
                var ip: Float64 = 0
                for I in range(e.ncomp):
                    ip += om[I] * w[I]
                var r = e.rows[f]
                out.unsafe_set(r, out.unsafe_get(r) + scale * ip)
    return out^


def l2_projection[
    F: def(List[Float64]) raises -> List[Float64]
](
    sc: SimplicialComplex,
    k: Int,
    degree: Int,
    omega: F,
    rtol: Float64 = 1e-12,
) raises -> Cochain:
    """The L2 projection of a k-form onto the Whitney k-forms.

    The cochain `c` whose Whitney form is closest to `omega` in the L2
    norm: `M c = b`, with `M = whitney_innerproduct(sc, k)` and
    `b = whitney_load(sc, k, degree, omega)`, solved by conjugate
    gradients with a Jacobi preconditioner (the Whitney mass matrix is
    uniformly well conditioned on shape-regular meshes). Unlike the de
    Rham map, it does not commute with `d`, but its error is the smallest
    possible: `||W c - omega|| <= ||W R omega - omega||`.

    ```mojo
    var c = l2_projection(sc, 1, 5, omega)
    print(l2_error(c, 7, omega))
    ```

    Parameters:
        F: The form's type: a function or closure.

    Args:
        sc: The complex.
        k: The degree of the form, `0..n`.
        degree: The quadrature degree of the load vector.
        omega: The form's components at a point.
        rtol: The relative residual at which CG stops.

    Returns:
        A primal k-cochain.

    Raises:
        If an argument is out of range, `omega` raises or gives the wrong
        number of components, or CG does not converge.
    """
    var b = whitney_load(sc, k, degree, omega)
    var M = whitney_innerproduct(sc, k)
    var res = cg(M, b, M=jacobi(M), rtol=rtol)
    if not res.converged():
        raise Error(
            String(
                "l2_projection: CG stopped (",
                res.stop,
                ") after ",
                res.iterations,
                " iterations, residual ",
                res.residual_norm,
            )
        )
    return Cochain(sc, k, Primality.PRIMAL, res^.take_x())


def l2_error[
    F: def(List[Float64]) raises -> List[Float64]
](c: Cochain, degree: Int, omega: F) raises -> Float64:
    """`||W c - omega||`, the L2 distance between a cochain's Whitney form
    and a k-form over the mesh.

    Integrated element by element with the Grundmann–Möller rule of the
    given degree; the squared difference of a polynomial form of degree
    `p` has degree `2 max(p, 1)`.

    Parameters:
        F: The form's type: a function or closure.

    Args:
        c: A primal k-cochain.
        degree: The quadrature degree.
        omega: The form's components at a point.

    Returns:
        The L2 norm of the difference.

    Raises:
        If `c` is dual, `degree` is out of range, `omega` gives the wrong
        number of components, or `omega` raises.
    """
    if c.primality() != Primality.PRIMAL:
        raise Error("l2_error: the cochain must be primal")
    ref sc = c.complex
    var n = sc.complex_dimension()
    var k = c.k
    var rule = grundmann_moeller(n, degree)
    var N = cols_of(sc.vertices())
    var x = List[Float64](length=N, fill=0.0)
    var bary = List[Float64](length=n + 1, fill=0.0)
    var w = List[Float64]()
    var total: Float64 = 0
    for t in range(sc[n].num_simplices):
        var e = _WhitneyElement(sc, k, t)
        if len(w) != e.ncomp:
            w = List[Float64](length=e.ncomp, fill=0.0)
        var acc = List[Float64](length=e.ncomp, fill=0.0)
        var part: Float64 = 0
        for q in range(len(rule)):
            for j in range(n + 1):
                bary[j] = rule.points.unsafe_get(q * (n + 1) + j)
            _point(sc, e.verts, bary, x)
            var om = omega(x)
            _check_components(len(om), e.ncomp, "l2_error")
            for I in range(e.ncomp):
                acc[I] = -om[I]
            for f in range(len(e.rows)):
                var cs = c.v.unsafe_get(e.rows[f])
                if cs == 0:
                    continue
                e.basis(f, bary, w)
                for I in range(e.ncomp):
                    acc[I] += cs * w[I]
            var sq: Float64 = 0
            for I in range(e.ncomp):
                sq += acc[I] * acc[I]
            part += rule.weights.unsafe_get(q) * sq
        total += e.volume * part
    return sqrt(total)


def whitney_load[
    n: Int, //, k: Int, F: def(List[Float64]) raises -> List[Float64]
](tc: TypedComplex[n], degree: Int, omega: F) raises -> TypedCochain[
    n, n - k, DUAL
]:
    """The load vector as a dual (n-k)-cochain, the kind
    `whitney_innerproduct[k](tc)` gives; see the untyped overload.

    Parameters:
        n: The complex dimension (inferred).
        k: The degree of the form.
        F: The form's type.

    Args:
        tc: The complex.
        degree: The quadrature degree.
        omega: The form's components at a point.

    Returns:
        `b_sigma = int <omega, W_sigma>` for every k-simplex.

    Raises:
        As the untyped overload.
    """
    comptime assert 0 <= k and k <= n, "whitney_load: k must be in 0..n"
    return TypedCochain[n, n - k, DUAL](
        tc.sc, whitney_load(tc.sc, k, degree, omega)
    )


def l2_projection[
    n: Int, //, k: Int, F: def(List[Float64]) raises -> List[Float64]
](
    tc: TypedComplex[n], degree: Int, omega: F, rtol: Float64 = 1e-12
) raises -> TypedCochain[n, k, PRIMAL]:
    """The L2 projection as a typed primal k-cochain; see the untyped
    overload.

    Parameters:
        n: The complex dimension (inferred).
        k: The degree of the form.
        F: The form's type.

    Args:
        tc: The complex.
        degree: The quadrature degree of the load vector.
        omega: The form's components at a point.
        rtol: The relative residual at which CG stops.

    Returns:
        The cochain whose Whitney form is L2-closest to `omega`.

    Raises:
        As the untyped overload.
    """
    comptime assert 0 <= k and k <= n, "l2_projection: k must be in 0..n"
    var c = l2_projection(tc.sc, k, degree, omega, rtol)
    return TypedCochain[n, k, PRIMAL](tc.sc, c.v.copy())


def l2_error[
    n: Int, k: Int, F: def(List[Float64]) raises -> List[Float64]
](c: TypedCochain[n, k, PRIMAL], degree: Int, omega: F) raises -> Float64:
    """`||W c - omega||` for a typed cochain; see the untyped overload.

    Parameters:
        n: The complex dimension.
        k: The degree.
        F: The form's type.

    Args:
        c: The cochain.
        degree: The quadrature degree.
        omega: The form's components at a point.

    Returns:
        The L2 norm of the difference.

    Raises:
        As the untyped overload.
    """
    return l2_error(
        Cochain(c.complex, k, Primality.PRIMAL, c.v.copy()), degree, omega
    )
