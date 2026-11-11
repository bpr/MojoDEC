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

from mojodec.core.cochain import Cochain
from mojodec.core.primality import Primality
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.typed import PRIMAL, TypedCochain, TypedComplex
from mojodec.core.utils import (
    RealArray,
    cols_of,
    fget,
    fset,
    real_array,
    real_vector,
)
from mojodec.routines.geometry import barycentric_gradients
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
    ref V = sc.vertices()
    var N = cols_of(V)
    var verts = List[Int](capacity=n + 1)
    var pts = real_array(n + 1, N)
    for j in range(n + 1):
        verts.append(Int(top.simplices.unsafe_get(element * (n + 1) + j)))
        for col in range(N):
            fset(pts, j, col, fget(V, verts[j], col))
    var grads = barycentric_gradients(pts)
    var sets = form_index_sets(N, k)
    var out = List[Float64](length=len(sets), fill=0.0)
    var kfact: Float64 = 1
    for j in range(2, k + 1):
        kfact *= Float64(j)
    ref level = sc[k]
    var face = List[Int](length=k + 1, fill=0)
    var local = List[Int](capacity=k + 1)
    for i in range(k + 1):
        local.append(i)
    var vecs = List[Float64](length=k * N, fill=0.0)
    while True:
        # the face with local vertices `local`, sorted like the element
        for j in range(k + 1):
            face[j] = verts[local[j]]
        var row = level.find_vertices(face)
        var cs = c.v.unsafe_get(row)
        if level.simplex_parity.unsafe_get(row) == 1:
            cs = -cs  # the value is on the oppositely oriented simplex
        if cs != 0:
            for i in range(k + 1):
                var coef = kfact * barycentric[local[i]]
                if i % 2 == 1:
                    coef = -coef
                # the wedge of the other k gradients
                var r = 0
                for j in range(k + 1):
                    if j == i:
                        continue
                    for col in range(N):
                        vecs[r * N + col] = fget(grads, local[j], col)
                    r += 1
                for I in range(len(sets)):
                    out[I] += cs * coef * _minor(vecs, k, N, sets[I])
        # next k-face of the element
        var p = k
        while p >= 0 and local[p] == n - k + p:
            p -= 1
        if p < 0:
            break
        local[p] += 1
        for j in range(p + 1, k + 1):
            local[j] = local[j - 1] + 1
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
