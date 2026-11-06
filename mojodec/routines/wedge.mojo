# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Wedge product (`mojodec.routines.wedge`).
=========================================
The discrete wedge product of primal cochains, which PyDEC lacks.

For a k-cochain `a` and an l-cochain `b`, its value on a (k+l)-simplex
`[v_0, ..., v_{k+l}]` is

    (a ^ b)[v_0..v_{k+l}] = 1/(k+l+1)!  sum over permutations t of 0..k+l of
                            sign(t) a[v_t(0)..v_t(k)] b[v_t(k)..v_t(k+l)],

the antisymmetrized cup product (A. N. Hirani, *Discrete Exterior
Calculus*, PhD thesis, Caltech 2003, ch. 7). It is bilinear, satisfies
`a ^ b = (-1)^(kl) b ^ a` and the Leibniz rule
`d(a ^ b) = da ^ b + (-1)^k a ^ db`, and is exact for constant forms
(`dx ^ dy` is the signed area of each triangle). With a 0-cochain `f` it
gives `(a ^ f)[s] = a[s]` times the mean of `f` over the vertices of `s`.
Unlike the smooth wedge, it is not associative in general.
"""

from mojodec.core.cochain import Cochain
from mojodec.core.primality import Primality
from mojodec.core.simplex import Simplex
from mojodec.core.typed import PRIMAL, TypedCochain
from mojodec.core.utils import real_vector, row_list
from mojodec.routines.combinatorial import permutation_parity, permutations


def wedge[
    n: Int, k: Int, l: Int
](
    a: TypedCochain[n, k, PRIMAL], b: TypedCochain[n, l, PRIMAL]
) raises -> TypedCochain[n, k + l, PRIMAL]:
    """The wedge product of two typed primal cochains.

    The degree of the result is in its type, and a product past the complex
    dimension does not compile:

    ```mojo
    var dx = tc.cochain[1, PRIMAL](...)
    var area = wedge(dx, dy)          # TypedCochain[2, 2, PRIMAL]
    # wedge(area, dx)                 # error: k + l exceeds n
    ```

    Parameters:
        n: The complex dimension.
        k: The degree of `a`.
        l: The degree of `b`.

    Args:
        a: A primal k-cochain.
        b: A primal l-cochain on the same complex.

    Returns:
        The primal (k+l)-cochain `a ^ b`.

    Raises:
        If the cochains live on different complexes.
    """
    comptime assert k + l <= n, "wedge: k + l exceeds the complex dimension"
    return TypedCochain[n, k + l, PRIMAL](wedge(a.untyped(), b.untyped()))


def wedge(a: Cochain, b: Cochain) raises -> Cochain:
    """The wedge product `a ^ b` of two primal cochains on the same complex.

    ```mojo
    var w = wedge(alpha, beta)      # a (k+l)-cochain
    ```

    Args:
        a: A primal k-cochain.
        b: A primal l-cochain on the same complex.

    Returns:
        The primal (k+l)-cochain `a ^ b`.

    Raises:
        If either cochain is dual, they live on different complexes, or
            `k + l` exceeds the complex dimension.
    """
    if a.primality().is_dual() or b.primality().is_dual():
        raise Error("wedge: only primal cochains are supported")
    if not a.complex.is_same(b.complex):
        raise Error("wedge: cochains live on different complexes")
    var k = a.k
    var l = b.k
    var sc = a.complex
    if k + l > sc.complex_dimension():
        raise Error(
            String(
                "wedge: degrees ",
                k,
                " + ",
                l,
                " exceed the complex dimension ",
                sc.complex_dimension(),
            )
        )
    var m = k + l + 1
    var order = List[Int](capacity=m)
    for i in range(m):
        order.append(i)
    var perms = permutations(order)
    var signs = List[Float64](capacity=len(perms))
    var scale: Float64 = 1
    for i in range(2, m + 1):
        scale *= Float64(i)
    for ref p in perms:
        signs.append(-1.0 if permutation_parity(p) == 1 else 1.0)
    ref target = sc[k + l]
    var out = real_vector(target.num_simplices)
    for r in range(target.num_simplices):
        var verts = row_list(target.simplices, r)
        var total: Float64 = 0
        for t in range(len(perms)):
            ref p = perms[t]
            var av = List[Int](capacity=k + 1)
            for i in range(k + 1):
                av.append(verts[p[i]])
            var bv = List[Int](capacity=l + 1)
            for i in range(k, m):
                bv.append(verts[p[i]])
            total += signs[t] * a[Simplex(av)] * b[Simplex(bv)]
        # values refer to each simplex's stored orientation, which for top
        # simplices is the mesh's rather than the sorted vertex order
        if Int(target.simplex_parity.unsafe_get(r)) == 1:
            total = -total
        out.unsafe_set(r, total / scale)
    return Cochain(sc, k + l, Primality.PRIMAL, out^)
