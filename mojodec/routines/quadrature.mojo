# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Quadrature on simplices (`mojodec.routines.quadrature`).
========================================================
Grundmann and Möller's rules (1978): for any dimension n and odd degree
`d = 2s + 1`, one closed formula gives a rule on the n-simplex that is
exact for every polynomial of degree at most `d`; not in PyDEC.

The rule's nodes, in barycentric coordinates, are

    x = (2 beta + 1) / (d + n - 2i),   i = 0..s,  beta in N^(n+1), |beta| = s - i,

and the nodes of one `i` share the weight (relative to the simplex volume)

    w_i = (-1)^i 4^(-s) n! (d + n - 2i)^d / (i! (d + n - i)!).

The weights sum to 1. From degree 3 on, some are negative, and the sum of
their absolute values grows with the degree, so high degrees amplify the
integrand's rounding; `QuadratureRule.abs_weight_sum` says by how much,
and `grundmann_moeller` refuses degrees above 21.

```mojo
var rule = grundmann_moeller(2, 5)              # triangles, exact to degree 5

def f(x: List[Float64]) -> Float64:
    return x[0] * x[0] * x[1]

var tri = nm.fromstring[f64]("[[0, 0], [1, 0], [0, 1]]")
print(rule.integrate(tri, f))                   # 1/60
var per_edge = integrate_simplices(sc, 1, 3, f) # one integral per edge
```
"""

from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import (
    RealArray,
    cols_of,
    fget,
    fset,
    real_array,
    real_vector,
    rows_of,
)
from mojodec.routines.geometry import unsigned_volume

comptime MAX_DEGREE = 21
"""The highest degree `grundmann_moeller` builds. There the absolute
weights sum to about 1200 on an edge and 3800 on a 4-simplex, and the
weights sum to 1 only to about 1e-11; at degree 31 that is 1e-9 on a
4-simplex."""


struct QuadratureRule(Copyable, Movable, Sized):
    """A quadrature rule on the n-simplex, in barycentric coordinates.

    `integrate` applies it to a simplex: the integral of `f` over a
    simplex `T` is approximately `vol(T) sum_q weights[q] f(x_q)`, `x_q`
    the point with barycentric coordinates `points[q]`.
    """

    var dim: Int
    """The simplex dimension n."""
    var degree: Int
    """The rule is exact for every polynomial of at most this degree."""
    var points: RealArray
    """`m x (n+1)` barycentric coordinates of the nodes, each row positive
    and summing to 1."""
    var weights: RealArray
    """The `m` weights, relative to the simplex volume: they sum to 1."""

    def __init__(
        out self,
        dim: Int,
        degree: Int,
        var points: RealArray,
        var weights: RealArray,
    ):
        """A rule from its parts; built by `grundmann_moeller`.

        Args:
            dim: The simplex dimension.
            degree: The degree of exactness.
            points: The barycentric nodes, one per row.
            weights: The weights, summing to 1.
        """
        self.dim = dim
        self.degree = degree
        self.points = points^
        self.weights = weights^

    def __len__(self) -> Int:
        """The number of nodes.

        Returns:
            The number of rows of `points`.
        """
        return self.weights.size

    def abs_weight_sum(self) -> Float64:
        """`sum |w_q|`, the factor by which the rule can amplify errors in
        the integrand's values (1 for a rule with positive weights).

        Returns:
            The sum of the absolute weights.
        """
        var s: Float64 = 0
        for q in range(self.weights.size):
            s += abs(self.weights.unsafe_get(q))
        return s

    def integrate[
        F: def(List[Float64]) raises -> Float64
    ](self, vertices: RealArray, f: F) raises -> Float64:
        """The integral of `f` over one simplex, by this rule.

        Parameters:
            F: The integrand's type: a function or closure.

        Args:
            vertices: `(n+1) x N` vertex coordinates, `N >= n`; the simplex
                may be embedded in a higher dimension.
            f: The integrand, given a point's `N` Cartesian coordinates.

        Returns:
            `vol * sum_q w_q f(x_q)` with the unsigned volume.

        Raises:
            If `vertices` does not have `n + 1` rows, or `f` raises.
        """
        if vertices.ndim != 2 or rows_of(vertices) != self.dim + 1:
            raise Error(
                String(
                    "QuadratureRule.integrate: expected ",
                    self.dim + 1,
                    " vertices of a ",
                    self.dim,
                    "-simplex",
                )
            )
        var x = List[Float64](length=cols_of(vertices), fill=0.0)
        return unsigned_volume(vertices) * self._sum(vertices, f, x)

    def _sum[
        F: def(List[Float64]) raises -> Float64
    ](self, vertices: RealArray, f: F, mut x: List[Float64]) raises -> Float64:
        """`sum_q w_q f(x_q)` on a simplex (shape checked by the caller)."""
        var n1 = self.dim + 1
        var N = len(x)
        var s: Float64 = 0
        for q in range(self.weights.size):
            for c in range(N):
                var v: Float64 = 0
                for j in range(n1):
                    v += self.points.unsafe_get(q * n1 + j) * fget(
                        vertices, j, c
                    )
                x[c] = v
            s += self.weights.unsafe_get(q) * f(x)
        return s


def grundmann_moeller(n: Int, degree: Int) raises -> QuadratureRule:
    """The Grundmann–Möller rule on the n-simplex exact to `degree`.

    The rules exist for odd degrees; an even `degree` gets the next odd one
    (`rule.degree` says which). The nodes are interior and symmetric under
    permutations of the vertices. The rule of degree `2s + 1` has
    `sum_{i=0}^s binomial(s - i + n, n)` nodes: the centroid alone for
    degree 1 (the midpoint rule), `n + 2` nodes for degree 3. From degree
    3 on, some weights are negative.

    Args:
        n: The simplex dimension, at least 0 (a point: one node, weight 1).
        degree: The polynomial degree to integrate exactly,
            `0..MAX_DEGREE` (21).

    Returns:
        The rule, with weights summing to 1.

    Raises:
        If `n` is negative or `degree` is outside `0..MAX_DEGREE`.
    """
    if n < 0:
        raise Error(String("grundmann_moeller: n = ", n, " is negative"))
    if degree < 0 or degree > MAX_DEGREE:
        raise Error(
            String(
                "grundmann_moeller: degree ",
                degree,
                " is outside 0..",
                MAX_DEGREE,
                " (higher degrees lose accuracy to cancellation)",
            )
        )
    var s = degree // 2
    var d = 2 * s + 1
    var count = 0
    for i in range(s + 1):
        count += _binomial(s - i + n, n)
    var points = real_array(count, n + 1)
    var weights = real_vector(count)
    var row = 0
    var beta = List[Int](length=n + 1, fill=0)
    for i in range(s + 1):
        var w = _weight(n, s, d, i)
        var denom = Float64(d + n - 2 * i)
        # every beta in N^(n+1) with |beta| = s - i
        _first_composition(beta, s - i)
        while True:
            for j in range(n + 1):
                fset(points, row, j, Float64(2 * beta[j] + 1) / denom)
            weights.unsafe_set(row, w)
            row += 1
            if not _next_composition(beta):
                break
    return QuadratureRule(n, d, points^, weights^)


def integrate_simplices[
    F: def(List[Float64]) raises -> Float64
](sc: SimplicialComplex, k: Int, degree: Int, f: F) raises -> RealArray:
    """The integral of `f` over each k-simplex of a complex.

    With the unsigned volumes: the integrals of a density, not of a
    k-form. For `k = 0` they are the values at the vertices; for the top
    dimension, the integrals over the elements, whose sum is the integral
    over the mesh.

    ```mojo
    def f(x: List[Float64]) -> Float64:
        return x[0] * x[0]

    var per_triangle = integrate_simplices(sc, 2, 2, f)
    ```

    Parameters:
        F: The integrand's type: a function or closure.

    Args:
        sc: The complex.
        k: The dimension of the simplices, `0..n`.
        degree: The degree of the Grundmann–Möller rule; exact when `f`
            is a polynomial of at most this degree.
        f: The integrand, given a point's Cartesian coordinates.

    Returns:
        One integral per k-simplex, in the order of `sc[k]`.

    Raises:
        If `k` or `degree` is out of range, or `f` raises.
    """
    if k < 0 or k > sc.complex_dimension():
        raise Error(
            String(
                "integrate_simplices: k = ",
                k,
                " is outside 0..",
                sc.complex_dimension(),
            )
        )
    var rule = grundmann_moeller(k, degree)
    ref level = sc[k]
    ref V = sc.vertices()
    var N = cols_of(V)
    var m = level.num_simplices
    var out = real_vector(m)
    var verts = real_array(k + 1, N)
    var x = List[Float64](length=N, fill=0.0)
    for i in range(m):
        for j in range(k + 1):
            var v = Int(level.simplices.unsafe_get(i * (k + 1) + j))
            for c in range(N):
                fset(verts, j, c, fget(V, v, c))
        var vol = abs(level.primal_volume.unsafe_get(i))
        out.unsafe_set(i, vol * rule._sum(verts, f, x))
    return out^


def _weight(n: Int, s: Int, d: Int, i: Int) -> Float64:
    """`(-1)^i 4^-s n! (d+n-2i)^d / (i! (d+n-i)!)`, as a product of
    factors taken so that the running value stays near 1 (no overflow, one
    rounding per factor)."""
    var up = List[Float64]()
    var down = List[Float64]()
    for _ in range(d):
        up.append(Float64(d + n - 2 * i))
    for j in range(2, n + 1):
        up.append(Float64(j))
    for j in range(2, i + 1):
        down.append(Float64(j))
    for j in range(2, d + n - i + 1):
        down.append(Float64(j))
    for _ in range(s):
        down.append(4.0)
    var w: Float64 = 1
    var a = 0
    var b = 0
    while a < len(up) or b < len(down):
        if b < len(down) and (w >= 1 or a == len(up)):
            w /= down[b]
            b += 1
        else:
            w *= up[a]
            a += 1
    return -w if i % 2 == 1 else w


def _binomial(a: Int, b: Int) -> Int:
    var r = 1
    for j in range(b):
        r = r * (a - j) // (j + 1)
    return r


def _first_composition(mut beta: List[Int], total: Int):
    """`beta = (total, 0, ..., 0)`."""
    for j in range(len(beta)):
        beta[j] = 0
    beta[0] = total


def _next_composition(mut beta: List[Int]) -> Bool:
    """Step to the next composition of the same total (all of them, from
    `(total, 0, ...)` to `(0, ..., total)`); False after the last."""
    var m = len(beta)
    if m == 1:
        return False
    # find the first nonzero entry before the last position
    var j = 0
    while j < m - 1 and beta[j] == 0:
        j += 1
    if j == m - 1:
        return False
    var t = beta[j]
    beta[j] = 0
    beta[0] = t - 1
    beta[j + 1] += 1
    return True
