# Quadrature on simplices

To put continuous data on a mesh, a load vector, the integral of a source
over each element, or the cochain of a given field, you integrate over
simplices. `mojodec.routines.quadrature` provides Grundmann and Möller's
rules: one closed formula gives a rule for any dimension and any odd
degree.

```mojo
from mojodec import grundmann_moeller, integrate_simplices

def f(x: List[Float64]) -> Float64:
    return x[0] * x[0] * x[1]

var rule = grundmann_moeller(2, 3)                  # triangles, exact to degree 3
var I = rule.integrate(triangle_vertices, f)        # one simplex
var per_element = integrate_simplices(sc, 2, 3, f)  # every triangle of a mesh
```

## Rules in barycentric coordinates

A quadrature rule on the n-simplex is a list of nodes, given by their
barycentric coordinates $\xi_q \in \mathbb R^{n+1}$, and weights $w_q$.
For a simplex $T$ with vertices $v_0, \dots, v_n$, it approximates

$$
\int_T f \, dV \;\approx\; |T| \sum_q w_q \, f\Big( \sum_j \xi_{q,j} \, v_j \Big),
$$

with $|T|$ the unsigned volume. The map from barycentric to Cartesian
coordinates is affine, so a polynomial of degree $p$ in $x$ is one of
degree $p$ in $\xi$. A rule exact for barycentric polynomials of degree $p$
is therefore exact for every polynomial of degree $p$ on every simplex.
That includes simplices embedded in a higher-dimensional space, such as
the triangles of a surface in $\mathbb R^3$ or the edges of any mesh.

With the weights normalized to sum to 1, the exactness conditions are

$$
\sum_q w_q \prod_j \xi_{q,j}^{\alpha_j} \;=\; \frac{n! \, \prod_j \alpha_j!}{(n + |\alpha|)!}
\qquad \text{for all } |\alpha| \le p,
$$

the mean of the barycentric monomial over the simplex.

## The Grundmann–Möller rules

For degree $d = 2s + 1$ (Grundmann and Möller, *SIAM J. Numer. Anal.* 15,
1978), the rule is

$$
\int_T f \;\approx\; |T| \sum_{i=0}^{s} (-1)^i \,
\frac{n! \, (d + n - 2i)^d}{4^s \, i! \, (d + n - i)!}
\sum_{\substack{\beta \in \mathbb N^{n+1} \\ |\beta| = s - i}}
f\Big( \frac{2\beta + 1}{d + n - 2i} \Big),
$$

the node's barycentric coordinates being $(2\beta_j + 1)/(d + n - 2i)$.
For each $i$, the nodes form a regular lattice strictly inside $T$, with
barycentric spacing $2/(d + n - 2i)$, all with the same weight. The terms
alternate in sign. Grundmann and Möller derived the weights so that the
combination is exact for every polynomial of degree $d$.

Properties:
- **Any n and d.** The same few lines build them all, which is why MojoDEC
  uses them rather than tables of special rules.
- **Symmetric and interior.** The nodes are symmetric under permutations of
  the vertices, and lie strictly inside the simplex.
- **Node count.** It is $\sum_{i=0}^{s} \binom{s - i + n}{n}$: the
  centroid alone for degree 1 (the midpoint rule), then $n + 2$ nodes for
  degree 3.
- **Even degrees round up.** `grundmann_moeller(n, 4)` returns the degree-5
  rule, and `rule.degree` says which degree you got.

### Negative weights and the degree limit

From degree 3 on, the terms with odd $i$ have negative weights. The
weights still sum to 1, but their absolute values sum to more, and that
sum bounds how much the rule amplifies errors in the integrand's values:

$$
\Big| \sum_q w_q \tilde f_q - \sum_q w_q f_q \Big| \le \Big( \sum_q |w_q| \Big) \max_q |\tilde f_q - f_q| .
$$

`rule.abs_weight_sum()` reports it. Measured (number of nodes, then
$\sum|w|$):

| degree | edge | triangle | tetrahedron | 4-simplex |
|---|---|---|---|---|
| 1 | 1, 1.0 | 1, 1.0 | 1, 1.0 | 1, 1.0 |
| 3 | 3, 1.7 | 4, 2.1 | 5, 2.6 | 6, 3.1 |
| 5 | 6, 3.1 | 10, 4.3 | 15, 5.6 | 21, 7.3 |
| 7 | 10, 6.2 | 20, 8.7 | 35, 11.9 | 56, 16.1 |
| 11 | 21, 26 | 56, 38 | 126, 54 | 252, 76 |
| 15 | 36, 119 | 120, 173 | 330, 251 | 792, 361 |
| 21 | 66, 1200 | 286, 1767 | 1001, 2592 | 3003, 3782 |

The weights themselves are computed in floating point, each to a few
units in the last place. With this much cancellation, at degree 21 their
sum is 1 only to $10^{-13}$ on an edge and $10^{-11}$ on a 4-simplex. At
degree 31 it would be $4 \times 10^{-9}$ on a 4-simplex. `grundmann_moeller` therefore refuses degrees
above `MAX_DEGREE = 21`.

For more accuracy on a non-polynomial integrand, consider refining the
mesh before raising the degree. A rule of degree $d$ on simplices of size
$h$ has an error of order $h^{d+1}$ for a smooth integrand, and moderate
degrees keep $\sum |w|$ small.

## The API

- `grundmann_moeller(n, degree) -> QuadratureRule`. The rule's fields are:
  - `dim`: the dimension n.
  - `degree`: the odd degree of exactness.
  - `points`: $m \times (n+1)$ barycentric coordinates.
  - `weights`: $m$ entries summing to 1.
- `rule.integrate(vertices, f)` integrates over one simplex, given as
  $(n+1) \times N$ vertex coordinates, $N \ge n$. The rule raises unless
  there are $n + 1$ vertices.
- `integrate_simplices(sc, k, degree, f)` returns one integral per
  k-simplex of a complex, in the order of `sc[k]`. For $k = 0$ these are
  the values $f(v)$; for the top dimension, the integrals over the
  elements. Their sum is the integral over the mesh.

The integrand is any function or closure taking the point's Cartesian
coordinates as a `List[Float64]` (of the ambient dimension) and returning
a `Float64`. It may raise, and the error passes through. Closures capture
parameters:

```mojo
var kappa = 2.5
def source(x: List[Float64]) raises {kappa} -> Float64:
    return kappa * sin(pi * x[0]) * sin(pi * x[1])

var load = integrate_simplices(sc, 2, 5, source)
```

## Densities, not forms

`integrate_simplices` integrates a **density**, a scalar against the
unsigned volume. That is what a load vector, a mass, or an average needs,
and its values do not depend on the simplices' orientation.

The cochain of a **k-form** $\omega$ is different. Its value on an
oriented simplex $\sigma$ is $\int_\sigma \omega$, which evaluates
$\omega$ on the simplex's tangent vectors and changes sign with the
orientation. That is the de Rham map, the next step on top of these rules.
For $k = 0$ the two coincide (the values at the vertices), and for $k = n$
in $\mathbb R^n$ they agree on positively oriented elements when
$\omega = f \, dx^1 \wedge \dots \wedge dx^n$.

## Pitfalls

- **Polynomial degree in Cartesian terms.** $x^2 y z^3$ has degree 6 and
  needs a rule of degree at least 6 (so 7) to be exact.
- **Cost** is the number of nodes times the number of simplices, each
  calling `f` once. A degree-7 rule on a 4-simplex mesh calls `f` 56 times
  per element.
- **Kinks and singularities.** A rule is exact only for polynomials. For an
  integrand with a kink inside an element, a higher degree helps little;
  align the mesh with the kink instead.

## Tests to read

`tests/test_quadrature.mojo`:
- `test_rules`: node counts, interior nodes, and weights summing to 1, for
  n = 0..4 and every degree up to 21.
- `test_exactness`: every barycentric monomial up to the degree, against
  $n! \prod \alpha_j! / (n + |\alpha|)!$, and a monomial of degree $d + 1$
  that is not exact.
- `test_integrate_on_a_simplex`: $\int x^2 y = 1/60$ on the unit triangle,
  an embedded triangle's area, and a closure.
- `test_integrate_simplices`: $\int_{[0,1]^3} x^2 y z^3 = 1/24$ on a Kuhn
  grid, edge lengths, and vertex values.
