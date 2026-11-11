# The de Rham map and Whitney forms

Discrete exterior calculus works with cochains: numbers attached to
simplices. Differential forms live on the continuous space. Two maps
connect them, both in `mojodec.routines.derham`:

- the **de Rham map** $R$ takes a k-form to the cochain of its integrals
  over the k-simplices (`derham_map`);
- **Whitney interpolation** $W$ takes a cochain back to a piecewise
  polynomial k-form (`whitney_form` evaluates it).

They are how you put a known field on a mesh (a source term, a boundary
value, a manufactured solution), and how you look at a discrete solution
as a field. Their algebraic properties ($R W = \mathrm{id}$, $d W = W d$,
$R d = d R$) are why DEC's discrete $d$ is exact rather than an
approximation.

```mojo
from mojodec import derham_map, whitney_form

def omega(x: List[Float64]) -> List[Float64]:    # x^2 dx + x y dy on R^2
    return [x[0] * x[0], x[0] * x[1]]

var c = derham_map(sc, 1, 3, omega)               # a primal 1-cochain
var w = whitney_form(c, 0, [0.2, 0.3, 0.5])       # [w_x, w_y] at a point of element 0
```

## Forms as component lists

A k-form on $\mathbb R^N$ is written in the basis
$dx^{i_1} \wedge \dots \wedge dx^{i_k}$ with $i_1 < \dots < i_k$:

$$
\omega = \sum_{I} \omega_I \, dx^I .
$$

Both functions use the components $\omega_I$ in lexicographic order of
the index sets, which `form_index_sets(N, k)` lists:

| $N$, $k$ | order of the components |
|---|---|
| any $N$, 0 | $[f]$ |
| 2, 1 | $[\omega_x, \omega_y]$ |
| 2, 2 | $[\omega_{xy}]$ |
| 3, 1 | $[\omega_x, \omega_y, \omega_z]$ |
| 3, 2 | $[\omega_{xy}, \omega_{xz}, \omega_{yz}]$ |
| 3, 3 | $[\omega_{xyz}]$ |

Note $\omega_{xz}$, not $\omega_{zx} = -\omega_{xz}$. The 2-form often
written $B_x \, dy \wedge dz + B_y \, dz \wedge dx + B_z \, dx \wedge dy$ is
$[B_z, -B_y, B_x]$ here.

A form is given to `derham_map` as a function or closure that takes a
point's Cartesian coordinates and returns its components. On a mesh
embedded in a higher dimension (a surface in $\mathbb R^3$), the form is
one on the ambient space, and only its restriction to the simplices
matters.

## The de Rham map

The cochain $R\omega$ has, on each oriented k-simplex
$\sigma = [x_0, \dots, x_k]$, the value $\int_\sigma \omega$. Parametrize
$\sigma$ over the reference simplex,
$x(\xi) = x_0 + \sum_j \xi_j (x_j - x_0)$. The pullback of $\omega$ is
$\omega(x)(v_1, \dots, v_k) \, d\xi_1 \wedge \dots \wedge d\xi_k$, with
edge vectors $v_j = x_j - x_0$. The reference simplex has volume $1/k!$,
so

$$
\int_\sigma \omega = \frac{1}{k!} \operatorname{mean}_\sigma \, \omega(x)(v_1, \dots, v_k),
\qquad
\omega(x)(v_1, \dots, v_k) = \sum_I \omega_I(x) \det\big[ (v_j)_{i} \big]_{j \in 1..k,\; i \in I}.
$$

The determinants are $k \times k$ minors of the edge vectors, computed once
per simplex. The mean is taken with the Grundmann–Möller rule of the given
`degree` (see [Quadrature](quadrature.md)). It is exact when the
components are polynomials of at most that degree, and otherwise has an
error of order $h^{\text{degree} + 1}$. For $k = 0$ the map samples
$f$ at the vertices.

### Orientation

A cochain's value depends on the orientation of its simplex, so $R$ must
use the orientation the complex uses. MojoDEC orients the k-simplices of
`sc[k]` by their sorted vertex lists, except the **elements** (`sc[n]`),
which keep the orientation of the mesh. `sc[n].simplex_parity` is 1 where
the mesh's order is an odd permutation of the sorted one, and `derham_map`
negates those values.

So on a positively oriented mesh in $\mathbb R^n$, the de Rham map of the
volume form $dx^1 \wedge \dots \wedge dx^n$ is each element's (positive)
volume. The tests check this.

### $R$ commutes with $d$

Stokes' theorem on each $(k+1)$-simplex $\tau$ says
$\int_\tau d\omega = \int_{\partial \tau} \omega$. In cochain terms:

$$
R(d\omega) = d \, R(\omega).
$$

The discrete $d$ (the coboundary, `d(c)`) is therefore exact, not an
approximation: it maps the cochain of $\omega$ to the cochain of
$d\omega$. With quadrature exact for both sides, MojoDEC's implementation
reproduces this to rounding (about $10^{-15}$):
- for $f \mapsto df$, $\omega \mapsto d\omega$ and $\eta \mapsto d\eta$ on
  a tetrahedral mesh;
- for $f \mapsto df$ and $\omega \mapsto d\omega$ on an icosphere.

## Whitney forms

For an oriented k-simplex $\sigma = [s_0, \dots, s_k]$ with barycentric
coordinates $\lambda$, the Whitney form is

$$
W_\sigma = k! \sum_{i=0}^{k} (-1)^i \, \lambda_{s_i} \,
d\lambda_{s_0} \wedge \dots \wedge \widehat{d\lambda_{s_i}} \wedge \dots \wedge d\lambda_{s_k},
$$

and the Whitney interpolant of a cochain is $W c = \sum_\sigma c_\sigma W_\sigma$.
Examples:
- **$k = 0$:** $W_v = \lambda_v$, the hat function, so $W$ of a
  0-cochain is the piecewise linear interpolant.
- **$k = 1$:** $W_{[i,j]} = \lambda_i \, d\lambda_j - \lambda_j \, d\lambda_i$,
  the lowest-order Nédélec (edge) element.
- **$k = n - 1$:** the Raviart–Thomas (face) element, through the
  identification of $(n-1)$-forms with vector fields.
- **$k = n$:** $W_t = n! \, d\lambda_1 \wedge \dots \wedge d\lambda_n$ is
  constant, $\pm 1/|t|$ times the volume form. A density is spread evenly
  over its element.

`whitney_form(c, element, barycentric)` evaluates $W c$ at the point of an
element with the given barycentric coordinates. Those are in the order of
the element's sorted vertices, and must sum to 1. The gradients
$d\lambda$ come from `barycentric_gradients`, which uses QR. The result is
the components of the form in the same order as above. MojoDEC does not
search for the element containing a point; you give the element.

### Properties

- **$R W = \mathrm{id}$.** $\int_\tau W_\sigma = \delta_{\sigma\tau}$:
  each Whitney form integrates to 1 on its own simplex and 0 on every
  other. If a vertex $v$ of $\sigma$ is not in $\tau$, then $\lambda_v$
  vanishes on $\tau$, and so does its pullback $d\lambda_v$. Every term
  of $W_\sigma$ contains one or the other, so $W_\sigma$ pulls back to
  zero on $\tau$. On $\sigma$ itself the integral is 1 by the $k!$
  normalization.
- **Tangential continuity.** The pullback of $W c$ to a shared face
  depends only on the cochain's values on that face's simplices. So the
  tangential part is the same from every element around the face, while
  the normal part can jump. This is the conformity of edge and face
  elements, and the test checks it from every element of each face.
- **$d W = W d$.** On each element, $d$ of a Whitney form is the Whitney
  form of the coboundary. With $R W = \mathrm{id}$, this makes $W$ and $R$
  a pair of cochain maps.
- **Partition of unity.** $\sum_v W_v = \sum_v \lambda_v = 1$.
- **Constants are reproduced.** For a constant form $\omega$,
  $W R \omega = \omega$ exactly. For a smooth form, $W R \omega$ is a
  first-order approximation, with error $O(h)$ in each element.
- **Mass matrices.** $\int \langle W_\sigma, W_\tau \rangle$ is the Whitney
  inner product. The test integrates it with `whitney_form` and a
  degree-2 rule, and matches `whitney_innerproduct` to $10^{-13}$ for
  every k on a perturbed mesh.

## The L² projection

$W R \omega$ interpolates: it matches $\omega$'s integrals over the
simplices exactly. The **L² projection** $P\omega$ is instead the
cochain whose Whitney form is closest to $\omega$ in the $L^2$ norm.
Setting the error orthogonal to every Whitney form gives the normal
equations

$$
\sum_\tau \Big( \int \langle W_\sigma, W_\tau \rangle \Big) c_\tau = \int \langle \omega, W_\sigma \rangle
\quad \text{for every } \sigma,
\qquad\text{that is}\qquad M c = b .
$$

Here $M$ is the Whitney mass matrix, `whitney_innerproduct(sc, k)`, and
$b$ is the **load vector**, `whitney_load(sc, k, degree, omega)`, integrated
element by element with the Grundmann–Möller rule. `l2_projection` solves
the system with conjugate gradients and a Jacobi preconditioner. On
shape-regular meshes the Whitney mass matrix is spectrally equivalent to
its diagonal, so CG converges in a number of iterations independent of the
mesh size. It raises if CG does not converge.

```mojo
var c = l2_projection(sc, 1, 5, omega)       # best L2 fit by Whitney 1-forms
print(l2_error(c, 7, omega))                  # ||W c - omega||
```

How it compares with $R$:
- **Best approximation.** $\|W P\omega - \omega\| \le \|W R\omega - \omega\|$
  always. The convergence tests check this on every mesh, degree, and
  dimension.
- **Not a cochain map.** $P$ does not commute with $d$, so a discrete
  derivative of $P\omega$ is not $P(d\omega)$.
- **Same for constants and for top degree.** Constants are Whitney forms,
  so $P\omega = R\omega$ for them. For $k = n$, both give each element's
  mean of $\omega$, so they coincide.

`l2_error(c, degree, omega)` computes $\|W c - \omega\|$ by quadrature
over the elements. For a smooth $\omega$, use a degree about twice that of
the integrand's polynomial approximation; the convergence tests use 7.

The load vector is also the right-hand side of any Galerkin problem with
Whitney forms. In the typed layer, `whitney_load[k](tc, ...)` is a
`TypedCochain[n, n - k, DUAL]`, the kind `whitney_innerproduct[k](tc)`
produces, so it pairs with the typed mass and stiffness operators.

## Using them

**Putting a field on a mesh.** Two examples:
- **A source term.** Sample it at the vertices (`k = 0`), or integrate it
  over the elements as an n-form, $f \, dx^1 \wedge \dots \wedge dx^n$.
- **A boundary flux.** Take the de Rham map of the flux 1-form, and keep
  its values on the boundary edges. In 2D, the flux of $u$ across an edge
  is $\int u_x \, dy - u_y \, dx$, measured to the right of the edge's
  direction:

```mojo
def flux(x: List[Float64]) raises {u} -> List[Float64]:
    return [-u[1], u[0]]                 # the 1-form whose integral is the flux of u

var g = derham_map(sc, 1, 3, flux)
```

**Looking at a solution.** Evaluate $W c$ at element barycenters
(barycentric coordinates all $1/(n+1)$) to get a vector per element for a
quiver plot or a VTK cell field.

**Measuring discretization error.** For a manufactured solution $u$,
`l2_error(u_h, degree, u)` gives $\|W u_h - u\|$. Comparing it with the
error of `l2_projection` (the best possible) separates the method's error
from the approximation limit; see [Convergence](convergence.md).

**Typed overloads.** `derham_map[k](tc, degree, omega)` returns a
`TypedCochain[n, k, PRIMAL]`, and `whitney_form` takes one.

## Pitfalls

- **Component order and sign.** $[\omega_{xy}, \omega_{xz}, \omega_{yz}]$,
  with $\omega_{xz} = -\omega_{zx}$. A wrong count of components raises;
  a wrong sign does not.
- **Elements keep the mesh's orientation.** Compute top-degree cochains with
  `derham_map` rather than by hand. A hand-made $\int_t \omega$ over the
  sorted vertices is off by $(-1)^{\text{parity}}$.
- **Quadrature degree.** Use at least the polynomial degree of the
  components. For non-polynomial forms, the quadrature error adds to the
  discretization error. In tests, check that it is the smaller of the two.
- **Points on faces.** Evaluating $W c$ on a face shared by two elements
  gives the same tangential part from both, but possibly different normal
  parts. Which element you name matters for the normal part.
- **Cost.** Each `whitney_form` call recomputes the element's barycentric
  gradients and visits all of its k-faces. That is fine for evaluation
  and plotting, but not meant for inner loops of assembly; use
  `whitney_innerproduct` for mass matrices.

## Tests to read

`tests/test_derham.mojo`:
- `test_stokes`: $R d = d R$ for 0-, 1-, and 2-forms on a tetrahedral mesh,
  and 0- and 1-forms on an icosphere; edge rises and element volumes.
- `test_derham_of_whitney_is_identity`: $R W c = c$ for every degree,
  evaluated from every element around each simplex (tangential
  continuity).
- `test_whitney_properties`: partition of unity, $d W = W d$, $W R
  \omega = \omega$ for constant forms.
- `test_mass_matrices_match_whitney_innerproduct`.
- `test_load_projection_and_error`: for constant forms, the load vector is
  $M R\omega$, the projection is $R\omega$, and both errors vanish.
- `test_typed_and_errors`.
