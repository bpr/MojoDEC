# Kuhn grids and flat tori

`kuhn_grid` and `flat_torus` (in `mojodec.routines.mesh_generation`) build
structured simplicial meshes in any dimension: a box, and a flat torus.
Their geometry and spectra are known in closed form, which makes them test
problems where the right answer is known exactly. This page explains:
- how the meshes are built;
- why the torus is exactly flat;
- what their circumcentric duals look like;
- why the Whitney stiffness matrix on them is the finite-difference
  Laplacian.

```mojo
from mojodec import SimplicialComplex, kuhn_grid, flat_torus

var box = SimplicialComplex(kuhn_grid([8, 8, 8]))                 # 3072 tetrahedra
var torus = SimplicialComplex(flat_torus([16, 16], [1.0, 2.0]))   # in R^4
print(torus.betti_numbers())                                      # [1, 2, 1]
```

## Kuhn's triangulation

Divide a box into cells of size $h_0 \times \dots \times h_{d-1}$. In a
cell's own coordinates $t_a = (x_a - c_a) / h_a \in [0, 1]$, each
permutation $\pi$ of the axes picks out the region

$$
1 \ge t_{\pi(0)} \ge t_{\pi(1)} \ge \dots \ge t_{\pi(d-1)} \ge 0,
$$

a simplex. Its vertices are the walk from the lowest corner that steps
along axis $\pi(0)$, then $\pi(1)$, and so on to the opposite corner:

$$
v_0 = c, \qquad v_k = v_{k-1} + h_{\pi(k-1)} e_{\pi(k-1)}.
$$

The $d!$ orderings of the coordinates cover the cell without overlap, so
each cell splits into $d!$ simplices of equal volume $V / d!$, with
$V = \prod_a h_a$. This is Kuhn's triangulation, also called Freudenthal's.
Two neighboring cells cut their shared face the same way, because the
face's own Kuhn triangulation is what both induce, so the mesh is
conforming.

**Orientation.** The edge vectors $v_k - v_0$ form a triangular matrix
after permuting columns by $\pi$, so the determinant is
$\operatorname{sign}(\pi) \, V$. `kuhn_grid` lists simplices of odd
permutations with their last two vertices swapped, so every simplex is
positively oriented and the mesh is coherent.

In 2D there are two triangles per square, both cut along the diagonal from
$(0,0)$ to $(1,1)$. These are the same triangles as `simplicial_grid_2d`,
which the tests check.

### Barycentric coordinates

On the simplex of $\pi$, the barycentric coordinates are differences of
consecutive $t$'s:

$$
\lambda_0 = 1 - t_{\pi(0)}, \qquad
\lambda_k = t_{\pi(k-1)} - t_{\pi(k)}, \qquad
\lambda_d = t_{\pi(d-1)},
$$

so their gradients are

$$
\nabla\lambda_0 = -\frac{e_{\pi(0)}}{h_{\pi(0)}}, \qquad
\nabla\lambda_k = \frac{e_{\pi(k-1)}}{h_{\pi(k-1)}} - \frac{e_{\pi(k)}}{h_{\pi(k)}}, \qquad
\nabla\lambda_d = \frac{e_{\pi(d-1)}}{h_{\pi(d-1)}} .
$$

Two gradients share an axis only when their vertices are consecutive in
the walk. Everything below follows from that.

## The circumcentric dual is the dual cube grid

Every vertex of a Kuhn simplex is at the same distance from the cell's
center $m$. Along each axis, $v_k$ sits at the cell's lower or upper face,
$h_a/2$ from $m_a$, so

$$
|v_k - m|^2 = \sum_a \frac{h_a^2}{4} \quad \text{for every } k .
$$

**All $d!$ simplices of a cell have the cell's center as their
circumcenter.** The circumcentric dual mesh is therefore the dual cube
grid, with one dual vertex per cell center. Consequences:

- A simplex lying inside a cell (a face diagonal, a body diagonal, a
  triangle cutting through the cube) has a **zero dual volume**. Its dual
  cell joins circumcenters that coincide. In 2D these are the
  hypotenuses. On a 2×2×2 grid of tetrahedra, 44 of the 98 edges and 48
  of the 120 triangles have zero dual volume.
- Simplices lying on the grid's own faces have the dual volumes of the cube
  grid. An axis edge's dual is the dual cube face through its middle.

The diagonal Hodge star is then zero on the diagonal simplices, and its
inverse is infinite there. MojoDEC builds the complex anyway, but `star`
from dual to primal raises and `sc[k].require_finite_star_inv()` reports
it. On these meshes use the **Whitney inner products**
(`whitney_innerproduct`), which are fine on right-angled simplices.

## The Whitney stiffness is the finite-difference Laplacian

The 0-form stiffness matrix $K = d_0^T W_1 d_0$ is, entry by entry,
$K_{uv} = \int \nabla\phi_u \cdot \nabla\phi_v$ for the hat functions
$\phi$. On a Kuhn grid it is exactly the $(2d+1)$-point finite-difference
Laplacian, scaled by the cell volume:

$$
K_{uv} = \begin{cases}
\displaystyle \sum_a \frac{2V}{h_a^2} & u = v, \\[1ex]
\displaystyle -\frac{V}{h_a^2} & u, v \text{ neighbors along axis } a, \\[1ex]
0 & \text{otherwise, including diagonal neighbors.}
\end{cases}
$$

**Proof sketch.**
- On one simplex, $\int \nabla\lambda_i \cdot \nabla\lambda_j$ is zero
  unless the gradients share an axis, which requires $|i - j| \le 1$.
- For consecutive vertices it is
  $-\frac{V}{d!} \frac{1}{h_a^2}$, $a$ the axis of their step.
- Vertices joined by a diagonal edge are never consecutive, so the
  diagonal couplings vanish.
- An axis edge $[u, u + h_a e_a]$ is a consecutive pair in exactly $d!$
  simplices. They lie in the $2^{d-1}$ cells around it, and, if $u$ is the
  cell's corner plus the steps along a set $S$ of other axes, a path must
  take the $|S|$ axes of $S$ first in any order, then $a$, then the rest:
  $\sum_{s} \binom{d-1}{s} s! \, (d-1-s)! = d!$. So the entry is
  $-V / h_a^2$.
- The diagonal follows because the rows sum to zero (constants are in the
  kernel).

`test_whitney_stiffness_is_finite_differences` checks this entry by entry
in 1, 2, and 3 dimensions with unequal spacings: at interior vertices of
`kuhn_grid`, and at every vertex of `flat_torus`. This is an independent
check of `whitney_innerproduct` and `d`. On these meshes, DEC with Whitney
forms reproduces the classical finite-difference scheme.

The mass matrix $W_0$ is not the FD identity. It is the consistent P1 mass,
coupling each vertex to its axis neighbors and its neighbors along
$(1, 1, \dots)$-type diagonals.

## Flat tori

`flat_torus(dims, lengths)` glues opposite faces of the Kuhn grid: vertex
$i$ along axis $a$ is vertex $i \bmod n_a$. The result is the flat torus
$\mathbb R^d / (L_0 \mathbb Z \times \dots \times L_{d-1} \mathbb Z)$: closed,
orientable, flat, with no boundary.

### Why at least 3 cells per axis

A Kuhn simplex steps each axis at most once, so its vertices use two
consecutive values $i, i+1$ along each axis. With $n_a \ge 3$, these stay
distinct modulo $n_a$ and no two cells produce the same vertex set, so the
quotient is a simplicial complex. With $n_a = 2$, the neighbors $i - 1$ and
$i + 1$ of a vertex are the same vertex. Two different edges would then
join the same pair of vertices, which a simplicial complex cannot
represent. `flat_torus` raises for $n_a < 3$.

### The embedding is exactly flat

A flat torus has no isometric embedding in $\mathbb R^d$, but MojoDEC needs
vertex coordinates. `flat_torus` puts axis $a$ on a circle in its own
coordinate plane $(x_{2a}, x_{2a+1})$ of $\mathbb R^{2d}$:

$$
p(i_0, \dots, i_{d-1}) = \big(r_a \cos(2\pi i_a / n_a),\; r_a \sin(2\pi i_a / n_a)\big)_{a = 0}^{d-1},
\qquad
r_a = \frac{h_a}{2 \sin(\pi / n_a)} .
$$

This is the Clifford embedding. The radius is chosen so that one step
along axis $a$, a chord of its circle, has length exactly $h_a$.

Within one simplex, every edge vector $v_k - v_j$ is a sum of single steps
along distinct axes. Each step is a chord lying in its own plane, and the
planes are orthogonal. So every inner product of edge vectors, and hence
every length, angle, volume, circumcenter, and Whitney inner product, is
that of the flat Kuhn simplex. The simplicial torus is not an
approximation of the flat torus but the flat torus itself, triangulated.
For example, its total volume is $\prod_a L_a$ to rounding. (The torus
lies on a curved surface in $\mathbb R^{2d}$, but DEC only sees each
simplex's intrinsic geometry, which is flat.)

### Topology and harmonic forms

The $d$-torus has Betti numbers $b_k = \binom{d}{k}$:

| $d$ | $b_0, \dots, b_d$ |
|---|---|
| 1 | 1, 1 (a circle) |
| 2 | 1, 2, 1 |
| 3 | 1, 3, 3, 1 |
| 4 | 1, 4, 6, 4, 1 |

The harmonic k-forms are the constant forms $dx^{a_1} \wedge \dots \wedge
dx^{a_k}$. Their de Rham cochains are discretely harmonic. Take the
cochain of $dx^a$, which is $\pm h_a$ on the edges along axis $a$
(signed by direction) and 0 elsewhere:
- it is closed, $d_1 \omega = 0$, since around each triangle the steps
  along $a$ cancel;
- it is coclosed in the Whitney inner product, $d_0^T W_1 \omega = 0$,
  since that is $\int \partial_a \phi_v \, dx = 0$ for every hat function
  on a closed manifold.

The tests check both on a 3-torus, so this mesh exercises the harmonic part
of a Hodge decomposition with a known answer.

### Spectrum

The eigenvalues of the Laplacian on functions are

$$
\lambda_m = (2\pi)^2 \sum_a \frac{m_a^2}{L_a^2}, \qquad m \in \mathbb Z^d,
$$

with eigenfunctions $e^{2\pi i \, m \cdot x / L}$. With Whitney 0-forms (P1
elements), solve $K x = \lambda W_0 x$. Grid translations are symmetries,
so the discrete eigenvectors are sampled Fourier modes.

For a mode along one axis ($m = m_a e_a$), both stencils reduce to the 1-D
ones, and the discrete eigenvalue is exactly

$$
\lambda_h = \frac{6 \, (1 - \cos\theta)}{h_a^2 \, (2 + \cos\theta)},
\qquad \theta = \frac{2\pi m_a h_a}{L_a},
$$

which exceeds $\lambda$ by a relative $\theta^2 / 12 + O(\theta^4)$. This
is the 1-D P1 result, in any dimension, for two reasons:
- **The interpolant is 1-D.** On a Kuhn simplex the barycentric
  coordinates telescope. If the walk steps axis $a$ at step $j$, then
  $\lambda_0 + \dots + \lambda_j = 1 - t_a$. So the P1 interpolant of a
  function of $x_a$ alone is $(1 - t_a) f(0) + t_a f(1)$, the 1-D hat
  interpolant.
- **Each hat contributes equally.** The hat functions of the vertices in
  one slice $x_a = \text{const}$ sum to the 1-D hat in $x_a$, and they are
  translates of each other. So each one integrates against a function of
  $x_a$ like $V / h_a$ times the 1-D hat.

The stiffness and mass symbols are then $V/h_a$ times the 1-D ones,
$2 h_a (1 - \cos\theta)/h_a^2$ and $h_a (2 + \cos\theta)/3$, and their
ratio is $\lambda_h$.

On `flat_torus([16, 16], [1.0, 2.0])`, the lowest nonzero eigenvalue is
$\pi^2$ with multiplicity 2, from $m = (0, \pm 1)$. The computed pair
equals $\lambda_h = 9.99708\ldots$ to $10^{-11}$, 1.3% above
$\pi^2 = 9.8696$.

## Gluing with a flip: the Möbius band and the Klein bottle

Gluing two sides of a 2D grid with a flip instead of a shift gives
non-orientable surfaces. MojoDEC has them as test meshes:

```mojo
var band = mobius_band(12, 2)        # 12 steps around, 2 across, in R^3
var klein = klein_bottle([6, 5])     # periodic in v, flipped in u, in R^4
```

- **`mobius_band(segments, across, width)`** glues the strip at $u = 2\pi$
  with $w \mapsto -w$. Its coordinates are
  $((1 + w\cos\tfrac u2)\cos u, (1 + w \cos\tfrac u2)\sin u, w\sin\tfrac u2)$.
  Its boundary is one circle, twice as long as the band.
- **`klein_bottle([n_u, n_v])`** is the flat torus's grid with the seam at
  $u = 2\pi$ flipped, $v \mapsto -v$. Its coordinates are
  $((2 + \cos v)\cos u, (2 + \cos v)\sin u, \sin v \cos\tfrac u2,
  \sin v\sin\tfrac u2)$, an embedding in $\mathbb R^4$: the half angles
  change sign together with $\sin v$ across the seam, so the map is
  well defined on the quotient.

Every triangle keeps the grid's orientation, so the two triangles at a
flipped seam induce the same orientation on their shared edge. No choice
of signs repairs that, which is what non-orientable means:
- `coherent_orientation` returns `None`;
- `orient_coherently` raises;
- `SimplicialComplex` refuses them with "the mesh is not orientable".

Their topology is available through `AbstractSimplicialComplex`, which
needs no orientation: both have rational Betti numbers $[1, 1, 0]$ and
Euler characteristic 0, and both are homology manifolds. Over
$\mathbb Z/2$, the Klein bottle's would be $[1, 2, 1]$. The rank
computation works modulo a large prime, so it sees the rational ones (see
[Homology](homology.md)).

## Pitfalls

- **Zero dual volumes.** Use `whitney_innerproduct`, not `sc[k].star_inv`
  or the dual-to-primal `star`, which raise. The same goes for anything
  built on them, such as `delta` from dual cochains and the Leapfrog with
  stars as masses.
- **Anisotropy is not a problem.** Unequal $h_a$ keep all the properties
  above, because the Kuhn simplices stay right-angled along the axes.
- **The vertex numbering** has axis 0 varying fastest, as in `cube_grid`:
  vertex $(i_0, \dots, i_{d-1})$ is $\sum_a i_a \prod_{b < a} (n_b + 1)$ in
  `kuhn_grid`, and $\sum_a i_a \prod_{b < a} n_b$ in `flat_torus`.
- **The cost grows like $d!$.** A 4-torus with 3×4×5×6 cells already has
  8,640 simplices. The homology-manifold check, which examines the link of
  every cell, takes about a second on it.
- **The torus's coordinates are in $\mathbb R^{2d}$.** Plot it through its
  grid indices, not its coordinates.

## Tests to read

`tests/test_mesh_generation.mojo`:
- `test_kuhn_grid`: 1–4 dimensions; positive volumes summing to the box;
  the same triangles as `simplicial_grid_2d`; argument errors.
- `test_flat_torus`: total volume $\prod L_a$, edge lengths $h_a$, no
  boundary, $b_k = \binom{d}{k}$ up to 4D.
- `test_whitney_stiffness_is_finite_differences`: the FD equivalence
  above.
- `test_flat_torus_spectrum_and_harmonic_forms`: the closed-form discrete
  eigenvalue, and the constant 1-forms being closed and coclosed.
- `test_non_orientable_surfaces`: the Möbius band and the Klein bottle
  refused by every orientation check; their Betti numbers, Euler
  characteristic, manifold check, and boundaries.
