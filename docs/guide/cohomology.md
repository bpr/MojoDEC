# Cohomology bases and harmonic forms

A Hodge decomposition splits a cochain into an exact part, a coexact part,
and a **harmonic** part. The harmonic part is the piece that sees the
topology: on a square with four holes, the harmonic 1-cochains are the
flows that circulate around the holes. [Homology](homology.md) counts
them, $b_k$ of them in degree k. This page is about computing them:
`cohomology_basis` finds one integer cocycle per class, and
`harmonic_basis` turns each into a harmonic cochain.

```mojo
from mojodec import cohomology_basis, harmonic_basis, whitney_innerproduct

var basis = cohomology_basis(sc, 1)   # b_1 integer cocycles and dual cycles
var H = harmonic_basis(sc, 1)         # harmonic in the diagonal Hodge star
var Hw = harmonic_basis(sc, 1, whitney_innerproduct(sc, 1))  # or in Whitney's
var p = basis.periods(H[0])           # [1, 0, 0, 0]
```

## Cocycles, cycles, and periods

A k-cochain $z$ is a **cocycle** (closed) when $dz = 0$; it is a
**coboundary** (exact) when $z = dp$ for a (k-1)-cochain $p$. Since
$d \circ d = 0$, coboundaries are cocycles. The k-th cohomology is the
quotient: cocycles, with two counted as the same **class** when they
differ by a coboundary. Over the reals its dimension is the Betti number
$b_k$, the same as that of homology.

Dually, a k-chain $c$ is a **cycle** when $\partial c = 0$. Pairing a
cochain with a chain, $\langle z, c\rangle = \sum_\sigma z_\sigma
c_\sigma$, is "integrating $z$ over $c$". This integral is the cochain's
**period** over the cycle. For a cycle, it depends only on the class of $z$,
by the discrete Stokes theorem:

$$
\langle z + dp, c \rangle = \langle z, c\rangle + \langle p, \partial c \rangle = \langle z, c \rangle .
$$

The converse is de Rham's theorem in its discrete form: a cocycle whose
periods over all cycles vanish is a coboundary. So a closed cochain's
class is fixed by its periods over a basis of cycles. In the plane with
holes, these are the circulations around the holes.

## Finding integer cocycles

`cohomology_basis(sc, k)` uses two exact integer column reductions. They
are the same lowest-pivot elimination that persistent homology uses.

1. **The gauge.** Reduce the boundary operator on k-simplices (column
   $\sigma$ lists $\partial\sigma$). The pivot columns $G$ are a column
   basis. Read the other way round, the k-simplices in $G$ are where
   $d p$ can be prescribed freely: for every cocycle $z$ there is exactly
   one coboundary $dp$ that agrees with it on $G$. So every class has
   exactly one representative that vanishes on $G$, and it is enough to
   look for cocycles among the other k-simplices, $N$.
2. **The cocycles.** Reduce the coboundary operator restricted to $N$
   (column $f$ lists the (k+1)-simplices that have $f$ as a face). Each
   column reduced to zero gives a combination of columns with zero
   coboundary: a cocycle $z_f$, with value 1 on $f$, 0 on the other
   zero columns, and support in $N$. There are exactly $b_k$ of them.

The dual cycle of $z_f$ is $c_f = f - \sum_{g \in G} y_g\, g$, the
combination of $G$ with the same boundary as $f$. The first reduction
records it. The cocycle $z_i$ is supported on $N$, with value 1 on $f_i$
and 0 on the other zero columns. The cycle $c_j$ is supported on
$\{f_j\} \cup G$. So their pairing is

$$
\langle z_i, c_j \rangle = z_i(f_j) = \delta_{ij},
$$

exactly, in integers. Hence the cocycles represent independent classes
and span the cohomology, and the cycles do the same for homology.

On a surface with $k = 1$ this is the classic **tree–cotree**
construction:
- $G$ is a spanning tree of the edges;
- the second reduction builds a spanning tree of the triangles across the
  remaining edges;
- each edge left over closes a loop around one handle or hole: $c_f$ is
  that loop, and $z_f$ is a "cut" crossing it once.

For $k = 0$ the cocycles are the indicator functions of the connected
pieces. For the top degree of a closed orientable manifold, the one cycle
is the whole oriented manifold.

### Integer pivots

Every step subtracts an integer multiple of an earlier column, which
needs the pivot to divide the entry it cancels. The pivots of a simplicial
complex's boundary operators are almost always $\pm 1$. On every complex
tried (flat tori in 2D and 3D, a solid torus, a sphere, squares with holes,
the HodgeDecomposition mesh) all cocycle and cycle entries are $0$ or
$\pm 1$. If a fraction would be needed (torsion, or an unlucky order),
`cohomology_basis` raises rather than return rational cocycles. It also
raises if entries grow past $2^{40}$.

The cost is small: the reductions of a 40 × 40 flat torus (4800 edges)
take 35 ms for $k = 1$, and the edges of an 8 × 8 × 8 flat 3-torus 20 ms.
The count needs no tolerance. It is the number of zero columns in exact
arithmetic, and the tests check it against `betti_numbers()`.

## From a cocycle to a harmonic cochain

With an inner product $M$ on k-cochains, a cochain is **harmonic** when
it is closed, $dh = 0$, and coclosed, $d^T M h = 0$ (orthogonal to every
coboundary; with $M$ the Hodge star this is $\delta h = 0$).
`harmonic_basis` keeps the class and removes the exact part. It sets
$h = z - dp$, with $p$ chosen so that $dp$ is the M-orthogonal projection
of $z$ onto the coboundaries:

$$
(d^T M d)\, p = d^T M z .
$$

Then:
- **$h$ is closed**, since $dh = dz - ddp = 0$. In floating point it is
  zero to rounding (about $10^{-16}$).
- **$h$ is coclosed**: $d^T M h = d^T M z - (d^T M d)p$ is the solve's
  residual. MINRES runs to `rtol = 1e-12`.
- **$h$ keeps the periods**: $\langle h, c_j \rangle = \langle z, c_j
  \rangle = \delta_{ij}$, by the Stokes identity above.

When $M$ is positive definite, these properties fix $h$. A closed and
coclosed cochain with zero periods is exact, $h = dq$, and then
$q^T d^T M d q = 0$ forces $h = 0$. So the basis does not depend on which
cocycles the elimination picked, only on the cycles: **$h_i$ is the
harmonic cochain with period 1 around cycle $i$ and 0 around the
others.** It is not orthonormal; orthonormalize it in $M$ if you need
that.

The matrix $d^T M d$ is singular: it vanishes on the closed (k-1)-cochains,
such as the constants for $k = 1$. The right-hand side lies in its range,
so the system is consistent. MINRES with a Jacobi preconditioner
$1/|a_{ii}|$ handles that. It also handles an indefinite Hodge star (see
the pitfalls). For $k = 0$ there is nothing to remove: the component
indicators are already harmonic.

### Which inner product

- **`harmonic_basis(sc, k)`** uses the diagonal Hodge star `sc[k].star`.
  It matches DEC's codifferential `delta`, so `delta(h)` vanishes to
  rounding.
- **`harmonic_basis(sc, k, M)`** takes any symmetric mass matrix, such as
  `whitney_innerproduct(sc, k)`. It is the right choice when the rest of
  the computation uses Whitney forms, and on meshes whose circumcentric
  dual degenerates (Kuhn grids).

The two bases have the same periods but are different cochains, except
where both inner products make the same forms harmonic. On a flat torus
both give exactly the constant forms $dx^a$, which the tests check.

### The harmonic part of a closed cochain

Since the periods determine the class, a closed cochain $\omega$ has the
harmonic part

$$
\omega_H = \sum_j \langle \omega, c_j \rangle\, h_j ,
$$

and `basis.periods(omega)` gives the coefficients. It needs no further
solve. This holds only for closed $\omega$. For a general cochain, the
coexact part also has periods, and the harmonic part comes from a full
Hodge decomposition.

## Dirichlet conditions: relative cohomology

The forms above satisfy the natural (Neumann) condition at a boundary,
$d^T M h = 0$ on every row. They count $b_k(K)$ and circulate around holes.
Harmonic forms with **Dirichlet** conditions instead vanish on the
boundary k-simplices. They count $b_k(K, \partial K)$, which by Lefschetz
duality is $b_{n-k}(K)$ for an orientable manifold. For a domain in the
plane, $b_1(K, \partial K)$ is again the number of holes, but the forms
are different: they flow *from* one boundary piece *to* another.

Every function above takes a subcomplex $L$, usually
`sc.boundary_cells()`:

```mojo
var bd = sc.boundary_cells()
var rel = cohomology_basis(sc, 1, bd)        # b_1(K, dK) relative cocycles
var D = harmonic_basis(sc, 1, bd)            # Dirichlet-harmonic, star
var Dw = harmonic_basis(sc, 1, whitney_innerproduct(sc, 1), bd)
```

**Relative cochains** vanish on $L$. Since $L$ is closed under taking
faces, $d$ maps them to relative cochains, and the elimination is the
same with the cells of $L$ removed from both reductions:
- a **relative cocycle** is zero on $L$ and closed;
- a **relative cycle** is a chain off $L$ whose boundary lies in $L$, such
  as a path from one boundary piece to another.

The pairing is again exactly the identity.

**The harmonic representative** removes $dp$ with $p$ also vanishing on
$L$. It solves $(d^T M d)\, p = d^T M z$ only on the rows and columns of
the (k-1)-simplices off $L$. So $h$ is zero on $L$, closed, and coclosed
at every (k-1)-simplex off $L$. At the boundary itself nothing is imposed,
which is what a Dirichlet condition means.

For the plane with holes and $k = 1$, these are exactly the gradients
$h = du$ of the functions $u$ that are discrete harmonic inside, 1 on the
boundary of one hole, and 0 on the other boundary pieces. The tests check
that the Whitney basis spans these to $10^{-10}$. In the top degree,
$b_n(K, \partial K) = b_0(K)$: one area form per connected piece, and with
the star it is the area form over the total area.

## Typed API

```mojo
var tc = TypedComplex[2](sc)
var H = harmonic_basis[1](tc)                          # star
var Hw = harmonic_basis(tc, whitney_innerproduct[1](tc))
```

Both give `List[TypedCochain[n, k, PRIMAL]]`, and both take a
subcomplex after the other arguments for the relative versions:
`harmonic_basis[1](tc, bd)`, `harmonic_basis(tc, W1, bd)`. The mass matrix must have
the Hodge star's kind, from primal k-cochains to dual (n-k)-cochains, and
must be built on the same complex.

## The HodgeDecomposition example

The example removes four triangles from a square mesh, so $b_1 = 4$.
PyDEC's driver decomposes four random cochains and orthogonalizes their
harmonic parts. MojoDEC's driver takes `harmonic_basis(sc, 1)` instead and
prints for each form:
- the periods around the four holes, the identity to $10^{-16}$;
- $|dh| \approx 2 \times 10^{-16}$ and $|\delta h| \approx 7 \times
  10^{-11}$.

It still decomposes one random cochain the PyDEC way. Its harmonic part is
closed, so it should equal $\sum_j p_j h_j$ with its own periods $p_j$. It
does to $2 \times 10^{-7}$, the accuracy of that decomposition's CG and
LSQR solves. The figures show each form circulating once around its own
hole.

## Pitfalls

- **Which boundary condition.** Without a subcomplex, the forms satisfy
  the natural (Neumann) condition and circulate around holes. With
  `sc.boundary_cells()`, they vanish on the boundary and flow between
  boundary pieces. Pass a different subcomplex for mixed conditions, such
  as only the outer boundary.
- **An indefinite star.** On a mesh that is not well-centered, some
  entries of `sc[k].star` can be negative. Then the star is not an inner
  product, and "harmonic" is with respect to an indefinite form. MINRES
  still solves the system when it is consistent, as on the
  HodgeDecomposition mesh. If $d^T M d$ has gained extra null vectors, it
  may not be, and `harmonic_basis` raises when MINRES stops. Use
  `whitney_innerproduct`, which is always positive definite.
- **Zero diagonal.** The Jacobi preconditioner raises on a zero diagonal
  entry of $d^T M d$. That happens for a (k-1)-simplex that is a face of
  no k-simplex, or whose k-simplices all have zero star (a Kuhn grid's
  diagonal edges have zero dual volume, but every vertex also has axis
  edges).
- **The basis depends on the simplex order** through the cycles the
  elimination picks. The span does not, and neither does the harmonic
  part of a given cochain.

## Tests to read

`tests/test_cohomology.mojo`:
- `test_cocycles_and_cycles`: on two- and three-dimensional tori, a solid
  torus, a sphere, a square with two holes, and two disjoint squares, the
  counts equal `betti_numbers()`. Every entry is $0$ or $\pm 1$,
  $dz = 0$ and $\partial c = 0$ exactly, and the pairing is exactly the
  identity.
- `test_constant_forms_on_flat_torus`: on a 3 × 4 × 5 flat torus, the
  Whitney and star harmonic 1-forms both span exactly the constant forms
  and agree to $10^{-12}$; the harmonic 2-forms are closed, coclosed, and
  have identity periods.
- `test_top_degree_is_the_volume_form`: the star-harmonic 2-cochain of a
  closed surface is the area form over the total area, and the 0-forms
  are the component indicators.
- `test_square_with_holes`: both inner products, periods, and the periods
  of a cocycle plus a coboundary.
- `test_relative_counts_and_duality`: on holed and disconnected squares,
  a solid torus, and a cube, the relative counts equal
  `relative_betti_numbers()` and $b_{n-k}$. The cocycles vanish on the
  boundary, the relative cycles have their boundary in it, and the
  pairing is the identity.
- `test_dirichlet_harmonic_forms`: the Dirichlet 1-forms against $du$,
  zero on boundary edges, coclosed inside; the relative 2-form is the
  area form.
- `test_typed` and `test_errors`, including a set that is not closed
  under faces.
