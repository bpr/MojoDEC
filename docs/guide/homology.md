# Homology: Betti numbers and manifold checks

Homology counts the holes in a complex: connected pieces, loops that bound
nothing, enclosed voids. In DEC it tells you how many harmonic forms to
expect before you compute any, and whether a mesh is the shape you think
it is. MojoDEC computes Betti numbers of complexes, subcomplexes, and pairs,
and checks whether a complex is a homology manifold.

## Chains, cycles, and boundaries

A k-chain is a formal combination of oriented k-cells, a vector with one
entry per k-cell. The boundary operator $B_k$ (`cc.boundary(k)`, an
$N_{k-1} \times N_k$ matrix) maps k-chains to (k−1)-chains, and the
boundary of a boundary is empty:

$$ B_k B_{k+1} = 0. $$

A **k-cycle** is a chain with no boundary, $B_k c = 0$; a **k-boundary**
is the boundary of something, $c = B_{k+1} s$. Every boundary is a cycle.
The k-th **homology** is the cycles that are not boundaries,

$$ H_k = \ker B_k \,/\, \operatorname{im} B_{k+1}, $$

and the **Betti number** $b_k$ is its dimension. By rank–nullity,

$$ b_k = N_k - \operatorname{rank} B_k - \operatorname{rank} B_{k+1}, $$

with $B_0 = 0$ and $B_{n+1} = 0$. Read them as:

| | counts | disk | circle | sphere | torus surface | solid torus |
|---|---|---|---|---|---|---|
| $b_0$ | connected pieces | 1 | 1 | 1 | 1 | 1 |
| $b_1$ | independent loops (tunnels) | 0 | 1 | 0 | 2 | 1 |
| $b_2$ | enclosed voids | 0 | — | 1 | 1 | 0 |

Their alternating sum is the Euler characteristic, whatever the complex:
$\sum_k (-1)^k b_k = \sum_k (-1)^k N_k = \chi$.

## Why it matters for DEC

By the discrete Hodge theorem, the harmonic k-cochains (those in the kernel
of the Hodge Laplacian `laplace_derham`) form a space of dimension $b_k$ on
a closed mesh, as long as the Hodge stars are positive (so they are inner
products; a well-centered mesh guarantees it). So:

- the Hodge decomposition of a 1-cochain on a square with four holes has a
  four-dimensional harmonic part (`b_1 = 4`; the HodgeDecomposition
  example);
- an eigensolver for the 1-form Laplacian on a torus should return
  exactly two zero eigenvalues;
- a Poisson problem on 0-cochains has one constant in its kernel per
  connected piece ($b_0$).

With a boundary, the boundary conditions decide which homology counts:
harmonic forms whose tangential part vanishes on the boundary (Dirichlet,
"relative" conditions) number $b_k(K, \partial K)$, and those whose normal
part vanishes (Neumann, "absolute") number $b_k(K)$. See *relative
homology* below.

## Computing Betti numbers

Every complex type has the methods: `SimplicialComplex`,
`RegularCubeComplex`, `AbstractSimplicialComplex`, and `RipsComplex`.

```mojo
var sc = SimplicialComplex(read_mesh("torus_tet.xml"))
print(sc.betti_numbers())                    # [1, 1, 0, 0]
print(sc.relative_betti_numbers())           # [0, 0, 1, 1]
print(sc.betti_numbers(sc.boundary_cells())) # [1, 2, 1, 0]: a torus surface
```

The free functions in `mojodec.routines.homology` take a `ChainComplex`
(`sc.chain_complex()`, or one you build from your own boundary matrices):

| Function | Gives |
|---|---|
| `betti_numbers(cc)` | $b_0, \dots, b_n$ of the whole complex |
| `betti_numbers(cc, cells)` | the Betti numbers of a subcomplex |
| `relative_betti_numbers(cc, sub)` | the Betti numbers of the pair $(K, L)$ |
| `non_manifold_cells(cc)` | the cells whose links fail the manifold test |

Each returns a `List[Int]` of length n + 1 (n the dimension of the whole
complex, even for a lower-dimensional subcomplex).

### Subcomplexes

`betti_numbers(cells)` takes a `CellSet` that must be **closed**: every face
of a cell in it is in it. Otherwise it is not a complex and has no
homology, and the call raises "the cells are not a subcomplex".
`closure_of(cx, cells)` (or `sc.closure(cells)`) makes any set closed; a
link, a closure, and `boundary_cells()` already are.

```mojo
var lk = sc.link(Simplex([v]))
print(sc.betti_numbers(lk))     # [1, 0, 1, 0] inside a tetrahedral mesh: a sphere
                                # [1, 0, 0, 0] on its boundary: a disk
```

### Relative homology

The homology of a pair $(K, L)$, $L$ a subcomplex of $K$, treats every
chain on $L$ as zero: the chain groups are $C_k(K)/C_k(L)$, which in
practice means deleting the cells of $L$ from both the rows and the columns
of every $B_k$. A relative cycle may have its boundary on $L$, so an arc
from one boundary component to another is a relative 1-cycle.

- `relative_betti_numbers()` is relative to the whole boundary,
  `boundary_cells()`.
- `relative_betti_numbers(sub)` is relative to any subcomplex, such as the
  part of the boundary carrying Dirichlet data (a Gmsh physical group,
  closed with `closure_of`).

For an orientable n-manifold $K$ with boundary, **Lefschetz duality**
relates the two kinds:

$$ b_k(K, \partial K) = b_{n-k}(K). $$

The solid torus above shows it: $[0, 0, 1, 1]$ is $[1, 1, 0, 0]$ reversed.
A disk relative to its boundary circle is $[0, 0, 1]$, a sphere: the one
2-cycle is the whole disk, whose boundary lies on $L$.

## Coefficients: the rationals

The Betti numbers are those of homology with **rational** (equivalently,
real) coefficients, which is what counts harmonic forms. Other coefficients
can differ. The 6-vertex projective plane, an `AbstractSimplicialComplex`
since it is not orientable, has Betti numbers $[1, 0, 0]$ over the
rationals but $[1, 1, 1]$ modulo 2, because its integral homology has
torsion: $H_1(\mathbb{RP}^2; \mathbb{Z}) = \mathbb{Z}/2$. The test suite
checks the rational answer.

### How the ranks are computed

The ranks of the boundary matrices are computed exactly, by sparse
Gaussian elimination modulo the prime $p = 1073741789$, the largest below
$2^{30}$. Residues are below $2^{30}$, so every product fits in 64 bits
and no big integers are needed. The elimination is column reduction as in
persistent homology: each column of $B_k$ is reduced against stored pivot
columns by its last nonzero row, and its pivot is stored when it survives.

The rank modulo $p$ equals the rational rank unless $p$ divides one of the
invariant factors of $B_k$ (the diagonal of its Smith normal form). The
invariant factors larger than 1 are the torsion coefficients of
$H_{k-1}(K; \mathbb{Z})$, so the answer is exact unless the complex has a
torsion group of order divisible by a 30-bit prime. Meshes of real objects
have no torsion at all, and the classic examples (projective spaces, lens
spaces) have small orders like 2 or 3.

Only the cells involved are touched, so the cost follows the size of the
complex or subcomplex: a 20,000-triangle sphere takes about 15 ms, and a
link a few microseconds.

The boundary entries must be integers (they are $\pm 1$ for every complex
MojoDEC builds); a `ChainComplex` with other entries raises.

### `ChainComplex` is checked when built

Homology is only defined when $B_k B_{k+1} = 0$. The `ChainComplex`
constructor raises unless that holds (to rounding, for real entries) and
each coboundary $d_k$ you pass is exactly $B_{k+1}^T$.

## Homology manifolds

A triangulated n-manifold looks the same near every point: a small
neighborhood is a ball, or a half-ball at the boundary. Combinatorially,
the neighborhood of a cell $\sigma$ is the join of $\sigma$ with its
**link**, the faces of cells around $\sigma$ that do not touch it (see the
neighborhoods in `mojodec.routines.topology`). In a manifold, the link of a
k-cell is an (n−k−1)-sphere inside and an (n−k−1)-ball on the boundary.

Recognizing a sphere is undecidable in dimension 5 and up (and hard
below), so the practical test compares homology. A complex is a **homology
n-manifold** (with boundary) when, for every k-cell $\sigma$ with $k < n$:

- $\sigma$ inside: the link has the rational homology of
  $S^{n-k-1}$, Betti numbers $[1, 0, \dots, 0, 1, 0, \dots]$ with the second
  1 in position $n - k - 1$ (for $n - k - 1 = 0$, the 0-sphere, two points:
  $b_0 = 2$);
- $\sigma$ on the boundary: the link has the homology of a point,
  $[1, 0, \dots, 0]$.

Every triangulated manifold passes; most broken meshes fail somewhere.

```mojo
var bad = sc.non_manifold_cells()     # a CellSet; empty for a manifold
if not sc.is_homology_manifold():
    print(bad.indices())              # per dimension, the failing cells
```

What gets flagged:

| Mesh | Flagged | Why |
|---|---|---|
| two triangles meeting at a vertex | the vertex | its link is two separate edges, not a path |
| two tetrahedra meeting at a vertex | the vertex | its link is two triangles, not a disk |
| two tetrahedra sharing only an edge | the edge | its link is two points apart, but on the boundary it should be one |
| three triangles on one edge | the edge | its link has three points |
| a triangle with a dangling edge | the edge's cells | the edge has an empty link |

### Why every dimension, not just vertices

Checking only vertex links misses pinches from dimension 3 on. Two
tetrahedra $[0,1,2,3]$ and $[0,1,4,5]$ share only the edge $[0, 1]$. The link
of vertex 0 is the two triangles $[1,2,3]$ and $[1,4,5]$, joined at vertex 1:
connected and contractible, exactly what a boundary vertex needs. Only the
link of the edge, the two separate edges $[2,3]$ and $[4,5]$, shows the
pinch. `non_manifold_cells` checks the link of every cell of dimension
below n (the links of n-cells are empty, which is always right).

### Not a constructor requirement

Unlike orientation, being a manifold is not required to build a complex.
A mesh pinched at a vertex, like two cones meeting at their tips, is a
coherently oriented pseudomanifold whose DEC operators are well defined; it
is just not a manifold. Checking every link also costs more than a build
should (about 30 ms for a 2,500-vertex sphere's vertices and edges). Call
it when the answer matters: before trusting a scanned mesh, or before
relying on Lefschetz duality or the harmonic-form counts above.

The test does not depend on orientation: the projective plane passes.

## Tests and examples to read

- `tests/test_homology.mojo`: Lefschetz duality on the solid torus, links
  (spheres inside, disks on the boundary, circles on a surface), the
  projective plane, cube complexes with holes, a Rips circle, the pinched
  meshes above, and timings on 20,000-triangle meshes.
- The Neighborhoods3D example computes links on the tetrahedral solid torus.
