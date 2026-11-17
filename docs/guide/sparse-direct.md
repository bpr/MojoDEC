# Sparse direct solvers: AMD and LDLᵀ

`mojodec.routines.sparse_ldlt` factors a sparse symmetric matrix and
solves with it. It suits DEC's mass, stiffness, and saddle-point systems
in 2D, and moderate ones in 3D:

```mojo
var F = SparseCholesky(K)        # K symmetric positive definite
var x = F.solve(b)               # K x = b; further solves are cheap
```

```mojo
var F = SparseLDLT(A)            # A symmetric, possibly indefinite
var x = F.solve(b)
print(F.inertia())               # (positive, negative) eigenvalue counts
```

The code is a port of [faer](https://codeberg.org/sarah-quinones/faer)'s
AMD ordering and its simplicial and supernodal Cholesky (MIT; faer's AMD
translates SuiteSparse's, BSD-3; both notices are in `NOTICE`); the dense
kernels of the supernodal factorization are MojoDEC's own.

## When to use it

| | direct (`SparseCholesky`, `SparseLDLT`) | iterative (`cg`, `minres`) |
|---|---|---|
| cost of the first solve | the factorization: fill-dependent | iterations × products |
| each further solve | two triangular solves | a whole new iteration |
| conditioning | barely matters | iteration count grows with $\kappa$ |
| memory | the factor $L$ | a few vectors |
| saddle points, indefinite | yes, with the right ordering | `minres` with a good block preconditioner |

The factorization pays off when the conditioning is bad, when one matrix
is solved many times (time stepping, inverse iteration), and for
saddle points. Measured on Kuhn grids, one thread, release build, for the
0-form stiffness plus $10^{-3}$ times the identity, to $10^{-10}$
(factor times without AMD; [Supernodes](#supernodes) below):

| mesh | unknowns | nnz(L) | AMD | factor: simplicial | supernodal | solve | Jacobi-CG |
|---|---|---|---|---|---|---|---|
| 300 × 300 | 90,601 | 3.6 M | 18 ms | 112 ms | 37 ms | 1.8 ms | 249 ms, 924 iterations |
| 30 × 30 × 30 | 29,791 | 11.9 M | 12 ms | 2.15 s | 322 ms | 6.3 ms | 40 ms, 230 iterations |

In 2D the factorization wins even for one solve. In 3D the factor fills
in: $O(n^{4/3})$ entries and $O(n^2)$ work for nested-dissection-like
orderings, against $O(n \log n)$ and $O(n^{3/2})$ in 2D. There it pays
when one matrix is solved many times (a 30³ factor is repaid after about
eight CG solves), and multigrid (`SmoothedAggregation`,
`GeometricMultigrid`) is the alternative for one solve.

## The factorization

For symmetric $A$ and a permutation $P$,

$$
P A P^T = L D L^T ,
$$

with $L$ unit lower triangular and $D$ diagonal. Solving $A x = b$ is then
a forward solve with $L$, a division by $D$, and a backward solve with
$L^T$, each touching every entry of $L$ once.

**Elimination tree.** Eliminating column $j$ creates fill in the rows
where column $j$ of $L$ has entries. The parent of $j$ in the elimination
tree is the row of the first entry below the diagonal in column $j$ of
$L$ (Liu, 1990). Row $k$ of $L$ has its entries exactly at the vertices
of the paths from the entries of row $k$ of $A$ (left of the diagonal) up
the tree, stopping at $k$. So the pattern of $L$, and the count of each
column, comes from the tree before any arithmetic.

**Up-looking factorization.** Row $k$ of $L$ solves a sparse triangular
system with the rows above it,

$$
L_{1:k-1,\,1:k-1} \, D_{1:k-1} \, \ell_k = a_{1:k-1,\,k} ,
$$

whose nonzero pattern is that set of paths (the "reach" of row $k$), so
the work is proportional to the flops, not to $n^2$. Then
$d_k = a_{kk} - \sum_j \ell_{kj}^2 d_j$.

**No pivoting.** The ordering is fixed before any number is seen, so a
pivot $d_k$ can be zero or tiny even when $A$ is nonsingular. For
positive definite $A$ that cannot happen: every $d_k > 0$ (they are
ratios of leading principal minors of $P A P^T$). For indefinite $A$ it
depends on the order.

## Supernodes

In the column-by-column factorization each entry of $L$ costs an indexed
load and store, so the arithmetic never reaches the processor's vector
units. But columns of $L$ come in runs with the same pattern: when $j+1$
is $j$'s only child in the elimination tree and column $j$'s pattern is
column $j+1$'s plus $j+1$ itself, the two columns are one dense
trapezoid. A maximal run is a (fundamental) **supernode** (Liu, Ng, and
Peyton, 1993), and $L$ is stored as one dense column-major block per
supernode: its triangle, with $D$ on the diagonal, then the rows of its
shared pattern.

**Relaxed supernodes.** A 3D Kuhn grid's fundamental supernodes are
mostly one or two columns wide. Merging a supernode with its parent in
the tree stores some zeros explicitly but makes bigger blocks
(Ashcraft and Grimes, 1989). In a postorder of the tree, a supernode's
last child comes just before it; the child is merged in when the merged
block's fraction of explicit zeros is at most 1.0 (up to 4 columns), 0.8
(16), 0.1 (48), or 0.05 (any size), faer's defaults. The factor
relabels the columns by that postorder first (`etree_postorder`), which
leaves the fill unchanged; `F.perm` is the relabeled ordering.
`stored_entries()` counts the blocks, zeros included: 15.2 M for the 30³
grid's 11.9 M entries of $L$.

**Left-looking factorization** (Ng and Peyton, 1993). For each
supernode, in order: scatter its columns of $P A P^T$ into its block;
subtract the update $L_b D_d L_m^T$ of each descendant $d$ whose pattern
meets its columns ($L_m$ are $d$'s rows inside the supernode, $L_b$ its
rows from there on), computed as a dense product and scattered by row;
then factor the dense block in place, by panels of 32 columns, each first
updated by all the columns before it in one product. The products run
in register tiles of 8 rows by 4 columns: per step, one vector load and
four fused multiply-adds. The solves gather each supernode's rows once,
so their inner loops run down contiguous columns too.

**Which to use.** `layout=FactorLayout.AUTO` (the default) counts the
flops, $\sum_j (c_j + c_j(c_j+1)/2)$ over the column counts, and goes
supernodal above 20 flops per entry of $L$ (faer's rule, with its 40
lowered to where MojoDEC's kernels measured even). Measured, one thread
(factor times without AMD, then solve):

| matrix | unknowns | flops / nnz(L) | simplicial | supernodal |
|---|---|---|---|---|
| path (1D) | 100,000 | 2.5 | 3.2 ms, 0.67 ms | 5.6 ms, 1.3 ms |
| 2D grid 20 × 20 | 441 | 10 | 0.07 ms | 0.10 ms |
| 2D Whitney 1-forms, 40 × 40 | 4,880 | 15 | 0.82 ms | 0.81 ms |
| 2D grid 60 × 60 | 3,721 | 21 | 1.37 ms | 0.89 ms |
| 2D 1-form mass, 100 × 100 | 30,200 | 35 | 12.0 ms | 5.9 ms |
| 2D grid 300 × 300 | 90,601 | 73 | 112 ms, 4.7 ms | 37 ms, 1.8 ms |
| 3D grid 20³ | 9,261 | 227 | 150 ms, 2.3 ms | 28 ms, 0.49 ms |
| 3D 1-forms 12³ | 13,428 | 306 | 304 ms, 3.7 ms | 55 ms, 0.77 ms |
| 3D grid 30³ | 29,791 | 550 | 2.15 s, 16.6 ms | 322 ms, 6.3 ms |
| Darcy saddle point, 20 × 20 | 2,040 | 258 | 45 ms | 6.6 ms |

The supernodal factor is 3 to 7 times faster on 3D meshes and 2D ones
past a few thousand vertices, and its solves 2 to 4 times. Both layouts
compute the same factorization; they round differently, so solutions
agree to rounding, not bit for bit.

## Orderings

The fill of $L$ depends entirely on $P$. `FillOrdering.AMD` (the default)
is Amestoy, Davis, and Duff's approximate minimum degree: at each step,
eliminate a vertex of nearly least degree in the graph of the remaining
Schur complement. It works on a quotient graph that never forms the fill
explicitly, merges vertices with identical neighborhoods, and bounds
degrees from above instead of computing them, so it costs about as much as
a few matrix-vector products (18 ms for 90,601 unknowns). On the 11 × 11
grid of `test_amd_is_a_permutation_with_zero_diagonals_last` it gives
fewer nonzeros than the natural (banded) order; on the 300 × 300 grid,
3.6 M.

`amd_ordering(A)` returns the permutation itself: row `p[k]` of $A$
becomes row $k$ of $P A P^T$. The port builds AMD's quotient graph in
faer's order, so ties break the same way, and its permutations equal
faer's entry for entry (`test_amd_matches_faer`, against
`tests/data/amd/`).

`FillOrdering.NATURAL` keeps $A$'s order, and `perm=` takes any
permutation, such as one `amd_ordering` already computed.

`cholesky_nonzeros(A, perm)` counts the entries of $L$ from the
elimination tree, before any arithmetic: the factor's memory and, with
it, whether factoring is worth it. The implicit time steppers' default
(`StepSolver.AUTO`, see [Time stepping](time-stepping.md)) factors when
it is at most 24 times $\operatorname{nnz}(A)$, which holds on 2D meshes
and on 3D ones up to about 30³ vertices.

### Saddle points: zero diagonals last

DarcyFlow's mixed system, and every mixed formulation, is

$$
A = \begin{bmatrix} -M & B^T \\ B & 0 \end{bmatrix},
$$

with $M$ symmetric positive definite (a Hodge star or a Whitney mass) and
$B$ of full row rank ($d_1$ on a disk). Ordered as written, eliminating
the first block gives pivots of $-M$'s factorization, all negative, and
leaves the Schur complement $B M^{-1} B^T$, positive definite: no pivot is
zero. Ordered with a constraint row first, the very first pivot is the
zero diagonal entry, and the factorization fails.

So `amd_ordering` puts every row whose diagonal entry is absent or zero
after all others, as Eigen's AMD does, and orders the rest by minimum
degree. Then `SparseLDLT(A)` factors a saddle point with no pivoting and
no regularization, and its inertia is ($c$ positive, $m$ negative), with
$c$ constraints and $m$ unknowns in the first block.

The price is that the constraint block is not reordered for fill: it
is eliminated last, as a dense Schur complement of $c^2/2$ entries. That
is fine for DarcyFlow (a few hundred triangles; 20 × 20 Kuhn cells,
2,040 unknowns, give 400,000 entries of $L$ and factor in 7 ms), and
hopeless past some ten thousand constraints (150 × 150 cells: $10^9$
entries). Its rows are numbered as the input numbers them, which for
cells of a mesh in Morton order (`reorder_vertices`) is spatially
coherent.

## Regularization and refinement

Without pivoting, an indefinite factorization is not backward stable:
eliminating the first block of a saddle point can make the Schur
complement's entries much larger than $A$'s, and rounding grows with
them. So every `solve` ends with iterative refinement against $A$
itself,

$$
x \leftarrow x + F^{-1} (b - A x) ,
$$

`refinement_steps` times (default 1). One step in working precision
restores a componentwise backward error of a few ulps when the factor
is not too unstable (Skeel, 1980), for the price of a product with $A$
and a second pair of triangular solves. On DarcyFlow it takes the
largest pressure error from $2.2 \cdot 10^{-14}$ to $4.4 \cdot 10^{-16}$
(the dense pivoted LU gives $7.8 \cdot 10^{-15}$). `SparseCholesky` sets
it to 0: positive definite factorizations have no growth, and the
preconditioner stays a single linear map.

For an indefinite matrix whose order cannot avoid small pivots, pass the
expected sign of each pivot and a regularization $\delta$:

```mojo
var F = SparseLDLT(A, signs=signs, regularization=1e-10, refinement_steps=3)
```

A pivot $d_k$ of the wrong sign, or with $|d_k| \le \delta$, becomes
$\pm\delta$ (dynamic regularization, as interior-point solvers use). The
factor is then that of a nearby matrix $A + E$, and the refinement steps
correct for $E$ as well; take 2 or 3. `F.num_regularized` counts the
replaced pivots. Without `signs`, a small pivot keeps its sign.

### Through `spsolve`

`spsolve` takes `method=DirectMethod.SPARSE_LDLT` on a `CSRMatrix`, a
`MatrixOp`, and a `BlockOp` (with or without fixed cells, whose reduced
matrix keeps the same rows as columns and so stays symmetric). DarcyFlow
solves its mixed system that way:

```mojo
var solution = spsolve(
    A, zero, values=known, fixed_first=boundary, fixed_second=pinned,
    method=DirectMethod.SPARSE_LDLT,
)
```

The default, `DirectMethod.DENSE_LU`, is the dense pivoted LU, for any
non-singular matrix of a few thousand unknowns.

## The API

```mojo
var F = SparseLDLT(
    A,                                   # both triangles stored
    ordering=FillOrdering.AMD,           # or .NATURAL
    perm=None,                           # or a permutation, replacing it
    signs=None,                          # +-1 per row, for regularization
    regularization=0,                    # delta; 0 turns it off
    refinement_steps=1,                  # per solve; 0 for bare solves
    layout=FactorLayout.AUTO,            # or .SIMPLICIAL, .SUPERNODAL
)
F.solve(b)                # A x = b
F.apply_into(b, x)        # the same, as a LinearOperator
F.inertia()               # Inertia(positive, negative)
F.factor_nonzeros()       # nnz(L), its diagonal included
F.layout                  # the layout used: SIMPLICIAL or SUPERNODAL
F.num_supernodes()        # dense blocks (0 if simplicial)
F.stored_entries()        # values stored, explicit zeros included
F.perm                    # the ordering (postordered if supernodal)
F.num_regularized

cholesky_nonzeros(A, amd_ordering(A))  # nnz(L) before factoring

var C = SparseCholesky(K)  # raises unless every pivot is positive
cg(K, b, M=C)              # an exact preconditioner: one iteration
BlockDiagonal(SparseCholesky(A1), SparseCholesky(A2))
```

### Typed

`TypedSparseLDLT(A)` and `TypedSparseCholesky(A)` factor a typed
`MatrixOp`, and `BlockSparseLDLT(A)` a `BlockOp`; their `solve` takes
cochains of the operator's output kind and gives cochains of its input
kind:

```mojo
var K = d0.T() @ whitney_innerproduct[1](tc) @ d0 + whitney_innerproduct[0](tc)
var C = TypedSparseCholesky(K)           # primal 0 -> dual n
var u = C.solve(f)                       # f: dual n, u: primal 0

var A = BlockOp(W1 * -1.0, d1.T(), d1, tc.zero[0, DUAL, 2, PRIMAL]())
var F = BlockSparseLDLT(A)               # factored once
var x = F.solve(b)                       # CochainPair of flux and pressure
print(F.inertia())                       # (triangles, edges)
```

A symmetric operator built from the complex ($B^T S B$, a mass, a
saddle point of them) takes each kind to its dual kind, primal
$k$-cochains to dual $(n-k)$-cochains or the reverse, and each block
system's equations are of the dual kind of their unknowns. Anything else
does not compile. A cochain from another complex is refused at the
solve. For one solve with some unknowns given, as in DarcyFlow, use
`spsolve(..., fixed_first=, fixed_second=, method=DirectMethod.SPARSE_LDLT)`.

`SparseCholesky` is a `Preconditioner`, so it can be a block of
`BlockDiagonal`: the Arnold–Falk–Winther preconditioner of the mixed
Hodge Laplacian (see [Linear solvers](linear-solvers.md)) with sparse
instead of dense exact blocks.

## Pitfalls

- **Singular matrices.** A Neumann or closed-surface Laplacian is
  singular, so in exact arithmetic its last pivot is zero, and
  `SparseLDLT` raises when it is. Rounding can instead leave a tiny
  nonzero pivot, which is not caught, and the solve then returns a huge
  kernel component. Fix a value (drop a row and column), or use `cg` with
  a `NullSpace`.
- **Symmetry.** Both triangles must be stored, and equal to $10^{-12}$
  of the largest entry; the constructor raises otherwise.
- **Explicit zeros.** A stored zero on the diagonal counts as a zero
  diagonal, and the row is ordered last.
- **3D fill.** The factor of a 3D problem grows fast (11.9 M entries for
  30³ vertices; the supernodal blocks store 15.2 M values, 122 MB, and
  few indices). Watch `factor_nonzeros()` and `stored_entries()`.
- **Serial.** Both factorizations run on one thread.
- **Saddle points past ~10⁴ constraints.** The constraint block is
  factored as a dense Schur complement (see above).

## Tests to read

- `tests/test_sparse_ldlt.mojo`:
  - `test_amd_matches_faer`: permutations equal to faer's, including a
    3D grid that makes AMD compress its workspace and a matrix with a
    dense row;
  - `test_cholesky_solves_dec_systems`: every degree's Hodge stiffness
    plus mass on 2D and 3D grids, against the dense solver, and as an
    exact preconditioner;
  - `test_ldlt_solves_saddle_points_without_pivoting`: the Darcy system
    with the Whitney mass in both block orders, its inertia, and the
    zero pivot of the natural order with constraints first;
  - `test_regularization_and_refinement`;
  - `test_ldlt_inertia_of_a_shifted_laplacian`: the negative pivots of
    $K - s M$ count the eigenvalues below $s$;
  - `test_symbolic_count_and_given_permutation`: `cholesky_nonzeros`
    against the factor, `perm=`, and its errors;
  - `test_supernodal_agrees_with_simplicial`: every degree in 2D and 3D,
    Darcy in both block orders, an indefinite shifted Laplacian: the
    same fill and inertia, solutions to rounding, and `AUTO`'s choice;
  - `test_supernodal_regularization_and_errors`; `test_etree_postorder`;
  - `test_typed_factorizations`: the three typed wrappers against
    `spsolve`, inertia of a shifted typed Laplacian and of the Darcy
    block system, and cochains from another complex refused;
  - `test_errors`.
- `tests/test_properties_robust.mojo`:
  `test_sparse_cholesky_matches_dense_solves` (both orderings, both
  layouts, `nnz(L)` against `cholesky_nonzeros`),
  `test_sparse_ldlt_solves_saddle_points` (random saddle points in either
  block order, either layout), and
  `test_etree_postorder_is_a_postorder` (random forests).

## References

- P. R. Amestoy, T. A. Davis, I. S. Duff, "An approximate minimum degree
  ordering algorithm", SIAM J. Matrix Anal. Appl. 17(4), 886–905, 1996.
- R. D. Skeel, "Iterative refinement implies numerical stability for
  Gaussian elimination", Math. Comp. 35(151), 817–832, 1980.
- J. W. H. Liu, "The role of elimination trees in sparse factorization",
  SIAM J. Matrix Anal. Appl. 11(1), 134–172, 1990.
- T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006,
  chapters 4 (Cholesky) and 7 (orderings).
- J. W. H. Liu, E. G. Ng, B. W. Peyton, "On finding supernodes for
  sparse matrix computations", SIAM J. Matrix Anal. Appl. 14(1),
  242–252, 1993.
- C. Ashcraft, R. Grimes, "The influence of relaxed supernode partitions
  on the multifrontal method", ACM Trans. Math. Softw. 15(4), 291–309,
  1989.
- E. G. Ng, B. W. Peyton, "Block sparse Cholesky algorithms on advanced
  uniprocessor computers", SIAM J. Sci. Comput. 14(5), 1034–1056, 1993.
- faer, `sparse::linalg::amd` and `sparse::linalg::cholesky::simplicial`
  and `::supernodal`.
