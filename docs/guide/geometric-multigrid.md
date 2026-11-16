# Geometric multigrid for k-forms

`GeometricMultigrid(A, hierarchy, k)` preconditions a symmetric positive
definite operator on Whitney $k$-forms, such as

$$
A = \alpha M_k + d_k^T M_{k+1} d_k ,
$$

the $H(\mathrm{grad})$ ($k = 0$), $H(\mathrm{curl})$ ($k = 1$), and
$H(\mathrm{div})$ ($k = 2$ in 3D) norms, on a mesh made by uniform
refinement. CG takes 7 to 10 iterations at every mesh size and every
degree, where Jacobi's count doubles with each refinement:

```mojo
var h = refinement_hierarchy(coarse, 6)          # 6 refinements
ref sc = h[len(h) - 1].fine
var A = whitney_innerproduct(sc, 1) + (
    sc[1].d.transpose() @ whitney_innerproduct(sc, 2) @ sc[1].d
)
var mg = GeometricMultigrid(A, h, 1)
var res = cg(A, b, M=mg, rtol=1e-10)             # 9 iterations
```

It is also the scalable block of the Arnold–Falk–Winther preconditioner
for mixed Hodge Laplacians (below). Written from the published
mathematics: Arnold, Falk, and Winther's vertex-star smoother, in the
patch form of PETSc's PCPATCH.

## Why point smoothers fail for $k \ge 1$

Multigrid works when the smoother removes the error the coarse grid cannot
represent. For $k = 0$ that is the oscillatory error, and Jacobi or
Gauss–Seidel remove it. For $k \ge 1$, $d_k$ has a large kernel: every
gradient $d_{k-1} \phi$ of a $(k-1)$-form. On that kernel $A$ is just
$\alpha M_k$, a mass matrix, while on its complement it is dominated by
$d^T M d \sim h^{-2}$. A point smoother scales each unknown by $A$'s
diagonal, which is $O(h^{-2})$, so on the kernel it barely moves: an
oscillatory gradient (a checkerboard $\phi$) is left untouched, and the
coarse grid cannot represent it either. Plain algebraic multigrid fails
the same way: `SmoothedAggregation`, whose coarse spaces reproduce
constants, needs 134, 245, 470, and 975 iterations on the $H(\mathrm{curl})$
operator as the mesh is refined (from 800 to 49,408 unknowns).

## The vertex-star smoother

Arnold, Falk, and Winther (2000) showed that a Schwarz smoother whose
subdomains are **vertex stars** works for every $k$. The patch of vertex
$v$ is the set of $k$-simplices containing $v$; their Whitney forms are
exactly those supported in the star of $v$. That space contains the local
gradients $d_{k-1}$ of the $(k-1)$-forms supported in the star, so the
kernel of $d$ is spanned by patch-local pieces, and solving exactly on
each patch removes the troublesome error. For $k = 0$ each patch is one
vertex, and the smoother is point Gauss–Seidel.

A smoothing step solves, for each patch $p$ with unknowns $I_p$,

$$
A_{pp}\, \delta_p = (b - A x)_{I_p}, \qquad x_{I_p} \leftarrow x_{I_p} + \delta_p ,
$$

with $A_{pp}$ the restriction of $A$ to $I_p$ (6 edges per vertex in 2D,
about 14 edges or 36 faces in 3D), factored once by a dense Cholesky.

- **`PatchSmoother.MULTIPLICATIVE`** (default): patch by patch, each with
  the residual updated by the ones before (block Gauss–Seidel), vertices
  in order before the coarse correction and in reverse after, which keeps
  the cycle symmetric. Serial.
- **`PatchSmoother.ADDITIVE`**: every patch from the same residual, the
  corrections summed and damped by `damping` (default $1/(k+2)$: each
  $k$-simplex lies in $k+1$ stars). Order-free, the parallel form; about
  twice the iterations.

## The levels

`refinement_hierarchy(coarse, m)` refines `m` times (`refine`); level $l$
is the mesh after $l$ refinements. The transfers are the exact Whitney
prolongations `Refinement.prolongation(k)`: the coarse Whitney forms are
fine Whitney forms, so $P$ loses nothing. $R = P^T$, and the coarser
operators are the Galerkin products $P^T A P$; with nested spaces and
exact mass matrices these equal the operators assembled on the coarser
meshes (`test_galerkin_equals_the_coarse_operator`), so the user supplies
only the finest $A$. The coarsest level is factored (`SparseCholesky`).

**Boundary conditions.** With `dirichlet=True`, $A$ acts on the
$k$-simplices off the boundary (tangential trace zero, as for a
perfectly conducting wall), and every level drops its boundary simplices:
a coarse Whitney form with zero tangential trace has a fine counterpart
with zero tangential trace, so $P$ restricted to the interior simplices
is still exact.

## Measured

One thread, release build, CG to $10^{-8}$, refinements of a 2 × 2 (or
2 × 2 × 2) Kuhn grid, $\alpha = 1$; setup + solve:

| problem | unknowns | Jacobi-CG | multiplicative | additive |
|---|---|---|---|---|
| 2D $H(\mathrm{grad})$ | 16,641 | 594 it, 50 ms | 9 it, 37 + 8 ms | 13 it |
| 2D $H(\mathrm{curl})$ | 12,416 | 1395 it, 69 ms | 9 it, 17 + 11 ms | 17 it |
| 2D $H(\mathrm{curl})$ | 49,408 | 2766 it, 517 ms | 9 it, 50 + 36 ms | 17 it |
| 3D $H(\mathrm{grad})$ | 4,913 | 132 it, 3.6 ms | 10 it | 17 it |
| 3D $H(\mathrm{curl})$ | 31,024 | 1033 it, 198 ms | 10 it, 54 + 38 ms | 23 it |
| 3D $H(\mathrm{div})$ | 50,688 | 831 it, 135 ms | 7 it, 113 + 107 ms | 15 it |

The iteration counts are the same at every level measured (2 to 6
refinements in 2D, 2 and 3 in 3D). For $k = n$ the operator is the
Whitney top-form mass alone, diagonal on these meshes: one iteration with
any preconditioner.

## The mixed Hodge Laplacian

Arnold, Falk, and Winther's block-diagonal preconditioner (see
[Linear solvers](linear-solvers.md#block-preconditioners-for-saddle-points))
needs the inverses of $M_{k-1} + d^T M_k d$ and $M_k + d^T M_{k+1} d$.
With a V-cycle for each, it is spectrally equivalent at linear cost:

```mojo
var P = BlockDiagonal(
    GeometricMultigrid(M0 + d0.transpose() @ M1 @ d0, h, 0),
    GeometricMultigrid(M1 + d1.transpose() @ M2 @ d1, h, 1),
)
var res = minres(K, b, M=P, rtol=1e-10)
```

On the mixed 1-form problem MINRES takes 26, 28, 30, 30, 30 iterations
from 289 to 66,049 unknowns (224 ms at the largest), where Jacobi takes
213 to 2602 (1.14 s). Exact blocks (`SparseCholesky`) take 7 iterations,
at a factorization's cost.

## Pitfalls

- **Refined meshes only.** The levels come from `refine`. A mesh from a
  generator such as Gmsh has no hierarchy; refine a coarse version of it,
  or use `SmoothedAggregation` for $k = 0$.
- **Pass the whole hierarchy.** The finest level is `h[-1].fine`; `A`
  must have its size (its interior size with `dirichlet`).
- **Positive definite.** The constructor factors every patch and the
  coarsest operator, and raises if one is not positive definite (a
  Hodge Laplacian without the mass term, $\alpha = 0$, is singular).
- **Cost.** Each multiplicative step reads about $k+1$ rows of $A$ per
  unknown; setup factors one small dense matrix per vertex per level.
  Both are serial.

## The API

```mojo
var h = refinement_hierarchy(coarse, refinements)   # List[Refinement]
var mg = GeometricMultigrid(
    A, h, k,
    dirichlet=False,
    smoother=PatchSmoother.MULTIPLICATIVE,          # or .ADDITIVE
    sweeps=1,                                       # before and after
    damping=None,                                   # additive: 1 / (k + 2)
)
mg.num_levels()
mg.level_sizes()                                    # finest first
mg.apply_into(r, z)                                 # one V-cycle
```

## Tests to read

- `tests/test_gmg.mojo`:
  - `test_iterations_do_not_grow_with_the_mesh_2d`: every degree, both
    smoothers, three refinements, and Jacobi's contrast;
  - `test_three_dimensions`: $H(\mathrm{curl})$ and $H(\mathrm{div})$;
  - `test_dirichlet_and_mass_weights`: interior edges, $\alpha$ from
    $10^{-3}$ to $10^3$;
  - `test_galerkin_equals_the_coarse_operator`;
  - `test_cycle_is_symmetric_positive_definite`;
  - `test_mixed_hodge_laplacian_block_preconditioner`: MINRES with
    multigrid blocks;
  - `test_errors`.
- `tests/test_properties_robust.mojo`,
  `test_geometric_multigrid_is_a_valid_preconditioner`: random coarse
  grids in 2D and 3D, any degree, weight, boundary condition, and
  smoother: the cycle is symmetric positive definite and CG with it
  reaches the direct solution.

## References

- D. N. Arnold, R. S. Falk, R. Winther, "Multigrid in H(div) and
  H(curl)", Numer. Math. 85, 197–217, 2000.
- R. Hiptmair, "Multigrid method for Maxwell's equations", SIAM J. Numer.
  Anal. 36(1), 204–225, 1998.
- P. E. Farrell, M. G. Knepley, L. Mitchell, F. Wechsung, "PCPATCH:
  software for the topological construction of multigrid relaxation
  methods", ACM Trans. Math. Softw. 47(3), 2021.
