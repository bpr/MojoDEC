# Algebraic multigrid: smoothed aggregation

`SmoothedAggregation(A)` is a preconditioner for sparse symmetric positive
definite matrices whose CG iteration count barely grows with the mesh: a
P1 Laplacian takes 9 or 10 iterations at 4,000 unknowns and at 261,000,
where Jacobi-CG takes 196 and 1485 (10 times the time). It needs only the matrix: no mesh
hierarchy, no geometry.

```mojo
var K = d0.transpose() @ whitney_innerproduct(sc, 1) @ d0   # (interior rows)
var amg = SmoothedAggregation(K)
print(amg.level_sizes(), amg.operator_complexity())
var res = cg(K, b, M=amg, rtol=1e-10)
```

The aggregation and the smoothed prolongation are ports of
[amgcl](https://github.com/ddemidov/amgcl)'s (MIT; notice in `NOTICE`),
and they agree with amgcl's bit for bit (`test_matches_amgcl`). The
smoother is PETSc's choice, Chebyshev in $D^{-1}A$.

## Why multigrid

Jacobi or Gauss–Seidel damp the oscillatory part of the error in a few
sweeps, but the smooth part, which varies over many cells, barely moves:
CG with Jacobi needs a number of iterations proportional to $1/h$ on a
mesh of size $h$, and more on graded meshes (the ratio of the largest to
the smallest cell enters). A smooth error is well represented on a
coarser space, where it looks oscillatory again. Multigrid smooths, moves
the residual to the coarse space, solves there (recursively), and
interpolates the correction back:

1. **pre-smooth:** a few smoothing steps on $A x = b$ from $x = 0$;
2. **restrict:** $r_c = R\,(b - A x)$;
3. **coarse solve:** $e_c = A_c^{-1} r_c$, by the same cycle one level down,
   or factored on the coarsest level;
4. **prolongate:** $x \leftarrow x + P e_c$;
5. **post-smooth:** the smoothing steps again, in mirror order.

That is one **V-cycle**. With $R = P^T$, $A_c = P^T A P$ (the Galerkin
product), and a post-smoother that is the adjoint of the pre-smoother, the
cycle is a symmetric positive definite operator, so CG can use it.

## Building the levels from the matrix

**Strength of connection.** Unknown $j$ is strongly coupled to $i$ when

$$
a_{ij}^2 > \varepsilon^2\, a_{ii}\, a_{jj},
$$

with $\varepsilon$ = `strength` (default 0.08), halved on each coarser
level. Weak couplings are ignored when grouping; rows with no strong
coupling at all (a decoupled or Dirichlet-like row) are left out of every
aggregate, and only the smoother touches them.

**Aggregation.** One greedy pass in row order: each row not yet in an
aggregate, with strong neighbors, starts one, takes its strong neighbors
(even from earlier aggregates), and tentatively its neighbors' strong
neighbors, which a later seed may claim. Aggregates emptied by later
claims are dropped. The pass is serial and depends only on the matrix, so
the hierarchy is the same on any number of threads. It also depends on
the row order: a spatially coherent numbering (`reorder_vertices`'s
Morton order) gives compact aggregates.

**Tentative prolongation.** $P_0$ has one entry, 1, per row, in the
column of the row's aggregate: piecewise constants over the aggregates.
Constants are the near-kernel of a Laplacian, and $P_0$ reproduces them
exactly.

**Smoothed prolongation.** One damped Jacobi step,

$$
P = (I - \omega D_F^{-1} A_F)\, P_0, \qquad \omega = \frac{4/3}{\lambda_{\max}(D^{-1} A)} ,
$$

where $A_F$ keeps $A$'s strong entries and adds the weak ones to the
diagonal (so $A_F$ has $A$'s row sums, and $P$ still reproduces
constants). Smoothing makes the coarse basis functions overlap, which
is what gives SA its convergence; without it, plain aggregation needs
many more iterations. $\lambda_{\max}$ comes from 10 Jacobi-CG steps
(`eigenvalue_range`), or Gershgorin's bound if CG breaks down. amgcl's
default is $\omega = 2/3$ (assuming $\lambda_{\max} \approx 2$); PyAMG
uses $4/3$ over the estimate, as here.

**Coarse operators.** $R = P^T$ and $A_c = P^T A P$. Coarsening stops
at `coarse_size` unknowns (default 1000), at `max_levels`, or when a
level does not shrink; the coarsest matrix is factored by
`SparseCholesky`.

## Smoothers

- **`Relaxation.CHEBYSHEV`** (default): the Chebyshev polynomial of
  degree `degree` (default 2) in $D^{-1} A$ that is smallest on
  $[0.1, 1.1]\,\lambda_{\max}$ (PETSc's interval; the 1.1 covers the
  estimate being low). It damps the upper part of the spectrum and leaves
  the bottom to the coarse grid. Each step is a product with $A$ and
  vector updates, no sweep order, so it parallelizes like the matrix
  product. The same polynomial before and after keeps the cycle
  symmetric.
- **`Relaxation.GAUSS_SEIDEL`**: forward sweeps before the coarse
  correction, backward after: symmetric, about as strong, and serial.

`sweeps` (default 1) applications before and after each correction.

## Measured

One thread, release build, $\mathrm{rtol} = 10^{-8}$, P1 Dirichlet
Laplacian $d_0^T M_1 d_0$ on the interior vertices:

| mesh | unknowns | levels | op. complexity | AMG-CG | setup + solve | Jacobi-CG |
|---|---|---|---|---|---|---|
| Kuhn 64² | 3,969 | 2 | 1.29 | 9 it | 1.0 + 0.9 ms | 196 it, 2.3 ms |
| Kuhn 256² | 65,025 | 4 | 1.33 | 9 it | 10 + 15 ms | 737 it, 140 ms |
| Kuhn 512² | 261,121 | 4 | 1.33 | 10 it | 46 + 75 ms | 1485 it, 1.31 s |
| Kuhn 32³ | 29,791 | 3 | 1.43 | 9 it | 18 + 13 ms | 116 it, 18 ms |
| Kuhn 48³ | 103,823 | 4 | 1.46 | 9 it | 47 + 48 ms | 175 it, 105 ms |
| graded plate, 3 holes | 355,385 | 4 | 1.26 | 10 it | 106 + 132 ms | 732 it, 1.03 s |

Symmetric Gauss–Seidel takes 0–2 iterations more. Operator complexity is
the stored entries of all levels over the finest's: the memory, and the
work of a cycle in products with $A$.

**PyAMG** (5.4.1) on the plate, as an independent check: its defaults
give levels 355,385 / 43,324 / 2,794 / 161 / 7, operator complexity 1.24,
12 CG iterations; with $\theta = 0.08$ and `max_coarse=1000` its second
level has exactly our 43,844 unknowns and it takes 8 iterations. Its
setup and solve took 0.85 s.

In 3D, Jacobi-CG on a uniform grid is already cheap (175 iterations at
100,000 unknowns): AMG's setup about cancels its gain there, and it pays
off on larger or graded meshes, or when the hierarchy serves many
solves.
Where the factor fits, a sparse direct solve (`SparseCholesky`) is
faster still in 2D; AMG's memory is $O(n)$.

## Singular matrices

A Neumann Laplacian has the constants in its kernel. The cycle keeps
them there ($P$ reproduces constants, so $A_c$ has them in its kernel
too), and the coarsest factorization regularizes its zero pivot. Give
`cg` the kernel and it removes the constant part the cycle adds:

```mojo
var res = cg(K, b, M=SmoothedAggregation(K), nullspace=NullSpace.constants(n))
```

## Pitfalls

- **Positive definite only.** The diagonal must be positive (the
  constructor raises otherwise); an indefinite or saddle-point matrix
  needs a block preconditioner with AMG on the definite blocks.
- **Scalar problems.** The tentative prolongation reproduces constants,
  the near-kernel of 0-form Laplacians. For $k$-form Laplacians with
  $k \ge 1$, whose near-kernel is the large range of $d$, plain SA
  converges slowly: that needs auxiliary-space methods (AMS/ADS) or
  geometric multigrid with a patch smoother.
- **Numbering.** Aggregation follows the row order. A random numbering
  gives scattered aggregates and a worse hierarchy; `reorder_vertices`.
- **Small matrices.** At or below `coarse_size` the "hierarchy" is one
  level, the exact factorization: CG converges in one step.
- **Serial setup.** Aggregation, the smoothed prolongation, and
  $P^T A P$ run on one thread; the cycle's products with $A$ go parallel
  above `PARALLEL_NNZ` like any matrix product.

## The API

```mojo
var amg = SmoothedAggregation(
    A,
    strength=0.08,                     # amgcl's eps_strong
    coarse_size=1000,                  # factor at or below this size
    max_levels=25,
    relaxation=Relaxation.CHEBYSHEV,   # or .GAUSS_SEIDEL
    degree=2,                          # Chebyshev degree
    sweeps=1,                          # before and after the correction
)
amg.num_levels()
amg.level_sizes()                      # finest first
amg.operator_complexity()
amg.apply_into(r, z)                   # z = one V-cycle applied to r
```

## Tests to read

- `tests/test_amg.mojo`:
  - `test_matches_amgcl`: aggregates and $P$ equal to amgcl's bit for bit,
    on a 2D Laplacian, a 1-form Hodge matrix (entries of both signs), and
    a random graph with decoupled rows (`tests/data/amgcl/`, with the
    C++ driver that made them);
  - `test_iterations_do_not_grow_with_the_mesh`: both smoothers, three
    refinements, and a 3D grid against Jacobi;
  - `test_cycle_is_symmetric_positive_definite`;
  - `test_neumann_laplacian_with_its_kernel`;
  - `test_small_matrices_are_factored`; `test_errors`.
- `tests/test_properties_robust.mojo`,
  `test_amg_cycle_is_a_valid_preconditioner`: random SPD graph matrices
  coarsened to a few unknowns: the cycle is symmetric and positive
  definite with either smoother, and CG with it reaches the dense
  solution.

## References

- P. Vaněk, J. Mandel, M. Brezina, "Algebraic multigrid by smoothed
  aggregation for second and fourth order elliptic problems", Computing
  56, 179–196, 1996.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM,
  2003: Algorithm 12.1 (Chebyshev acceleration) and chapter 13
  (multigrid).
- D. Demidov, "AMGCL: an efficient, flexible, and extensible algebraic
  multigrid implementation", Lobachevskii J. Math. 40(5), 535–546, 2019.
