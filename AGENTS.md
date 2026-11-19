# AGENTS.md

Project context for AI coding assistants working on MojoDEC. Tool-specific
workflow instructions live in each tool's own file, e.g.
CLAUDE.md.

## What this is

MojoDEC is largely inspired by PyDEC (Discrete Exterior Calculus). It runs
on Mojo 1.3, using
its own `RealArray` (real vectors and matrices) and NuMojo integer arrays
in place of PyDEC's NumPy arrays, and its own `CSRMatrix` in place of
PyDEC's `scipy.sparse` matrices. README.md has the full MojoDEC ↔
PyDEC file mapping, the list of deliberate differences from PyDEC, and the
features added beyond PyDEC.

## Commands

All tooling runs through pixi (linux-64 only):

```sh
pixi install                          # Mojo 1.3 nightly + MAX, Python, NumPy, matplotlib
pixi run numojo                       # precompile vendored NuMojo -> build/numojo.mojoc
pixi run test                         # every tests/test_*.mojo (builds numojo first)
pixi run examples                     # all examples; figures saved as PNG
pixi run examples DarcyFlow --show    # one example, figures in windows
pixi run test-release                 # tests with -D ASSERT=none (also examples-release)
pixi run bench                        # timings of the parallel loops (release build)
pixi run bench-compare                # vs PyDEC, dexterior, formoniq (benchmarks/compare)
pixi run package                      # build/mojodec.mojoc
pixi run format                       # mojo format mojodec tests examples
pixi run doc-check                    # doc string warnings vs docs/docstring-baseline.txt
pixi run docs                         # API reference -> docs/api/ (Markdown, not committed)
pixi run docs-site                    # MkDocs site -> build/site (pixi's `docs` env)
```

Run a single test file from the repo root (after `pixi run numojo`):

```sh
pixi run mojo run -I . -I build tests/test_cochain.mojo
```

The property tests need proptest at `../proptest`; set `PROPTEST_PATH` to
use another checkout.

## Project layout

Layout follows NuMojo: `mojodec/core` holds data structures,
`mojodec/routines` holds algorithms, `mojodec/vis` holds plotting.
`mojodec/__init__.mojo` re-exports the public API (add new public names
there); `mojodec/prelude.mojo` is the small common subset.

- `tests/`: one `test_*.mojo` per area; mesh fixtures in `tests/meshes/`.
- `examples/`: each example runs from its own directory
  (`examples/run.sh` cds into it) and writes outputs that `.gitignore`
  excludes.
- `third_party/numojo`: vendored NuMojo (see Toolchain).
- `benchmarks/`: timings of the parallel loops; `benchmarks/compare/`
  times MojoDEC against PyDEC, dexterior, and formoniq on shared meshes
  (Rust crate pinned by git rev, PyDEC cloned and patched into
  `build/compare`, scipy from the `bench` pixi environment).
- `docs/`: doc string guide, doc-check baseline, MkDocs sources.
- `conda.recipe/`: rattler-build recipe and package smoke test; see
  `docs/publishing.md` and `docs/community-edition.md` (the submission
  checklist).

## Tech stack and toolchain

- `pixi.toml` asks for `mojo >=1.3.0.dev0` and `max-core >=26.8.0.dev0`
  (plain `>=1.3` would exclude the 1.3 nightlies, which conda orders before
  1.3.0); `pixi.lock` records the build actually tested. Don't pin exact
  versions in `pixi.toml` except temporarily while testing one. `max-core`
  is needed by the vendored NuMojo (`max.algorithm`, `max.gpu`); MojoDEC
  uses `max.algorithm.parallelize` only through `core/parallel.mojo`.
  `benchmarks/parallel_for.mojo` is a pure-Mojo POSIX-threads baseline to
  time it against.
- `-D ASSERT=none` removes `debug_assert` bounds checks. `mojo run`/`build`
  accept it (the `*-release` tasks pass it via `MOJO_FLAGS`); `mojo
  precompile` does not, so it applies only to code compiled from source.
- NuMojo v0.12.0 is vendored in `third_party/numojo` with a Mojo 1.3 patch
  (`third_party/numojo-mojo-1.3.patch`; see `third_party/README.md`) because
  the published package requires Mojo < 1.3. Code is compiled against
  `build/numojo.mojoc`, so rebuild it after touching the vendored copy, and
  keep the patch file in sync with any change there.
- NuMojo's `linalg.solve` (no pivoting) and Jacobi `eigh` are unsuitable for
  DEC systems; use MojoDEC's own `routines/dense.mojo` (pivoted LU,
  Householder/QL `eigh`) and `routines/linalg_core.mojo` (`cg`, `minres`,
  `lsqr`, dense `spsolve`, `eigh_generalized`).

### Known toolchain bugs

NuMojo bugs MojoDEC has found are written up, with repros for its
maintainers, in `third_party/numojo-bugs.md`; add new ones there.

- Write floats as text with `io/scan.mojo`'s `write_float`, not
  `String(x)`: Mojo prints about one in five integral doubles between
  ~1.8e16 and 1e23 with a digit too few, so they don't read back, and
  some Float32 values too (widen them to Float64 first). Parse integer
  text with `atol`, not `atof`, which loses digits past 2^53.
- `nm.fromstring` (and `real_array_from_string`, which parses with it)
  reads `1e4` as 14 (it drops the `e`); write test literals without
  exponents or use `real_array_from_rows`.
- `assert_almost_equal` has a default `rtol=1e-5` on top of `atol`, so
  every call passes `rtol` explicitly (normally `rtol=0`, with an `atol`
  from the problem's conditioning); `tests/run_all.sh` fails on a call
  without one.
- NuMojo's elementwise array arithmetic was slow (0.63 ms for 4,000
  entries, 50 times the sparse product itself). Real arrays are now
  `RealArray`, whose `+`, `-`, and scaling are plain loops, but each
  still allocates its result: in hot loops, write the loop over
  `unsafe_get`/`unsafe_set` (the AMG cycle went from 3 ms to 0.09).
- Write `sqrt(x)` (`std.math`), never `x ** 0.5`: Mojo's `**` with a
  fractional exponent is off by up to ~5e-12 relative, which Cholesky and
  Gram-matrix solves amplify (it cost DarcyFlow two digits and Sphere's
  spectrum check three).
- Mojo's float operations allow contraction, so LLVM may fuse `a * b + c`
  into one rounding, or not, depending on how the code around it is
  laid out: the dual walk's loops fused all three of their multiply-adds,
  and the same arithmetic unrolled fused different ones and changed the
  last bits. Where two code paths must agree bit for bit (the dual walk's
  `_walk` and `_walk_soa`), write the multiply-adds as explicit
  `x.fma(y, acc)`.

## Architecture

`docs/architecture.md` is the overview, with the complex's memory and
build times. `docs/plan.md` lists future ideas and rejected designs;
check it before proposing a design change. The complex caches nothing:
operators other than `d`, `boundary`, and the stars are functions whose
results the caller owns (caching Whitney sparsity was rejected, see the
plan). Key design points that span several files:

- **Linear algebra boundary** (to become a package of its own):
  `core/{sparse,operator,diagonal,parallel,utils,index_set}.mojo` and
  `routines/{dense,linalg_core,sparse_ldlt_core,supernodal,ordering,lanczos_core,amg}.mojo`
  import nothing else from MojoDEC: no complexes, cochains, subsets, or
  typed layer, and no NuMojo: real vectors and dense matrices are
  `RealArray` (`core/real_array.mojo`: C-contiguous `Float64`, rank 1 or
  2, NuMojo's method names for flat access, `+`/`-`/scaling entrywise,
  `to_numpy`), which the rest of MojoDEC uses too (`utils` re-exports
  it, with `real_vector`/`real_array`; `from_ndarray`/`to_ndarray` and
  `real_array_from_string` convert to and from NuMojo, which integer
  arrays still are). `read_array(path)` and `loadtxt(path)` without a
  dtype give a `RealArray`; with one, a NuMojo array of that dtype.
  `CSRMatrix` and `DiagonalOperator`'s
  `exclude_subset`/`restrict_to_subset` take any `IndexSet`
  (`core/index_set.mojo`: `universe_size()` and `in`), which `Subset`
  implements. The typed overloads stay in `routines/linalg.mojo`,
  `lanczos.mojo`, and `sparse_ldlt.mojo`, which re-export the core
  modules' names and define the overloaded ones (`cg`, `minres`,
  `jacobi`, `lsqr`, `spsolve`, `eigh_generalized`, `eigsh`) again as
  forwarders: Mojo does not merge overloads of one name imported from
  two modules (the second import is ambiguous). Put new solver code in
  the core modules, importing nothing from the DEC side; a forwarder's
  signature must match its core function's.

- **`SimplicialComplex`** (`core/simplicial_complex.mojo`) stores its
  `ComplexData` (mesh + one `SimplexLevel` per dimension) behind an
  `ArcPointer`, so copies and cochains share one complex like Python
  references. `sc[k]` gives a `SimplexLevel` with simplices, parity,
  `d`/`boundary` `CSRMatrix`es, `simplex_to_index(s)`/`index_to_simplex(i)`
  lookups (binary search over the sorted rows, no per-level dictionary), and geometry
  (circumcenters, primal/dual volumes, diagonal Hodge `star`/`star_inv`).
  The constructor raises unless the elements form a coherently
  oriented pseudomanifold (`require_coherent_orientation` in
  `routines/simplex_array.mojo`, on the top-level boundary);
  `orient_coherently(mesh)` fixes an orientable mesh, so tests of
  non-manifold or non-orientable input use `coherent_orientation` on the
  element array instead.
  Geometry is computed eagerly at construction, unlike PyDEC's lazy
  `__getattr__`; `compute_geometry=False` skips it for degenerate meshes.
  `simplex_array_boundary` builds `d` directly (k+1 entries per row) and
  `boundary` is its transpose. Circumcenters are closed-form up to
  triangles and for tetrahedra in 3D (cross products, each weight from
  its own face's normal); other cases solve PyDEC's system with positions
  relative to the first vertex, and from triangles up the Cartesian center
  is built relative to it too (absolute positions lose precision far from
  the origin). Embedded volumes use Householder QR
  (`parallelotope_volume`), or for triangles in 3D the cross product
  `|u x w|` (its components are 2 x 2 minors, as accurate as QR; the
  triangle circumcenter's area uses it too), never the Gram determinant;
  tetrahedra in 3D take the triple product. The Whitney top-form mass
  (QR gradients) and the primal volume therefore agree to the
  conditioning `eps L / h`, not to the last bit. Barycentric
  gradients use QR too (`dual_basis`, `frame_dual_basis` in `core/utils.mojo`),
  never `(E E^T)^-1 E`. Dual volumes come from one walk down the chains of
  circumcenters (`_add_dual_pieces`; `compute_dual_volume` runs it as
  `_DualWalk`, breadth first and four chains per SIMD vector in
  `_walk_soa` for triangles in 2D and 3D and tetrahedra in 3D, the
  recursive `_walk` otherwise, the same sums bit for bit; check a change
  there against the committed build on jittered meshes, serial and
  parallel, by the bits of every level's dual volumes). `_run_fixed`
  gathers one top simplex ahead into `_FixedTables`, laid out so that
  each lane vector is one or two vector loads and a shuffle (`_lanes`,
  `_face_slot`); the scatter sums each face's pieces in a register, in
  chain order. Its two sets of tables are separate allocations: in one,
  the walk took 1.3x as long. The next simplex is gathered in three
  parts (`_gather_part`) between `_walk_soa`'s levels, where the
  processor would otherwise idle on the square roots and divisions, and
  the divisions by 3! are `_div6` (a multiplication and two fused
  multiply-adds, correctly rounded for magnitudes in [2^-1000, 2^1000];
  others, zeros included, are divided);
  `dual_edge_piece_values` runs the
  same walk to keep the edge pieces per top simplex, which `flat` and
  `dual_edge_pieces` use. In loops, look faces up with
  `level.find_vertices(sorted_list)`, not `simplex_to_index(Simplex(...))`.
- **Cochains** (`core/cochain.mojo`): PyDEC's dual-typed `cochain.v` is split
  into `Cochain` (a `RealArray`) and `CochainBasis` (`CSRMatrix`), both
  implementing the `CochainLike` trait. `d`, `star`, `delta`,
  `laplace_derham`, `laplace_beltrami` are generic over `CochainLike`.
- **Operators**: on `CSRMatrix`, `A @ B` / `A @ x` are matrix products
  (PyDEC's `A * B` on SciPy matrices); `A * s` scales. Hodge stars
  (`sc[k].star`, `star_inv`) are `DiagonalOperator` (`core/diagonal.mojo`):
  `D @ A` scales rows, `A @ D` (via `__rmatmul__`) scales columns;
  `to_csr()` when a `CSRMatrix` is required. `CochainLike.mapped` is overloaded for both.
- **`Primality`** (`core/primality.mojo`): `.PRIMAL`/`.DUAL` in place of
  PyDEC's `is_primal` Bool, on cochains, `get_cochain(_basis)`, and
  `Subset` (read with `.primality()`; `p.level(n, k)` gives the level).
  The typed layer keeps `primal: Bool` parameters with `PRIMAL`/`DUAL`
  Bool aliases, because Mojo 1.3 does not fold `p.dual()` in a parameter
  (`star`'s result type would not match `DUAL`); convert with
  `Primality(primal)` at the untyped boundary.
- **`Subset`** (`core/subset.mojo`): bitset of primal k-simplices or dual
  k-cells (indexed by primal level k or n-k). Boundary/coboundary walk the
  rows of `sc[m-1].d` / `sc[m+1].boundary`, swapped for dual cells.
  `CSRMatrix`/`DiagonalOperator` have `exclude_subset`/`restrict_to_subset`,
  `Cochain.overwrite`, `sc.boundary_cells()[k]`. `subset.mojo` and
  `sparse.mojo` import each other, as do `cochain` and `simplicial_complex`.
- **`CellSet`** (`core/cell_set.mojo`): one primal `Subset` per dimension
  `0..n`; the neighborhood queries take and return it. They are generic
  over the `CellComplex` trait (`num_cells`, `faces_of`, `cofaces_of`,
  `boundary_faces_of`), which the four complex types and `ChainComplex`
  implement by walking incidence rows (`cells_in_rows`), so nothing is
  copied per query. A trait cannot return a reference to a field in Mojo
  1.3, hence these value-returning primitives.
- **`ChainComplex`** (`core/chain_complex.mojo`): `boundary(k)` = `B_k`
  (k = 1..n) and `coboundary(k)` = `d_k` (k = 0..n-1), no PyDEC padding.
  Abstract and Rips complexes store one and return it by reference from
  `chain_complex()`; `SimplicialComplex` and `RegularCubeComplex` keep
  `sc[k].boundary`/`sc[k].d` per level and build a copy.
- **Typed layer** (`core/typed.mojo`): `TypedComplex[n]` wraps a
  `SimplicialComplex` (`.sc`); `TypedCochain[n, k, primal]`,
  `MatrixOp`/`DiagonalOp[n, in_k, in_primal, out_k, out_primal]` wrap
  `RealArray`/`CSRMatrix`/`DiagonalOperator`. Composition is checked by
  `__matmul__` signatures; degree ranges by `comptime assert`. Ops built
  by `TypedComplex` record its complex (`Optional`, `None` when wrapped
  from a bare matrix) and refuse cochains and ops from another one.
  `TypedSubset[n, k, primal]` types subsets the same way; typed `wedge`,
  `flat`, `whitney_innerproduct[k](tc)`, solver (`cg`, `lsqr`,
  `spsolve`, `eigh_generalized` -> `SolveResult[TypedCochain[...]]`,
  `TypedEighResult`) and `GmshMesh.subset[k](tc, ...)` overloads live next
  to the untyped ones. A typed solve takes `A: in -> out` and `b: out` and
  gives `x: in`. `MatrixOp`/`DiagonalOp.T()` maps `(k, p) -> (l, q)` to
  `(n-l, !q) -> (n-k, !p)`; build symmetric operators as `B.T() @ S @ B`
  rather than with a dual `d`, which carries a sign. `visit_typed[min_n, max_n](sc, visitor)` turns a runtime
  dimension into the parameter of a `TypedComplexVisitor.visit[n]`; it
  compiles `visit` for every n in the range, so its `comptime assert`s
  must hold for all of them. `core/typed_block.mojo` has `BlockOp` (2 x 2
  typed blocks, zero blocks from `tc.zero[...]()`) and `CochainPair`;
  its `spsolve` overloads are in `routines/linalg.mojo`. Mojo
  simplifies parameter expressions (`n - (n - k + 1)` = `k - 1`), but
  `comptime if k == 0` does not refine `k` inside the branch, so write
  branches in terms of `k`. Keep it a thin layer: operators are built
  from the untyped levels and must match `d`/`star`/`delta`.
- Other complex types (`AbstractSimplicialComplex`, `RipsComplex`,
  `RegularCubeComplex`) expose the same neighborhood API
  (`star`/`closure`/`link`/`boundary`/`interior`, `euler_characteristic`),
  implemented on top of `routines/topology.mojo`, and the same
  `betti_numbers`/`relative_betti_numbers`/`non_manifold_cells`/
  `is_homology_manifold` methods, which pass
  `chain_complex()` to `routines/homology.mojo` (sparse column reduction
  modulo a 30-bit prime; `_betti_of` touches only the given cells, so
  per-cell link checks cost the size of the link). `ChainComplex` checks `B_k B_{k+1} = 0` and
  `d_k = B_{k+1}^T` when built. `len(cells)` on a `CellSet` counts cells;
  `len(cells.levels)` is the number of dimensions.
- **Operators and Krylov solvers** (`core/operator.mojo`,
  `routines/linalg_core.mojo`; typed overloads in `routines/linalg.mojo`): `LinearOperator` is `num_rows`/`num_cols`/
  `apply_into(x, mut y)` on `RealArray`; `CSRMatrix`, `DiagonalOperator`,
  `Identity`, `MatrixOp`, `DiagonalOp`, and `BlockOp` (applied block by
  block) implement it. `Preconditioner` is a method-less marker for
  symmetric positive definite operators (`DiagonalOperator`, `DiagonalOp`,
  `Identity`; never a general `CSRMatrix`). `BlockDiagonal(P1, P2)`
  (`core/operator.mojo`) stacks two for 2 x 2 systems; `DenseCholesky`
  (`routines/linalg_core.mojo`) is an exact dense inverse for small problems. `cg` and `minres` are generic
  over both, with `M=` keyword-only; the unpreconditioned overloads pass
  `Identity`, which reproduces the old iterates bit for bit. Both raise on
  a non-symmetric operator (`_require_symmetric`, two probe products) and
  stop with `BREAKDOWN` on `r^T M r < 0`. Typed preconditioners map `A`'s
  output kind back to its input kind (`jacobi(A)` does). Keyword options
  on both (and their typed `MatrixOp` overloads, not the `BlockOp` one):
  `x0=` (tolerance stays relative to `||b||`), `nullspace=` (a
  `NullSpace`, orthonormalized by Gram–Schmidt twice and shared behind an
  `ArcPointer`; removed from `b`, `x0`, and every preconditioned residual
  or Lanczos vector, and checked against `||A q|| <= sqrt(eps) ||A||`
  with `||A||` from the symmetry probes), and `ritz=True` (the Ritz
  values of the Lanczos tridiagonal, from CG's `alpha`/`beta` or MINRES's
  `alfa`/`beta`, by `dense.tridiagonal_eigenvalues`, QL without vectors).
  `eigenvalue_range` runs 10 CG steps on `probe_vector(n, 3)` for the
  extreme Ritz values (inner bounds). Without the options, every solve is
  bit for bit what it was before them; keep it so.
  `eigsh(K, M, nev, sigma=)` (`routines/lanczos_core.mojo`, from the papers:
  Ericsson–Ruhe, Wu–Simon) factors `K - sigma M` once (`SparseLDLT`)
  and runs thick-restart Lanczos on `OP = (K - sigma M)^-1 M` in the `M`
  inner product (basis and `M V` stored, full reorthogonalization twice,
  `H` dense, Ritz pairs by `dense.eigh`, accepted at `tol |theta|`).
  `_run` finds the `want` pairs of largest `|theta|` in the complement
  of a locked set; `eigsh` runs it once for `nev`, then repeatedly with
  the found pairs locked (`_PROBE = 4`) and swaps in anything nearer
  sigma (`_merge`), because single-vector Lanczos misses copies of
  multiple eigenvalues (the icosphere did). `deflate=G` projects every
  vector `M`-orthogonally off `range(G)` with `SparseCholesky(G^T M G)`.
  Eigenvectors come back `n x nev`; `residuals` and `mode` take `n x m`.
  `EighResult.residuals(K, M)` (and `TypedEighResult`'s) gives each
  pair's normwise backward error with norms from 30 power steps (lower
  bounds, so the errors err high); dense arrays are wrapped in a private
  `_Dense` operator. Pass sparse `K` to keep it at n sparse products.
- **Sparse direct** (`routines/ordering.mojo`, `routines/sparse_ldlt_core.mojo`
  (typed wrappers in `routines/sparse_ldlt.mojo`),
  `routines/supernodal.mojo`; ports of faer's AMD and simplicial and
  supernodal Cholesky, notices in `NOTICE`):
  `amd_ordering(A)` builds AMD's quotient graph exactly as faer's
  `aat`/`amd_1` do from the sorted pattern of `A + A^T`, so its
  permutations equal faer's (`tests/data/amd/`, generated by the Rust
  driver there); rows with an absent or zero diagonal are ordered last
  (treated as AMD's dense rows), which makes saddle points factor
  without pivoting. `SparseLDLT` checks symmetry entrywise (`1e-12` of
  the largest entry), takes the lower triangle of `P A P^T`, computes the
  elimination tree and column counts, then each row of `L` by `_ereach`
  (stack order is a valid topological order); `D[j]` is stored in place
  of `L`'s unit diagonal. Dynamic regularization (`signs`,
  `regularization`) as faer's. Every `solve` takes `refinement_steps`
  (default 1) of iterative refinement against a stored copy of `A`: the
  unpivoted indefinite factor is not backward stable, and DarcyFlow's
  pressures need the step to agree to 1e-14. `perm=` replaces the
  ordering; `cholesky_nonzeros(A, perm)` runs only the symbolic phase
  (`_PermutedLower`, `_etree_and_counts`, shared with the factor).
  `TypedSparseLDLT`, `TypedSparseCholesky` (`MatrixOp`) and
  `BlockSparseLDLT` (`BlockOp`, assembled by `to_csr`) wrap them,
  `comptime assert` that each output kind is the dual of its input kind,
  and refuse cochains from another complex at `solve`.
  `SparseCholesky` wraps it
  with no refinement, requires positive pivots, and is a
  `Preconditioner`. Simplicial factors and all solves are serial.
  `spsolve(..., method=DirectMethod.SPARSE_LDLT)`
  (CSR, `MatrixOp`, `BlockOp`, fixed cells) uses it; DarcyFlow does.
  `layout=FactorLayout.AUTO` goes supernodal above 20 flops per entry
  of `L` (`_SUPERNODAL_RATIO`, measured break-even; faer uses 40): the
  permutation is relabeled by `etree_postorder` (same fill), and
  `_Supernodal` finds fundamental supernodes, merges each with the
  supernode before it in the postorder under faer's relaxation
  thresholds, finds patterns as faer's `ereach_super`, and factors
  left-looking: per supernode, scatter `C`, subtract each descendant's
  `L_b D L_m^T` (`_Frame.update`: a dense product into scratch,
  scattered by `rmap`), then the dense block by panels of 32 columns
  (`panel_update`: the columns before the panel in one product;
  `factor_diagonal`: the panel's diagonal block, left-looking, with the
  pivot rule; `panel_eliminate`: the rows below it). `_sub_product` is
  the one kernel: register tiles of 8 rows x 4 columns, lower trapezoid
  (`top` offsets the diagonal for blocks of rows below it). The numeric
  phase is a task list (`_Tasks`) run in one `parallel_ranges` call on
  the physical cores, above `_PARALLEL_WORK` (5e6 multiply-adds; serial
  below): Geist-Ng subtrees (heaviest first), then each supernode above
  them as one task, or, past `_SPLIT_WORK`, column windows (each applies
  every descendant's update to its columns), then per panel a diagonal
  task and `_ROW_BLOCK` (64) row tasks; a panel's diagonal task waits
  only for the row block holding its rows (and the panel two back,
  whose coefficient buffer it reuses). Threads claim tasks from an
  atomic counter in list order and spin on each one's pending count
  (`Atomic[Int64]` statics on untracked pointers); predecessors come
  first, so a waiting task's predecessors are held by running threads
  and it cannot deadlock. A call per panel instead cost 225 us each, 50
  ms at 30^3. In 3D under AMD the subtrees hold ~3% of the work: the
  split supernodes are what scale. Bits equal serial at every thread
  count (row and column splits keep each entry's operations and order);
  check a change with `test_parallel_supernodal_matches_serial`. A failed
  pivot sets a flag every thread polls, then the whole factorization
  reruns serially so the error is serial's (first in supernode order).
  `vals` is allocated uninitialized and each task zeroes the block or
  columns it assembles (parallel first touch). The frame's pointers are
  untracked: keep the lists they point into alive (`_ = x^`) past the
  parallel call. Solves gather each supernode's rows once. `solve(B)`
  (n x m) is `_solve_many`: `_solve_block(c_lo, c_hi)` per block of
  columns does everything for them: gathers `B` by columns (permuted),
  `_solve_columns(Y, c_lo, c_hi)` (each supernode, or simplicial column,
  applied to every column in turn; `_solve_factored` is its one-column
  case), scatters, and per refinement step the residuals (the loop of
  `CSRMatrix.matvec_into`, same sums), the solve, and `x += dx`; so
  column c is the single solve bit for bit. The supernodal
  `solve_columns` gathers each supernode's rows below its triangle for
  all the block's columns (`_gather`, deg x mb by columns) and multiplies
  as matrices: `_sub_product` (the factor's kernel, `W` = the
  supernode's solved rows, row-major) going down, `_sub_dots` (2 x 2
  tiles of `L` columns by `Y` columns, `_W` lanes then `reduce_add`,
  odd ones by `_dot_lanes`) going up. The triangles take groups of 4
  columns of `Y`: down by blocks of `_W` rows, the columns of `L`
  before a block by `_down_tiles` (8 x 4 register tiles starting from
  `Y`, a zero `x_j` skipped per column as the single solve skips it),
  then the block's own triangle; up by `_up_triangle4` (one load of `L`
  for four dots); the other `mb % 4` columns column by column (a
  one-column tile reads `L` across its rows, slower than the sweep down
  each column). Every entry's chain is fixed whatever `mb` or tile, so keep
  explicit `fma`s there (contraction differs between tile and tail
  code). `mb == 1` (single solves, and a core's one-column block) has
  its own loops with the same chains, built column by column of `L`
  (`_down_triangle`, `_up_triangle`, `_dot_lanes`) without `W` or
  `_sub_product`: through the matrix kernels a single solve was 20-28%
  slower. Above `_PARALLEL_SOLVE_WORK`
  ((stored + steps (stored + nnz A)) x m = 4e6) the blocks go to the
  physical cores in one `parallel_ranges` call. 30^3, m = 64, no
  refinement (i7-10875H): 915 ms as single solves, 304 ms one thread,
  53 ms on 8. The typed wrappers overload `solve` on a `List` of
  cochains (`BlockSparseLDLT`: of `CochainPair`s), packed into one n x m
  solve (`_typed_solve_many`, `_columns_of`, `_column`).
  3-7x faster factors than simplicial in 3D on one thread, about 4x more
  on 8 cores; results differ from simplicial in the last bits. The
  supernodal path relabels `etree` and `counts` by the postorder instead
  of recomputing them (an equivalent ordering). `_etree_and_counts` is
  Liu's elimination tree (ancestors path-compressed, in `visited`) and
  Gilbert-Ng-Peyton column counts (row-subtree leaves in a postorder,
  least common ancestors by a disjoint-set forest; bookkeeping as faer's
  `column_counts_aat`), nearly linear in nnz(C): 2.2 ms at 30^3 where
  walking every path of L took 13; `test_column_counts_match_dense_elimination`
  checks it against dense symbolic elimination.
- **Algebraic multigrid** (`routines/amg.mojo`): `SmoothedAggregation`
  ports amgcl's `plain_aggregates` (`_aggregates`: strong when
  `a_ij^2 > eps^2 a_ii a_jj`, eps halved per level, one serial greedy
  pass in row order, rows without strong neighbors left out, id `-1`)
  and its smoothed prolongation (`_smoothed_prolongation`: weak entries
  lumped into the diagonal, `-omega * (1 / dia)` as amgcl rounds it,
  rows sorted), equal to amgcl bit for bit (`tests/data/amgcl/`, C++
  driver there; keep it so). `omega = 4/3 / lambda_max(D^-1 A)` from
  `eigenvalue_range` (Gershgorin if CG breaks down); `R = P^T`,
  `A_c = R @ (A @ P)`; coarsest at `coarse_size` by `SparseCholesky`,
  regularized `SparseLDLT` if singular. V-cycle from zero: Chebyshev on
  `[0.1, 1.1] lambda` (Saad 12.1) or forward/backward Gauss-Seidel.
  Residuals by loops (`_residual`), not `RealArray`'s `-`.
- **Geometric multigrid** (`routines/gmg.mojo`, from the papers):
  `GeometricMultigrid(A, hierarchy, k)` takes the finest operator and a
  `List[Refinement]` (`refinement_hierarchy`), checks the chain
  (`coarse.is_same(previous.fine)`), and builds levels finest first:
  `P = prolongation(k)` (`select`ed to interior simplices with
  `dirichlet`), `R = P^T`, `A_c = R @ (A @ P)`; coarsest by
  `SparseCholesky`. `_Patches`: per vertex, the free k-simplices
  containing it (counting sort, so each patch is sorted), with the dense
  Cholesky of `A` on them stored row-major. Multiplicative smoothing
  visits vertices forward before the coarse correction and backward
  after (symmetric cycle); additive sums all patches from one residual,
  damped by `1/(k+2)`. Serial.
- **Kuhn grids and flat tori** (`routines/mesh_generation.mojo`):
  `kuhn_grid(dims[, lengths])` cuts each cell into d! simplices (one per
  axis permutation, odd ones with the last two vertices swapped);
  `flat_torus` is the same grid glued periodically (at least 3 cells per
  axis) and placed in R^(2d), each axis on a circle of radius
  `h / (2 sin(pi / n))` so the chords have length h and the complex is
  exactly flat. All simplices of a cell share the cell center as
  circumcenter, so diagonal simplices have zero dual volume: test with
  `whitney_innerproduct`, not `star_inv`. `mobius_band` and
  `klein_bottle` (2D grids glued with a flip) are non-orientable test
  meshes: `SimplicialComplex` refuses them; use
  `AbstractSimplicialComplex`.
- **Quadrature** (`routines/quadrature.mojo`): `grundmann_moeller(n,
  degree)` gives barycentric nodes and weights summing to 1 (degree
  rounded up to odd, capped at `MAX_DEGREE = 21` because the negative
  weights' cancellation costs digits); integrands are
  `def(List[Float64]) raises -> Float64` functions or closures taking
  Cartesian coordinates. `integrate_simplices` integrates densities
  (unsigned volume), not k-forms.
- **de Rham map and Whitney forms** (`routines/derham.mojo`): forms are
  component lists over increasing index sets in lexicographic order
  (`form_index_sets`). Both maps orient k-simplices by sorted vertices
  but the elements by the mesh (`simplex_parity`, negated when 1), as
  the cochains of `sc[k]` are; hand-written top-degree integrals must do
  the same. `tests/test_derham.mojo` checks R d = d R, R W = id from every
  element of a face, d W = W d, and the mass matrices against
  `whitney_innerproduct`.
  `_WhitneyElement` precomputes an element's Whitney basis (wedge minors
  of the gradients, signs) for `whitney_form`, `whitney_load`, and
  `l2_error`; `l2_projection` solves `M c = b` with Jacobi-PCG and raises
  if CG stops. `tests/test_convergence.mojo` holds the rate tests
  (thresholds just below the measured rates in `docs/guide/convergence.md`).
- **Cohomology** (`routines/cohomology.mojo`): `cohomology_basis` runs
  two exact integer column reductions (lowest pivot, as persistence):
  the boundary of the k-simplices gives the gauge set `G` (and, tracked,
  the dual cycles), the coboundary of the rest gives the cocycles. It
  raises if a pivot does not divide (torsion) instead of going rational.
  `harmonic_basis` solves `d^T M d p = d^T M z` with Jacobi-MINRES
  (singular, consistent; the star may be indefinite) and returns
  `z - d p`; the periods stay the identity. The `relative_to` overloads
  (Dirichlet) drop the subcomplex's cells from both reductions and solve
  for `p` only off it.
- **Refinement** (`routines/refinement.mojo`): `refine` splits each
  element in its sorted vertex order by the Kuhn simplices of the doubled
  simplex (`_template`, vertex codes `i (n+1) + j` for the midpoint of
  `i`, `j`); fine vertices are numbered in lexicographic order of their
  coarse pairs `(a, a)`/`(a, b)`, which makes sorted order the inherited
  (Bey) order, so shapes stay bounded; don't renumber them old-first.
  `prolongation(k)` integrates coarse Whitney forms over fine simplices
  from barycentric coordinates alone, in integers (`_whitney_integral`).
- **Time stepping** (`routines/time.mojo`): `Leapfrog` takes diagonal
  masses `Mq`, `Mp` and one coupling `K` and steps `Mq q' = K p`,
  `Mp p' = -K^T q`, so the system is skew by construction; it raises on
  non-positive masses (stars on meshes that are not well-centered) and on
  `dt` at or above the stability limit (power iteration). `TypedLeapfrog`
  infers its kinds from the operators and checks them with
  `comptime assert`. `ImplicitMidpoint` (any SPD `Mq`, diagonal `Mp`)
  and `CrankNicolson` assemble one SPD matrix `S`.
  `solver=StepSolver.AUTO` (default) counts `nnz(L)` under AMD
  (`cholesky_nonzeros`, symbolic only) and factors `S` once
  (`SparseCholesky`, reusing the permutation) when it is at most
  `_FACTOR_FILL = 24` times `nnz(S)` (Kuhn grids: 2D 2.4-5.7, 3D 16^3
  9.1, 28^3 22.8, 32^3 28; break-even about 28 with the parallel supernodal
  factor on 8 cores, 26 on one),
  recording the choice in `.solver`; `CHOLESKY` and `CG`
  force one. CG solves each step with Jacobi-PCG for the increment (the tolerance is relative to the change,
  not the state), started from Fischer's guess (`_Guess`: an
  `S`-orthonormal basis of the last 16 solutions, the projection of the
  new one; never worse than zero in the `S` norm), so a step's bits
  depend on the history, its value only within `rtol`.
- **`mojodec.vis`** uses matplotlib via Python interop and is intentionally
  *not* re-exported from `mojodec`, so non-plotting programs never load
  Python. `pyplot(interactive=show_requested())` uses Agg unless `--show`
  is passed. `simplex_quivers`/`cube_quivers` are pure Mojo.
- I/O (`routines/io/`): PyDEC array/mesh formats, VTK `.vtu`
  (`VTKFormat.ASCII`/`.BINARY`/`.APPENDED`, pure Mojo), PLY, and Gmsh `.msh` input
  (4.1 and 2.2, ASCII and binary; physical groups -> `GmshMesh.subset`).
  Formats are small structs with `comptime` members (`ArrayFormat`,
  `SparseFormat`, `PlyFormat`, `VTKFormat`), passed as `format=.BINARY`.
  Readers go through `io/scan.mojo`'s `ByteReader` (in-place number
  parsing, block copies of binary data); check a count read from a file
  against the bytes left before allocating for it, with
  `require_items(count, size)` rather than `require(count * size)`, which
  can overflow. File bytes need not be UTF-8: never build a `String` or
  `StringSlice` from them with `unsafe_from_utf8` (an assertion, or worse,
  on invalid input); use `from_utf8_lossy` for text and messages and the
  raising `from_utf8` where invalid bytes are an error (number fallbacks,
  Gmsh names), and search for ASCII markers by byte (`_find_byte`).
  Never index a table by a number read from a file unless
  the table is bounded by the file's size (Gmsh node tags index a table
  only when they span less than 4 times their number, and are hashed
  otherwise), and bound sizes the file cannot vouch for (`read_sparse`'s
  `max_dim`). `ByteReader` reads the file with its size given (8x faster
  than reading to the end) and parses numbers in one pass, decimals by
  Clinger's exact fast path with `atof` for the rest; keep parsed values
  equal to `atof`'s bit for bit (`tests/test_io.mojo` checks). A loop per
  digit beat SWAR (8 digits a word): each token's start waits on the
  last one's length, and SWAR's chain is longer. Indexing a SIMD
  constant at run time spills it to the stack on every use; `_pow10`
  builds powers by selects instead. `read_gmsh` parses ASCII `$Nodes`
  and `$Elements` sections of 256 KB or more with `parse_numbers`
  (chunks per core: count tokens by SIMD masks, then parse into one
  array), and the section readers take the numbers from it
  (`_Cursor.tokenized`; bulk `ints`/`floats`); any token the serial
  reader might read differently makes it None, and any error reruns the
  section serially, so results and messages are the serial reader's
  (`test_parallel_parse_agrees` checks with the threshold at 0).
  `read_gmsh` drops format 2.2's repeated elements by `_unique_rows`:
  rows into parts by their smallest vertex position, then a hash table
  per part, threads balanced by rows (smallest entries skew low);
  `_unique_rows_sorted` (by `lexsort_rows`) when rows do not pack into
  63 bits. Grow lists per block with `_reserve_more` (doubling), never
  `reserve(len + n)`, which grows to exactly that: 2.2 starts a block
  whenever the physical group changes, possibly per element. VTK ASCII output raises on NaN
  and infinity, which VTK's text reader cannot parse.
  The Gmsh fixtures in `tests/meshes/gmsh/` are generated from the `.geo`
  files there: `pixi exec --spec gmsh gmsh -2 square.geo -format msh41
  [-bin] -o square_v41[_bin].msh` (and `msh22`; `-3` for the cube).

### Parallel loops

`parallel_ranges(n, serial_below, body)` (`core/parallel.mojo`) runs
`body(chunk, lo, hi)` on one chunk per logical core (per physical core
with `memory_bound=True`, which the matvec passes: a second hardware
thread speeds up arithmetic, not memory streams), or serially when
`n < serial_below`. A MAX `parallelize` call costs about 200 µs whatever the
work (POSIX threads cost 20–50 µs), so thresholds are set where the serial
loop takes a few hundred µs: matvec at `PARALLEL_NNZ` (1M stored entries),
circumcenters and primal volumes at 50k simplices, `whitney_innerproduct`
at 4096 top simplices (element loop and row assembly), dual volumes at
4096 top simplices, the supernodal factor at 5e6 multiply-adds (one call
for the whole factorization). `MOJODEC_NUM_THREADS=k` caps the chunks at k (1 runs
everything serially, for single-threaded benchmarks). Bodies must not
raise: record failures per chunk and raise afterwards. Keep a hot serial loop out of a function that also
holds the parallel branch (`CSRMatrix._matvec_parallel` is separate): in
the same function the serial loop ran about 15% slower. The pool's threads flush
subnormals to zero, inputs and results (FTZ and DAZ; the calling thread
does not), so a parallel loop matches the serial one only away from
subnormals (`parse_numbers` leaves `atof` results under 1e-307 to the
serial reader). Loops that sum
into shared entries stay bit for bit serial by ownership: each chunk owns
a block of outputs and adds only to those, in the serial order (dual
volumes: faces by first vertex id, which can exceed the number of
0-simplices when vertices are unused; `whitney_innerproduct`: matrix rows,
each summed over its top simplices in increasing order, as `from_coo`
would). Faces of codimension 1 take `_assemble_codim1`: in a
pseudomanifold each lies in one or two top simplices (the rows of the
top level's `boundary`), which share no other face, so a row is their
faces merged, each simplex's faces being in increasing order. Other
faces take `_assemble`: `_incidence` (a stable counting sort, parallel
by blocks of faces, 32-bit entries), then two passes, one counting each
row's columns (each slot marked with the row, no branch) and one
writing them straight into the result through a per-chunk slot array. `whitney_innerproduct`'s element loop for
triangles in 2D and 3D and tetrahedra in 3D is `_whitney_element`,
unrolled at compile time with the general loop's arithmetic and its
fused multiply-adds explicit, four top simplices to a SIMD vector
(`_whitney_range`; the remainder, and a batch with a degenerate
simplex, take the one-lane form of the same code). `_minor_det`'s row
pivoting becomes selects, the term tables are compile-time constants
(`_term_pair`, `_term_coef`, `_comb_mask`; bind them with `comptime`),
and `_element_faces` reads each face from one row of d
(`_first_container`), the four simplices' rows level by level. The
bits are the general loop's (check a change by hashing every k's
matrix against the committed build, on meshes whose element counts
leave remainders too). Ownership by vertex blocks assumes the vertex numbering is
spatially coherent: with randomly numbered vertices a tetrahedron touches
3.6 blocks instead of 1.4 and the 3D walk on all cores takes 2.6× as long
(`tet_40_shuffled` in `benchmarks/compare`), and every loop in face order
loses locality; `reorder_vertices` (`routines/mesh_generation.mojo`:
Morton order by a radix sort of the codes, which are computed in
parallel; elements sorted by their sorted rows, orientations kept)
restores it (see `docs/guide/vertex-numbering.md`). `numbering_spread`
measures a numbering (blocks per element, 16 blocks; in parallel, each
vertex's block from a byte table filled from the bounds ceil(k nv / 16),
which equals `v * 16 // nv` exactly; a 64-bit division per entry was
4.5 ns), and `read_gmsh`
prints a warning to stderr for 50k+ elements above 1.9 (Gmsh's own
numbering is 2.1–3.6; `warn=False` silences it, and tests check the
message through `_numbering_warning`). It must not run
between refinements, whose numbering keeps shapes bounded. Mojo's `>>`
by 64 or more is undefined (it gave garbage): bound shift loops. Allocate a body's scratch with `chunk_scratch(n, fill)`, never a
bare `List(length=...)`: two chunks' small lists can share a cache line,
and writing them in an inner loop (false sharing) made the 3D dual walk
slower on two threads than on one; helpers called per element take
their scratch as an argument (`frame_dual_basis(..., diag)`,
`parallelotope_volume(..., diag)`) instead of allocating it. Large
outputs the parallel loops write in full are created with
`List(unsafe_uninit_length=n)`: a `fill` is a serial memset that also
takes every page fault on one thread (it doubled the first call of
`whitney_innerproduct`). Topology is parallel too:
`simplex_array_boundary` sorts face occurrences into one bucket per
first vertex (`_boundary_by_buckets`, from 16384 occurrences with vertex
ids in a narrow range; occurrences are `r << sh | i`, as divisions by
the row width dominated the sort; each chunk counts only the band of
buckets from its rows' smallest first entry to their largest second) and
reads faces, `d`, and the boundary off the sorted runs, equal entry for
entry to `_boundary_by_lexsort`. Each bucket sorts occurrences packed
into one integer key (the rest of the face, the occurrence, and the sign
of its boundary entry) when that fits in 62 bits, and falls back to the
`_occurrence_less` comparator otherwise (`pack=False` forces it, for
tests); bit 62 marks each run's first occurrence, and the faces are read
back from the keys. A sorted simplex's faces come in decreasing
lexicographic order of the entry left out, so the runs write `d`'s
columns in place, with no pass sorting its rows; with rows of 2 entries
the buckets arrive sorted. The scatter, the runs, and `d`'s values are
unrolled for rows of 2–4 entries (`_scatter_packed`, `_runs_packed`,
`_d_rows`); check a change by hashing every level's simplices, parity,
`d`, boundary, and `_order` against the committed build. Sorts in
topology avoid data-dependent branches, which mispredict more or less
with the vertex numbering: sorting networks for rows of 3 and 4
(`sort_each_row`), parity by counting inversions, and rank sort (SIMD)
for buckets of 9 to 96 keys. The rank sort compares keys as doubles
(key + 2^52 is a positive normal double's bit pattern, so keys must stay
below 2^62), since AVX2 compares 64-bit integers on one port only. `require_distinct_simplices` skips its
sort when the rows are already in strictly increasing order; otherwise
`lexsort_rows` (a counting sort by the first column, in parallel like
topology's, which also packs each row's other entries into one key
when they fit in 63 bits; then each run by insertion on those keys, in
parallel, by merging past 32 rows) also reports the first repeated row, and its order becomes the top level's `_order`. `tests/test_parallel.mojo` uses meshes past
every threshold, and compares each parallel path with the serial one
exactly; the other tests' meshes run serially.

## Coding standards

- Use American spelling everywhere (code, comments, doc strings, docs,
  commit messages): "-ize", not "-ise" (`normalize`, `initialize`).
- Return a small struct with named fields (`SimplicialMesh`,
  `EighResult`, `SolveResult`, `Quivers`, ...) rather than a tuple, and an
  enum-like struct with `comptime` members (`Primality`, `StopReason`, the
  I/O formats) rather than a Bool or String flag.
- Public symbols get Google-style doc strings (`Args:`, `Parameters:`,
  `Returns:`, `Raises:`; see `docs/writing-docs.md`). The doc-check
  warning count must not get worse than `docs/docstring-baseline.txt`.
- Features beyond PyDEC (wedge, flat, curvature, quality, etc.) are
  implemented from the published mathematics, never from GPL-licensed
  code — keep it that way, and don't name GPL projects as sources.

## Testing

- Each test file ends with
  `TestSuite.discover_tests[__functions_in_module()]().run()`, so any
  `def test_*() raises` in the file is picked up automatically. Mesh
  fixtures live in `tests/meshes/`.
- `tests/test_properties.mojo` (combinatorics, simplex geometry, sparse
  kernels against a dense model, complexes) and
  `tests/test_properties_dec.mojo` (rigid-motion and relabeling
  invariance, volume identities, thin and far-away simplices, the
  codifferential's sign, curvature, Whitney, wedge, flat, I/O round trips)
  and `tests/test_properties_robust.mojo` (readers given truncated and
  corrupted files, raw CSR input, kd-tree and eigensolver against brute
  force or their definitions, every dtype through every format) are
  property tests using proptest, a sibling `../proptest` checkout.
- The examples double as regression checks against PyDEC; README.md's
  Examples table lists the expected results (e.g. DarcyFlow pressures to
  1e-14, ResonantCavity eigenvalues m²+n², MakeMesh matches
  `examples/MakeMesh/ExpectedOutput/`). DarcyFlow, HeatConduction,
  ResonantCavity, and CavityWaves use the typed layer. HeatConduction imposes Dirichlet rows
  with `exclude_subset(fixed.dual())` plus `tc.dual_diagonal[...]`
  restricted to the same cells; DarcyFlow solves the mixed `BlockOp` with
  `spsolve(..., values=, fixed_first=, fixed_second=,
  method=DirectMethod.SPARSE_LDLT)`: PyDEC's elimination (the same rows
  and columns dropped, the same right-hand side), then the reduced saddle
  point factored sparsely with one refinement step (pressures to 4e-16).
