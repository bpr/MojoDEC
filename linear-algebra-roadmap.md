# Refinement, multigrid, and where sparse linear algebra belongs

Notes for later review (2026-11-11). The question: is item 8 of
`formoniq.md` (refinement with prolongation, then multigrid, sparse
factorization, and Lanczos) worth doing, given that the last three look
like jobs for the linear-algebra dependency (NuMojo)?

## Refinement with prolongation: worth doing, as one modest item

The pieces mostly exist.

- **Refinement.** Freudenthal (Kuhn) subdivision of each n-simplex into
  2^n children works in any dimension and stays shape-regular (finitely
  many similarity classes). It reuses the permutation machinery of
  `kuhn_grid`. Today `loop_subdivision` handles only surfaces.
- **Prolongation is exact.** The coarse Whitney space sits inside the
  fine one, so the prolongation of k-cochains is `P_k = R_fine W_coarse`
  (`derham_map` after `whitney_form`). It is sparse, has small dyadic
  entries, and commutes with d (`P d = d P`), a strong test. It even
  needs no geometry: the integral of a Whitney form over a sub-simplex
  depends only on barycentric coordinates.

What it buys:
- refining a user's mesh (from Gmsh, say) in any dimension, which MojoDEC
  cannot do now;
- nested-mesh convergence tests (`docs/guide/convergence.md` lists them
  as not covered);
- error estimates and Richardson extrapolation without an exact
  solution, from comparing `P u_coarse` with `u_fine`;
- transferring fields between levels, e.g. initial conditions in time
  stepping.

Caveat: refining a well-centered mesh does not keep it well-centered in
3D, so this pays off mostly with Whitney inner products, not with the
diagonal star.

## Multigrid belongs in MojoDEC, if anywhere

Geometric multigrid for k-forms is not generic linear algebra. It needs:
- the mesh hierarchy;
- the prolongation above;
- smoothers that respect the large kernel of d (Hiptmair's, or
  Arnold–Falk–Winther's).

That is DEC-specific. Algebraic multigrid would be generic, but it
handles those kernels badly. Defer it until someone needs to solve
systems with 10^5 or more unknowns.

## Sparse factorization and Lanczos: generic, but not NuMojo's today

They are indeed domain-independent. But NuMojo has no sparse matrices,
and its dense `solve` (no pivoting) and Jacobi `eigh` were weak enough
that MojoDEC already carries its own:
- `CSRMatrix`;
- pivoted LU and Householder/QL `eigh`;
- CG and MINRES with the operator traits.

So "belongs in the dependency" really means "belongs in a separate sparse
linear-algebra package". The clean move, if it ever matters, is to split
MojoDEC's `CSRMatrix`, the `LinearOperator`/`Preconditioner` traits, and
the Krylov solvers into such a package, possibly offered to NuMojo.
Nothing in the examples needs a sparse factorization or Lanczos yet: the
dense solvers handle those sizes.

## What the block preconditioners showed (2026-11-11)

The mixed Hodge Laplacian on 1-forms (Kuhn grids, 4, 8, 16 cells per
side) measured MINRES iterations:
- Jacobi: 79, 205, 424.
- Diagonal blocks (`diag(M0)` and the diagonal of the Schur complement
  approximation): about the same as Jacobi.
- `BlockDiagonal(DenseCholesky(M0 + d0^T M1 d0), DenseCholesky(M1 + A))`,
  the Arnold–Falk–Winther preconditioner with exact blocks: 7, 7, 7.

So the block structure is right, and all of its value lies in applying
the H(grad) and H(curl) inverses well. Diagonal approximations are no
substitute, and inner CG is not a fixed linear operator. The scalable
version is exactly the multigrid of the section above (Hiptmair or AFW
smoothers on the refinement hierarchy), which now has its prolongation.
The framework (`BlockDiagonal`, any `Preconditioner` as a block) is in
place for it.

## Order

1. Refinement and prolongation, one round. **Done.**
2. The smaller items, **all done 2026-11-11**:
   - implicit midpoint time stepping (`ImplicitMidpoint`, `CrankNicolson`);
   - Dirichlet (relative) harmonic forms (`relative_to` overloads);
   - mixed Hodge–Laplace convergence (`test_mixed_hodge_laplacian`);
   - Möbius and Klein quotients as negative orientation tests
     (`mobius_band`, `klein_bottle`);
   - block preconditioners (`BlockDiagonal`, `DenseCholesky`; see above).
3. Multigrid, sparse factorization, and Lanczos only when a problem size
   calls for them; the generic parts in a separate package. **Now
   called for:** see the next sections.

## Real meshes need them now (2026-11-13)

Item 3's trigger has arrived. End-to-end tests on Gmsh meshes (a box
with a spherical hole up to 1.05M tetrahedra, a graded 2D plate with
holes at 323k triangles, a sphere surface at 133k) found the geometry
and assembly sound, and the solvers the limit:

- **Graded 2D meshes:** Jacobi-CG took 834 iterations (1.3 s) on the
  plate's Poisson problem, 160k unknowns. 3D Poisson on 378k tetrahedra
  was fine (122 iterations, 90 ms): the pain is condition numbers that
  grow with `h_max / h_min`, and higher-degree forms.
- **Eigenproblems:** `eigh_generalized` is dense, so the sphere's
  spectrum at 66k vertices was out of reach; only Rayleigh quotients
  could be checked.
- **Mixed problems:** DarcyFlow now solves with `SparseLDLT`
  (`spsolve(..., method=DirectMethod.SPARSE_LDLT)`); the block preconditioner above needs scalable H(grad) and H(curl)
  inverses.

Decisions (2026-11-14):
- **Pure Mojo only.** No bindings to C libraries (Hypre, SuiteSparse,
  PETSc): ports of permissively licensed code, or our own from papers.
- **Not through NuMojo.** It is dense-only; improving it would not touch
  these problems, and MojoDEC would lose control of guarantees it relies
  on (parallel equal to serial, bit for bit).
- **faer's MIT code may be ported directly** (2026-11-15), its sparse
  Cholesky included, though its design follows Davis's LGPL codes (see
  below); its MPL-2.0 files stay out.

## Audit of Rust sources (2026-11-14)

Three libraries were read at these commits; nothing was built.

### faer: the source to port

`codeberg.org/sarah-quinones/faer`, commit `0ecd846` (faer 0.24.4,
October 2026), MIT (© sarah quiñones el kazdadi), actively maintained.
About 99k lines in `faer/src`: 22k sparse, 53k dense, 6k iterative.

Has:
- **Orderings:** AMD (with the aggressive-absorption and dense-row
  options) and COLAMD, both credited translations of SuiteSparse's BSD-3
  AMD and COLAMD (`COPYING.SUITE_SPARSE.*.BSD`). No nested dissection.
- **Sparse Cholesky** (`sparse/linalg/cholesky.rs`, 5.6k lines):
  - simplicial: elimination tree, column counts, `ereach`, up-looking
    numeric factorization;
  - supernodal: left-looking, each update a GEMM scattered into the
    target supernode, then dense Cholesky and a triangular solve per
    supernode, parallel inside;
  - LLᵀ and LDLᵀ, both with optional dynamic regularization (a pivot
    below ε becomes ±δ), and LDLᵀ with a per-row sign vector: the
    standard treatment of saddle-point (KKT) systems such as mixed Darcy;
  - Bunch–Kaufman (or rook) pivoting only within a supernode, no delayed
    pivots: a zero (2,2) block needs the regularized LDLᵀ.
- **Sparse LU** (Gilbert–Peierls with partial pivoting; supernodal with
  COLAMD) and supernodal Householder QR.
- **Lanczos** (`operator/self_adjoint_eigen`, 351 lines): thick restart,
  full reorthogonalization, locking. Standard problem only, largest
  eigenvalues, no M-inner product, no shift-invert, no LOBPCG.
- **Dense:** blocked LLᵀ, LDLᵀ, and LBLᵀ (Bunch–Kaufman, rook or
  diagonal pivoting), LU with partial and full pivoting, blocked QR,
  symmetric eigen by tridiagonalization and divide and conquer, QZ.
- Iterative solvers (CG, BiCGSTAB, LSMR) with a `Precond` trait, but no
  preconditioners beyond identity and diagonal.

Lacks: AMG, incomplete factorizations, auxiliary-space preconditioners.

Provenance:
- AMD and COLAMD: BSD-3, credited.
- Dense SVD divide and conquer: from Eigen, `COPYING.EIGEN.MPL2`; the
  dense eigensolver's divide-and-conquer step shares its secular-equation
  code. Do not port these files (we keep QL `eigh` or write our own).
- Givens rotations, pivoted LLᵀ, column-pivoted QR, Schur: LAPACK, BSD.
- The simplicial symbolic step reproduces the loop of Davis's LDL
  (`ldl_symbolic`), `ereach` is CSparse's `cs_ereach`, and the default
  supernode-merging thresholds `[(4, 1.0), (16, 0.8), (48, 0.1),
  (MAX, 0.05)]` are CHOLMOD Supernodal's `nrelax`/`zrelax` defaults (all
  LGPL-2.1+). No header cites them, and the code is a fresh
  reimplementation (left-looking, no CHOLMOD Map/RelativeMap), not a
  translation. Accepted as MIT (decision above); credit faer, and cite
  the published algorithms (Liu, Gilbert–Ng–Peyton, Ng–Peyton,
  Ashcraft–Grimes for merging) in the doc strings and the guide page.

Porting: faer is generic over the scalar (complex included), brands
dimensions with `generativity` ghost lifetimes, and takes scratch from
`dyn-stack`, SIMD from `pulp`, GEMM from the author's separate `gemm`
crates, threads from rayon. Restricted to Float64 and Int, nearly all of
that disappears and the sparse code is plain index arithmetic. Estimated
Mojo sizes:

| Piece | Lines |
|---|---|
| AMD | 700–900 |
| Simplicial LLᵀ/LDLᵀ with regularization and solves | ~400 |
| Supernodal symbolic + numeric LLᵀ/LDLᵀ + solves | ~1,500 |
| Dense blocked Cholesky, triangular solves, indexed SYRK/GEMM (our SIMD) | ~600 |
| Dense Bunch–Kaufman LBLᵀ (rook) | ~600 |
| Thick-restart Lanczos, M-inner product, shift-invert | ~450 |
| COLAMD (if needed) | ~500 |
| Supernodal LU (if needed) | ~2,000 |

### sprs: little to take

`github.com/sparsemat/sprs`, commit `92f95fd` (October 2026), MIT or
Apache-2.0, in maintenance mode. No eigensolvers, no preconditioners;
an unpreconditioned BiCGSTAB; SpGEMM (Bank–Douglas SMMP) and triangular
solves we already match or exceed. Its one factorization, `sprs-ldl`, is
**LGPL-2.1**, adapted line for line from Davis's LDL, with no pivoting:
do not port it. The UMFPACK bindings link GPL code. Worth a glance only:
reverse Cuthill–McKee with a pseudo-peripheral start (`ordering.rs`,
MIT), short enough to write from a textbook.

### scirs: do not use

`github.com/cool-japan/scirs`, commit `467efc0` (July 2026),
Apache-2.0. The sparse and linalg crates (136k and 259k lines) are
machine-generated at scale, and every algorithm opened was a
placeholder, wrong, or dense in disguise:
- `CsrMatrix::matmul` converts both operands to dense;
- the generalized eigensolvers ignore `B`, and shift-invert ignores
  `sigma`; LOBPCG does not B-orthogonalize consistently and falls back
  silently to the standard problem; the Lanczos tridiagonal solver is
  Numerical Recipes' `tqli` (a restrictive license);
- the "sparse" Cholesky builds a dense matrix; "AMD" is an O(n²)
  minimum-degree scan; smoothed aggregation and Ruge–Stüben build dense
  prolongators, and the RS interpolation formula is wrong; the IC
  diagonal update is wrong;
- tests check shapes and that residuals decrease, not answers.

### Preconditioners: other sources

None of the three has them.
- **Smoothed-aggregation AMG** for 0-forms (Vaněk–Mandel–Brezina):
  amgcl (MIT) as the source, PyAMG (MIT) as the test oracle (see the
  amgcl audit below). Plugs
  into `cg(M=...)` as a `Preconditioner` (the V-cycle must be symmetric:
  symmetric smoothers, same pre- and post-smoothing).
- **Hiptmair–Xu auxiliary-space preconditioners** (AMS for H(curl), ADS
  for H(div)): from the papers, with hypre's AMS (MIT or Apache-2.0)
  as a reference for a port, never a binding. Needs the discrete
  gradient (`sc[0].d`) and the vertex coordinates, both at hand; the
  lowest-order Whitney 1-forms are the Nédélec elements AMS assumes.
  This is the scalable form of the block preconditioner above (7 MINRES
  iterations with exact blocks).
- **LOBPCG** (Knyazev; the Hetmaniuk–Lehoucq and Duersch et al.
  stabilizations), as the eigensolver that needs only a preconditioner,
  not a factorization. SciPy's (BSD) as a reference.
- PETSc (BSD-2) is next to read, for its preconditioner designs
  (GAMG, PCFIELDSPLIT/Schur, PCBDDC).

## Audit of PETSc (2026-11-15)

`gitlab.com/petsc/petsc`, commit `ba01f43` (November 2026), BSD-2.
Read, not built. PETSc is mostly MPI and object infrastructure; the
algorithms below are small once that is stripped. Line references are
to that commit.

### Provenance

Everything listed under "Take" is PETSc's own BSD-2 code. Exceptions:
- `src/mat/graphops/order/{genrcm,gennd,genqmd,gen1wd,...}.c` are f2c
  translations of SPARSPAK (George–Liu), headerless; PETSc's only
  license basis is "the PUBLIC DOMAIN version of SPARSPAK"
  (`doc/install/external_software.md:48`). Write from George–Liu's
  book if ever needed, don't port the f2c code.
- `amd/amd.c` and `metisnd/` are interfaces to SuiteSparse and METIS,
  no code. `wbm.c` calls HSL's MC64 (restrictive license): avoid.
- MINRES-QLP (`ksp/impls/minres/minres.c:89`) is "adapted from" SOL's
  MATLAB `minresQLP.m`: implement from Choi–Paige–Saunders (SISC 2011)
  instead of porting it.
- `gamg/geo.c` can include Shewchuk's Triangle (non-OSI): skip.
- `pc/impls/hpddm` wraps the external HPDDM library: skip.
- Cite the papers PETSc cites: Farrell–Knepley–Mitchell–Wechsung (TOMS
  2021) for PCPATCH, Lottes (2023) for the optimized fourth-kind
  Chebyshev coefficients (`cheby/betas.c`), Fischer (1998) for the
  initial guesses.

### Take: multigrid for k-forms (the DEC-native preconditioner)

- **Vertex-star patch smoother** (`pc/impls/patch/pcpatch.c`, 3.7k
  lines, ~350 of algorithm). The default patch is the open star of a
  vertex (`PCPatchConstruct_Star`, :31); dofs in the closure but not the
  star are zero Dirichlet. For DEC: patch v = the k-simplices containing
  v, and the patch operator is just the principal submatrix
  `A[I_v, I_v]` of the assembled `M_k + d_kᵀ M_{k+1} d_k` (any element
  coupling two simplices of the star contains v). Patches are small
  (3D Kuhn: ~14 edges, ~36 faces), so dense Cholesky per patch. Additive
  (the default) or symmetric multiplicative (Gauss–Seidel over patches,
  forward and back, SPD). This is Arnold–Falk–Winther's smoother: it
  handles the kernel of d because that kernel is spanned by d of
  (k−1)-forms, each supported in one star. Bit-for-bit parallel: sort
  each patch's dofs; in additive mode each dof gathers its patches in
  vertex order (ownership, as `_assemble`); the multiplicative sweep
  needs a deterministic coloring.
- **Chebyshev smoother** (`ksp/impls/cheby/cheby.c`, ~200 lines to
  port). First kind on `[0.1, 1.1]·λmax(D⁻¹A)` (`:201`, `:908–910`);
  λmax from 10 Jacobi-CG steps on a deterministic hashed right-hand side
  (`KSPNoisyHash_Private`, `kspimpl.h:265`), read off CG's Lanczos
  tridiagonal. The optimized fourth kind (tabulated β up to order 16)
  needs no lower bound. Polynomial, hence symmetric: the V-cycle stays
  a valid `Preconditioner` for `cg`.
- **V/W cycle** (`pc/impls/mg/mg.c:66–180`, ~60 lines of logic):
  defaults 2 smoothing steps, Galerkin coarse operators `PᵀAP`. Over
  `refine()` with the exact `P_k` this, plus the star smoother, is the
  geometric multigrid this file planned in its first sections, now with
  a concrete design. It needs the user's mesh to come from refinement.
- **Smoothed aggregation** (`pc/impls/gamg/agg.c`, ~700 lines of core
  among ~4.3k): graph `D^-1/2 (|A|+|A|ᵀ) D^-1/2`, greedy MIS (on the
  squared graph on the finest level), tentative P by a QR per aggregate
  on the near-null space, one Jacobi smoothing step `(I − 1.4/λmax
  D⁻¹A)` (the 1.4 is undocumented, `agg.c:2142`; PyAMG uses 4/3),
  `PᵀAP`, stop at ≤ 50 unknowns, Chebyshev–Jacobi smoothing reusing the
  λ estimates. PETSc's MIS order is a seeded shuffle that depends on the
  MPI layout; use natural (Morton) order serially, hashed-priority Luby
  MIS in parallel. Scalar only: no help with the kernel of d. Its jobs
  are 0-forms on unstructured meshes (the 834-iteration plate) and the
  H1 solves inside AMS. Compare with PyAMG before choosing which to
  follow.

### Take: block and auxiliary-space structure

- **Schur `diag` with `selfp`** (`pc/impls/fieldsplit/fieldsplit.c`,
  `ksp/utils/schurm/schurm.c:780`). `selfp` is `S ≈ A11 − A10
  diag(A00)⁻¹ A01`. In DarcyFlow `A00 = −(μ/k) star1` is diagonal, so
  `selfp` is the exact Schur complement, a weighted graph Laplacian on
  the dual: `diag(A00, −S)` with MINRES is optimal given an AMG or
  multigrid solve of S. With Whitney `M1`, `diag(M1)` is spectrally
  equivalent, so iteration counts stay bounded in h. PETSc scales S by
  −1 for MINRES and switches to +1 when the matrix is known SPD
  (`:849`, `:1210`): copy that. The `lower`/`upper`/`full`
  factorizations are nonsymmetric and need FGMRES or GCR, which we lack.
  Core ~250 lines over `BlockOp`. The DarcyFlow example keeps a direct
  solve (it must match PyDEC to 1e-14); this is for large meshes.
- **Galerkin and composite combinators** (`pc/impls/galerkin`,
  `composite`, ~200 lines together): `y = P (PᵀAP)⁻¹ Pᵀ x` as a
  preconditioner, and additive or multiplicative combinations. AMS is
  exactly "smoother, then `G (GᵀAG)⁻¹ Gᵀ`, then the Π correction", so
  these are the pieces to build it from.
- **The AMS/ADS interface** (`pc/impls/hypre/impl/ihypre.c`), as the API
  our own AMS/ADS should take: the discrete gradient G (`sc[0].d`,
  required); vertex coordinates, or the edge constant vectors `G x`,
  `G y`, `G z`, or the Nédélec interpolation Π; optional alpha and beta
  Poisson matrices; ADS also takes the discrete curl (`sc[1].d`). Zero
  beta (no mass term) must be declared, never detected. Defaults are
  MFEM's, not hypre's (`:2116`): cycle type 13, ℓ1 symmetric
  Gauss–Seidel, θ = 0.25.
- **Deflation** (`pc/impls/deflation`, ~60 lines of core): a coarse
  correction `W (WᵀAW)⁻¹ Wᵀ`. With `W` from `harmonic_basis` or
  `cohomology_basis` it removes near-kernel modes of Hodge Laplacians on
  domains with holes.

### Take: Krylov conveniences (small, useful now)

- **Null-space projection** (`mat/interface/matnull.c:405`,
  `kspimpl.h:360,450`): with an orthonormal basis of the kernel, project
  it out of b and out of every `M⁻¹r`. Keeps CG/MINRES in the range for
  singular Hodge Laplacians (`harmonic_basis` already uses MINRES on a
  consistent singular system). Under 100 lines.
- **Lanczos coefficients from CG** (`ksp/impls/cg/cg.c:357–376`,
  `cgeig.c`): record CG's tridiagonal while iterating, eigenvalues from
  our QL `eigh`. Free extreme-eigenvalue estimates for Chebyshev,
  Leapfrog's stability limit (now power iteration) and shift choices.
- **Fischer's initial guess** (`ksp/guess/impls/fischer`, method 2):
  keep an A-orthonormal basis of the last ~10 solutions; `x0 = Σ (x̃ᵢᵀb)
  x̃ᵢ` is the A-norm-best guess from them. `ImplicitMidpoint` and
  `CrankNicolson` solve with the same matrix every step: one matvec and
  a few dots per step for fewer CG iterations. ~100 lines.
- **MINRES-QLP** (from the paper; PETSc's thresholds `TranCond =
  maxxnorm = 1/√eps` as reference): the minimum-norm solution of
  singular, possibly inconsistent systems. ~400 lines.
- **Flexible CG** (Notay; `ksp/impls/fcg`): makes inner-CG block solves
  legitimate inside a preconditioner. Note PETSc's indefinite-
  preconditioner test there, `|β·β_old| < 0` (`fcg.c:197`), can never
  fire.
- Convergence-test details worth matching: a divergence tolerance
  (`rnorm > 1e4·rnorm0`) and a distinct reason for NaN/Inf.

### Take: time stepping

- **Higher-order symplectic tableaus**
  (`ts/impls/symplectic/basicsymplectic/basicsymplectic.c:70–77`):
  Ruth's order 3 and Forest–Ruth's order 4 as alternating kicks with
  coefficients (c, d). `Leapfrog` generalizes to a staged tableau in
  ~60 lines, same matrices, same skew structure. Cite Ruth (1983) and
  Forest–Ruth (1990) / Candy–Rozmus (1991).
- The discrete-gradient integrator (`ts/impls/discgrad`) reduces to
  implicit midpoint for quadratic energies, which is all our linear
  dynamics: skip until there is a nonlinear Hamiltonian.

### Skip

BDDC (19k lines, needs unassembled subdomain matrices, a partitioner
and sparse local factorizations; its Nédélec change of basis is worth
reading only if domain decomposition returns), LSC (for nonsymmetric
Navier–Stokes), ASM (its default restricted variant is nonsymmetric),
Eisenstat (needs split preconditioning), classical AMG, PETSc's native
Cholesky (up-looking, not supernodal: faer's is better), ICC/ILU (ICC(0)
with the bisected positive-definite shift, `aijfact.c:338`, would help
the time-stepping matrices but is serial and h-dependent; multigrid
supersedes it), the SPARSPAK orderings (AMD beats them), the spectral
ordering (dense `geev`), the pipelined CG variants (MPI latency hiding).

## Audit of Eigen (2026-11-15)

Local checkout `/home/bpr/src/c++/eigen`, commit `77bee9307` (November
15, 2026). **MPL-2.0**, file-level copyleft: a Mojo translation of an
Eigen file would have to stay MPL-2.0 inside MIT MojoDEC, the same
reason faer's MPL files stay out. Worse, `SimplicialCholesky_impl.h`
(from Davis's LDL), `OrderingMethods/Amd.h` (from CSparse) and
`IncompleteLUT.h` (from SPARSKIT) are LGPL code relicensed to MPL only
"as part of Eigen". **Decision: read for design, port nothing**; cite
the papers. The tree has moved since older docs: GMRES, DGMRES, MINRES,
IDR(s) are in `Eigen/src/IterativeLinearSolvers`, `SparseExtra` and
`ArpackSupport` in `contrib/`.

Findings:
- **`MultifrontalLDLT.h`** (1745 lines, added 2026-11-14, one day old
  when read): multifrontal LDLᵀ with Duff–Reid threshold 1×1/2×2
  pivoting (u = 0.01) and delayed pivots, Ashcraft–Grimes front
  amalgamation, panel updates of 32 columns, inertia and determinant.
  The better algorithm for indefinite systems (exact factorization,
  correct inertia, no regularization), but young, serial, unscaled, and
  tested only on synthetic KKT matrices. Not rank-revealing: a
  rounding-size pivot passes, so singular Hodge Laplacians still need
  projection or pinning. Write it from Duff–Reid (1983) and Liu (1992)
  later, only if faer's regularized LDLᵀ fails on non-well-centered
  meshes (~1,600 Mojo lines).
- **The trick that makes Darcy easy:** Eigen's AMD orders every column
  without a structural diagonal last (`Amd.h:146`). With the zero (2,2)
  block absent from the pattern, pressures come last, the edges are
  eliminated on `M1`'s positive pivots, and then the Schur complement
  `−d M1⁻¹ dᵀ` is factored: no zero pivot ever, so faer's LDLᵀ without
  pivoting suffices when `M1` is positive. Add this constraint to the
  faer AMD port (an idea, not code). Cost: the pressures' fill is not
  reduced; Morton numbering softens it.
- `SimplicialCholesky`: up-looking LDL, no pivoting, a static
  `setShift`. Nothing beyond faer. `Amd.h` is CSparse's `cs_amd`, not
  SuiteSparse AMD; faer's BSD-3 translation is the clean source.
  `Cholesky/LDLT.h` is diagonal (semidefinite) pivoting, not
  Bunch–Kaufman; a new blocked `BunchKaufman.h` duplicates faer's LBLᵀ.
- Iterative solvers are weaker than MojoDEC's: MINRES mixes the
  Euclidean `‖r0‖` with an M-norm recurrence and can produce NaN on
  singular systems (`MINRES.h:56,113,127`); CG has no curvature check;
  `DiagonalPreconditioner` substitutes 1 for zero diagonals; ILUT drops
  L entries by an absolute tolerance but U entries relative to the row
  norm (`IncompleteLUT.h:531,598`).
- No Lanczos or LOBPCG anywhere. New tridiagonal bisection and inverse
  iteration (2026) are the right kernels for Lanczos's T; take them from
  LAPACK's `dstebz`/`dstein` (BSD) if QL `eigh` ever falls short.
- Expression templates buy little here: sparse×sparse is evaluated
  eagerly in Eigen too. What pays off (Mojo traits with default
  methods, no evaluator layer):
  - **fused Krylov kernels**: `y = Ap` with `pᵀAp` in one pass; `x +=
    αp`, `r −= αAp`, `‖r‖²` in one; `z = Dr` with `rᵀz` in one. CG's
    vector traffic exceeds the SpMV's on 2D meshes, so ~1.5× per
    iteration. Reductions over fixed blocks (e.g. 4096 entries)
    combined in order, so any thread count gives the serial bits;
  - **starting guesses** (`x0=` in `cg`/`minres`, which now always
    start from zero, `linalg.mojo:273`), needed for Fischer's guesses;
  - **Ruiz equilibration** as a `DiagonalOperator`: `DAD` then MINRES
    equals MINRES with `M = D²`, an SPD diagonal preconditioner where
    `jacobi` raises on the zero block, and a prescaling before LDLᵀ
    (from Ruiz, RAL-TR-2001-034; ~40 lines);
  - `PAPᵀ` written straight into one triangle by two counting passes
    (`SparseSelfAdjointView.h:513`), the input faer's Cholesky takes.
    Keep full storage for SpMV: the one-triangle SpMV scatters, which
    breaks row ownership.
- SpGEMM (`ConservativeSparseSparseProduct.h`) is Gustavson with a
  dense marker, unsorted output fixed by two transposes; no triple
  product. Ours should be row-parallel in two passes (count, then fill)
  with sorted short rows and a reusable symbolic phase.

## Audit of amgcl (2026-11-15)

`github.com/ddemidov/amgcl`, commit `c68350e` (November 9, 2026), MIT
(© Denis Demidov), header-only C++, ~39k lines. **The source for
smoothed aggregation**: compact, Vaněk–Mandel–Brezina's algorithm as
published, and already deterministic where it matters. Its defaults
(tuned for GPUs and throughput) are the weak part.

Take:
- **Aggregation** (`coarsening/plain_aggregates.hpp`, 213 lines): j is
  strong for i when `a_ij² > ε² a_ii a_jj` (`:136`), ε = 0.08 halved
  per level (`smoothed_aggregation.hpp:140`); one serial greedy pass in
  natural order, so Morton numbering gives compact aggregates and the
  result is independent of threads (GAMG's MIS order depends on the MPI
  layout: don't port it). Rows with no strong neighbor (Dirichlet) get
  empty rows of P. ~120 lines.
- **Tentative and smoothed P** (`tentative_prolongation.hpp`,
  `smoothed_aggregation.hpp`): 0/1 P₀ for scalar problems (QR per
  aggregate when a near-null space is given, parallel by aggregate);
  `P = (I − ω D⁻¹ A_F) P₀` with weak entries lumped into the diagonal,
  so P keeps the strong pattern. Default ω = 2/3 (assumes λmax ≈ 2);
  use `4/3 / λmax` with a CG-Lanczos λ instead (PyAMG's 4/3; GAMG's
  1.4). Gershgorin is the deterministic fallback; amgcl's power method
  seeds per thread (`builtin.hpp:839`). ~90 lines.
- **Galerkin `PᵀAP`** by Saad's row-wise two-pass SpGEMM
  (`detail/spgemm.hpp:62`), bit-identical for any thread count. Above
  16 threads amgcl switches to a merge SpGEMM with different bits unless
  `AMGCL_DETERMINISTIC_REDUCTIONS` is set (`builtin.hpp:391`): we keep
  the Saad form only.
- **Symmetric Gauss–Seidel** (`relaxation/gauss_seidel.hpp`): forward
  sweep before, backward after; parallel by level scheduling, which
  preserves the serial update order, hence bit-identical. ~150 lines.
- **Hierarchy and V-cycle** (`amg.hpp`): 1 pre- and 1 post-smoothing,
  symmetric when post is pre's transpose and R = Pᵀ. Coarse solve with
  `DenseCholesky` below ~1000 unknowns, later the sparse Cholesky.
- **`selfp` Schur assembly** (`preconditioner/schur_pressure_correction.hpp:479`,
  `adjust_p = 2`): ~40 lines for Darcy's `diag(A00, −S)`.
- `spai0` (`m_i = a_ii / ‖a_i‖²`, 25 lines) as a cheap option only: the
  V-cycle is SPD only if λmax(MA) < 2, which holds for diagonally
  dominant M-matrices, not in general.
- Fused residual `b − Ax` (`backend/builtin.hpp`) for smoothers.

Skip: skyline LU and Cuthill–McKee (adopted from Kratos, whose BSD
license has an advertising clause), energy-minimizing SA (R ≠ Pᵀ,
breaks CG), the ILU family, CPR, `deflated_solver` (nonsymmetric), the
Chebyshev defaults (polynomial in unscaled A, interval `[λ/30, λ]`:
use PETSc's D⁻¹A on `[0.1, 1.1]λ`), its Krylov solvers (no MINRES, no
breakdown checks; thread-count-dependent dot products by default).

Verdict: **SA from amgcl, smoothers and λ estimates per PETSc, PyAMG as
the test oracle** (hierarchy sizes and iteration counts on the plate).

## Revised order (2026-11-15)

1. **AMD and simplicial LDLᵀ** with sign-aware regularization, from
   faer: sparse DarcyFlow (replacing the dense `spsolve`, which must
   keep its 1e-14 agreement with PyDEC), and the inner solve for
   shift-invert. Test against the dense `spsolve` and `DenseCholesky`,
   and on random SPD and saddle-point matrices with known solutions.
2. **Lanczos with an M-inner product and shift-invert** through item 1:
   the smallest eigenvalues of `K x = λ M x` (sphere `l(l+1)`,
   ResonantCavity `m² + n²` at real sizes), checked with
   `EighResult.residuals`.
3. **Smoothed-aggregation AMG** for 0-forms, from amgcl (superseded: see below); then LOBPCG
   preconditioned by it.
4. **Supernodal Cholesky/LDLᵀ** with dense blocked kernels, when
   simplicial factorization time matters (3D, past ~10⁵ unknowns).
5. **AMS/ADS** for H(curl)/H(div), making the block preconditioner
   scale; geometric multigrid on the refinement hierarchy remains the
   alternative.
6. Later: split `CSRMatrix`, the operator traits, Krylov solvers, and
   the above into a sparse package.

After the PETSc audit (proposed, not yet agreed):
- **0. Small Krylov items first** (each under ~100 lines, useful now):
  null-space projection in `cg`/`minres`, Lanczos coefficients from
  `cg` (eigenvalue estimates), Fischer initial guesses in the implicit
  time steppers.
- **3a. Chebyshev smoother, vertex-star patch smoother, and a V-cycle
  on the `refine()` hierarchy**, before or alongside SA-AMG: together
  they precondition `M_k + dᵀ M d` for every k on refined meshes, the
  scalable form of the AFW block preconditioner, with no
  auxiliary-space machinery.
- **3b. Schur `diag`/`selfp` for Darcy** once a 0-form AMG exists.
- **5. AMS/ADS** built from Galerkin and composite combinators, with
  hypre's interface (via PETSc) as the API and MFEM's defaults; needed
  for meshes that do not come from refinement.
- Later: MINRES-QLP, flexible CG, higher-order symplectic tableaus,
  deflation by the harmonic basis.

## Recommended order after all audits (2026-11-15, proposed)

Sources settled: faer (MIT) for AMD and Cholesky/LDLᵀ, amgcl (MIT) for
smoothed aggregation, PETSc (BSD-2) for smoothers, patches, Schur and
Krylov designs, papers for the rest; Eigen (MPL-2.0) and the LGPL
codes for reading only.

1. **Krylov groundwork** (one session, each piece small, all
   prerequisites of what follows): `x0=` in `cg`/`minres`; null-space
   projection with an orthonormal harmonic basis; Lanczos coefficients
   recorded by `cg` for eigenvalue estimates; the over/underflow
   scaling factor. Optionally the fused vector kernels with fixed-block
   reductions (measure on the quiet machine).
   **Done 2026-11-15** (uncommitted): `x0=`, `nullspace=` (`NullSpace`),
   `ritz=` on `cg`/`minres` and their typed `MatrixOp` overloads,
   `eigenvalue_range`, `dense.tridiagonal_eigenvalues`; guide sections in
   `linear-solvers.md`. Left out: the scaling factor (norms of RealArrays
   in DEC are far from Float64's limits), fused kernels (wait for a
   benchmark), `BlockOp` overloads of the options.
   Fischer's guesses (16 solutions) in `ImplicitMidpoint` and
   `CrankNicolson`: **done 2026-11-15**, 2.4–2.9× fewer CG iterations
   for Crank–Nicolson, up to 35% for the midpoint rule at small steps.
   Property tests in `test_properties_robust.mojo` against dense solves;
   they found three bugs, fixed: CG iterating on rounding when `b` lies in
   the null space (now `x = 0`), MINRES reporting breakdown at an
   invariant Krylov space under projection, and readers asserting on
   invalid UTF-8.
2. **AMD (with zero-diagonal columns last) and simplicial LDLᵀ from
   faer**, with sign-aware regularization and one or two refinement
   steps; `PAPᵀ` into one triangle. **Done 2026-11-15** (uncommitted):
   `amd_ordering` (permutations equal to faer's, checked against faer
   itself), `SparseLDLT`, `SparseCholesky` (a `Preconditioner`),
   `docs/guide/sparse-direct.md`. 2D Laplacian, 90,601 unknowns: 128 ms
   to factor, 4 ms per solve (Jacobi-CG 249 ms); 3D 30³: 2.3 s, 11.9 M
   entries in L (supernodal territory). DarcyFlow on `SparseLDLT`
   through `spsolve(..., method=DirectMethod.SPARSE_LDLT)`: one
   refinement step per solve (now the default) takes its pressure error
   from 2.2e-14 to 4.4e-16 (dense LU: 7.8e-15). Implicit time steppers
   factor once (`StepSolver.AUTO`: when nnz(L) <= 8 nnz(S), i.e. 2D),
   6-16x faster than CG over 100 steps on 128²-256² grids; 3D keeps CG,
   which wins there until supernodal. Typed wrappers: `TypedSparseLDLT`,
   `TypedSparseCholesky`, `BlockSparseLDLT`. Item done. Payoff across every degree and
   problem type at once: sparse DarcyFlow (keeping 1e-14 against PyDEC),
   implicit time stepping factored once and solved per step, Hodge
   Laplacians of any degree, coarse solves for multigrid.
3. **Lanczos with M-inner product and shift-invert** through item 2:
   the sphere and cavity spectra at real sizes. **Done (2026-11-15):**
   `eigsh` (`routines/lanczos.mojo`, written from Ericsson–Ruhe and
   Wu–Simon), thick restart, locking plus a search of the locked
   complement (plain Lanczos missed copies of the icosphere's multiple
   eigenvalues), `deflate=G` for curl-curl kernels. Sphere example on
   it; 20 pairs of a 119,600-edge cavity in 1.1 s, backward error 4e-16.
   ResonantCavity stays dense (336 triangles).
4. **Smoothed aggregation** (amgcl's aggregation and P, Chebyshev–Jacobi
   with a CG-Lanczos λ, symmetric Gauss–Seidel), with a row-parallel
   deterministic SpGEMM for `PᵀAP`: the 834-iteration plate, Darcy's
   `selfp` Schur block, the H1 solves for AMS; then LOBPCG.
   **Done (2026-11-15):** `SmoothedAggregation` (`routines/amg.mojo`),
   aggregates and P equal to amgcl's bit for bit, Chebyshev on
   [0.1, 1.1]λ (default) or symmetric Gauss–Seidel, coarsest by
   `SparseCholesky`. 9–10 CG iterations from 4k to 355k unknowns; a
   regenerated graded plate (355,385 unknowns): 238 ms against Jacobi's
   732 iterations and 1.03 s; PyAMG 5.4.1 on the same matrix: 8–12
   iterations, 0.85 s. Not yet: parallel setup (serial SpGEMM), typed
   overload for `cg`, Darcy `selfp`, LOBPCG.
5. **Geometric multigrid for k-forms**: vertex-star patch smoother
   (PETSc's PCPATCH design) and a V-cycle on the `refine()` hierarchy.
   **Done (2026-11-15):** `GeometricMultigrid` (`routines/gmg.mojo`),
   multiplicative and additive vertex-star smoothing, Galerkin levels
   from the exact Whitney prolongations, `dirichlet=`. CG 7–10
   iterations for every degree and size measured (2D up to 49k, 3D up
   to 51k unknowns); AFW block preconditioner with V-cycle blocks:
   MINRES 26–30 up to 66k. SA-AMG on H(curl) for comparison: 134 → 975.
   Not yet: parallel (additive, colored multiplicative) smoothing, W
   cycles, typed overloads.
6. **Supernodal factorization** when 3D factorization time matters.
   **Done (2026-11-16):** `layout=FactorLayout.AUTO|SIMPLICIAL|SUPERNODAL`
   on `SparseLDLT`/`SparseCholesky` and the typed wrappers
   (`routines/supernodal.mojo`: faer's supernode partition, relaxation
   and patterns; left-looking with our own 8 x 4 register-tiled kernel).
   One thread: 30³ factor 2.15 s -> 322 ms, solve 16.6 -> 6.3 ms; 2D 300²
   112 -> 37 ms. AUTO above 20 flops per nnz(L). Steppers' `_FACTOR_FILL`
   8 -> 24 (3D factored up to ~30³). Not yet: tree-parallel
   factorization, multiple right-hand sides. Then:
   **AMS/ADS** for unrefined meshes; threshold-pivoting multifrontal
   LDLᵀ only if regularization fails on non-well-centered meshes.

## Reading list

Free copies, where one exists:
- T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006
  (no authorized free copy). Free companions by the author: his 42
  recorded lectures on the book, with notes
  (`people.engr.tamu.edu/davis/classes.html`), and Davis, Rajamanickam,
  Sid-Lakhdar, "A survey of direct methods for sparse linear systems",
  Acta Numerica 2016, technical report version:
  `people.engr.tamu.edu/davis/publications_files/survey_tech_report.pdf`.
- J. W. H. Liu, "The role of elimination trees in sparse factorization",
  SIAM J. Matrix Anal. Appl. 11(1), 134–172, 1990,
  doi:10.1137/0611010 (no free copy found; the survey covers it).
- E. G. Ng, B. W. Peyton, "Block sparse Cholesky algorithms on advanced
  uniprocessor computers", SIAM J. Sci. Comput. 14(5), 1993; report
  ORNL/TM-11960: `osti.gov/servlets/purl/6097949`.
- J. R. Gilbert, E. G. Ng, B. W. Peyton, "An efficient algorithm to
  compute row and column counts for sparse Cholesky factorization",
  SIAM J. Matrix Anal. Appl. 15(4), 1994; report ORNL/TM-12195:
  `digital.library.unt.edu/ark:/67531/metadc1450948/m2/1/high_res_d/7258591.pdf`.
- P. Amestoy, T. A. Davis, I. Duff, "An approximate minimum degree
  ordering algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996 (report
  TR-94-039, University of Florida); AMD's user guide describes the
  code.
- D. N. Arnold, R. S. Falk, R. Winther, "Multigrid in H(div) and
  H(curl)", Numer. Math. 85, 197–217, 2000 (the vertex-star smoother).
- P. E. Farrell, M. G. Knepley, L. Mitchell, F. Wechsung, "PCPATCH:
  software for the topological construction of multigrid relaxation
  methods", ACM TOMS 47(3), 2021 (arXiv:1912.08516).
- R. Hiptmair, J. Xu, "Nodal auxiliary space preconditioning in
  H(curl) and H(div) spaces", SIAM J. Numer. Anal. 45(6), 2007.
- S.-C. T. Choi, C. C. Paige, M. A. Saunders, "MINRES-QLP: a Krylov
  subspace method for indefinite or singular symmetric systems", SIAM
  J. Sci. Comput. 33(4), 2011 (arXiv:1003.4042).
- P. F. Fischer, "Projection techniques for iterative solution of
  Ax = b with successive right-hand sides", Comput. Methods Appl. Mech.
  Engrg. 163, 1998.
- M. Lottes, "Optimal polynomial smoothers for multigrid V-cycles",
  Numer. Linear Algebra Appl. 30(6), 2023 (arXiv:2202.08830).
