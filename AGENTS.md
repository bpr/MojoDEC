# AGENTS.md

Project context for AI coding assistants working on MojoDEC. Tool-specific
workflow instructions live in each tool's own file, e.g.
CLAUDE.md.

## What this is

MojoDEC is largely inspired by PyDEC (Discrete Exterior Calculus). It runs
on Mojo 1.3, using
NuMojo arrays in place of PyDEC's NumPy arrays and its own `CSRMatrix` in
place of PyDEC's `scipy.sparse` matrices. README.md has the full MojoDEC ↔
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
- `benchmarks/`: timings of the parallel loops.
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
  Householder/QL `eigh`) and `routines/linalg.mojo` (`cg`, `lsqr`, dense
  `spsolve`, `eigh_generalized`).

### Known toolchain bugs

NuMojo bugs MojoDEC has found are written up, with repros for its
maintainers, in `third_party/numojo-bugs.md`; add new ones there.

- Write floats as text with `io/scan.mojo`'s `write_float`, not
  `String(x)`: Mojo prints about one in five integral doubles between
  ~1.8e16 and 1e23 with a digit too few, so they don't read back, and
  some Float32 values too (widen them to Float64 first). Parse integer
  text with `atol`, not `atof`, which loses digits past 2^53.
- `nm.fromstring` reads `1e4` as 14 (it drops the `e`); write test
  literals without exponents or use `real_array_from_rows`.
- `assert_almost_equal` has a default `rtol=1e-5` on top of `atol`, so
  every call passes `rtol` explicitly (normally `rtol=0`, with an `atol`
  from the problem's conditioning); `tests/run_all.sh` fails on a call
  without one.
- Write `sqrt(x)` (`std.math`), never `x ** 0.5`: Mojo's `**` with a
  fractional exponent is off by up to ~5e-12 relative, which Cholesky and
  Gram-matrix solves amplify (it cost DarcyFlow two digits and Sphere's
  spectrum check three).

## Architecture

Key design points that span several files:

- **`SimplicialComplex`** (`core/simplicial_complex.mojo`) stores its
  `ComplexData` (mesh + one `SimplexLevel` per dimension) behind an
  `ArcPointer`, so copies and cochains share one complex like Python
  references. `sc[k]` gives a `SimplexLevel` with simplices, parity,
  `d`/`boundary` `CSRMatrix`es, `simplex_to_index(s)`/`index_to_simplex(i)`
  lookups (binary search over the sorted rows, no per-level dictionary), and geometry
  (circumcenters, primal/dual volumes, diagonal Hodge `star`/`star_inv`).
  Geometry is computed eagerly at construction, unlike PyDEC's lazy
  `__getattr__`; `compute_geometry=False` skips it for degenerate meshes.
  `simplex_array_boundary` builds `d` directly (k+1 entries per row) and
  `boundary` is its transpose. Circumcenters are closed-form up to
  triangles; higher dimensions solve PyDEC's system with positions
  relative to the first vertex, and from triangles up the Cartesian center
  is built relative to it too (absolute positions lose precision far from
  the origin). Embedded volumes use Householder QR
  (`parallelotope_volume`), never the Gram determinant, and barycentric
  gradients likewise (`dual_basis`, `frame_dual_basis` in `core/utils.mojo`),
  never `(E E^T)^-1 E`. Dual volumes come from one walk down the chains of
  circumcenters (`_add_dual_pieces`); `dual_edge_piece_values` runs the
  same walk to keep the edge pieces per top simplex, which `flat` and
  `dual_edge_pieces` use. In loops, look faces up with
  `level.find_vertices(sorted_list)`, not `simplex_to_index(Simplex(...))`.
- **Cochains** (`core/cochain.mojo`): PyDEC's dual-typed `cochain.v` is split
  into `Cochain` (NuMojo vector) and `CochainBasis` (`CSRMatrix`), both
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
  implemented on top of `routines/topology.mojo`.
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
  can overflow. Never index a table by a number read from a file (Gmsh
  node tags are sorted and searched), and bound sizes the file cannot
  vouch for (`read_sparse`'s `max_dim`). VTK ASCII output raises on NaN
  and infinity, which VTK's text reader cannot parse.
  The Gmsh fixtures in `tests/meshes/gmsh/` are generated from the `.geo`
  files there: `pixi exec --spec gmsh gmsh -2 square.geo -format msh41
  [-bin] -o square_v41[_bin].msh` (and `msh22`; `-3` for the cube).

### Parallel loops

`parallel_ranges(n, serial_below, body)` (`core/parallel.mojo`) runs
`body(chunk, lo, hi)` on one chunk per physical core, or serially when
`n < serial_below`. A MAX `parallelize` call costs about 200 µs whatever the
work (POSIX threads cost 20–50 µs), so thresholds are set where the serial
loop takes a few hundred µs: matvec at `PARALLEL_NNZ` (1M stored entries),
circumcenters and primal volumes at 50k simplices, `whitney_innerproduct`
at 4096 top simplices. Bodies must not raise: record failures per chunk
and raise afterwards. Keep a hot serial loop out of a function that also
holds the parallel branch (`CSRMatrix._matvec_parallel` is separate): in
the same function the serial loop ran about 15% slower. Dual volumes stay
serial (each top simplex adds into shared faces). `tests/test_parallel.mojo`
uses meshes past every threshold; the other tests' meshes run serially.

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
  `examples/MakeMesh/ExpectedOutput/`). DarcyFlow, HeatConduction, and
  ResonantCavity use the typed layer. HeatConduction imposes Dirichlet rows
  with `exclude_subset(fixed.dual())` plus `tc.dual_diagonal[...]`
  restricted to the same cells; DarcyFlow solves the mixed `BlockOp` with
  `spsolve(..., values=, fixed_first=, fixed_second=)`, which reproduces
  PyDEC's elimination bit for bit.
