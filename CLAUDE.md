# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MojoDEC is a port of PyDEC (Discrete Exterior Calculus) to Mojo 1.3, using
NuMojo arrays in place of NumPy and its own `CSRMatrix` in place of
`scipy.sparse`. README.md has the full MojoDEC ↔ PyDEC file mapping, the list
of deliberate differences from PyDEC, and the features added beyond PyDEC.

## Commands

All tooling runs through pixi (linux-64 only):

```sh
pixi install                          # Mojo 1.3 nightly + MAX, Python, NumPy, matplotlib
pixi run numojo                       # precompile vendored NuMojo -> build/numojo.mojoc
pixi run test                         # every tests/test_*.mojo (builds numojo first)
pixi run examples                     # all examples; figures saved as PNG
pixi run examples DarcyFlow --show    # one example, figures in windows
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

Each test file ends with `TestSuite.discover_tests[__functions_in_module()]().run()`,
so any `def test_*() raises` in the file is picked up automatically. Test
mesh fixtures live in `tests/meshes/`. Examples run from their own directory
(`examples/run.sh` cds into it) and write outputs that `.gitignore` excludes.

## Toolchain constraints

- `pixi.toml` pins the nightly `mojo ==1.3.0.dev2026110305` and the matching
  `max-core`; change both together when moving versions.
- NuMojo v0.12.0 is vendored in `third_party/numojo` with a Mojo 1.3 patch
  (`third_party/numojo-mojo-1.3.patch`; see `third_party/README.md`) because
  the published package requires Mojo < 1.3. Code is compiled against
  `build/numojo.mojoc`, so rebuild it (`pixi run numojo`) after touching the
  vendored copy, and keep the patch file in sync with any change there.
- NuMojo's `linalg.solve` (no pivoting) and Jacobi `eigh` are unsuitable for
  DEC systems; use MojoDEC's own `routines/dense.mojo` (pivoted LU,
  Householder/QL `eigh`) and `routines/linalg.mojo` (`cg`, `lsqr`, dense
  `spsolve`, `eigh_generalized`).

## Architecture

Layout follows NuMojo: `mojodec/core` holds data structures,
`mojodec/routines` holds algorithms, `mojodec/vis` holds plotting.
`mojodec/__init__.mojo` re-exports the public API (add new public names
there); `mojodec/prelude.mojo` is the small common subset.

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
- **Cochains** (`core/cochain.mojo`): PyDEC's dual-typed `cochain.v` is split
  into `Cochain` (NuMojo vector) and `CochainBasis` (`CSRMatrix`), both
  implementing the `CochainLike` trait. `d`, `star`, `delta`,
  `laplace_derham`, `laplace_beltrami` are generic over `CochainLike`.
- **Operators**: on `CSRMatrix`, `A @ B` / `A @ x` are matrix products
  (SciPy's `A * B`); `A * s` scales. Hodge stars (`sc[k].star`,
  `star_inv`) are `DiagonalOperator` (`core/diagonal.mojo`): `D @ A` scales
  rows, `A @ D` (via `__rmatmul__`) scales columns; `to_csr()` when a
  `CSRMatrix` is required. `CochainLike.mapped` is overloaded for both.
- **`Subset`** (`core/subset.mojo`): bitset of primal k-simplices or dual
  k-cells (indexed by primal level k or n-k). Boundary/coboundary walk the
  rows of `sc[m-1].d` / `sc[m+1].boundary`, swapped for dual cells.
  `CSRMatrix`/`DiagonalOperator` have `exclude_subset`/`restrict_to_subset`,
  `Cochain.overwrite`, `sc.boundary_subset(k)`. `subset.mojo` and
  `sparse.mojo` import each other, as do `cochain` and `simplicial_complex`.
- **Typed layer** (`core/typed.mojo`): `TypedComplex[n]` wraps a
  `SimplicialComplex` (`.sc`); `TypedCochain[n, k, primal]`,
  `MatrixOp`/`DiagonalOp[n, in_k, in_primal, out_k, out_primal]` wrap
  `RealArray`/`CSRMatrix`/`DiagonalOperator`. Composition is checked by
  `__matmul__` signatures; degree ranges by `comptime assert`. Ops built
  by `TypedComplex` record its complex (`Optional`, `None` when wrapped
  from a bare matrix) and refuse cochains and ops from another one. Mojo
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
  (`ascii`/`binary`/`appended`, pure Mojo), PLY, and Gmsh `.msh` input
  (4.1 and 2.2, ASCII and binary; physical groups -> `GmshMesh.subset`).
  The Gmsh fixtures in `tests/meshes/gmsh/` are generated from the `.geo`
  files there: `pixi exec --spec gmsh gmsh -2 square.geo -format msh41
  [-bin] -o square_v41[_bin].msh` (and `msh22`; `-3` for the cube).

## Conventions

- Public symbols get Google-style doc strings (`Args:`, `Parameters:`,
  `Returns:`, `Raises:`; see `docs/writing-docs.md`). `pixi run doc-check`
  must not get worse; after documenting a file, `pixi run doc-check
  --update` lowers the baseline.
- Features taken from Dxtr (wedge, flat, curvature, quality, etc.) are
  implemented from the published mathematics, not Dxtr's GPL code — keep it
  that way.
- The examples double as regression checks against PyDEC; README.md's
  Examples table lists the expected results (e.g. DarcyFlow pressures to
  1.5e-12, ResonantCavity eigenvalues m²+n², MakeMesh matches
  `examples/MakeMesh/ExpectedOutput/`).
