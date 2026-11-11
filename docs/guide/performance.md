# Performance against other DEC libraries

How long MojoDEC takes to build a complex, its geometry, and the Whitney
mass matrices, next to three other libraries given the same meshes:

- [PyDEC](https://github.com/hirani/pydec): Python with NumPy and SciPy,
  the library MojoDEC follows;
- [dexterior](https://codeberg.org/molentum/dexterior): Rust, DEC with
  diagonal Hodge stars;
- [formoniq](https://github.com/luiswirth/formoniq): Rust, finite
  element exterior calculus with Whitney forms, parallel assembly with
  rayon.

The harness is in `benchmarks/compare/`; its README lists the versions,
what each phase calls in each library, and how the checks work. Run it
with

```sh
pixi run bench-compare
```

## The method

Every library reads the same files: Kuhn grids of the unit square and
cube with the interior vertices jittered by up to 5% of the cell size, so
that no dual volume is exactly zero (the meshes are not well-centered, so
some are negative). With each time, every library prints check values:
simplex counts, the sum of each primal star's diagonal, and the Frobenius
norm of each Whitney matrix. They agree to at least 1e-9 relative (mostly
to the last few digits) wherever a library computes the quantity, so the
libraries are doing the same arithmetic, not just similar work.

Times are the best of three runs, in each library's own release build
(Rust with `target-cpu=native`, as Mojo compiles for the host CPU).
MojoDEC and formoniq run once on one thread and once on all cores;
`MOJODEC_NUM_THREADS=1` makes MojoDEC's parallel loops serial (see
`core/parallel.mojo`). PyDEC and dexterior are single-threaded.

## Results

AMD Ryzen 7 9800X3D (8 cores, 16 threads), 2026-11-11: MojoDEC
`c7bbcce` on Mojo 1.3.0.dev2026110305, PyDEC `c181202`, dexterior
`9b0f379`, formoniq `710139e`, rustc 1.99.1. In brackets: the time
relative to MojoDEC with the same number of threads.

### 320k triangles (`tri_400`)

| Phase | MojoDEC, 1 thread | MojoDEC, all cores | PyDEC | dexterior | formoniq, 1 thread | formoniq, all cores |
|---|---:|---:|---:|---:|---:|---:|
| topology | 29 ms | 31 ms | 78 ms (2.7×) | — | 233 ms (7.9×) | 215 ms (7.0×) |
| geometry | 65 ms | 53 ms | 57.2 s (886×) | — | (54 ms) | (30 ms) |
| build | 87 ms | 80 ms | 57.2 s (654×) | 140 ms (1.6×) | — | — |
| Whitney, k = 0 | 20 ms | 19 ms | 579 ms (28×) | — | 309 ms (15×) | 65 ms (3.4×) |
| Whitney, k = 1 | 43 ms | 30 ms | 5.24 s (122×) | — | 379 ms (8.8×) | 80 ms (2.6×) |
| Whitney, k = 2 | 20 ms | 10 ms | 7.96 s (394×) | — | 231 ms (11×) | 27 ms (2.7×) |

### 384k tetrahedra (`tet_40`)

| Phase | MojoDEC, 1 thread | MojoDEC, all cores | PyDEC | dexterior | formoniq, 1 thread | formoniq, all cores |
|---|---:|---:|---:|---:|---:|---:|
| topology | 125 ms | 125 ms | 376 ms (3.0×) | — | 941 ms (7.5×) | 941 ms (7.5×) |
| geometry | 390 ms | 337 ms | 200 s (512×) | — | (37 ms) | (45 ms) |
| build | 539 ms | 410 ms | 200 s (372×) | 666 ms (1.2×), NaN | — | — |
| Whitney, k = 0 | 96 ms | 60 ms | 782 ms (8.1×) | — | 553 ms (5.7×) | 161 ms (2.7×) |
| Whitney, k = 1 | 269 ms | 197 ms | 8.09 s (30×) | — | 1.22 s (4.5×) | 369 ms (1.9×) |
| Whitney, k = 2 | 195 ms | 115 ms | 23.2 s (119×) | — | 1.29 s (6.6×) | 245 ms (2.1×) |
| Whitney, k = 3 | 83 ms | 24 ms | 13.4 s (161×) | — | 452 ms (5.4×) | 52 ms (2.2×) |

The smaller meshes (20k and 80k triangles, 6k and 48k tetrahedra) give
similar ratios, within a factor of about 2; `pixi run bench-compare`
prints them all.

## Reading the results

- **PyDEC.** Its topology is vectorized NumPy and only about 3× slower.
  Its geometry walks every top simplex in Python (`compute_dual_volume`)
  and is 500–1000× slower: over three minutes for 384k tetrahedra.
  Whitney matrices are 10–400× slower, worst for the top degree.
- **dexterior.** Builds topology and geometry in one serial call, 1.2–2×
  slower than MojoDEC on one thread. On every 3D mesh some of its vertex
  and edge dual volumes are NaN (519 of 1331 vertices on `tet_10`): it
  takes the volume of each piece of a dual cell as `sqrt(det(VᵀV))`, a
  Gram determinant, which rounds below zero for nearly flat pieces.
  MojoDEC uses Householder QR for volumes to avoid exactly this.
- **formoniq.** Whitney assembly is 5–15× slower than MojoDEC on one
  thread and 2–5× slower on all cores: formoniq parallelizes well,
  MojoDEC starts from a faster serial kernel. Its `geometry` is only the
  squared edge lengths (it computes metric quantities per element during
  assembly and has no circumcentric dual), so those times, in
  parentheses, are not comparable with the others'.

## What it says about MojoDEC

- **Parallel scaling is weak.** On 8 cores, geometry speeds up by only
  1.2× and the Whitney matrices by 1.05–3.5×. Topology and dual volumes
  are serial, and in `whitney_innerproduct` only the element loop is
  parallel: the sparsity pattern (`massmatrix_rowcols`) and the CSR
  conversion (`CSRMatrix.from_coo`) are not. formoniq gains 3–9× from
  the same cores.
- **Geometry is now the largest phase**, three times topology in 3D. On
  `tet_40`, one thread, its 395 ms split into circumcenters 88 ms, primal
  volumes 53 ms, and the serial dual-volume walk 245 ms.

These are the places to look for the next speedups.
