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

Intel Core i7-10875H laptop (8 cores, 16 threads, `powersave` governor),
2026-11-11: MojoDEC `e400d86` plus the changes that come with this page
(parallel Whitney assembly, one chunk per logical core), on Mojo
1.3.0.dev2026110305; dexterior `9b0f379`, formoniq `710139e`, rustc
1.98.1. In brackets: the time relative to MojoDEC with the same number of
threads. PyDEC was not rerun on this machine; its ratios below come from
the earlier run on an AMD Ryzen 7 9800X3D (MojoDEC `c7bbcce`), which is
about twice as fast as this laptop throughout.

### 320k triangles (`tri_400`)

| Phase | MojoDEC, 1 thread | MojoDEC, all cores | dexterior | formoniq, 1 thread | formoniq, all cores |
|---|---:|---:|---:|---:|---:|
| topology | 76 ms | 78 ms | — | 705 ms (9.2×) | 711 ms (9.2×) |
| geometry | 145 ms | 51 ms | — | (68 ms) | (68 ms) |
| build | 210 ms | 116 ms | 330 ms (1.6×) | — | — |
| Whitney, k = 0 | 44 ms | 18 ms | — | 649 ms (15×) | 144 ms (7.9×) |
| Whitney, k = 1 | 92 ms | 33 ms | — | 888 ms (9.7×) | 208 ms (6.4×) |
| Whitney, k = 2 | 44 ms | 13 ms | — | 566 ms (13×) | 66 ms (5.1×) |

### 384k tetrahedra (`tet_40`)

| Phase | MojoDEC, 1 thread | MojoDEC, all cores | dexterior | formoniq, 1 thread | formoniq, all cores |
|---|---:|---:|---:|---:|---:|
| topology | 225 ms | 222 ms | — | 2.27 s (10×) | 2.29 s (10×) |
| geometry | 742 ms | 248 ms | — | (79 ms) | (79 ms) |
| build | 935 ms | 440 ms | 1.45 s (1.6×), NaN | — | — |
| Whitney, k = 0 | 105 ms | 32 ms | — | 1.17 s (11×) | 365 ms (11×) |
| Whitney, k = 1 | 302 ms | 95 ms | — | 2.83 s (9.4×) | 824 ms (8.7×) |
| Whitney, k = 2 | 234 ms | 69 ms | — | 3.07 s (13×) | 584 ms (8.5×) |
| Whitney, k = 3 | 115 ms | 25 ms | — | 1.00 s (8.7×) | 118 ms (4.8×) |

The smaller meshes (20k and 80k triangles, 6k and 48k tetrahedra) give
similar ratios; `pixi run bench-compare` prints them all.

## Reading the results

- **PyDEC.** Its topology is vectorized NumPy and only about 3× slower.
  Its geometry walks every top simplex in Python (`compute_dual_volume`)
  and is 500–1000× slower: over three minutes for 384k tetrahedra on the
  Ryzen. Whitney matrices are 10–400× slower, worst for the top degree.
- **dexterior.** Builds topology and geometry in one serial call,
  1.4–1.6× slower than MojoDEC on one thread. On every 3D mesh some of its
  vertex and edge dual volumes are NaN (519 of 1331 vertices on
  `tet_10`): it takes the volume of each piece of a dual cell as
  `sqrt(det(VᵀV))`, a Gram determinant, which rounds below zero for
  nearly flat pieces. MojoDEC uses Householder QR for volumes to avoid
  exactly this.
- **formoniq.** Whitney assembly is 9–15× slower than MojoDEC on one
  thread and 5–11× slower on all cores. Its topology is 9–10× slower.
  Its `geometry` is only the squared edge lengths (it computes metric
  quantities per element during assembly and has no circumcentric dual),
  so those times, in parentheses, are not comparable with the others'.

## What it says about MojoDEC

Each change below was timed against the build before it, alternating
the two binaries in one session (`benchmarks/compare/ab.py`), and gives
results equal to the last bit: the same dual volumes, and Whitney
matrices with the same `indptr`, `indices`, and data bits.

- **Dual volumes in parallel** (vertex ownership, `e400d86`): geometry on
  all cores 0.59–0.69× of before on `tri_400`, `tet_20`, `tet_40`; one
  thread 0.87–0.96×.
- **Whitney assembly in parallel.** The element loop now also finds each
  top simplex's k-faces from the rows of `d` (no `massmatrix_rowcols`,
  whose two COO index lists were 220 MB for `tet_40`, k = 1), and each
  row of the matrix is summed in parallel from the top simplices
  containing its face, in increasing order, which is the order
  `CSRMatrix.from_coo` sums duplicates in. k < n: 0.33–0.63× of before on
  all cores, 0.75–0.90× on one thread; the top degree, a diagonal, is
  built directly and unchanged.
- **One chunk per logical core**, except the sparse matrix-vector
  product, which stays at one per physical core (`memory_bound=True`):
  with 16 chunks it made the 4.5M-entry CG solve 15% slower. Geometry
  takes 0.52–0.81× the time with 16 chunks, Whitney k = n 0.47–0.81×,
  the other k > 0 0.75–0.97× (`tet_10`, k = 1 measured 1.11× once and
  0.75× on a rerun of 8 rounds), and k = 0 is unchanged (0.98–1.05×).

On all cores geometry now speeds up 3.0× (`tet_40`) to 2.8× (`tri_400`)
and Whitney 2.4–4.6×. Where the time goes now:

- **Topology is serial** and about half of `build` on all cores (222 of
  440 ms on `tet_40`).
- **The 3D dual walk scales poorly.** On `tet_40` it takes 458 ms on one
  thread and 153 ms on all cores (3.0×; 5.7× on `tri_400`): a chunk walks
  every top simplex touching its vertex block, and its chains down from
  faces it does not own, so a tetrahedron is walked by up to four
  chunks. Circumcenters take 156 → 34 ms and primal volumes 102 → 27 ms.

These are the places to look for the next speedups.
