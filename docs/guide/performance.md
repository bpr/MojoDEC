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
2026-11-11: MojoDEC `03e097c` plus the changes that come with this page
(parallel topology, scratch space without false sharing, no serial
fills of large outputs), on Mojo 1.3.0.dev2026110305; dexterior
`9b0f379`, formoniq `710139e`, rustc 1.98.1. In brackets: the time
relative to MojoDEC with the same number of threads. PyDEC was not rerun
on this machine; its ratios below come from an earlier run on an AMD
Ryzen 7 9800X3D (MojoDEC `c7bbcce`), which is about twice as fast as
this laptop throughout.

### 320k triangles (`tri_400`)

| Phase | MojoDEC, 1 thread | MojoDEC, all cores | dexterior | formoniq, 1 thread | formoniq, all cores |
|---|---:|---:|---:|---:|---:|
| topology | 58 ms | 34 ms | — | 723 ms (12×) | 719 ms (21×) |
| geometry | 132 ms | 42 ms | — | (68 ms) | (69 ms) |
| build | 174 ms | 62 ms | 331 ms (1.9×) | — | — |
| Whitney, k = 0 | 35 ms | 13 ms | — | 664 ms (19×) | 161 ms (12×) |
| Whitney, k = 1 | 76 ms | 24 ms | — | 892 ms (12×) | 200 ms (8.5×) |
| Whitney, k = 2 | 33 ms | 6.5 ms | — | 581 ms (18×) | 65 ms (10×) |

### 384k tetrahedra (`tet_40`)

| Phase | MojoDEC, 1 thread | MojoDEC, all cores | dexterior | formoniq, 1 thread | formoniq, all cores |
|---|---:|---:|---:|---:|---:|
| topology | 172 ms | 76 ms | — | 2.30 s (13×) | 2.31 s (31×) |
| geometry | 695 ms | 174 ms | — | (79 ms) | (78 ms) |
| build | 828 ms | 203 ms | 1.47 s (1.8×), NaN | — | — |
| Whitney, k = 0 | 95 ms | 22 ms | — | 1.21 s (13×) | 372 ms (17×) |
| Whitney, k = 1 | 268 ms | 65 ms | — | 2.87 s (11×) | 893 ms (14×) |
| Whitney, k = 2 | 205 ms | 49 ms | — | 3.11 s (15×) | 559 ms (11×) |
| Whitney, k = 3 | 100 ms | 18 ms | — | 1.05 s (10×) | 122 ms (6.9×) |

The smaller meshes (20k and 80k triangles, 6k and 48k tetrahedra) give
similar ratios, a little smaller on all cores, where a parallel loop's
fixed cost weighs more (formoniq's Whitney assembly is 3.7× slower on
`tet_10`); `pixi run bench-compare` prints them all.

## Reading the results

- **PyDEC.** Its topology is vectorized NumPy and only about 3× slower
  (on the Ryzen, before MojoDEC's topology was parallel). Its geometry
  walks every top simplex in Python (`compute_dual_volume`) and is
  500–1000× slower: over three minutes for 384k tetrahedra on the
  Ryzen. Whitney matrices are 10–400× slower, worst for the top degree.
- **dexterior.** Builds topology and geometry in one serial call,
  1.6–1.9× slower than MojoDEC on one thread. On every 3D mesh some of
  its vertex and edge dual volumes are NaN (519 of 1331 vertices on
  `tet_10`): it takes the volume of each piece of a dual cell as
  `sqrt(det(VᵀV))`, a Gram determinant, which rounds below zero for
  nearly flat pieces. MojoDEC uses Householder QR for volumes to avoid
  exactly this.
- **formoniq.** Whitney assembly is 10–19× slower than MojoDEC on one
  thread and 5–18× on all cores on the two large meshes (8–19× and
  3.7–18× over all six). Short all-core phases vary by up to 25% from
  run to run on this laptop: a second run, with the same code, gave
  `tet_40` Whitney k = 1, 2, 3 74, 61, and 22 ms on all cores, against
  65, 49, and 18 ms above. Its topology is 5–13× slower on one thread. Its `geometry`
  is only the squared edge lengths (it computes metric quantities per
  element during assembly and has no circumcentric dual), so those
  times, in parentheses, are not comparable with the others'.

## What it says about MojoDEC

Each change below was timed against the build before it, alternating
the two binaries in one session (`benchmarks/compare/ab.py`), and gives
results equal to the last bit: the same simplices, `d`, and boundary
matrices at every level, the same circumcenters and primal and dual
volumes, and Whitney matrices with the same `indptr`, `indices`, and
data bits, on one thread and on all cores.

- **Dual volumes in parallel** (vertex ownership, `e400d86`): geometry on
  all cores 0.59–0.69× of before on `tri_400`, `tet_20`, `tet_40`; one
  thread 0.87–0.96×.
- **Whitney assembly in parallel** (`03e097c`). The element loop also
  finds each top simplex's k-faces from the rows of `d` (no
  `massmatrix_rowcols`, whose two COO index lists were 220 MB for
  `tet_40`, k = 1), and each row of the matrix is summed in parallel
  from the top simplices containing its face, in increasing order, the
  order `CSRMatrix.from_coo` sums duplicates in. k < n: 0.33–0.63× on all
  cores, 0.75–0.90× on one thread.
- **One chunk per logical core** (`03e097c`), except the sparse
  matrix-vector product (`memory_bound=True`), which 16 chunks made 15%
  slower. Geometry 0.52–0.81×, Whitney k = n 0.47–0.81× on all cores.
- **Scratch space without false sharing.** Each chunk's small scratch
  lists could share a cache line with another chunk's, so writing them
  in the inner loop made the 3D dual walk slower on two threads (619 ms)
  than on one (463 ms). `chunk_scratch` leaves 128 bytes after each list;
  the dual walk on `tet_40` went from 149 to 77 ms on all cores, and
  geometry 0.70–0.96× overall. Passing the per-element helpers
  (`frame_dual_basis`, `parallelotope_volume`) their scratch instead of
  allocating it: geometry another 0.75–0.93× and Whitney k = 1, 2
  0.86–0.95×, on one thread too.
- **Topology in parallel.** `simplex_array_boundary` puts face occurrences
  into one bucket per first vertex (a parallel counting sort), sorts
  each bucket, and reads the faces, `d`, and the boundary matrix off the
  runs of equal faces, with no lexicographic sort of all faces and no
  transpose; `require_distinct_simplices` skips its sort when the
  elements are already in increasing order, and parities and row sorts
  run in parallel. Topology 0.32–0.53× on all cores and 0.76–0.84× on one
  thread (the larger meshes), build 0.58–0.79×.
- **No serial fills of large outputs.** Buffers the parallel loops write
  in full are no longer zero-filled first, a memset on one thread that
  also took every page fault there: the first call of
  `whitney_innerproduct` for each k on `tet_40` went from 439 to 191 ms
  in all, and repeated calls 0.65–0.92× on the larger meshes.

From the first committed parallel results on this page (`e400d86`, same
machine) to now, on all cores: `tet_40` build 662 → 203 ms, Whitney k = 1
281 → 65 ms; `tri_400` build 145 → 62 ms, Whitney k = 1 73 → 24 ms. All
cores now speed up topology 1.7–2.3×, geometry 3.1–4.0×, and Whitney
2.7–5.6× over one thread on the large meshes.

Where the time goes now, on `tet_40`:

- **Dual volumes** are the largest piece of geometry: 454 ms on one
  thread, 80 ms on all cores (5.7×). Circumcenters take 155 → 31 ms and
  primal volumes 49 → 7 ms.
- **Topology** is 76 ms on all cores, a third of `build`; what is left
  serial in it is the prefix sums over vertices, the mesh copy and
  orientation check, and the CSR bookkeeping between the parallel passes.
- **First calls** still pay page faults the warm benchmark does not see:
  a fresh `SimplicialComplex(V, E)` on `tet_40` took 244 ms against the
  203 ms best of three.
