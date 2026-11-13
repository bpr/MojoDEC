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
  primal volumes 49 → 7 ms. (The dual walk has since gone breadth first:
  116 ms and 34 ms; see [below](#the-dual-walk-breadth-first).)
- **Topology** is 76 ms on all cores, a third of `build`; what is left
  serial in it is the prefix sums over vertices, the mesh copy and
  orientation check, and the CSR bookkeeping between the parallel passes.
- **First calls** still pay page faults the warm benchmark does not see:
  a fresh `SimplicialComplex(V, E)` on `tet_40` took 244 ms against the
  203 ms best of three.

## Shuffled vertex numbers

A Kuhn grid numbers its vertices in spatial order. `tet_40_shuffled` and
`tri_400_shuffled` are the same meshes with their vertex numbers randomly
permuted (element order kept), the worst a mesh file can do. The results
are the same up to the last bits (sums in another order), and every check
still agrees across libraries. MojoDEC's times (best of 4 alternating
rounds, `ca6c9f9`, same laptop), shuffled over ordered:

| `tet_40` | 1 thread | all cores |
|---|---:|---:|
| topology | 172 → 447 ms (2.6×) | 68 → 200 ms (3.0×) |
| geometry | 690 → 770 ms (1.1×) | 154 → 281 ms (1.8×) |
| … dual walk | 454 → 539 ms (1.2×) | 78 → 200 ms (2.6×) |
| build | 823 → 1191 ms (1.45×) | 194 → 439 ms (2.3×) |
| Whitney k = 0, 1, 2, 3 | 2.5×, 1.75×, 1.26×, 1.0× | 1.7×, 1.35×, 1.4×, 1.2–1.4× (13–18 ms, noisy) |

`tri_400` is alike: build 1.8× on one thread and 2.5× on all cores.

Two costs, both from numbering:

- **The dual walk repeats work.** Each chunk walks every top simplex with
  a vertex in its block of vertex numbers. With 16 chunks a tetrahedron
  is walked 3.64 times on average when shuffled, against 1.38 ordered
  (triangles 2.82 against 1.04), so the walk on all cores takes 2.6× as
  long; on one thread there is a single block and only locality is lost.
- **Locality everywhere else.** Face numbers follow vertex numbers, and the
  loops that run in face order (topology's bucket sorts, the Whitney rows
  for k = 0, 1) read per-element data scattered over the whole mesh.
  Topology on one thread: the bucket sorts take 2–2.5× as long, and
  `require_distinct_simplices` 1.6 → 48 ms, since the sorted element rows
  are no longer in increasing order and it falls back to a sort. Whitney
  k = n runs in element order and is unaffected.

The other libraries slow down too (dexterior's build 1.46 → 1.93 s,
formoniq's topology 1.7×, its Whitney 1.1–1.3×), so MojoDEC still leads,
by 1.6× (dexterior's build) to 19× (formoniq).

**Renumbering fixes most of it.** The shuffled meshes renumbered in Morton
order of their vertex coordinates (vertices only, elements kept in their
order) against the ordered ones: geometry 1.03× on one thread and 0.93× on
all cores (the walk revisits 1.18 times per tetrahedron, fewer than the
grid's lexicographic numbering), Whitney 1.0–1.15× on one thread except
k = 0 on `tet_40` (1.48×), build 1.1×. Topology is still 1.2–1.5×, as the
element order no longer matches the vertex order; sorting the elements
too (by their first vertex, say) would be needed for that.

**`reorder_vertices`** does both: Morton order for the vertices and the
elements sorted by their sorted vertex numbers (see
[Vertex numbering](vertex-numbering.md)). On `tet_40_shuffled`, reordered
against `tet_40` (best of 3 alternating rounds): build 1.06× on one thread
and 1.00× on all cores, topology 1.21× and 1.11×, the dual walk on all
cores 0.82× (69 against 84 ms), Whitney k = 0 1.33× on one thread and the
rest within 1.05×. `tri_400_shuffled` reordered: build 1.04× and 0.98×.
Reordering takes 30–40 ms.

## Sorting without branches

Most of topology is sorting: each element's vertices, the parity of
that sort, and in `simplex_array_boundary` the face occurrences in each
bucket (one bucket per first vertex). Comparison sorts branch on the
data, and how often those branches mispredict depends on the vertex
numbering, which is why a Hilbert-numbered mesh had the slowest
topology (see [Vertex numbering](vertex-numbering.md)). Three changes take the branches out:

- **Rows of 3 and 4** are sorted by fixed sorting networks of min/max
  exchanges, and their parity is the parity of the number of pairs out
  of order, counted with comparisons that add 0 or 1. Longer rows use an
  odd-even transposition network and the same count.
- **Each face occurrence is one integer key**: the face's entries after
  the first, then the simplex and the entry left out, when that fits in
  62 bits (up to about 4M tetrahedra in 3D). A bucket then sorts integers
  instead of rebuilding both faces from the simplex array at every
  comparison, and the runs of equal faces are marked once, while
  counting them.
- **Buckets of 17 to 64 keys** are sorted by rank: each key's place is the
  number of smaller keys, counted 4 at a time in SIMD, more comparisons
  than insertion sort but none that branch. Smaller buckets keep
  insertion sort, larger ones merge.

`tet_40` and `tri_400`, before and after (best of 3 alternating rounds,
same laptop), in ms:

| Mesh | topology, 1 thread | topology, all cores | build, 1 thread | build, all cores |
|---|---:|---:|---:|---:|
| `tet_40` | 172 → 127 | 68 → 65 | 825 → 779 | 200 → 188 |
| `tet_40`, Morton order | 209 → 131 | 72 → 62 | 873 → 797 | 180 → 178 |
| `tet_40`, Hilbert order | 258 → 139 | 80 → 66 | 927 → 807 | 199 → 178 |
| `tet_40_shuffled` | 440 → 284 | 199 → 178 | 1196 → 1031 | 454 → 433 |
| `tri_400` | 58 → 47 | 32 → 31 | 171 → 161 | 57 → 62 |
| `tri_400`, Morton order | 51 → 49 | 35 → 30 | 178 → 167 | 56 → 56 |
| `tri_400_shuffled` | 156 → 133 | 93 → 91 | 311 → 289 | 144 → 135 |

The results are the same, entry for entry. On one thread, topology on
`tet_40` is 0.54–0.74× in every numbering, and a Morton-numbered mesh
is within 4% of the grid's own numbering (it was 1.22×). On all cores
the sorts were already a smaller part, and topology is 0.83–0.98× (the
`tri_400` build, 57 → 62 ms, is within its round-to-round noise of
57–62 ms). Whitney is unchanged.

## The dual walk, breadth first

`compute_dual_volume` walks down every chain of faces from each top
simplex (for a tetrahedron: 4 triangles, 12 triangle-edge pairs, 24
triangle-edge-vertex chains) and, for each, orthogonalizes the edge from
the top simplex's circumcenter to the face's against the chain's edges
so far. The piece it adds is the chain's volume. Depth first, each step
waits on its parent's square root and division, and the processor sees
only a few pieces at a time: about 30 ns a piece, 455 ms on one thread
for `tet_40`, more than half of `build`.

For triangles in 2D and 3D and tetrahedra in 3D, the walk now runs
breadth first: all chains of a level at once, four to a SIMD vector
(1, 3, and 6 vectors for a tetrahedron), each lane doing the scalar
walk's arithmetic. The chains of a level are independent, so their
square roots and divisions overlap. The face masks, the chains, and the
gathering of each top simplex's faces are fixed at compile time, so the
loops unroll with constant offsets. The pieces are then added to the
dual volumes level by level in the depth-first order, so every sum is
added in the same order: the dual volumes equal the old ones bit for
bit, on one thread and on all cores. Other dimensions keep the
recursive walk.

One trap on the way: Mojo lets LLVM fuse `a * b + c` into one rounding,
and whether it does depends on the code around it. The old loops fused
all three of their multiply-adds (the dot products' accumulations and
the projection's update), and the first unrolled version fused
different ones, which moved the last bits. Both walks now write those
as explicit fused multiply-adds.

`tet_40` and `tri_400`, before and after (best of 3 alternating rounds,
same laptop; the dual walk timed alone), in ms:

| Mesh | dual walk, 1 thread | dual walk, all cores | build, 1 thread | build, all cores |
|---|---:|---:|---:|---:|
| `tet_40` | 454 → 116 | 83 → 34 | 779 → 444 | 192 → 152 |
| `tet_40`, reordered | 478 → 127 | 75 → 26 | 799 → 458 | 176 → 131 |
| `tet_40_shuffled` | 541 → 178 | 203 → 90 | 1032 → 674 | 447 → 283 |
| `tri_400` | 69 → 21 | 12 → 5.8 | 162 → 114 | 58 → 49 |
| `tri_400`, reordered | | | 167 → 118 | 60 → 49 |
| `tri_400_shuffled` | | | 291 → 225 | 147 → 124 |

On one thread the walk is 3.0–3.9× faster and `build` 0.57–0.77× of
before; geometry on `tet_40` went from 690 to 355 ms, and circumcenters
(155 ms) are now its largest piece. On all cores the walk is 2.1–2.9×
faster and `build` 0.63–0.84×. Topology and Whitney are unchanged.

## Closed-form circumcenters of tetrahedra

Tetrahedra in 3D used to get their circumcenters from PyDEC's 5 × 5
system `[[2 P Pᵀ, 1], [1ᵀ, 0]]`, solved with partial pivoting. They now
use the closed form: with edges `a`, `b`, `c` from the first vertex and
`D = a · (b × c)`, the center is the first vertex plus
`(|a|² b × c + |b|² c × a + |c|² a × b) / 2D`, and the barycentric
weights are its dot products with the face normals over `D`, vertex 0's
from its own face's normal rather than as 1 minus the others. It is
faster and much more accurate: the system's matrix is a Gram matrix,
which squares a sliver's conditioning. Against exact rational
arithmetic on 3000 random tetrahedra flattened down to 1e-10 of their
size, the closed form's center was within 1e-11 circumradii throughout,
the system's off by 3e-7 on the best-shaped ones and up to 30 on the
thinnest. Tetrahedra in 4D and 5D keep the system.

Triangles' circumcenters no longer allocate scratch for their QR on
every triangle; their results are unchanged, bit for bit.

The circumcenters, and from them the dual volumes and the diagonal
Hodge stars, change in the last bits: at most 1.5e-14 of each level's
largest dual volume on `tet_40`. On one thread, tetrahedra's
circumcenters take 70 → 9.2 ms on `tet_40`, triangles' 73 → 49 ms on
`tet_40` and 26 → 16 ms on `tri_400`. Best of 3 alternating rounds, in
ms:

| Mesh | geometry, 1 thread | geometry, all cores | build, 1 thread | build, all cores |
|---|---:|---:|---:|---:|
| `tet_40` | 355 → 271 | 109 → 90 | 446 → 362 | 140 → 122 |
| `tet_40`, reordered | 367 → 284 | 100 → 79 | 460 → 377 | 143 → 127 |
| `tet_40_shuffled` | 434 → 348 | 148 → 127 | 684 → 596 | 303 → 307 |
| `tri_400` | 82 → 72 | 33 → 26 | 115 → 104 | 49 → 49 |

On all cores, `build` is within its noise on `tet_40_shuffled` and
`tri_400`, where topology dominates.

## Closed-form areas and volumes

Triangle circumcenters and the primal volumes of triangles in 3D took
their areas from a Householder QR of the two edges, and tetrahedra in
3D their volumes from an LU determinant with pivoting. They now use
`|u × w|` and `a · (b × c)`. Each component of a cross product is a 2 × 2
minor, as well conditioned as QR's distance from one edge to the line of
the other, so slivers lose nothing (the Gram form `|u|²|w|² - (u · w)²`
would cancel). Against exact rational arithmetic on 3000 triangles in
3D flattened down to 1e-11 of their size, the areas and barycentric
weights are 2–4 times more accurate than QR's, and the circumcenters and
tetrahedron volumes as accurate; both lose accuracy as `eps L / h`, from
rounding the inputs' differences. Edge lengths and triangles in 2D
already had closed forms; they now skip the general loops too.

On one thread, triangle circumcenters take 49 → 14 ms on `tet_40` and
16 → 4.1 ms on `tri_400`; primal volumes of triangles in 3D 25 → 6.1 ms
and of tetrahedra 17 → 2.9 ms. Best of 3 alternating rounds, in ms:

| Mesh | geometry, 1 thread | geometry, all cores | build, 1 thread | build, all cores |
|---|---:|---:|---:|---:|
| `tet_40` | 272 → 203 | 88 → 85 | 359 → 295 | 129 → 120 |
| `tet_40_shuffled` | 351 → 282 | 126 → 123 | 597 → 541 | 272 → 274 |
| `tri_400` | 72 → 60 | 27 → 26 | 105 → 93 | 45 → 50 |
| `tri_400_shuffled` | 102 → 87 | 39 → 38 | 218 → 205 | 121 → 121 |

On all cores geometry was already short, and `build` is within its
noise (topology, unchanged, varies by several ms between rounds).

The results change in the last bits, and one test had to change with
them: a triangle of height 1e-8 checked that its Whitney top-form mass
times its area is 1 to 1e-14, which held only because both came from
the same QR. The area is now the more accurate of the two (5e-10 from
exact against QR's 7e-9), and the two agree to 5e-9, within the
conditioning `eps L / h = 2e-8`; the test now checks the area against its
exact value and the product to 1e-7.

## Whitney matrices

`whitney_innerproduct` loops over the top simplices, computing each
one's barycentric gradients (Householder QR of its edges), the minors of
their Gram matrix by Cauchy-Binet, and its local matrix as signed sums
of those minors from a precomputed table; then it assembles the rows of
the sparse matrix, each from the top simplices containing its face. Two
changes, with the same matrices bit for bit:

- **The element loop is unrolled** for triangles in 2D and 3D and
  tetrahedra in 3D: the dimensions, the degree, and the face masks are
  known at compile time, the small matrices stay on the stack, and the
  general loop's arithmetic is kept operation for operation, with the
  multiply-adds the compiler fused there written as explicit fused
  multiply-adds. On `tet_40` with one thread it takes 15, 72, 74, and
  55 ms for k = 0 to 3, against 50, 153, 155, and 101.
- **Assembly writes the result directly.** It summed each row into a
  scratch array of `fps` slots per term (two arrays of 110 MB for
  `tet_40`, k = 1), then copied the rows into the result; faulting in
  that fresh memory cost about as much as the summing. It now counts
  each row's columns in one pass and writes the row into its place in a
  second. Rows of up to 16 terms on average (triangles' edges, and
  tetrahedra's triangles) keep their columns sorted by insertion, as
  before; longer ones find a column's slot through a per-chunk array
  indexed by column, where insertion would be quadratic.

Best of 3 alternating rounds, in ms:

| Mesh | k | 1 thread | all cores |
|---|---|---:|---:|
| `tet_40` | 0 | 96 → 55 | 23 → 21 |
| | 1 | 266 → 186 | 70 → 46 |
| | 2 | 206 → 137 | 50 → 32 |
| | 3 | 100 → 54 | 13 → 6.8 |
| `tet_40_shuffled` | 0 | 244 → 173 | 41 → 38 |
| | 1 | 460 → 402 | 89 → 86 |
| | 2 | 261 → 221 | 67 → 50 |
| | 3 | 102 → 57 | 16 → 6.3 |
| `tri_400` | 0 | 40 → 30 | 14 → 12 |
| | 1 | 76 → 54 | 22 → 15 |
| | 2 | 32 → 18 | 5.3 → 2.3 |
| `tri_400_shuffled` | 0 | 107 → 114 | 25 → 27 |
| | 1 | 121 → 117 | 34 → 29 |
| | 2 | 36 → 21 | 6.7 → 2.8 |

The exception is k = 0 on the shuffled triangles, 6% slower: with
randomly numbered vertices a row's top simplices are scattered, and the
two passes read them twice. `reorder_vertices` removes that case.

