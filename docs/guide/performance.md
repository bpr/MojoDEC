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


## Topology

`SimplicialComplex(V, E, compute_geometry=False)` builds every level's
faces, `d`, and boundary matrix. Before the changes below, on `tet_40`
it took 127 ms on one thread and 66 on all cores, 106 and 46 of them in
the three bucket sorts (`_boundary_by_buckets`) and the rest in serial
setup. Each change keeps every level's simplices, parities, `d`,
boundary, and sort order equal to the last bit. That was checked by
hashing them on `tet_40`, `tri_400`, and their shuffled versions, on one
thread and on all cores.

- **No pass sorting `d`'s rows.** A sorted simplex's faces come in
  decreasing lexicographic order of the vertex left out. Leaving out a
  later vertex leaves the smaller vertex at the first place the two
  faces differ. So the face without vertex i is entry K − 1 − i of the
  simplex's row in `d`. The pass over the sorted runs writes `d`'s
  column indices in place, and its values (±1, by the simplex's
  orientation and i) are written in simplex order. This replaced an
  array of one face index per occurrence and a pass that sorted each
  row.
- **Faces and signs from the keys.** Each occurrence's packed key also
  carries the sign of its boundary entry in its lowest bit. The pass
  over the runs reads the face's vertices and the sign from the key, so
  it never reads the simplices or their parities at random.
- **Counters only for a chunk's band.** Each chunk used to count
  occurrences in an array over all vertices. On `tri_400` that is 16 ×
  160,801 entries, 20 MB per level, zeroed and then summed across chunks:
  9 of the 23 ms of the bucket sorts on all cores. A chunk's rows only
  reach the buckets from their smallest first vertex to their largest
  second vertex. That band is narrow when the numbering is spatially
  coherent, and now only it is counted. With a single chunk the band is
  every vertex, and it is not computed.
- **Unrolled for rows of 2–4 vertices.** The pass that scatters
  occurrences into buckets, the pass over the runs, and `d`'s values
  are compiled for each row width, without inner loops over a runtime
  width. On `tet_40` with one thread, the unrolled scatter saved 17 ms
  and the runs 10. `d`'s values, together with skipping the band on a
  single chunk, saved 6.
- **No sort for edges.** When the faces are vertices, each bucket is one
  face, and its occurrences arrive in order.
- **Serial setup.** Several steps ran serially or repeated work:
  - `SimplicialComplex(V, E)` copied the mesh twice;
  - `SimplicialMesh` checked the vertex numbers with scalar min and max
    passes;
  - every level's constructor checked serially whether its rows were
    sorted, though faces are sorted by construction and
    `require_distinct_simplices` had already checked the top level;
  - the orientation check ran serially;
  - each level scanned for its largest vertex number serially.

  Now one SIMD pass copies the elements and checks their range (3.3 →
  1.2 ms), the top level reuses the distinctness check's answer, and
  the orientation check and the largest-vertex scan run in parallel.
- **Elements out of order.** When the elements are not in lexicographic
  order, as in the shuffled meshes and in most mesh files,
  `require_distinct_simplices` sorted them by an LSD radix sort. Its
  four passes read rows at random: 53 ms on `tet_40_shuffled`. It then
  scanned the sorted rows for repeats, and the top level sorted them a
  second time for its `_order`. Now `lexsort_rows` does a counting sort
  by the first vertex, then sorts each run of equal first vertices by
  insertion, in parallel (by merging past 32 rows). It finds repeated
  rows as it goes, and its order becomes `_order`: 53 → 9 ms on all
  cores.

Best of 5 alternating rounds against the committed build (`7651d47`),
in ms:

| Mesh | | topology, 1 thread | topology, all cores | build, 1 thread | build, all cores |
|---|---|---:|---:|---:|---:|
| `tet_40` | old | 127.0 | 61.4 | 292.2 | 117.8 |
| | new | 73.6 (0.58×) | 40.1 (0.65×) | 241.9 (0.83×) | 94.0 (0.80×) |
| `tet_40_shuffled` | old | 293.3 | 186.9 | 533.8 | 276.3 |
| | new | 122.0 (0.42×) | 58.8 (0.31×) | 365.9 (0.69×) | 147.4 (0.53×) |
| `tri_400` | old | 47.8 | 33.2 | 92.5 | 49.0 |
| | new | 27.8 (0.58×) | 19.2 (0.58×) | 75.5 (0.82×) | 33.8 (0.69×) |
| `tri_400_shuffled` | old | 131.4 | 95.0 | 205.1 | 124.5 |
| | new | 54.7 (0.42×) | 35.1 (0.37×) | 127.0 (0.62×) | 56.8 (0.46×) |

**The benchmark's separate geometry phase reads slower.** On `tri_400`
it is 1.10× on one thread and 1.23× on all cores, though the geometry
code did not change. That phase times `compute_geometry` on a complex
built while the previous one is still alive. In the same binary,
`compute_geometry` takes 32 ms that way and 16 ms on a freshly built
complex, before and after the change. Inside `build`, which is what a
program pays, geometry takes about 17 ms both before and after (build
minus topology).

**Tried and not kept:**

- **32-bit rank sort.** The bucket sort's ranking was rewritten on
  32-bit keys, 8 per SIMD vector, holding the face's vertices relative
  to the bucket's smallest and the key's position. The keys arrive in
  occurrence order, so position breaks ties correctly. It saved at most
  3% on one thread and nothing on all cores. In the level 2 bucket
  sort on `tet_40`, ranking takes about 11 of 16 ms, about 13 cycles
  per key, however the comparisons are arranged.
- **A separate array for the first face.** A row's face without its
  first vertex is in another bucket than its other faces, often one
  owned by another thread. Writing it to its own array and copying it
  into `d` in the pass over rows was slower in 7 of the 8 timings
  above.

**What is left.**

- **One thread, `tet_40` (about 74 ms):** the bucket sorts take about
  26 ms, the passes over the runs 14, the scatter 8, counting 4, `d`'s
  values 4, and the serial setup about 8.
- **All cores (about 40 ms):** the passes write the complex's 229 MB
  of topology, and they stop scaling early. The pass over the runs at
  level 2 takes the same time on 2 threads as on 16, so it is limited
  by memory bandwidth.

## The dual walk's gather and scatter

After the walk went [breadth first](#the-dual-walk-breadth-first), most
of its time was spent around the arithmetic rather than in it. On
`tet_40` with one thread, taking a part out showed what it cost: of
117 ms, the scatter of the pieces into the dual volumes took about
40 ms, and building the SIMD lanes from the gathered tables about 14.
Four changes, all with the same bits:

- **Each face's pieces are summed in a register.** A tetrahedron adds
  40 pieces to 14 faces, 6 of them to each vertex. Each piece was a
  load, an add, and a store to the dual volume, and the next piece for
  the same vertex waited on that store. Now each face's pieces are
  added in a register, in the same order, and the sum is stored once.
  When every vertex of the tetrahedron is in the block, no face's
  ownership is checked.
- **Tables laid out for the lanes.** Each SIMD lane takes one chain, and
  the chains' faces were gathered into a table by face, so each vector
  took four scalar loads and three shuffles. Now:
  - the circumcenters are stored by coordinate, with the faces of each
    size in the order they first appear among the chains, so a lane
    vector is one or two vector loads and a shuffle;
  - the barycentric coordinates that sign the chains are stored in
    chain order, so they are one vector load;
  - the parents' edges, volumes, and signs are a shuffle of one of the
    parent level's vectors.
- **One simplex ahead.** The faces of the next top simplex are gathered
  into a second set of tables before the current one is walked. The
  gather's loads overlap the walk. Without this, the vector loads read
  the gather's scalar stores before they had left the store buffer, and
  `tri_400` got slower (16 → 17 ms).
- **Fewer loads in the gather.** Each edge's index is read from one
  triangle's row of `d`, not two: three of a tetrahedron's four rows
  name all six edges. The factorial that divides each piece is a
  constant, so the division by 2 is an exact multiplication by 1/2.

The dual walk timed alone (best of 15, alternating) and `build` (best
of 5 alternating rounds), against the committed build (`2a00011`), in
ms:

| Mesh | walk, 1 thread | walk, all cores | build, 1 thread | build, all cores |
|---|---:|---:|---:|---:|
| `tet_40` | 116 → 62 | 33 → 18 | 231 → 191 | 96 → 85 |
| `tet_40_shuffled` | 178 → 109 | 61 → 37 | 365 → 301 | 152 → 123 |
| `tet_40`, reordered | 126 → 66 | 26 → 15 | 256 → 201 | 90 → 79 |
| `tri_400` | 21 → 14 | 5.6 → 4.7 | 74 → 68 | 36 → 36 |
| `tri_400_shuffled` | 37 → 24 | 13 → 10 | 127 → 113 | 63 → 63 |
| `tri_400`, reordered | 22 → 15 | 5.1 → 4.4 | 78 → 71 | 35 → 38 |

On one thread the walk is 0.53–0.66× of before. On all cores it is
0.55–0.84×, and `build` on `tri_400` doesn't change, as the walk is
about 5 of its 36 ms. (The `tri_400` reordered build on all cores,
35 → 38 ms, is within its round-to-round noise; the walk in it got
faster.) Every level's dual volumes were checked against the committed
build by their bits, plain and jittered, on all six meshes, a Hilbert
order, a triangulated surface in 3D, and a small grid, on 1 thread and
on all cores.

**Tried and not kept:**

- **A branch-free scatter**, summing the faces a block doesn't own into
  a spare slot instead of skipping them: 82 → 91 ms on one thread.
- **Gathering two simplices ahead**, with three sets of tables: no
  faster than one ahead (66 against 65 ms).
- **Gathering after the walk** instead of before it: `tri_400` 14.5 →
  16 ms.
- **Both sets of tables in one allocation:** the walk took 1.3× as long
  as with separate ones.

**What is left.** On `tet_40` with one thread (about 62 ms), the 10
square roots and 18 divisions per tetrahedron, 4 lanes each, take about
26 ms: the divider handles one 4-lane square root or division every 8
to 12 cycles. Replacing them with multiplications would change the
bits. The gather takes about 20 ms, 13 of it the circumcenters, and
the scatter about 7.

## Whitney elements four at a time

After the [unrolling](#whitney-matrices), the element loop of
`whitney_innerproduct` still took 14, 65, 72, and 53 ms for k = 0 to 3
on `tet_40` with one thread. Each element was computed alone. Its 3
square roots and 9 divisions form one chain of dependencies, too long
for the processor to overlap two elements. Its local matrix's terms
were read from run-time tables. Four changes, all with the same bits:

- **The term tables are compile-time constants.** Each entry of the
  local matrix is a signed sum of determinants of pairs of forms. Which
  pairs, and which signs, depend only on the dimension and the degree.
  They were read from tables at run time, three loads per term: the
  coefficient, the pair's index, and the determinant. Now they are
  constants, the determinants stay in registers, and the coefficients
  (±1 and ±2) are folded in. Alone, this took k = 1 to 3 from 65, 72,
  and 53 ms to 40, 43, and 36.
- **Four elements per SIMD vector.** Each lane holds one top simplex,
  and every operation is the one-element code's, lane by lane. The sign
  in each Householder step is a select. So is the row pivoting in the
  3 × 3 minors of k = 3, with a lane whose pivot is zero giving 0, as
  before. A batch with a degenerate simplex is redone one at a time, to
  name the simplex. The one-lane form is the same code and also takes
  the last few simplices of each chunk.
- **Each face is gathered once, four simplices at a time.** The k-faces
  come from the rows of `d`, from the top simplex down. Each face is
  now read from one row only (three of a tetrahedron's four triangle
  rows name all six edges), and the four simplices' rows are read level
  by level, so their loads overlap.
- **Whole rows are stored.** For tetrahedra, whose local matrices have
  16 or 36 entries, four entries of four simplices are transposed and
  stored as one vector per simplex.

The element loop alone (an instrumented build, best of 7) and the whole
call (best of 5 alternating rounds), against the committed build
(`c291d71`), in ms:

| Mesh | k | loop, 1 thread | loop, all cores | call, 1 thread | call, all cores |
|---|---|---:|---:|---:|---:|
| `tet_40` | 0 | 14 → 7.6 | 6.6 → 5.0 | 55 → 49 | 20 → 19 |
| | 1 | 65 → 16 | 12 → 8.8 | 185 → 137 | 46 → 48 |
| | 2 | 72 → 14 | 8.5 → 5.2 | 138 → 80 | 31 → 28 |
| | 3 | 53 → 9.5 | 5.4 → 1.3 | 54 → 10 | 5.8 → 2.0 |
| `tet_40_shuffled` | 0 | 44 → 22 | 8.3 → 6.8 | 171 → 152 | 36 → 37 |
| | 1 | 83 → 30 | 13 → 12 | 398 → 347 | 82 → 83 |
| | 2 | 79 → 18 | 9.3 → 5.0 | 219 → 160 | 52 → 45 |
| | 3 | 56 → 10 | 7.6 → 1.3 | 58 → 11 | 6.6 → 2.9 |
| `tri_400` | 0 | 6.2 → 3.7 | 2.6 → 2.6 | 31 → 29 | 11 → 11 |
| | 1 | 24 → 5.9 | 2.8 → 2.7 | 54 → 36 | 14 → 16 |
| | 2 | 17 → 2.9 | 1.7 → 0.6 | 18 → 3.4 | 2.3 → 1.5 |

On one thread the element loop is 0.17–0.6× of before, and the
top-degree matrices (k = n), which are nothing but the element loop,
0.19×. On all cores the calls for k = 0 and 1 don't change: there
assembly takes nearly all the time, limited by memory bandwidth (the
`tri_400` k = 1 call, 14 → 16 ms, is within its noise; on the
reordered mesh it went 14.7 → 14.5). Every k's matrix was checked
against the committed build by its bits: `tet_40`, `tri_400`, and
`tet_40_shuffled`, plain and jittered, the triangles also lifted onto
a surface in 3D, and small Kuhn grids whose element counts leave
remainders, on 1 thread and on all cores.

**What is left.** Of k = 1's 137 ms on `tet_40` with one thread, the
element loop is now 16: about 6 gathering the faces, 3 storing the
110 MB of local matrices, and the rest arithmetic. Assembly takes the
other 120 or so: counting each row's columns, and filling the rows.

## Whitney assembly

With the [element loop](#whitney-elements-four-at-a-time) down to
16 ms, assembly was most of `whitney_innerproduct`. On `tet_40` with
one thread, k = 1 spent 7.5 ms finding the top simplices containing each
edge (the incidence), 38 counting each row's columns, and 78 filling the
rows. k = 2 spent 27 counting and 33 filling. Three changes, all with
the same bits:

- **Faces of codimension 1 are merged, not counted.** In a
  pseudomanifold, a face of dimension n − 1 lies in one or two top
  simplices, and two top simplices share at most one such face. So its
  row has 1 + (fps − 1)·m entries for m top simplices: the faces of
  those simplices, merged, with the shared face counted once. Each
  simplex's faces are already in increasing order, so a column's place
  is its place in its own simplex plus the number of smaller faces of
  the other. The shared face's entry is v₁ + v₂, in increasing order of
  the simplex, as before. The top simplices containing each face are the
  rows of the top level's `boundary`, so this needs no incidence, no
  counting pass, and no search. It covers k = 2 for tetrahedra and
  k = 1 for triangles.
- **Counting marks columns with the row.** The counting pass looks up
  each term's column in an array of slots. About half the terms bring a
  new column, so the branch on that mispredicted. Now each slot holds the
  last row that saw its column: a term is new if its slot holds another
  row, and the slot is set to this row either way, with no branch and no
  pass to clear the slots. k = 1: 38 → 26 ms.
- **The incidence is built on all cores**, with 32-bit entries. It was a
  serial counting sort, 8.4 ms of k = 1's 41 on all cores. Now each
  chunk of terms counts its faces by blocks of faces, places its terms
  into their blocks after those of the chunks before it, and each block
  is sorted by face. Each step keeps the order, so each face's top
  simplices stay in increasing order. All cores: 8.4 → 6.0 ms.

The path that kept short rows sorted by insertion is gone: only faces
of codimension 1 had rows that short.

The whole call, best of 5 alternating rounds, against the committed
build (`bb4571c`), in ms:

| Mesh | k | 1 thread | all cores |
|---|---|---:|---:|
| `tet_40` | 0 | 50 → 46 | 19 → 18 |
| | 1 | 138 → 126 | 46 → 43 |
| | 2 | 80 → 52 | 31 → 16 |
| `tet_40_shuffled` | 0 | 152 → 137 | 33 → 30 |
| | 1 | 349 → 313 | 83 → 72 |
| | 2 | 163 → 107 | 41 → 24 |
| `tri_400` | 0 | 28 → 26 | 12 → 11 |
| | 1 | 36 → 22 | 15 → 7.7 |
| `tri_400_shuffled` | 0 | 105 → 91 | 27 → 22 |
| | 1 | 90 → 65 | 30 → 14 |

The top degree (k = n), whose matrix is diagonal, is unchanged. Every
k's matrix was checked against the committed build by its bits, on the
same meshes as for the element loop and the shuffled triangles too, on
1 thread and on all cores.

**Tried and not kept**, all on `tet_40`, k = 1, one thread:

- **Faces per simplex known at compile time** in the general path: the
  counting went 26 → 30 ms and the filling 77 → 86. The codimension-1
  merge does use it: with the faces' order, it went 45 → 38 ms.
- **A fill without branches**, its column's place packed into the slot
  with the row's mark: 77 → 88 ms. Each term's place came from a load
  that waited on the previous term's store.
- **Sorting each row's terms** by column and term (eight keys at a time,
  by SIMD ranks) and summing each column in one scan: 77 → 95 ms. It is
  simply more instructions per term.
- **Counting with ranks**, recording each term's place in its sorted row
  so that the fill needs no search: 116 → 148 ms for both passes.

**What is left.** On `tet_40`, k = 1, one thread, the fill takes about
75 ms: about 45 finding each term's column in the row (half of those
branches still mispredict), 11 reading the local matrices, and 17
sorting each row. Page faults are not what limits it: touching the
result's pages first did not make the fill faster. On all cores it takes
about 18 ms and moves about 250 MB, which is close to this machine's
memory bandwidth.

## Bucket sorts in topology

After the [topology changes](#topology), the bucket sorts were the
largest part left of building `tet_40`'s topology on one thread: about
24 of 74 ms. Its buckets hold 24 keys (level 3, triangles in
tetrahedra) and 36 (level 2, edges in triangles). Each bucket of 17 to
64 keys was sorted by rank: each key's place is the number of smaller
keys, counted by comparing it with every key of the bucket, four at a
time. Three changes, all with the same bits:

- **The keys are compared as doubles.** AVX2 compares 64-bit integers
  (`vpcmpgtq`) on one port of this CPU, and doubles (`vcmppd`) on two. A
  key is below 2⁶², so key + 2⁵² is the bit pattern of a positive normal
  double. Positive doubles order as their bit patterns do, so the
  comparison is exact.
- **Sixteen keys are ranked at a time.** The old loop took one key,
  compared it with the whole bucket, and summed its count's four lanes:
  a broadcast and a horizontal sum for every key, on the same port as
  the compares. Now the counts of 16 keys stay in four registers while
  each key of the bucket is loaded once, already broadcast, and compared
  with all 16. The last block takes 8 or 4 keys, so a bucket is padded
  to a multiple of 4, not 16.
- **New thresholds.** Ranking now takes over from insertion above 8
  keys, not 16, and from merging two ranked halves at up to 96 keys, not
  64. On keys that arrive nearly in order, insertion takes about 1.2 ns
  a key and ranking 3. On keys in random order, insertion takes 7–8 ns a
  key at 6–8 keys, as its branches mispredict. Ranking takes 4.7 ns a key
  at 64 keys against 5.8 for merging ranked halves, and 8.2 against 7.6
  at 128.

The sort pass alone, one thread, in ms (the keys after the scatter,
sorted bucket by bucket, best of 5):

| Mesh | level 2 | level 3 |
|---|---:|---:|
| `tet_40` | 15.0 → 10.5 | 8.5 → 5.8 |
| `tet_40_shuffled` | 21.3 → 15.5 | 13.4 → 9.1 |
| `tri_400` | 3.1 → 3.0 | |
| `tri_400_shuffled` | 8.3 → 5.8 | |

`tri_400`'s buckets hold 6 keys, still sorted by insertion.

Topology, `SimplicialComplex(V, E, compute_geometry=False)`, best of 5
alternating rounds against the committed build (`b7c8940`), in ms:

| Mesh | 1 thread | all cores |
|---|---:|---:|
| `tet_40` | 72.9 → 65.9 | 38.7 → 38.0 |
| `tet_40_shuffled` | 119.5 → 106.9 | 60.2 → 58.0 |
| `tet_40_reordered` | 76.3 → 69.6 | 42.8 → 41.7 |
| `tri_400` | 27.3 → 27.6 | 20.0 → 20.7 |
| `tri_400_shuffled` | 52.9 → 50.1 | 34.0 → 36.0 |
| `tri_400_reordered` | 28.2 → 28.1 | 20.4 → 20.7 |

The changes on all cores, and on `tri_400` and `tri_400_reordered`, are
within the noise of this machine. On all cores the sorts were already a
small part, and the passes around them are limited by memory bandwidth.
`build` on `tet_40` with one thread went 186 → 181 ms. Every level's
simplices, parities, `d`, boundary, and `_order` were checked against
the committed build by their bits, on these meshes and on Morton-ordered
ones, on 1 thread and on all cores.

**Tried and not kept:**

- **A table of each bucket's distinct faces.** A bucket of 36 edge
  occurrences holds only about 7 distinct edges. So each key could look
  up its face in a small table of the bucket's faces so far, and the
  keys could then be placed by counting. Level 3 of `tet_40` took
  20 ms against 8.5. Half the keys there bring a new face, and that
  branch mispredicts.
- **32-bit keys for edges.** When the faces are edges (v, w), the key
  (w − v) · 128 + position fits in 30 bits and can be ranked as a float,
  eight per vector. Level 2 of `tet_40` went 10.5 → 9.0 ms. That is a
  second ranking path for one face width, for 2% of the topology.

**What is left.** On `tet_40` with one thread the sorts take about
16 ms. About 6 ms of that is reading the keys, writing them back, and
marking the runs, which a pass over the buckets does without sorting
them.

## Sorting elements out of order

When a mesh's elements are not in lexicographic order, as in the
shuffled meshes and in most mesh files, `require_distinct_simplices`
sorts their rows with `lexsort_rows`, and the order becomes the top
level's `_order` (see [Topology](#topology)). That was a serial counting
sort by first vertex, then each bucket sorted by insertion in parallel,
comparing rows by reading them from the array. On `tet_40_shuffled`
with one thread, in ms: the count 1.0, the scatter 1.6, and the buckets
16.7, as each comparison read two rows at random. Three changes, with
the same order:

- **The count and the scatter run in parallel**, as topology's do: each
  chunk of rows counts only the buckets between its smallest and
  largest first vertex, and a bucket takes each chunk's rows in turn,
  so the order is the serial one. They take one chunk per physical
  core: with a shuffled numbering every chunk's band is every vertex,
  and 16 chunks' counters cost more than they saved on `tri_400`.
- **Each row's other entries are packed into one integer key** when
  they fit in 63 bits, and the scatter writes the key next to the row's
  index. The buckets then sort (key, index) pairs that lie side by side,
  and equal keys are equal rows. The scatter takes 2.8 ms instead of
  1.6, and the buckets 5.2 instead of 16.7.
- **The range of the entries** is found by the SIMD scan the element
  copy already uses, in parallel, and the permutation is not filled
  with the identity first, since the scatter writes all of it.

`lexsort_rows` on the elements, with their rows sorted, best of 7, in
ms:

| Mesh | 1 thread | all cores |
|---|---:|---:|
| `tet_40_shuffled` | 21.4 → 10.4 | 8.5 → 4.4 |
| `tri_400_shuffled` | 9.6 → 7.5 | 4.8 → 4.5 |

Topology, best of 5 alternating rounds against the committed build
(`2bcde4b`), in ms:

| Mesh | 1 thread | all cores |
|---|---:|---:|
| `tet_40_shuffled` | 109.6 → 98.5 | 58.7 → 55.2 |
| `tri_400_shuffled` | 52.1 → 48.3 | 35.4 → 34.6 |

The meshes in grid order don't sort their elements, and didn't change
beyond the noise. In the benchmark (5 alternating rounds), `build` on
`tet_40_shuffled` read 271.5 → 272.1 ms on one thread and 124.6 →
114.9 on all cores: on a shuffled mesh the geometry varies by more than
the 10 ms saved. Every level's simplices, parities, `d`, boundary, and
`_order` were checked against the committed build by their bits, on 14
meshes, on 1 thread and on all cores.

**What is left.** On `tri_400_shuffled` on all cores, the count and the
sums of the counters take about 2 ms of the 4.5, most of it clearing
and summing 8 chunks' counters for all 160,801 vertices. The buckets of
`tet_40_shuffled` (5.2 ms on one thread) are sorted by insertion, whose
branches mispredict on keys in random order; ranking pairs, with ties
broken by position, would avoid that.
