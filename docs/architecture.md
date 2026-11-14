# Architecture

How MojoDEC is put together: the layers, what a complex stores and
what it computes on demand, where the time and memory go, and the rules
that keep parallel results equal to serial ones bit for bit. This page is
the overview. AGENTS.md has the file-by-file details, conventions, and
toolchain traps, and the [plan](plan.md) has what may change and what was
decided against.

Measurements on this page use the Kuhn grids from `benchmarks/compare`.
They were taken on an 8-core i7-10875H (16 threads, AVX2) at commit
`48c8542`, built with `-D ASSERT=none`, best of 3. Topology, dual
volumes, and build were measured again after later work (best of 5).

- **`tet_40`**: the unit cube, 40³ cells, 384,000 tetrahedra, 68,921
  vertices.
- **`tri_400`**: the unit square, 400² cells, 320,000 triangles, 160,801
  vertices.

## Layers

| Layer | Where | What |
|---|---|---|
| Data structures | `mojodec/core/` | complexes, sparse and diagonal matrices, cochains, subsets, the typed layer |
| Algorithms | `mojodec/routines/` | topology, geometry, Whitney forms, solvers, I/O, mesh generation |
| Plotting | `mojodec/vis/` | matplotlib through Python interop, not re-exported |
| Public API | `mojodec/__init__.mojo` | re-exports `core` and `routines`; `prelude.mojo` is a small subset |

The untyped API follows PyDEC: a complex, levels indexed by dimension,
cochains holding NuMojo vectors, and operators as `CSRMatrix` and
`DiagonalOperator`. The typed layer (`core/typed.mojo`) is a thin wrapper
on top. It carries the degree and primality of every cochain and
operator as parameters, so the compiler checks how they compose.
Operators are always built from the untyped levels, so the two layers
cannot drift apart.

Other complex types (`AbstractSimplicialComplex`, `RipsComplex`,
`RegularCubeComplex`) share the topological API through the
`CellComplex` trait and `ChainComplex`. Only `SimplicialComplex` has
geometry.

## The complex

`SimplicialComplex` is a handle: an `ArcPointer` to one `ComplexData`,
which holds the mesh and one `SimplexLevel` per dimension. Copying a
complex, or keeping one in a cochain, shares the data, as a Python
reference would. Once it is built, nothing changes it. The one
exception is the `compute_*` methods, which fill in geometry when the
complex was built with `compute_geometry=False`.

Each level holds the following:

- **Topology:** the sorted vertex lists of its simplices, their parity
  against the mesh's orientation, `d` (to level k + 1) and `boundary`
  (to level k − 1) as `CSRMatrix`es, and the permutation that sorts the
  top level when the mesh's elements are not in order.
- **Geometry:** circumcenters (Cartesian and barycentric), primal and
  dual volumes, and the diagonal Hodge stars `star` and `star_inv`.

Geometry is computed at construction, unlike PyDEC's lazy attributes.
`compute_geometry=False` skips it for degenerate meshes.

**The design rule:** a level stores what nearly every DEC program needs,
and is built once. Everything else is a function that returns a value
the caller owns: Whitney matrices, wedge products, the flat operator,
curvature, quality measures. Nothing is cached on the complex. So its
memory is fixed by the mesh, and two programs that use different
operators don't pay for each other's.

### Memory

Exact byte counts of the arrays, in MB. Indices and values are both 8
bytes.

| `tet_40` | k = 0 | k = 1 | k = 2 | k = 3 | total |
|---|---:|---:|---:|---:|---:|
| simplices | 68,921 | 462,520 | 777,600 | 384,000 | |
| topology | 19.6 | 70.0 | 93.6 | 46.2 | 229 |
| geometry | 4.4 | 33.3 | 62.2 | 33.8 | 134 |
| mesh (vertices and elements) | | | | | 14 |
| **complex** | | | | | **377** |

`tri_400` takes 171 MB: 97 for topology, 63 for geometry, and 10 for
the mesh.

Topology is about 60% of the complex, and most of that is `d` and
`boundary`. Each stores every incidence once, as an 8-byte column index
and an 8-byte value, plus the row pointers.

A program that reads `tet_40` and builds the complex peaks at about
417 MB of resident memory. That includes the input arrays, which the
complex copies, and the runtime.

## Building a complex

`SimplicialComplex(V, E)` runs these steps, in ms on `tet_40`:

| Step | Where | 1 thread | all cores |
|---|---|---:|---:|
| topology: faces, `d`, `boundary`, orientation check | `simplex_array_boundary`, `require_coherent_orientation` | 74 | 40 |
| circumcenters | `compute_circumcenters` | 33 | 12 |
| primal volumes | `compute_primal_volume` | 16 | 4.5 |
| dual volumes | `compute_dual_volume` (`_DualWalk`) | 62 | 18 |
| Hodge stars | `construct_hodge` | 5.6 | 6.0 |
| **build** | | **191** | **85** |

On `tri_400` the build takes 68 ms on one thread and 36 ms on all
cores. Topology is 28 and 19 ms of that.

- **Topology** never sorts all the faces. A parallel counting sort puts
  face occurrences into one bucket per first vertex, each bucket is
  sorted with branch-free sorts, and the faces, `d`, and `boundary` are
  read off the runs of equal faces. `d` needs no sort of its own: a
  sorted simplex's faces come in reverse order of the vertex left out.
  It is still the largest step on all cores, where it is limited by
  memory bandwidth.
- **Circumcenters and volumes** are closed form where that is cheap and
  accurate: triangles in any dimension, and tetrahedra in 3D. Other
  shapes use Householder QR on positions relative to the first vertex.
  Volumes never come from a Gram determinant.
- **Dual volumes** come from one walk down the chains of circumcenters,
  from each top simplex to its vertices. For triangles in 2D and 3D and
  tetrahedra in 3D the walk runs breadth first, four chains per SIMD
  vector. Each top simplex's faces are gathered one simplex ahead, into
  tables laid out so that each vector is a vector load and a shuffle,
  and each face's pieces are summed in a register. What is left is
  mostly the square roots and divisions.

## Whitney inner products

`whitney_innerproduct(sc, k)` returns a new `CSRMatrix` each call, in
three steps:

1. **Incidence:** each top simplex's k-faces, from the rows of `d`.
2. **Element loop:** each top simplex's local matrix, `fps × fps`
   values, where `fps` is the number of its k-faces. Triangles in 2D and
   3D and tetrahedra in 3D take `_whitney_element`, which is unrolled at
   compile time. Other shapes take the general loop.
3. **Assembly:** each row of the result is the sum over the top
   simplices containing its face, in increasing order. One pass counts
   each row's columns. A second writes the row into its place in the
   result: by sorted insertion when rows average up to 16 terms, and
   through a per-chunk marker array indexed by column otherwise.

Times in ms on `tet_40`:

| k | 1 thread | all cores | nonzeros | result (MB) | peak memory (MB) |
|---|---:|---:|---:|---:|---:|
| 0 | 55 | 21 | 1.0 M | 16 | 484 |
| 1 | 186 | 46 | 7.4 M | 123 | 694 |
| 2 | 137 | 32 | 5.4 M | 92 | 574 |
| 3 | 54 | 6.8 | 0.38 M | 9 | 417 |

"Peak memory" is the whole process: the complex plus one call.

- **k = 1, one thread:** the element loop takes about 72 ms. Assembly
  takes the rest: incidence 8 ms, counting 38 ms, and filling 76 ms,
  about 35 of which is page faults on the fresh result.
- **The call's own peak:** about 280 MB over the complex for k = 1. The
  pieces are the result (123 MB), the local matrices (111 MB), the
  incidence (18 MB), and one marker array per chunk (16 × 1.9 MB).
- **Freeing memory:** all of it but the result is freed on return.

The counting and placing of columns depends only on the topology.
Caching it was considered and rejected; the [plan](plan.md#rejected-caching-whitney-sparsity-on-the-complex)
has the measurements.

## Parallelism and determinism

Every loop over simplices that is large enough goes through
`parallel_ranges` (`core/parallel.mojo`).

- **Chunks:** one per logical core, or one per physical core for
  memory-bound loops.
- **Serial threshold:** a loop runs serially below a size where it would
  take a few hundred µs, since starting the threads costs about 200 µs.
- **Single thread:** `MOJODEC_NUM_THREADS=1` makes every loop serial.

**Results do not depend on the thread count, bit for bit.** Loops that
add into shared entries get this by ownership. Each chunk owns a block
of outputs, such as matrix rows or dual volumes by first vertex, and
sums into them in the serial order.

- **Testing:** `tests/test_parallel.mojo` compares each parallel path
  with the serial one exactly.
- **Optimizations:** every change on the [performance](guide/performance.md)
  page was checked against the build before it by hashing the outputs'
  bits.
- **Explicit fused multiply-adds:** where two code paths must agree to
  the last bit, multiply-adds are written as explicit `fma`. This covers
  the dual walk's two forms and the unrolled Whitney element. LLVM
  otherwise fuses or splits them depending on the surrounding code.

**Vertex numbering:** ownership by vertex blocks assumes the numbering
is spatially coherent. A mesh numbered at random builds 1.5–1.7× slower,
and `reorder_vertices` fixes it. See
[vertex numbering](guide/vertex-numbering.md).

## Numerical policy

- **Precision away from the origin:** compute relative to a simplex's
  first vertex. Absolute positions lose digits far from the origin.
- **Barycentric gradients and volumes:** QR or cross products, never
  `(E Eᵀ)⁻¹ E` or Gram determinants.
- **Matching computations:** where two computations of the same quantity
  are compared, the tests allow the conditioning (`eps L / h`), not the
  last bit. One example is the Whitney top-form mass against the primal
  volume.
- **Solvers:** `routines/dense.mojo` and `routines/linalg.mojo` provide
  pivoted LU, Householder `eigh`, and Krylov methods. NuMojo's unpivoted
  `solve` and Jacobi `eigh` are not used.

## Verification

- **Unit and property tests:** `pixi run test` runs 34 test files. The
  property tests cover rigid motions, relabeling, thin and far-away
  simplices, and corrupted files.
- **Examples:** the examples are regression checks against PyDEC.
  README.md's Examples table lists their expected results.
- **Cross-library benchmark:** `benchmarks/compare` checks that MojoDEC,
  PyDEC, dexterior, and formoniq compute the same numbers on the same
  meshes, and times them.
