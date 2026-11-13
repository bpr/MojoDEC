# Plan

Ideas not yet done, each with the measurements behind it, and ideas
that were considered and rejected, with the reasons. The
[architecture](architecture.md) page describes how things are now. When
an idea lands, its results go on the [performance](guide/performance.md)
page, and it comes off this list.

The machine and meshes are the architecture page's: an 8-core i7-10875H,
`tet_40` (384,000 tetrahedra), and `tri_400` (320,000 triangles). Times
are best of 3, release build.

## Rejected: caching Whitney sparsity on the complex

*Decided 2026-11-13.*

About 75 ms of `whitney_innerproduct(sc, 1)`'s 186 ms on `tet_40` (one
thread) is counting each row's columns and placing each term. Both depend
only on the topology, so they could be computed once and stored on the
`SimplicialComplex`. We decided not to. The complex stays free of caches
(see [the design rule](architecture.md#the-complex)).

### What a cache would hold

There are two possible caches.

- **The pattern only:** the result's row pointers and column indices.
  It saves the counting pass, about 38 ms. Each call would still have to
  find each term's slot in its row and copy the indices into the matrix
  it returns, so the net saving would be around 30 ms.
- **The pattern plus a scatter map:** the map holds, for each term of
  each top simplex, its position in the result. Assembly then becomes a
  plain scatter-add, which saves most of the 75 ms.

Sizes on `tet_40`, in MB, against a 377 MB complex:

| | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---:|---:|---:|---:|
| nonzeros | 1.0 M | 7.4 M | 5.4 M | 0.38 M |
| pattern, 64-bit indices | 8.5 | 63 | 49 | (diagonal) |
| pattern, 32-bit indices | 4.3 | 32 | 25 | (diagonal) |
| scatter map, 32-bit | 25 | 55 | 25 | (diagonal) |

- **k = 1 alone:** 87–118 MB, a quarter to a third of the complex.
- **Every k:** about half of the complex.
- **`tri_400`:** the k = 1 pattern is 23 MB, against a 171 MB complex.

### Why not

**The saving is small.**

- **One thread:** the saving is 75 ms at most, on the second and later
  calls.
- **All cores:** the whole call already takes 46 ms.
- **First call:** it saves nothing, since the first call builds the
  cache.

**Real programs don't make a second call.** In each common pattern of
use, a matrix is assembled once per complex:

- **A fixed mesh:** time stepping, eigenproblems, Poisson solves. Each
  matrix is built once and kept by the caller.
- **A multigrid or refinement hierarchy:** each 3D level is an eighth of
  the next, so the hierarchy is at most 8/7 of the finest complex. Each
  level's matrices are built once.
- **Adaptive refinement or remeshing:** each mesh is a new complex with
  a new pattern.
- **A moving mesh with fixed connectivity:** this is the one pattern
  that reassembles on the same topology. But new vertex positions mean a
  new `SimplicialComplex`, so a cache stored on the complex would not
  carry over.

**The memory would be held for the life of the complex,** and every
program would pay it, including the ones that never benefit.

### If repeated assembly becomes a use case

The likely trigger is a mass matrix with coefficients that vary per
element, reassembled every nonlinear iteration on the same mesh. The
right shape for that is opt-in.

- **Building the pattern:** `whitney_pattern(sc, k)` would return a
  value holding the pattern and the scatter map.
- **Using it:** `whitney_innerproduct(sc, k, pattern)` and a weighted
  variant would take that value.
- **Paying for it:** only callers that reassemble would pay the memory,
  and they would choose how long to keep it. A pattern built from one
  complex could also be used on another complex with the same elements
  and moved vertices.

## Speed

Ordered by the time at stake. Times are in ms on `tet_40`, one thread /
all cores.

- **Topology** (123 / 63) is the largest step of `build` on all cores.
  The parts still serial at last count are the prefix sums over
  vertices, the mesh copy, the orientation check, and the CSR
  bookkeeping between the parallel passes.
- **Dual walk** (117 / 34): about 25 ms of the one-thread time is
  gathering vertex coordinates into the SIMD lanes.
- **Whitney element loop**, several elements per SIMD vector. It takes
  72 ms of k = 1's 186, and 74 of k = 2's 137. Today each element is
  unrolled but computed alone. Any change must keep the bits of every k
  equal to the committed build (see AGENTS.md).
- **Whitney assembly page faults:** about 35 ms of k = 1's 76 ms fill is
  first-touch faults on the 123 MB result. The memory is unavoidable;
  faulting it in parallel ahead of the fill might hide some of the cost.
- **`tri_400_shuffled`, Whitney k = 0:** 114 ms against 107 before the
  two-pass assembly, because with random numbering a row's triangles
  are scattered and are read twice. `reorder_vertices` avoids it; fix it
  only if it costs nothing on the ordered meshes.
- **`reorder_vertices`:** its sort is serial.

## Memory

The `tet_40` complex is 377 MB. Topology is 229 MB of that, and `d` and
`boundary` alone are about 180 MB.

- **±1 stored as doubles:** `d` and `boundary` hold 4.8 M incidences
  each, every one with an 8-byte ±1 value, 77 MB in all. A signed
  incidence type would store these in a byte or a sign bit.
- **The transpose stored twice:** `boundary` is the transpose of the
  next level's `d`, and keeping both doubles about 90 MB.
- **32-bit column indices** would halve the index arrays, about 38 MB.

All three change public types (`sc[k].d` is a `CSRMatrix` today) and
every routine that reads them, so they need a real use case: meshes
large enough that the complex no longer fits.

- **Whitney k = 1 peak:** the call peaks at about 280 MB over the
  complex. The local matrices are 111 MB of that, kept until assembly
  has read them. No plan yet.

## Usability

- **Warn about badly numbered meshes in `read_gmsh`.** A mesh numbered
  without spatial coherence builds 2–3× slower (see
  [vertex numbering](guide/vertex-numbering.md)). A cheap check, such as
  how many ownership blocks the average element touches, could point the
  user to `reorder_vertices`.
