# Vertex numbering and `reorder_vertices`

A mesh's vertex numbers change nothing in the mathematics: renumbering
gives the same complex, the same operators up to a permutation, and the
same results up to the last bits of sums taken in another order. They
change a lot in speed. `reorder_vertices` renumbers a mesh along a
space-filling curve, so that vertices, faces, and elements that are close
in space are close in memory.

```mojo
from mojodec import SimplicialComplex, read_gmsh, reorder_vertices

var r = reorder_vertices(mesh)          # r.mesh, r.old_vertex, r.new_vertex, r.old_element
var sc = SimplicialComplex(r.mesh)

var msh = read_gmsh("part.msh").reorder_vertices()   # node tags and groups follow
```

## Why numbering matters

A `SimplicialComplex` names every face by its sorted vertex numbers and
keeps each level sorted, so face numbers follow vertex numbers. Most of
the work of building a complex and its Whitney matrices is a loop over
vertices, faces, or elements that reads data belonging to their
neighbors: the faces of each element, the elements around each face, the
circumcenters of a chain of faces. When neighbors have nearby numbers,
those reads hit the cache; when the numbering jumps around in space, each
one can be a miss.

The parallel dual-volume walk adds a second cost. It gives each chunk a
block of vertex numbers, and a chunk walks every top simplex with a
vertex in its block (so that every sum is added in the serial order, and
the results equal the serial walk's bit for bit). A top simplex whose
vertices fall in several blocks is walked once per block. With 16 chunks,
a tetrahedron of a spatially numbered grid is walked 1.38 times on
average; with random numbers, 3.64 times.

Meshes generated in code (`kuhn_grid`, `simplicial_grid_2d`, `refine`)
are numbered in space already. Meshes read from files may not be: Gmsh
numbers nodes by the geometric entity they lie on, and meshes that went
through other tools can arrive in any order.

## Morton order

`morton_keys(vertices)` gives each point a Morton (Z-order) code:

1. Scale each coordinate to the points' bounding box and quantize it to
   $b = \min(21, \lfloor 63 / E \rfloor)$ bits, for $E$ coordinates per
   point (21 bits in 2D and 3D). An axis on which all points agree
   quantizes to 0.
2. Interleave the bits, most significant first, axis 0 lowest within each
   bit: bit $l E + a$ of the code is bit $l$ of axis $a$.

Sorting by the code visits the bounding box in a recursive Z: the four
quadrants (in 2D) in the order lower left, lower right, upper left, upper
right, each of them in the same order, and so on down. Points with
nearby codes are close in space. The converse fails where the curve
jumps from one quadrant to the next, so some neighbors get distant codes,
but few: on `tet_40` the walk revisits each tetrahedron 1.18 times with
16 chunks, fewer than the grid's own numbering (1.38).

A Hilbert curve has no jumps, and puts the two ends of an edge a little
closer in number (median 16 apart on `tet_40`, against 27 in Morton
order), but it is slower to build with. On `tet_40` with one thread, in
ms:

| | grid order | Morton | Hilbert |
|---|---:|---:|---:|
| topology | 127 | 130 | 140 |
| Whitney k = 0 | 55 | 74 | 91 |
| build | 295 | 314 | 321 |

The two curves walk each tetrahedron in the dual walk equally often
(1.18 times), and on all cores they are within a few percent of each
other. Hilbert keys also take twice as long to compute, so Morton order
is the better choice here.

The gap was wider when topology sorted with branches: topology took 171,
208, and 257 ms. The Hilbert curve rotates and reflects each subcube, so
the order of the vertices within an element changes from one subcube to
the next, and sorts that branch on that order mispredict more often;
Morton order keeps the axes in the same order in every subcube, and the
grid's Kuhn numbering repeats only a handful of patterns. Topology's
sorts no longer branch on the order of the data (see
[Performance](performance.md#sorting-without-branches)). Whitney k = 0
still runs fastest in grid order, even though the two ends of an edge
are farthest apart in it (median 1681).

`reorder_vertices(mesh)`:

- orders the vertices by their codes, equal codes in their old order
  (unused vertices are kept and placed by their coordinates);
- orders the elements by their new vertex numbers sorted, compared
  lexicographically (ties in their old order). This puts each element
  next to the others with the same smallest vertex, which the
  topology's sort by first vertex reads together, and leaves the rows in
  increasing order, which `SimplicialComplex` then need not sort to
  check for repeated elements;
- keeps each element's vertices in the same positions, so its
  orientation is unchanged and a coherently oriented mesh stays one.

The result does not depend on the numbering the mesh came with, when the
codes are distinct, and reordering it again changes nothing. It takes
30–40 ms for 384k tetrahedra or 320k triangles, on one thread.

## Carrying data across

`ReorderedMesh` holds the permutations:

| Field | Meaning | Use |
|---|---|---|
| `old_vertex[i]` | old number of new vertex `i` | vertex data: `take_rows(x, r.old_vertex)` (`mojodec.core.utils`) |
| `new_vertex[j]` | new number of old vertex `j` | lists of old vertex numbers: `r.new_vertex[j]` |
| `old_element[t]` | old row of new element `t` | element data: `y[r.old_element[t]]` |

`GmshMesh.reorder_vertices()` returns a `GmshMesh` with the node tags
and the physical groups renumbered too, so `msh.subset(sc, name, k)`
works on the complex built from the reordered `msh.mesh()`, and each
vertex keeps its node tag from the file.

## Results

`tet_40` (384k tetrahedra) and `tri_400` (320k triangles) from
`benchmarks/compare`, with vertex numbers randomly permuted
(`_shuffled`), then reordered; best of 3 alternating rounds on an
i7-10875H (8 cores, 16 threads), in ms:

| `tet_40` | grid order | shuffled | reordered |
|---|---:|---:|---:|
| build, 1 thread | 295 | 541 | 314 |
| build, all cores | 120 | 274 | 112 |
| dual walk, all cores | 34 | 90 | 26 |
| topology, 1 thread | 126 | 284 | 130 |
| topology, all cores | 64 | 180 | 63 |
| Whitney k = 0, 1 thread | 55 | 173 | 74 |
| Whitney k = 1, 1 thread | 186 | 402 | 203 |

| `tri_400` | grid order | shuffled | reordered |
|---|---:|---:|---:|
| build, 1 thread | 93 | 205 | 96 |
| build, all cores | 50 | 121 | 48 |
| Whitney k = 0, 1 thread | 30 | 114 | 34 |

Reordering brings a shuffled mesh back to the speed of the grid's own
numbering, and the 3D dual walk on all cores below it (26 against 34 ms,
from the fewer revisits). It takes 30–40 ms, against 227 ms saved in a
one-thread build of `tet_40_shuffled` and 99–199 ms in each Whitney
matrix.
[Performance](performance.md#shuffled-vertex-numbers) has the full
comparison.

## Pitfalls

- **Everything indexed by vertex or element changes.** Cochains,
  boundary vertex lists, element data, and anything else computed on the
  old numbering must go through the permutations. Reorder first, then
  build the complex and everything else from `r.mesh`.
- **Don't reorder between refinements.** `refine` keeps element shapes
  bounded only because it numbers the fine vertices itself, so that each
  child is split in the order it inherited (Bey's rule; see
  [Uniform refinement](refinement.md)). Reorder the coarsest mesh before
  refining, or the finest at the end, never a level in between; and a
  reordered level no longer matches the `Refinement` that produced it.
- **The last bits change.** Sums over faces and elements are taken in
  another order, so dual volumes, stars, and Whitney matrices agree with
  the unreordered mesh's to rounding, not bit for bit. `tests/test_mesh_generation.mojo` checks agreement to 1e-12.
