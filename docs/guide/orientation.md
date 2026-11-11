# Orientation and pseudomanifolds

A `SimplicialComplex` is built only from meshes that are **coherently
oriented pseudomanifolds**. This page explains what that means, why
discrete exterior calculus needs it, what the constructor checks, and how
to fix a mesh that fails.

## Oriented simplices

An *oriented* n-simplex is an ordered list of its vertices
$[v_0, v_1, \dots, v_n]$, where two orders give the same orientation when
they differ by an even permutation and opposite orientations when they
differ by an odd one. A triangle $[0, 1, 2]$ is the same as $[1, 2, 0]$
and the opposite of $[1, 0, 2]$. In a mesh, the order in which an element's
vertices are listed is its orientation.

An oriented simplex *induces* an orientation on each of its faces: the
face opposite vertex $v_i$ gets the sign $(-1)^i$,

$$
\partial [v_0, \dots, v_n] = \sum_{i=0}^{n} (-1)^i \,
  [v_0, \dots, \hat v_i, \dots, v_n].
$$

The triangle $[0, 1, 2]$ induces $[1, 2]$, $-[0, 2] = [2, 0]$, and
$[0, 1]$ on its edges: it walks around its boundary $0 \to 1 \to 2 \to 0$.

## Coherent orientation

Two n-simplices that share an (n−1)-face are **coherently oriented** when
they induce *opposite* orientations on it. For triangles: the shared edge
is walked in opposite directions, so both triangles turn the same way
(both counterclockwise, seen from the same side). A mesh is coherently
oriented when every pair of neighbors is.

In matrix terms, the top boundary operator $B_n$ (`sc[n].boundary`) has a
row per (n−1)-face and a column per element, with the induced signs as
entries. Coherence says every row with two entries holds one $+1$ and one
$-1$. Summing all the elements' columns then cancels every interior face:
the boundary of the whole mesh is just its boundary faces, as it should be.

## Pseudomanifolds

A mesh is a **pseudomanifold** when every (n−1)-face lies in at most two
elements: one on the boundary, two inside. Three triangles on one edge, or
three edges at one vertex of a curve, break it. There, "the other side" of
a face is not defined, so neither is orientation, nor the dual cell of the
face as a piece of a manifold.

## Why DEC needs both

The dual operators are built from the orientation:

- **The dual derivative.** Dual 0-cells are the elements, dual 1-cells
  cross the interior faces, and the dual derivative from dual 0-cochains
  to dual 1-cochains is, up to sign, $B_n$ on the interior faces' rows. A
  constant dual
  0-cochain is closed exactly when every interior row of $B_n$ sums to
  zero, which is coherence. On an incoherent mesh, `d` of a constant is
  not zero, and every dual Laplacian, codifferential `delta`, and Hodge
  decomposition built from it is wrong without any error.
- **The top Hodge star and `wedge`.** Both read each element's orientation
  as the sign of the n-form on it; flipping one element flips the sign of
  the field there.
- **Stokes' theorem** on dual cells, which all of the above encode, needs
  the cells to fit together with matching orientations.

PyDEC accepts any mesh and produces these wrong operators silently.
MojoDEC refuses to build them.

## What the constructor checks

`SimplicialComplex(mesh)` (and `TypedComplex`, `read_ply`, the Gmsh path,
everything that builds a complex) raises in three cases, each with the
offending face and elements in the message:

| Problem | Example message |
|---|---|
| A face shared by three or more elements | `SimplicialComplex: not a pseudomanifold: the face [0, 1] is shared by 3 elements` |
| Two neighbors induce the same orientation on their face, and the mesh can be fixed | `SimplicialComplex: elements 2 and 3 induce the same orientation on their face [5, 6]; orient_coherently(mesh) reorients the elements` |
| No choice of orientations is coherent | `SimplicialComplex: the mesh is not orientable (elements … induce the same orientation on their face …)` |

The check reads the top boundary matrix the constructor builds anyway, so
it costs one pass over it. It applies with `compute_geometry=False` too.

Points (n = 0) are always accepted.

## Fixing a mesh

```mojo
from mojodec import SimplicialComplex, coherent_orientation, orient_coherently

var mesh = read_ply("scan.ply")
var sc = SimplicialComplex(orient_coherently(mesh))
```

`coherent_orientation(elements)` (in `mojodec.routines.simplex_array`)
works on the raw element array, before any complex exists:

- It walks from element to element across shared faces (a breadth-first
  search over the dual graph). The first element of each connected piece
  keeps its orientation, and the walk propagates it.
- It returns one sign per element: `1` to keep, `-1` to reverse. All ones
  means the mesh is already coherent.
- It returns `None` when some loop of neighbors comes back with the wrong
  sign: the mesh is not orientable (a Möbius band, a Klein bottle, the
  projective plane).
- It raises if the mesh is not a pseudomanifold, or an element repeats a
  vertex, or two elements have the same vertices.

`orient_coherently(mesh)` (in `mojodec.routines.mesh_generation`) applies
those signs: each element to reverse has its last two vertices swapped.
Vertex coordinates and the order of the elements are unchanged, so data
indexed by element still lines up.

"Connected" here means connected through (n−1)-faces. Two triangles that
touch only at a vertex are separate pieces, oriented independently.

### Which way is "positive"?

Coherence fixes orientations only relative to each other. Each piece keeps
the orientation of its first element, so a planar mesh may come out
entirely clockwise. For an n-dimensional mesh in $\mathbb{R}^n$, the signed
volume `sc[n].primal_volume` shows which: if it is negative, reverse every
element (swap two columns of the element array) to make the mesh
positively oriented. Both are valid DEC inputs; only the signs of top-degree
quantities change.

## Non-orientable spaces

A Möbius band, a Klein bottle, or the projective plane cannot be a
`SimplicialComplex`, since there is no coherent orientation for the dual
operators to use. Their *topology* is still available through
`AbstractSimplicialComplex`, which has no orientation requirement: Betti
numbers, links, and the homology-manifold check all work on it (see
[Homology](homology.md)).

## Tests to read

`tests/test_topology.mojo`: `test_orient_coherently` flips every third
element of a grid, an icosphere, and a tetrahedral torus and checks that the
walk finds exactly those; `test_moebius_band_is_rejected` uses the 5-vertex
Möbius band, triangles $\{i, i+1, i+2\} \bmod 5$.
