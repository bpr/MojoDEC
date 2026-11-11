# Uniform refinement and prolongation

Refining a mesh, splitting every element into smaller ones of the same
shapes, gives a sequence of nested meshes. On nested meshes the coarse
Whitney forms are also fine Whitney forms, so a coarse cochain has an
exact fine counterpart. `refine` builds the finer mesh in any dimension,
and `Refinement.prolongation(k)` gives the matrix that maps the cochains.

```mojo
from mojodec import SimplicialComplex, refine

var r = refine(sc)               # r.coarse, r.fine, r.parent
var P1 = r.prolongation(1)       # fine edges x coarse edges, sparse
var c_fine = P1 @ c.v            # the same Whitney 1-form on the finer mesh
var r2 = refine(r.fine)          # the next level; P from sc: r2.prolongation(1) @ P1
```

## Freudenthal's subdivision

Every edge gets its midpoint. The children of an n-simplex are the
simplices whose vertices are its vertices and edge midpoints, chosen as
follows. Map the simplex to
$S = \{1 \ge x_1 \ge \dots \ge x_n \ge 0\}$, whose vertices
$p_0, \dots, p_n$ have $i$ ones followed by zeros. Double it:
- the lattice points of $2S$ are the sums $p_i + p_j$ ($i \le j$), which
  are the images of the vertices ($i = j$) and the edge midpoints;
- the Kuhn triangulation of the unit cubes (see
  [Kuhn grids](kuhn-grids-and-tori.md)) cuts $2S$ into $2^n$ simplices.

The result:
- 1D: the two halves of the edge.
- 2D: the usual four triangles, three at the corners and one in the
  middle.
- 3D: eight tetrahedra. Four are at the corners, and the interior
  octahedron is cut into four along one of its diagonals, the one from the
  midpoint of edge (0, 2) to the midpoint of edge (1, 3).

Every child is the parent shrunk by half in each lattice direction:
- **Volume.** Each has exactly $2^{-n}$ of its parent's volume.
- **Orientation.** `refine` orients each child as its parent, so a
  coherently oriented mesh stays coherently oriented.
- **Conformity.** The split depends on the order of the parent's vertices.
  `refine` uses the order of their global numbers, so the two elements
  sharing a face split it the same way.

### Shapes stay bounded

Refining again and again must not degrade the elements. Freudenthal's
subdivision guarantees that only when each child is split again in the
vertex order it inherited from the path through its parent's lattice. This
is Bey's rule. With an arbitrary order, the number of distinct shapes
grows: on one tetrahedron, re-sorting by vertex number gave 1, 3, 5, then
11 shapes over three levels.

`refine` gets the inherited order without storing any order. It numbers
the fine vertices in lexicographic order of their coarse vertex pairs:
$(a, a)$ for coarse vertex $a$, and $(a, b)$ for the midpoint of edge $(a,
b)$ with $a < b$. Within a child, consecutive vertices $p_i + p_j$ along
the path increase $i$ or $j$, so any order that respects the componentwise
order of pairs lists them in path order. The lexicographic order does. So
at the next level, sorting a child's vertices by number recovers its path
order, and the split is Bey's.

Measured on one irregular tetrahedron, the descendants have 1, 3, 3, 3
shapes at levels 0–3. Triangles keep their single shape.

The price is that coarse vertex $a$ is not fine vertex $a$. Coarse vertex
0 is fine vertex 0, but the midpoints of the edges at vertex 0 come before
coarse vertex 1. The prolongation $P_0$ gives the mapping: its row for an
old vertex has a single 1.

## The prolongation

A coarse k-cochain $c$ defines the Whitney form $W c$, and its de Rham map
on the fine mesh is

$$
P_k = R_{\text{fine}} \, W_{\text{coarse}}, \qquad (P_k)_{\tau\sigma} =
\int_\tau W_\sigma ,
$$

for fine k-simplices $\tau$ and coarse k-simplices $\sigma$. Every fine
simplex lies in some coarse element, and $W_\sigma$ is affine there. So the
integral is its value at $\tau$'s centroid applied to $\tau$'s edge
vectors, divided by $k!$.

In barycentric coordinates $\beta_0, \dots, \beta_k$ of $\tau$'s vertices
and with $b$ their mean, the entry is

$$
\int_\tau W_\sigma = \sum_{i=0}^{k} (-1)^i \, b_{s_i}
\det\big[\beta_m(s_j) - \beta_0(s_j)\big]_{j \ne i,\; m = 1..k},
$$

for $\sigma = [s_0, \dots, s_k]$. The gradients of the barycentric
coordinates applied to an edge vector are differences of barycentric
coordinates, so no geometry enters. The entries are the same on every
mesh, and `prolongation` computes them in integers with one division at
the end: they are $\pm 2^{-k}$ and $\pm 2^{-k-1}$.

For a fine simplex inside a coarse face, any element around the face gives
the same row. Whitney forms have continuous tangential traces, and the
forms of coarse simplices outside the face have zero trace on it.

Properties, all checked by the tests:
- **$P_0$ interpolates**: old vertices copy their value, and midpoints
  average the two ends.
- **$P$ commutes with $d$**: $P_{k+1} d = d P_k$ exactly, entry by entry.
  The reason is $W d = d W$ and $R d = d R$.
- **$P$ preserves the Whitney inner product**: $P_k^T M_k^{\text{fine}}
  P_k = M_k^{\text{coarse}}$ to rounding. It is the same form, measured on
  either mesh.
- **Constant forms**: $P R_{\text{coarse}} \omega = R_{\text{fine}}
  \omega$, since $W R \omega = \omega$ for constant $\omega$.

Prolongations compose. From level 0 to level 2, multiply them:
`r2.prolongation(k) @ r1.prolongation(k)`.

## Uses

- **Finer meshes in any dimension.** Refine a mesh from Gmsh or a
  generator. The boundary is refined with it, as every boundary face is
  split into $2^{n-1}$ faces.
- **Convergence on nested meshes.** Refining an irregular mesh repeatedly
  gives a sequence with shape-regular elements of every orientation, a
  harder test than structured grids (see [Convergence](convergence.md)).
- **Error estimates without an exact solution.** For a method of order
  $p$, the difference $\|u_h - P u_{2h}\|$ between two levels is about
  $(1 - 2^{-p})$ of the coarse error. For second-order linear elements
  that is $3/4$, and the measured ratio is 0.67–0.77.
- **Transferring fields** between levels: initial conditions, or a coarse
  solution as a starting guess on the fine mesh.

## Pitfalls

- **Well-centeredness is lost.** A child of a well-centered tetrahedron
  need not be well-centered, so the diagonal Hodge star may get negative
  entries on refined 3D meshes. Use `whitney_innerproduct` there. In 2D,
  the children of an acute triangle are similar to it, so a well-centered
  2D mesh stays well-centered.
- **Curved boundaries are not followed.** Midpoints of boundary edges lie
  on the straight edge, not on the curved boundary. To refine an
  icosphere toward the sphere, use `icosphere` with more subdivisions, or
  project the new boundary vertices yourself. The prolongation then no
  longer maps exactly onto the moved mesh's Whitney forms.
- **Size.** Every level multiplies the number of elements by $2^n$: 8
  levels of a triangle mesh is 65536 times as many triangles, 4 levels of
  a tetrahedral mesh is 4096 times.
- **Vertex numbers change**, as described above. `r.parent[t]` gives a
  fine element's coarse element, and the fine elements are stored in
  blocks of $2^n$ per parent.

## Tests to read

- `tests/test_refinement.mojo`:
  - children's volumes and orientation in 1–4D;
  - bounded shapes over three levels;
  - counts, Betti numbers and boundary splitting on grids, a torus, a
    sphere and a solid torus;
  - the prolongation's properties above in 1–4D;
  - constant forms.
- `tests/test_convergence.mojo`, `test_poisson_on_nested_meshes`: Poisson
  on refinements of a jittered grid in 2D and 3D, with the two-level
  error estimate.
