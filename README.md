## **MojoDEC: A Mojo Library for Discretizations of Exterior Calculus.**

MojoDEC is largely inspired by [PyDEC](https://github.com/hirani/pydec).
It implements Discrete Exterior Calculus (DEC) and lowest order finite element
exterior calculus (FEEC), i.e., Whitney forms. It runs on Mojo 1.3, with
[NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) arrays
in place of PyDEC's NumPy arrays and its own `CSRMatrix` in place of
PyDEC's `scipy.sparse` matrices. The main functionality, as in PyDEC:
- simplicial complexes of dimension n embedded in dimension N >= n
- abstract simplicial complexes (no embedding needed), and Rips complexes
- cubical complexes in all dimensions
- boundary operators for all the above types of complexes
- discrete exterior derivative (i.e., coboundary operator)
- DEC discrete Hodge star (i.e., primal-dual diagonal mass matrix)
- FEEC mass matrix for Whitney forms
- mesh I/O in PyDEC's formats, mesh generation and subdivision, and
  matplotlib plotting

Beyond PyDEC, MojoDEC adds:
- the discrete wedge product of primal cochains
- the discrete flat, from vector fields to primal 1-cochains
- Gaussian and mean curvature of triangulated surfaces
- mesh quality measures (well-centeredness, shape regularity)
- Euler characteristics, Betti numbers (absolute, of subcomplexes, and
  relative), a homology-manifold check, and simplicial neighborhoods
  (star, closure, link, boundary, interior) on every complex type
- coherent orientation checks: complexes must be coherently oriented
  pseudomanifolds, and `orient_coherently` fixes an orientable mesh
- explicit leapfrog time stepping (Yee's scheme) with an exactly
  conserved energy, for Maxwell's equations and other wave systems, and
  the implicit midpoint rule for the same systems (with a Whitney mass, any
  step) and for diffusion (Crank–Nicolson)
- cell subsets for boundary conditions: bitsets of primal or dual cells
  with set operations, boundary and coboundary, the mesh boundary in every
  dimension, and `exclude_subset` / `restrict_to_subset` on operators
- typed cochains and operators, whose degree and primal/dual kind are
  checked at compile time, so `d` and `star` compose only correctly
- VTK (`.vtu`) output for ParaView, PLY mesh input/output, and Gmsh
  `.msh` input with its physical groups as subsets, in pure Mojo
- icosahedron and icosphere meshes
- its own dense and sparse linear algebra (pivoted LU, symmetric
  eigensolver, CG, LSQR, generalized eigenproblems), and a
  `DiagonalOperator` type for the Hodge stars
- faster complex construction: geometry computed in place without
  per-simplex allocation, and simplex lookup by binary search instead of
  per-level dictionaries
- examples beyond PyDEC's: the spectrum of the Hodge Laplacian on the
  sphere, neighborhoods on 2D and 3D meshes, and Maxwell's equations
  stepped in time in a cavity

The algorithms follow the PyDEC paper: Bell & Hirani, *PyDEC: Software and
Algorithms for Discretization of Exterior Calculus*, ACM Transactions on
Mathematical Software, Vol. 39, No. 1, pp. 3:1-3:41, 2012,
[DOI: 10.1145/2382585.2382588](http://dx.doi.org/10.1145/2382585.2382588).

```mojo
import numojo as nm
from mojodec import SimplicialComplex, d, delta, cg

def main() raises:
    var V = nm.fromstring[nm.f64]("[[0,0],[1,0],[1,1],[0,1]]")
    var S = nm.fromstring[DType.int]("[[0,1,3],[1,2,3]]")
    var sc = SimplicialComplex(V, S)
    print(sc[1].num_simplices)                              # 5 edges
    var L = delta(d(sc.get_cochain_basis(0))).v.copy()      # 0-form Laplacian
    var star1 = sc[1].star.copy()                           # diagonal Hodge star
```

Installation (linux-64; everything runs through [pixi](https://pixi.sh)):
- `cd` to the folder where you cloned MojoDEC
- `pixi install` (Mojo 1.3 nightly + MAX, Python, NumPy, matplotlib)
- `pixi run numojo` (builds `build/numojo.mojoc` from the vendored NuMojo)
- `pixi run package` (builds `build/mojodec.mojoc`)

Other tasks:

```sh
pixi run test                        # all unit tests
pixi run examples                    # every example; figures saved as PNG
pixi run examples DarcyFlow --show   # one example, figures in windows
pixi run examples-release            # the same with bounds checks compiled out
pixi run bench                       # timings of the parallel loops
pixi run format
```

Notes about installation:
- `pixi.toml` accepts any Mojo from the 1.3 nightlies on (`>=1.3.0.dev0`);
  `pixi.lock` holds the nightly it was tested with (`1.3.0.dev2026110305`),
  and `pixi update` moves to a later one.
- The published NuMojo 0.12 package requires Mojo < 1.3, so NuMojo v0.12.0
  is vendored in `third_party/numojo` with a small 1.3 patch; see
  `third_party/README.md`. The `test`, `examples`, and `package` tasks build
  it first.
- To use MojoDEC from another project, put both `build/numojo.mojoc` and
  `build/mojodec.mojoc` (or `-I /path/to/MojoDEC -I /path/to/MojoDEC/build`)
  on the import path.
- There is also a conda recipe (`conda.recipe/`); MojoDEC will go on the
  modular-community channel once Mojo 1.3 is released and NuMojo supports
  it (see `docs/publishing.md`).

## Layout

Modeled on NuMojo (`core` for data structures, `routines` for algorithms)
rather than PyDEC's `pydec/pydec/<subpackage>` nesting.

| MojoDEC | PyDEC |
|---|---|
| `mojodec/core/sparse.mojo` — `CSRMatrix` | `scipy.sparse.csr_matrix` (PyDEC's sparse matrices) |
| `mojodec/core/diagonal.mojo` — `DiagonalOperator` | (new; PyDEC uses `scipy.sparse` diagonal matrices) |
| `mojodec/core/operator.mojo` — `LinearOperator`, `Preconditioner`, `Identity` | `scipy.sparse.linalg.LinearOperator` |
| `mojodec/core/subset.mojo` — `Subset` | (new) |
| `mojodec/core/typed.mojo` — `TypedComplex`, `TypedCochain`, `MatrixOp`, `DiagonalOp` | (new) |
| `mojodec/core/typed_block.mojo` — `BlockOp`, `CochainPair` | (new) |
| `mojodec/core/simplex.mojo` — `Simplex` | `mesh/simplex.py` (`simplex`) |
| `mojodec/core/mesh.mojo` — `SimplicialMesh`, `RegularCubeMesh` | `mesh/simplex.py`, `mesh/regular_cube.py` |
| `mojodec/core/ncube.mojo` — `NCube` | `mesh/ncube.py` |
| `mojodec/core/simplicial_complex.mojo` — `SimplicialComplex`, `SimplexLevel` | `dec/simplicial_complex.py` |
| `mojodec/core/cochain.mojo` — `Cochain`, `CochainBasis`, `d`, `star`, `delta`, `laplace_*` | `dec/cochain.py` |
| `mojodec/core/abstract_simplicial_complex.mojo` | `dec/abstract_simplicial_complex.py` |
| `mojodec/core/rips_complex.mojo` | `dec/rips_complex.py` |
| `mojodec/core/regular_cube_complex.mojo` | `dec/regular_cube_complex.py` |
| `mojodec/core/kd_tree.mojo` — `KDTree` | `math/kd_tree.py` |
| `mojodec/core/utils.mojo` | array helpers |
| `mojodec/routines/simplex_array.mojo`, `cube_array.mojo` | `dec/simplex_array.py`, `dec/cube_array.py` |
| `mojodec/routines/geometry.mojo` | `math/volume.py`, `math/circumcenter.py`, `barycentric_gradients` |
| `mojodec/routines/combinatorial.mojo` | `math/combinatorial.py`, `math/parity.py` |
| `mojodec/routines/innerproduct.mojo` | `fem/innerproduct.py` |
| `mojodec/routines/mesh_generation.mojo` | `mesh/generation.py`, `mesh/subdivision.py` |
| `mojodec/routines/linalg.mojo` — `cg`, `minres`, `jacobi`, `lsqr`, `spsolve`, `eigh_generalized` | `scipy.sparse.linalg`, `scipy.linalg.eig` (called by PyDEC) |
| `mojodec/routines/dense.mojo` — pivoted LU, `det`, `inv`, `cholesky`, `eigh`, `EighResult.residuals` | `numpy.linalg` (called by PyDEC) |
| `mojodec/routines/io/` — `read_array`, `write_array`, `read_mesh`, `write_mesh`, `loadtxt` | `io/arrayio.py`, `io/meshio.py` |
| `mojodec/routines/io/vtk.mojo` — `write_vtu`, `write_vtu_cubes`, `VTKField` | (new) |
| `mojodec/routines/io/ply.mojo` — `read_ply`, `write_ply` | (new) |
| `mojodec/routines/io/gmsh.mojo` — `read_gmsh`, `GmshMesh`, `GmshGroup` | (new) |
| `mojodec/routines/wedge.mojo` — `wedge` | (new) |
| `mojodec/routines/musical.mojo` — `flat`, `dual_edge_pieces` | (new) |
| `mojodec/routines/curvature.mojo` — `angle_defect`, `gaussian_curvature`, `mean_curvature`, `mean_curvature_normal` | (new) |
| `mojodec/routines/quality.mojo` — `well_centeredness`, `well_centered_fraction`, `shape_regularity` | (new) |
| `mojodec/core/chain_complex.mojo` — `ChainComplex` | the lists of `chain_complex()`, `cochain_complex()` |
| `mojodec/core/cell_set.mojo` — `CellSet`, `CellComplex` | (new) |
| `mojodec/routines/topology.mojo` — `star_of`, `closure_of`, `link_of`, `boundary_of`, `interior_of` | (new) |
| `mojodec/routines/homology.mojo` — `betti_numbers`, `relative_betti_numbers`, `non_manifold_cells` | (new) |
| `mojodec/routines/cohomology.mojo` — `cohomology_basis`, `CohomologyBasis`, `harmonic_basis` | (new) |
| `mojodec/routines/refinement.mojo` — `refine`, `Refinement` (`prolongation`) | (new) |
| `mojodec/routines/time.mojo` — `Leapfrog`, `ImplicitMidpoint`, `CrankNicolson` (and typed versions) | (new) |
| `mojodec/routines/quadrature.mojo` — `grundmann_moeller`, `QuadratureRule`, `integrate_simplices` | (new) |
| `mojodec/routines/derham.mojo` — `derham_map`, `whitney_form`, `l2_projection`, `whitney_load`, `l2_error`, `form_index_sets` | (new) |
| `mojodec/vis/draw.mojo` — `triplot`, `lineplot`, `lineplot2`, `simplex_quivers`, `cube_quivers` | `vis/draw.py` |

## Plotting

`mojodec.vis` draws with matplotlib through Mojo's Python interop, so it
needs the Python, NumPy, and matplotlib that `pixi install` provides. As in
PyDEC, `triplot` and the line plots add to matplotlib's current axes;
`simplex_quivers` and `cube_quivers` are pure Mojo and return NuMojo arrays
for `quiver`. `mojodec` does not re-export `vis`, so programs that do not
plot never load Python.

```mojo
from mojodec.vis import pyplot, show_requested, simplex_quivers, triplot

var plt = pyplot(interactive=show_requested())  # Agg unless run with --show
_ = triplot(sc.vertices(), sc.simplices())
var q = simplex_quivers(sc, h)
var b = q.bases.to_numpy()
var v = q.vectors.to_numpy()
plt.quiver(b.T[0], b.T[1], v.T[0], v.T[1])
plt.savefig("h.png")
if show_requested():
    plt.show()
```

`pyplot` returns `matplotlib.pyplot` with the non-interactive Agg backend,
so figures can be saved without a display, or with matplotlib's GUI backend
when `interactive` is true. A complete program, and how to run it, is in
[`examples/README.md`](examples/README.md#plotting-from-your-own-program).

## VTK output

`write_vtu` writes a simplicial mesh (vertices, edges, triangles, or
tetrahedra) as a VTK XML unstructured grid (`.vtu`) for ParaView or
VisIt; `write_vtu_cubes` writes the k-cubes of a `RegularCubeComplex`. It is
pure Mojo, with no Python. Cochains go in as named point or cell data: a
k-cochain is cell data on `sc[k].simplices`, and a `simplex_quivers` field
is cell data on `sc.simplices()`. 2D points and vectors are padded to 3D.
[`examples/README.md`](examples/README.md#viewing-the-vtk-files-in-paraview)
shows how to view them in ParaView.

```mojo
from mojodec import VTKField, write_vtu

write_vtu(
    "h.vtu",
    sc.vertices(),
    sc[1].simplices,
    cell_data=[VTKField("h", h)],
    format=.APPENDED,
)
```

`format` is a `VTKFormat`: `.ASCII` (the default), `.BINARY` (base64 inside
the XML), or `.APPENDED` (raw bytes after the XML). All three hold the same
values exactly. Binary output is fixed-size and faster to read, but it is
not compressed: `.APPENDED` is about a quarter smaller than ASCII for
floating-point fields, while connectivity of small meshes, whose indices are
short in ASCII, can come out larger.

Differences from `pydec.vis`: `lineplot` takes either one width or a
`RealArray` of widths per segment; `lineplot2`'s labels work (PyDEC's
referenced an undefined variable); and `triplot` draws black edges, which
PyDEC got from matplotlib's pre-2.0 defaults.

## Beyond PyDEC

The extensions listed at the top, in more detail. Each is implemented from
the published mathematics:

- **`wedge(a, b)`**: the wedge product of primal cochains, the
  antisymmetrized cup product of Hirani's thesis (ch. 7). It is graded
  anticommutative, satisfies the Leibniz rule with `d`, and is exact for
  constant forms; the tests check all three on 2D and 3D meshes.
- **`flat(sc, X)`**: Hirani's DPP flat (thesis, sec. 5.5), a vector field
  given per top simplex to a primal 1-cochain, weighting each top simplex
  by its signed share of the edge's dual cell. Constant fields flatten
  exactly, on meshes that are not well-centered too. `simplex_quivers` is
  the opposite direction.
- **Surface curvature** at the vertices of a triangulated surface in any
  dimension: `angle_defect` (which adds up to `2 pi` times the Euler
  characteristic), `gaussian_curvature` (defect over dual area), and
  `mean_curvature_normal`, `1/2 delta d x` (the cotangent formula).
- **Mesh quality**: `well_centeredness` (the smallest barycentric
  coordinate of each circumcenter), `well_centered_fraction`, and
  `shape_regularity`.
- **Orienting meshes.** `coherent_orientation(elements)` walks across
  shared faces and gives each element a sign (`-1` to reverse it) that
  makes neighbors induce opposite orientations on their shared face, or
  `None` for a non-orientable mesh such as a Möbius band.
  `orient_coherently(mesh)` returns the mesh with those elements reversed,
  ready for `SimplicialComplex`.
- **Betti numbers** over the rationals on every complex type:
  `betti_numbers()`, `betti_numbers(cells)` for a subcomplex such as a
  link, and `relative_betti_numbers()` (relative to the boundary) or
  `relative_betti_numbers(sub)` for a pair. The free functions in
  `routines/homology.mojo` take a `ChainComplex`. Ranks come from sparse
  elimination modulo a prime near 2^30, exact unless the integral
  homology has torsion of that order; 20k triangles take milliseconds.
  The tests check Lefschetz duality on the solid torus, spheres as links
  of interior vertices, and that the projective plane gets `[1, 0, 0]`
  (over Z/2 it would be `[1, 1, 1]`).
- **Cohomology bases and harmonic forms** (`cohomology_basis`,
  `harmonic_basis`): `b_k` integer cocycles and dual cycles with
  `<z_i, c_j> = delta_ij`, from exact integer elimination (on a surface,
  a tree and cotree), and the harmonic cochain with period 1 around each
  cycle, `h = z - d p` with `d^T M d p = d^T M z` solved by MINRES, in the
  Hodge star or any mass matrix. The count needs no eigenvalue tolerance.
  HodgeDecomposition uses it in place of random cochains. Given a
  subcomplex such as `sc.boundary_cells()`, both give relative classes
  and the harmonic forms with Dirichlet conditions (`b_k(K, dK)` of them).
- **Homology-manifold check** on every complex type:
  `non_manifold_cells()` gives the cells whose link does not have the
  homology of a sphere (inside) or a point (on the boundary), such as the
  tip where two cones meet, and `is_homology_manifold()` says whether
  there are none. Cells of every dimension below n are checked, not only
  vertices: two tetrahedra sharing just an edge have disks for all vertex
  links, and only the edge's link (two separate edges) shows the pinch.
  The constructors don't require it, since pinched meshes are valid DEC
  input; it is a query, a few tens of milliseconds on a 2,500-vertex
  surface.
- **`euler_characteristic()`** on `SimplicialComplex`,
  `AbstractSimplicialComplex`, and `RegularCubeComplex`.
- **Neighborhoods**: `star`, `closure`, and `link` methods on every
  complex type, simplicial or cubical, for a `CellSet` (cells of every
  dimension, one `Subset` bitset per dimension) or, on a
  `SimplicialComplex`, a `Simplex` (`sc.link(Simplex([v]))`).
  `CellSet.from_lists(sc, [[], [e]])`, `from_indices`, `from_simplices`,
  and `from_subset` build sets; `cells[k]` is the `Subset` of k-cells,
  sets support `|`, `&`, `-`, and `==`, and `euler_characteristic()`
  and `simplices(sc)` read them. Each query walks only the incidence of
  the cells it reaches, so a vertex's link costs microseconds on any
  mesh. The link of an interior vertex is a circle on a surface and a
  sphere in a tetrahedral mesh, which the tests check. `boundary` and
  `interior` do the same for a pure set of cells, and `boundary_cells()`
  and `interior_cells()` for the whole complex: the boundary subcomplex,
  and every cell not on it. The free functions `star_of`, `link_of`,
  etc. take any `CellComplex`, a `ChainComplex` included.
- **Subsets** (`Subset`): a set of primal k-simplices or dual k-cells,
  stored as a bitset over their indices, for boundary conditions and
  regions with their own parameters. `sc.boundary_cells()[k]` gives the
  k-simplices on the mesh boundary; `Subset.from_indices`, `from_mask`,
  `from_simplices`, `full`, and `empty` build others. Subsets support
  `|`, `&`, `-`, `~`, `dual()`, `boundary(sc)`, `coboundary(sc)`,
  `inner_coboundary(sc)` (cells whose whole boundary is in the set), and
  `manifold_boundary(sc)` (the boundary of a region of n-simplices).
  `A.exclude_subset(s)` empties the rows of `s` in a `CSRMatrix` or
  `DiagonalOperator`, `restrict_to_subset` keeps only them, and
  `c.overwrite(s, g)` copies values on `s` into a cochain. Dirichlet
  conditions for `L u = f`:

  ```mojo
  var fixed = sc.boundary_cells()[0].copy()
  var I = DiagonalOperator.identity(N).restrict_to_subset(fixed)
  var A = L.exclude_subset(fixed) + I.to_csr()
  f.overwrite(fixed, g)                  # boundary rows read u = g
  var u = spsolve(A, f.v)
  ```
- **Typed cochains and operators** (`mojodec.core.typed`), a layer over
  the untyped core. `TypedComplex[n]` is a complex whose dimension is a
  compile-time parameter (checked when built), `TypedCochain[n, k,
  primal]` a k-cochain, and `MatrixOp` / `DiagonalOp[n, in_k, in_primal,
  out_k, out_primal]` a sparse or diagonal operator between two kinds of
  cochain. `tc.d[k, primal]()`, `tc.star[k, primal]()`,
  `tc.delta[k, primal]()`, `tc.laplacian[k, primal]()`, and
  `tc.laplace_beltrami[k, primal]()` agree with the untyped `d`, `star`,
  `delta`, `laplace_derham`, and `laplace_beltrami`. `A @ B` and `A @ c`
  compile only when the kinds match, and `d` past degree n−1 fails a
  compile-time check:

  ```mojo
  var tc = TypedComplex[2](vertices, triangles)
  var L = tc.star[2, DUAL]() @ tc.d[1, DUAL]() @ tc.star[1, PRIMAL]() @ tc.d[0, PRIMAL]()
  var f = L @ tc.cochain[0, PRIMAL]()       # TypedCochain[2, 0, PRIMAL]
  # tc.d[0, PRIMAL]() @ tc.star[1, PRIMAL]()   # error: no matching __matmul__
  ```

  Diagonal products stay diagonal, and `inverse()` swaps the input and
  output kinds. `TypedSubset[n, k, primal]` is a set of cells with its kind
  in its type: `tc.boundary_cells[k]()` and `tc.subset[k, primal](indices)`
  build one, `dual()`, `boundary()`, and `coboundary()` type their
  results, and `exclude_subset` / `restrict_to_subset` and
  `TypedCochain.overwrite` accept only subsets of the cells they act on.
  `wedge(a, b)` on typed primal cochains has degree `k + l` in its type
  and does not compile past the complex dimension, and `flat(tc, X)`
  returns a `TypedCochain[n, 1, PRIMAL]`.

  `whitney_innerproduct[k](tc)` is the Whitney mass matrix as an operator
  from primal k-cochains to dual (n−k)-cochains, the slot of
  `tc.star[k, PRIMAL]()`. `cg`, `lsqr`, `spsolve`, and `eigh_generalized`
  have typed overloads: an operator from one kind of cochain to another
  takes a right-hand side of the output kind and gives a solution (or
  eigenvectors, `TypedEighResult.mode(i)`) of the input kind, so a
  stiffness matrix from primal 0-cochains to dual n-cochains cannot be
  handed a primal right-hand side. `A.T()` transposes an operator from
  k-cochains to l-cochains into one from the duals of l-cochains to the
  duals of k-cochains (the adjoint under the pairing of primal k-cochains
  with dual (n−k)-cochains), so `d0.T()` takes dual (n−1)- to dual
  n-cochains:

  ```mojo
  var d0 = tc.d[0, PRIMAL]()
  var A = d0.T() @ tc.star[1, PRIMAL]() @ d0   # primal 0 -> dual 2
  var u = cg(A, b, rtol=1e-10).take_x()        # b dual 2, u primal 0
  var K = tc.d[0, DUAL]() @ whitney_innerproduct[2](tc) @ tc.d[1, PRIMAL]()
  var eig = eigh_generalized(K, whitney_innerproduct[1](tc))
  ```

  For a mesh read from a file, `TypedComplex[n](read_mesh(path))` checks
  the dimension, and `visit_typed(sc, visitor)` calls a
  `TypedComplexVisitor`'s `visit[n]` with the dimension of `sc` as a
  parameter, for code written once for every dimension.
  `BlockOp` is a 2 × 2 block of typed operators for mixed formulations,
  acting on a `CochainPair`; the blocks' kinds must agree along each row
  and column. `spsolve(A, b, values=..., fixed_first=..., fixed_second=...)`
  solves it with some unknowns given (Dirichlet data, a pinned pressure),
  dropping the equations on the same cells, which needs each equation to be
  of the dual kind of its unknown. DarcyFlow's mixed system:

  ```mojo
  var d1 = tc.d[1, PRIMAL]()
  var A = BlockOp(
      tc.star[1, PRIMAL]().to_matrix_op() * (-mu / k),  # primal 1 -> dual 1
      d1.T(),                                          # dual 0 -> dual 1
      d1,                                              # primal 1 -> primal 2
      tc.zero[0, DUAL, 2, PRIMAL](),
  )
  ```

  `GmshMesh.subset[k](tc, name)` gives a physical group as a
  `TypedSubset[n, k, PRIMAL]`. `complex()`, `untyped()`, `.mat`, and
  `.diag` lead back to the untyped API, and `TypedCochain(c)` checks an
  untyped cochain's kind at run time.
- **Parallel loops** on large meshes: circumcenters and primal volumes
  (from 50k simplices), Whitney mass matrices (from 4096 top simplices),
  and sparse matrix-vector products in `cg` and `lsqr` (from 1M stored
  entries), through MAX's `parallelize`. Smaller problems run serially, and
  the results are the same either way.
- **`icosahedron` and `icosphere`**, closed surfaces to test on.
- **`read_ply` and `write_ply`**, for meshes from MeshLab, Blender, and
  scanners (ASCII and binary, either byte order; `write_ply` takes
  `format=.BINARY`).
- **`read_gmsh`**, for triangle and tetrahedral meshes made by
  [Gmsh](https://gmsh.info), in `.msh` format 4.1 or 2.2, ASCII or binary.
  Gmsh's physical groups ("inlet", "walls", a material region) are kept,
  and `subset(sc, name, k)` gives a group's k-simplices as a `Subset`,
  faces of its cells included, ready for `exclude_subset` or a
  per-region coefficient:

  ```mojo
  var msh = read_gmsh("channel.msh", dim=2)
  var sc = SimplicialComplex(msh.mesh())
  var walls = msh.subset(sc, "walls", 0)        # vertices of the walls
  var core = msh.subset(sc, "core", 2)          # triangles of a region
  ```

  Once a `.geo` file defines physical groups, Gmsh saves only their
  elements, so tag the domain as well (or set `Mesh.SaveAll = 1`);
  `dim=2` makes a file without triangles an error rather than a mesh of
  edges. 2D meshes come back with two coordinate columns. Unnamed groups
  are reached by tag: `msh.subset(sc, group_dim=1, tag=42, dim=0)`.
- **The Sphere example**, a check of the Hodge-Laplacian spectrum against
  the spherical harmonics, for 0-, 1-, and 2-forms.
- **The Neighborhoods and Neighborhoods3D examples**, which draw stars,
  links, and boundaries on a grid of the square and on a tetrahedral solid
  torus, and check their topology.
- **The HeatConduction example**, steady heat flow through a plate of two
  materials meshed by Gmsh, whose physical groups supply the boundary
  conditions and the conductivity of each triangle.
- **Leapfrog time stepping** (`Leapfrog`, `TypedLeapfrog`) for
  `Mq q' = K p`, `Mp p' = -K^T q` with diagonal masses: Maxwell's
  equations with `e` on edges and `b` on triangles, or the acoustic wave
  equation. Each step is two sparse products, no solve. The system is
  skew by construction, and the scheme conserves a staggered energy
  `energy(q, p)` to rounding. The constructor raises unless the masses are
  positive (Hodge stars are, on a well-centered mesh) and `dt` is below
  the stability limit `max_stable_dt`, found by power iteration:

  ```mojo
  var K = P.T() @ tc.d[1, PRIMAL]().T() @ tc.star[2, PRIMAL]().to_matrix_op()
  var lf = TypedLeapfrog(tc.star[1, PRIMAL](), tc.star[2, PRIMAL](), K, dt)
  lf.step(e, b, steps=100)                 # e: primal 1, b: primal 2
  ```

  Mixing up the kinds (a mass that doesn't map its field to the dual
  kind, or a `K` into the wrong kind) fails to compile.
- **The CavityWaves example**, Maxwell's equations stepped in time in an
  equilateral-triangle cavity.
- **Implicit midpoint and Crank–Nicolson** (`ImplicitMidpoint`,
  `CrankNicolson`, typed versions): the wave system with any symmetric
  positive definite mass for `q` (the Whitney mass) and a diagonal one for
  `p`, one SPD solve per step, unconditionally stable and conserving the
  plain energy exactly. Maxwell then runs on meshes that are not
  well-centered. `CrankNicolson` steps `M u' = -K u + f` with monotone
  energy decay and the steady state as its limit.
- **Operator traits, preconditioned CG, and MINRES.** `cg` and `minres`
  take any `LinearOperator` (`CSRMatrix`, `DiagonalOperator`, the typed
  ops, a `BlockOp` applied block by block) and, as `M=`, any
  `Preconditioner`, a trait for symmetric positive definite operators
  (`DiagonalOperator`, `jacobi(A)`, `Identity`). `minres` solves symmetric
  indefinite systems, such as a mixed formulation's saddle-point
  `BlockOp`, without assembling it:

  ```mojo
  var res = cg(A, b, M=jacobi(A), rtol=1e-10)   # typed: M maps A's output kind back
  var x = minres(darcy, b, rtol=1e-12).take_x() # CochainPair of flux and pressure
  ```

  Both raise on an operator that is not symmetric (two probe products), so
  a matrix such as `delta(d(.))`, self-adjoint only in the Hodge-star
  inner product, cannot be passed by mistake.
- **Kuhn grids and flat tori** (`kuhn_grid`, `flat_torus`) in any
  dimension: each box cell split into d! simplices, and the periodic
  version embedded in R^(2d) so that it is exactly the flat torus (Betti
  numbers `binomial(d, k)`, Laplacian eigenvalues in closed form). The
  circumcentric dual is the dual cube grid, so diagonal simplices have
  zero dual volume; with Whitney forms the 0-form stiffness is exactly the
  finite-difference Laplacian, which the tests check entry by entry.
  `mobius_band` and `klein_bottle` glue a 2D grid with a flip: test meshes
  that every orientation check refuses.
- **Quadrature on simplices** (`grundmann_moeller`, `integrate_simplices`):
  Grundmann–Möller rules for any dimension and odd degree up to 21, exact
  for polynomials of that degree, applied to one simplex or to every
  k-simplex of a complex with a function or closure as the integrand.
- **The de Rham map and Whitney forms** (`derham_map`, `whitney_form`):
  integrate a k-form, given as a function or closure of its components,
  over every k-simplex into a cochain (exact for polynomial forms, so
  `R d = d R` to rounding), and evaluate a cochain's Whitney form at a
  point of an element (`R W = id`, `d W = W d`; the Whitney inner products
  built from it match `whitney_innerproduct`).
- **L² projection and convergence tests** (`l2_projection`,
  `whitney_load`, `l2_error`): the best L² fit of a form by Whitney forms
  (Jacobi-preconditioned CG on the mass matrix), the Galerkin load vector,
  and `||W c - omega||`. `tests/test_convergence.mojo` checks the rates on
  Kuhn grids in 1–3D for every degree: 2 for 0-forms, 1 above, the
  projection never worse than `W R`, and Poisson's equation second order
  within 5 times the best approximation.
- **Uniform refinement and prolongation** (`refine`,
  `Refinement.prolongation`): Freudenthal subdivision of every element
  into 2^n children in any dimension, conforming, coherently oriented,
  and with a vertex numbering that keeps the number of shapes bounded
  (3 for tetrahedra over any number of levels); the exact prolongation
  `P_k = R_fine W_coarse` of k-cochains, which commutes with `d` and
  preserves the Whitney inner products. The convergence tests use it for
  Poisson on nested irregular meshes and a two-level error estimate.
- **Mixed Hodge Laplacian and block preconditioners.** The convergence
  tests solve Arnold–Falk–Winther's mixed formulation on 1- and 2-forms
  (first order in `sigma`, `u`, and `du`). `BlockDiagonal(P1, P2)` makes a
  preconditioner of two, and `DenseCholesky(A)` applies `A^-1` exactly on
  small problems; with the blocks' natural norms MINRES takes 7
  iterations at every mesh size, where Jacobi takes 79 to 424.
- **Backward errors of eigenpairs.** `eig.residuals(K, M)` gives each
  pair's `||K x - lambda M x|| / ((||K|| + |lambda| ||M||) ||x||)`, the
  relative change of `K` and `M` that would make it exact: scale-free, so
  one threshold (a modest multiple of n times 2.2e-16) serves every
  problem. It takes sparse or typed operators, so the check costs n
  sparse products. Sphere and ResonantCavity report it (5e-15, 7e-15).

## Differences from PyDEC

- **Cochains.** PyDEC's `cochain.v` is either a vector or a sparse basis.
  Mojo's static types split this into `Cochain` (a NuMojo vector) and
  `CochainBasis` (a `CSRMatrix`); `d`, `star`, `delta`, and the Laplacians
  are generic over both. Adding or subtracting cochains also requires
  both to be primal or both dual, and `c[s]` requires a simplex of the
  cochain's own dimension (k, or n-k for a dual cochain); both raise
  otherwise.
- **Eager geometry.** PyDEC computes circumcenters, volumes, and Hodge stars
  lazily via `__getattr__`. `SimplicialComplex` computes them at
  construction; pass `compute_geometry=False` to skip that, e.g. for
  degenerate meshes.
- **Precise geometry.** The results agree with PyDEC's to rounding, but
  are computed so as not to lose digits that PyDEC loses. Volumes of
  simplices below the embedding dimension come from a Householder QR
  factorization rather than the Gram determinant, which cannot tell a
  very thin simplex from a flat one. Circumcenters are solved for with
  positions relative to a vertex, not absolute positions, which keeps
  their precision on meshes far from the origin; triangles use a closed
  form whose denominator, the squared area, also comes from QR, so even
  slivers get their circumcenters. Barycentric gradients, and the Whitney
  mass matrices built from them, come from the same QR rather than from
  the Gram matrix of the edges (`barycentric_gradients` in PyDEC), which
  squares a thin simplex's conditioning: for a triangle 1e-8 high, PyDEC's
  formula gives gradients that are wrong in the first digit, and the
  Whitney minors are taken in the simplex's own frame by Cauchy-Binet.
- **Zero dual volumes raise.** Where a dual cell has zero volume (the
  hypotenuses of `simplicial_grid_2d`'s right triangles), the Hodge star
  is zero and its inverse infinite. PyDEC carries on and returns
  infinities and NaNs from `star` and `delta`; MojoDEC's `star` from dual
  to primal (and so `delta` where it needs it) raises, naming the
  simplex. `sc[k].star_inv` itself still holds the infinities, and
  `sc[k].require_finite_star_inv()` checks a level.
- **The dual `d` and `delta` in odd dimensions.** PyDEC's `d` on dual
  k-cochains is `(-1)^k` times the boundary operator, which makes its
  `delta` minus the adjoint of `d` in odd dimensions: in 1-D and 3-D its
  `laplace_beltrami` is negative semidefinite. MojoDEC's dual `d` carries
  `(-1)^(n-k)`, so `delta` is the adjoint of `d` under the Hodge star
  inner products, and the Laplacians are positive semidefinite, in every
  dimension. In 2-D (and any even dimension) the two agree exactly.
- **Malformed meshes raise.** `SimplicialComplex` and the top simplices of
  `AbstractSimplicialComplex` reject an element that repeats a vertex and
  two elements with the same vertices; PyDEC accepts both and builds wrong
  operators (`[[0,1,2],[0,1,2]]` gets an empty boundary and doubled dual
  volumes). `SimplicialComplex` also requires a pseudomanifold (no
  (n−1)-simplex is a face of three or more elements) with coherently
  oriented elements, which the dual `d`, `delta`, and `wedge` assume;
  PyDEC builds wrong dual operators otherwise. The error names the
  offending face and elements, and says whether `orient_coherently` can
  fix the mesh.
- **Simplex lookup.** PyDEC's `simplex_to_index` dict and
  `index_to_simplex` list are methods here: `sc[k].simplex_to_index(s)`
  binary-searches the sorted simplex rows and `sc[k].index_to_simplex(i)`
  builds the `Simplex`, so no per-level dictionary is kept in memory.
  `s in sc[k]` tests membership.
- **Shared complexes.** A `SimplicialComplex` holds its data behind an
  `ArcPointer`, so copies and cochains share one complex, as Python
  references do.
- **Operators.** `A @ B` and `A @ x` are matrix products (PyDEC's `A * B`, on
  SciPy matrices); `A * s` scales.
- **Hodge stars.** `sc[k].star` and `sc[k].star_inv` are
  `DiagonalOperator`s rather than sparse matrices. `D @ x` is elementwise,
  `D @ A` and `A @ D` scale the rows or columns of a `CSRMatrix`, and
  `D @ E` stays diagonal, so none of them needs a general sparse product.
  It also has `inverse()`, `inner_product(a, b)` (`aᵀ D b`), and `norm(x)`.
  Use `to_csr()` where a `CSRMatrix` is needed, e.g. as a `bmat` block.
- **Whitney mass matrices** weight each simplex by its unsigned volume, so
  inconsistently oriented meshes still give positive-definite matrices.
- **`AbstractSimplicialComplex`** maps `B1`'s rows to vertex ids, so vertex
  numberings with gaps work.
- **Chain complexes.** PyDEC's `chain_complex()` and `cochain_complex()`
  return lists padded with zero matrices (a `1 x N_0` `B_0` and a
  `1 x N_n` `d_n`). Here `chain_complex()` returns a `ChainComplex`
  holding both kinds without padding: `cc.boundary(k)` is `B_k` for
  k = 1..n, `cc.coboundary(k)` is `d_k` for k = 0..n-1, and
  `cc.num_cells(k)` is `N_k`. `sc[k].boundary` and `sc[k].d` still give
  the operators of one level directly. The constructor raises unless
  `B_k B_{k+1} = 0` and each `d_k` is `B_{k+1}^T`.
- **File formats are types.** `write_array` and `write_mesh` take an
  `ArrayFormat` (`.BASIC`, `.ASCII`, `.BINARY`), `write_sparse` a
  `SparseFormat`, `write_ply` a `PlyFormat`, and `write_vtu` a
  `VTKFormat`, in place of PyDEC's strings, so a misspelled format does not
  compile. `read_array` converts binary data from the file's dtype
  directly, so 64-bit integers keep all their digits, as in PyDEC.
- **Primal or dual is a type.** PyDEC's `is_primal` flags are a
  `Primality` here: `sc.get_cochain(1, .DUAL)`,
  `Subset.empty(sc, 0, .DUAL)`, and `c.primality()`. `p.level(n, k)` is
  the level holding the cells (`k`, or `n - k` for dual cells), and
  `p.dual()` the other kind.
- **Named results.** Functions that return several things return a
  struct with named fields rather than a tuple:
  - `simplicial_grid_2d`, `icosahedron`, `icosphere`, `triangulate_ncube`,
    and `loop_subdivision` return a `SimplicialMesh`, so
    `SimplicialComplex(simplicial_grid_2d(8))` works;
  - `cube_grid` returns a `CubeGrid` (`vertices`, `corners`);
  - `dense.eigh` and `eigh_generalized` return an `EighResult` (`values`,
    `vectors`);
  - `cg` and `lsqr` return a `SolveResult` whose `stop` is a `StopReason`
    (`.CONVERGED`, `.MAX_ITER`, `.BREAKDOWN`), where SciPy's `info` cannot
    tell a CG breakdown from the iteration limit. `residual_norm` is
    `||b - A x||` for both, LSQR also reports `normal_residual_norm`
    (`||A^T r||`), and `res^.take_x()` moves the solution out;
  - `simplex_quivers` and `cube_quivers` return `Quivers` (`bases`,
    `vectors`), `circumcenter` a `Circumsphere`,
    `simplex_array_boundary` and `cube_array_boundary` a
    `FacesAndBoundary`, `to_coo()` a `COOEntries`,
    `massmatrix_rowcols` a `COOPattern`, and `dual_edge_pieces` a
    `DualEdgePieces` that also records each piece's edge.
- **Rips complexes without edges** are 0-dimensional: the chain complex
  has no operators, where PyDEC appends a zero `N_0 x 1` boundary matrix
  for an edge that does not exist.
- **Python-3 breakage.** Code that no longer runs under current Python or
  NumPy works here: `simplicial_grid_2d`, `triangulate_ncube`, and
  `cube_array_boundary`.
- **Linear algebra.** NuMojo 0.12's `linalg.solve` factors without row
  exchanges, which fails on DEC's saddle-point systems, and its Jacobi
  `eigh` is slow for a few hundred unknowns. MojoDEC has its own pivoted LU
  and Householder/QL `eigh` (`routines/dense.mojo`). `spsolve` and
  `eigh_generalized` are dense, which suits the example sizes;
  `eigh_generalized(K, M)` also takes a diagonal `M` (a Hodge star) and
  then only rescales `K`, with no Cholesky factorization.
- **Symmetric solvers check symmetry.** SciPy's `cg` accepts any matrix;
  MojoDEC's `cg` and `minres` raise unless `u^T A w = w^T A u` on two
  probe vectors. PyDEC's HodgeDecomposition passes `delta(d(.))`
  (`star^-1 d^T star d`, not symmetric) to `cg`; MojoDEC's driver solves
  the symmetric `(star d^T star d) alpha = star b` with `M = star^-1`,
  which is CG in the star inner product, and gets the same harmonic space
  (to a principal angle of 3e-6).

## Examples

```sh
pixi run examples                      # all eleven; figures saved as PNG files
pixi run examples DarcyFlow --show     # one, with its figures in windows
```

Each writes its results to text files and draws PyDEC's figures to PNG
files, so it runs without a display; `--show` also opens them in
matplotlib windows. RipsComplex, HodgeDecomposition, DarcyFlow,
HeatConduction, ResonantCavity, CavityWaves, Sphere, and both Neighborhoods examples also write their fields to `.vtu` files for
ParaView.
[`examples/README.md`](examples/README.md) covers running them by hand,
what each one produces, viewing the VTK files in ParaView, and plotting
from your own program.

| Example | Result |
|---|---|
| Ranking | matches PyDEC's ranking |
| RipsComplex | same complex as PyDEC (300 / 2831 / 10955 simplices), harmonic chain with norm 7.6274 |
| HodgeDecomposition | four harmonic 1-cochains from integer cocycles, periods around the holes the identity, `|d h|` 2e-16, `|δ h|` 7e-11; a random cochain's harmonic part is their combination to 2e-7 (see below) |
| DarcyFlow | pressures agree with the exact solution to 1e-14 |
| HeatConduction | mesh and boundary pieces from a Gmsh file; the piecewise-linear temperature through two materials to 7e-15, and the heat in through the hot edge equals the heat out through the cold one and the exact 25/26 (not from PyDEC) |
| ResonantCavity | eigenvalues 1, 1, 2, 4, 4, 5, 5, 8, 9, 9, 10, 10, 13, 13, … (the exact m²+n²); largest backward error 7e-15 |
| CavityWaves | resonances 0.9986, 0.9986, 2.987, 3.977, 3.977, 6.93, 6.93, 8.88 times (4π/3)² (the exact m²+mn+n² of the equilateral triangle); leapfrog from the lowest mode follows cos(ω_dt t) to 3e-12 over five periods, staggered energy constant to 2e-15 (not from PyDEC) |
| Sphere | Hodge-Laplacian spectra near n(n+1) for 0-, 1-, and 2-forms; the 1-form spectrum is the 0- and 2-form spectra merged (to 3e-12); Gauss-Bonnet exact; Gaussian and mean curvature 1.02 and 1.0001; largest backward error of any eigenpair 5e-15 (not from PyDEC) |
| Neighborhoods | interior links are hexagons (χ = 0), boundary links paths (χ = 1); Bd(Cl(St(v))) = Lk(v); every ring boundary has χ = 0; the square's boundary has 4n edges and an empty boundary (not from PyDEC) |
| Neighborhoods3D | interior vertex links are spheres (χ = 2), boundary links disks (χ = 1); growing rings are balls (χ = 1) until they meet themselves around the hole (one pinched step), then the solid torus (χ = 0) bounded by a torus; χ(Bd R) = 2 χ(R) whenever Bd R is a surface (not from PyDEC) |
| MakeMesh | writes a mesh and reads it back; matches `ExpectedOutput/` |

**HodgeDecomposition.** The mesh is not well-centered, so `d(δ(·))` on
2-cochains is nonsymmetric and indefinite. PyDEC's CG solve of that system
does not converge (PyDEC's call to SciPy's `cg` returns `info > 0`), and
the resulting "harmonic" forms are not harmonic. MojoDEC solves that
system with LSQR, and the 0-cochain system in its symmetric form with
preconditioned CG (see *Symmetric solvers check symmetry* above). The
basis itself comes from `harmonic_basis`, one form per hole, rather than
from four random cochains; the decomposition of one random cochain checks
it.

**DarcyFlow, HeatConduction, ResonantCavity, and CavityWaves** are written with the
typed layer (`TypedComplex`, typed operators and solves); the others use
the untyped API, as PyDEC's drivers do.

**ResonantCavity.** The original PyDEC driver no longer runs on current
NumPy: `whitney_innerproduct` assigns a 1×1 array to a scalar. This one
keeps every edge and decouples the boundary edges (`P^T K P`,
`P^T M P + Q^T M Q`) rather than deleting their rows and columns; they add
zero eigenvalues, which the report skips with the gradients' ones.

## License

The MIT License (MIT)

PyDEC, which MojoDEC is largely inspired by, is under its own license; see
[NOTICE](NOTICE).
