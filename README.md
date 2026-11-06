## **MojoDEC: A Mojo Library for Discretizations of Exterior Calculus.**

MojoDEC is a Mojo port of [PyDEC](https://github.com/hirani/pydec),
implementing Discrete Exterior Calculus (DEC) and lowest order finite element
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
- Euler characteristics and simplicial neighborhoods (star, closure, link,
  boundary, interior) on every complex type
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
  sphere, and neighborhoods on 2D and 3D meshes

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

## Layout

Modelled on NuMojo (`core` for data structures, `routines` for algorithms)
rather than PyDEC's `pydec/pydec/<subpackage>` nesting.

| MojoDEC | PyDEC |
|---|---|
| `mojodec/core/sparse.mojo` — `CSRMatrix` | `scipy.sparse.csr_matrix` (PyDEC's sparse matrices) |
| `mojodec/core/diagonal.mojo` — `DiagonalOperator` | (new; PyDEC uses `scipy.sparse` diagonal matrices) |
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
| `mojodec/routines/linalg.mojo` — `cg`, `lsqr`, `spsolve`, `eigh_generalized` | `scipy.sparse.linalg`, `scipy.linalg.eig` (called by PyDEC) |
| `mojodec/routines/dense.mojo` — pivoted LU, `det`, `inv`, `cholesky`, `eigh` | `numpy.linalg` (called by PyDEC) |
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
  the operators of one level directly.
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

## Examples

```sh
pixi run examples                      # all ten; figures saved as PNG files
pixi run examples DarcyFlow --show     # one, with its figures in windows
```

Each writes its results to text files and draws PyDEC's figures to PNG
files, so it runs without a display; `--show` also opens them in
matplotlib windows. RipsComplex, HodgeDecomposition, DarcyFlow,
HeatConduction, ResonantCavity, Sphere, and both Neighborhoods examples also write their fields to `.vtu` files for
ParaView.
[`examples/README.md`](examples/README.md) covers running them by hand,
what each one produces, viewing the VTK files in ParaView, and plotting
from your own program.

| Example | Result |
|---|---|
| Ranking | matches PyDEC's ranking |
| RipsComplex | same complex as PyDEC (300 / 2831 / 10955 simplices), harmonic chain with norm 7.6274 |
| HodgeDecomposition | four harmonic 1-cochains with `|d h|, |δ h|` ≈ 0 (see below) |
| DarcyFlow | pressures agree with the exact solution to 1e-14 |
| HeatConduction | mesh and boundary pieces from a Gmsh file; the piecewise-linear temperature through two materials to 7e-15, and the heat in through the hot edge equals the heat out through the cold one and the exact 25/26 (not from PyDEC) |
| ResonantCavity | eigenvalues 1, 1, 2, 4, 4, 5, 5, 8, 9, 9, 10, 10, 13, 13, … (the exact m²+n²) |
| Sphere | Hodge-Laplacian spectra near n(n+1) for 0-, 1-, and 2-forms; the 1-form spectrum is the 0- and 2-form spectra merged (to 3e-12); Gauss-Bonnet exact; Gaussian and mean curvature 1.02 and 1.0001 (not from PyDEC) |
| Neighborhoods | interior links are hexagons (χ = 0), boundary links paths (χ = 1); Bd(Cl(St(v))) = Lk(v); every ring boundary has χ = 0; the square's boundary has 4n edges and an empty boundary (not from PyDEC) |
| Neighborhoods3D | interior vertex links are spheres (χ = 2), boundary links disks (χ = 1); growing rings are balls (χ = 1) until they meet themselves around the hole (one pinched step), then the solid torus (χ = 0) bounded by a torus; χ(Bd R) = 2 χ(R) whenever Bd R is a surface (not from PyDEC) |
| MakeMesh | writes a mesh and reads it back; matches `ExpectedOutput/` |

**HodgeDecomposition.** The mesh is not well-centered, so `d(δ(·))` on
2-cochains is nonsymmetric and indefinite. PyDEC's CG solve of that system
does not converge (PyDEC's call to SciPy's `cg` returns `info > 0`), and
the resulting "harmonic" forms are not harmonic. This port solves that
system with LSQR.

**DarcyFlow, HeatConduction, and ResonantCavity** are written with the
typed layer (`TypedComplex`, typed operators and solves); the others use
the untyped API, as PyDEC's drivers do.

**ResonantCavity.** The original PyDEC driver no longer runs on current
NumPy: `whitney_innerproduct` assigns a 1×1 array to a scalar. This one
keeps every edge and decouples the boundary edges (`P^T K P`,
`P^T M P + Q^T M Q`) rather than deleting their rows and columns; they add
zero eigenvalues, which the report skips with the gradients' ones.

## License

BSD 3-Clause, as PyDEC.
