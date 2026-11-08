# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
MojoDEC Top-Level Package (`mojodec`).
======================================
Discrete exterior calculus on simplicial and cubical complexes, largely
inspired by PyDEC, with NuMojo arrays in place of NumPy and `CSRMatrix` in
place of `scipy.sparse`.

```mojo
import numojo as nm
from mojodec import SimplicialComplex, d, star, delta

def main() raises:
    var V = nm.fromstring[nm.f64]("[[0,0],[1,0],[1,1],[0,1]]")
    var S = nm.fromstring[DType.int]("[[0,1,3],[1,2,3]]")
    var sc = SimplicialComplex(V, S)
    var L = delta(d(sc.get_cochain_basis(0))).v.copy()   # 0-form Laplacian
```

Layout
------
- `mojodec.core`: data structures (sparse matrices, simplices, meshes,
  simplicial / abstract / Rips / cube complexes, cochains, k-d trees).
- `mojodec.routines`: algorithms (simplex and cube arrays, geometry,
  combinatorics, Whitney inner products, solvers, mesh generation, I/O,
  the wedge product and flat, surface curvature, mesh quality).
- `mojodec.vis`: matplotlib drawing through Python interop, and sampling
  of 1-cochains for `quiver` plots. Not re-exported here, so programs that
  do not plot never touch Python.
"""

comptime __version__: String = "0.1.0"
"""The MojoDEC version."""

from mojodec.core.sparse import COOEntries, CSRMatrix
from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.primality import Primality
from mojodec.core.subset import Subset
from mojodec.core.cell_set import CellComplex, CellSet
from mojodec.core.chain_complex import ChainComplex
from mojodec.core.simplex import Simplex
from mojodec.core.mesh import SimplicialMesh, RegularCubeMesh
from mojodec.core.ncube import NCube
from mojodec.core.simplicial_complex import SimplicialComplex, SimplexLevel
from mojodec.core.abstract_simplicial_complex import AbstractSimplicialComplex
from mojodec.core.rips_complex import (
    RipsComplex,
    rips_simplices,
    rips_chain_complex,
)
from mojodec.core.regular_cube_complex import RegularCubeComplex, CubeLevel
from mojodec.core.cochain import (
    Cochain,
    CochainBasis,
    CochainLike,
    d,
    star,
    delta,
    laplace_derham,
    laplace_beltrami,
)
from mojodec.core.typed import (
    TypedComplex,
    TypedCochain,
    TypedSubset,
    TypedComplexVisitor,
    visit_typed,
    MatrixOp,
    DiagonalOp,
    PRIMAL,
    DUAL,
)
from mojodec.core.typed_block import BlockOp, CochainPair
from mojodec.core.kd_tree import KDTree
from mojodec.core.utils import IntArray, RealArray

from mojodec.routines.simplex_array import (
    FacesAndBoundary,
    simplex_array_searchsorted,
    simplex_array_parity,
    simplex_array_boundary,
)
from mojodec.routines.cube_array import cube_array_search, cube_array_boundary
from mojodec.routines.combinatorial import (
    combinations,
    permutations,
    permutation_parity,
    relative_parity,
)
from mojodec.routines.geometry import (
    Circumsphere,
    unsigned_volume,
    signed_volume,
    circumcenter,
    circumcenter_barycentric,
    is_wellcentered,
    barycentric_gradients,
)
from mojodec.routines.innerproduct import (
    COOPattern,
    whitney_innerproduct,
    regular_cube_innerproduct,
    massmatrix_rowcols,
)
from mojodec.routines.linalg import (
    cg,
    lsqr,
    spsolve,
    eigh_generalized,
    SolveResult,
    StopReason,
    TypedEighResult,
)
from mojodec.routines.dense import (
    EighResult,
)
from mojodec.routines.mesh_generation import (
    CubeGrid,
    simplicial_grid_2d,
    cube_grid,
    triangulate_ncube,
    loop_subdivision,
    icosahedron,
    icosphere,
)
from mojodec.routines.wedge import wedge
from mojodec.routines.musical import DualEdgePieces, flat, dual_edge_pieces
from mojodec.routines.curvature import (
    angle_defect,
    gaussian_curvature,
    mean_curvature,
    mean_curvature_normal,
)
from mojodec.routines.topology import (
    star_of,
    closure_of,
    link_of,
    boundary_of,
    interior_of,
)
from mojodec.routines.quality import (
    well_centeredness,
    well_centered_fraction,
    shape_regularity,
)
from mojodec.routines.io.arrayio import (
    ArrayFormat,
    SparseFormat,
    read_array,
    read_sparse,
    read_header,
    write_array,
    write_sparse,
)
from mojodec.routines.io.meshio import read_mesh, write_mesh
from mojodec.routines.io.text import loadtxt
from mojodec.routines.io.vtk import (
    VTKField,
    VTKFormat,
    write_vtu,
    write_vtu_cubes,
)
from mojodec.routines.io.ply import PlyFormat, read_ply, write_ply
from mojodec.routines.io.gmsh import read_gmsh, GmshMesh, GmshGroup
