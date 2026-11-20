# ===----------------------------------------------------------------------=== #
# molina: linear algebra in Mojo (sparse and dense matrices, direct and
# iterative solvers, eigensolvers, multigrid), split from MojoDEC.
# ===----------------------------------------------------------------------=== #
"""
Linear Algebra in Mojo (`molina`).
==================================
Linear algebra in pure Mojo: `RealArray` vectors and dense matrices,
`CSRMatrix`, the `LinearOperator` and `Preconditioner` traits, dense
factorizations and `eigh`, AMD ordering, simplicial and supernodal sparse
LDL^T and Cholesky, Krylov solvers (`cg`, `minres`, `lsqr`), shift-invert
Lanczos (`eigsh`), and smoothed-aggregation multigrid. It imports
nothing from MojoDEC, which builds on it, nor from NuMojo.

```mojo
from molina import CSRMatrix, cg, jacobi, real_vector_from_list

def main() raises:
    var A = CSRMatrix.from_coo(
        2, 2, [0, 0, 1, 1], [0, 1, 0, 1], [4.0, 1.0, 1.0, 3.0]
    )
    var b = real_vector_from_list([1.0, 2.0])
    var x = cg(A, b, M=jacobi(A), rtol=1e-12).take_x()
```

Modules
-------
- `real_array`, `utils`: `RealArray` and its builders and level-1
  kernels (`dot`, `norm2`, `axpy`).
- `operator`, `diagonal`, `sparse`, `index_set`: the operator traits,
  `DiagonalOperator`, `CSRMatrix`, and the `IndexSet` trait that row
  selection takes.
- `dense`: pivoted LU, Cholesky, Householder/QL `eigh`, tridiagonal
  eigenvalues.
- `ordering`, `sparse_ldlt`, `supernodal`: the sparse direct solvers.
- `linalg`: the iterative solvers, `spsolve`, `eigh_generalized`.
- `lanczos`: `eigsh`. `amg`: `SmoothedAggregation`.
- `parallel`: `parallel_ranges`, the loops on MAX's thread pool
  (`MOLINA_NUM_THREADS` caps them).
"""

comptime __version__: String = "0.1.0"
"""The molina version."""

from molina.real_array import RealArray
from molina.utils import (
    axpy,
    dot,
    norm2,
    real_array,
    real_vector,
    real_vector_from_list,
)
from molina.index_set import IndexSet
from molina.operator import (
    BlockDiagonal,
    Identity,
    LinearOperator,
    Preconditioner,
)
from molina.diagonal import DiagonalOperator
from molina.sparse import COOEntries, CSRMatrix
from molina.dense import EighResult
from molina.ordering import amd_ordering
from molina.sparse_ldlt import (
    FactorLayout,
    FillOrdering,
    Inertia,
    SparseCholesky,
    SparseLDLT,
    cholesky_nonzeros,
)
from molina.linalg import (
    DenseCholesky,
    DirectMethod,
    EigenvalueRange,
    NullSpace,
    SolveResult,
    StopReason,
    cg,
    eigenvalue_range,
    eigh_generalized,
    jacobi,
    lsqr,
    minres,
    spsolve,
)
from molina.lanczos import eigsh
from molina.amg import Relaxation, SmoothedAggregation
