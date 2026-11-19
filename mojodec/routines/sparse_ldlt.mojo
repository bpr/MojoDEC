# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Typed sparse factorizations (`mojodec.routines.sparse_ldlt`).
=============================================================
`TypedSparseLDLT`, `TypedSparseCholesky`, and `BlockSparseLDLT`: the
factorizations of `mojodec.routines.sparse_ldlt_core` for typed operators,
whose solves take and give cochains. The untyped `SparseLDLT`,
`SparseCholesky`, `FactorLayout`, `FillOrdering`, `Inertia`, and
`cholesky_nonzeros` are re-exported from there.
"""

from std.math import isfinite, sqrt

from mojodec.core.operator import LinearOperator, Preconditioner
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.typed import MatrixOp, TypedCochain
from mojodec.core.typed_block import BlockOp, CochainPair
from mojodec.core.parallel import parallel_ranges
from mojodec.core.utils import (
    RealArray,
    cols_of,
    fget,
    fset,
    real_array,
    real_vector,
    rows_of,
)
from mojodec.routines.ordering import amd_ordering
from mojodec.routines.supernodal import _Supernodal, etree_postorder
from mojodec.routines.sparse_ldlt_core import (
    FactorLayout,
    FillOrdering,
    Inertia,
    SparseCholesky,
    SparseLDLT,
    cholesky_nonzeros,
)


# ===----------------------------------------------------------------------=== #
# Typed factorizations
# ===----------------------------------------------------------------------=== #


def _same_complex(
    op: Optional[SimplicialComplex], b: SimplicialComplex, who: String
) raises:
    if op and not op.value().is_same(b):
        raise Error(String(who, ".solve: b is on a different complex"))


def _columns_of(vs: List[RealArray], n: Int, who: String) raises -> RealArray:
    """The vectors as the columns of an `n x m` array."""
    var B = real_array(n, len(vs))
    for c in range(len(vs)):
        if vs[c].size != n:
            raise Error(
                String(
                    who,
                    ".solve: right-hand side ",
                    c,
                    " has length ",
                    vs[c].size,
                    ", not ",
                    n,
                )
            )
        for i in range(n):
            fset(B, i, c, vs[c].unsafe_get(i))
    return B^


def _column(X: RealArray, c: Int, lo: Int, hi: Int) raises -> RealArray:
    """Rows `lo..hi` of column `c` of `X`."""
    var x = real_vector(hi - lo)
    for i in range(lo, hi):
        x.unsafe_set(i - lo, fget(X, i, c))
    return x^


def _typed_solve_many[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](
    factor: SparseLDLT,
    complex: Optional[SimplicialComplex],
    bs: List[TypedCochain[n, out_k, out_primal]],
    who: String,
) raises -> List[TypedCochain[n, in_k, in_primal]]:
    """`factor.solve` of the cochains as one `n x m` block."""
    var vs = List[RealArray](capacity=len(bs))
    for c in range(len(bs)):
        _same_complex(complex, bs[c].complex, who)
        vs.append(bs[c].v.copy())
    var out = List[TypedCochain[n, in_k, in_primal]](capacity=len(bs))
    if len(bs) == 0:
        return out^
    var X = factor.solve(_columns_of(vs, factor.n, who))
    for c in range(len(bs)):
        out.append(
            TypedCochain[n, in_k, in_primal](
                bs[c].complex, _column(X, c, 0, factor.n)
            )
        )
    return out^


struct TypedSparseLDLT[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](Movable):
    """`SparseLDLT` of a typed operator: solves take cochains of the
    operator's output kind and give cochains of its input kind.

    ```mojo
    var W1 = whitney_innerproduct[1](tc)       # primal 1 -> dual 1
    var F = TypedSparseLDLT(W1)
    var e = F.solve(f)                         # f: dual 1, e: primal 1
    ```

    A symmetric operator built from the complex, `B^T S B` or a mass,
    takes a kind to its dual kind (primal k-cochains to dual
    (n-k)-cochains, or the reverse); others do not compile.

    Parameters:
        n: The complex dimension.
        in_k: The degree of the solution.
        in_primal: Whether the solution is primal.
        out_k: The degree of the right-hand side, `n - in_k`.
        out_primal: Whether the right-hand side is primal, `not in_primal`.
    """

    var factor: SparseLDLT
    """The untyped factorization."""
    var complex: Optional[SimplicialComplex]
    """The operator's complex (`None` if it was built from a bare
    matrix)."""

    def __init__(
        out self,
        A: MatrixOp[
            Self.n, Self.in_k, Self.in_primal, Self.out_k, Self.out_primal
        ],
        *,
        ordering: FillOrdering = FillOrdering.AMD,
        perm: Optional[List[Int]] = None,
        signs: Optional[List[Int]] = None,
        regularization: Float64 = 0,
        refinement_steps: Int = 1,
        layout: FactorLayout = FactorLayout.AUTO,
    ) raises:
        """Factor `A`; the options are `SparseLDLT`'s.

        Args:
            A: A symmetric operator from a kind to its dual kind.
            ordering: The fill-reducing permutation.
            perm: A permutation to use instead of `ordering`.
            signs: The expected sign of each pivot, by cell.
            regularization: The `delta` of dynamic regularization.
            refinement_steps: Iterative refinement steps per solve.
            layout: Simplicial or supernodal `L`, or `AUTO` to choose.

        Raises:
            As `SparseLDLT`.
        """
        comptime assert (
            Self.out_k == Self.n - Self.in_k
            and Self.out_primal == (not Self.in_primal)
        ), "TypedSparseLDLT: A must map a kind to its dual kind"
        self.factor = SparseLDLT(
            A.mat,
            ordering=ordering,
            perm=perm,
            signs=signs,
            regularization=regularization,
            refinement_steps=refinement_steps,
            layout=layout,
        )
        self.complex = A.complex

    def solve(
        self, b: TypedCochain[Self.n, Self.out_k, Self.out_primal]
    ) raises -> TypedCochain[Self.n, Self.in_k, Self.in_primal]:
        """Solve `A x = b`.

        Args:
            b: The right-hand side, of `A`'s output kind.

        Returns:
            `x`, of `A`'s input kind.

        Raises:
            If `b` is on another complex than `A`.
        """
        _same_complex(self.complex, b.complex, "TypedSparseLDLT")
        return TypedCochain[Self.n, Self.in_k, Self.in_primal](
            b.complex, self.factor.solve(b.v)
        )

    def solve(
        self, bs: List[TypedCochain[Self.n, Self.out_k, Self.out_primal]]
    ) raises -> List[TypedCochain[Self.n, Self.in_k, Self.in_primal]]:
        """Solve `A x = b` for every cochain of `bs` at once, as
        `SparseLDLT.solve` of an `n x m` array: each block of `L` read once
        for all of them, each result `solve(bs[c])` bit for bit.

        Args:
            bs: The right-hand sides, of `A`'s output kind.

        Returns:
            The solutions, of `A`'s input kind, in the same order.

        Raises:
            If a cochain is on another complex than `A`.
        """
        return _typed_solve_many[
            Self.n, Self.in_k, Self.in_primal, Self.out_k, Self.out_primal
        ](self.factor, self.complex, bs, "TypedSparseLDLT")

    def inertia(self) -> Inertia:
        """The signs of `A`'s eigenvalues; see `SparseLDLT.inertia`.

        Returns:
            The counts of positive and negative pivots.
        """
        return self.factor.inertia()

    def factor_nonzeros(self) -> Int:
        """The entries of `L`, its diagonal included.

        Returns:
            `nnz(L)`.
        """
        return self.factor.factor_nonzeros()


struct TypedSparseCholesky[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](Movable):
    """`SparseCholesky` of a typed operator, such as a stiffness plus mass
    or a Hodge Laplacian; see `TypedSparseLDLT`.

    ```mojo
    var K = tc.d[0, PRIMAL]().T() @ whitney_innerproduct[1](tc) @ tc.d[0, PRIMAL]()
    var F = TypedSparseCholesky(K + whitney_innerproduct[0](tc))
    var u = F.solve(f)                         # f: dual n, u: primal 0
    ```

    Parameters:
        n: The complex dimension.
        in_k: The degree of the solution.
        in_primal: Whether the solution is primal.
        out_k: The degree of the right-hand side, `n - in_k`.
        out_primal: Whether the right-hand side is primal, `not in_primal`.
    """

    var factor: SparseCholesky
    """The untyped factorization."""
    var complex: Optional[SimplicialComplex]
    """The operator's complex (`None` if it was built from a bare
    matrix)."""

    def __init__(
        out self,
        A: MatrixOp[
            Self.n, Self.in_k, Self.in_primal, Self.out_k, Self.out_primal
        ],
        ordering: FillOrdering = FillOrdering.AMD,
        *,
        perm: Optional[List[Int]] = None,
        layout: FactorLayout = FactorLayout.AUTO,
    ) raises:
        """Factor `A`.

        Args:
            A: A symmetric positive definite operator from a kind to its
                dual kind.
            ordering: The fill-reducing permutation.
            perm: A permutation to use instead of `ordering`.
            layout: Simplicial or supernodal `L`, or `AUTO` to choose.

        Raises:
            As `SparseCholesky`.
        """
        comptime assert (
            Self.out_k == Self.n - Self.in_k
            and Self.out_primal == (not Self.in_primal)
        ), "TypedSparseCholesky: A must map a kind to its dual kind"
        self.factor = SparseCholesky(A.mat, ordering, perm=perm, layout=layout)
        self.complex = A.complex

    def solve(
        self, b: TypedCochain[Self.n, Self.out_k, Self.out_primal]
    ) raises -> TypedCochain[Self.n, Self.in_k, Self.in_primal]:
        """Solve `A x = b`.

        Args:
            b: The right-hand side, of `A`'s output kind.

        Returns:
            `x`, of `A`'s input kind.

        Raises:
            If `b` is on another complex than `A`.
        """
        _same_complex(self.complex, b.complex, "TypedSparseCholesky")
        return TypedCochain[Self.n, Self.in_k, Self.in_primal](
            b.complex, self.factor.solve(b.v)
        )

    def solve(
        self, bs: List[TypedCochain[Self.n, Self.out_k, Self.out_primal]]
    ) raises -> List[TypedCochain[Self.n, Self.in_k, Self.in_primal]]:
        """Solve `A x = b` for every cochain of `bs` at once, as
        `SparseLDLT.solve` of an `n x m` array: each block of `L` read once
        for all of them, each result `solve(bs[c])` bit for bit.

        Args:
            bs: The right-hand sides, of `A`'s output kind.

        Returns:
            The solutions, of `A`'s input kind, in the same order.

        Raises:
            If a cochain is on another complex than `A`.
        """
        return _typed_solve_many[
            Self.n, Self.in_k, Self.in_primal, Self.out_k, Self.out_primal
        ](self.factor.factor, self.complex, bs, "TypedSparseCholesky")


struct BlockSparseLDLT[
    n: Int,
    in1_k: Int,
    in1_p: Bool,
    in2_k: Int,
    in2_p: Bool,
    out1_k: Int,
    out1_p: Bool,
    out2_k: Int,
    out2_p: Bool,
](Movable):
    """`SparseLDLT` of a 2 x 2 block operator, such as the saddle point of a
    mixed formulation, factored once for many solves.

    ```mojo
    var A = BlockOp(W1 * -1.0, d1.T(), d1, tc.zero[0, DUAL, 2, PRIMAL]())
    var F = BlockSparseLDLT(A)       # the cells of the zero block last
    var x = F.solve(b)               # CochainPair of flux and pressure
    ```

    The assembled matrix must be symmetric, and each equation of the dual
    kind of its unknown, as `spsolve`'s fixed-cells overload requires.
    With a zero second block, AMD orders its cells last and the saddle
    point factors without pivoting (see `SparseLDLT`).

    Parameters:
        n: The complex dimension.
        in1_k: The degree of the first unknown.
        in1_p: Whether the first unknown is primal.
        in2_k: The degree of the second unknown.
        in2_p: Whether the second unknown is primal.
        out1_k: The degree of the first equation, `n - in1_k`.
        out1_p: Whether the first equation is primal, `not in1_p`.
        out2_k: The degree of the second equation, `n - in2_k`.
        out2_p: Whether the second equation is primal, `not in2_p`.
    """

    var factor: SparseLDLT
    """The factorization of the assembled matrix, unknowns stacked."""
    var complex: Optional[SimplicialComplex]
    """The operator's complex."""
    var _n1: Int

    def __init__(
        out self,
        A: BlockOp[
            Self.n,
            Self.in1_k,
            Self.in1_p,
            Self.in2_k,
            Self.in2_p,
            Self.out1_k,
            Self.out1_p,
            Self.out2_k,
            Self.out2_p,
        ],
        *,
        ordering: FillOrdering = FillOrdering.AMD,
        signs: Optional[List[Int]] = None,
        regularization: Float64 = 0,
        refinement_steps: Int = 1,
        layout: FactorLayout = FactorLayout.AUTO,
    ) raises:
        """Assemble and factor `A`; the options are `SparseLDLT`'s.

        Args:
            A: A symmetric block operator.
            ordering: The fill-reducing permutation.
            signs: The expected sign of each pivot, by stacked unknown.
            regularization: The `delta` of dynamic regularization.
            refinement_steps: Iterative refinement steps per solve.
            layout: Simplicial or supernodal `L`, or `AUTO` to choose.

        Raises:
            As `SparseLDLT`.
        """
        comptime assert (
            Self.out1_k == Self.n - Self.in1_k
            and Self.out1_p == (not Self.in1_p)
            and Self.out2_k == Self.n - Self.in2_k
            and Self.out2_p == (not Self.in2_p)
        ), (
            "BlockSparseLDLT: each equation must be of the dual kind of its"
            " unknown"
        )
        self.factor = SparseLDLT(
            A.to_csr(),
            ordering=ordering,
            signs=signs,
            regularization=regularization,
            refinement_steps=refinement_steps,
            layout=layout,
        )
        self.complex = A.complex
        self._n1 = A.a11.mat.ncols

    def solve(
        self,
        b: CochainPair[
            Self.n, Self.out1_k, Self.out1_p, Self.out2_k, Self.out2_p
        ],
    ) raises -> CochainPair[
        Self.n, Self.in1_k, Self.in1_p, Self.in2_k, Self.in2_p
    ]:
        """Solve `A x = b`.

        Args:
            b: The right-hand side, of `A`'s output kinds.

        Returns:
            The unknowns, of `A`'s input kinds.

        Raises:
            If `b` is on another complex than `A`.
        """
        var complex = b.first.complex
        _same_complex(self.complex, complex, "BlockSparseLDLT")
        var x = self.factor.solve(b.stacked())
        var n1 = self._n1
        var first = real_vector(n1)
        var second = real_vector(x.size - n1)
        for i in range(n1):
            first.unsafe_set(i, x.unsafe_get(i))
        for i in range(x.size - n1):
            second.unsafe_set(i, x.unsafe_get(n1 + i))
        return CochainPair(
            TypedCochain[Self.n, Self.in1_k, Self.in1_p](complex, first^),
            TypedCochain[Self.n, Self.in2_k, Self.in2_p](complex, second^),
        )

    def solve(
        self,
        bs: List[
            CochainPair[
                Self.n, Self.out1_k, Self.out1_p, Self.out2_k, Self.out2_p
            ]
        ],
    ) raises -> List[
        CochainPair[Self.n, Self.in1_k, Self.in1_p, Self.in2_k, Self.in2_p]
    ]:
        """Solve `A x = b` for every pair of `bs` at once, as
        `SparseLDLT.solve` of an `n x m` array: each block of `L` read once
        for all of them, each result `solve(bs[c])` bit for bit.

        Args:
            bs: The right-hand sides, of `A`'s output kinds.

        Returns:
            The unknowns, of `A`'s input kinds, in the same order.

        Raises:
            If a pair is on another complex than `A`.
        """
        var vs = List[RealArray](capacity=len(bs))
        for c in range(len(bs)):
            _same_complex(self.complex, bs[c].first.complex, "BlockSparseLDLT")
            vs.append(bs[c].stacked())
        var out = List[
            CochainPair[Self.n, Self.in1_k, Self.in1_p, Self.in2_k, Self.in2_p]
        ](capacity=len(bs))
        if len(bs) == 0:
            return out^
        var N = self.factor.n
        var X = self.factor.solve(_columns_of(vs, N, "BlockSparseLDLT"))
        var n1 = self._n1
        for c in range(len(bs)):
            var complex = bs[c].first.complex
            out.append(
                CochainPair(
                    TypedCochain[Self.n, Self.in1_k, Self.in1_p](
                        complex, _column(X, c, 0, n1)
                    ),
                    TypedCochain[Self.n, Self.in2_k, Self.in2_p](
                        complex, _column(X, c, n1, N)
                    ),
                )
            )
        return out^

    def inertia(self) -> Inertia:
        """The signs of `A`'s eigenvalues: for a saddle point
        `[[-M, B^T], [B, 0]]`, as many negative as first unknowns.

        Returns:
            The counts of positive and negative pivots.
        """
        return self.factor.inertia()
