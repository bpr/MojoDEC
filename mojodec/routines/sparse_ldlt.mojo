# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
# The symbolic analysis (elimination tree, column counts, `ereach`) and the
# up-looking numeric factorization with dynamic regularization are ported
# from faer's `sparse::linalg::cholesky::simplicial` (faer 0.24, commit
# 0ecd846, MIT License, Copyright (c) 2026 sarah quiñones el kazdadi); see
# NOTICE. The algorithms are Liu's elimination tree (SIAM J.
# Matrix Anal. Appl. 11(1), 1990), Gilbert, Ng, and Peyton's column counts
# ("An efficient algorithm to compute row and column counts for sparse
# Cholesky factorization", SIAM J. Matrix Anal. Appl. 15(4), 1994), and the row-by-row ("up-looking") sparse
# Cholesky of Davis, *Direct Methods for Sparse Linear Systems*, ch. 4.
# ===----------------------------------------------------------------------=== #
"""
Sparse LDL^T and Cholesky factorizations (`mojodec.routines.sparse_ldlt`).
=========================================================================
`SparseLDLT(A)` factors a sparse symmetric matrix as `P A P^T = L D L^T`,
with `P` a fill-reducing permutation (`amd_ordering` by default), `L` unit
lower triangular and `D` diagonal, and then solves `A x = b` by two sparse
triangular solves. `D` may have entries of both signs, so symmetric
indefinite systems (the saddle points of mixed formulations) factor
without pivoting when the ordering puts their constraint block last, which
AMD does. `SparseCholesky(A)` is the same factorization for symmetric
positive definite `A`, refusing any other; it is a `Preconditioner`, an
exact inverse.

Two layouts of `L` (`FactorLayout`). Simplicial: one column at a time,
each row `k` of `L` the solution of a sparse triangular system whose
pattern, the "reach" of row `k` of `A` in the elimination tree, is found
first, so the work is proportional to the flops; it suits 2D meshes.
Supernodal: columns with the same pattern below a dense triangle stored
as one dense block and factored by dense kernels; it suits 3D meshes,
where the factor fills in. `FactorLayout.AUTO` picks by the flops per
entry of `L`, from the symbolic analysis.
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


comptime _NONE = -1
comptime _PARALLEL_SOLVE_WORK = 4.0e6
"""Solves of fewer stored entries times right-hand sides run serially."""
comptime _SUPERNODAL_RATIO = 20.0
"""`FactorLayout.AUTO` goes supernodal above this many flops per entry of
`L` (faer's rule, with faer's 40 lowered to where MojoDEC's kernels
measured even: simplicial was 1.3x faster at 10, the two tied at 12-16,
supernodal was 1.5x faster at 21 and 2x at 35)."""


struct FactorLayout(Equatable, ImplicitlyCopyable, Writable):
    """How `SparseLDLT` stores and computes `L`."""

    var _code: UInt8

    comptime AUTO = FactorLayout(0)
    """Supernodal when the symbolic analysis counts more than 20 flops
    per entry of `L` (3D meshes, 2D ones past a few thousand vertices),
    simplicial otherwise."""
    comptime SIMPLICIAL = FactorLayout(1)
    """One sparse column at a time (up-looking)."""
    comptime SUPERNODAL = FactorLayout(2)
    """Dense blocks of columns sharing a pattern (left-looking)."""

    def __init__(out self, code: UInt8):
        """A layout from its code; use the named members instead.

        Args:
            code: 0, 1, or 2.
        """
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        """Whether two layouts are the same.

        Args:
            other: The other layout.

        Returns:
            True if they are equal.
        """
        return self._code == other._code

    def write_to(self, mut writer: Some[Writer]):
        """Write the layout's name.

        Args:
            writer: The destination.
        """
        if self._code == 0:
            writer.write("auto")
        elif self._code == 1:
            writer.write("simplicial")
        else:
            writer.write("supernodal")


struct FillOrdering(Equatable, ImplicitlyCopyable, Writable):
    """How `SparseLDLT` permutes the matrix before factoring it."""

    var _code: UInt8

    comptime AMD = FillOrdering(0)
    """Approximate minimum degree (`amd_ordering`), zero diagonals last."""
    comptime NATURAL = FillOrdering(1)
    """No permutation: the matrix's own order."""

    def __init__(out self, code: UInt8):
        """An ordering from its code; use the named members instead.

        Args:
            code: 0 or 1.
        """
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        """Whether two orderings are the same.

        Args:
            other: The other ordering.

        Returns:
            True if they are equal.
        """
        return self._code == other._code

    def write_to(self, mut writer: Some[Writer]):
        """Write the ordering's name.

        Args:
            writer: The destination.
        """
        writer.write("AMD" if self._code == 0 else "natural")


struct Inertia(ImplicitlyCopyable, Movable, Writable):
    """The numbers of positive and negative eigenvalues of a symmetric
    matrix, read off the signs of `D` in `P A P^T = L D L^T` (Sylvester's
    law of inertia)."""

    var positive: Int
    """Positive eigenvalues."""
    var negative: Int
    """Negative eigenvalues."""

    def __init__(out self, positive: Int, negative: Int):
        """An inertia from its counts.

        Args:
            positive: Positive eigenvalues.
            negative: Negative eigenvalues.
        """
        self.positive = positive
        self.negative = negative

    def write_to(self, mut writer: Some[Writer]):
        """Write `(positive, negative)`.

        Args:
            writer: The destination.
        """
        writer.write(
            "(", self.positive, " positive, ", self.negative, " negative)"
        )


struct SparseLDLT(LinearOperator, Movable):
    """`P A P^T = L D L^T` for a sparse symmetric `A`, and solves with it.

    ```mojo
    var F = SparseLDLT(K)               # AMD ordering, no pivoting
    var x = F.solve(b)                  # K x = b
    print(F.inertia())                  # the signs of K's eigenvalues
    ```

    No pivoting is done: the ordering is fixed before the numbers are
    seen, so a pivot can be zero or small even when `A` is not singular.
    That cannot happen for positive definite `A`, nor for a saddle point
    `[[M, B^T], [B, 0]]` with `M` definite and `B` of full rank when the
    zero block is ordered last (the AMD ordering does so). For other
    indefinite matrices, pass `signs` and a `regularization`: a pivot of
    the wrong sign, or smaller than the regularization, is replaced by
    `+-regularization` (dynamic regularization, as in interior-point
    solvers), and each solve then corrects the perturbation by iterative
    refinement against `A`.

    Without pivoting, an indefinite factorization is not backward stable:
    the Schur complement's entries can grow. So every solve takes one step
    of iterative refinement by default, `x += F^-1 (b - A x)`, which costs
    a product with `A` and a second pair of triangular solves and restores
    a backward error of a few ulps (Skeel, 1980). On DarcyFlow's saddle
    point it takes the pressure error from 2.2e-14 to 4.4e-16.

    The factor is a `LinearOperator` applying `A^-1`, so it can serve as an
    exact inner solve (e.g. for shift-invert); for a preconditioner of a
    positive definite matrix, use `SparseCholesky`.
    """

    var n: Int
    """The dimension."""
    var perm: List[Int]
    """The ordering: row `perm[k]` of `A` is row `k` of `P A P^T`."""
    var num_regularized: Int
    """Pivots replaced by the regularization (0 without one)."""
    var refinement_steps: Int
    """Steps of iterative refinement each solve takes."""
    var layout: FactorLayout
    """The layout used: `SIMPLICIAL` or `SUPERNODAL`, never `AUTO`."""
    var _nnz: Int
    var _super: Optional[_Supernodal]
    var _perm_inv: List[Int]
    var _Lp: List[Int]
    var _Li: List[Int]
    var _Lx: List[Float64]
    """Column `j` of `L`: `_Lx[_Lp[j]]` holds `D[j]`, then the entries
    below the diagonal, rows in `_Li`, increasing."""
    var _A: CSRMatrix

    def __init__(
        out self,
        A: CSRMatrix,
        *,
        ordering: FillOrdering = FillOrdering.AMD,
        perm: Optional[List[Int]] = None,
        signs: Optional[List[Int]] = None,
        regularization: Float64 = 0,
        refinement_steps: Int = 1,
        layout: FactorLayout = FactorLayout.AUTO,
    ) raises:
        """Factor `A`.

        Args:
            A: An `n x n` symmetric matrix, both triangles stored.
            ordering: The fill-reducing permutation.
            perm: A permutation to use instead of `ordering` (row
                `perm[k]` of `A` becomes row `k`), such as one
                `amd_ordering(A)` already computed. The supernodal layout
                relabels it by a postorder of the elimination tree, which
                keeps the fill; `self.perm` is the ordering used.
            signs: The expected sign (`+1` or `-1`) of each pivot, by row
                of `A`, for the regularization; e.g. `-1` on a mass block
                and `+1` on a constraint block, or the reverse. Without it
                the regularization keeps each small pivot's sign.
            regularization: The `delta` of dynamic regularization: a pivot
                `d` with `|d| <= delta`, or of the wrong sign, becomes
                `+-delta`. 0 (the default) turns it off. A useful value is
                about `sqrt(eps)` times the size of `A`'s entries.
            refinement_steps: Iterative refinement steps per solve; 0
                for the bare triangular solves. With a regularization,
                2 or 3.
            layout: Simplicial or supernodal `L`, or `AUTO` to choose.

        Raises:
            If `A` is not square or not symmetric (an entry differs from
            its transpose by more than `1e-12` times the largest), a pivot
            is zero or not finite (with the row of `A` it belongs to), the
            lengths of `signs` or its entries are wrong, `perm` is not a
            permutation of `0..n-1`, or `regularization` or
            `refinement_steps` is negative.
        """
        var n = A.nrows
        if A.ncols != n:
            raise Error(String("SparseLDLT: the matrix is ", n, " x ", A.ncols))
        if not (regularization >= 0) or not isfinite(regularization):
            raise Error("SparseLDLT: regularization must be >= 0")
        if refinement_steps < 0:
            raise Error("SparseLDLT: refinement_steps must be >= 0")
        if signs:
            ref sg = signs.value()
            if len(sg) != n:
                raise Error(
                    String("SparseLDLT: ", len(sg), " signs for ", n, " rows")
                )
            for i in range(n):
                if sg[i] != 1 and sg[i] != -1:
                    raise Error(
                        String(
                            "SparseLDLT: sign ", i, " is ", sg[i], ", not +-1"
                        )
                    )
        _require_symmetric_entries(A)
        self.n = n
        self.num_regularized = 0
        self.refinement_steps = refinement_steps
        if perm:
            self.perm = perm.value().copy()
        elif ordering == FillOrdering.AMD:
            self.perm = amd_ordering(A)
        else:
            self.perm = List[Int](capacity=n)
            for i in range(n):
                self.perm.append(i)
        self._perm_inv = _inverse_permutation(self.perm, n, "SparseLDLT")
        self._A = A.copy()
        var C = _PermutedLower(A, self._perm_inv)
        var etree = List[Int](length=n, fill=_NONE)
        var counts = List[Int](length=n, fill=1)
        var visited = List[Int](length=n, fill=_NONE)
        _etree_and_counts(C.Cp, C.Ci, etree, counts, visited)
        self._nnz = 0
        var flops: Float64 = 0
        for j in range(n):
            self._nnz += counts[j]
            var c = Float64(counts[j])
            flops += c + c * (c + 1) / 2
        self.layout = layout
        if layout == FactorLayout.AUTO:
            if flops > _SUPERNODAL_RATIO * Float64(max(self._nnz, 1)):
                self.layout = FactorLayout.SUPERNODAL
            else:
                self.layout = FactorLayout.SIMPLICIAL
        self._super = None
        if self.layout == FactorLayout.SUPERNODAL:
            self._Lp = List[Int]()
            self._Li = List[Int]()
            self._Lx = List[Float64]()
            var post = etree_postorder(etree)
            var identity = True
            for k in range(n):
                if post[k] != k:
                    identity = False
                    break
            if not identity:
                var p2 = List[Int](capacity=n)
                for k in range(n):
                    p2.append(self.perm[post[k]])
                self.perm = p2^
                self._perm_inv = _inverse_permutation(
                    self.perm, n, "SparseLDLT"
                )
                C = _PermutedLower(A, self._perm_inv)
                # a postorder is an equivalent ordering: the same tree and
                # column counts, relabeled (ancestors keep larger labels)
                var label = List[Int](length=n, fill=0)
                for k in range(n):
                    label[post[k]] = k
                var etree2 = List[Int](length=n, fill=_NONE)
                var counts2 = List[Int](length=n, fill=0)
                for k in range(n):
                    var p = etree[post[k]]
                    etree2[k] = label[p] if p != _NONE else _NONE
                    counts2[k] = counts[post[k]]
                etree = etree2^
                counts = counts2^
            var S = _Supernodal(
                C.Cp,
                C.Ci,
                C.Cx,
                etree,
                counts,
                self.perm,
                signs,
                regularization,
            )
            self.num_regularized = S.num_regularized
            self._super = S^
            return
        ref Cp = C.Cp
        ref Ci = C.Ci
        ref Cx = C.Cx
        self._Lp = List[Int](length=n + 1, fill=0)
        for j in range(n):
            self._Lp[j + 1] = self._Lp[j] + counts[j]
        var nnz = self._Lp[n]
        self._Li = List[Int](length=nnz, fill=0)
        self._Lx = List[Float64](length=nnz, fill=0)
        # numeric: row k of L from the sparse triangular solve L x = C[k, :k]
        var x = List[Float64](length=n, fill=0)
        var current = List[Int](length=n, fill=0)
        for j in range(n):
            current[j] = self._Lp[j]
            self._Li[self._Lp[j]] = j
        var stack = List[Int](length=n, fill=0)
        for k in range(n):
            visited[k] = _NONE
        for k in range(n):
            var top = _ereach(Cp, Ci, etree, k, visited, stack)
            for t in range(Cp[k], Cp[k + 1]):
                x[Ci[t]] += Cx[t]
            var d = x[k]
            x[k] = 0
            for s in range(top, n):
                var j = stack[s]
                var j_start = self._Lp[j]
                current[j] += 1
                var row = current[j]
                var xj = x[j]
                x[j] = 0
                var lkj = xj * (1 / self._Lx[j_start])
                for p in range(j_start + 1, row):
                    x[self._Li[p]] -= self._Lx[p] * xj
                d = d - lkj * xj
                self._Lx[row] = lkj
                self._Li[row] = k
            if regularization > 0:
                var delta = regularization
                if signs:
                    var sg = signs.value()[self.perm[k]]
                    if sg > 0 and d <= delta:
                        d = delta
                        self.num_regularized += 1
                    elif sg < 0 and d >= -delta:
                        d = -delta
                        self.num_regularized += 1
                elif abs(d) <= delta:
                    d = -delta if d < 0 else delta
                    self.num_regularized += 1
            if d == 0 or not isfinite(d):
                raise Error(
                    String(
                        "SparseLDLT: pivot ",
                        k,
                        " (row ",
                        self.perm[k],
                        " of the matrix) is ",
                        d,
                        (
                            "; the matrix is singular, or needs another"
                            " ordering or regularization"
                        ),
                    )
                )
            self._Lx[self._Lp[k]] = d

    def num_rows(self) -> Int:
        """The number of rows.

        Returns:
            `n`.
        """
        return self.n

    def num_cols(self) -> Int:
        """The number of columns.

        Returns:
            `n`.
        """
        return self.n

    def factor_nonzeros(self) -> Int:
        """The stored entries of `L`, its unit diagonal counted (it holds
        `D`): the fill the ordering achieved.

        Returns:
            `nnz(L)`.
        """
        return self._nnz

    def num_supernodes(self) -> Int:
        """The supernodes of a supernodal factor: blocks of columns of `L`
        stored dense; 0 for a simplicial one.

        Returns:
            The number of blocks.
        """
        if self._super:
            return self._super.value().num_supernodes()
        return 0

    def stored_entries(self) -> Int:
        """The values the factor stores: `factor_nonzeros()` for a
        simplicial one; for a supernodal one, its dense blocks, the zeros
        above each block's diagonal and those that relaxed merging adds
        included.

        Returns:
            The number of stored values.
        """
        if self._super:
            return self._super.value().stored_entries()
        return len(self._Lx)

    def _pivot(self, k: Int) -> Float64:
        """`D[k]`."""
        if self._super:
            return self._super.value().pivot(k)
        return self._Lx[self._Lp[k]]

    def inertia(self) -> Inertia:
        """The numbers of positive and negative eigenvalues of `A`, or of the
        regularized matrix when `num_regularized > 0`.

        Returns:
            The signs of `D`.
        """
        var pos = 0
        for j in range(self.n):
            if self._pivot(j) > 0:
                pos += 1
        return Inertia(pos, self.n - pos)

    def _solve_factored(self, b: RealArray, mut out: RealArray):
        """`out = (P^T L D L^T P)^-1 b`: `_solve_columns` with one column."""
        var n = self.n
        var y = List[Float64](length=n, fill=0)
        for k in range(n):
            y[k] = b.unsafe_get(self.perm[k])
        self._solve_columns(
            y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 0, 1
        )
        for k in range(n):
            out.unsafe_set(self.perm[k], y[k])

    def _solve_columns(
        self, Y: Pointer[Float64, MutUntrackedOrigin], c_lo: Int, c_hi: Int
    ):
        """`Y[:, c] = (L D L^T)^-1 Y[:, c]` for columns `c_lo..c_hi` of a
        column-major `n`-row `Y`, in the permuted numbering. Each column of
        `L` (or supernode) is applied to every column of `Y` in turn, and
        each column of `Y` gets the same operations as when solved alone."""
        if self._super:
            self._super.value().solve_columns(Y, c_lo, c_hi)
            return
        var n = self.n
        for j in range(n):  # L y = P b
            var lo = self._Lp[j] + 1
            var hi = self._Lp[j + 1]
            for c in range(c_lo, c_hi):
                var y = Y.unsafe_offset(c * n)
                var yj = y[unsafe_offset=j]
                if yj != 0:
                    for p in range(lo, hi):
                        y[unsafe_offset=self._Li[p]] -= self._Lx[p] * yj
        for c in range(c_lo, c_hi):  # D
            var y = Y.unsafe_offset(c * n)
            for j in range(n):
                y[unsafe_offset=j] = y[unsafe_offset=j] / self._Lx[self._Lp[j]]
        for j in range(n - 1, -1, -1):  # L^T
            var lo = self._Lp[j] + 1
            var hi = self._Lp[j + 1]
            for c in range(c_lo, c_hi):
                var y = Y.unsafe_offset(c * n)
                var s = y[unsafe_offset=j]
                for p in range(lo, hi):
                    s -= self._Lx[p] * y[unsafe_offset=self._Li[p]]
                y[unsafe_offset=j] = s

    def _solve_all_columns(self, mut Y: List[Float64], m: Int):
        """`_solve_columns` on all `m` columns of `Y`, split into blocks of
        columns across the physical cores when the work is large enough
        (`MOJODEC_NUM_THREADS` caps them). The bits do not depend on the
        split."""
        var Yp = Y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var stored = self.stored_entries()
        if m < 2 or Float64(stored) * Float64(m) < _PARALLEL_SOLVE_WORK:
            self._solve_columns(Yp, 0, m)
            return

        def columns(chunk: Int, lo: Int, hi: Int) {self, Yp}:
            self._solve_columns(Yp, lo, hi)

        parallel_ranges(m, 2, columns, memory_bound=True)

    def solve(self, b: RealArray) raises -> RealArray:
        """Solve `A x = b`, or `A X = B` for an `n x m` array `B`.

        After the triangular solves, `refinement_steps` steps of
        `x += F^-1 (b - A x)` correct the rounding of an unpivoted
        factorization and, with regularized pivots, the perturbation.

        With several right-hand sides, each block of `L` is read once for
        all of them, and blocks of columns run on separate cores; column
        `c` of `X` is `solve(B[:, c])` bit for bit.

        Args:
            b: The length-`n` right-hand side, or an `n x m` array of `m`
                right-hand sides (C-contiguous).

        Returns:
            `x`, of `b`'s shape.

        Raises:
            If `b` does not have `n` entries (or rows), or is a non-
            contiguous 2-D array.
        """
        if b.ndim == 2:
            return self._solve_many(b)
        if b.size != self.n:
            raise Error(
                String(
                    "SparseLDLT.solve: b has length ", b.size, ", not ", self.n
                )
            )
        var x = real_vector(self.n)
        self.apply_into(b, x)
        return x^

    def _solve_many(self, B: RealArray) raises -> RealArray:
        """`solve` for an `n x m` `B`."""
        var n = self.n
        var m = cols_of(B)
        if rows_of(B) != n:
            raise Error(
                String("SparseLDLT.solve: B has ", rows_of(B), " rows, not ", n)
            )
        if not B.flags.C_CONTIGUOUS:
            raise Error("SparseLDLT.solve: B is not C-contiguous")
        var X = real_array(n, m)
        var Y = List[Float64](length=n * m, fill=0)  # by columns, permuted
        for c in range(m):
            for k in range(n):
                Y[c * n + k] = fget(B, self.perm[k], c)
        self._solve_all_columns(Y, m)
        for c in range(m):
            for k in range(n):
                fset(X, self.perm[k], c, Y[c * n + k])
        # refinement, column by column as `apply_into` does it
        var x = real_vector(n)
        var r = real_vector(n)
        for _ in range(self.refinement_steps):
            for c in range(m):
                for i in range(n):
                    x.unsafe_set(i, fget(X, i, c))
                self._A.apply_into(x, r)
                for i in range(n):
                    r.unsafe_set(i, fget(B, i, c) - r.unsafe_get(i))
                for k in range(n):
                    Y[c * n + k] = r.unsafe_get(self.perm[k])
            self._solve_all_columns(Y, m)
            for c in range(m):
                for k in range(n):
                    var i = self.perm[k]
                    fset(X, i, c, fget(X, i, c) + Y[c * n + k])
        return X^

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """`y = A^-1 x`, as `solve`.

        Args:
            x: The right-hand side, length `n`.
            y: The solution, overwritten.

        Raises:
            If `x` or `y` does not have length `n`.
        """
        if x.size != self.n or y.size != self.n:
            raise Error(
                String(
                    "SparseLDLT.solve: vectors of length ",
                    x.size,
                    " and ",
                    y.size,
                    " for n = ",
                    self.n,
                )
            )
        self._solve_factored(x, y)
        var r = real_vector(self.n)
        var dx = real_vector(self.n)
        for _ in range(self.refinement_steps):
            self._A.apply_into(y, r)
            for i in range(self.n):
                r.unsafe_set(i, x.unsafe_get(i) - r.unsafe_get(i))
            self._solve_factored(r, dx)
            for i in range(self.n):
                y.unsafe_set(i, y.unsafe_get(i) + dx.unsafe_get(i))


struct SparseCholesky(Movable, Preconditioner):
    """`P A P^T = L D L^T` for a sparse symmetric positive definite `A`, with
    every pivot checked positive: an exact `A^-1` that can serve as a
    preconditioner (`cg(K, b, M=SparseCholesky(K))` converges in one step)
    or as the block of a `BlockDiagonal`.

    ```mojo
    var F = SparseCholesky(K)
    var x = F.solve(b)
    ```
    """

    var factor: SparseLDLT
    """The factorization."""

    def __init__(
        out self,
        A: CSRMatrix,
        ordering: FillOrdering = FillOrdering.AMD,
        *,
        perm: Optional[List[Int]] = None,
        layout: FactorLayout = FactorLayout.AUTO,
    ) raises:
        """Factor `A`.

        Args:
            A: An `n x n` symmetric positive definite matrix.
            ordering: The fill-reducing permutation.
            perm: A permutation to use instead of `ordering`, as
                `SparseLDLT` takes it.
            layout: Simplicial or supernodal `L`, or `AUTO` to choose.

        Raises:
            If `A` is not square or not symmetric, or a pivot is not
            positive (`A` is not positive definite), naming the row.
        """
        # positive definite: no growth, so the bare solves are backward
        # stable, and the preconditioner stays one linear map
        self.factor = SparseLDLT(
            A, ordering=ordering, perm=perm, refinement_steps=0, layout=layout
        )
        ref F = self.factor
        for k in range(F.n):
            var d = F._pivot(k)
            if not d > 0:
                raise Error(
                    String(
                        "SparseCholesky: pivot ",
                        k,
                        " (row ",
                        F.perm[k],
                        " of the matrix) is ",
                        d,
                        "; the matrix is not positive definite",
                    )
                )

    def num_rows(self) -> Int:
        """The number of rows.

        Returns:
            `n`.
        """
        return self.factor.n

    def num_cols(self) -> Int:
        """The number of columns.

        Returns:
            `n`.
        """
        return self.factor.n

    def solve(self, b: RealArray) raises -> RealArray:
        """Solve `A x = b`, or `A X = B` for an `n x m` array `B` (each
        column as `solve` of it alone, bit for bit; see
        `SparseLDLT.solve`).

        Args:
            b: The length-`n` right-hand side, or an `n x m` array of `m`
                right-hand sides (C-contiguous).

        Returns:
            `x`, of `b`'s shape.

        Raises:
            If `b` does not have `n` entries (or rows).
        """
        return self.factor.solve(b)

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """`y = A^-1 x`.

        Args:
            x: The right-hand side, length `n`.
            y: The solution, overwritten.

        Raises:
            If `x` or `y` does not have length `n`.
        """
        self.factor.apply_into(x, y)


def cholesky_nonzeros(A: CSRMatrix, perm: List[Int]) raises -> Int:
    """The number of nonzeros `SparseLDLT(A, perm=perm)` would store in
    `L` (its diagonal included), from the elimination tree alone: no
    arithmetic, about the cost of reading `A`.

    ```mojo
    var p = amd_ordering(A)
    if cholesky_nonzeros(A, p) <= 8 * A.nnz():  # little fill: factor
        var F = SparseCholesky(A, perm=p^)
    ```

    Args:
        A: An `n x n` matrix with a symmetric pattern; only the entries on
            and below the diagonal of `P A P^T` are read.
        perm: The ordering: row `perm[k]` of `A` becomes row `k`.

    Returns:
        `nnz(L)`, as `factor_nonzeros()` would report it.

    Raises:
        If `A` is not square or `perm` is not a permutation of `0..n-1`.
    """
    var n = A.nrows
    if A.ncols != n:
        raise Error(
            String("cholesky_nonzeros: the matrix is ", n, " x ", A.ncols)
        )
    var C = _PermutedLower(
        A, _inverse_permutation(perm, n, "cholesky_nonzeros")
    )
    var etree = List[Int](length=n, fill=_NONE)
    var counts = List[Int](length=n, fill=1)
    var visited = List[Int](length=n, fill=_NONE)
    _etree_and_counts(C.Cp, C.Ci, etree, counts, visited)
    var total = 0
    for j in range(n):
        total += counts[j]
    return total


def _inverse_permutation(
    perm: List[Int], n: Int, who: String
) raises -> List[Int]:
    """`perm`'s inverse; raises unless it is a permutation of `0..n-1`."""
    if len(perm) != n:
        raise Error(String(who, ": perm has ", len(perm), " entries for ", n))
    var inv = List[Int](length=n, fill=_NONE)
    for k in range(n):
        var i = perm[k]
        if i < 0 or i >= n or inv[i] != _NONE:
            raise Error(
                String(who, ": perm is not a permutation (entry ", k, ")")
            )
        inv[i] = k
    return inv^


struct _PermutedLower(Movable):
    """`C = P A P^T`, its lower triangle by rows (= upper, by columns)."""

    var Cp: List[Int]
    var Ci: List[Int]
    var Cx: List[Float64]

    def __init__(out self, A: CSRMatrix, perm_inv: List[Int]):
        var n = A.nrows
        self.Cp = List[Int](length=n + 1, fill=0)
        for i in range(n):
            var ki = perm_inv[i]
            for t in range(A.indptr[i], A.indptr[i + 1]):
                if perm_inv[A.indices[t]] <= ki:
                    self.Cp[ki + 1] += 1
        for k in range(n):
            self.Cp[k + 1] += self.Cp[k]
        self.Ci = List[Int](length=self.Cp[n], fill=0)
        self.Cx = List[Float64](length=self.Cp[n], fill=0)
        var at = self.Cp.copy()
        for i in range(n):
            var ki = perm_inv[i]
            for t in range(A.indptr[i], A.indptr[i + 1]):
                var kj = perm_inv[A.indices[t]]
                if kj <= ki:
                    self.Ci[at[ki]] = kj
                    self.Cx[at[ki]] = A.data[t]
                    at[ki] += 1


def _etree_and_counts(
    Cp: List[Int],
    Ci: List[Int],
    mut etree: List[Int],
    mut counts: List[Int],
    mut visited: List[Int],
):
    """The elimination tree of `C` and the column counts of `L` (with the
    diagonal), in time nearly linear in `nnz(C)` rather than `nnz(L)`.

    The tree by Liu's algorithm: each row's entries climb to their roots
    through ancestors compressed to that row (`visited` holds them). The
    counts by Gilbert, Ng, and Peyton's: `L[i, j]` is nonzero exactly when
    `j` is in the row subtree of `i`, the union of the tree paths from the
    columns of row `i`'s entries up to `i`. Visiting the columns in a
    postorder, `j` gets +1 for each row subtree it is a leaf of, and the
    least common ancestor of that leaf and the row's previous leaf gets -1
    (found with a disjoint-set forest of the visited columns); each child
    gives its parent -1; summing up the tree then counts, at each `j`, the
    row subtrees containing it. The leaf and ancestor bookkeeping follows
    faer's `column_counts_aat`."""
    var n = len(etree)
    for k in range(n):
        etree[k] = _NONE
        visited[k] = _NONE
        for t in range(Cp[k], Cp[k + 1]):
            var i = Ci[t]
            while i != _NONE and i < k:
                var nxt = visited[i]
                visited[i] = k
                if nxt == _NONE:
                    etree[i] = k
                i = nxt
    var post = etree_postorder(etree)
    # the strictly lower entries by columns: rows i > j of column j
    var Tp = List[Int](length=n + 1, fill=0)
    for k in range(n):
        for t in range(Cp[k], Cp[k + 1]):
            if Ci[t] < k:
                Tp[Ci[t] + 1] += 1
    for j in range(n):
        Tp[j + 1] += Tp[j]
    var Ti = List[Int](length=Tp[n], fill=0)
    var at = Tp.copy()
    for k in range(n):
        for t in range(Cp[k], Cp[k + 1]):
            var j = Ci[t]
            if j < k:
                Ti[at[j]] = k
                at[j] += 1
    # first[j]: the postorder position of j's first descendant
    var first = List[Int](length=n, fill=_NONE)
    for k in range(n):
        var j = post[k]
        counts[j] = 1 if first[j] == _NONE else 0  # leaves of the tree
        while j != _NONE and first[j] == _NONE:
            first[j] = k
            j = etree[j]
    var max_first = List[Int](length=n, fill=_NONE)
    var prev_leaf = List[Int](length=n, fill=_NONE)
    ref ancestor = visited
    for i in range(n):
        ancestor[i] = i
    for k in range(n):
        var j = post[k]
        var p = etree[j]
        if p != _NONE:
            counts[p] -= 1
        for t in range(Tp[j], Tp[j + 1]):
            var i = Ti[t]
            # j is a leaf of i's row subtree unless a descendant of j
            # (first[j] or later in the postorder) already was
            if first[j] <= max_first[i]:
                continue
            max_first[i] = first[j]
            var jprev = prev_leaf[i]
            prev_leaf[i] = j
            counts[j] += 1
            if jprev == _NONE:
                continue  # i's first leaf: its subtree is this path so far
            var lca = jprev
            while lca != ancestor[lca]:
                lca = ancestor[lca]
            var node = jprev
            while node != lca:
                var nxt = ancestor[node]
                ancestor[node] = lca
                node = nxt
            counts[lca] -= 1
        if p != _NONE:
            ancestor[j] = p
    for j in range(n):  # children come before their parents
        var p = etree[j]
        if p != _NONE:
            counts[p] += counts[j]


def _ereach(
    Cp: List[Int],
    Ci: List[Int],
    etree: List[Int],
    k: Int,
    mut visited: List[Int],
    mut stack: List[Int],
) -> Int:
    """The pattern of row `k` of `L` (columns `j < k`), in `stack[top:]`, in
    an order where each column comes before those that depend on it (the
    paths from row `k`'s entries up the elimination tree)."""
    var n = len(etree)
    var top = n
    visited[k] = k
    for t in range(Cp[k], Cp[k + 1]):
        var i = Ci[t]
        if i >= k:
            continue
        var length = 0
        while visited[i] != k:
            stack[length] = i
            length += 1
            visited[i] = k
            i = etree[i]
        # move the path to the top of the stack, in the same order
        for q in range(length - 1, -1, -1):
            top -= 1
            stack[top] = stack[q]
    return top


def _require_symmetric_entries(A: CSRMatrix) raises:
    """Raise unless `|a_ij - a_ji| <= 1e-12 max |a|` for every stored entry
    (an entry stored on one side only is compared with zero)."""
    var big: Float64 = 0
    for t in range(len(A.data)):
        big = max(big, abs(A.data[t]))
    var T = A.transpose()
    var D = A - T
    var tol = 1e-12 * big
    for i in range(D.nrows):
        for t in range(D.indptr[i], D.indptr[i + 1]):
            var e = abs(D.data[t])
            if not (e <= tol):
                raise Error(
                    String(
                        "SparseLDLT: the matrix is not symmetric (entries (",
                        i,
                        ", ",
                        D.indices[t],
                        ") and its transpose differ by ",
                        e,
                        ")",
                    )
                )


# ===----------------------------------------------------------------------=== #
# Typed factorizations
# ===----------------------------------------------------------------------=== #


def _same_complex(
    op: Optional[SimplicialComplex], b: SimplicialComplex, who: String
) raises:
    if op and not op.value().is_same(b):
        raise Error(String(who, ".solve: b is on a different complex"))


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

    def inertia(self) -> Inertia:
        """The signs of `A`'s eigenvalues: for a saddle point
        `[[-M, B^T], [B, 0]]`, as many negative as first unknowns.

        Returns:
            The counts of positive and negative pivots.
        """
        return self.factor.inertia()
