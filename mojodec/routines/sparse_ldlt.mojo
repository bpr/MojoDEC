# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
# The symbolic analysis (elimination tree, column counts, `ereach`) and the
# up-looking numeric factorization with dynamic regularization are ported
# from faer's `sparse::linalg::cholesky::simplicial` (faer 0.24, commit
# 0ecd846, MIT License, Copyright (c) 2026 sarah quiñones el kazdadi); see
# NOTICE. The algorithms are Liu's elimination tree (SIAM J.
# Matrix Anal. Appl. 11(1), 1990) and the row-by-row ("up-looking") sparse
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

The factorization is simplicial: one column of `L` at a time, which suits
2D meshes and moderate 3D ones. Each row `k` of `L` is the solution of a
sparse triangular system whose pattern, the "reach" of row `k` of `A` in
the elimination tree, is found first, so the work is proportional to the
flops, not to `n^2`.
"""

from std.math import isfinite, sqrt

from mojodec.core.operator import LinearOperator, Preconditioner
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import RealArray, real_vector
from mojodec.routines.ordering import amd_ordering


comptime _NONE = -1


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
    """Steps of iterative refinement each solve takes when a pivot was
    regularized."""
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
        signs: Optional[List[Int]] = None,
        regularization: Float64 = 0,
        refinement_steps: Int = 2,
    ) raises:
        """Factor `A`.

        Args:
            A: An `n x n` symmetric matrix, both triangles stored.
            ordering: The fill-reducing permutation.
            signs: The expected sign (`+1` or `-1`) of each pivot, by row
                of `A`, for the regularization; e.g. `-1` on a mass block
                and `+1` on a constraint block, or the reverse. Without it
                the regularization keeps each small pivot's sign.
            regularization: The `delta` of dynamic regularization: a pivot
                `d` with `|d| <= delta`, or of the wrong sign, becomes
                `+-delta`. 0 (the default) turns it off. A useful value is
                about `sqrt(eps)` times the size of `A`'s entries.
            refinement_steps: Iterative refinement steps per solve, taken
                only when a pivot was regularized.

        Raises:
            If `A` is not square or not symmetric (an entry differs from
            its transpose by more than `1e-12` times the largest), a pivot
            is zero or not finite (with the row of `A` it belongs to), the
            lengths of `signs` or its entries are wrong, or
            `regularization` or `refinement_steps` is negative.
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
        if ordering == FillOrdering.AMD:
            self.perm = amd_ordering(A)
        else:
            self.perm = List[Int](capacity=n)
            for i in range(n):
                self.perm.append(i)
        self._perm_inv = List[Int](length=n, fill=0)
        for k in range(n):
            self._perm_inv[self.perm[k]] = k
        self._A = A.copy()
        # C = P A P^T, lower triangle, by rows (= upper, by columns)
        var Cp = List[Int](length=n + 1, fill=0)
        for i in range(n):
            var ki = self._perm_inv[i]
            for t in range(A.indptr[i], A.indptr[i + 1]):
                if self._perm_inv[A.indices[t]] <= ki:
                    Cp[ki + 1] += 1
        for k in range(n):
            Cp[k + 1] += Cp[k]
        var Ci = List[Int](length=Cp[n], fill=0)
        var Cx = List[Float64](length=Cp[n], fill=0)
        var at = Cp.copy()
        for i in range(n):
            var ki = self._perm_inv[i]
            for t in range(A.indptr[i], A.indptr[i + 1]):
                var kj = self._perm_inv[A.indices[t]]
                if kj <= ki:
                    Ci[at[ki]] = kj
                    Cx[at[ki]] = A.data[t]
                    at[ki] += 1
        # elimination tree and column counts of L
        var etree = List[Int](length=n, fill=_NONE)
        var counts = List[Int](length=n, fill=1)
        var visited = List[Int](length=n, fill=_NONE)
        for k in range(n):
            visited[k] = k
            for t in range(Cp[k], Cp[k + 1]):
                var i = Ci[t]
                if i < k:
                    while visited[i] != k:
                        var nxt: Int
                        if etree[i] != _NONE:
                            nxt = etree[i]
                        else:
                            etree[i] = k
                            nxt = k
                        counts[i] += 1
                        visited[i] = k
                        i = nxt
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
        return self._Lp[self.n]

    def inertia(self) -> Inertia:
        """The numbers of positive and negative eigenvalues of `A`, or of the
        regularized matrix when `num_regularized > 0`.

        Returns:
            The signs of `D`.
        """
        var pos = 0
        for j in range(self.n):
            if self._Lx[self._Lp[j]] > 0:
                pos += 1
        return Inertia(pos, self.n - pos)

    def _solve_factored(self, b: RealArray, mut out: RealArray):
        var n = self.n
        var y = List[Float64](length=n, fill=0)
        for k in range(n):
            y[k] = b.unsafe_get(self.perm[k])
        for j in range(n):  # L y = P b
            var yj = y[j]
            if yj != 0:
                for p in range(self._Lp[j] + 1, self._Lp[j + 1]):
                    y[self._Li[p]] -= self._Lx[p] * yj
        for j in range(n):  # D
            y[j] = y[j] / self._Lx[self._Lp[j]]
        for j in range(n - 1, -1, -1):  # L^T
            var s = y[j]
            for p in range(self._Lp[j] + 1, self._Lp[j + 1]):
                s -= self._Lx[p] * y[self._Li[p]]
            y[j] = s
        for k in range(n):
            out.unsafe_set(self.perm[k], y[k])

    def solve(self, b: RealArray) raises -> RealArray:
        """Solve `A x = b`.

        With regularized pivots, the factor is that of a nearby matrix;
        `refinement_steps` steps of `x += F^-1 (b - A x)` then bring `x`
        back toward the solution of `A x = b`.

        Args:
            b: The length-`n` right-hand side.

        Returns:
            `x`.

        Raises:
            If `b` does not have length `n`.
        """
        if b.size != self.n:
            raise Error(
                String(
                    "SparseLDLT.solve: b has length ", b.size, ", not ", self.n
                )
            )
        var x = real_vector(self.n)
        self.apply_into(b, x)
        return x^

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
        if self.num_regularized == 0:
            return
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
        out self, A: CSRMatrix, ordering: FillOrdering = FillOrdering.AMD
    ) raises:
        """Factor `A`.

        Args:
            A: An `n x n` symmetric positive definite matrix.
            ordering: The fill-reducing permutation.

        Raises:
            If `A` is not square or not symmetric, or a pivot is not
            positive (`A` is not positive definite), naming the row.
        """
        self.factor = SparseLDLT(A, ordering=ordering)
        ref F = self.factor
        for k in range(F.n):
            var d = F._Lx[F._Lp[k]]
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
        """Solve `A x = b`.

        Args:
            b: The length-`n` right-hand side.

        Returns:
            `x`.

        Raises:
            If `b` does not have length `n`.
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
