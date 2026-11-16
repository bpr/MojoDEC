# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
# The aggregation (`_aggregates`) and the smoothed prolongation
# (`_smoothed_prolongation`) are ported from amgcl
# (https://github.com/ddemidov/amgcl, commit c68350e, MIT License,
# Copyright (c) 2012-2022 Denis Demidov): `coarsening/plain_aggregates.hpp`
# and `coarsening/smoothed_aggregation.hpp`. Its license is reproduced in
# NOTICE. The Chebyshev smoother follows PETSc's design (interval
# [0.1, 1.1] lambda_max of D^-1 A); the method is Vanek, Mandel, and
# Brezina's (Computing 56, 1996).
# ===----------------------------------------------------------------------=== #
"""
Smoothed-aggregation algebraic multigrid (`mojodec.routines.amg`).
=================================================================
`SmoothedAggregation(A)` builds a multigrid hierarchy from a sparse
symmetric positive definite matrix alone, such as a 0-form stiffness
`d0^T M1 d0` (plus a mass), and applies one V-cycle as a preconditioner.

```mojo
var M = SmoothedAggregation(K)
var res = cg(K, b, M=M, rtol=1e-10)   # iterations nearly independent of h
```

Each level groups the unknowns into aggregates of strongly coupled
neighbors, interpolates piecewise constants over them (the tentative
prolongation), smooths that interpolation with one damped Jacobi step,
and takes the Galerkin product `P^T A P` as the next level's matrix. The
coarsest level is factored (`SparseCholesky`).
"""

from std.math import isfinite, sqrt

from mojodec.core.operator import LinearOperator, Preconditioner
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import RealArray, real_vector
from mojodec.routines.linalg import eigenvalue_range, jacobi
from mojodec.routines.sparse_ldlt import SparseCholesky, SparseLDLT


struct Relaxation(Equatable, ImplicitlyCopyable, Writable):
    """The smoother of a `SmoothedAggregation` V-cycle."""

    var _code: UInt8

    comptime CHEBYSHEV = Relaxation(0)
    """Chebyshev polynomial in `D^-1 A` on `[0.1, 1.1] lambda_max` (PETSc's
    choice): no sweep order, so parallel like a matrix product, and the
    same polynomial before and after keeps the cycle symmetric."""
    comptime GAUSS_SEIDEL = Relaxation(1)
    """Forward Gauss-Seidel before the coarse correction, backward after:
    symmetric, strong, serial."""

    def __init__(out self, code: UInt8):
        """A relaxation from its code; use the named members instead.

        Args:
            code: 0 or 1.
        """
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        """Whether two relaxations are the same.

        Args:
            other: The relaxation to compare with.

        Returns:
            True if they are equal.
        """
        return self._code == other._code

    def write_to(self, mut writer: Some[Writer]):
        """Write the relaxation's name.

        Args:
            writer: The destination.
        """
        if self == Self.CHEBYSHEV:
            writer.write("Chebyshev")
        else:
            writer.write("symmetric Gauss-Seidel")


struct _Aggregates(Movable):
    """Aggregate ids (`-1`: left out, no strong neighbor) and the strong
    entries of `A`, by stored position."""

    var count: Int
    var id: List[Int]
    var strong: List[Bool]

    def __init__(
        out self, count: Int, var id: List[Int], var strong: List[Bool]
    ):
        self.count = count
        self.id = id^
        self.strong = strong^


comptime _UNDEFINED = -1
comptime _REMOVED = -2


def _aggregates(A: CSRMatrix, diag: RealArray, eps: Float64) -> _Aggregates:
    """amgcl's plain aggregation: `j` is strongly coupled to `i` when
    `a_ij^2 > eps^2 a_ii a_jj`; one greedy pass in row order makes each
    unaggregated row with strong neighbors a seed, takes its strong
    neighbors (from other aggregates too), and tentatively the strong
    neighbors of those, which later seeds may claim."""
    var n = A.nrows
    var eps2 = eps * eps
    var strong = List[Bool](length=len(A.data), fill=False)
    for i in range(n):
        var eps_dia_i = eps2 * diag.unsafe_get(i)
        for t in range(A.indptr[i], A.indptr[i + 1]):
            var c = A.indices[t]
            var v = A.data[t]
            strong[t] = c != i and eps_dia_i * diag.unsafe_get(c) < v * v
    var id = List[Int](length=n, fill=_REMOVED)
    for i in range(n):
        for t in range(A.indptr[i], A.indptr[i + 1]):
            if strong[t]:
                id[i] = _UNDEFINED
                break
    var count = 0
    var neib = List[Int]()
    for i in range(n):
        if id[i] != _UNDEFINED:
            continue
        var cur = count
        count += 1
        id[i] = cur
        neib.clear()
        for t in range(A.indptr[i], A.indptr[i + 1]):
            var c = A.indices[t]
            if strong[t] and id[c] != _REMOVED:
                id[c] = cur
                neib.append(c)
        for c in neib:
            for t in range(A.indptr[c], A.indptr[c + 1]):
                var cc = A.indices[t]
                if strong[t] and id[cc] == _UNDEFINED:
                    id[cc] = cur
    # aggregates emptied by later claims are dropped, the rest renumbered
    var used = List[Int](length=count, fill=0)
    for i in range(n):
        if id[i] >= 0:
            used[id[i]] = 1
    var total = 0
    for a in range(count):
        total += used[a]
        used[a] = total
    if total < count:
        for i in range(n):
            if id[i] >= 0:
                id[i] = used[id[i]] - 1
    for i in range(n):
        if id[i] < 0:
            id[i] = -1
    return _Aggregates(total, id^, strong^)


def _smoothed_prolongation(
    A: CSRMatrix, agg: _Aggregates, omega: Float64
) -> CSRMatrix:
    """`P = (I - omega D_F^-1 A_F) P0`: `P0` the 0/1 tentative
    prolongation, `A_F` the strong part of `A` with its weak entries lumped
    into the diagonal (amgcl's), so `P` keeps the strong pattern."""
    var n = A.nrows
    var nc = agg.count
    var indptr = List[Int](length=n + 1, fill=0)
    var indices = List[Int]()
    var data = List[Float64]()
    var marker = List[Int](length=nc, fill=-1)
    for i in range(n):
        var dia: Float64 = 0
        for t in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[t] == i or not agg.strong[t]:
                dia += A.data[t]
        if dia != 0:
            dia = -omega * (1 / dia)  # amgcl's rounding
        var row_beg = len(indices)
        for t in range(A.indptr[i], A.indptr[i + 1]):
            var c = A.indices[t]
            if c != i and not agg.strong[t]:
                continue
            var a = agg.id[c]  # P0 has at most one entry per row, 1
            if a < 0:
                continue
            var va = (1 - omega) if c == i else dia * A.data[t]
            if marker[a] < row_beg:
                marker[a] = len(indices)
                indices.append(a)
                data.append(va)
            else:
                data[marker[a]] += va
        # sort the row's columns (they arrive in A's column order)
        var row_end = len(indices)
        for p in range(row_beg + 1, row_end):
            var ci = indices[p]
            var vi = data[p]
            var q = p
            while q > row_beg and indices[q - 1] > ci:
                indices[q] = indices[q - 1]
                data[q] = data[q - 1]
                q -= 1
            indices[q] = ci
            data[q] = vi
        indptr[i + 1] = row_end
    var P = CSRMatrix(n, nc)
    P.indptr = indptr^
    P.indices = indices^
    P.data = data^
    return P^


def _lambda_max(A: CSRMatrix, diag: RealArray) raises -> Float64:
    """`lambda_max(D^-1 A)`: 10 Jacobi-CG steps' largest Ritz value, or
    Gershgorin's bound when CG breaks down (a semidefinite `A`)."""
    try:
        return eigenvalue_range(A, M=jacobi(A)).highest
    except:
        var best: Float64 = 0
        for i in range(A.nrows):
            var s: Float64 = 0
            for t in range(A.indptr[i], A.indptr[i + 1]):
                s += abs(A.data[t])
            best = max(best, s / diag.unsafe_get(i))
        return best


struct _Level(Movable):
    var A: CSRMatrix
    var P: CSRMatrix
    var R: CSRMatrix
    var dinv: RealArray
    var lam: Float64
    """The estimate of `lambda_max(D^-1 A)`."""

    def __init__(
        out self,
        var A: CSRMatrix,
        var P: CSRMatrix,
        var R: CSRMatrix,
        var dinv: RealArray,
        lam: Float64,
    ):
        self.A = A^
        self.P = P^
        self.R = R^
        self.dinv = dinv^
        self.lam = lam


struct SmoothedAggregation(Movable, Preconditioner):
    """One V-cycle of smoothed-aggregation algebraic multigrid: a
    `Preconditioner` for a sparse symmetric positive definite matrix.

    ```mojo
    var amg = SmoothedAggregation(K)            # K = d0^T M1 d0 + M0, say
    print(amg.level_sizes(), amg.operator_complexity())
    var res = cg(K, b, M=amg, rtol=1e-10)
    ```

    The hierarchy follows amgcl's smoothed aggregation: strength
    threshold `strength` (halved on each coarser level), greedy
    aggregation in row order (so a spatially coherent numbering, such as
    `reorder_vertices`'s, gives compact aggregates), piecewise-constant
    tentative prolongation smoothed by one Jacobi step with
    `omega = 4/3 / lambda_max(D^-1 A)`, `R = P^T`, and `A_c = P^T A P`.
    Levels are added until one has at most `coarse_size` unknowns, which
    is factored. The smoother is applied `sweeps` times before and after
    the coarse correction, in mirror order, so the cycle is symmetric.

    The cycle preserves constants when `A`'s rows sum to zero (a Neumann
    Laplacian): for a singular `A`, give `cg` the kernel
    (`nullspace=NullSpace.constants(n)`), and the coarse factor is
    regularized.
    """

    var levels: List[_Level]
    """The levels above the coarsest, finest first."""
    var coarse: CSRMatrix
    """The coarsest level's matrix, factored."""
    var relaxation: Relaxation
    """The smoother."""
    var degree: Int
    """The Chebyshev polynomial's degree (products with `A` per sweep)."""
    var sweeps: Int
    """Smoother applications before and after each coarse correction."""
    var _cholesky: Optional[SparseCholesky]
    var _ldlt: Optional[SparseLDLT]

    def __init__(
        out self,
        A: CSRMatrix,
        *,
        strength: Float64 = 0.08,
        coarse_size: Int = 1000,
        max_levels: Int = 25,
        relaxation: Relaxation = Relaxation.CHEBYSHEV,
        degree: Int = 2,
        sweeps: Int = 1,
    ) raises:
        """Build the hierarchy.

        Args:
            A: An `n x n` symmetric positive (semi)definite matrix with a
                positive diagonal.
            strength: The strength threshold, amgcl's `eps_strong`: `j` is
                strong for `i` when
                `a_ij^2 > strength^2 a_ii a_jj`.
            coarse_size: Stop coarsening at this many unknowns or fewer.
            max_levels: The most levels, the coarsest included.
            relaxation: `Relaxation.CHEBYSHEV` or `.GAUSS_SEIDEL`.
            degree: The Chebyshev degree.
            sweeps: Smoother applications before and after each coarse
                correction.

        Raises:
            If `A` is not square, a diagonal entry is not positive, the
            parameters are out of range, or the coarsest matrix cannot be
            factored even regularized.
        """
        var n = A.nrows
        if A.ncols != n:
            raise Error(
                String("SmoothedAggregation: the matrix is ", n, " x ", A.ncols)
            )
        if not (strength > 0 and strength < 1):
            raise Error("SmoothedAggregation: strength must be in (0, 1)")
        if coarse_size < 1 or max_levels < 1 or degree < 1 or sweeps < 1:
            raise Error(
                "SmoothedAggregation: coarse_size, max_levels, degree, and"
                " sweeps must be positive"
            )
        self.levels = List[_Level]()
        self.relaxation = relaxation
        self.degree = degree
        self.sweeps = sweeps
        self._cholesky = None
        self._ldlt = None
        var current = A.copy()
        var eps = strength
        while current.nrows > coarse_size and len(self.levels) + 1 < max_levels:
            var diag = current.diagonal()
            for i in range(diag.size):
                var d = diag.unsafe_get(i)
                if not (d > 0) or not isfinite(d):
                    raise Error(
                        String(
                            "SmoothedAggregation: diagonal entry ",
                            i,
                            " is ",
                            d,
                            " on level ",
                            len(self.levels),
                            "; the matrix must be positive definite",
                        )
                    )
            var agg = _aggregates(current, diag, eps)
            eps *= 0.5
            if agg.count == 0 or agg.count >= current.nrows:
                break  # nothing to coarsen: factor this level
            var lam = _lambda_max(current, diag)
            var P = _smoothed_prolongation(current, agg, 4.0 / 3.0 / lam)
            var R = P.transpose()
            var next = R @ (current @ P)
            var dinv = real_vector(diag.size)
            for i in range(diag.size):
                dinv.unsafe_set(i, 1 / diag.unsafe_get(i))
            self.levels.append(_Level(current^, P^, R^, dinv^, lam))
            current = next^
        self.coarse = current^
        if self.coarse.nrows == 0:
            return
        try:
            self._cholesky = SparseCholesky(self.coarse)
        except:
            # semidefinite (a Neumann Laplacian's constants): regularize
            # the zero pivot; the kernel component of the result is the
            # caller's to remove (cg's nullspace=)
            var big: Float64 = 0
            for i in range(self.coarse.nrows):
                big = max(big, abs(self.coarse[i, i]))
            var signs = List[Int](length=self.coarse.nrows, fill=1)
            self._ldlt = SparseLDLT(
                self.coarse,
                signs=signs^,
                regularization=1e-10 * big,
                refinement_steps=0,
            )

    def num_rows(self) -> Int:
        """The number of rows.

        Returns:
            `n`.
        """
        if len(self.levels) == 0:
            return self.coarse.nrows
        return self.levels[0].A.nrows

    def num_cols(self) -> Int:
        """The number of columns.

        Returns:
            `n`.
        """
        return self.num_rows()

    def num_levels(self) -> Int:
        """The number of levels, the coarsest included.

        Returns:
            The hierarchy's depth.
        """
        return len(self.levels) + 1

    def level_sizes(self) -> List[Int]:
        """The unknowns on each level, finest first.

        Returns:
            One size per level.
        """
        var out = List[Int]()
        for l in range(len(self.levels)):
            out.append(self.levels[l].A.nrows)
        out.append(self.coarse.nrows)
        return out^

    def operator_complexity(self) -> Float64:
        """The stored entries of all levels' matrices over the finest's:
        the memory and per-cycle work relative to one product with `A`.

        Returns:
            `sum nnz(A_l) / nnz(A_0)`, 1 or more.
        """
        var total = self.coarse.nnz()
        for l in range(len(self.levels)):
            total += self.levels[l].A.nnz()
        var finest = (
            self.levels[0].A.nnz() if len(self.levels)
            > 0 else self.coarse.nnz()
        )
        return Float64(total) / Float64(max(finest, 1))

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """`y = B x`, one V-cycle from a zero guess.

        Args:
            x: The residual, length `n`.
            y: The correction, overwritten.

        Raises:
            If `x` or `y` does not have length `n`.
        """
        var n = self.num_rows()
        if x.size != n or y.size != n:
            raise Error(
                String(
                    "SmoothedAggregation: vectors of length ",
                    x.size,
                    " and ",
                    y.size,
                    " for n = ",
                    n,
                )
            )
        y = self._cycle(0, x)

    def _cycle(self, l: Int, b: RealArray) raises -> RealArray:
        if l == len(self.levels):
            return self._coarse_solve(b)
        ref level = self.levels[l]
        var x = real_vector(b.size)
        for _ in range(self.sweeps):
            self._smooth(level, b, x, True)
        var r = _residual(level.A, x, b)
        var e = self._cycle(l + 1, level.R @ r)
        var pe = level.P @ e
        for i in range(x.size):
            x.unsafe_set(i, x.unsafe_get(i) + pe.unsafe_get(i))
        for _ in range(self.sweeps):
            self._smooth(level, b, x, False)
        return x^

    def _coarse_solve(self, b: RealArray) raises -> RealArray:
        if b.size == 0:
            return real_vector(0)
        if self._cholesky:
            return self._cholesky.value().solve(b)
        return self._ldlt.value().solve(b)

    def _smooth(
        self, level: _Level, b: RealArray, mut x: RealArray, pre: Bool
    ) raises:
        if self.relaxation == Relaxation.GAUSS_SEIDEL:
            _gauss_seidel(level.A, level.dinv, b, x, forward=pre)
        else:
            _chebyshev(level.A, level.dinv, level.lam, b, x, self.degree)


def _residual(A: CSRMatrix, x: RealArray, b: RealArray) raises -> RealArray:
    """`b - A x`, by a loop: NuMojo's elementwise `-` costs 50 times the
    product on small vectors."""
    var r = A @ x
    for i in range(r.size):
        r.unsafe_set(i, b.unsafe_get(i) - r.unsafe_get(i))
    return r^


def _gauss_seidel(
    A: CSRMatrix, dinv: RealArray, b: RealArray, mut x: RealArray, forward: Bool
):
    var n = A.nrows
    for step in range(n):
        var i = step if forward else n - 1 - step
        var s = b.unsafe_get(i)
        for t in range(A.indptr[i], A.indptr[i + 1]):
            var c = A.indices[t]
            if c != i:
                s -= A.data[t] * x.unsafe_get(c)
        x.unsafe_set(i, s * dinv.unsafe_get(i))


def _chebyshev(
    A: CSRMatrix,
    dinv: RealArray,
    lam: Float64,
    b: RealArray,
    mut x: RealArray,
    degree: Int,
) raises:
    """`degree` steps of Chebyshev iteration on `D^-1 A x = D^-1 b` for the
    interval `[0.1, 1.1] lam` (Saad, Algorithm 12.1)."""
    var lo = 0.1 * lam
    var hi = 1.1 * lam
    var theta = (hi + lo) / 2
    var delta = (hi - lo) / 2
    var sigma = theta / delta
    var rho = 1 / sigma
    var n = x.size
    var r = _residual(A, x, b)
    var d = real_vector(n)
    for i in range(n):
        d.unsafe_set(i, r.unsafe_get(i) * dinv.unsafe_get(i) / theta)
    for k in range(degree):
        for i in range(n):
            x.unsafe_set(i, x.unsafe_get(i) + d.unsafe_get(i))
        if k + 1 == degree:
            break
        var ad = A @ d
        for i in range(n):
            r.unsafe_set(i, r.unsafe_get(i) - ad.unsafe_get(i))
        var rho_new = 1 / (2 * sigma - rho)
        var c1 = rho_new * rho
        var c2 = 2 * rho_new / delta
        for i in range(n):
            d.unsafe_set(
                i,
                c1 * d.unsafe_get(i)
                + c2 * dinv.unsafe_get(i) * r.unsafe_get(i),
            )
        rho = rho_new
