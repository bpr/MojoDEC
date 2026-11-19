# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
# Written from the published algorithms: the spectral transformation of
# Ericsson and Ruhe (Math. Comp. 35(152), 1980), the thick-restart Lanczos
# method of Wu and Simon (SIAM J. Matrix Anal. Appl. 22(2), 2000), and the
# usual M-orthogonal deflation of gradients for Maxwell eigenproblems.
# ===----------------------------------------------------------------------=== #
"""
Sparse symmetric eigenproblems (`mojodec.routines.lanczos_core`).
=================================================================
`eigsh(K, M, nev, sigma=s)` finds the `nev` eigenvalues of
`K x = lambda M x` nearest `s`, with `K` symmetric and `M` symmetric
positive definite.

It runs Lanczos on the shift-inverted operator

    OP = (K - s M)^-1 M,

which is self-adjoint in the `M` inner product and has eigenvalues
`theta = 1 / (lambda - s)`: the eigenvalues nearest `s` become the
largest in magnitude, and Lanczos finds those first. `K - s M` is
factored once (`SparseLDLT`), so each Lanczos step costs two triangular
solves and a product with `M`. The basis is `M`-orthonormalized in full
(twice, classical Gram-Schmidt) and restarted thickly: the wanted Ritz
vectors are kept, the rest dropped. Converged pairs are then locked and
their complement searched again, which finds every copy of a multiple
eigenvalue.

The overloads on typed operators are in `mojodec.routines.lanczos`, which
also forwards the ones here.
"""

from std.math import isfinite, sqrt

from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    RealArray,
    axpy,
    dot,
    probe_vector,
    real_array,
    real_vector,
)
from mojodec.routines import dense
from mojodec.routines.dense import EighResult
from mojodec.routines.sparse_ldlt_core import SparseCholesky, SparseLDLT


comptime _BREAKDOWN = 1e-10
"""A new Lanczos vector whose `M` norm is below this fraction of `OP v`'s
spans nothing new: the basis is an invariant subspace."""


struct _Deflation(Movable):
    """The `M`-orthogonal projection onto the complement of `range(G)`:
    `x - G (G^T M G)^-1 G^T M x`."""

    var G: CSRMatrix
    var GT: CSRMatrix
    var C: SparseCholesky

    def __init__(out self, G: CSRMatrix, M: CSRMatrix) raises:
        self.G = G.copy()
        self.GT = G.transpose()
        var GMG = self.GT @ M @ G
        try:
            self.C = SparseCholesky(GMG)
        except e:
            raise Error(
                String(
                    (
                        "eigsh: G^T M G is not positive definite (G must have"
                        " full column rank): "
                    ),
                    e,
                )
            )

    def project(self, mut x: RealArray, Mx: RealArray) raises:
        """`x -= G (G^T M G)^-1 G^T Mx`, with `Mx = M x` given."""
        var c = self.C.solve(self.GT @ Mx)
        axpy(-1.0, self.G @ c, x)


def _m_normalize(
    mut v: RealArray, mut Mv: RealArray, M: CSRMatrix
) raises -> Float64:
    """Set `Mv = M v`, then scale both to `v^T M v = 1`; returns the norm
    before scaling (0 leaves them)."""
    M.apply_into(v, Mv)
    var nn = dot(v, Mv)
    if not (nn > 0) or not isfinite(nn):
        return 0
    var s = 1 / sqrt(nn)
    for i in range(v.size):
        v.unsafe_set(i, s * v.unsafe_get(i))
        Mv.unsafe_set(i, s * Mv.unsafe_get(i))
    return sqrt(nn)


def _orthogonalize(
    mut w: RealArray, V: List[RealArray], MV: List[RealArray], count: Int
) -> List[Float64]:
    """Remove from `w` its `M`-components along `V[0:count]`, twice
    (classical Gram-Schmidt with reorthogonalization), and return them."""
    var h = List[Float64](length=count, fill=0)
    for _ in range(2):
        for i in range(count):
            var c = dot(MV[i], w)
            axpy(-c, V[i], w)
            h[i] += c
    return h^


def _fresh_vector(
    n: Int,
    seed: Int,
    locked: _Ritz,
    V: List[RealArray],
    MV: List[RealArray],
    count: Int,
    M: CSRMatrix,
    deflation: Optional[_Deflation],
    mut v: RealArray,
    mut Mv: RealArray,
) raises -> Bool:
    """A start vector `M`-orthonormal to `locked`, `V[0:count]`, and
    `range(G)`, in `v` and `Mv`; False when none is left (the space is
    spanned)."""
    v = probe_vector(n, UInt64(1000 + seed))
    Mv = M @ v
    var before = sqrt(abs(dot(v, Mv)))
    for _ in range(2):
        if deflation:
            M.apply_into(v, Mv)
            deflation.value().project(v, Mv)
        _ = _orthogonalize(v, locked.X, locked.MX, len(locked.theta))
        _ = _orthogonalize(v, V, MV, count)
    var after = _m_normalize(v, Mv, M)
    return after > _BREAKDOWN * before


def eigsh(
    K: CSRMatrix,
    M: CSRMatrix,
    nev: Int,
    *,
    sigma: Float64,
    tol: Float64 = 1e-12,
    ncv: Optional[Int] = None,
    max_restarts: Int = 300,
) raises -> EighResult:
    """The `nev` eigenpairs of `K x = lambda M x` with `lambda` nearest
    `sigma`, by shift-invert Lanczos.

    ```mojo
    # the 6 smallest eigenvalues of a Laplacian with a mass matrix; the
    # shift sits below the spectrum (0 is an eigenvalue, and K - 0 M is
    # singular)
    var eig = eigsh(K, M, 6, sigma=-0.1)
    var eta = eig.residuals(K, M)          # backward errors, ~1e-15
    ```

    `K - sigma M` is factored once with `SparseLDLT` (AMD ordering), so
    each step is two triangular solves and a product with `M`. Its zero
    and negative pivots make `sigma` at an eigenvalue an error, and any
    other `sigma` works, inside the spectrum too.

    With `deflate=G`, the eigenvectors are sought `M`-orthogonal to
    `range(G)`: for the curl-curl operator `K = d1^T M2 d1` on edges, whose
    kernel is the gradients `range(d0)`, pass `G = d0` (restricted to the
    free cells) and the kernel's eigenvalue 0 never appears. `OP` maps
    `range(G)` and its `M`-orthogonal complement to themselves, so each
    new Lanczos vector is projected (one solve with `G^T M G`, factored
    once). The shift must not be the kernel's eigenvalue either: `K - 0 M`
    is singular there, so shift below 0.

    Lanczos from one vector finds one copy of each eigenvalue; further
    copies of a multiple one come only from rounding. So the converged
    pairs are locked and their `M`-orthogonal complement searched again
    from a fresh vector, until nothing nearer `sigma` turns up: every copy
    is found (on an icosphere, whose `l(l+1)` are exactly multiple, plain
    Lanczos missed some).

    Args:
        K: The `n x n` symmetric matrix.
        M: The `n x n` symmetric positive definite mass matrix
            (`CSRMatrix.identity(n)` for `K x = lambda x`).
        nev: The number of eigenpairs, `1 <= nev <= n`.
        sigma: The shift: the eigenvalues nearest it are found.
        tol: A Ritz pair is accepted when `||OP x - theta x||_M <=
            tol |theta|`; the backward error of `(lambda, x)` is then
            about `tol` times `||K - sigma M|| / ||K||`, or rounding.
        ncv: The Lanczos basis size, `nev < ncv <= n`; default
            `max(2 nev + 1, 20)`, at most `n`.
        max_restarts: Restarts before giving up.

    Returns:
        The eigenvalues in ascending order, and the eigenvectors as the
        columns of an `n x nev` array, `M`-orthonormal.

    Raises:
        If the shapes do not match, `nev` or `ncv` is out of range,
        `K - sigma M` is singular (`sigma` an eigenvalue), `M` is not
        positive definite on the start vector, or the Ritz pairs have not
        converged after `max_restarts`.
    """
    return _eigsh(K, M, nev, sigma, None, tol, ncv, max_restarts)


def eigsh(
    K: CSRMatrix,
    M: CSRMatrix,
    nev: Int,
    *,
    sigma: Float64,
    deflate: CSRMatrix,
    tol: Float64 = 1e-12,
    ncv: Optional[Int] = None,
    max_restarts: Int = 300,
) raises -> EighResult:
    """`eigsh` with eigenvectors `M`-orthogonal to `range(deflate)`; see the
    overload without it.

    Args:
        K: The `n x n` symmetric matrix.
        M: The `n x n` symmetric positive definite mass matrix.
        nev: The number of eigenpairs.
        sigma: The shift: the eigenvalues nearest it are found.
        deflate: `G`, `n x p` of full column rank, such as `d0` restricted
            to the free cells for the curl-curl operator.
        tol: The relative Ritz residual at which a pair is accepted.
        ncv: The Lanczos basis size.
        max_restarts: Restarts before giving up.

    Returns:
        The eigenvalues ascending and the eigenvectors as columns.

    Raises:
        As the overload without `deflate`, or if `G^T M G` is not positive
        definite (`G` rank deficient), or fewer than `nev` eigenpairs are
        left.
    """
    return _eigsh(K, M, nev, sigma, deflate.copy(), tol, ncv, max_restarts)


def _eigsh(
    K: CSRMatrix,
    M: CSRMatrix,
    nev: Int,
    sigma: Float64,
    deflate: Optional[CSRMatrix],
    tol: Float64,
    ncv: Optional[Int],
    max_restarts: Int,
) raises -> EighResult:
    var n = K.nrows
    if K.ncols != n or M.nrows != n or M.ncols != n:
        raise Error(
            String(
                "eigsh: K is ",
                K.nrows,
                " x ",
                K.ncols,
                " and M ",
                M.nrows,
                " x ",
                M.ncols,
                "; both must be the same square size",
            )
        )
    if nev < 1 or nev > n:
        raise Error(String("eigsh: nev = ", nev, " for n = ", n))
    var m = ncv.value() if ncv else min(n, max(2 * nev + 1, 20))
    if m < nev or m > n or (m == nev and m < n):
        raise Error(
            String(
                "eigsh: ncv = ",
                m,
                " must be in (nev, n] = (",
                nev,
                ", ",
                n,
                "]",
            )
        )
    if not (tol > 0):
        raise Error("eigsh: tol must be positive")
    if not isfinite(sigma):
        raise Error("eigsh: sigma must be finite")
    var deflation = Optional[_Deflation](None)
    if deflate:
        ref G = deflate.value()
        if G.nrows != n:
            raise Error(
                String("eigsh: deflate has ", G.nrows, " rows, not ", n)
            )
        deflation = _Deflation(G, M)
    var F: SparseLDLT
    try:
        F = SparseLDLT(K - M * sigma)
    except e:
        raise Error(
            String(
                "eigsh: K - sigma M does not factor at sigma = ",
                sigma,
                " (is sigma an eigenvalue? move it): ",
                e,
            )
        )

    # Lanczos converges to one copy of each eigenvalue at a time; further
    # copies of a multiple one appear only through rounding, and may be
    # missing when the wanted pairs have converged. So lock the pairs
    # found and search their M-orthogonal complement from a fresh vector:
    # a missing copy is nearest sigma there. Swap in whatever is nearer
    # than the farthest pair kept, until nothing is
    var seed = 0
    var found = _run(
        F, M, deflation, _Ritz(), nev, m, tol, max_restarts, n, seed
    )
    if len(found.theta) < nev:
        raise Error(
            String(
                "eigsh: only ",
                len(found.theta),
                " eigenpairs exist (deflated space), not ",
                nev,
            )
        )
    var probe_size = min(nev, _PROBE)
    var probe_ncv = min(n, max(2 * probe_size + 1, 20))
    while True:
        var probe = _run(
            F,
            M,
            deflation,
            found,
            probe_size,
            probe_ncv,
            tol,
            max_restarts,
            n,
            seed,
        )
        var weakest = abs(found.theta[nev - 1])
        if len(probe.theta) == 0 or not (abs(probe.theta[0]) > weakest):
            break
        found = _merge(found^, probe^, nev)

    # lambda = sigma + 1 / theta, ascending
    var pick = List[Int]()
    for i in range(nev):
        pick.append(i)
    for a in range(1, nev):
        var x = pick[a]
        var b = a
        while b > 0 and (1 / found.theta[pick[b - 1]] > 1 / found.theta[x]):
            pick[b] = pick[b - 1]
            b -= 1
        pick[b] = x
    var values = real_vector(nev)
    var vectors = real_array(n, nev)
    for c in range(nev):
        var col = pick[c]
        values.unsafe_set(c, sigma + 1 / found.theta[col])
        ref x = found.X[col]
        for i in range(n):
            vectors.unsafe_set(i * nev + c, x.unsafe_get(i))
    return EighResult(values^, vectors^)


comptime _PROBE = 4
"""Pairs each locked search of the complement looks for."""


struct _Ritz(Movable):
    """Converged Ritz pairs of `OP`, by decreasing `|theta|`, with `M X`."""

    var theta: List[Float64]
    var X: List[RealArray]
    var MX: List[RealArray]

    def __init__(out self):
        self.theta = List[Float64]()
        self.X = List[RealArray]()
        self.MX = List[RealArray]()


def _merge(var a: _Ritz, var b: _Ritz, keep: Int) -> _Ritz:
    """The `keep` pairs of largest `|theta|` from both, in that order."""
    var out = _Ritz()
    var i = 0
    var j = 0
    while len(out.theta) < keep and (i < len(a.theta) or j < len(b.theta)):
        var from_a = j >= len(b.theta) or (
            i < len(a.theta) and abs(a.theta[i]) >= abs(b.theta[j])
        )
        if from_a:
            out.theta.append(a.theta[i])
            out.X.append(a.X[i].copy())
            out.MX.append(a.MX[i].copy())
            i += 1
        else:
            out.theta.append(b.theta[j])
            out.X.append(b.X[j].copy())
            out.MX.append(b.MX[j].copy())
            j += 1
    return out^


def _run(
    F: SparseLDLT,
    M: CSRMatrix,
    deflation: Optional[_Deflation],
    locked: _Ritz,
    want: Int,
    m: Int,
    tol: Float64,
    max_restarts: Int,
    n: Int,
    mut seed: Int,
) raises -> _Ritz:
    """Thick-restart Lanczos for the `want` pairs of `OP` with largest
    `|theta|`, `M`-orthogonal to `locked` (and to `range(G)`); fewer when
    that space has fewer dimensions than the basis would need."""
    # the basis V (M-orthonormal), M V, and the projected matrix H (m x m,
    # symmetric: tridiagonal, with an arrow after each restart)
    var V = List[RealArray]()
    var MV = List[RealArray]()
    for _ in range(m + 1):
        V.append(real_vector(n))
        MV.append(real_vector(n))
    var H = List[Float64](length=m * m, fill=0)
    seed += 1
    if not _fresh_vector(n, seed, locked, V, MV, 0, M, deflation, V[0], MV[0]):
        if len(locked.theta) == 0:
            raise Error("eigsh: M is not positive definite on the start vector")
        return _Ritz()  # the locked pairs span the space
    var k = 0  # basis vectors kept from the last restart
    var active = m  # basis size; smaller if the space is exhausted
    var beta: Float64 = 0  # the residual coupling of the last vector
    var theta = real_vector(0)
    var Y = real_vector(0)
    var order = List[Int]()
    for restart in range(max_restarts + 1):
        var j = k
        while j < active:
            var w = F.solve(MV[j])  # OP v_j
            var Mw = M @ w
            if deflation:
                deflation.value().project(w, Mw)
            _ = _orthogonalize(w, locked.X, locked.MX, len(locked.theta))
            var h = _orthogonalize(w, V, MV, j + 1)
            for i in range(j + 1):
                H[i * m + j] = h[i]
                H[j * m + i] = h[i]
            var size: Float64 = 0
            for i in range(j + 1):
                size += h[i] * h[i]
            beta = _m_normalize(w, Mw, M)
            size = sqrt(size + beta * beta)
            if beta > _BREAKDOWN * size:
                V[j + 1] = w^
                MV[j + 1] = Mw^
            else:
                # an invariant subspace: continue from a fresh vector,
                # uncoupled (beta = 0), unless nothing is left
                beta = 0
                seed += 1
                if not _fresh_vector(
                    n,
                    seed,
                    locked,
                    V,
                    MV,
                    j + 1,
                    M,
                    deflation,
                    V[j + 1],
                    MV[j + 1],
                ):
                    active = j + 1
                    break
            if j + 1 < active:
                H[(j + 1) * m + j] = beta
                H[j * m + j + 1] = beta
            j += 1
        # Rayleigh-Ritz on the active block
        var T = real_array(active, active)
        for r in range(active):
            for c in range(active):
                T.unsafe_set(r * active + c, H[r * m + c])
        var eig = dense.eigh(T)
        theta = eig.values.copy()
        Y = eig.vectors.copy()
        # wanted: largest |theta| (lambda nearest sigma)
        order = List[Int]()
        for i in range(active):
            order.append(i)
        for a in range(1, active):
            var x = order[a]
            var b = a
            while b > 0 and abs(theta.unsafe_get(order[b - 1])) < abs(
                theta.unsafe_get(x)
            ):
                order[b] = order[b - 1]
                b -= 1
            order[b] = x
        if active < m:
            break  # an invariant subspace: every Ritz pair is exact
        var converged = 0
        for i in range(want):
            var c = order[i]
            var r = abs(beta * Y.unsafe_get((active - 1) * active + c))
            if r <= tol * abs(theta.unsafe_get(c)):
                converged += 1
        if converged == want:
            break
        if restart == max_restarts:
            raise Error(
                String(
                    "eigsh: ",
                    converged,
                    " of ",
                    want,
                    " eigenpairs converged after ",
                    max_restarts,
                    " restarts; raise ncv or max_restarts, or move sigma",
                )
            )
        # thick restart: keep the k best Ritz vectors, then the residual
        k = min(want + (m - want) // 2, m - 1)
        var newV = List[RealArray]()
        var newMV = List[RealArray]()
        for i in range(k):
            var c = order[i]
            var x = real_vector(n)
            var Mx = real_vector(n)
            for t in range(m):
                var y = Y.unsafe_get(t * m + c)
                axpy(y, V[t], x)
                axpy(y, MV[t], Mx)
            newV.append(x^)
            newMV.append(Mx^)
        for i in range(m * m):
            H[i] = 0
        for i in range(k):
            var c = order[i]
            H[i * m + i] = theta.unsafe_get(c)
            var s = beta * Y.unsafe_get((m - 1) * m + c)
            H[i * m + k] = s
            H[k * m + i] = s
        V[k] = V[m].copy()
        MV[k] = MV[m].copy()
        for i in range(k):
            V[i] = newV[i].copy()
            MV[i] = newMV[i].copy()
    var out = _Ritz()
    for i in range(min(want, active)):
        var c = order[i]
        var x = real_vector(n)
        var Mx = real_vector(n)
        for t in range(active):
            var y = Y.unsafe_get(t * active + c)
            axpy(y, V[t], x)
            axpy(y, MV[t], Mx)
        out.theta.append(theta.unsafe_get(c))
        out.X.append(x^)
        out.MX.append(Mx^)
    return out^


def eigsh(
    K: CSRMatrix,
    M: DiagonalOperator,
    nev: Int,
    *,
    sigma: Float64,
    tol: Float64 = 1e-12,
    ncv: Optional[Int] = None,
    max_restarts: Int = 300,
) raises -> EighResult:
    """`eigsh` with a diagonal mass matrix, such as a Hodge star; see the
    `CSRMatrix` overload.

    Args:
        K: The `n x n` symmetric matrix.
        M: The diagonal, positive mass matrix.
        nev: The number of eigenpairs.
        sigma: The shift: the eigenvalues nearest it are found.
        tol: The relative Ritz residual at which a pair is accepted.
        ncv: The Lanczos basis size.
        max_restarts: Restarts before giving up.

    Returns:
        The eigenvalues ascending and the eigenvectors as columns.

    Raises:
        As the `CSRMatrix` overload.
    """
    return _eigsh(K, M.to_csr(), nev, sigma, None, tol, ncv, max_restarts)
