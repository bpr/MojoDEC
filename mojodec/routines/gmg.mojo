# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
# Written from the published mathematics: the vertex-star (Arnold-Falk-
# Winther) Schwarz smoother for H(curl) and H(div) (D. N. Arnold, R. S.
# Falk, R. Winther, Numer. Math. 85, 2000), in the patch form of PETSc's
# PCPATCH (Farrell, Knepley, Mitchell, Wechsung, ACM TOMS 47(3), 2021).
# ===----------------------------------------------------------------------=== #
"""
Geometric multigrid for k-forms (`mojodec.routines.gmg`).
========================================================
`GeometricMultigrid(A, hierarchy, k)` preconditions a symmetric positive
definite operator on Whitney k-forms, such as `alpha M_k + d^T M_{k+1} d`,
on a mesh made by uniform refinement.

The levels are the meshes of the `refine` hierarchy, the transfers the
exact Whitney prolongations, and the coarse operators the Galerkin
products `P^T A P`. Point smoothers (Jacobi, Gauss-Seidel) fail for
`k >= 1`: the large kernel of `d` (the gradients of `(k-1)`-forms) makes
the `d^T M d` part invisible to them, and the error there is not smooth.
The vertex-star smoother solves exactly on the k-forms supported in the
star of each vertex, which contains the local gradients, so the cycle
converges in a number of iterations independent of the mesh size, for
every `k` and both the mass and the stiffness parts.
"""

from std.math import isfinite, sqrt

from molina.operator import LinearOperator, Preconditioner
from mojodec.core.simplicial_complex import SimplicialComplex
from molina.sparse import CSRMatrix
from mojodec.core.utils import RealArray, real_vector
from mojodec.routines.refinement import Refinement, refine
from mojodec.routines.sparse_ldlt import SparseCholesky


struct PatchSmoother(Equatable, ImplicitlyCopyable, Writable):
    """How the vertex-star patches are combined in a smoothing step."""

    var _code: UInt8

    comptime MULTIPLICATIVE = PatchSmoother(0)
    """Patch by patch, each with the residual updated by the ones before
    (block Gauss-Seidel): vertex order before the coarse correction,
    reverse order after, so the cycle is symmetric. Serial."""
    comptime ADDITIVE = PatchSmoother(1)
    """All patches from one residual, their corrections summed and damped
    (block Jacobi with overlap): order-free, the parallel form."""

    def __init__(out self, code: UInt8):
        """A smoother from its code; use the named members instead.

        Args:
            code: 0 or 1.
        """
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        """Whether two smoothers are the same.

        Args:
            other: The smoother to compare with.

        Returns:
            True if they are equal.
        """
        return self._code == other._code

    def write_to(self, mut writer: Some[Writer]):
        """Write the smoother's name.

        Args:
            writer: The destination.
        """
        if self == Self.MULTIPLICATIVE:
            writer.write("multiplicative vertex-star")
        else:
            writer.write("additive vertex-star")


def refinement_hierarchy(
    sc: SimplicialComplex, refinements: Int
) raises -> List[Refinement]:
    """Refine `sc` `refinements` times; the hierarchy for
    `GeometricMultigrid`, coarse to fine.

    ```mojo
    var h = refinement_hierarchy(coarse, 5)
    ref fine = h[len(h) - 1].fine           # 32 times finer in each axis
    ```

    Args:
        sc: The coarsest complex.
        refinements: How many times to refine, at least 1.

    Returns:
        One `Refinement` per step; each one's `fine` is the next one's
        `coarse`.

    Raises:
        If `refinements` is not positive, or as `refine`.
    """
    if refinements < 1:
        raise Error("refinement_hierarchy: refinements must be positive")
    var out = List[Refinement]()
    out.append(refine(sc))
    for _ in range(refinements - 1):
        var next = refine(out[len(out) - 1].fine)
        out.append(next^)
    return out^


def _free_cells(
    sc: SimplicialComplex, k: Int, dirichlet: Bool
) raises -> List[Int]:
    """The k-simplices that carry unknowns: all, or those off the
    boundary."""
    var out = List[Int]()
    if not dirichlet:
        for i in range(sc[k].num_simplices):
            out.append(i)
        return out^
    var boundary = sc.boundary_cells()[k].copy()
    for i in range(sc[k].num_simplices):
        if i not in boundary:
            out.append(i)
    return out^


struct _Patches(Movable):
    """The vertex stars of one level: patch `p` holds the unknowns
    `dofs[ptr[p]:ptr[p+1]]` (increasing), and the Cholesky factor of
    `A` restricted to them, `m x m` row-major from `factor_at[p]`."""

    var ptr: List[Int]
    var dofs: List[Int]
    var factor_at: List[Int]
    var factors: List[Float64]

    def __init__(
        out self,
        sc: SimplicialComplex,
        k: Int,
        free: List[Int],
        A: CSRMatrix,
    ) raises:
        var nv = sc[0].num_simplices
        var nk = sc[k].num_simplices
        var local = List[Int](length=nk, fill=-1)
        for i in range(len(free)):
            local[free[i]] = i
        # the free k-simplices at each vertex, by a counting sort
        ref S = sc[k].simplices
        var w = k + 1
        var count = List[Int](length=nv + 1, fill=0)
        for s in range(nk):
            if local[s] >= 0:
                for j in range(w):
                    count[Int(S.unsafe_get(s * w + j)) + 1] += 1
        for v in range(nv):
            count[v + 1] += count[v]
        self.ptr = count.copy()
        self.dofs = List[Int](length=count[nv], fill=0)
        var at = count^
        for s in range(nk):
            if local[s] >= 0:
                for j in range(w):
                    var v = Int(S.unsafe_get(s * w + j))
                    self.dofs[at[v]] = local[s]
                    at[v] += 1
        # simplices are visited in increasing order: each patch is sorted.
        # Factor A on each patch
        self.factor_at = List[Int](capacity=nv + 1)
        self.factors = List[Float64]()
        var pos = List[Int](length=A.nrows, fill=-1)
        for p in range(nv):
            self.factor_at.append(len(self.factors))
            var lo = self.ptr[p]
            var m = self.ptr[p + 1] - lo
            if m == 0:
                continue
            for a in range(m):
                pos[self.dofs[lo + a]] = a
            var base = len(self.factors)
            for _ in range(m * m):
                self.factors.append(0)
            for a in range(m):
                var i = self.dofs[lo + a]
                for t in range(A.indptr[i], A.indptr[i + 1]):
                    var b = pos[A.indices[t]]
                    if b >= 0:
                        self.factors[base + a * m + b] = A.data[t]
            for a in range(m):
                pos[self.dofs[lo + a]] = -1
            _cholesky_in_place(self.factors, base, m, p)
        self.factor_at.append(len(self.factors))

    def solve(self, p: Int, mut r: List[Float64]):
        """`r = A_pp^-1 r` by the stored factor."""
        var m = self.ptr[p + 1] - self.ptr[p]
        var base = self.factor_at[p]
        for a in range(m):
            var s = r[a]
            for b in range(a):
                s -= self.factors[base + a * m + b] * r[b]
            r[a] = s / self.factors[base + a * m + a]
        for a in range(m - 1, -1, -1):
            var s = r[a]
            for b in range(a + 1, m):
                s -= self.factors[base + b * m + a] * r[b]
            r[a] = s / self.factors[base + a * m + a]


def _cholesky_in_place(
    mut F: List[Float64], base: Int, m: Int, patch: Int
) raises:
    """The lower Cholesky factor of the `m x m` block at `base`, over it."""
    for j in range(m):
        var d = F[base + j * m + j]
        for t in range(j):
            d -= F[base + j * m + t] * F[base + j * m + t]
        if not (d > 0) or not isfinite(d):
            raise Error(
                String(
                    (
                        "GeometricMultigrid: the operator is not positive"
                        " definite on the star of vertex "
                    ),
                    patch,
                )
            )
        var l = sqrt(d)
        F[base + j * m + j] = l
        for i in range(j + 1, m):
            var s = F[base + i * m + j]
            for t in range(j):
                s -= F[base + i * m + t] * F[base + j * m + t]
            F[base + i * m + j] = s / l


struct _Level(Movable):
    var A: CSRMatrix
    var P: CSRMatrix
    """From the next coarser level to this one."""
    var R: CSRMatrix
    var patches: _Patches

    def __init__(
        out self,
        var A: CSRMatrix,
        var P: CSRMatrix,
        var R: CSRMatrix,
        var patches: _Patches,
    ):
        self.A = A^
        self.P = P^
        self.R = R^
        self.patches = patches^


struct GeometricMultigrid(Movable, Preconditioner):
    """One V-cycle of geometric multigrid with vertex-star smoothing, for a
    symmetric positive definite operator on Whitney k-forms of a refined
    mesh: a `Preconditioner`.

    ```mojo
    var h = refinement_hierarchy(coarse, 5)
    ref sc = h[len(h) - 1].fine
    var A = whitney_innerproduct(sc, 1) + (
        sc[1].d.transpose() @ whitney_innerproduct(sc, 2) @ sc[1].d
    )
    var mg = GeometricMultigrid(A, h, 1)       # H(curl)
    var res = cg(A, b, M=mg, rtol=1e-10)       # about 10 iterations, any h
    ```

    Level `l` is the complex `h[l].coarse` (and the finest, `A`'s,
    `h[-1].fine`); `P_l = h[l].prolongation(k)` maps level `l` to `l + 1`
    exactly (nested Whitney spaces), `R = P^T`, and the coarser operators
    are `P^T A P`, which for nested spaces equals the operator assembled
    on the coarser mesh. The coarsest is factored. Each smoothing step
    solves exactly on every vertex star: the free k-simplices containing
    the vertex (for `k = 0`, the vertex itself: point Gauss-Seidel).

    With `dirichlet=True`, `A` acts on the k-simplices off the boundary
    of the finest mesh (tangential trace zero), and every level drops its
    boundary k-simplices the same way.
    """

    var levels: List[_Level]
    """The levels above the coarsest, finest first."""
    var coarse: CSRMatrix
    """The coarsest operator, factored."""
    var smoother: PatchSmoother
    """How the patches are combined."""
    var sweeps: Int
    """Smoothing steps before and after each coarse correction."""
    var damping: Float64
    """The additive smoother's damping (unused by the multiplicative one)."""
    var _factor: SparseCholesky

    def __init__(
        out self,
        A: CSRMatrix,
        hierarchy: List[Refinement],
        k: Int,
        *,
        dirichlet: Bool = False,
        smoother: PatchSmoother = PatchSmoother.MULTIPLICATIVE,
        sweeps: Int = 1,
        damping: Optional[Float64] = None,
    ) raises:
        """Build the levels.

        Args:
            A: The symmetric positive definite operator on the finest
                level's k-cochains (its interior ones with `dirichlet`).
            hierarchy: The refinements, coarse to fine, each one's `fine`
                the next one's `coarse` (`refinement_hierarchy`).
            k: The degree of the forms.
            dirichlet: Whether the boundary k-simplices are left out.
            smoother: `PatchSmoother.MULTIPLICATIVE` or `.ADDITIVE`.
            sweeps: Smoothing steps before and after each correction.
            damping: The additive smoother's weight; default `1 / (k + 2)`
                (each k-simplex lies in `k + 1` stars).

        Raises:
            If the hierarchy is empty or not a chain, `k` is out of range,
            `A` does not have the finest level's size, or `A` is not
            positive definite on a star or the coarsest level.
        """
        if len(hierarchy) == 0:
            raise Error("GeometricMultigrid: the hierarchy is empty")
        var n = hierarchy[0].coarse.complex_dimension()
        if k < 0 or k > n:
            raise Error(
                String("GeometricMultigrid: k = ", k, " for dimension ", n)
            )
        for l in range(1, len(hierarchy)):
            if not hierarchy[l].coarse.is_same(hierarchy[l - 1].fine):
                raise Error(
                    String(
                        "GeometricMultigrid: refinement ",
                        l,
                        " does not refine refinement ",
                        l - 1,
                        "'s fine complex",
                    )
                )
        if sweeps < 1:
            raise Error("GeometricMultigrid: sweeps must be positive")
        var L = len(hierarchy)
        ref finest = hierarchy[L - 1].fine
        var free = _free_cells(finest, k, dirichlet)
        if A.nrows != len(free) or A.ncols != len(free):
            raise Error(
                String(
                    "GeometricMultigrid: A is ",
                    A.nrows,
                    " x ",
                    A.ncols,
                    "; the finest level has ",
                    len(free),
                    " free ",
                    k,
                    "-simplices",
                )
            )
        self.smoother = smoother
        self.sweeps = sweeps
        self.damping = damping.value() if damping else 1.0 / Float64(k + 2)
        if not (self.damping > 0):
            raise Error("GeometricMultigrid: damping must be positive")
        self.levels = List[_Level]()
        var current = A.copy()
        for l in range(L - 1, -1, -1):
            ref r = hierarchy[l]
            var coarse_free = _free_cells(r.coarse, k, dirichlet)
            var P = r.prolongation(k)
            if dirichlet:
                P = P.select(free, coarse_free)
            var R = P.transpose()
            var next = R @ (current @ P)
            var patches = _Patches(r.fine, k, free, current)
            self.levels.append(_Level(current^, P^, R^, patches^))
            current = next^
            free = coarse_free^
        self.coarse = current^
        try:
            self._factor = SparseCholesky(self.coarse)
        except e:
            raise Error(
                String(
                    (
                        "GeometricMultigrid: the coarsest operator does not"
                        " factor: "
                    ),
                    e,
                )
            )

    def num_rows(self) -> Int:
        """The number of rows.

        Returns:
            The finest level's unknowns.
        """
        return self.levels[0].A.nrows

    def num_cols(self) -> Int:
        """The number of columns.

        Returns:
            The finest level's unknowns.
        """
        return self.levels[0].A.nrows

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
                    "GeometricMultigrid: vectors of length ",
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
            return self._factor.solve(b)
        ref level = self.levels[l]
        var x = real_vector(b.size)
        for _ in range(self.sweeps):
            self._smooth(level, b, x, True)
        var r = level.A @ x
        for i in range(r.size):
            r.unsafe_set(i, b.unsafe_get(i) - r.unsafe_get(i))
        var e = self._cycle(l + 1, level.R @ r)
        var pe = level.P @ e
        for i in range(x.size):
            x.unsafe_set(i, x.unsafe_get(i) + pe.unsafe_get(i))
        for _ in range(self.sweeps):
            self._smooth(level, b, x, False)
        return x^

    def _smooth(
        self, level: _Level, b: RealArray, mut x: RealArray, forward: Bool
    ) raises:
        ref A = level.A
        ref patches = level.patches
        var np = len(patches.ptr) - 1
        var local = List[Float64]()
        if self.smoother == PatchSmoother.MULTIPLICATIVE:
            for step in range(np):
                var p = step if forward else np - 1 - step
                var lo = patches.ptr[p]
                var hi = patches.ptr[p + 1]
                if hi == lo:
                    continue
                local.clear()
                for a in range(lo, hi):
                    var i = patches.dofs[a]
                    var s = b.unsafe_get(i)
                    for t in range(A.indptr[i], A.indptr[i + 1]):
                        s -= A.data[t] * x.unsafe_get(A.indices[t])
                    local.append(s)
                patches.solve(p, local)
                for a in range(lo, hi):
                    var i = patches.dofs[a]
                    x.unsafe_set(i, x.unsafe_get(i) + local[a - lo])
            return
        # additive: every patch from the same residual
        var r = A @ x
        for i in range(r.size):
            r.unsafe_set(i, b.unsafe_get(i) - r.unsafe_get(i))
        var dx = real_vector(x.size)
        for p in range(np):
            var lo = patches.ptr[p]
            var hi = patches.ptr[p + 1]
            if hi == lo:
                continue
            local.clear()
            for a in range(lo, hi):
                local.append(r.unsafe_get(patches.dofs[a]))
            patches.solve(p, local)
            for a in range(lo, hi):
                var i = patches.dofs[a]
                dx.unsafe_set(i, dx.unsafe_get(i) + local[a - lo])
        for i in range(x.size):
            x.unsafe_set(i, x.unsafe_get(i) + self.damping * dx.unsafe_get(i))
