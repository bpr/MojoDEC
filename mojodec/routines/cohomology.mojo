# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Cohomology and harmonic forms (`mojodec.routines.cohomology`).
==============================================================
Integer cocycles and a harmonic basis built from them; not in PyDEC.

- `cohomology_basis(sc, k)` gives `b_k` integer k-cocycles `z_i` whose
  classes span the k-th cohomology, and `b_k` integer k-cycles `c_j`
  dual to them: `<z_i, c_j> = delta_ij`. Both come from exact integer
  elimination of the boundary and coboundary operators.
- `harmonic_basis(sc, k)` makes each cocycle harmonic by removing its
  exact part: `h_i = z_i - d p_i`, with `d^T M d p_i = d^T M z_i` solved by
  MINRES, `M` the Hodge star (or any symmetric mass matrix, such as
  `whitney_innerproduct`). Then `d h_i = 0` because `d d = 0`,
  `d^T M h_i = 0` is the solve, and the periods stay
  `<h_i, c_j> = delta_ij` because `<d p, c> = <p, boundary c> = 0`.

The count comes from the elimination, so no eigenvalue tolerance decides
how many harmonic forms there are, and the basis is the same each run.

```mojo
var basis = cohomology_basis(sc, 1)     # len(basis) == sc.betti_numbers()[1]
var H = harmonic_basis(sc, 1)           # List[Cochain], star-harmonic
var p = basis.periods(H[0])             # [1, 0, 0, ...]
```
"""

from mojodec.core.cell_set import CellSet
from mojodec.core.cochain import Cochain
from mojodec.core.primality import Primality
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.typed import (
    DUAL,
    PRIMAL,
    MatrixOp,
    TypedCochain,
    TypedComplex,
)
from mojodec.core.utils import RealArray, real_vector
from mojodec.routines.linalg import jacobi, minres
from mojodec.routines.topology import closure_of

comptime _MAX_ENTRY = 1 << 40
"""Above this, an entry of the elimination is taken as runaway growth."""


struct CohomologyBasis(Movable, Sized):
    """Integer k-cocycles spanning the k-th cohomology, with dual cycles.

    `cocycles[i]` is a primal k-cochain with integer values and
    `d cocycles[i] = 0` exactly. `cycles[j]` holds one integer coefficient
    per k-simplex of `sc[k]`, with `boundary cycles[j] = 0`. The pairing
    is the identity: `<cocycles[i], cycles[j]> = delta_ij`, so the cycles
    span the k-th homology and the cocycles its dual.
    """

    var k: Int
    """The degree."""
    var cocycles: List[Cochain]
    """`b_k` integer cocycles, one class each."""
    var cycles: List[RealArray]
    """`b_k` integer cycles, coefficients on the k-simplices."""

    def __init__(
        out self,
        k: Int,
        var cocycles: List[Cochain],
        var cycles: List[RealArray],
    ):
        """Assemble a basis; `cohomology_basis` builds them.

        Args:
            k: The degree.
            cocycles: The cocycles.
            cycles: The dual cycles, as many as the cocycles.
        """
        self.k = k
        self.cocycles = cocycles^
        self.cycles = cycles^

    def __len__(self) -> Int:
        """The Betti number `b_k`: the number of cocycles.

        Returns:
            `len(self.cocycles)`.
        """
        return len(self.cocycles)

    def periods(self, c: Cochain) raises -> List[Float64]:
        """The integrals of a k-cochain over the basis cycles.

        For a cocycle `c` they are its coordinates in cohomology:
        `c - sum_j periods[j] cocycles[j]` is exact. For a harmonic basis
        from `harmonic_basis` they are the identity's rows.

        Args:
            c: A primal k-cochain on the complex of the basis.

        Returns:
            `<c, cycles[j]>` for each j.

        Raises:
            If `c` is not a primal k-cochain of the right size.
        """
        if c.k != self.k or c.primality() != Primality.PRIMAL:
            raise Error(
                String("periods: expected a primal ", self.k, "-cochain")
            )
        var out = List[Float64](capacity=len(self.cycles))
        for ref z in self.cycles:
            if z.size != c.v.size:
                raise Error("periods: the cochain has the wrong size")
            var s: Float64 = 0
            for i in range(z.size):
                s += z.unsafe_get(i) * c.v.unsafe_get(i)
            out.append(s)
        return out^


struct _Sparse(Copyable, Movable, Sized):
    """An integer vector: sorted indices and their nonzero values."""

    var idx: List[Int]
    var val: List[Int]

    def __init__(out self):
        self.idx = List[Int]()
        self.val = List[Int]()

    def __len__(self) -> Int:
        return len(self.idx)


def _unit(i: Int) -> _Sparse:
    """The i-th unit vector."""
    var u = _Sparse()
    u.idx.append(i)
    u.val.append(1)
    return u^


def _minus_multiple(
    a: _Sparse, c: Int, b: _Sparse, who: String
) raises -> _Sparse:
    """`a - c b`, dropping zeros."""
    var out = _Sparse()
    var i = 0
    var j = 0
    while i < len(a) or j < len(b):
        var r: Int
        var v: Int
        if j == len(b) or (i < len(a) and a.idx[i] < b.idx[j]):
            r = a.idx[i]
            v = a.val[i]
            i += 1
        elif i == len(a) or b.idx[j] < a.idx[i]:
            r = b.idx[j]
            v = -c * b.val[j]
            j += 1
        else:
            r = a.idx[i]
            v = a.val[i] - c * b.val[j]
            i += 1
            j += 1
        if v != 0:
            if abs(v) > _MAX_ENTRY:
                raise Error(who + ": integer elimination entries overflowed")
            out.idx.append(r)
            out.val.append(v)
    return out^


def _row(A: CSRMatrix, i: Int, drop: List[Bool], who: String) raises -> _Sparse:
    """Row i of an integer matrix, sorted by column, without the columns
    `drop` marks (none if it is empty)."""
    var out = _Sparse()
    for t in range(A.indptr[i], A.indptr[i + 1]):
        if len(drop) > 0 and drop[A.indices[t]]:
            continue
        var x = A.data[t]
        var v = Int(x)
        if Float64(v) != x:
            raise Error(String(who, ": the entry ", x, " is not an integer"))
        if v == 0:
            continue
        # insertion: the rows of d have a few entries each
        var at = len(out)
        out.idx.append(A.indices[t])
        out.val.append(v)
        while at > 0 and out.idx[at - 1] > out.idx[at]:
            out.idx.swap_elements(at - 1, at)
            out.val.swap_elements(at - 1, at)
            at -= 1
    return out^


struct _Reduction(Movable):
    """The outcome of reducing integer columns."""

    var is_pivot: List[Bool]
    """Whether each column kept a nonzero reduced form (a column basis)."""
    var kernel: List[Int]
    """The columns reduced to zero, in order."""
    var vectors: List[_Sparse]
    """For each kernel column asked for, in order: the integer combination
    of columns that is zero, with coefficient 1 on that column."""

    def __init__(out self, ncols: Int):
        self.is_pivot = List[Bool](length=ncols, fill=False)
        self.kernel = List[Int]()
        self.vectors = List[_Sparse]()


def _reduce(
    columns: List[_Sparse],
    nrows: Int,
    want: List[Bool],
    who: String,
) raises -> _Reduction:
    """Column-reduce an integer matrix by exact integer operations.

    Each column in turn has its lowest entry cancelled by an earlier
    reduced column with the same lowest row, as persistent homology does,
    until it is zero or has a new lowest row. A multiple must be an
    integer: the pivot has to divide the entry, which holds whenever the
    pivots are +-1. The combinations are tracked only when `want` is
    nonempty, and kept for the kernel columns it marks.
    """
    var ncols = len(columns)
    var track = len(want) > 0
    var out = _Reduction(ncols)
    var pivot_of_row = List[Int](length=nrows, fill=-1)
    var R = List[_Sparse](capacity=ncols)
    var V = List[_Sparse](capacity=ncols if track else 0)
    for i in range(ncols):
        var col = columns[i].copy()
        var v = _unit(i) if track else _Sparse()
        while len(col) > 0:
            var low = col.idx[len(col) - 1]
            var j = pivot_of_row[low]
            if j < 0:
                break
            var a = col.val[len(col) - 1]
            var b = R[j].val[len(R[j]) - 1]
            if a % b != 0:
                raise Error(
                    String(
                        who,
                        ": elimination needs the non-integer multiple ",
                        a,
                        "/",
                        b,
                        " (torsion, or pivots other than +-1)",
                    )
                )
            var c = a // b
            col = _minus_multiple(col, c, R[j], who)
            if track:
                v = _minus_multiple(v, c, V[j], who)
        if len(col) > 0:
            pivot_of_row[col.idx[len(col) - 1]] = i
            out.is_pivot[i] = True
            R.append(col^)
            V.append(v^ if track else _Sparse())
        else:
            out.kernel.append(i)
            if track and want[i]:
                out.vectors.append(v^)
            R.append(_Sparse())
            if track:
                V.append(_Sparse())
    return out^


def cohomology_basis(sc: SimplicialComplex, k: Int) raises -> CohomologyBasis:
    """Integer cocycles spanning the k-th cohomology, and dual cycles.

    Two exact integer column reductions:
    1. The boundary of the k-simplices: the pivot columns `G` are a basis
       of the image of `d` on (k-1)-cochains, read the other way round.
       Every cohomology class has exactly one representative vanishing
       on `G`, so the cocycles are sought among the other k-simplices `N`.
    2. The coboundary of the simplices of `N`: its kernel is the
       cocycles vanishing on `G`, one per non-pivot column `f`, with
       value 1 on `f` and 0 on the other non-pivot columns. There are
       `b_k` of them.
    The cycle dual to the cocycle of `f` is `f` minus the combination of
    `G` with the same boundary, from the first reduction. On a surface
    with k = 1 this is a spanning tree with its cotree (each cocycle a cut
    across one handle or hole, each cycle a loop around it).

    The pivots of a simplicial complex's operators are almost always +-1,
    so the elimination stays integral; if it needs a fraction (torsion,
    or an unlucky order), the function raises rather than return
    rational cocycles.

    Args:
        sc: The complex.
        k: The degree, `0..n`.

    Returns:
        The cocycles and cycles; `len` is the rational Betti number `b_k`.

    Raises:
        If `k` is outside `0..n`, or the elimination needs a fraction or
        its entries exceed 2^40.
    """
    return _cohomology(sc, k, None, "cohomology_basis")


def cohomology_basis(
    sc: SimplicialComplex, k: Int, relative_to: CellSet
) raises -> CohomologyBasis:
    """Integer cocycles spanning the k-th cohomology relative to a
    subcomplex, and dual relative cycles.

    Relative cochains vanish on the subcomplex `L`; a relative cocycle has
    `d z = 0` there too, and a relative cycle has its boundary in `L`. The
    elimination is that of the absolute overload with the cells of `L`
    removed. Relative to the boundary (`sc.boundary_cells()`) these are
    the classes of harmonic forms with Dirichlet conditions; their number
    is `b_k(K, L)` (`sc.relative_betti_numbers()`), which for an
    orientable manifold equals `b_{n-k}(K)` (Lefschetz duality). Every
    cocycle is zero on the cells of `L`.

    ```mojo
    var rel = cohomology_basis(sc, 1, sc.boundary_cells())
    ```

    Args:
        sc: The complex.
        k: The degree, `0..n`.
        relative_to: A subcomplex of `sc`: every face of a cell in it is
            in it.

    Returns:
        The relative cocycles and cycles; `len` is `b_k(K, L)`.

    Raises:
        If `k` is outside `0..n`, `relative_to` does not belong to `sc` or
        is not closed, or as the absolute overload.
    """
    _require_subcomplex(sc, relative_to, "cohomology_basis")
    return _cohomology(sc, k, relative_to.copy(), "cohomology_basis")


def _check_degree(sc: SimplicialComplex, k: Int, who: String) raises:
    var n = sc.complex_dimension()
    if k < 0 or k > n:
        raise Error(String(who, ": k = ", k, " is outside 0..", n))


def _require_subcomplex(
    sc: SimplicialComplex, cells: CellSet, who: String
) raises:
    cells.check_complex(sc)
    if closure_of(sc, cells) != cells:
        raise Error(
            who,
            ": the cells are not a subcomplex (a face of a cell is missing)",
        )


def _outside(
    sc: SimplicialComplex, k: Int, sub: Optional[CellSet]
) raises -> List[Bool]:
    """Which k-simplices are in the subcomplex (empty if there is none)."""
    if not sub or k < 0 or k > sc.complex_dimension():
        return List[Bool]()
    return sub.value()[k].mask()


def _cohomology(
    sc: SimplicialComplex, k: Int, sub: Optional[CellSet], who: String
) raises -> CohomologyBasis:
    """The two reductions, on the cells outside `sub`."""
    _check_degree(sc, k, who)
    var n = sc.complex_dimension()
    var m = sc[k].num_simplices
    var in_sub = _outside(sc, k, sub)
    var below = _outside(sc, k - 1, sub)
    # the k-simplices that carry values
    var cells = List[Int](capacity=m)
    for f in range(m):
        if len(in_sub) == 0 or not in_sub[f]:
            cells.append(f)
    # 1. the boundaries of the k-simplices: rows of d on (k-1)-cochains
    var faces = List[_Sparse](capacity=len(cells))
    var nfaces = 0
    if k > 0:
        ref D = sc[k - 1].d
        nfaces = D.ncols
        for f in cells:
            faces.append(_row(D, f, below, who))
    else:
        for _ in range(len(cells)):
            faces.append(_Sparse())
    var first = _reduce(faces, nfaces, List[Bool](), who)
    # 2. the coboundaries of the non-pivot simplices (outside a subcomplex,
    # so are their cofaces)
    var N = List[Int]()  # positions in `cells`
    for f in range(len(cells)):
        if not first.is_pivot[f]:
            N.append(f)
    var cofaces = List[_Sparse](capacity=len(N))
    var ncofaces = 0
    if k < n:
        ref B = sc[k + 1].boundary
        ncofaces = B.ncols
        for f in N:
            cofaces.append(_row(B, cells[f], List[Bool](), who))
    else:
        for _ in range(len(N)):
            cofaces.append(_Sparse())
    var second = _reduce(
        cofaces, ncofaces, List[Bool](length=len(N), fill=True), who
    )
    var cocycles = List[Cochain](capacity=len(second.kernel))
    var want = List[Bool](length=len(cells), fill=False)
    for i in range(len(second.kernel)):
        want[N[second.kernel[i]]] = True
        var z = real_vector(m)
        ref vec = second.vectors[i]
        for t in range(len(vec)):
            z.unsafe_set(cells[N[vec.idx[t]]], Float64(vec.val[t]))
        cocycles.append(Cochain(sc, k, Primality.PRIMAL, z^))
    # the dual cycles, from the first reduction with its combinations kept
    var again = _reduce(faces, nfaces, want, who)
    var cycles = List[RealArray](capacity=len(cocycles))
    for i in range(len(again.vectors)):
        var c = real_vector(m)
        ref vec = again.vectors[i]
        for t in range(len(vec)):
            c.unsafe_set(cells[vec.idx[t]], Float64(vec.val[t]))
        cycles.append(c^)
    # both lists follow the simplex order of their marked columns
    return CohomologyBasis(k, cocycles^, cycles^)


def harmonic_basis(
    sc: SimplicialComplex, k: Int, rtol: Float64 = 1e-12
) raises -> List[Cochain]:
    """A basis of the harmonic k-cochains in the diagonal Hodge star.

    Each `h_i` is the harmonic representative of a `cohomology_basis`
    cocycle: `d h_i = 0`, `delta h_i = 0`, and `<h_i, c_j> = delta_ij` on
    the basis cycles. See the overload with a mass matrix.

    Args:
        sc: The complex.
        k: The degree, `0..n`.
        rtol: The relative residual at which MINRES stops.

    Returns:
        `b_k` primal k-cochains.

    Raises:
        As `cohomology_basis`, if a diagonal entry of `d^T star d` is zero,
        or if MINRES does not converge.
    """
    _check_degree(sc, k, "harmonic_basis")
    return _harmonic(sc, k, sc[k].star.to_csr(), None, rtol)


def harmonic_basis(
    sc: SimplicialComplex, k: Int, M: CSRMatrix, rtol: Float64 = 1e-12
) raises -> List[Cochain]:
    """A basis of the harmonic k-cochains in the inner product `M`.

    For each cocycle `z` of `cohomology_basis(sc, k)`, solve the singular
    but consistent `(d^T M d) p = d^T M z` on (k-1)-cochains with
    Jacobi-preconditioned MINRES (which also copes with an indefinite
    star on a mesh that is not well-centered), and return `h = z - d p`.
    It is closed, M-orthogonal to every exact cochain, and has the
    cocycle's periods. For k = 0 the cocycles (indicators of the
    connected pieces) are already harmonic.

    Args:
        sc: The complex.
        k: The degree, `0..n`.
        M: A symmetric mass matrix on k-cochains, such as
            `whitney_innerproduct(sc, k)`.
        rtol: The relative residual at which MINRES stops.

    Returns:
        `b_k` primal k-cochains.

    Raises:
        As `cohomology_basis`, if `M` is not `N_k x N_k` or a diagonal
        entry of `d^T M d` is zero, or if MINRES does not converge.
    """
    return _harmonic(sc, k, M, None, rtol)


def harmonic_basis(
    sc: SimplicialComplex,
    k: Int,
    relative_to: CellSet,
    rtol: Float64 = 1e-12,
) raises -> List[Cochain]:
    """Harmonic k-cochains vanishing on a subcomplex, in the Hodge star.

    Relative to the boundary these are the harmonic forms with Dirichlet
    conditions: zero on the boundary k-simplices, closed, and coclosed at
    the (k-1)-simplices off the boundary. See the overload with a mass
    matrix.

    Args:
        sc: The complex.
        k: The degree, `0..n`.
        relative_to: A subcomplex, such as `sc.boundary_cells()`.
        rtol: The relative residual at which MINRES stops.

    Returns:
        `b_k(K, L)` primal k-cochains.

    Raises:
        As `cohomology_basis` with `relative_to`, or as the absolute
        overload.
    """
    _check_degree(sc, k, "harmonic_basis")
    _require_subcomplex(sc, relative_to, "harmonic_basis")
    return _harmonic(sc, k, sc[k].star.to_csr(), relative_to.copy(), rtol)


def harmonic_basis(
    sc: SimplicialComplex,
    k: Int,
    M: CSRMatrix,
    relative_to: CellSet,
    rtol: Float64 = 1e-12,
) raises -> List[Cochain]:
    """Harmonic k-cochains vanishing on a subcomplex, in the inner
    product `M`.

    For each cocycle `z` of `cohomology_basis(sc, k, relative_to)`, the
    exact part is removed with (k-1)-cochains `p` that also vanish on the
    subcomplex: `h = z - d p`, `(d^T M d) p = d^T M z` on the rows and
    columns of the (k-1)-simplices outside it. So `h` is zero on the
    subcomplex, closed, M-orthogonal to `d p` for every such `p`, and has
    the relative periods of `z`. Relative to the boundary of a domain in
    the plane, the harmonic 1-forms are `d u` for the functions `u` that
    are discrete harmonic inside and constant on each boundary piece.

    Args:
        sc: The complex.
        k: The degree, `0..n`.
        M: A symmetric mass matrix on k-cochains.
        relative_to: A subcomplex, such as `sc.boundary_cells()`.
        rtol: The relative residual at which MINRES stops.

    Returns:
        `b_k(K, L)` primal k-cochains.

    Raises:
        As `cohomology_basis` with `relative_to`, if `M` is not
        `N_k x N_k`, or as the absolute overload.
    """
    _require_subcomplex(sc, relative_to, "harmonic_basis")
    return _harmonic(sc, k, M, relative_to.copy(), rtol)


def _harmonic(
    sc: SimplicialComplex,
    k: Int,
    M: CSRMatrix,
    sub: Optional[CellSet],
    rtol: Float64,
) raises -> List[Cochain]:
    """`h = z - d p` for each cocycle, with `p` off the subcomplex."""
    var basis = _cohomology(sc, k, sub, "harmonic_basis")
    var m = sc[k].num_simplices
    if M.nrows != m or M.ncols != m:
        raise Error(
            String("harmonic_basis: M is not ", m, " x ", m, " (N_", k, ")")
        )
    if k == 0 or len(basis) == 0:
        return basis.cocycles.copy()
    var below = _outside(sc, k - 1, sub)
    var free = List[Int]()  # the (k-1)-simplices p may live on
    for i in range(sc[k - 1].num_simplices):
        if len(below) == 0 or not below[i]:
            free.append(i)
    var rows = List[Int](capacity=m)
    for i in range(m):
        rows.append(i)
    var D = sc[k - 1].d.select(rows, free)  # (k-1)-cochains -> k-cochains
    var Dt = D.transpose()
    var K = Dt @ (M @ D)
    var P = jacobi(K)
    var out = List[Cochain](capacity=len(basis))
    for ref z in basis.cocycles:
        var res = minres(K, Dt @ (M @ z.v), M=P, rtol=rtol)
        if not res.converged():
            raise Error(
                String(
                    "harmonic_basis: MINRES stopped (",
                    res.stop,
                    ") at residual ",
                    res.residual_norm,
                )
            )
        var dp = D @ res^.take_x()
        var h = z.v.copy()
        for t in range(h.size):
            h.unsafe_set(t, h.unsafe_get(t) - dp.unsafe_get(t))
        out.append(Cochain(sc, k, Primality.PRIMAL, h^))
    return out^


def harmonic_basis[
    n: Int, //, k: Int
](tc: TypedComplex[n], rtol: Float64 = 1e-12) raises -> List[
    TypedCochain[n, k, PRIMAL]
]:
    """The star-harmonic basis as typed cochains; see the untyped overload.

    Parameters:
        n: The complex dimension (inferred).
        k: The degree.

    Args:
        tc: The complex.
        rtol: The relative residual at which MINRES stops.

    Returns:
        `b_k` harmonic primal k-cochains.

    Raises:
        As the untyped overload.
    """
    comptime assert 0 <= k and k <= n, "harmonic_basis: k must be in 0..n"
    return _typed[n, k](tc.sc, harmonic_basis(tc.sc, k, rtol))


def harmonic_basis[
    n: Int, k: Int
](
    tc: TypedComplex[n],
    M: MatrixOp[n, k, PRIMAL, n - k, DUAL],
    rtol: Float64 = 1e-12,
) raises -> List[TypedCochain[n, k, PRIMAL]]:
    """The harmonic basis in a typed mass matrix, such as
    `whitney_innerproduct[k](tc)`; see the untyped overload.

    Parameters:
        n: The complex dimension.
        k: The degree.

    Args:
        tc: The complex.
        M: The mass matrix, from primal k-cochains to dual (n-k)-cochains
            as the Hodge star.
        rtol: The relative residual at which MINRES stops.

    Returns:
        `b_k` harmonic primal k-cochains.

    Raises:
        If `M` was built on another complex, or as the untyped overload.
    """
    if M.complex and not M.complex.value().is_same(tc.sc):
        raise Error("harmonic_basis: M is on a different complex")
    return _typed[n, k](tc.sc, harmonic_basis(tc.sc, k, M.mat, rtol))


def harmonic_basis[
    n: Int, //, k: Int
](
    tc: TypedComplex[n], relative_to: CellSet, rtol: Float64 = 1e-12
) raises -> List[TypedCochain[n, k, PRIMAL]]:
    """The star-harmonic basis relative to a subcomplex as typed cochains;
    see the untyped overload.

    Parameters:
        n: The complex dimension (inferred).
        k: The degree.

    Args:
        tc: The complex.
        relative_to: A subcomplex, such as `tc.sc.boundary_cells()`.
        rtol: The relative residual at which MINRES stops.

    Returns:
        `b_k(K, L)` harmonic primal k-cochains vanishing on the subcomplex.

    Raises:
        As the untyped overload.
    """
    comptime assert 0 <= k and k <= n, "harmonic_basis: k must be in 0..n"
    return _typed[n, k](tc.sc, harmonic_basis(tc.sc, k, relative_to, rtol))


def harmonic_basis[
    n: Int, k: Int
](
    tc: TypedComplex[n],
    M: MatrixOp[n, k, PRIMAL, n - k, DUAL],
    relative_to: CellSet,
    rtol: Float64 = 1e-12,
) raises -> List[TypedCochain[n, k, PRIMAL]]:
    """The harmonic basis in a typed mass matrix, relative to a
    subcomplex; see the untyped overload.

    Parameters:
        n: The complex dimension.
        k: The degree.

    Args:
        tc: The complex.
        M: The mass matrix, from primal k-cochains to dual (n-k)-cochains.
        relative_to: A subcomplex, such as `tc.sc.boundary_cells()`.
        rtol: The relative residual at which MINRES stops.

    Returns:
        `b_k(K, L)` harmonic primal k-cochains vanishing on the subcomplex.

    Raises:
        If `M` was built on another complex, or as the untyped overload.
    """
    if M.complex and not M.complex.value().is_same(tc.sc):
        raise Error("harmonic_basis: M is on a different complex")
    return _typed[n, k](
        tc.sc, harmonic_basis(tc.sc, k, M.mat, relative_to, rtol)
    )


def _typed[
    n: Int, k: Int
](sc: SimplicialComplex, H: List[Cochain]) -> List[TypedCochain[n, k, PRIMAL]]:
    """Wrap untyped primal k-cochains."""
    var out = List[TypedCochain[n, k, PRIMAL]](capacity=len(H))
    for ref h in H:
        out.append(TypedCochain[n, k, PRIMAL](sc, h.v.copy()))
    return out^
