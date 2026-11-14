# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Finite element inner products (`mojodec.routines.innerproduct`).
================================================================
Largely inspired by `pydec.fem.innerproduct`: mass matrices of Whitney k-forms
on a simplicial complex, and of the analogous tensor-product elements on a
regular cube complex. `whitney_innerproduct[k](tc)` gives the Whitney mass
matrix as a typed operator from primal k-cochains to dual (n-k)-cochains.
"""


from std.math import sqrt
from std.memory import stack_allocation

from mojodec.core.parallel import chunk_scratch, num_chunks, parallel_ranges
from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.typed import DUAL, PRIMAL, MatrixOp, TypedComplex
from mojodec.core.utils import (
    factorial,
    frame_dual_basis,
    iget,
    int_array,
    int_array_from_rows,
    real_array,
    rows_of,
    cols_of,
)
from mojodec.routines.combinatorial import combinations
from mojodec.routines.cube_array import cube_array_boundary, cube_array_search
from mojodec.routines.simplex_array import simplex_array_searchsorted


struct COOPattern(Copyable, Movable):
    """The row and column indices of a matrix's entries, without values."""

    var rows: List[Int]
    """The row of each entry."""
    var cols: List[Int]
    """The column of each entry."""

    def __init__(out self, var rows: List[Int], var cols: List[Int]):
        """A pattern from its two lists.

        Args:
            rows: The rows.
            cols: The columns.
        """
        self.rows = rows^
        self.cols = cols^

    def __len__(self) -> Int:
        """The number of entries.

        Returns:
            `len(rows)`.
        """
        return len(self.rows)


def massmatrix_rowcols(sc: SimplicialComplex, k: Int) raises -> COOPattern:
    """COO row and column indices of the Whitney k-form mass matrix.

    For each top simplex `s` and each pair `(a, b)` of its k-faces the entry
    `(index(a), index(b))` is listed, simplex-major.

    Args:
        sc: A simplicial complex.
        k: The form degree, `0 <= k <= complex_dimension()`.

    Returns:
        The row and column index lists, of equal length.

    Raises:
        If a k-face of a top simplex is missing from `sc[k]`.
    """
    var p = sc.complex_dimension()
    ref simplices = sc[p].simplices
    var num_simplices = rows_of(simplices)
    var rows = List[Int]()
    var cols = List[Int]()
    if k == p:
        for i in range(num_simplices):
            rows.append(i)
            cols.append(i)
        return COOPattern(rows^, cols^)
    var local = List[Int]()
    for i in range(p + 1):
        local.append(i)
    var k_faces = combinations(local, k + 1)
    var fps = len(k_faces)
    var faces = int_array(num_simplices * fps, k + 1)
    for s in range(num_simplices):
        for n in range(fps):
            for m in range(k + 1):
                faces.unsafe_set(
                    (s * fps + n) * (k + 1) + m,
                    simplices.unsafe_get(s * (p + 1) + k_faces[n][m]),
                )
    var indices = simplex_array_searchsorted(sc[k].simplices, faces)
    rows = List[Int](length=num_simplices * fps * fps, fill=0)
    cols = List[Int](length=num_simplices * fps * fps, fill=0)
    var rp = rows.unsafe_ptr()
    var cp = cols.unsafe_ptr()
    var at = 0
    for s in range(num_simplices):
        for n in range(fps):
            var r = Int(indices.unsafe_get(s * fps + n))
            for m in range(fps):
                rp[unsafe_offset=at] = r
                cp[unsafe_offset=at] = Int(indices.unsafe_get(s * fps + m))
                at += 1
    return COOPattern(rows^, cols^)


def whitney_innerproduct[
    n: Int, //, k: Int
](tc: TypedComplex[n]) raises -> MatrixOp[n, k, PRIMAL, n - k, DUAL]:
    """The Whitney mass matrix as a typed operator.

    It takes primal k-cochains to dual (n-k)-cochains, as the diagonal Hodge
    star `tc.star[k, PRIMAL]()` does, and can replace it in a composition:

    ```mojo
    var M2 = whitney_innerproduct[2](tc)     # primal 2 -> dual 0, in 2D
    var K = tc.d[0, DUAL]() @ M2 @ tc.d[1, PRIMAL]()
    var eig = eigh_generalized(K, whitney_innerproduct[1](tc))
    ```

    Parameters:
        n: The complex dimension (inferred).
        k: The form degree, `0..n`.

    Args:
        tc: The complex, with geometry.

    Returns:
        The symmetric mass matrix, on `tc`'s complex.

    Raises:
        As the untyped overload.
    """
    comptime assert (
        0 <= k and k <= n
    ), "whitney_innerproduct: degree out of range"
    return MatrixOp[n, k, PRIMAL, n - k, DUAL](
        whitney_innerproduct(tc.sc, k), tc.sc
    )


def whitney_innerproduct(sc: SimplicialComplex, k: Int) raises -> CSRMatrix:
    """The mass matrix of Whitney k-forms on a simplicial complex.

    Entry `(i, j)` is the L2 inner product of the Whitney forms of k-simplices
    `i` and `j` (in their sorted orientation). Each top simplex contributes in
    proportion to its unsigned volume.

    Args:
        sc: A simplicial complex with geometry.
        k: The form degree.

    Returns:
        The symmetric `N_k x N_k` mass matrix.

    Raises:
        If `k` is out of range, `sc` was built without geometry, or a top
        simplex is degenerate.
    """
    var p = sc.complex_dimension()
    if k < 0 or k > p:
        raise Error("whitney_innerproduct: invalid form degree")
    if not sc.has_geometry():
        raise Error(
            "whitney_innerproduct: the complex was built without geometry"
        )

    # precomputation on the reference simplex
    var scale_integration = factorial(k) ** 2 / Float64((p + 2) * (p + 1))
    var local = List[Int]()
    for i in range(p + 1):
        local.append(i)
    var k_forms = combinations(local, k)
    var k_faces = combinations(local, k + 1)
    var num_forms = len(k_forms)
    var num_k_faces = len(k_faces)
    var nfp = num_k_faces * num_k_faces
    var forms = List[Int](capacity=num_forms * k)
    for ref f in k_forms:
        for v in f:
            forms.append(v)

    # Local entry (f1, f2) is a signed sum of the determinants of form pairs
    # obtained by dropping vertex n from face f1 and m from face f2: entry t
    # sums terms[term_start[t]:term_start[t + 1]], each a coefficient and an
    # index a * num_forms + b into the table of pair determinants.
    var term_start = List[Int](capacity=nfp + 1)
    var term_pair = List[Int](capacity=nfp * (k + 1) * (k + 1))
    var term_coef = List[Float64](capacity=nfp * (k + 1) * (k + 1))
    term_start.append(0)
    for f1 in range(num_k_faces):
        for f2 in range(num_k_faces):
            for n in range(k + 1):
                for m in range(k + 1):
                    var a = _form_index(k_forms, _without(k_faces[f1], n))
                    var b = _form_index(k_forms, _without(k_faces[f2], m))
                    var same = 1 if k_faces[f1][n] == k_faces[f2][m] else 0
                    var sign: Float64 = 1.0 if (n + m) % 2 == 0 else -1.0
                    term_pair.append(a * num_forms + b)
                    term_coef.append(sign * Float64(same + 1))
            term_start.append(len(term_pair))

    # The determinant of pair (a, b) is that of the k x k block, in the rows
    # of form a and the columns of form b, of the Gram matrix D D^T of the
    # barycentric gradients D of a top simplex (1 when k = 0). With the
    # edge vectors E[r] = x_{r+1} - x_0 as rows and E^T = Q R (Householder
    # QR), gradients 1..p are the rows of R^-1 Q^T and gradient 0 is minus
    # their sum. In the orthonormal frame Q they are the rows of C = R^-1
    # (and minus their sum), and by Cauchy-Binet the block determinant is
    # the sum over k-subsets S of the frame's p axes of
    # det(C[a, S]) det(C[b, S]). Forming D D^T instead squares the
    # conditioning L / h of a thin simplex.
    var axes = List[Int]()
    for i in range(p):
        axes.append(i)
    var axis_sets = combinations(axes, k)
    var num_sets = len(axis_sets)
    var sets = List[Int](capacity=num_sets * k)
    for ref a in axis_sets:
        for v in a:
            sets.append(v)
    var dim = cols_of(sc.vertices())
    var tsp = term_start.unsafe_ptr()
    var tpp = term_pair.unsafe_ptr()
    var tcp = term_coef.unsafe_ptr()

    ref top = sc[p]
    var num_top = top.num_simplices
    # the parallel loops below write every entry of these (a degenerate
    # simplex raises before any is read), and so touch their pages first
    var data = List[Float64](unsafe_uninit_length=num_top * nfp)
    var datap = data.unsafe_ptr()
    # The k-face indices of each top simplex, in the order of k_faces, as
    # `massmatrix_rowcols` finds them, but from the rows of d: faces are
    # named by bit masks of vertex positions in the (sorted) top simplex,
    # and the row of an L-simplex in d_{L-1} lists its faces by increasing
    # index; levels below the top are in lexicographic order, so entry q
    # omits the vertex of rank L - q.
    var w = p + 1
    var size = 1 << w
    var pop = List[Int](length=size, fill=0)
    var bitpos = List[Int](length=size * w, fill=0)
    for M in range(size):
        var r = 0
        for j in range(w):
            if (M >> j) & 1 == 1:
                bitpos[M * w + r] = j
                r += 1
        pop[M] = r
    var face_mask = List[Int](capacity=num_k_faces)
    for ref f in k_faces:
        var M = 0
        for v in f:
            M |= 1 << v
        face_mask.append(M)
    var d_indptr = List[Pointer[Int, MutUntrackedOrigin]]()
    var d_indices = List[Pointer[Int, MutUntrackedOrigin]]()
    for L in range(p):
        d_indptr.append(
            sc[L].d.indptr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        )
        d_indices.append(
            sc[L]
            .d.indices.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
    var faces = List[Int](unsafe_uninit_length=num_top * num_k_faces)
    var facesp = faces.unsafe_ptr()
    var Xp = sc.vertices().unsafe_ptr()
    var Sp = top.simplices.unsafe_ptr()
    var volp = top.primal_volume.unsafe_ptr()
    # top simplices in parallel chunks (meshes of a few thousand run
    # serially), each with its own scratch; a chunk records its first
    # degenerate simplex, raised after the loop
    comptime SERIAL_BELOW = 4096
    var degenerate = List[Int](
        length=num_chunks(num_top, SERIAL_BELOW), fill=-1
    )
    var badp = degenerate.unsafe_ptr()
    var fixed = (p == 3 and dim == 3) or (p == 2 and (dim == 2 or dim == 3))

    def simplices(
        chunk: Int, lo: Int, hi: Int
    ) {
        fixed,
        k,
        p,
        dim,
        num_forms,
        nfp,
        num_sets,
        scale_integration,
        imm forms,
        imm sets,
        tsp,
        tpp,
        tcp,
        datap,
        Xp,
        Sp,
        volp,
        badp,
        w,
        size,
        num_k_faces,
        imm pop,
        imm bitpos,
        imm face_mask,
        imm d_indptr,
        imm d_indices,
        facesp,
    }:
        var edges = chunk_scratch[Float64](p * dim, 0)
        var frame = chunk_scratch[Float64](p * p, 0)
        var grads = chunk_scratch[Float64]((p + 1) * p, 0)
        var minors = chunk_scratch[Float64](num_forms * num_sets, 0)
        var dets = chunk_scratch[Float64](num_forms * num_forms, 1)
        var work = chunk_scratch[Float64](k * k, 0)
        var diag = chunk_scratch[Float64](p, 0)
        var ep = edges.unsafe_ptr()
        var Dp = grads.unsafe_ptr()
        var mp = minors.unsafe_ptr()
        var dp = dets.unsafe_ptr()
        # the fixed shapes' face masks, four simplices at a time
        var gidx = chunk_scratch[Int](4 * size, 0)
        var gp = gidx.unsafe_ptr()
        # triangles in 2D and 3D and tetrahedra in 3D: the same arithmetic
        # unrolled at compile time, four simplices to a SIMD vector
        if fixed:
            var args = _ElementArgs(
                Xp.unsafe_origin_cast[MutUntrackedOrigin](),
                Sp.unsafe_origin_cast[MutUntrackedOrigin](),
                volp.unsafe_origin_cast[MutUntrackedOrigin](),
                scale_integration,
                facesp.unsafe_origin_cast[MutUntrackedOrigin](),
                datap.unsafe_origin_cast[MutUntrackedOrigin](),
                gp.unsafe_origin_cast[MutUntrackedOrigin](),
            )
            var bad: Int
            if p == 3:
                bad = _whitney_fixed[3, 3](k, lo, hi, args, d_indptr, d_indices)
            elif dim == 3:
                bad = _whitney_fixed[2, 3](k, lo, hi, args, d_indptr, d_indices)
            else:
                bad = _whitney_fixed[2, 2](k, lo, hi, args, d_indptr, d_indices)
            if bad >= 0:
                badp[unsafe_offset=chunk] = bad
            return
        for i in range(lo, hi):
            gp[unsafe_offset=size - 1] = i
            for L in range(p, k, -1):
                var ip = d_indptr[L - 1]
                var ix = d_indices[L - 1]
                for M in range(size):
                    if pop.unsafe_get(M) != L + 1:
                        continue
                    var base = ip[unsafe_offset=gp[unsafe_offset=M]]
                    for q in range(L + 1):
                        var bit = bitpos.unsafe_get(M * w + L - q)
                        gp[unsafe_offset=M ^ (1 << bit)] = ix[
                            unsafe_offset=base + q
                        ]
            for a in range(num_k_faces):
                facesp[unsafe_offset=i * num_k_faces + a] = gp[
                    unsafe_offset=face_mask.unsafe_get(a)
                ]
            if k > 0:
                var v0 = Int(Sp[unsafe_offset=i * (p + 1)])
                for r in range(p):
                    var v = Int(Sp[unsafe_offset=i * (p + 1) + r + 1])
                    for c in range(dim):
                        ep[unsafe_offset=r * dim + c] = (
                            Xp[unsafe_offset=v * dim + c]
                            - Xp[unsafe_offset=v0 * dim + c]
                        )
                if not frame_dual_basis(edges, p, dim, frame, diag):
                    badp[unsafe_offset=chunk] = i
                    return
                for c in range(p):
                    var t: Float64 = 0
                    for r in range(p):
                        var g = frame[r * p + c]
                        Dp[unsafe_offset=(r + 1) * p + c] = g
                        t += g
                    Dp[unsafe_offset=c] = -t
                for a in range(num_forms):
                    for q in range(num_sets):
                        mp[unsafe_offset=a * num_sets + q] = _minor_det(
                            grads, p, forms, a * k, sets, q * k, k, work
                        )
                for a in range(num_forms):
                    for b in range(a, num_forms):
                        var det: Float64 = 0
                        for q in range(num_sets):
                            det += (
                                mp[unsafe_offset=a * num_sets + q]
                                * mp[unsafe_offset=b * num_sets + q]
                            )
                        dp[unsafe_offset=a * num_forms + b] = det
                        dp[unsafe_offset=b * num_forms + a] = det
            var weight = abs(volp[unsafe_offset=i]) * scale_integration
            for t in range(nfp):
                var s: Float64 = 0
                for q in range(tsp[unsafe_offset=t], tsp[unsafe_offset=t + 1]):
                    s += (
                        tcp[unsafe_offset=q]
                        * dp[unsafe_offset=tpp[unsafe_offset=q]]
                    )
                datap[unsafe_offset=i * nfp + t] = s * weight

    parallel_ranges(num_top, SERIAL_BELOW, simplices)
    for i in degenerate:
        if i >= 0:
            raise Error(
                String(
                    "whitney_innerproduct: top simplex ", i, " is degenerate"
                )
            )
    if k == p:
        # one entry per top simplex, on the diagonal
        var A = CSRMatrix(num_top, num_top)
        var ip = A.indptr.unsafe_ptr()
        A.indices = List[Int](unsafe_uninit_length=num_top)
        var ixp = A.indices.unsafe_ptr()
        for i in range(num_top):
            ip[unsafe_offset=i + 1] = i + 1
            ixp[unsafe_offset=i] = i
        A.data = data^
        return A^
    if k == p - 1:
        var Nk = sc[k].num_simplices
        var A: Optional[CSRMatrix]
        if num_k_faces == 3:
            A = _assemble_codim1[3](Nk, 3, faces, data, sc[p].boundary)
        elif num_k_faces == 4:
            A = _assemble_codim1[4](Nk, 4, faces, data, sc[p].boundary)
        else:
            A = _assemble_codim1[0](
                Nk, num_k_faces, faces, data, sc[p].boundary
            )
        if A:
            return A.take()
    return _assemble(sc[k].num_simplices, num_k_faces, faces, data)


comptime _FPtr = Pointer[Float64, MutUntrackedOrigin]
comptime _NPtr = Pointer[Int, MutUntrackedOrigin]
comptime _SPtr = Pointer[Scalar[DType.int], MutUntrackedOrigin]


@fieldwise_init
struct _ElementArgs(Copyable, Movable):
    """The tables and buffers `_whitney_element` reads and writes."""

    var X: _FPtr
    """Vertex coordinates."""
    var S: _SPtr
    """Top simplices' vertex rows."""
    var vol: _FPtr
    """Top simplices' primal volumes."""
    var scale: Float64
    """`k!^2 / ((p + 2)(p + 1))`."""
    var faces: _NPtr
    """Output: the k-face indices of each top simplex."""
    var data: _FPtr
    """Output: the local matrices."""
    var gidx: _NPtr
    """Scratch: global index of each face mask."""


@always_inline
def _whitney_fixed[
    p: Int, dim: Int
](
    k: Int,
    lo: Int,
    hi: Int,
    args: _ElementArgs,
    d_indptr: List[_NPtr],
    d_indices: List[_NPtr],
) -> Int:
    """`_whitney_range` for the run-time degree `k`."""
    comptime for kk in range(p + 1):
        if k == kk:
            return _whitney_range[p, dim, kk](lo, hi, args, d_indptr, d_indices)
    return -1


def _whitney_range[
    p: Int, dim: Int, k: Int
](
    lo: Int,
    hi: Int,
    args: _ElementArgs,
    d_indptr: List[_NPtr],
    d_indices: List[_NPtr],
) -> Int:
    """`_whitney_element` for top simplices `lo` to `hi - 1`, four to a SIMD
    vector and the rest one at a time. Returns the first degenerate one, or
    -1."""
    comptime W = 4
    var i = lo
    while i + W <= hi:
        if not _whitney_element[p, dim, k, W](i, args, d_indptr, d_indices):
            for j in range(i, i + W):
                if not _whitney_one[p, dim, k](j, args, d_indptr, d_indices):
                    return j
        i += W
    while i < hi:
        if not _whitney_one[p, dim, k](i, args, d_indptr, d_indices):
            return i
        i += 1
    return -1


def _whitney_one[
    p: Int, dim: Int, k: Int
](
    i: Int, args: _ElementArgs, d_indptr: List[_NPtr], d_indices: List[_NPtr]
) -> Bool:
    """`_whitney_element` for top simplex `i` alone."""
    return _whitney_element[p, dim, k, 1](i, args, d_indptr, d_indices)


@always_inline
def _ld[W: Int](ptr: _FPtr, idx: Int) -> SIMD[DType.float64, W]:
    """Vector `idx` of a stack array of `W`-lane vectors."""
    return ptr.unsafe_load[width=W](idx * W)


@always_inline
def _st[W: Int](ptr: _FPtr, idx: Int, v: SIMD[DType.float64, W]):
    """Stores vector `idx` of a stack array of `W`-lane vectors."""
    ptr.unsafe_store(idx * W, v)


@always_inline
def _element_faces[
    p: Int, k: Int, W: Int
](i: Int, args: _ElementArgs, d_indptr: List[_NPtr], d_indices: List[_NPtr]):
    """The k-faces of top simplices `i` to `i + W - 1`, from the rows of d,
    as the general loop of `whitney_innerproduct` finds them: level by
    level, the `W` simplices' rows together so that their loads overlap.
    Each face is read from one row only, that of `_first_container`, and
    rows naming no face that way are skipped."""
    comptime w = p + 1
    comptime size = 1 << w
    comptime nkf = _binom(p + 1, k + 1)
    var gp = args.gidx
    comptime for l in range(W):
        gp[unsafe_offset=l * size + size - 1] = i + l
    comptime for jj in range(p - k):
        comptime L = p - jj
        var ip = d_indptr[L - 1]
        var ix = d_indices[L - 1]
        comptime for M in range(size):
            # a row without position 0 names no face by _first_container
            comptime if _bit_count(M) == L + 1 and M & 1 == 1:
                comptime for l in range(W):
                    var base = ip[unsafe_offset=gp[unsafe_offset=l * size + M]]
                    comptime for q in range(L + 1):
                        comptime face = M ^ (1 << _nth_bit(M, L - q))
                        comptime if _first_container(w, face) == M:
                            gp[unsafe_offset=l * size + face] = ix[
                                unsafe_offset=base + q
                            ]
    comptime for a in range(nkf):
        comptime mask = _comb_mask(w, k + 1, a)
        comptime for l in range(W):
            args.faces[unsafe_offset=(i + l) * nkf + a] = gp[
                unsafe_offset=l * size + mask
            ]


def _first_container(w: Int, F: Int) -> Int:
    """The face mask whose row of d `_element_faces` reads face mask `F`
    from: `F` with its lowest missing position (below `w`) added, so it
    always contains position 0."""
    for b in range(w):
        if (F >> b) & 1 == 0:
            return F | (1 << b)
    return -1


@always_inline
def _whitney_element[
    p: Int, dim: Int, k: Int, W: Int
](
    i: Int, args: _ElementArgs, d_indptr: List[_NPtr], d_indices: List[_NPtr]
) -> Bool:
    """The k-faces and local mass matrices of top simplices `i` to
    `i + W - 1`, as the general loop of `whitney_innerproduct` computes
    them, with the dimensions known at compile time: the loops unroll and
    the small matrices stay on the stack, one SIMD lane per simplex. The
    arithmetic is the general loop's, operation for operation (lane by
    lane: the row pivoting of `_minor_det` becomes selects); the
    multiply-adds the compiler fuses there (in `_householder_rows`, the
    back substitution of `frame_dual_basis`, `_minor_det`, and the sums of
    pair determinants and of terms) are explicit fused multiply-adds here,
    so the results are the same bit for bit, whatever `W`. Returns False
    if any of the simplices is degenerate."""
    comptime V = SIMD[DType.float64, W]
    comptime w = p + 1
    comptime nkf = _binom(p + 1, k + 1)
    comptime nfp = nkf * nkf
    comptime nforms = _binom(p + 1, k)
    comptime nsets = _binom(p, k)
    comptime T = (k + 1) * (k + 1)
    _element_faces[p, k, W](i, args, d_indptr, d_indices)
    var dets = stack_allocation[nforms * nforms * W, DType.float64]()
    comptime if k == 0:
        _st[W](dets, 0, V(1.0))
    else:
        # edges from vertex 0, then Householder QR as _householder_rows
        var A = stack_allocation[p * dim * W, DType.float64]()
        comptime for r in range(p):
            comptime for c in range(dim):
                var e = V()
                comptime for l in range(W):
                    var v0 = Int(args.S[unsafe_offset=(i + l) * w])
                    var v = Int(args.S[unsafe_offset=(i + l) * w + r + 1])
                    e[l] = (
                        args.X[unsafe_offset=v * dim + c]
                        - args.X[unsafe_offset=v0 * dim + c]
                    )
                _st[W](A, r * dim + c, e)
        var diag = stack_allocation[p * W, DType.float64]()
        comptime for j in range(p):
            var norm2 = V(0)
            comptime for c in range(j, dim):
                var x = _ld[W](A, j * dim + c)
                norm2 = x.fma(x, norm2)
            var norm = sqrt(norm2)
            if norm.eq(0).reduce_or():
                return False
            var x0 = _ld[W](A, j * dim + j)
            var alpha = x0.ge(0).select(-norm, norm)
            _st[W](diag, j, alpha)
            _st[W](A, j * dim + j, x0 - alpha)
            var vv = V(0)
            comptime for c in range(j, dim):
                var x = _ld[W](A, j * dim + c)
                vv = x.fma(x, vv)
            comptime for l in range(j + 1, p):
                var dot = V(0)
                comptime for c in range(j, dim):
                    dot = _ld[W](A, j * dim + c).fma(
                        _ld[W](A, l * dim + c), dot
                    )
                var f = 2 * dot / vv
                comptime for c in range(j, dim):
                    _st[W](
                        A,
                        l * dim + c,
                        (-f).fma(
                            _ld[W](A, j * dim + c), _ld[W](A, l * dim + c)
                        ),
                    )
        # R^-1 by back substitution, as frame_dual_basis
        var Rinv = stack_allocation[p * p * W, DType.float64]()
        comptime for c in range(p):
            comptime for ii in range(p):
                comptime r = p - 1 - ii
                comptime if r <= c:
                    var t = V(1 if r == c else 0)
                    comptime for kk in range(r + 1, c + 1):
                        t = (-_ld[W](A, kk * dim + r)).fma(
                            _ld[W](Rinv, kk * p + c), t
                        )
                    _st[W](Rinv, r * p + c, t / _ld[W](diag, r))
                else:
                    _st[W](Rinv, r * p + c, V(0))
        # the gradients in the frame: rows 1..p of R^-1, row 0 minus their sum
        var D = stack_allocation[(p + 1) * p * W, DType.float64]()
        comptime for c in range(p):
            var t = V(0)
            comptime for r in range(p):
                var g = _ld[W](Rinv, r * p + c)
                _st[W](D, (r + 1) * p + c, g)
                t += g
            _st[W](D, c, -t)
        var minors = stack_allocation[nforms * nsets * W, DType.float64]()
        comptime for a in range(nforms):
            comptime for q in range(nsets):
                _st[W](minors, a * nsets + q, _minor_fixed[p, k, a, q, W](D))
        comptime for a in range(nforms):
            comptime for b in range(a, nforms):
                var det = V(0)
                comptime for q in range(nsets):
                    det = _ld[W](minors, a * nsets + q).fma(
                        _ld[W](minors, b * nsets + q), det
                    )
                _st[W](dets, a * nforms + b, det)
                _st[W](dets, b * nforms + a, det)
    var vol = V()
    comptime for l in range(W):
        vol[l] = args.vol[unsafe_offset=i + l]
    var weight = abs(vol) * args.scale

    @always_inline
    def entry[t: Int]() {dets, weight} -> SIMD[DType.float64, W]:
        var s = V(0)
        comptime for q in range(T):
            comptime coef = _term_coef(p, k, t, q)
            comptime pair = _term_pair(p, k, t, q)
            s = V(coef).fma(_ld[W](dets, pair), s)
        return s * weight

    comptime if W == 4 and nfp % 4 == 0:
        # four entries at a time, transposed to four simplices' rows
        comptime for g in range(nfp // 4):
            var x0 = rebind[SIMD[DType.float64, 4]](entry[4 * g]())
            var x1 = rebind[SIMD[DType.float64, 4]](entry[4 * g + 1]())
            var x2 = rebind[SIMD[DType.float64, 4]](entry[4 * g + 2]())
            var x3 = rebind[SIMD[DType.float64, 4]](entry[4 * g + 3]())
            var a = x0.shuffle[0, 4, 2, 6](x1)
            var b = x0.shuffle[1, 5, 3, 7](x1)
            var c = x2.shuffle[0, 4, 2, 6](x3)
            var d = x2.shuffle[1, 5, 3, 7](x3)
            var at = i * nfp + 4 * g
            args.data.unsafe_store(at, a.shuffle[0, 1, 4, 5](c))
            args.data.unsafe_store(at + nfp, b.shuffle[0, 1, 4, 5](d))
            args.data.unsafe_store(at + 2 * nfp, a.shuffle[2, 3, 6, 7](c))
            args.data.unsafe_store(at + 3 * nfp, b.shuffle[2, 3, 6, 7](d))
    else:
        comptime for t in range(nfp):
            var x = entry[t]()
            comptime for l in range(W):
                args.data[unsafe_offset=(i + l) * nfp + t] = x[l]
    return True


@always_inline
def _minor_fixed[
    p: Int, k: Int, form: Int, axes: Int, W: Int
](D: _FPtr) -> SIMD[DType.float64, W]:
    """`_minor_det` of the `(p + 1) x p` gradients `D` (`W`-lane vectors)
    for a degree `k` known at compile time, with its fused multiply-adds
    explicit and its row pivoting done by selects: the minor in the rows of
    k-form `form` (vertex positions, in the order of `combinations`) and
    the columns of k-subset `axes` of the frame's axes."""
    comptime V = SIMD[DType.float64, W]
    comptime N = SIMD[DType.int, W]

    @always_inline
    def g[a: Int, b: Int]() {D} -> SIMD[DType.float64, W]:
        comptime row = _nth_bit(_comb_mask(p + 1, k, form), a)
        comptime col = _nth_bit(_comb_mask(p, k, axes), b)
        return _ld[W](D, row * p + col)

    comptime if k == 1:
        return g[0, 0]()
    elif k == 2:
        return g[0, 0]().fma(g[1, 1](), -(g[0, 1]() * g[1, 0]()))
    else:
        var work = stack_allocation[k * k * W, DType.float64]()
        comptime for a in range(k):
            comptime for b in range(k):
                _st[W](work, a * k + b, g[a, b]())
        var det = V(1)
        var zero = SIMD[DType.bool, W](fill=False)
        comptime for c in range(k):
            # the first row from c with the largest magnitude in column c
            var piv = N(c)
            var best = abs(_ld[W](work, c * k + c))
            comptime for r in range(c + 1, k):
                var x = abs(_ld[W](work, r * k + c))
                var m = x.gt(best)
                piv = m.select(N(r), piv)
                best = m.select(x, best)
            det = piv.ne(N(c)).select(-det, det)
            comptime for r in range(c + 1, k):
                var m = piv.eq(N(r))
                comptime for b in range(k):
                    var x = _ld[W](work, c * k + b)
                    var y = _ld[W](work, r * k + b)
                    _st[W](work, c * k + b, m.select(y, x))
                    _st[W](work, r * k + b, m.select(x, y))
            var pv = _ld[W](work, c * k + c)
            zero |= pv.eq(0)
            det *= pv
            comptime for r in range(c + 1, k):
                var f = _ld[W](work, r * k + c) / pv
                comptime for b in range(c + 1, k):
                    _st[W](
                        work,
                        r * k + b,
                        (-f).fma(
                            _ld[W](work, c * k + b), _ld[W](work, r * k + b)
                        ),
                    )
        return zero.select(V(0), det)


def _binom(n: Int, r: Int) -> Int:
    """`C(n, r)`, 1 when `r <= 0` (as `combinations` yields one empty
    combination)."""
    if r <= 0:
        return 1
    if r > n:
        return 1
    var c = 1
    for j in range(r):
        c = c * (n - j) // (j + 1)
    return c


def _bit_count(M: Int) -> Int:
    """The number of set bits of `M`."""
    var c = 0
    for j in range(64):
        c += (M >> j) & 1
    return c


def _nth_bit(M: Int, r: Int) -> Int:
    """The position of the `r`-th lowest set bit of `M`."""
    var c = 0
    for j in range(64):
        if (M >> j) & 1 == 1:
            if c == r:
                return j
            c += 1
    return -1


def _lex_less(A: Int, B: Int) -> Bool:
    """Whether the sorted positions of mask `A` come before those of mask
    `B` (of the same size) in lexicographic order."""
    var a = A
    var b = B
    while a != 0 and b != 0:
        var la = _nth_bit(a, 0)
        var lb = _nth_bit(b, 0)
        if la != lb:
            return la < lb
        a &= a - 1
        b &= b - 1
    return False


def _comb_rank(n: Int, M: Int) -> Int:
    """The index of mask `M` (positions below `n`) among the combinations of
    its size, in the order of `combinations`."""
    var r = 0
    for M2 in range(1 << n):
        if _bit_count(M2) == _bit_count(M) and _lex_less(M2, M):
            r += 1
    return r


def _comb_mask(n: Int, r: Int, idx: Int) -> Int:
    """The mask of combination `idx` of `r` of the positions `0..n-1`, in
    the order of `combinations` (0 when `r <= 0`)."""
    for M in range(1 << n):
        if _bit_count(M) == max(r, 0) and _comb_rank(n, M) == idx:
            return M
    return -1


def _term_parts(p: Int, k: Int, t: Int, q: Int) -> Tuple[Int, Int]:
    """Term `q` of local entry `t`, as `whitney_innerproduct`'s tables have
    it: the index `a * num_forms + b` of its pair determinant, and its
    coefficient."""
    var nkf = _binom(p + 1, k + 1)
    var nforms = _binom(p + 1, k)
    var M1 = _comb_mask(p + 1, k + 1, t // nkf)
    var M2 = _comb_mask(p + 1, k + 1, t % nkf)
    var n = q // (k + 1)
    var m = q % (k + 1)
    var v1 = _nth_bit(M1, n)
    var v2 = _nth_bit(M2, m)
    var a = _comb_rank(p + 1, M1 ^ (1 << v1))
    var b = _comb_rank(p + 1, M2 ^ (1 << v2))
    var coef = (2 if v1 == v2 else 1) * (1 if (n + m) % 2 == 0 else -1)
    return (a * nforms + b, coef)


def _term_pair(p: Int, k: Int, t: Int, q: Int) -> Int:
    """The pair-determinant index of term `q` of local entry `t`."""
    return _term_parts(p, k, t, q)[0]


def _term_coef(p: Int, k: Int, t: Int, q: Int) -> Float64:
    """The coefficient of term `q` of local entry `t`."""
    return Float64(_term_parts(p, k, t, q)[1])


def _assemble_codim1[
    F: Int
](
    N: Int,
    fps_: Int,
    faces: List[Int],
    values: List[Float64],
    cofaces: CSRMatrix,
) -> Optional[CSRMatrix]:
    """`_assemble` for the faces of codimension 1, from the top simplices
    containing each (the rows of `cofaces`, the top level's boundary).
    In a pseudomanifold a face lies in one or two top simplices, which
    share no other face, so its row has `1 + (fps - 1) m` entries for `m`
    top simplices: their faces merged, with the face itself summed
    `v_1 + v_2` in increasing `s`. The same matrix as `_assemble`, bit for
    bit; None if some face lies in more than two top simplices. `F` is
    `fps` when it is known at compile time, 0 otherwise. Each top
    simplex's faces are in increasing order (its vertices are sorted, and
    its faces listed in lexicographic order)."""
    var fps = F if F > 0 else fps_
    var cp = cofaces.indptr.unsafe_ptr()
    var cx = cofaces.indices.unsafe_ptr()
    var indptr = List[Int](unsafe_uninit_length=N + 1)
    var pp = indptr.unsafe_ptr()
    pp[unsafe_offset=0] = 0
    for r in range(N):
        var m = cp[unsafe_offset=r + 1] - cp[unsafe_offset=r]
        if m < 1 or m > 2:
            return None
        pp[unsafe_offset=r + 1] = pp[unsafe_offset=r] + 1 + (fps - 1) * m
    var nnz = pp[unsafe_offset=N]
    var indices = List[Int](unsafe_uninit_length=nnz)
    var data = List[Float64](unsafe_uninit_length=nnz)
    var ixp = indices.unsafe_ptr()
    var dp = data.unsafe_ptr()
    var fp = faces.unsafe_ptr()
    var vp = values.unsafe_ptr()

    def rows(chunk: Int, lo: Int, hi: Int) {fps, cp, cx, pp, ixp, dp, fp, vp}:
        for r in range(lo, hi):
            var j = cp[unsafe_offset=r]
            var base = pp[unsafe_offset=r]
            var A = cx[unsafe_offset=j] * fps
            var a1 = 0
            for b in range(fps):
                a1 += b * Int(fp[unsafe_offset=A + b] == r)
            var V1 = (A + a1) * fps
            if cp[unsafe_offset=r + 1] - j == 1:
                for b in range(fps):
                    ixp[unsafe_offset=base + b] = fp[unsafe_offset=A + b]
                    dp[unsafe_offset=base + b] = vp[unsafe_offset=V1 + b]
                continue
            var B = cx[unsafe_offset=j + 1] * fps
            var a2 = 0
            for b in range(fps):
                a2 += b * Int(fp[unsafe_offset=B + b] == r)
            var V2 = (B + a2) * fps
            # a column's place: its place in its own simplex, plus the
            # smaller columns of the other, counting the shared face once;
            # the second simplex first, so that the first overwrites the
            # shared face's entry
            for b in range(fps):
                var x = fp[unsafe_offset=B + b]
                var at = base + b - Int(r < x)
                for q in range(fps):
                    at += Int(fp[unsafe_offset=A + q] < x)
                ixp[unsafe_offset=at] = x
                dp[unsafe_offset=at] = vp[unsafe_offset=V2 + b]
            var shared = vp[unsafe_offset=V1 + a1] + vp[unsafe_offset=V2 + a2]
            for b in range(fps):
                var x = fp[unsafe_offset=A + b]
                var at = base + b - Int(r < x)
                for q in range(fps):
                    at += Int(fp[unsafe_offset=B + q] < x)
                ixp[unsafe_offset=at] = x
                dp[unsafe_offset=at] = (
                    shared if b == a1 else vp[unsafe_offset=V1 + b]
                )

    parallel_ranges(N, 4096, rows)
    var M = CSRMatrix(N, N)
    M.indptr = indptr^
    M.indices = indices^
    M.data = data^
    return M^


struct _Incidence(Movable):
    """The top simplices containing each face, as `_incidence` finds them."""

    var start: List[Int]
    """Face `r`'s entries are `entries[start[r]:start[r + 1]]`."""
    var entries: List[Int32]
    """Entry `s * fps + a` for face `faces[s * fps + a]`, in increasing
    order within each face."""

    def __init__(out self, var start: List[Int], var entries: List[Int32]):
        self.start = start^
        self.entries = entries^


def _incidence(N: Int, fps: Int, faces: List[Int]) -> _Incidence:
    """The top simplices containing each of the `N` faces, by a counting
    sort of `faces`, stable so that each face's entries are in increasing
    order. On several cores: each chunk of entries counts its faces by
    block of faces, places its entries into their blocks after those of
    the chunks before it, and each block is then sorted by face."""
    var total = len(faces)
    var fp = faces.unsafe_ptr()
    var start = List[Int](length=N + 1, fill=0)
    var sp = start.unsafe_ptr()
    var entries = List[Int32](unsafe_uninit_length=total)
    var ep = entries.unsafe_ptr()
    comptime SERIAL_BELOW = 1 << 17
    var T = num_chunks(total, SERIAL_BELOW)
    if T <= 1:
        for e in range(total):
            var r = fp[unsafe_offset=e]
            sp[unsafe_offset=r + 1] = sp[unsafe_offset=r + 1] + 1
        for r in range(N):
            sp[unsafe_offset=r + 1] = (
                sp[unsafe_offset=r + 1] + sp[unsafe_offset=r]
            )
        var next = start.copy()
        var np = next.unsafe_ptr()
        for e in range(total):
            var r = fp[unsafe_offset=e]
            var at = np[unsafe_offset=r]
            ep[unsafe_offset=at] = Int32(e)
            np[unsafe_offset=r] = at + 1
        return _Incidence(start^, entries^)
    # blocks of 2^shift faces, about one per chunk
    var shift = 0
    while (N >> shift) >= T:
        shift += 1
    var B = (N >> shift) + 1
    var offsets = List[Int](length=T * B, fill=0)
    var op = offsets.unsafe_ptr()

    def count(chunk: Int, lo: Int, hi: Int) {fp, op, B, shift}:
        var h = chunk_scratch[Int](B, 0)
        var hp = h.unsafe_ptr()
        for e in range(lo, hi):
            var b = fp[unsafe_offset=e] >> shift
            hp[unsafe_offset=b] = hp[unsafe_offset=b] + 1
        for b in range(B):
            op[unsafe_offset=chunk * B + b] = hp[unsafe_offset=b]

    parallel_ranges(total, SERIAL_BELOW, count)
    # block b's entries start at bstart[b], chunk by chunk
    var bstart = List[Int](length=B + 1, fill=0)
    var at = 0
    for b in range(B):
        bstart[b] = at
        for c in range(T):
            var n = op[unsafe_offset=c * B + b]
            op[unsafe_offset=c * B + b] = at
            at += n
    bstart[B] = at
    var by_block = List[Int32](unsafe_uninit_length=total)
    var bp = by_block.unsafe_ptr()

    def place(chunk: Int, lo: Int, hi: Int) {fp, op, bp, B, shift}:
        var nx = chunk_scratch[Int](B, 0)
        var np = nx.unsafe_ptr()
        for b in range(B):
            np[unsafe_offset=b] = op[unsafe_offset=chunk * B + b]
        for e in range(lo, hi):
            var b = fp[unsafe_offset=e] >> shift
            var a = np[unsafe_offset=b]
            bp[unsafe_offset=a] = Int32(e)
            np[unsafe_offset=b] = a + 1

    parallel_ranges(total, SERIAL_BELOW, place)
    var bsp = bstart.unsafe_ptr()

    def sort_blocks(
        chunk: Int, lo: Int, hi: Int
    ) {N, fp, sp, ep, bp, bsp, shift}:
        var nx = chunk_scratch[Int](1 << shift, 0)
        var np = nx.unsafe_ptr()
        for b in range(lo, hi):
            var r0 = b << shift
            var r1 = min(N, (b + 1) << shift)
            var e0 = bsp[unsafe_offset=b]
            var e1 = bsp[unsafe_offset=b + 1]
            for r in range(r1 - r0):
                np[unsafe_offset=r] = 0
            for q in range(e0, e1):
                var r = fp[unsafe_offset=Int(bp[unsafe_offset=q])] - r0
                np[unsafe_offset=r] = np[unsafe_offset=r] + 1
            # each face's start; then the next free place of each
            var at = e0
            for r in range(r1 - r0):
                var n = np[unsafe_offset=r]
                sp[unsafe_offset=r0 + r] = at
                np[unsafe_offset=r] = at
                at += n
            for q in range(e0, e1):
                var e = bp[unsafe_offset=q]
                var r = fp[unsafe_offset=Int(e)] - r0
                var a = np[unsafe_offset=r]
                ep[unsafe_offset=a] = e
                np[unsafe_offset=r] = a + 1

    parallel_ranges(B, 2, sort_blocks)
    sp[unsafe_offset=N] = total
    return _Incidence(start^, entries^)


def _assemble(
    N: Int, fps: Int, faces: List[Int], values: List[Float64]
) raises -> CSRMatrix:
    """The `N x N` matrix of the local matrices of the top simplices:
    top simplex `s` adds `values[(s * fps + a) * fps + b]` at `(faces[s *
    fps + a], faces[s * fps + b])`.

    The same matrix, bit for bit, as `CSRMatrix.from_coo` of the triplets
    listed simplex-major (`massmatrix_rowcols`' order): each entry sums its
    terms in increasing `s`, starting from the first. Rows are assembled
    in parallel, each from the top simplices containing its face. Faces
    of codimension 1 take `_assemble_codim1` instead.
    """
    if len(faces) > Int(Int32.MAX) or N > Int(Int32.MAX):
        raise Error("whitney_innerproduct: too many faces for 32-bit indices")
    var fp = faces.unsafe_ptr()
    var vp = values.unsafe_ptr()
    # the top simplices containing each face, in increasing order: entry
    # s * fps + a for face faces[s * fps + a]
    var inc = _incidence(N, fps, faces)
    var sp = inc.start.unsafe_ptr()
    var ip = inc.entries.unsafe_ptr()
    # Two passes over the rows, in parallel: the first counts each row's
    # distinct columns, the second sums its entries and writes them, sorted
    # by column, straight into the result; the terms of each entry are
    # added in increasing s, as from_coo's stable sort orders them. A
    # column is found in the row through a per-chunk array `slot[c]`. The
    # count marks it with the row, without a branch (about half the terms
    # bring a new column, and the branch mispredicted). The fill keeps the
    # column's place in the row (-1 when absent) and branches: without the
    # branch each term's place waited on the last store, and it ran slower.
    var max_terms = 0
    for r in range(N):
        max_terms = max(
            max_terms, (sp[unsafe_offset=r + 1] - sp[unsafe_offset=r]) * fps
        )
    var counts = List[Int](unsafe_uninit_length=N)
    var cnp = counts.unsafe_ptr()
    comptime SERIAL_BELOW = 4096

    def count_rows(chunk: Int, lo: Int, hi: Int) {N, fps, fp, sp, ip, cnp}:
        var slot = chunk_scratch[Int32](N, -1)
        var slp = slot.unsafe_ptr()
        for r in range(lo, hi):
            var n = 0
            var mark = Int32(r)
            for j in range(sp[unsafe_offset=r], sp[unsafe_offset=r + 1]):
                var s = Int(ip[unsafe_offset=j]) // fps
                for b in range(fps):
                    var c = fp[unsafe_offset=s * fps + b]
                    n += Int(slp[unsafe_offset=c] != mark)
                    slp[unsafe_offset=c] = mark
            cnp[unsafe_offset=r] = n

    parallel_ranges(N, SERIAL_BELOW, count_rows)
    var indptr = List[Int](unsafe_uninit_length=N + 1)
    indptr[0] = 0
    var pp = indptr.unsafe_ptr()
    for r in range(N):
        pp[unsafe_offset=r + 1] = pp[unsafe_offset=r] + cnp[unsafe_offset=r]
    var nnz = indptr[N]
    var indices = List[Int](unsafe_uninit_length=nnz)
    var data = List[Float64](unsafe_uninit_length=nnz)
    var ixp = indices.unsafe_ptr()
    var dp = data.unsafe_ptr()

    def fill_rows(
        chunk: Int, lo: Int, hi: Int
    ) {N, fps, fp, vp, sp, ip, pp, ixp, dp, max_terms}:
        var slot = chunk_scratch[Int32](N, -1)
        var cols = chunk_scratch[Int](max_terms, 0)
        var vals = chunk_scratch[Float64](max_terms, 0)
        var slp = slot.unsafe_ptr()
        var cp = cols.unsafe_ptr()
        var vlp = vals.unsafe_ptr()
        for r in range(lo, hi):
            var n = 0
            for j in range(sp[unsafe_offset=r], sp[unsafe_offset=r + 1]):
                var e = Int(ip[unsafe_offset=j])
                var s = e // fps
                for b in range(fps):
                    var c = fp[unsafe_offset=s * fps + b]
                    var v = vp[unsafe_offset=e * fps + b]
                    var at = Int(slp[unsafe_offset=c])
                    if at < 0:
                        slp[unsafe_offset=c] = Int32(n)
                        cp[unsafe_offset=n] = c
                        vlp[unsafe_offset=n] = v
                        n += 1
                    else:
                        vlp[unsafe_offset=at] = vlp[unsafe_offset=at] + v
            for t in range(n):
                slp[unsafe_offset=cp[unsafe_offset=t]] = -1
            # sort the row by column (insertion: rows are short) into place
            var dst = pp[unsafe_offset=r]
            for t in range(n):
                var c = cp[unsafe_offset=t]
                var v = vlp[unsafe_offset=t]
                var at = t
                while at > 0 and ixp[unsafe_offset=dst + at - 1] > c:
                    ixp[unsafe_offset=dst + at] = ixp[
                        unsafe_offset=dst + at - 1
                    ]
                    dp[unsafe_offset=dst + at] = dp[unsafe_offset=dst + at - 1]
                    at -= 1
                ixp[unsafe_offset=dst + at] = c
                dp[unsafe_offset=dst + at] = v

    parallel_ranges(N, SERIAL_BELOW, fill_rows)
    _ = inc^
    _ = counts^
    # canonical by construction (sorted, no duplicates)
    var A = CSRMatrix(N, N)
    A.indptr = indptr^
    A.indices = indices^
    A.data = data^
    return A^


def regular_cube_innerproduct(
    rcc: RegularCubeComplex, k: Int
) raises -> CSRMatrix:
    """The k-form inner product matrix on a regular cube complex.

    The elements are the cubical analogue of Whitney forms (multilinear for
    0-forms).

    Args:
        rcc: A regular cube complex.
        k: The form degree, `0 <= k <= complex_dimension()`.

    Returns:
        The symmetric `N_k x N_k` inner product matrix.

    Raises:
        If `k > complex_dimension()`, or a k-face of a top cube is missing
            from `rcc[k]`.
    """
    var N = rcc.complex_dimension()
    # the standard cube [0,...,0, 0,1,...,N-1] and its k-faces
    var std_row = List[Int]()
    for _ in range(N):
        std_row.append(0)
    for i in range(N):
        std_row.append(i)
    var standard = int_array_from_rows([std_row^], 2 * N)
    for i in range(N, k, -1):
        standard = cube_array_boundary(standard, i).faces.copy()
    var kfpc = rows_of(standard)
    var width = cols_of(standard)

    # local stiffness matrix
    var K = real_array(kfpc, kfpc)
    var scale = (1.0 / 3.0) ** (N - k)
    for i in range(kfpc):
        for j in range(kfpc):
            var same_axes = True
            for c in range(N, width):
                if iget(standard, i, c) != iget(standard, j, c):
                    same_axes = False
            if not same_axes:
                continue
            var spanned = List[Bool](length=N, fill=False)
            for c in range(N, width):
                spanned[iget(standard, i, c)] = True
            var diffs = 0
            for c in range(N):
                if not spanned[c] and iget(standard, i, c) != iget(
                    standard, j, c
                ):
                    diffs += 1
            K.unsafe_set(i * kfpc + j, scale * 0.5**diffs)

    ref CA = rcc[N].cube_array
    var num_cubes = rows_of(CA)
    var k_faces = int_array(num_cubes * kfpc, width)
    for c in range(num_cubes):
        for f in range(kfpc):
            var r = c * kfpc + f
            for j in range(width):
                var v = iget(standard, f, j)
                if j < N:
                    v += iget(CA, c, j)
                k_faces.unsafe_set(r * width + j, Scalar[DType.int](v))
    var idx = cube_array_search(rcc[k].cube_array, k_faces)
    var rows = List[Int]()
    var cols = List[Int]()
    var vals = List[Float64]()
    for c in range(num_cubes):
        for f in range(kfpc):
            for g in range(kfpc):
                var v = K.unsafe_get(f * kfpc + g)
                if v != 0:
                    rows.append(Int(idx.unsafe_get(c * kfpc + f)))
                    cols.append(Int(idx.unsafe_get(c * kfpc + g)))
                    vals.append(v)
    var n = rows_of(rcc[k].cube_array)
    return CSRMatrix.from_coo(n, n, rows, cols, vals)


def _without(xs: List[Int], n: Int) -> List[Int]:
    var out = List[Int](capacity=len(xs))
    for i in range(len(xs)):
        if i != n:
            out.append(xs[i])
    return out^


def _form_index(forms: List[List[Int]], form: List[Int]) raises -> Int:
    """The position of `form` in `forms`."""
    for i in range(len(forms)):
        if forms[i] == form:
            return i
    raise Error("whitney_innerproduct: k-form not found")


def _minor_det(
    mat: List[Float64],
    ncols: Int,
    row_list: List[Int],
    rows: Int,
    col_list: List[Int],
    cols: Int,
    k: Int,
    mut scratch: List[Float64],
) -> Float64:
    """The determinant of the k x k block of the row-major matrix `mat`
    (`ncols` columns) in rows `row_list[rows:rows + k]` and columns
    `col_list[cols:cols + k]` (1 when k = 0); `scratch` holds `k * k`
    values.
    """
    var G = mat.unsafe_ptr()
    var rl = row_list.unsafe_ptr()
    var cl = col_list.unsafe_ptr()
    var work = scratch.unsafe_ptr()

    @always_inline
    def g(a: Int, b: Int) {G, ncols, rl, cl, rows, cols} -> Float64:
        return G[
            unsafe_offset=rl[unsafe_offset=rows + a] * ncols
            + cl[unsafe_offset=cols + b]
        ]

    if k == 0:
        return 1
    if k == 1:
        return g(0, 0)
    if k == 2:
        return g(0, 0) * g(1, 1) - g(0, 1) * g(1, 0)
    # Gaussian elimination with partial pivoting: the cofactor expansion
    # loses about a digit on sliver tetrahedra at k = 3
    for a in range(k):
        for b in range(k):
            work[unsafe_offset=a * k + b] = g(a, b)
    var det: Float64 = 1
    for c in range(k):
        var piv = c
        for r in range(c + 1, k):
            if abs(work[unsafe_offset=r * k + c]) > abs(
                work[unsafe_offset=piv * k + c]
            ):
                piv = r
        var pv = work[unsafe_offset=piv * k + c]
        if pv == 0:
            return 0
        if piv != c:
            det = -det
            for b in range(k):
                var t = work[unsafe_offset=c * k + b]
                work[unsafe_offset=c * k + b] = work[unsafe_offset=piv * k + b]
                work[unsafe_offset=piv * k + b] = t
        det *= pv
        for r in range(c + 1, k):
            var f = work[unsafe_offset=r * k + c] / pv
            for b in range(c + 1, k):
                work[unsafe_offset=r * k + b] -= (
                    f * work[unsafe_offset=c * k + b]
                )
    return det
