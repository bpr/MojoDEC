# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Simplex arrays (`mojodec.routines.simplex_array`).
==================================================
Largely inspired by `pydec.dec.simplex_array`. A simplex array is an
`N x (k+1)` integer array whose rows are the vertex lists of `N` k-simplices.
"""

from numojo.prelude import Shape
from mojodec.core.parallel import num_chunks, parallel_ranges
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    _bounds,
    compare_rows,
    int_array,
    int_vector,
    lexsort_rows,
    require_distinct_simplices,
    rows_of,
    cols_of,
    sort_each_row,
)


def simplex_array_searchsorted(s: IntArray, v: IntArray) raises -> IntArray:
    """Row indices in `s` of the simplices stored in the rows of `v`.

    The rows of `s` must be in lexicographic order; each lookup is a
    binary search.

    ```mojo
    # s = [[0,1],[0,2],[1,2],[1,3]], v = [[1,2],[0,2]]  ->  [2, 1]
    ```

    Args:
        s: Simplices in lexicographic row order.
        v: Simplices to look up, with the same number of columns.

    Returns:
        For each row of `v`, the index of the equal row of `s`.

    Raises:
        If either array is not 2-D, their column counts differ, or a row of
            `v` does not occur in `s`.
    """
    if s.ndim != 2 or v.ndim != 2:
        raise Error("simplex_array_searchsorted: expected rank 2 arrays")
    if cols_of(s) != cols_of(v):
        raise Error("simplex_array_searchsorted: number of columns must agree")
    var ns = rows_of(s)
    var nv = rows_of(v)
    var c = cols_of(s)
    var out = int_vector(nv)
    # The rows with leading vertex f are rows start[f]:start[f + 1] of s, so
    # each search covers only those (a vertex's few simplices). Skipped for
    # negative or very large vertex labels, where the table would not fit.
    var start = List[Int]()
    var first = Int(s.unsafe_get(0)) if ns > 0 and c > 0 else -1
    var last = Int(s.unsafe_get((ns - 1) * c)) if ns > 0 and c > 0 else -1
    if last > 4 * ns + 1024:
        first = -1
    if first >= 0:
        start = List[Int](length=last + 2, fill=0)
        for r in range(ns):
            start[Int(s.unsafe_get(r * c)) + 1] += 1
        for f in range(last + 1):
            start[f + 1] += start[f]
    for i in range(nv):
        var lo = 0
        var hi = ns
        if first >= 0:
            var f = Int(v.unsafe_get(i * c))
            if f < 0 or f > last:
                raise Error("simplex_array_searchsorted: simplex not found")
            lo = start.unsafe_get(f)
            hi = start.unsafe_get(f + 1)
        var end = hi
        while lo < hi:
            var mid = (lo + hi) // 2
            if compare_rows(s, mid, v, i) < 0:
                lo = mid + 1
            else:
                hi = mid
        if lo >= end or compare_rows(s, lo, v, i) != 0:
            raise Error("simplex_array_searchsorted: simplex not found")
        out.unsafe_set(i, Scalar[DType.int](lo))
    return out^


def simplex_array_parity(s: IntArray) raises -> IntArray:
    """Parity of each row of `s` relative to its sorted order (0 or 1).

    Args:
        s: An `N x (k+1)` simplex array with distinct entries in each row.

    Returns:
        A length-`N` vector: 0 where sorting the row takes an even
        permutation, 1 where it takes an odd one.

    Raises:
        Does not raise in practice.
    """
    var M = rows_of(s)
    var N = cols_of(s)
    var out = int_vector(M)
    var sp = s.unsafe_ptr()
    var op = out.unsafe_ptr()

    def rows(chunk: Int, lo: Int, hi: Int) {sp, op, N}:
        # the parity of the number of pairs out of order, counted with no
        # branch to mispredict
        if N == 3:
            for i in range(lo, hi):
                var x0 = sp[unsafe_offset=3 * i]
                var x1 = sp[unsafe_offset=3 * i + 1]
                var x2 = sp[unsafe_offset=3 * i + 2]
                var inversions = Int(x1 < x0) + Int(x2 < x0) + Int(x2 < x1)
                op[unsafe_offset=i] = Scalar[DType.int](inversions % 2)
            return
        if N == 4:
            for i in range(lo, hi):
                var x0 = sp[unsafe_offset=4 * i]
                var x1 = sp[unsafe_offset=4 * i + 1]
                var x2 = sp[unsafe_offset=4 * i + 2]
                var x3 = sp[unsafe_offset=4 * i + 3]
                var inversions = (
                    Int(x1 < x0)
                    + Int(x2 < x0)
                    + Int(x3 < x0)
                    + Int(x2 < x1)
                    + Int(x3 < x1)
                    + Int(x3 < x2)
                )
                op[unsafe_offset=i] = Scalar[DType.int](inversions % 2)
            return
        for i in range(lo, hi):
            var base = i * N
            var inversions = 0
            for j in range(N - 1):
                var x = sp[unsafe_offset=base + j]
                for k in range(j + 1, N):
                    inversions += Int(sp[unsafe_offset=base + k] < x)
            op[unsafe_offset=i] = Scalar[DType.int](inversions % 2)

    parallel_ranges(M, 50_000, rows)
    return out^


struct FacesAndBoundary(Copyable, Movable):
    """The faces of an array of simplices or cubes, and its boundary
    operator."""

    var faces: IntArray
    """The unique faces, one per row."""
    var boundary: CSRMatrix
    """The `num_faces x num_cells` boundary operator."""
    var coboundary: CSRMatrix
    """Its transpose, `num_cells x num_faces`: the coboundary `d` on the
    faces."""

    def __init__(
        out self,
        var faces: IntArray,
        var boundary: CSRMatrix,
        var coboundary: CSRMatrix,
    ):
        """A result from its parts.

        Args:
            faces: The faces.
            boundary: The boundary operator.
            coboundary: Its transpose.
        """
        self.faces = faces^
        self.boundary = boundary^
        self.coboundary = coboundary^

    def __init__(out self, var faces: IntArray, var boundary: CSRMatrix) raises:
        """A result from the faces and the boundary operator; the
        coboundary is its transpose.

        Args:
            faces: The faces.
            boundary: The boundary operator.

        Raises:
            Does not raise in practice.
        """
        self.coboundary = boundary.transpose()
        self.faces = faces^
        self.boundary = boundary^


def simplex_array_boundary(
    s: IntArray, parity: IntArray
) raises -> FacesAndBoundary:
    """Boundary faces and boundary operator of an array of sorted simplices.

    `s` must have sorted rows; `parity` gives each simplex's orientation
    relative to that order. Returns the unique faces (sorted, in
    lexicographic order) and the `num_faces x num_simplices` boundary matrix.

    ```mojo
    # simplex_array_boundary([[0,1,2],[1,2,3]], [0,1])
    ```

    Args:
        s: `N x (k+1)` simplices with sorted rows.
        parity: Length-`N` orientations, 0 or 1.

    Returns:
        The unique (k-1)-faces in lexicographic order, and the
        `num_faces x N` boundary matrix.

    Raises:
        Does not raise in practice.
    """
    # From a few thousand simplices with vertex ids in a narrow range, as a
    # complex's levels have, the faces are sorted by buckets, in parallel;
    # the result is the same either way.
    if cols_of(s) >= 2 and rows_of(s) * cols_of(s) >= _BUCKETS_FROM:
        var V = _vertex_bound(s)
        if V > 0:
            return _boundary_by_buckets(s, parity, V)
    return _boundary_by_lexsort(s, parity)


comptime _BUCKETS_FROM = 16384
"""Face occurrences (simplices x faces per simplex) from which
`simplex_array_boundary` sorts by buckets."""


def _boundary_by_lexsort(
    s: IntArray, parity: IntArray
) raises -> FacesAndBoundary:
    """`simplex_array_boundary` by one lexicographic sort of every face."""
    var num_simplices = rows_of(s)
    var faces_per_simplex = cols_of(s)
    var num_faces = num_simplices * faces_per_simplex
    var fw = faces_per_simplex - 1
    # all faces, with the owning simplex and orientation alongside
    var faces = int_array(num_faces, fw)
    var owner = List[Int](length=num_faces, fill=0)
    var orient = List[Float64](length=num_faces, fill=0)
    for i in range(faces_per_simplex):
        for r in range(num_simplices):
            var row = num_simplices * i + r
            var c = 0
            for j in range(faces_per_simplex):
                if j != i:
                    faces.unsafe_set(
                        row * fw + c, s.unsafe_get(r * faces_per_simplex + j)
                    )
                    c += 1
            owner[row] = r
            var o = 1 - 2 * Int(parity.unsafe_get(r))
            orient[row] = Float64(o if i % 2 == 0 else -o)

    var perm = lexsort_rows(faces)
    var unique_rows = List[Int]()
    var face_index = List[Int](length=num_faces, fill=0)
    for t in range(num_faces):
        var p = perm[t]
        if t == 0 or compare_rows(faces, p, faces, perm[t - 1]) != 0:
            unique_rows.append(p)
        face_index[t] = len(unique_rows) - 1

    var unique_faces = int_array(len(unique_rows), fw)
    for i in range(len(unique_rows)):
        for j in range(fw):
            unique_faces.unsafe_set(
                i * fw + j, faces.unsafe_get(unique_rows[i] * fw + j)
            )

    # d (simplices x faces) has k+1 entries per row; visiting the faces in
    # sorted order fills each row's columns in increasing order
    var indptr = List[Int](capacity=num_simplices + 1)
    for r in range(num_simplices + 1):
        indptr.append(r * faces_per_simplex)
    var indices = List[Int](length=num_faces, fill=0)
    var data = List[Float64](length=num_faces, fill=0)
    var filled = List[Int](length=num_simplices, fill=0)
    for t in range(num_faces):
        var p = perm.unsafe_get(t)
        var r = owner.unsafe_get(p)
        var at = r * faces_per_simplex + filled.unsafe_get(r)
        indices.unsafe_set(at, face_index.unsafe_get(t))
        data.unsafe_set(at, orient.unsafe_get(p))
        filled.unsafe_set(r, filled.unsafe_get(r) + 1)
    var d = CSRMatrix(num_simplices, len(unique_rows), indptr^, indices^, data^)
    var B = d.transpose()
    return FacesAndBoundary(unique_faces^, B^, d^)


comptime _IntPtr = Pointer[Scalar[DType.int], ImmUntrackedOrigin]
comptime _Ptr = Pointer[Int, MutUntrackedOrigin]


def _vertex_bound(s: IntArray) -> Int:
    """One more than the largest vertex id of `s`, or -1 if an id is
    negative or the range is wider than the number of entries (then one
    bucket per id would cost more than it saves)."""
    if s.size == 0:
        return 0
    var lo = 0
    var hi = -1
    try:
        _bounds(s, lo, hi)
    except:
        return -1
    if lo < 0:
        return -1
    if hi + 1 > s.size + 1024:
        return -1
    return hi + 1


@always_inline
def _scatter_packed[
    K: Int
](
    sp: _IntPtr,
    pp: _IntPtr,
    cp: _Ptr,
    stp: _Ptr,
    op: _Ptr,
    base: Int,
    lo: Int,
    hi: Int,
    sh: Int,
    bV: Int,
    kE: Int,
):
    """`_boundary_by_buckets`'s scatter of packed keys for rows of `K`
    entries, unrolled: the same keys in the same places. Occurrences
    1..K-1 of a row all go to the bucket of its first entry, in order."""
    for r in range(lo, hi):
        comptime assert K <= 4
        var x = SIMD[DType.int, 4](0)
        comptime for j in range(K):
            x[j] = sp[unsafe_offset=r * K + j]
        var par = Int(pp[unsafe_offset=r])
        # occurrence 0, in the bucket of the row's second entry
        var x0 = Int(x[0])
        var x1 = Int(x[1])
        var at0 = stp[unsafe_offset=x1] + cp[unsafe_offset=base + x1]
        cp[unsafe_offset=base + x1] += 1
        var key0 = 0
        comptime for j in range(2, K):
            key0 = (key0 << bV) | Int(x[j])
        op[unsafe_offset=at0] = (key0 << kE) | ((r << sh) << 1) | (par & 1)
        # the others, in the bucket of its first
        var at = stp[unsafe_offset=x0] + cp[unsafe_offset=base + x0]
        cp[unsafe_offset=base + x0] += K - 1
        comptime for i in range(1, K):
            var key = 0
            comptime for j in range(1, K):
                comptime if j != i:
                    key = (key << bV) | Int(x[j])
            op[unsafe_offset=at + i - 1] = (
                (key << kE) | (((r << sh) | i) << 1) | ((par + i) & 1)
            )


@always_inline
def _runs_packed[
    K: Int
](
    op: _Ptr,
    stp: _Ptr,
    dsp: _Ptr,
    fp: Pointer[Scalar[DType.int], MutUntrackedOrigin],
    bip: _Ptr,
    bix: _Ptr,
    bdv: Pointer[Float64, MutUntrackedOrigin],
    dix: _Ptr,
    v0: Int,
    v1: Int,
    bV: Int,
    kE: Int,
):
    """`_boundary_by_buckets`'s pass over the sorted packed keys of buckets
    `v0..v1`, for rows of `K` entries, unrolled: the faces, the boundary,
    and the column indices of d."""
    comptime fw = K - 1
    comptime sh = 1 if K <= 2 else 2
    var e_mask = (1 << kE) - 1
    var v_mask = (1 << bV) - 1
    for v in range(v0, v1):
        var f = dsp[unsafe_offset=v] - 1
        for t in range(stp[unsafe_offset=v], stp[unsafe_offset=v + 1]):
            var x = op[unsafe_offset=t]
            var key = x & (_NEW_FACE - 1)
            var occ_e = (key & e_mask) >> 1
            if x != key:
                f += 1
                fp[unsafe_offset=f * fw] = Scalar[DType.int](v)
                # the face's entries after the first, bV bits each, the
                # last lowest
                var rest = key >> kE
                comptime for j in range(fw - 1, 0, -1):
                    fp[unsafe_offset=f * fw + j] = Scalar[DType.int](
                        rest & v_mask
                    )
                    rest >>= bV
                bip[unsafe_offset=f] = t
            var r = occ_e >> sh
            var i = occ_e & ((1 << sh) - 1)
            bix[unsafe_offset=t] = r
            bdv[unsafe_offset=t] = Float64(1 - 2 * (key & 1))
            dix[unsafe_offset=r * K + K - 1 - i] = f


@always_inline
def _d_rows[
    K: Int
](
    pp: _IntPtr,
    dip: _Ptr,
    ddv: Pointer[Float64, MutUntrackedOrigin],
    lo: Int,
    hi: Int,
):
    """Row pointers and values of d for rows `lo..hi` of `K` entries: the
    face without entry i, at `K - 1 - i`, has sign (-1)^i times the
    simplex's orientation."""
    for r in range(lo, hi):
        var o = Float64(1 - 2 * Int(pp[unsafe_offset=r]))
        dip[unsafe_offset=r] = r * K
        comptime for i in range(K):
            ddv[unsafe_offset=r * K + K - 1 - i] = o if i % 2 == 0 else -o


def _face_entry(sp: _IntPtr, K: Int, sh: Int, e: Int, j: Int) -> Int:
    """Entry `j` of the face of occurrence `e = (r << sh) | i`: row `r` of
    the simplices without its entry `i`."""
    var r = e >> sh
    var i = e & ((1 << sh) - 1)
    return Int(sp[unsafe_offset=r * K + (j if j < i else j + 1)])


def _occurrence_less(sp: _IntPtr, K: Int, sh: Int, a: Int, b: Int) -> Bool:
    """Whether occurrence `a` sorts before `b` in their bucket: by the face's
    entries after the first (the bucket's), then by simplex."""
    for j in range(1, K - 1):
        var x = _face_entry(sp, K, sh, a, j)
        var y = _face_entry(sp, K, sh, b, j)
        if x != y:
            return x < y
    return (a >> sh) < (b >> sh)


def _same_face(sp: _IntPtr, K: Int, sh: Int, a: Int, b: Int) -> Bool:
    """Whether occurrences `a` and `b` of one bucket have the same face."""
    for j in range(1, K - 1):
        if _face_entry(sp, K, sh, a, j) != _face_entry(sp, K, sh, b, j):
            return False
    return True


def _sort_occurrences(
    sp: _IntPtr, K: Int, sh: Int, op: _Ptr, lo: Int, hi: Int, tmp: _Ptr
):
    """Sort occurrences `op[lo:hi]` with `_occurrence_less` (merge sort over
    insertion-sorted runs; `tmp` holds at least `(hi - lo) / 2 + 1`)."""
    if hi - lo <= 24:
        for a in range(lo + 1, hi):
            var key = op[unsafe_offset=a]
            var b = a - 1
            while b >= lo and _occurrence_less(
                sp, K, sh, key, op[unsafe_offset=b]
            ):
                op[unsafe_offset=b + 1] = op[unsafe_offset=b]
                b -= 1
            op[unsafe_offset=b + 1] = key
        return
    var mid = (lo + hi) // 2
    _sort_occurrences(sp, K, sh, op, lo, mid, tmp)
    _sort_occurrences(sp, K, sh, op, mid, hi, tmp)
    if not _occurrence_less(
        sp, K, sh, op[unsafe_offset=mid], op[unsafe_offset=mid - 1]
    ):
        return
    var n = mid - lo
    for t in range(n):
        tmp[unsafe_offset=t] = op[unsafe_offset=lo + t]
    var a = 0
    var b = mid
    var out = lo
    while a < n and b < hi:
        if _occurrence_less(
            sp, K, sh, op[unsafe_offset=b], tmp[unsafe_offset=a]
        ):
            op[unsafe_offset=out] = op[unsafe_offset=b]
            b += 1
        else:
            op[unsafe_offset=out] = tmp[unsafe_offset=a]
            a += 1
        out += 1
    while a < n:
        op[unsafe_offset=out] = tmp[unsafe_offset=a]
        a += 1
        out += 1


def _sort_keys(op: _Ptr, lo: Int, hi: Int, tmp: _Ptr):
    """Sort the distinct integers `op[lo:hi]`, all below `Int.MAX`.

    Up to `_INSERTION_SORT_TO` keys by insertion; up to `_RANK_SORT_TO` by
    rank; longer runs by merge sort over halves. `tmp` holds at least
    `max((hi - lo) / 2 + 1, _RANK_SORT_TO)` entries.
    """
    var n0 = hi - lo
    if n0 <= _INSERTION_SORT_TO:
        for a in range(lo + 1, hi):
            var key = op[unsafe_offset=a]
            var b = a - 1
            while b >= lo and key < op[unsafe_offset=b]:
                op[unsafe_offset=b + 1] = op[unsafe_offset=b]
                b -= 1
            op[unsafe_offset=b + 1] = key
        return
    if n0 <= _RANK_SORT_TO:
        # each key's place is the number of smaller keys (they are
        # distinct): more comparisons than insertion sort, but in SIMD and
        # with no branch to mispredict, so the cost does not depend on how
        # the keys arrive; padded with Int.MAX, which counts for none
        comptime W = 4
        var padded = (n0 + W - 1) // W * W
        for a in range(n0):
            tmp[unsafe_offset=a] = op[unsafe_offset=lo + a]
        for a in range(n0, padded):
            tmp[unsafe_offset=a] = Int.MAX
        var tv = rebind[Pointer[Scalar[DType.int], MutUntrackedOrigin]](tmp)
        for a in range(n0):
            var key = tmp[unsafe_offset=a]
            var kv = SIMD[DType.int, W](key)
            var acc = SIMD[DType.int, W](0)
            for b in range(0, padded, W):
                var v = tv.unsafe_load[width=W](b)
                acc += v.lt(kv).cast[DType.int]()
            op[unsafe_offset=lo + Int(acc.reduce_add())] = key
        return
    var mid = (lo + hi) // 2
    _sort_keys(op, lo, mid, tmp)
    _sort_keys(op, mid, hi, tmp)
    if op[unsafe_offset=mid - 1] < op[unsafe_offset=mid]:
        return
    var n = mid - lo
    for t in range(n):
        tmp[unsafe_offset=t] = op[unsafe_offset=lo + t]
    var a = 0
    var b = mid
    var out = lo
    while a < n and b < hi:
        var x = op[unsafe_offset=b]
        var y = tmp[unsafe_offset=a]
        if x < y:
            op[unsafe_offset=out] = x
            b += 1
        else:
            op[unsafe_offset=out] = y
            a += 1
        out += 1
    while a < n:
        op[unsafe_offset=out] = tmp[unsafe_offset=a]
        a += 1
        out += 1


comptime _INSERTION_SORT_TO = 16
"""Runs up to this long are sorted by insertion in `_sort_keys`: few
comparisons, so mispredicted ones cost little."""
comptime _RANK_SORT_TO = 64
"""Longer runs up to this long are sorted by rank in `_sort_keys` (a
multiple of the SIMD width 4). With ranking instead of insertion above
16, the topology of `tet_40` on one thread takes 125 ms instead of 133 in
Morton order and 238 instead of 264 numbered at random, but 120 instead
of 114 in grid order."""


def _bit_length(x: Int) -> Int:
    """The number of bits of `x >= 0` (0 for 0)."""
    var n = 0
    while n < 63 and (1 << n) <= x:
        n += 1
    return n


comptime _NEW_FACE = 1 << 62
"""Marks an occurrence that starts a run of equal faces in its bucket."""


def _boundary_by_buckets(
    s: IntArray, parity: IntArray, V: Int, pack: Bool = True
) raises -> FacesAndBoundary:
    """`simplex_array_boundary` with the faces sorted by buckets, in parallel.

    Occurrence `e = (r << sh) | i` is the face of simplex `r` without its
    entry `i` (a shift, not `r * K + i`, as divisions by `K` dominated the
    sort); its first vertex is `s[r, 0]`, or `s[r, 1]` when `i = 0`. A
    counting sort puts the occurrences into one bucket per first vertex,
    each bucket is sorted by the rest of the face and then by simplex, and
    the runs of equal faces are the unique faces, in lexicographic order.
    A run's position is its row's start in the boundary matrix, whose
    entries are then in increasing simplex order, as a transpose gives;
    each row of `d` is the simplex's faces sorted by index. The result is
    `_boundary_by_lexsort`'s, entry for entry.

    When they fit in 62 bits, each occurrence is sorted as one integer key:
    the face's entries after the first, `bV` bits each, then `e`, then a
    bit set when its boundary entry is -1 (so that `runs` need not look up
    the simplex's parity). Keys
    order as `_occurrence_less` does (by the rest of the face, then by
    simplex), and a bucket sorts with integer comparisons instead of
    rebuilding both faces from `s` at every one. Otherwise (or with
    `pack=False`, for tests) the occurrences sort with
    `_occurrence_less`. Either way, sorting marks each run's first
    occurrence with `_NEW_FACE`.
    """
    var N = rows_of(s)
    var K = cols_of(s)
    var fw = K - 1
    var M = N * K
    var sp = s.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
    var pp = parity.unsafe_ptr()
    comptime SERIAL_BELOW = 4096
    var C = num_chunks(N, SERIAL_BELOW)
    var sh = 1
    while (1 << sh) < K:
        sh += 1
    var bV = _bit_length(V - 1)
    var bE = _bit_length(N - 1) + sh
    # packed keys end in e and the sign bit of the occurrence's entry in
    # the boundary matrix
    var packed = pack and (K - 2) * bV + bE + 1 <= 62
    var kE = bE + 1
    var e_mask = (1 << kE) - 1

    # the buckets each chunk of simplices puts occurrences into lie between
    # its smallest first vertex and its largest second one (the rows are
    # sorted): a narrow band when the numbering is spatially coherent
    var band_lo = List[Int](unsafe_uninit_length=C)
    var band_hi = List[Int](unsafe_uninit_length=C)
    var blp = band_lo.unsafe_ptr()
    var bhp = band_hi.unsafe_ptr()

    def bands(chunk: Int, lo: Int, hi: Int) {sp, blp, bhp, K}:
        var a = Int(sp[unsafe_offset=lo * K])
        var b = Int(sp[unsafe_offset=lo * K + 1])
        for r in range(lo + 1, hi):
            a = min(a, Int(sp[unsafe_offset=r * K]))
            b = max(b, Int(sp[unsafe_offset=r * K + 1]))
        blp[unsafe_offset=chunk] = a
        bhp[unsafe_offset=chunk] = b + 1

    if C > 1:
        parallel_ranges(N, SERIAL_BELOW, bands)
    else:
        band_lo[0] = 0
        band_hi[0] = V
    # occurrences per (chunk of simplices, first vertex in its band); the
    # counter of chunk c's bucket v is at band_base[c] + v
    var band_base = List[Int](unsafe_uninit_length=C)
    var total_band = 0
    for c in range(C):
        band_base[c] = total_band - band_lo[c]
        total_band += band_hi[c] - band_lo[c]
    var counts = List[Int](unsafe_uninit_length=total_band)
    var cp = counts.unsafe_ptr()
    var bbp = band_base.unsafe_ptr()

    def count(chunk: Int, lo: Int, hi: Int) {sp, cp, blp, bhp, bbp, K}:
        var base = bbp[unsafe_offset=chunk]
        for v in range(blp[unsafe_offset=chunk], bhp[unsafe_offset=chunk]):
            cp[unsafe_offset=base + v] = 0
        for r in range(lo, hi):
            cp[unsafe_offset=base + Int(sp[unsafe_offset=r * K])] += K - 1
            cp[unsafe_offset=base + Int(sp[unsafe_offset=r * K + 1])] += 1

    parallel_ranges(N, SERIAL_BELOW, count)
    # bucket v holds occurrences start[v]:start[v + 1]; within it, chunk c
    # writes from start[v] + counts[band_base[c] + v] on
    var start = List[Int](length=V + 1, fill=0)
    var stp = start.unsafe_ptr()

    def totals(chunk: Int, lo: Int, hi: Int) {cp, stp, blp, bhp, bbp, C}:
        for v in range(lo, hi):
            var t = 0
            for c in range(C):
                if blp[unsafe_offset=c] <= v < bhp[unsafe_offset=c]:
                    var at = bbp[unsafe_offset=c] + v
                    var x = cp[unsafe_offset=at]
                    cp[unsafe_offset=at] = t
                    t += x
            stp[unsafe_offset=v + 1] = t

    parallel_ranges(V, 65536, totals)
    for v in range(V):
        stp[unsafe_offset=v + 1] += stp[unsafe_offset=v]
    var occ = List[Int](unsafe_uninit_length=M)
    var op = occ.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    var ppi = parity.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
    var cpm = cp.unsafe_origin_cast[MutUntrackedOrigin]()
    var stpm = stp.unsafe_origin_cast[MutUntrackedOrigin]()

    def scatter(
        chunk: Int, lo: Int, hi: Int
    ) {sp, pp, ppi, cp, cpm, stp, stpm, op, bbp, K, sh, bV, kE, packed}:
        var base = bbp[unsafe_offset=chunk]
        if packed and K == 3:
            _scatter_packed[3](sp, ppi, cpm, stpm, op, base, lo, hi, sh, bV, kE)
            return
        if packed and K == 4:
            _scatter_packed[4](sp, ppi, cpm, stpm, op, base, lo, hi, sh, bV, kE)
            return
        if packed and K == 2:
            _scatter_packed[2](sp, ppi, cpm, stpm, op, base, lo, hi, sh, bV, kE)
            return
        for r in range(lo, hi):
            for i in range(K):
                var v = Int(sp[unsafe_offset=r * K + (1 if i == 0 else 0)])
                var at = stp[unsafe_offset=v] + cp[unsafe_offset=base + v]
                cp[unsafe_offset=base + v] = cp[unsafe_offset=base + v] + 1
                var e = (r << sh) | i
                if packed:
                    # the face's entries after its first, then e, then
                    # whether the boundary entry is -1
                    var key = 0
                    for j in range(2 if i == 0 else 1, K):
                        if j != i:
                            key = (key << bV) | Int(sp[unsafe_offset=r * K + j])
                    var odd = (Int(pp[unsafe_offset=r]) + i) & 1
                    e = (key << kE) | (e << 1) | odd
                op[unsafe_offset=at] = e

    parallel_ranges(N, SERIAL_BELOW, scatter)
    _ = counts^
    _ = band_lo^
    _ = band_hi^
    _ = band_base^

    # blocks of buckets with about equal numbers of occurrences
    var blocks = List[Int](length=C + 1, fill=V)
    blocks[0] = 0
    var c = 1
    for v in range(V):
        while c < C and stp[unsafe_offset=v] * C >= c * M:
            blocks[c] = v
            c += 1
    var bp = blocks.unsafe_ptr()
    # sort each bucket and count its distinct faces
    var distinct = List[Int](length=V + 1, fill=0)
    var dsp = distinct.unsafe_ptr()

    def sort_buckets(
        chunk: Int, lo: Int, hi: Int
    ) {sp, stp, op, bp, dsp, K, sh, kE, packed}:
        for b in range(lo, hi):
            var widest = 0
            for v in range(bp[unsafe_offset=b], bp[unsafe_offset=b + 1]):
                widest = max(
                    widest, stp[unsafe_offset=v + 1] - stp[unsafe_offset=v]
                )
            var tmp = List[Int](
                length=max(widest // 2 + 1, _RANK_SORT_TO), fill=0
            )
            var tp = tmp.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            for v in range(bp[unsafe_offset=b], bp[unsafe_offset=b + 1]):
                var a = stp[unsafe_offset=v]
                var e = stp[unsafe_offset=v + 1]
                var n = 0
                if packed:
                    # a bucket's keys arrive in increasing order of their
                    # occurrence, so with K = 2, faces with no entries
                    # after the first, they are already sorted
                    if K != 2:
                        _sort_keys(op, a, e, tp)
                    # run starts marked; the keys stay, and `runs` reads
                    # the faces from them
                    var prev = -1
                    for t in range(a, e):
                        var key = op[unsafe_offset=t]
                        if t == a or (key >> kE) != prev:
                            op[unsafe_offset=t] = key | _NEW_FACE
                            n += 1
                        prev = key >> kE
                else:
                    _sort_occurrences(sp, K, sh, op, a, e, tp)
                    var prev = -1
                    for t in range(a, e):
                        var occ_e = op[unsafe_offset=t]
                        if t == a or not _same_face(sp, K, sh, prev, occ_e):
                            op[unsafe_offset=t] = occ_e | _NEW_FACE
                            n += 1
                        prev = occ_e
                dsp[unsafe_offset=v + 1] = n
            _ = tmp^

    parallel_ranges(C, 0, sort_buckets)
    for v in range(V):
        dsp[unsafe_offset=v + 1] += dsp[unsafe_offset=v]
    var F = distinct[V]

    # the unique faces, the boundary (one row per run), and d. A sorted
    # simplex's faces, without entry i, are in decreasing lexicographic
    # order of i (without a later entry, a face has the smaller entry at
    # the first place they differ), so row r of d holds the face without
    # entry i at r * K + K - 1 - i, in increasing face index.
    # every entry is written in the parallel loops below, which then also
    # touch the pages first
    var faces = IntArray(Shape(F, fw))
    var fp = faces.unsafe_ptr()
    var b_indptr = List[Int](unsafe_uninit_length=F + 1)
    b_indptr[F] = M
    var b_indices = List[Int](unsafe_uninit_length=M)
    var b_data = List[Float64](unsafe_uninit_length=M)
    var d_indptr = List[Int](unsafe_uninit_length=N + 1)
    var d_indices = List[Int](unsafe_uninit_length=M)
    var d_data = List[Float64](unsafe_uninit_length=M)
    var bip = b_indptr.unsafe_ptr()
    var bix = b_indices.unsafe_ptr()
    var bdv = b_data.unsafe_ptr()
    var dip = d_indptr.unsafe_ptr()
    var dix = d_indices.unsafe_ptr()
    var unrolled = packed and 2 <= K <= 4
    var ddv = d_data.unsafe_ptr()

    var fpm = fp.unsafe_origin_cast[MutUntrackedOrigin]()
    var bipm = bip.unsafe_origin_cast[MutUntrackedOrigin]()
    var bixm = bix.unsafe_origin_cast[MutUntrackedOrigin]()
    var bdvm = bdv.unsafe_origin_cast[MutUntrackedOrigin]()
    var dixm = dix.unsafe_origin_cast[MutUntrackedOrigin]()
    var dspm = dsp.unsafe_origin_cast[MutUntrackedOrigin]()

    def runs(
        chunk: Int, lo: Int, hi: Int
    ) {
        sp,
        pp,
        stp,
        stpm,
        op,
        bp,
        dsp,
        dspm,
        fp,
        fpm,
        bip,
        bipm,
        bix,
        bixm,
        bdv,
        bdvm,
        dix,
        dixm,
        unrolled,
        K,
        fw,
        sh,
        bV,
        kE,
        e_mask,
        packed,
    }:
        if unrolled:
            var v0 = bp[unsafe_offset=lo]
            var v1 = bp[unsafe_offset=hi]
            if K == 3:
                _runs_packed[3](
                    op, stpm, dspm, fpm, bipm, bixm, bdvm, dixm, v0, v1, bV, kE
                )
            elif K == 4:
                _runs_packed[4](
                    op, stpm, dspm, fpm, bipm, bixm, bdvm, dixm, v0, v1, bV, kE
                )
            else:
                _runs_packed[2](
                    op, stpm, dspm, fpm, bipm, bixm, bdvm, dixm, v0, v1, bV, kE
                )
            return
        for b in range(lo, hi):
            for v in range(bp[unsafe_offset=b], bp[unsafe_offset=b + 1]):
                var f = dsp[unsafe_offset=v] - 1
                for t in range(stp[unsafe_offset=v], stp[unsafe_offset=v + 1]):
                    var x = op[unsafe_offset=t]
                    var key = x & (_NEW_FACE - 1)
                    var occ_e = (key & e_mask) >> 1 if packed else key
                    if x != key:
                        f += 1
                        fp[unsafe_offset=f * fw] = Scalar[DType.int](v)
                        if packed:
                            # the face's entries after the first, bV bits
                            # each, the last lowest
                            var rest = key >> kE
                            for j in range(fw - 1, 0, -1):
                                fp[unsafe_offset=f * fw + j] = Scalar[
                                    DType.int
                                ](rest & ((1 << bV) - 1))
                                rest >>= bV
                        else:
                            for j in range(1, fw):
                                fp[unsafe_offset=f * fw + j] = Scalar[
                                    DType.int
                                ](_face_entry(sp, K, sh, occ_e, j))
                        bip[unsafe_offset=f] = t
                    var r = occ_e >> sh
                    var i = occ_e & ((1 << sh) - 1)
                    var odd = (
                        key
                        & 1 if packed else (Int(pp[unsafe_offset=r]) + i)
                        & 1
                    )
                    var sign = Float64(1 - 2 * odd)
                    bix[unsafe_offset=t] = r
                    bdv[unsafe_offset=t] = sign
                    dix[unsafe_offset=r * K + K - 1 - i] = f

    parallel_ranges(C, 0, runs)
    _ = occ^
    _ = start^
    _ = blocks^
    _ = distinct^

    # d's values in simplex order: the face without entry i has sign
    # (-1)^i, times the simplex's orientation
    var ppr = parity.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()
    var dipm = dip.unsafe_origin_cast[MutUntrackedOrigin]()
    var ddvm = ddv.unsafe_origin_cast[MutUntrackedOrigin]()

    def rows(chunk: Int, lo: Int, hi: Int) {ppr, dipm, ddvm, K, N}:
        if K == 3:
            _d_rows[3](ppr, dipm, ddvm, lo, hi)
        elif K == 4:
            _d_rows[4](ppr, dipm, ddvm, lo, hi)
        elif K == 2:
            _d_rows[2](ppr, dipm, ddvm, lo, hi)
        else:
            for r in range(lo, hi):
                var o = Float64(1 - 2 * Int(ppr[unsafe_offset=r]))
                dipm[unsafe_offset=r] = r * K
                for i in range(K):
                    ddvm[unsafe_offset=r * K + K - 1 - i] = (
                        o if i % 2 == 0 else -o
                    )
        if hi == N:
            dipm[unsafe_offset=N] = N * K

    parallel_ranges(N, SERIAL_BELOW, rows)
    # canonical by construction: sorted columns, no duplicates
    var d = CSRMatrix(N, F)
    d.indptr = d_indptr^
    d.indices = d_indices^
    d.data = d_data^
    var B = CSRMatrix(F, N)
    B.indptr = b_indptr^
    B.indices = b_indices^
    B.data = b_data^
    return FacesAndBoundary(faces^, B^, d^)


def _row_string(a: IntArray, i: Int) -> String:
    var out = String("[")
    var c = cols_of(a)
    for j in range(c):
        if j > 0:
            out += ", "
        out += String(Int(a.unsafe_get(i * c + j)))
    return out + "]"


def _element_signs(
    boundary: CSRMatrix, coboundary: CSRMatrix
) -> Optional[List[Int]]:
    """Signs making the elements coherent, or None if there are none.

    `boundary` is the `num_faces x num_elements` boundary of the elements,
    with at most two entries per row; `coboundary` its transpose. The first
    element of each piece connected through faces keeps its sign.
    """
    var N = boundary.ncols
    var sign = List[Int](length=N, fill=0)
    var stack = List[Int]()
    for root in range(N):
        if sign[root] != 0:
            continue
        sign[root] = 1
        stack.append(root)
        while len(stack) > 0:
            var t = stack.pop()
            for p in range(coboundary.indptr[t], coboundary.indptr[t + 1]):
                var f = coboundary.indices[p]
                var lo = boundary.indptr[f]
                if boundary.indptr[f + 1] - lo != 2:
                    continue
                var a = boundary.indices[lo]
                var u = boundary.indices[lo + 1] if a == t else a
                # s_t B[f,t] + s_u B[f,u] = 0, entries +-1
                var want = -sign[t] * Int(
                    boundary.data[lo] * boundary.data[lo + 1]
                )
                if sign[u] == 0:
                    sign[u] = want
                    stack.append(u)
                elif sign[u] != want:
                    return None
    return sign^


def _require_pseudomanifold(
    faces: IntArray, boundary: CSRMatrix, who: String
) raises:
    for f in range(boundary.nrows):
        var count = boundary.indptr[f + 1] - boundary.indptr[f]
        if count > 2:
            raise Error(
                who,
                ": not a pseudomanifold: the face ",
                _row_string(faces, f),
                " is shared by ",
                count,
                " elements",
            )


def require_coherent_orientation(
    faces: IntArray, boundary: CSRMatrix, coboundary: CSRMatrix, who: String
) raises:
    """Raise unless the elements form a coherently oriented pseudomanifold.

    Args:
        faces: The `(n-1)`-faces of the elements, sorted rows.
        boundary: The `num_faces x num_elements` boundary of the elements,
            signed by their orientations (`simplex_array_boundary`).
        coboundary: Its transpose.
        who: The caller, named in the error message.

    Raises:
        If a face is shared by three or more elements, or two elements
        induce the same orientation on their shared face (the message says
        whether the mesh is orientable at all).
    """
    # one parallel pass: each chunk's first face with three or more
    # elements, and its first face two elements orient the same way
    var F = boundary.nrows
    comptime SERIAL_BELOW = 50_000
    var chunks = num_chunks(F, SERIAL_BELOW)
    var crowded = List[Int](length=chunks, fill=-1)
    var same = List[Int](length=chunks, fill=-1)
    var ip = boundary.indptr.unsafe_ptr()
    var dp = boundary.data.unsafe_ptr()
    var cp = crowded.unsafe_ptr()
    var sp = same.unsafe_ptr()

    def faces_of_chunk(chunk: Int, lo: Int, hi: Int) {ip, dp, cp, sp}:
        for f in range(lo, hi):
            var a = ip[unsafe_offset=f]
            var count = ip[unsafe_offset=f + 1] - a
            if count > 2:
                cp[unsafe_offset=chunk] = f
                return
            if (
                count == 2
                and sp[unsafe_offset=chunk] < 0
                and dp[unsafe_offset=a] + dp[unsafe_offset=a + 1] != 0
            ):
                sp[unsafe_offset=chunk] = f

    parallel_ranges(F, SERIAL_BELOW, faces_of_chunk)
    for c in range(chunks):
        if crowded[c] >= 0:
            _require_pseudomanifold(faces, boundary, who)
    for c in range(chunks):
        var f = same[c]
        if f < 0:
            continue
        var lo = boundary.indptr[f]
        var t = boundary.indices[lo]
        var u = boundary.indices[lo + 1]
        if not _element_signs(boundary, coboundary):
            raise Error(
                who,
                ": the mesh is not orientable (elements ",
                t,
                " and ",
                u,
                " induce the same orientation on their face ",
                _row_string(faces, f),
                ")",
            )
        raise Error(
            who,
            ": elements ",
            t,
            " and ",
            u,
            " induce the same orientation on their face ",
            _row_string(faces, f),
            "; orient_coherently(mesh) reorients the elements",
        )


def coherent_orientation(elements: IntArray) raises -> Optional[List[Int]]:
    """Signs that make a mesh's elements coherently oriented.

    Coherent means two elements sharing an (n-1)-face induce opposite
    orientations on it, as `SimplicialComplex` requires. Walks from element
    to element across shared faces; the first element of each piece
    connected through faces keeps its orientation.

    ```mojo
    # coherent_orientation([[0,1,2],[1,2,3]])  ->  [1, 1]
    # coherent_orientation([[0,1,2],[2,1,3]])  ->  [1, -1]
    ```

    Args:
        elements: `N x (n+1)` vertex lists; their order gives the
            orientation.

    Returns:
        One sign per element, `1` to keep its orientation and `-1` to
        reverse it, or `None` if no choice is coherent (e.g. a Möbius band).
        All ones for points (n = 0).

    Raises:
        If an element repeats a vertex, two elements have the same
        vertices, or a face is shared by three or more elements.
    """
    var N = rows_of(elements)
    if cols_of(elements) < 2:
        return List[Int](length=N, fill=1)
    var s = elements.copy()
    var parity = simplex_array_parity(s)
    sort_each_row(s)
    require_distinct_simplices(s, "coherent_orientation")
    var fb = simplex_array_boundary(s, parity)
    _require_pseudomanifold(fb.faces, fb.boundary, "coherent_orientation")
    return _element_signs(fb.boundary, fb.coboundary)
