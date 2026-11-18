# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
# The supernode partition, the relaxed amalgamation (its thresholds), the row
# patterns by `ereach_super`, and the left-looking numeric factorization with
# one dense update per descendant supernode follow faer's
# `sparse::linalg::cholesky::supernodal` (faer 0.24, commit 0ecd846, MIT
# License, Copyright (c) 2026 sarah quiñones el kazdadi); see NOTICE. The
# dense kernels are MojoDEC's own. The algorithms: J. W. H. Liu, E. G. Ng,
# B. W. Peyton, "On finding supernodes for sparse matrix computations", SIAM
# J. Matrix Anal. Appl. 14(1), 1993; C. Ashcraft, R. Grimes, "The influence
# of relaxed supernode partitions on the multifrontal method", ACM Trans.
# Math. Softw. 15(4), 1989; E. G. Ng, B. W. Peyton, "Block sparse Cholesky
# algorithms on advanced uniprocessor computers", SIAM J. Sci. Comput.
# 14(5), 1993; for the subtrees given to threads, G. A. Geist, E. G. Ng,
# "Task scheduling for parallel sparse Cholesky factorization", Int. J.
# Parallel Programming 18(4), 1989. The task graph is MojoDEC's own.
# ===----------------------------------------------------------------------=== #
"""
Supernodal LDL^T (`mojodec.routines.supernodal`).
==================================================
The numeric engine behind `SparseLDLT(A, layout=FactorLayout.SUPERNODAL)`.
Columns of `L` with the same pattern below a dense triangle (a supernode)
are stored together as one dense column-major block, so the factorization
and the solves run dense kernels (vector loads, one pass over each block)
instead of one sparse column at a time. Private: use `SparseLDLT`.

The factorization runs on every physical core (`MOJODEC_NUM_THREADS` caps
them) as a list of tasks: disjoint subtrees of the supernode tree, then
each supernode above them, split when large into windows of columns for
its updates and, per panel, a diagonal-block task and tasks of 64 rows
below it. Threads claim the tasks in order and wait for their
predecessors, all in one parallel call. Every entry gets the same
operations in the same order as serially, so the factor's bits do not
depend on the number of threads.
"""

from std.atomic import Atomic
from std.math import isfinite

from mojodec.core.parallel import max_threads, parallel_ranges

comptime _NONE = -1
comptime _W = 8
"""SIMD width of the dense kernels (two AVX2 registers, or one AVX-512)."""
comptime _PANEL = 32
"""Columns per panel of a supernode's dense factorization."""


def _sub_product(
    o: Pointer[mut=True, Float64, ...],
    o_stride: Int,
    a: Pointer[Float64, ...],
    a_stride: Int,
    w: Pointer[Float64, ...],
    k: Int,
    m: Int,
    nc: Int,
    top: Int = 0,
):
    """`O[r, c] -= sum_t A[r, t] W[t, c]` for `r + top >= c` (the lower
    trapezoid of a block whose first row is `top` rows below the diagonal;
    entries above the diagonal are left alone), with `O` and `A`
    column-major (`O[r, c]` at `o[c * o_stride + r]`, `A[r, t]` at
    `a[t * a_stride + r]`), `W` row-major `k x nc`, and `m` rows.

    Register tiles of `_W` rows by 4 columns: per step of `t`, one vector
    load of `A` and four fused multiply-adds. Each entry is the same
    chain of fused multiply-adds, in `t` order, whatever the tiling, so a
    block computed in pieces of rows has the same bits."""
    comptime V = SIMD[DType.float64, _W]
    var c0 = 0
    while c0 + 4 <= nc:
        var r0 = max(c0 - top, 0)
        while r0 + _W <= m:
            var acc0 = V(0)
            var acc1 = V(0)
            var acc2 = V(0)
            var acc3 = V(0)
            for t in range(k):
                var av = a.unsafe_load[width=_W](t * a_stride + r0)
                var wt = t * nc + c0
                acc0 = av.fma(V(w[unsafe_offset=wt]), acc0)
                acc1 = av.fma(V(w[unsafe_offset=wt + 1]), acc1)
                acc2 = av.fma(V(w[unsafe_offset=wt + 2]), acc2)
                acc3 = av.fma(V(w[unsafe_offset=wt + 3]), acc3)
            var b = r0 + top - c0
            _store_sub(o, c0 * o_stride + r0, b, acc0)
            _store_sub(o, (c0 + 1) * o_stride + r0, b - 1, acc1)
            _store_sub(o, (c0 + 2) * o_stride + r0, b - 2, acc2)
            _store_sub(o, (c0 + 3) * o_stride + r0, b - 3, acc3)
            r0 += _W
        _sub_rows(o, o_stride, a, a_stride, w, k, nc, r0, m, c0, c0 + 4, top)
        c0 += 4
    while c0 < nc:  # the last columns, one at a time
        var r0 = max(c0 - top, 0)
        while r0 + _W <= m:
            var acc = V(0)
            for t in range(k):
                acc = a.unsafe_load[width=_W](t * a_stride + r0).fma(
                    V(w[unsafe_offset=t * nc + c0]), acc
                )
            _store_sub(o, c0 * o_stride + r0, r0 + top - c0, acc)
            r0 += _W
        _sub_rows(o, o_stride, a, a_stride, w, k, nc, r0, m, c0, c0 + 1, top)
        c0 += 1


@always_inline
def _store_sub(
    o: Pointer[mut=True, Float64, ...],
    at: Int,
    below: Int,
    acc: SIMD[DType.float64, _W],
):
    """`o[at:at+_W] -= acc`, skipping the first `-below` entries when the
    tile starts above the diagonal (`below < 0`)."""
    if below >= 0:
        o.unsafe_store(at, o.unsafe_load[width=_W](at) - acc)
    else:
        for i in range(-below, _W):
            o[unsafe_offset=at + i] -= acc[i]


def _sub_rows(
    o: Pointer[mut=True, Float64, ...],
    o_stride: Int,
    a: Pointer[Float64, ...],
    a_stride: Int,
    w: Pointer[Float64, ...],
    k: Int,
    nc: Int,
    r_lo: Int,
    m: Int,
    c_lo: Int,
    c_hi: Int,
    top: Int,
):
    """`_sub_product` on rows `r_lo..m`, columns `c_lo..c_hi`, by scalars."""
    for r in range(r_lo, m):
        for c in range(c_lo, c_hi):
            if r + top < c:
                continue
            var acc: Float64 = 0
            for t in range(k):
                acc = a[unsafe_offset=t * a_stride + r].fma(
                    w[unsafe_offset=t * nc + c], acc
                )
            o[unsafe_offset=c * o_stride + r] -= acc


def _sub_dots(
    res: Pointer[mut=True, Float64, ...],
    mb: Int,
    a: Pointer[Float64, ...],
    a_stride: Int,
    b: Pointer[Float64, ...],
    b_stride: Int,
    deg_v: Int,
    nc: Int,
):
    """`res[j * mb + c] = sum_r A[r, j] B[r, c]` over the first `deg_v`
    rows (a multiple of `_W`), `A` and `B` column-major (`A[r, j]` at
    `a[j * a_stride + r]`, `B[r, c]` at `b[c * b_stride + r]`), for `nc`
    columns of `A` and `mb` of `B`.

    Each entry is `_W` lanes of fused multiply-adds down the rows, then
    `reduce_add`, whatever the tiling: tiles of two columns of `A` by two
    of `B` share each load, and the odd ones are done alone."""
    var j = 0
    while j + 2 <= nc:
        var a0 = a.unsafe_offset(j * a_stride)
        var a1 = a.unsafe_offset((j + 1) * a_stride)
        var c = 0
        while c + 2 <= mb:
            var b0 = b.unsafe_offset(c * b_stride)
            var b1 = b.unsafe_offset((c + 1) * b_stride)
            comptime V = SIMD[DType.float64, _W]
            var s00 = V(0)
            var s01 = V(0)
            var s10 = V(0)
            var s11 = V(0)
            var r = 0
            while r < deg_v:
                var av0 = a0.unsafe_load[width=_W](r)
                var av1 = a1.unsafe_load[width=_W](r)
                var bv0 = b0.unsafe_load[width=_W](r)
                var bv1 = b1.unsafe_load[width=_W](r)
                s00 = av0.fma(bv0, s00)
                s01 = av0.fma(bv1, s01)
                s10 = av1.fma(bv0, s10)
                s11 = av1.fma(bv1, s11)
                r += _W
            res[unsafe_offset=j * mb + c] = s00.reduce_add()
            res[unsafe_offset=j * mb + c + 1] = s01.reduce_add()
            res[unsafe_offset=(j + 1) * mb + c] = s10.reduce_add()
            res[unsafe_offset=(j + 1) * mb + c + 1] = s11.reduce_add()
            c += 2
        if c < mb:
            var bc = b.unsafe_offset(c * b_stride)
            res[unsafe_offset=j * mb + c] = _dot_lanes(a0, bc, deg_v)
            res[unsafe_offset=(j + 1) * mb + c] = _dot_lanes(a1, bc, deg_v)
        j += 2
    if j < nc:
        var aj = a.unsafe_offset(j * a_stride)
        for c in range(mb):
            res[unsafe_offset=j * mb + c] = _dot_lanes(
                aj, b.unsafe_offset(c * b_stride), deg_v
            )


@always_inline
def _down_triangle(
    vp: Pointer[Float64, ...],
    col: Int,
    yp: Pointer[mut=True, Float64, ...],
    s0: Int,
    lo: Int,
    nc: Int,
    xj: Float64,
):
    """`yp[s0 + i] -= vp[col + i] xj` for `i` in `lo..nc` by fused
    multiply-adds: a column of a supernode's triangle in the forward
    solve."""
    comptime V = SIMD[DType.float64, _W]
    var xv = V(-xj)
    var i = lo
    while i + _W <= nc:
        yp.unsafe_store(
            s0 + i,
            vp.unsafe_load[width=_W](col + i).fma(
                xv, yp.unsafe_load[width=_W](s0 + i)
            ),
        )
        i += _W
    while i < nc:
        yp[unsafe_offset=s0 + i] = vp[unsafe_offset=col + i].fma(
            -xj, yp[unsafe_offset=s0 + i]
        )
        i += 1


@always_inline
def _up_triangle(
    vp: Pointer[Float64, ...],
    col: Int,
    yp: Pointer[Float64, ...],
    s0: Int,
    lo: Int,
    nc: Int,
    acc: Float64,
) -> Float64:
    """`acc - sum_i vp[col + i] yp[s0 + i]` over `i` in `lo..nc`: `_W`
    lanes of fused multiply-adds, reduced and subtracted, then the rest
    one at a time; a column of a supernode's triangle in the backward
    solve."""
    comptime V = SIMD[DType.float64, _W]
    var i = lo
    var tv = V(0)
    while i + _W <= nc:
        tv = vp.unsafe_load[width=_W](col + i).fma(
            yp.unsafe_load[width=_W](s0 + i), tv
        )
        i += _W
    var out = acc - tv.reduce_add()
    while i < nc:
        out = vp[unsafe_offset=col + i].fma(-yp[unsafe_offset=s0 + i], out)
        i += 1
    return out


@always_inline
def _dot_lanes(
    a: Pointer[Float64, ...], b: Pointer[Float64, ...], deg_v: Int
) -> Float64:
    """One entry of `_sub_dots`."""
    comptime V = SIMD[DType.float64, _W]
    var s = V(0)
    var r = 0
    while r < deg_v:
        s = a.unsafe_load[width=_W](r).fma(b.unsafe_load[width=_W](r), s)
        r += _W
    return s.reduce_add()


def etree_postorder(etree: List[Int]) -> List[Int]:
    """A postorder of a forest: every node after its descendants, each
    subtree contiguous, children in increasing order.

    Relabeling the columns of a factorization by a postorder of its
    elimination tree leaves the fill unchanged and makes each chain of the
    tree a run of consecutive columns, which supernodes need.

    Args:
        etree: `etree[j]` is the parent of `j`, or -1 for a root.

    Returns:
        `post`: `post[k]` is the `k`-th node visited.
    """
    var n = len(etree)
    var head = List[Int](length=n, fill=_NONE)
    var nxt = List[Int](length=n, fill=_NONE)
    for j in range(n - 1, -1, -1):  # pushed in reverse: increasing order
        var p = etree[j]
        if p != _NONE:
            nxt[j] = head[p]
            head[p] = j
    var post = List[Int](capacity=n)
    var stack = List[Int](capacity=n)
    for root in range(n):
        if etree[root] != _NONE:
            continue
        stack.append(root)
        while len(stack) > 0:
            var p = stack[len(stack) - 1]
            var c = head[p]
            if c == _NONE:
                _ = stack.pop()
                post.append(p)
            else:
                head[p] = nxt[c]
                stack.append(c)
    return post^


def _relax_accepts(combined: Int, zeros: Int, expanded: Int) -> Bool:
    """Whether a merged supernode of `combined` columns, `expanded` stored
    entries and `zeros` of them explicit zeros, is acceptable: the
    thresholds `(4, 1.0), (16, 0.8), (48, 0.1), (any, 0.05)` (size up to,
    fraction of zeros up to), faer's defaults."""
    var e = Float64(expanded)
    var z = Float64(zeros)
    if combined <= 4 and e * 1.0 >= z:
        return True
    if combined <= 16 and e * 0.8 >= z:
        return True
    if combined <= 48 and e * 0.1 >= z:
        return True
    return e * 0.05 >= z


struct _Supernodal(Movable):
    """`P A P^T = L D L^T` stored by supernodes.

    Supernode `s` holds columns `begin[s]..begin[s+1]` (`nc` of them) and
    the rows below them in `rows[row_ptr[s]:row_ptr[s+1]]` (`deg`,
    increasing); its block `vals[val_ptr[s]:]` is `(nc + deg) x nc`,
    column-major: the dense triangle (with `D` on its diagonal, zeros
    above), then the rows of the pattern."""

    var n: Int
    var begin: List[Int]
    var row_ptr: List[Int]
    var rows: List[Int]
    var val_ptr: List[Int]
    var vals: List[Float64]
    var diag_at: List[Int]
    """Where `D[k]` is stored in `vals`."""
    var num_regularized: Int

    def __init__(
        out self,
        Cp: List[Int],
        Ci: List[Int],
        Cx: List[Float64],
        etree: List[Int],
        counts: List[Int],
        perm: List[Int],
        signs: Optional[List[Int]],
        regularization: Float64,
    ) raises:
        """Factor `C = P A P^T`, given by its lower triangle by rows, with
        its elimination tree and column counts (postordered for supernodes
        of the most columns)."""
        var n = len(etree)
        self.n = n
        self.num_regularized = 0
        # -- fundamental supernodes: j joins j - 1 when it is j - 1's
        # parent, its only child, and has j - 1's pattern less j
        var nchild = List[Int](length=n, fill=0)
        for j in range(n):
            if etree[j] != _NONE:
                nchild[etree[j]] += 1
        var sizes = List[Int]()
        for j in range(n):
            if (
                j > 0
                and etree[j - 1] == j
                and nchild[j] == 1
                and counts[j - 1] == counts[j] + 1
            ):
                sizes[len(sizes) - 1] += 1
            else:
                sizes.append(1)
        var ns = len(sizes)
        var col_super = List[Int](length=n, fill=0)
        var sparent = List[Int](length=ns, fill=_NONE)
        var degree = List[Int](length=ns, fill=0)
        _supernode_tree(sizes, etree, counts, col_super, sparent, degree)
        # -- relaxed amalgamation: in a postorder, a supernode's last child
        # is the one just before it; merge it in when the explicit zeros
        # that adds stay under the thresholds
        var zeros = List[Int](length=ns, fill=0)
        var merged = False
        for p in range(1, ns):
            var c = p - 1
            if sparent[c] != p:
                continue
            var cs = sizes[c]
            var ps = sizes[p]
            var new_zeros = (ps + degree[p] - degree[c]) * cs
            var total = zeros[c] + zeros[p] + new_zeros
            var combined = cs + ps
            var expanded = combined * (combined + 1) // 2 + degree[p] * combined
            if new_zeros == 0 or _relax_accepts(combined, total, expanded):
                sizes[p] += cs
                sizes[c] = 0
                zeros[p] = total
                merged = True
        if merged:
            var kept = List[Int]()
            for s in range(ns):
                if sizes[s] > 0:
                    kept.append(sizes[s])
            sizes = kept^
            ns = len(sizes)
            sparent = List[Int](length=ns, fill=_NONE)
            degree = List[Int](length=ns, fill=0)
            _supernode_tree(sizes, etree, counts, col_super, sparent, degree)
        # -- layout
        self.begin = List[Int](length=ns + 1, fill=0)
        self.row_ptr = List[Int](length=ns + 1, fill=0)
        self.val_ptr = List[Int](length=ns + 1, fill=0)
        for s in range(ns):
            var nc = sizes[s]
            self.begin[s + 1] = self.begin[s] + nc
            self.row_ptr[s + 1] = self.row_ptr[s] + degree[s]
            self.val_ptr[s + 1] = self.val_ptr[s] + (nc + degree[s]) * nc
        # -- row patterns: row k is in supernode s's pattern exactly when s
        # is on a path from an entry of row k of C up to k's supernode
        self.rows = List[Int](length=self.row_ptr[ns], fill=0)
        var pos = self.row_ptr.copy()
        var visited = List[Int](length=ns, fill=_NONE)
        for k in range(n):
            visited[col_super[k]] = k
            for t in range(Cp[k], Cp[k + 1]):
                var i = Ci[t]
                if i >= k:
                    continue
                var s = col_super[i]
                while visited[s] != k:
                    self.rows[pos[s]] = k
                    pos[s] += 1
                    visited[s] = k
                    s = sparent[s]
        for s in range(ns):
            if pos[s] != self.row_ptr[s + 1]:
                raise Error(
                    String(
                        "SparseLDLT: internal error: supernode ",
                        s,
                        " has ",
                        pos[s] - self.row_ptr[s],
                        " rows below it, not ",
                        degree[s],
                    )
                )
        # -- descendants: the supernodes whose patterns meet each one
        var dptr = List[Int](length=ns + 1, fill=0)
        for d in range(ns):
            var prev = _NONE
            for t in range(self.row_ptr[d], self.row_ptr[d + 1]):
                var s = col_super[self.rows[t]]
                if s != prev:
                    dptr[s + 1] += 1
                    prev = s
        for s in range(ns):
            dptr[s + 1] += dptr[s]
        var desc = List[Int](length=dptr[ns], fill=0)
        var dpos = dptr.copy()
        for d in range(ns):
            var prev = _NONE
            for t in range(self.row_ptr[d], self.row_ptr[d + 1]):
                var s = col_super[self.rows[t]]
                if s != prev:
                    desc[dpos[s]] = d
                    dpos[s] += 1
                    prev = s
        # -- C's lower triangle by columns (entries i >= j of column j)
        var Lp = List[Int](length=n + 1, fill=0)
        for t in range(len(Ci)):
            Lp[Ci[t] + 1] += 1
        for j in range(n):
            Lp[j + 1] += Lp[j]
        var Li = List[Int](length=len(Ci), fill=0)
        var Lx = List[Float64](length=len(Ci), fill=0)
        var at = Lp.copy()
        for i in range(n):
            for t in range(Cp[i], Cp[i + 1]):
                var j = Ci[t]
                Li[at[j]] = i
                Lx[at[j]] = Cx[t]
                at[j] += 1
        # -- numeric, left-looking: per supernode, assemble it from C,
        # subtract every descendant's update, factor its dense block.
        # Disjoint subtrees run on separate threads, then the supernodes
        # above them in order, each split across the threads when large
        # each block is zeroed by the thread that assembles it, so the
        # pages are first touched in parallel
        self.vals = List[Float64](unsafe_uninit_length=self.val_ptr[ns])
        self.diag_at = List[Int](length=n, fill=0)
        var perm_c = perm.copy()
        var signs_c = signs.value().copy() if signs else List[Int]()
        var f = _Frame(
            n,
            _ints(self.begin),
            _ints(self.row_ptr),
            _ints(self.rows),
            _ints(self.val_ptr),
            _floats(self.vals),
            _ints(self.diag_at),
            _ints(dptr),
            _ints(desc),
            _ints(Lp),
            _ints(Li),
            _floats(Lx),
            _ints(perm_c),
            _ints(signs_c),
            True if signs else False,
            regularization,
        )
        var bad = _NONE
        var bad_d: Float64 = 0
        self.num_regularized = _factor_all(f, sparent, bad, bad_d)
        _ = dptr^  # the frame's pointers are untracked: keep these alive
        _ = desc^
        _ = Lp^
        _ = Li^
        _ = Lx^
        _ = signs_c^
        if bad != _NONE:
            raise Error(
                String(
                    "SparseLDLT: pivot ",
                    bad,
                    " (row ",
                    perm_c[bad],
                    " of the matrix) is ",
                    bad_d,
                    (
                        "; the matrix is singular, or needs another"
                        " ordering or regularization"
                    ),
                )
            )

    def pivot(self, k: Int) -> Float64:
        """`D[k]`."""
        return self.vals[self.diag_at[k]]

    def solve_in_place(self, mut y: List[Float64]):
        """`y = (L D L^T)^-1 y`, in the permuted numbering: `solve_columns`
        with one column."""
        self.solve_columns(
            y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 0, 1
        )

    def solve_columns(self, Y: _FloatPtr, c_lo: Int, c_hi: Int):
        """`Y[:, c] = (L D L^T)^-1 Y[:, c]` for the columns `c_lo..c_hi` of
        a column-major `n`-row `Y`, in the permuted numbering.

        Supernode by supernode, each block is applied to all the columns
        while it is in cache. The rows below a supernode's triangle are
        gathered once per column into a dense block, and the products with
        them are matrix products: `_sub_product` (register tiles of rows by
        columns of `Y`) on the way down, `_sub_dots` (pairs of columns of
        `L` by pairs of columns of `Y`) on the way up. Each entry is the
        same chain of fused multiply-adds whatever the number of columns,
        so a column gets the same bits as when solved alone. One column
        builds those chains column by column of `L` instead, without the
        copies the products need (a single solve was 20-28% slower
        through them)."""
        var ns = len(self.begin) - 1
        var n = self.n
        var mb = c_hi - c_lo
        if mb <= 0:
            return
        var vp = self.vals.unsafe_ptr()
        var maxdeg = 0
        var maxnc = 0
        for s in range(ns):
            maxdeg = max(maxdeg, self.row_ptr[s + 1] - self.row_ptr[s])
            maxnc = max(maxnc, self.begin[s + 1] - self.begin[s])
        var buf = List[Float64](length=maxdeg * mb, fill=0)
        var wbuf = List[Float64](length=maxnc * mb, fill=0)
        var bp = buf.unsafe_ptr()
        var wp = wbuf.unsafe_ptr()
        var Y0 = Y.unsafe_offset(c_lo * n)
        comptime V = SIMD[DType.float64, _W]
        for s in range(ns):  # L
            var s0 = self.begin[s]
            var nc = self.begin[s + 1] - s0
            var rp = self.row_ptr[s]
            var deg = self.row_ptr[s + 1] - rp
            var nr = nc + deg
            var base = self.val_ptr[s]
            if mb == 1:
                # one column: the rows below the triangle get the chains
                # `_sub_product` builds (from 0, in column order, then
                # subtracted), built column by column of `L`, each while
                # it is in cache, with no copies of `Y`
                for r in range(deg):
                    bp[unsafe_offset=r] = 0
                for j in range(nc):
                    var xj = Y0[unsafe_offset=s0 + j]
                    var col = base + j * nr
                    if xj != 0:
                        _down_triangle(vp, col, Y0, s0, j + 1, nc, xj)
                    var c = col + nc
                    var xv = V(xj)
                    var r = 0
                    while r + _W <= deg:
                        bp.unsafe_store(
                            r,
                            vp.unsafe_load[width=_W](c + r).fma(
                                xv, bp.unsafe_load[width=_W](r)
                            ),
                        )
                        r += _W
                    while r < deg:
                        bp[unsafe_offset=r] = vp[unsafe_offset=c + r].fma(
                            xj, bp[unsafe_offset=r]
                        )
                        r += 1
                for r in range(deg):
                    var at = self.rows[rp + r]
                    Y0[unsafe_offset=at] = (
                        Y0[unsafe_offset=at] - bp[unsafe_offset=r]
                    )
                continue
            for cy in range(mb):  # the triangle, column by column
                var yp = Y0.unsafe_offset(cy * n)
                for j in range(nc):
                    var xj = yp[unsafe_offset=s0 + j]
                    if xj != 0:
                        _down_triangle(vp, base + j * nr, yp, s0, j + 1, nc, xj)
            if deg == 0:
                continue
            for t in range(nc):  # W = Y[s0:s0+nc, :], row-major
                for cy in range(mb):
                    wp[unsafe_offset=t * mb + cy] = Y0[
                        unsafe_offset=cy * n + s0 + t
                    ]
            self._gather(Y0, n, bp, rp, deg, mb)
            _sub_product(
                bp,
                deg,
                vp.unsafe_offset(base + nc),
                nr,
                wp,
                nc,
                deg,
                mb,
                top=mb,
            )
            for cy in range(mb):
                var yp = Y0.unsafe_offset(cy * n)
                var b = bp.unsafe_offset(cy * deg)
                for r in range(deg):
                    yp[unsafe_offset=self.rows[rp + r]] = b[unsafe_offset=r]
        for cy in range(mb):  # D
            var yp = Y0.unsafe_offset(cy * n)
            for k in range(n):
                yp[unsafe_offset=k] = (
                    yp[unsafe_offset=k] / vp[unsafe_offset=self.diag_at[k]]
                )
        for s in range(ns - 1, -1, -1):  # L^T
            var s0 = self.begin[s]
            var nc = self.begin[s + 1] - s0
            var rp = self.row_ptr[s]
            var deg = self.row_ptr[s + 1] - rp
            var nr = nc + deg
            var base = self.val_ptr[s]
            var deg_v = deg - deg % _W
            self._gather(Y0, n, bp, rp, deg, mb)
            if mb == 1:
                # one column: each column of `L`'s dot (`_dot_lanes`, as
                # `_sub_dots` takes it) just before it is used
                for j in range(nc - 1, -1, -1):
                    var col = base + j * nr
                    var c = col + nc
                    var acc = Y0[unsafe_offset=s0 + j] - _dot_lanes(
                        vp.unsafe_offset(c), bp, deg_v
                    )
                    for r in range(deg_v, deg):
                        acc = vp[unsafe_offset=c + r].fma(
                            -bp[unsafe_offset=r], acc
                        )
                    Y0[unsafe_offset=s0 + j] = _up_triangle(
                        vp, col, Y0, s0, j + 1, nc, acc
                    )
                continue
            _sub_dots(
                wp, mb, vp.unsafe_offset(base + nc), nr, bp, deg, deg_v, nc
            )
            for cy in range(mb):  # the rest, and the triangle
                var yp = Y0.unsafe_offset(cy * n)
                var b = bp.unsafe_offset(cy * deg)
                for j in range(nc - 1, -1, -1):
                    var col = base + j * nr
                    var c = col + nc
                    var acc = (
                        yp[unsafe_offset=s0 + j] - wp[unsafe_offset=j * mb + cy]
                    )
                    for r in range(deg_v, deg):
                        acc = vp[unsafe_offset=c + r].fma(
                            -b[unsafe_offset=r], acc
                        )
                    yp[unsafe_offset=s0 + j] = _up_triangle(
                        vp, col, yp, s0, j + 1, nc, acc
                    )
        _ = buf^
        _ = wbuf^

    def _gather(
        self,
        Y0: _FloatPtr,
        n: Int,
        bp: Pointer[mut=True, Float64, ...],
        rp: Int,
        deg: Int,
        mb: Int,
    ):
        """`bp[c * deg + r] = Y0[c * n + rows[rp + r]]`: the rows below a
        supernode's triangle, by columns."""
        for cy in range(mb):
            var yp = Y0.unsafe_offset(cy * n)
            var b = bp.unsafe_offset(cy * deg)
            for r in range(deg):
                b[unsafe_offset=r] = yp[unsafe_offset=self.rows[rp + r]]

    def num_supernodes(self) -> Int:
        """The number of supernodes."""
        return len(self.begin) - 1

    def stored_entries(self) -> Int:
        """The values stored: the blocks, explicit zeros included."""
        return len(self.vals)


def _supernode_tree(
    sizes: List[Int],
    etree: List[Int],
    counts: List[Int],
    mut col_super: List[Int],
    mut sparent: List[Int],
    mut degree: List[Int],
):
    """The supernode of each column, each supernode's parent (that of the
    parent of its last column), and its degree (its last column's count
    below the diagonal)."""
    var j = 0
    for s in range(len(sizes)):
        for _ in range(sizes[s]):
            col_super[j] = s
            j += 1
    var first = 0
    for s in range(len(sizes)):
        var last = first + sizes[s] - 1
        degree[s] = counts[last] - 1
        first += sizes[s]
        var p = etree[last]
        sparent[s] = col_super[p] if p != _NONE else _NONE


# ===----------------------------------------------------------------------=== #
# The numeric factorization, serial and parallel
# ===----------------------------------------------------------------------=== #

comptime _IntPtr = Pointer[Int, MutUntrackedOrigin]
comptime _FloatPtr = Pointer[Float64, MutUntrackedOrigin]

comptime _PARALLEL_WORK = 5.0e6
"""Factorizations of less work (sum of squared column counts, about the
multiply-adds) run serially: a parallel call costs about 200 µs."""
comptime _SPLIT_WORK = 2.0e5
"""A supernode above the subtrees of more work (its own and its updates')
is split into tasks."""
comptime _ROWS_WORK = 2.0e5
"""A panel splits the rows below its diagonal block into tasks above this
much work."""
comptime _ROW_BLOCK = 64
"""Rows per task below a panel's diagonal block."""
comptime _GRAIN = 4.0
"""Subtrees are split until the heaviest holds at most `1 / _GRAIN` of a
thread's share of the work they hold."""


def _ints(mut v: List[Int]) -> _IntPtr:
    """`v`'s storage, untracked: the caller keeps `v` alive and unresized."""
    return v.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _floats(mut v: List[Float64]) -> _FloatPtr:
    """`v`'s storage, untracked: the caller keeps `v` alive and unresized."""
    return v.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _lower_bound(v: _IntPtr, lo: Int, hi: Int, x: Int) -> Int:
    """The first position in `v[lo:hi]` (sorted) holding at least `x`."""
    var a = lo
    var b = hi
    while a < b:
        var m = (a + b) >> 1
        if v[unsafe_offset=m] < x:
            a = m + 1
        else:
            b = m
    return a


@fieldwise_init
struct _Frame(Copyable, Movable):
    """A factorization in progress, by pointers that threads share: the
    layout of `_Supernodal`, its values, the descendant lists, `C` by
    columns, and the pivot rule. Each method touches only the blocks it
    names, so threads may run them on disjoint supernodes (or disjoint
    columns or rows of one)."""

    var n: Int
    var begin: _IntPtr
    var row_ptr: _IntPtr
    var rows: _IntPtr
    var val_ptr: _IntPtr
    var vals: _FloatPtr
    var diag_at: _IntPtr
    var dptr: _IntPtr
    var desc: _IntPtr
    var Lp: _IntPtr
    var Li: _IntPtr
    var Lx: _FloatPtr
    var perm: _IntPtr
    var signs: _IntPtr
    var has_signs: Bool
    var regularization: Float64

    def cols(self, s: Int) -> Int:
        """The number of columns of supernode `s`."""
        return self.begin[unsafe_offset=s + 1] - self.begin[unsafe_offset=s]

    def height(self, s: Int) -> Int:
        """The number of rows of supernode `s`'s block."""
        return (
            self.cols(s)
            + self.row_ptr[unsafe_offset=s + 1]
            - self.row_ptr[unsafe_offset=s]
        )

    def work(self, s: Int) -> Float64:
        """The multiply-adds that eliminating `s`'s columns costs: the sum
        of their squared column counts."""
        var nc = self.cols(s)
        var nr = self.height(s)
        var w: Float64 = 0
        for j in range(nc):
            var c = Float64(nr - j)
            w += c * c
        return w

    def mark(self, s: Int, g2l: _IntPtr):
        """Point `g2l` at the local rows of `s`'s pattern (entries for
        other rows are stale and never read: every row looked up while
        assembling `s` is in its pattern)."""
        var nc = self.cols(s)
        var rp = self.row_ptr[unsafe_offset=s]
        for r in range(self.row_ptr[unsafe_offset=s + 1] - rp):
            g2l[unsafe_offset=self.rows[unsafe_offset=rp + r]] = nc + r

    def scatter(self, s: Int, ca: Int, cb: Int, g2l: _IntPtr):
        """Set columns `ca..cb` (local) of `s`'s block to `C`'s (zero
        outside its entries); `g2l` marked for `s`."""
        var s0 = self.begin[unsafe_offset=s]
        var s1 = self.begin[unsafe_offset=s + 1]
        var nr = self.height(s)
        var base = self.val_ptr[unsafe_offset=s]
        for i in range(base + ca * nr, base + cb * nr):
            self.vals[unsafe_offset=i] = 0
        for j in range(s0 + ca, s0 + cb):
            var col = base + (j - s0) * nr
            for t in range(
                self.Lp[unsafe_offset=j], self.Lp[unsafe_offset=j + 1]
            ):
                var i = self.Li[unsafe_offset=t]
                var li = i - s0 if i < s1 else g2l[unsafe_offset=i]
                self.vals[unsafe_offset=col + li] += self.Lx[unsafe_offset=t]

    def update_work(self, s: Int) -> Float64:
        """About the multiply-adds of `s`'s descendant updates."""
        var s0 = self.begin[unsafe_offset=s]
        var s1 = self.begin[unsafe_offset=s + 1]
        var w: Float64 = 0
        for q in range(
            self.dptr[unsafe_offset=s], self.dptr[unsafe_offset=s + 1]
        ):
            var d = self.desc[unsafe_offset=q]
            var drp = self.row_ptr[unsafe_offset=d]
            var dend = self.row_ptr[unsafe_offset=d + 1]
            var start = _lower_bound(self.rows, drp, dend, s0)
            var mid = _lower_bound(self.rows, start, dend, s1) - start
            var m = dend - start
            w += Float64(self.cols(d)) * Float64(mid) * Float64(m)
        return w

    def update(
        self,
        d: Int,
        s: Int,
        ca: Int,
        cb: Int,
        g2l: _IntPtr,
        mut rmap: List[Int],
        mut tmp: List[Float64],
        mut coef: List[Float64],
    ):
        """Subtract descendant `d`'s contribution `L_b D_d L_m^T` from
        columns `ca..cb` (local) of `s`'s scattered block: `L_m` are `d`'s
        rows inside those columns, `L_b` its rows from the first of them
        on, lower part only."""
        var s0 = self.begin[unsafe_offset=s]
        var s1 = self.begin[unsafe_offset=s + 1]
        var nr = self.height(s)
        var base = self.val_ptr[unsafe_offset=s]
        var dnc = self.cols(d)
        var drp = self.row_ptr[unsafe_offset=d]
        var dend = self.row_ptr[unsafe_offset=d + 1]
        var dnr = self.height(d)
        var dbase = self.val_ptr[unsafe_offset=d]
        var start = _lower_bound(self.rows, drp, dend, s0 + ca)
        var mid = _lower_bound(self.rows, start, dend, s0 + cb) - start
        if mid == 0:
            return
        var m = dend - start
        # local row of each of d's rows in the target block
        rmap.resize(m, 0)
        for r in range(m):
            var g = self.rows[unsafe_offset=start + r]
            rmap[r] = g - s0 if g < s1 else g2l[unsafe_offset=g]
        var vp = self.vals
        var off = dnc + (start - drp)  # row of L_b's first row in d's block
        # coef[t, c] = D_t L_m[c, t]
        coef.resize(dnc * mid, 0)
        var cp = coef.unsafe_ptr()
        for t in range(dnc):
            var col = dbase + t * dnr
            var dt = vp[unsafe_offset=col + t]
            for c in range(mid):
                cp[unsafe_offset=t * mid + c] = (
                    dt * vp[unsafe_offset=col + off + c]
                )
        # tmp (m x mid, column-major, rows r >= c) = -L_b coef
        tmp.resize(m * mid, 0)
        var tp = tmp.unsafe_ptr()
        for i in range(m * mid):
            tp[unsafe_offset=i] = 0
        _sub_product(tp, m, vp.unsafe_offset(dbase + off), dnr, cp, dnc, m, mid)
        for c in range(mid):  # scatter into the target
            var col = base + rmap[c] * nr
            var tc = c * m
            for r in range(c, m):
                vp[unsafe_offset=col + rmap[r]] += tp[unsafe_offset=tc + r]

    def panel_coef(self, s: Int, j0: Int, j1: Int, cp: _FloatPtr):
        """`cp[t * pw + c] = D_t L[j0 + c, t]` for the columns `t < j0` of
        `s` before the panel `j0..j1` (`pw = j1 - j0`)."""
        var pw = j1 - j0
        var nr = self.height(s)
        var base = self.val_ptr[unsafe_offset=s]
        for t in range(j0):
            var col = base + t * nr
            var dt = self.vals[unsafe_offset=col + t]
            for c in range(pw):
                cp[unsafe_offset=t * pw + c] = (
                    dt * self.vals[unsafe_offset=col + j0 + c]
                )

    def panel_update(
        self, s: Int, j0: Int, j1: Int, lo: Int, hi: Int, cp: _FloatPtr
    ):
        """Rows `lo..hi` (`lo >= j0`) of the panel `j0..j1` of `s` minus
        `L[:, :j0] (D L[j0:j1, :j0])^T`, the columns before it (`cp` from
        `panel_coef`); on and below the diagonal."""
        if j0 == 0 or lo >= hi:
            return
        var nr = self.height(s)
        var base = self.val_ptr[unsafe_offset=s]
        _sub_product(
            self.vals.unsafe_offset(base + j0 * nr + lo),
            nr,
            self.vals.unsafe_offset(base + lo),
            nr,
            cp,
            j0,
            hi - lo,
            j1 - j0,
            lo - j0,
        )

    def panel_eliminate(self, s: Int, j0: Int, j1: Int, lo: Int, hi: Int):
        """Rows `lo..hi` (`lo >= j1`, below the panel's diagonal block,
        which `factor_diagonal` has factored) of the panel `j0..j1` of
        `s`: eliminate the panel's earlier columns from each, then divide
        by the pivot. Row by row the same arithmetic as the diagonal
        block's own columns."""
        if lo >= hi:
            return
        var nr = self.height(s)
        var base = self.val_ptr[unsafe_offset=s]
        var vp = self.vals
        for j in range(j0, j1):
            var cj = base + j * nr
            for c in range(j0, j):
                var cc = base + c * nr
                var ljc = vp[unsafe_offset=cc + j]
                if ljc == 0:
                    continue
                _axpy(vp, cj, cc, lo, hi, vp[unsafe_offset=cc + c] * ljc)
            var inv = 1 / vp[unsafe_offset=cj + j]
            for r in range(lo, hi):
                vp[unsafe_offset=cj + r] *= inv

    def factor_diagonal(
        self, s: Int, j0: Int, j1: Int, cp: _FloatPtr, mut w: _Worker
    ) -> Bool:
        """Update and factor the diagonal block of the panel `j0..j1` of
        `s` (its rows `j0..j1`) column by column, with the pivot rule.
        Returns False, with the pivot in `w.bad`, `w.bad_d`, if one is
        zero or not finite."""
        self.panel_update(s, j0, j1, j0, j1, cp)
        var s0 = self.begin[unsafe_offset=s]
        var nr = self.height(s)
        var base = self.val_ptr[unsafe_offset=s]
        var vp = self.vals
        for j in range(j0, j1):
            var cj = base + j * nr
            # column j -= L[:, c] d_c L[j, c] for c in j0..j, rows j..j1
            for c in range(j0, j):
                var cc = base + c * nr
                var ljc = vp[unsafe_offset=cc + j]
                if ljc == 0:
                    continue
                _axpy(vp, cj, cc, j, j1, vp[unsafe_offset=cc + c] * ljc)
            var k = s0 + j
            var d = vp[unsafe_offset=cj + j]
            var delta = self.regularization
            if delta > 0:
                if self.has_signs:
                    var sg = self.signs[
                        unsafe_offset=self.perm[unsafe_offset=k]
                    ]
                    if sg > 0 and d <= delta:
                        d = delta
                        w.regularized += 1
                    elif sg < 0 and d >= -delta:
                        d = -delta
                        w.regularized += 1
                elif abs(d) <= delta:
                    d = -delta if d < 0 else delta
                    w.regularized += 1
            if d == 0 or not isfinite(d):
                w.bad = k
                w.bad_d = d
                return False
            vp[unsafe_offset=cj + j] = d
            self.diag_at[unsafe_offset=k] = cj + j
            var inv = 1 / d
            for r in range(j + 1, j1):
                vp[unsafe_offset=cj + r] *= inv
        return True


@always_inline
def _axpy(vp: _FloatPtr, cj: Int, cc: Int, lo: Int, hi: Int, w: Float64):
    """`vp[cj + r] -= vp[cc + r] w` for `r` in `lo..hi`, by fused
    multiply-adds."""
    var wv = SIMD[DType.float64, _W](-w)
    var r = lo
    while r + _W <= hi:
        vp.unsafe_store(
            cj + r,
            vp.unsafe_load[width=_W](cc + r).fma(
                wv, vp.unsafe_load[width=_W](cj + r)
            ),
        )
        r += _W
    while r < hi:
        vp[unsafe_offset=cj + r] = vp[unsafe_offset=cc + r].fma(
            -w, vp[unsafe_offset=cj + r]
        )
        r += 1


struct _Worker(Movable):
    """One thread's scratch: the global-to-local row map of the supernode
    it is assembling, the dense buffers of an update, and its pivot
    count and failure."""

    var g2l: List[Int]
    var rmap: List[Int]
    var tmp: List[Float64]
    var coef: List[Float64]
    var regularized: Int
    var bad: Int
    var bad_d: Float64

    def __init__(out self, n: Int):
        self.g2l = List[Int](length=n, fill=_NONE)
        self.rmap = List[Int]()
        self.tmp = List[Float64]()
        self.coef = List[Float64]()
        self.regularized = 0
        self.bad = _NONE
        self.bad_d = 0


def _factor_supernode(f: _Frame, s: Int, mut w: _Worker) -> Bool:
    """Assemble, update, and factor supernode `s` on this thread. Returns
    False on a failed pivot (recorded in `w`)."""
    var g = _ints(w.g2l)
    var nc = f.cols(s)
    var nr = f.height(s)
    f.mark(s, g)
    f.scatter(s, 0, nc, g)
    for q in range(f.dptr[unsafe_offset=s], f.dptr[unsafe_offset=s + 1]):
        f.update(f.desc[unsafe_offset=q], s, 0, nc, g, w.rmap, w.tmp, w.coef)
    for j0 in range(0, nc, _PANEL):
        var j1 = min(j0 + _PANEL, nc)
        w.coef.resize(j0 * (j1 - j0), 0)
        var cp = _floats(w.coef)
        f.panel_coef(s, j0, j1, cp)
        if not f.factor_diagonal(s, j0, j1, cp, w):
            return False
        f.panel_update(s, j0, j1, j1, nr, cp)
        f.panel_eliminate(s, j0, j1, j1, nr)
    return True


comptime _SUBTREE = 0
"""Task: factor a whole subtree (`node` its root)."""
comptime _WHOLE = 1
"""Task: factor supernode `node` on one thread."""
comptime _WINDOW = 2
"""Task: assemble and update local columns `a..b` of `node`."""
comptime _DIAG = 3
"""Task: factor the diagonal block of the panel `a..b` of `node` (and the
rows below it when `c` is 1)."""
comptime _ROWS = 4
"""Task: rows `c..d` of the panel `a..b` of `node`, below its diagonal
block."""


struct _Tasks(Movable):
    """The factorization as a list of tasks, each after every task it
    depends on. Disjoint subtrees of the supernode tree are one task each
    (split from the roots down until the heaviest is small next to a
    thread's share, as in Geist and Ng's subtree-to-processor mapping),
    heaviest first; each supernode above them is one task when small, and
    otherwise windows of columns for its updates, then per panel a
    diagonal-block task and tasks of rows below it."""

    var kind: List[Int]
    var node: List[Int]
    var a: List[Int]
    var b: List[Int]
    var c: List[Int]
    var d: List[Int]
    var pending: List[Int64]
    """How many of each task's predecessors have not finished."""
    var succ_ptr: List[Int]
    var succ: List[Int]
    var first: List[Int]
    """The first supernode of each subtree (supernodes are postordered,
    so subtree `r` is `first[r]..r`)."""
    var coef_at: List[Int]
    """Where each split supernode's panel coefficients go in `coef`."""
    var coef: List[Float64]

    def __init__(out self, f: _Frame, sparent: List[Int], threads: Int):
        """The tasks for `threads` threads; none when the factorization is
        too small to share."""
        var ns = len(sparent)
        self.kind = List[Int]()
        self.node = List[Int]()
        self.a = List[Int]()
        self.b = List[Int]()
        self.c = List[Int]()
        self.d = List[Int]()
        self.pending = List[Int64]()
        self.succ_ptr = List[Int](length=1, fill=0)
        self.succ = List[Int]()
        self.first = List[Int](capacity=ns)
        self.coef_at = List[Int](length=ns, fill=0)
        self.coef = List[Float64]()
        for s in range(ns):
            self.first.append(s)
        var W = List[Float64](capacity=ns)
        for s in range(ns):
            W.append(f.work(s))
        var size = List[Int](length=ns, fill=1)
        for s in range(ns):  # children come before their parents
            var p = sparent[s]
            if p != _NONE:
                W[p] += W[s]
                size[p] += size[s]
                self.first[p] = min(self.first[p], self.first[s])
        var total: Float64 = 0
        var set = List[Int]()
        for s in range(ns):
            if size[s] != s - self.first[s] + 1:
                return  # not postordered: serial
            if sparent[s] == _NONE:
                set.append(s)
                total += W[s]
        var T = threads
        if T <= 1 or total < _PARALLEL_WORK:
            return
        var head = List[Int](length=ns, fill=_NONE)
        var nxt = List[Int](length=ns, fill=_NONE)
        for s in range(ns - 1, -1, -1):
            var p = sparent[s]
            if p != _NONE:
                nxt[s] = head[p]
                head[p] = s
        # -- subtrees: split the heaviest until it is small
        var in_set = total
        while len(set) < 64 * T:
            var h = 0
            for i in range(1, len(set)):
                if W[set[i]] > W[set[h]]:
                    h = i
            var r = set[h]
            if W[r] * _GRAIN * Float64(T) <= in_set or head[r] == _NONE:
                break
            set[h] = set[len(set) - 1]
            _ = set.pop()
            in_set -= W[r]
            var c = head[r]
            while c != _NONE:
                set.append(c)
                in_set += W[c]
                c = nxt[c]
        for i in range(1, len(set)):  # heaviest first
            var x = set[i]
            var j = i
            while j > 0 and W[set[j - 1]] < W[x]:
                set[j] = set[j - 1]
                j -= 1
            set[j] = x
        var in_subtree = List[Bool](length=ns, fill=False)
        var exit_lo = List[Int](length=ns, fill=0)
        var exit_hi = List[Int](length=ns, fill=0)
        var edges_from = List[Int]()
        var edges_to = List[Int]()
        for i in range(len(set)):
            var r = set[i]
            for s in range(self.first[r], r + 1):
                in_subtree[s] = True
            exit_lo[r] = len(self.kind)
            exit_hi[r] = exit_lo[r] + 1
            self._add(_SUBTREE, r, 0, 0, 0, 0)
        # -- the supernodes above the subtrees, in order
        var coef_size = 0
        for s in range(ns):
            if in_subtree[s]:
                continue
            var entry_lo = len(self.kind)
            var nc = f.cols(s)
            var nr = f.height(s)
            if nc == 1 or f.work(s) + f.update_work(s) < _SPLIT_WORK:
                self._add(_WHOLE, s, 0, 0, 0, 0)
                exit_lo[s] = entry_lo
                exit_hi[s] = entry_lo + 1
            else:
                # windows of columns of about equal area below the diagonal
                var parts = min(T, nc)
                var area: Float64 = 0
                for c in range(nc):
                    area += Float64(nr - c)
                var acc: Float64 = 0
                var ca = 0
                var p = 1
                for c in range(nc):
                    if p < parts and acc * Float64(parts) >= Float64(p) * area:
                        if c > ca:
                            self._add(_WINDOW, s, ca, c, 0, 0)
                            ca = c
                        p += 1
                    acc += Float64(nr - c)
                self._add(_WINDOW, s, ca, nc, 0, 0)
                # panels: rows below each diagonal block in fixed blocks of
                # _ROW_BLOCK, so a panel's diagonal task waits only for the
                # block holding its rows (and the panel before the last,
                # whose coefficient buffer it reuses)
                self.coef_at[s] = coef_size
                coef_size += 2 * nc * _PANEL
                var nb = (nr + _ROW_BLOCK - 1) // _ROW_BLOCK
                var last_rows = List[Int](length=nb, fill=_NONE)
                var prev_diag = _NONE
                var rows_lo = 0  # the previous panel's row tasks
                var rows_hi = 0
                var older_lo = 0  # the panel's before it
                var older_hi = 0
                for j0 in range(0, nc, _PANEL):
                    var j1 = min(j0 + _PANEL, nc)
                    var below = nr - j1
                    var work = Float64(below) * Float64(j1 - j0) * Float64(j1)
                    var split = below > _ROW_BLOCK and work > _ROWS_WORK
                    var diag = len(self.kind)
                    self._add(_DIAG, s, j0, j1, 0 if split else 1, 0)
                    var preds = List[Int]()
                    if prev_diag == _NONE:
                        for t in range(entry_lo, diag):
                            preds.append(t)
                    else:
                        preds.append(prev_diag)
                    for b in range(nb):
                        var t = last_rows[b]
                        if t == _NONE:
                            continue
                        if not split or (b + 1) * _ROW_BLOCK > j0:
                            if not split or b * _ROW_BLOCK < j1:
                                preds.append(t)
                    for t in range(older_lo, older_hi):
                        preds.append(t)
                    for t in preds:
                        edges_from.append(t)
                        edges_to.append(diag)
                    older_lo = rows_lo
                    older_hi = rows_hi
                    rows_lo = diag + 1
                    for b in range(nb):
                        var lo = max(b * _ROW_BLOCK, j1)
                        var hi = min((b + 1) * _ROW_BLOCK, nr)
                        if not split or lo >= hi:
                            last_rows[b] = _NONE
                            continue
                        var t = len(self.kind)
                        self._add(_ROWS, s, j0, j1, lo, hi)
                        edges_from.append(diag)
                        edges_to.append(t)
                        if last_rows[b] != _NONE:
                            edges_from.append(last_rows[b])
                            edges_to.append(t)
                        last_rows[b] = t
                    rows_hi = len(self.kind)
                    prev_diag = diag
                var prev_lo = rows_lo
                var prev_hi = rows_hi
                if rows_lo == rows_hi:
                    prev_lo = prev_diag
                    prev_hi = prev_diag + 1
                exit_lo[s] = prev_lo
                exit_hi[s] = prev_hi
            # every task that finishes a child precedes every entry task
            var c = head[s]
            while c != _NONE:
                for t in range(exit_lo[c], exit_hi[c]):
                    for e in range(entry_lo, exit_hi[s]):
                        if self.kind[e] == _DIAG or self.kind[e] == _ROWS:
                            break
                        edges_from.append(t)
                        edges_to.append(e)
                c = nxt[c]
        self.coef = List[Float64](length=coef_size, fill=0)
        var m = len(self.kind)
        self.pending = List[Int64](length=m, fill=0)
        self.succ_ptr = List[Int](length=m + 1, fill=0)
        for e in range(len(edges_from)):
            self.succ_ptr[edges_from[e] + 1] += 1
            self.pending[edges_to[e]] += 1
        for t in range(m):
            self.succ_ptr[t + 1] += self.succ_ptr[t]
        self.succ = List[Int](length=len(edges_from), fill=0)
        var at = self.succ_ptr.copy()
        for e in range(len(edges_from)):
            self.succ[at[edges_from[e]]] = edges_to[e]
            at[edges_from[e]] += 1

    def _add(mut self, kind: Int, node: Int, a: Int, b: Int, c: Int, d: Int):
        self.kind.append(kind)
        self.node.append(node)
        self.a.append(a)
        self.b.append(b)
        self.c.append(c)
        self.d.append(d)

    def __len__(self) -> Int:
        """The number of tasks."""
        return len(self.kind)


def _run_task(
    f: _Frame,
    mut w: _Worker,
    kind: Int,
    s: Int,
    a: Int,
    b: Int,
    c: Int,
    d: Int,
    first: _IntPtr,
    cp: _FloatPtr,
) -> Bool:
    """Run one task of `_Tasks` (`cp`: the supernode's panel coefficients).
    Returns False on a failed pivot."""
    if kind == _SUBTREE:
        for t in range(first[unsafe_offset=s], s + 1):
            if not _factor_supernode(f, t, w):
                return False
        return True
    if kind == _WHOLE:
        return _factor_supernode(f, s, w)
    if kind == _WINDOW:
        var g = _ints(w.g2l)
        f.mark(s, g)
        f.scatter(s, a, b, g)
        for q in range(f.dptr[unsafe_offset=s], f.dptr[unsafe_offset=s + 1]):
            f.update(f.desc[unsafe_offset=q], s, a, b, g, w.rmap, w.tmp, w.coef)
        return True
    var buf = cp.unsafe_offset(((a // _PANEL) % 2) * f.cols(s) * _PANEL)
    if kind == _DIAG:
        f.panel_coef(s, a, b, buf)
        if not f.factor_diagonal(s, a, b, buf, w):
            return False
        if c == 1:
            var nr = f.height(s)
            f.panel_update(s, a, b, b, nr, buf)
            f.panel_eliminate(s, a, b, b, nr)
        return True
    f.panel_update(s, a, b, c, d, buf)  # _ROWS
    f.panel_eliminate(s, a, b, c, d)
    return True


def _factor_all(
    f: _Frame, sparent: List[Int], mut bad: Int, mut bad_d: Float64
) -> Int:
    """Factor every supernode: by the tasks of `_Tasks` on every thread,
    or serially in order when it makes none. Threads claim tasks in order
    from a shared counter and wait for each one's predecessors to finish;
    as every predecessor comes earlier in the list, it is claimed by a
    running thread, so waiting cannot deadlock. Each entry gets the same
    arithmetic in the same order either way, so the bits are the serial
    ones. On a failed pivot, the factorization is redone serially, which
    finds the one a serial factorization reports (`bad`, `bad_d`, the
    first in supernode order). Returns the number of pivots regularized."""
    var ns = len(sparent)
    var T = max_threads(memory_bound=True)
    var tasks = _Tasks(f, sparent, T)
    var m = len(tasks.kind)
    if m > 0:
        var state = List[Int64](length=2, fill=0)  # next task, failed
        var reg = List[Int](length=T, fill=0)
        var sp = state.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var pp = tasks.pending.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var regp = _ints(reg)
        var kp = _ints(tasks.kind)
        var np = _ints(tasks.node)
        var ap = _ints(tasks.a)
        var bp = _ints(tasks.b)
        var cp = _ints(tasks.c)
        var dp = _ints(tasks.d)
        var sup = _ints(tasks.succ_ptr)
        var sucp = _ints(tasks.succ)
        var firstp = _ints(tasks.first)
        var coefat = _ints(tasks.coef_at)
        var coefp = _floats(tasks.coef)

        def worker(
            chunk: Int, lo: Int, hi: Int
        ) {
            f,
            m,
            sp,
            pp,
            regp,
            kp,
            np,
            ap,
            bp,
            cp,
            dp,
            sup,
            sucp,
            firstp,
            coefat,
            coefp,
        }:
            var w = _Worker(f.n)
            var failed = sp.unsafe_offset(1)
            while True:
                var i = Int(Atomic[Int64].fetch_add(sp, 1))
                if i >= m:
                    break
                var ready = True
                while Atomic[Int64].load(pp.unsafe_offset(i)) > 0:
                    if Atomic[Int64].load(failed) != 0:
                        ready = False
                        break
                if not ready or Atomic[Int64].load(failed) != 0:
                    break
                var s = np[unsafe_offset=i]
                if not _run_task(
                    f,
                    w,
                    kp[unsafe_offset=i],
                    s,
                    ap[unsafe_offset=i],
                    bp[unsafe_offset=i],
                    cp[unsafe_offset=i],
                    dp[unsafe_offset=i],
                    firstp,
                    coefp.unsafe_offset(coefat[unsafe_offset=s]),
                ):
                    Atomic[Int64].store(failed, 1)
                    break
                for e in range(sup[unsafe_offset=i], sup[unsafe_offset=i + 1]):
                    _ = Atomic[Int64].fetch_add(
                        pp.unsafe_offset(sucp[unsafe_offset=e]), -1
                    )
            regp[unsafe_offset=chunk] = w.regularized

        parallel_ranges(T, 0, worker)
        _ = tasks^  # its storage was used through untracked pointers
        if state[1] == 0:
            var total = 0
            for c in range(T):
                total += reg[c]
            return total
    var w = _Worker(f.n)
    for s in range(ns):
        if not _factor_supernode(f, s, w):
            bad = w.bad
            bad_d = w.bad_d
            break
    return w.regularized
