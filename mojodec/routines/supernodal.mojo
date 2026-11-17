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
# 14(5), 1993.
# ===----------------------------------------------------------------------=== #
"""
Supernodal LDL^T (`mojodec.routines.supernodal`).
==================================================
The numeric engine behind `SparseLDLT(A, layout=FactorLayout.SUPERNODAL)`.
Columns of `L` with the same pattern below a dense triangle (a supernode)
are stored together as one dense column-major block, so the factorization
and the solves run dense kernels (vector loads, one pass over each block)
instead of one sparse column at a time. Private: use `SparseLDLT`.
"""

from std.math import isfinite

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
):
    """`O[r, c] -= sum_t A[r, t] W[t, c]` for `r >= c` (the lower
    trapezoid; entries above the diagonal are left alone), with `O` and `A`
    column-major (`O[r, c]` at `o[c * o_stride + r]`, `A[r, t]` at
    `a[t * a_stride + r]`), `W` row-major `k x nc`, and `m` rows.

    Register tiles of `_W` rows by 4 columns: per step of `t`, one vector
    load of `A` and four fused multiply-adds."""
    comptime V = SIMD[DType.float64, _W]
    var c0 = 0
    while c0 + 4 <= nc:
        var r0 = c0
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
            _store_sub(o, c0 * o_stride + r0, r0 - c0, acc0)
            _store_sub(o, (c0 + 1) * o_stride + r0, r0 - c0 - 1, acc1)
            _store_sub(o, (c0 + 2) * o_stride + r0, r0 - c0 - 2, acc2)
            _store_sub(o, (c0 + 3) * o_stride + r0, r0 - c0 - 3, acc3)
            r0 += _W
        _sub_rows(o, o_stride, a, a_stride, w, k, nc, r0, m, c0, c0 + 4)
        c0 += 4
    while c0 < nc:  # the last columns, one at a time
        var r0 = c0
        while r0 + _W <= m:
            var acc = V(0)
            for t in range(k):
                acc = a.unsafe_load[width=_W](t * a_stride + r0).fma(
                    V(w[unsafe_offset=t * nc + c0]), acc
                )
            _store_sub(o, c0 * o_stride + r0, r0 - c0, acc)
            r0 += _W
        _sub_rows(o, o_stride, a, a_stride, w, k, nc, r0, m, c0, c0 + 1)
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
):
    """`_sub_product` on rows `r_lo..m`, columns `c_lo..c_hi`, by scalars."""
    for r in range(r_lo, m):
        for c in range(c_lo, c_hi):
            if r < c:
                continue
            var acc: Float64 = 0
            for t in range(k):
                acc = a[unsafe_offset=t * a_stride + r].fma(
                    w[unsafe_offset=t * nc + c], acc
                )
            o[unsafe_offset=c * o_stride + r] -= acc


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


def _lower_bound(v: List[Int], lo: Int, hi: Int, x: Int) -> Int:
    """The first position in `v[lo:hi]` (sorted) holding at least `x`."""
    var a = lo
    var b = hi
    while a < b:
        var m = (a + b) >> 1
        if v[m] < x:
            a = m + 1
        else:
            b = m
    return a


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
        # -- numeric, left-looking: assemble each supernode from C, subtract
        # every descendant's update, factor its dense block
        self.vals = List[Float64](length=self.val_ptr[ns], fill=0)
        self.diag_at = List[Int](length=n, fill=0)
        var g2l = List[Int](length=n, fill=_NONE)
        var rmap = List[Int]()
        var tmp = List[Float64]()
        var coef = List[Float64]()
        for s in range(ns):
            var s0 = self.begin[s]
            var s1 = self.begin[s + 1]
            var nc = s1 - s0
            var rp = self.row_ptr[s]
            var deg = self.row_ptr[s + 1] - rp
            var nr = nc + deg
            var base = self.val_ptr[s]
            for r in range(deg):
                g2l[self.rows[rp + r]] = nc + r
            for j in range(s0, s1):
                var col = base + (j - s0) * nr
                for t in range(Lp[j], Lp[j + 1]):
                    var i = Li[t]
                    var li = i - s0 if i < s1 else g2l[i]
                    self.vals[col + li] += Lx[t]
            for q in range(dptr[s], dptr[s + 1]):
                self._update(desc[q], s0, s1, nr, base, g2l, rmap, tmp, coef)
            self._factor_block(
                base, s0, nc, nr, perm, signs, regularization, coef
            )
            for r in range(deg):
                g2l[self.rows[rp + r]] = _NONE

    def _update(
        mut self,
        d: Int,
        s0: Int,
        s1: Int,
        nr: Int,
        base: Int,
        g2l: List[Int],
        mut rmap: List[Int],
        mut tmp: List[Float64],
        mut coef: List[Float64],
    ):
        """Subtract descendant `d`'s contribution `L_b D_d L_m^T` from the
        block at `base` (`nr` rows, columns `s0..s1`): `L_m` are `d`'s rows
        inside `s0..s1`, `L_b` its rows from `s0` on, lower part only."""
        var d0 = self.begin[d]
        var dnc = self.begin[d + 1] - d0
        var drp = self.row_ptr[d]
        var dend = self.row_ptr[d + 1]
        var dnr = dnc + dend - drp
        var dbase = self.val_ptr[d]
        var start = _lower_bound(self.rows, drp, dend, s0)
        var mid = _lower_bound(self.rows, start, dend, s1) - start
        var m = dend - start
        # local row of each of d's rows in the target block
        rmap.resize(m, 0)
        for r in range(m):
            var g = self.rows[start + r]
            rmap[r] = g - s0 if r < mid else g2l[g]
        var vp = self.vals.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
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

    def _factor_block(
        mut self,
        base: Int,
        s0: Int,
        nc: Int,
        nr: Int,
        perm: List[Int],
        signs: Optional[List[Int]],
        regularization: Float64,
        mut coef: List[Float64],
    ) raises:
        """Factor the assembled `nr x nc` block in place: `D` on the
        diagonal, `L` below it, rows of the pattern included. By panels of
        `_PANEL` columns: each panel first takes the columns before it in
        one tiled product, then is factored column by column."""
        var vp = self.vals.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        for j0 in range(0, nc, _PANEL):
            var j1 = min(j0 + _PANEL, nc)
            var pw = j1 - j0
            if j0 > 0:
                # panel -= L[j0:, :j0] (D L[j0:j1, :j0])^T
                coef.resize(j0 * pw, 0)
                var cp = coef.unsafe_ptr()
                for t in range(j0):
                    var col = base + t * nr
                    var dt = vp[unsafe_offset=col + t]
                    for c in range(pw):
                        cp[unsafe_offset=t * pw + c] = (
                            dt * vp[unsafe_offset=col + j0 + c]
                        )
                _sub_product(
                    vp.unsafe_offset(base + j0 * nr + j0),
                    nr,
                    vp.unsafe_offset(base + j0),
                    nr,
                    cp,
                    j0,
                    nr - j0,
                    pw,
                )
            for j in range(j0, j1):
                var cj = base + j * nr
                # column j -= L[:, c] d_c L[j, c] for c in j0..j, rows j..
                for c in range(j0, j):
                    var cc = base + c * nr
                    var ljc = vp[unsafe_offset=cc + j]
                    if ljc == 0:
                        continue
                    var w = vp[unsafe_offset=cc + c] * ljc
                    var wv = SIMD[DType.float64, _W](-w)
                    var r = j
                    while r + _W <= nr:
                        vp.unsafe_store(
                            cj + r,
                            vp.unsafe_load[width=_W](cc + r).fma(
                                wv, vp.unsafe_load[width=_W](cj + r)
                            ),
                        )
                        r += _W
                    while r < nr:
                        vp[unsafe_offset=cj + r] = vp[unsafe_offset=cc + r].fma(
                            -w, vp[unsafe_offset=cj + r]
                        )
                        r += 1
                var k = s0 + j
                var d = vp[unsafe_offset=cj + j]
                if regularization > 0:
                    var delta = regularization
                    if signs:
                        var sg = signs.value()[perm[k]]
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
                            perm[k],
                            " of the matrix) is ",
                            d,
                            (
                                "; the matrix is singular, or needs another"
                                " ordering or regularization"
                            ),
                        )
                    )
                vp[unsafe_offset=cj + j] = d
                self.diag_at[k] = cj + j
                var inv = 1 / d
                for r in range(j + 1, nr):
                    vp[unsafe_offset=cj + r] *= inv

    def pivot(self, k: Int) -> Float64:
        """`D[k]`."""
        return self.vals[self.diag_at[k]]

    def solve_in_place(self, mut y: List[Float64]):
        """`y = (L D L^T)^-1 y`, in the permuted numbering. Per supernode,
        the rows below its triangle are gathered once into a dense vector,
        so the inner loops run down contiguous columns."""
        var ns = len(self.begin) - 1
        var vp = self.vals.unsafe_ptr()
        var yp = y.unsafe_ptr()
        var maxdeg = 0
        for s in range(ns):
            maxdeg = max(maxdeg, self.row_ptr[s + 1] - self.row_ptr[s])
        var buf = List[Float64](length=maxdeg, fill=0)
        var bp = buf.unsafe_ptr()
        comptime V = SIMD[DType.float64, _W]
        for s in range(ns):  # L
            var s0 = self.begin[s]
            var nc = self.begin[s + 1] - s0
            var rp = self.row_ptr[s]
            var deg = self.row_ptr[s + 1] - rp
            var nr = nc + deg
            var base = self.val_ptr[s]
            for r in range(deg):
                bp[unsafe_offset=r] = 0
            for j in range(nc):
                var xj = yp[unsafe_offset=s0 + j]
                if xj == 0:
                    continue
                var col = base + j * nr
                var xv = V(-xj)
                var i = j + 1
                while i + _W <= nc:
                    yp.unsafe_store(
                        s0 + i,
                        vp.unsafe_load[width=_W](col + i).fma(
                            xv, yp.unsafe_load[width=_W](s0 + i)
                        ),
                    )
                    i += _W
                while i < nc:
                    yp[unsafe_offset=s0 + i] -= vp[unsafe_offset=col + i] * xj
                    i += 1
                var c = col + nc
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
                        -xj, bp[unsafe_offset=r]
                    )
                    r += 1
            for r in range(deg):
                yp[unsafe_offset=self.rows[rp + r]] += bp[unsafe_offset=r]
        for k in range(self.n):  # D
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
            for r in range(deg):
                bp[unsafe_offset=r] = yp[unsafe_offset=self.rows[rp + r]]
            for j in range(nc - 1, -1, -1):
                var col = base + j * nr
                var c = col + nc
                var accv = V(0)
                var r = 0
                while r + _W <= deg:
                    accv = vp.unsafe_load[width=_W](c + r).fma(
                        bp.unsafe_load[width=_W](r), accv
                    )
                    r += _W
                var acc = yp[unsafe_offset=s0 + j] - accv.reduce_add()
                while r < deg:
                    acc -= vp[unsafe_offset=c + r] * bp[unsafe_offset=r]
                    r += 1
                var i = j + 1
                var tv = V(0)
                while i + _W <= nc:
                    tv = vp.unsafe_load[width=_W](col + i).fma(
                        yp.unsafe_load[width=_W](s0 + i), tv
                    )
                    i += _W
                acc -= tv.reduce_add()
                while i < nc:
                    acc -= vp[unsafe_offset=col + i] * yp[unsafe_offset=s0 + i]
                    i += 1
                yp[unsafe_offset=s0 + j] = acc

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
