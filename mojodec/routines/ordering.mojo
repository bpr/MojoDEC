# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
# The approximate minimum degree ordering below is a port of faer's
# `sparse::linalg::amd` (faer 0.24, commit 0ecd846, MIT License, Copyright (c)
# 2026 sarah quiñones el kazdadi), itself a translation of SuiteSparse's AMD:
#
#   AMD, Copyright (c), 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
#   Iain S. Duff. All Rights Reserved. BSD 3-clause license.
#
# Both notices and licenses are reproduced in NOTICE.
# ===----------------------------------------------------------------------=== #
"""
Fill-reducing orderings (`mojodec.routines.ordering`).
=====================================================
`amd_ordering(A)` permutes a sparse symmetric matrix so that its Cholesky
or LDL^T factor has few nonzeros: Amestoy, Davis, and Duff's approximate
minimum degree (AMD, SIAM J. Matrix Anal. Appl. 17(4), 1996). Each step
eliminates a vertex of (nearly) least degree in the graph of the
remaining Schur complement, with degrees bounded from above rather than
computed, which makes the whole ordering about as cheap as a few sparse
matrix-vector products.
"""

from std.math import sqrt

from mojodec.core.sparse import CSRMatrix


comptime _NONE = -1


@always_inline
def _flip(i: Int) -> Int:
    return -2 - i


def _clear_flag(wflg: Int, wbig: Int, mut w: List[Int]) -> Int:
    if wflg < 2 or wflg >= wbig:
        for x in range(len(w)):
            if w[x] != 0:
                w[x] = 1
        return 2
    return wflg


def amd_ordering(
    A: CSRMatrix,
    dense: Float64 = 10.0,
    aggressive: Bool = True,
) raises -> List[Int]:
    """A fill-reducing permutation of a sparse symmetric matrix: approximate
    minimum degree, with rows of zero diagonal last.

    Only the pattern of `A + A^T` is used (values only to find zero
    diagonals), so `A` may be given whole or as one triangle. Rows whose
    diagonal entry is absent or zero are ordered after all others. In a
    saddle-point matrix `[[M, B^T], [B, 0]]` that puts the constraint block
    last: its pivots then come from the Schur complement `-B M^-1 B^T`,
    which is nonsingular when `M` is definite and `B` has full rank, so an
    LDL^T factorization meets no zero pivot (the rule Eigen's AMD uses).

    ```mojo
    var p = amd_ordering(K)        # p[k] = the row of K eliminated k-th
    ```

    Args:
        A: An `n x n` matrix, symmetric in pattern (its pattern is
            symmetrized otherwise).
        dense: Rows with more than `max(16, dense * sqrt(n))` off-diagonal
            entries are treated as dense and ordered last (SuiteSparse's
            default 10); a negative value turns this off.
        aggressive: Aggressive absorption of elements (SuiteSparse's
            default), which usually gives slightly better orderings.

    Returns:
        The permutation `p`: row `p[k]` of `A` becomes row `k` of `P A P^T`.

    Raises:
        If `A` is not square.
    """
    var n = A.nrows
    if A.ncols != n:
        raise Error(String("amd_ordering: the matrix is ", n, " x ", A.ncols))
    if n == 0:
        return List[Int]()
    # rows whose diagonal is absent or zero go last
    var last_rows = List[Bool](length=n, fill=True)
    for i in range(n):
        for t in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[t] == i and A.data[t] != 0:
                last_rows[i] = False
    # the pattern of A + A^T, rows sorted: symmetric, so its rows are also
    # its columns, the column-major input faer's code reads
    var S = _symmetric_pattern(A)
    ref Sp = S.indptr
    ref Si = S.indices
    # amd's `aat`: the length of each vertex's adjacency list
    var lens = List[Int](length=n, fill=0)
    var tp = List[Int](length=n, fill=0)
    for k in range(n):
        var seen = 0
        for q in range(Sp[k], Sp[k + 1]):
            var j = Si[q]
            if j < k:
                seen += 1
                lens[j] += 1
                lens[k] += 1
            else:
                if j == k:
                    seen += 1
                break
            var seen_j = 0
            for r in range(Sp[j] + tp[j], Sp[j + 1]):
                var i = Si[r]
                if i < k:
                    lens[i] += 1
                    lens[j] += 1
                    seen_j += 1
                else:
                    if i == k:
                        seen_j += 1
                    break
            tp[j] += seen_j
        tp[k] = seen
    for j in range(n):
        for r in range(Sp[j] + tp[j], Sp[j + 1]):
            lens[Si[r]] += 1
            lens[j] += 1
    var nzaat = 0
    for i in range(n):
        nzaat += lens[i]
    # amd's `amd_1`: the adjacency lists, in the same order as faer's
    var iwlen = nzaat + nzaat // 5 + n
    var iw = List[Int](length=iwlen, fill=0)
    var pe = List[Int](length=n, fill=0)
    var sp = List[Int](length=n, fill=0)
    var pfree = 0
    for j in range(n):
        pe[j] = pfree
        sp[j] = pfree
        pfree += lens[j]
    for j in range(n):
        tp[j] = 0
    for k in range(n):
        var seen = 0
        for q in range(Sp[k], Sp[k + 1]):
            var j = Si[q]
            if j < k:
                iw[sp[j]] = k
                sp[j] += 1
                iw[sp[k]] = j
                sp[k] += 1
                seen += 1
            else:
                if j == k:
                    seen += 1
                break
            var seen_j = 0
            for r in range(Sp[j] + tp[j], Sp[j + 1]):
                var i = Si[r]
                if i < k:
                    iw[sp[i]] = j
                    sp[i] += 1
                    iw[sp[j]] = i
                    sp[j] += 1
                    seen_j += 1
                else:
                    if i == k:
                        seen_j += 1
                    break
            tp[j] += seen_j
        tp[k] = seen
    for j in range(n):
        for r in range(Sp[j] + tp[j], Sp[j + 1]):
            var i = Si[r]
            iw[sp[i]] = j
            sp[i] += 1
            iw[sp[j]] = i
            sp[j] += 1
    var perm = List[Int](length=n, fill=0)
    var perm_inv = List[Int](length=n, fill=0)
    _amd_2(pe, iw, lens, pfree, perm_inv, perm, dense, aggressive, last_rows)
    return perm^


def _symmetric_pattern(A: CSRMatrix) raises -> CSRMatrix:
    """The pattern of `A + A^T` (values 1), rows sorted and unique."""
    var rows = List[Int](capacity=2 * len(A.indices))
    var cols = List[Int](capacity=2 * len(A.indices))
    for i in range(A.nrows):
        for t in range(A.indptr[i], A.indptr[i + 1]):
            var j = A.indices[t]
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
    var ones = List[Float64](length=len(rows), fill=1.0)
    return CSRMatrix.from_coo(A.nrows, A.nrows, rows, cols, ones)


def _post_tree(
    root: Int,
    k_in: Int,
    mut child: List[Int],
    sibling: List[Int],
    mut order: List[Int],
    mut stack: List[Int],
) -> Int:
    """Number the subtree of `root` in postorder from `k_in` (iteratively);
    returns the next number."""
    var k = k_in
    var top = 1
    stack[0] = root
    while top > 0:
        var i = stack[top - 1]
        var c = child[i]
        if c != _NONE:
            var f = c
            while True:
                top += 1
                var nf = sibling[f]
                if nf != _NONE:
                    f = nf
                else:
                    break
            var t = top
            f = c
            while True:
                t -= 1
                stack[t] = f
                var nf = sibling[f]
                if nf != _NONE:
                    f = nf
                else:
                    break
            child[i] = _NONE
        else:
            top -= 1
            order[i] = k
            k += 1
    return k


def _postorder(
    mut order: List[Int],
    etree: List[Int],
    nv: List[Int],
    f_size: List[Int],
):
    """Postorder the assembly tree, each node's largest child last."""
    var n = len(order)
    if n == 0:
        return
    var child = List[Int](length=n, fill=_NONE)
    var sibling = List[Int](length=n, fill=_NONE)
    var stack = List[Int](length=n, fill=0)
    for j in range(n - 1, -1, -1):
        if nv[j] > 0:
            var parent = etree[j]
            if parent != _NONE:
                sibling[j] = child[parent]
                child[parent] = j
    for i in range(n):
        if nv[i] > 0 and child[i] != _NONE:
            var fprev = _NONE
            var bigfprev = _NONE
            var bigf = _NONE
            var maxfrsize = _NONE
            var f = child[i]
            while f != _NONE:
                var frsize = f_size[f]
                if frsize >= maxfrsize:
                    maxfrsize = frsize
                    bigfprev = fprev
                    bigf = f
                fprev = f
                f = sibling[f]
            var fnext = sibling[bigf]
            if fnext != _NONE:
                if bigfprev != _NONE:
                    sibling[bigfprev] = fnext
                else:
                    child[i] = fnext
                sibling[bigf] = _NONE
                sibling[fprev] = bigf
    for i in range(n):
        order[i] = _NONE
    var k = 0
    for i in range(n):
        if etree[i] == _NONE and nv[i] > 0:
            k = _post_tree(i, k, child, sibling, order, stack)


def _amd_2(
    mut pe: List[Int],
    mut iw: List[Int],
    mut lens: List[Int],
    pfree_in: Int,
    mut next: List[Int],
    mut last: List[Int],
    alpha: Float64,
    aggressive: Bool,
    last_rows: List[Bool],
):
    """SuiteSparse's `amd_2` (through faer): the minimum degree loop on the
    quotient graph in `pe`/`iw`/`lens`. On return `last` is the permutation
    (new -> old) and `next` its inverse."""
    var n = len(pe)
    var pfree = pfree_in
    var iwlen = len(iw)
    var mindeg = 0
    var lemax = 0
    var nel = 0
    var me = _NONE
    var dense: Int
    if alpha < 0:
        dense = n - 2
    else:
        dense = Int(alpha * sqrt(Float64(n)))
    dense = max(dense, 16)
    dense = min(dense, n)
    var w = List[Int](length=n, fill=1)
    var nv = List[Int](length=n, fill=1)
    var elen = List[Int](length=n, fill=0)
    var wbig = Int.MAX - n
    var wflg = 0
    var ndense = 0
    var head = List[Int](length=n, fill=_NONE)
    var degree = List[Int](length=n, fill=_NONE)
    for i in range(n):
        last[i] = _NONE
        next[i] = _NONE
        degree[i] = lens[i]
    for i in range(n):
        var deg = degree[i]
        if deg == 0 and not last_rows[i]:
            elen[i] = _flip(1)
            nel += 1
            pe[i] = _NONE
            w[i] = 0
        elif deg > dense or last_rows[i]:
            # dense, or a zero diagonal: eliminated last, after the rest
            ndense += 1
            nv[i] = 0
            elen[i] = _NONE
            pe[i] = _NONE
            nel += 1
        else:
            var inext = head[deg]
            if inext != _NONE:
                last[inext] = i
            next[i] = inext
            head[deg] = i
    while nel < n:
        var deg = mindeg
        while deg < n:
            me = head[deg]
            if me != _NONE:
                break
            deg += 1
        mindeg = deg
        var inext = next[me]
        if inext != _NONE:
            last[inext] = _NONE
        head[deg] = inext
        var elenme = elen[me]
        var nvpiv = nv[me]
        nel += nvpiv
        nv[me] = -nvpiv
        var degme = 0
        var pme1: Int
        var pme2: Int
        if elenme == 0:
            pme1 = pe[me]
            pme2 = pme1 - 1
            for p in range(pme1, pme1 + lens[me]):
                var i = iw[p]
                var nvi = nv[i]
                if nvi > 0:
                    degme += nvi
                    nv[i] = -nvi
                    pme2 += 1
                    iw[pme2] = i
                    var ilast = last[i]
                    var inx = next[i]
                    if inx != _NONE:
                        last[inx] = ilast
                    if ilast != _NONE:
                        next[ilast] = inx
                    else:
                        head[degree[i]] = inx
        else:
            var p = pe[me]
            pme1 = pfree
            var slenme = lens[me] - elenme
            for knt1 in range(1, elenme + 2):
                var e: Int
                var pj: Int
                var ln: Int
                if knt1 > elenme:
                    e = me
                    pj = p
                    ln = slenme
                else:
                    e = iw[p]
                    p += 1
                    pj = pe[e]
                    ln = lens[e]
                for knt2 in range(1, ln + 1):
                    var i = iw[pj]
                    pj += 1
                    var nvi = nv[i]
                    if nvi > 0:
                        if pfree >= iwlen:
                            # compress the workspace
                            pe[me] = p
                            lens[me] -= knt1
                            if lens[me] == 0:
                                pe[me] = _NONE
                            pe[e] = pj
                            lens[e] = ln - knt2
                            if lens[e] == 0:
                                pe[e] = _NONE
                            for j in range(n):
                                var pn = pe[j]
                                if pn >= 0:
                                    pe[j] = iw[pn]
                                    iw[pn] = _flip(j)
                            var psrc = 0
                            var pdst = 0
                            var pend = pme1
                            while psrc < pend:
                                var j = _flip(iw[psrc])
                                psrc += 1
                                if j >= 0:
                                    iw[pdst] = pe[j]
                                    pe[j] = pdst
                                    pdst += 1
                                    var lenj = lens[j]
                                    if lenj > 0:
                                        for q in range(lenj - 1):
                                            iw[pdst + q] = iw[psrc + q]
                                        psrc += lenj - 1
                                        pdst += lenj - 1
                            var p1 = pdst
                            for q in range(pfree - pme1):
                                iw[pdst + q] = iw[pme1 + q]
                            pdst += pfree - pme1
                            pme1 = p1
                            pfree = pdst
                            pj = pe[e]
                            p = pe[me]
                        degme += nvi
                        nv[i] = -nvi
                        iw[pfree] = i
                        pfree += 1
                        var ilast = last[i]
                        var inx = next[i]
                        if inx != _NONE:
                            last[inx] = ilast
                        if ilast != _NONE:
                            next[ilast] = inx
                        else:
                            head[degree[i]] = inx
                if e != me:
                    pe[e] = _flip(me)
                    w[e] = 0
            pme2 = pfree - 1
        degree[me] = degme
        pe[me] = pme1
        lens[me] = pme2 - pme1 + 1
        elen[me] = _flip(nvpiv + degme)
        wflg = _clear_flag(wflg, wbig, w)
        # degrees of the elements adjacent to the new one
        for pme in range(pme1, pme2 + 1):
            var i = iw[pme]
            var eln = elen[i]
            if eln > 0:
                var nvi = -nv[i]
                var wnvi = wflg - nvi
                for t in range(pe[i], pe[i] + eln):
                    var e = iw[t]
                    var we = w[e]
                    if we >= wflg:
                        we -= nvi
                    elif we != 0:
                        we = degree[e] + wnvi
                    w[e] = we
        # approximate degrees of the variables in the new element
        for pme in range(pme1, pme2 + 1):
            var i = iw[pme]
            var p1 = pe[i]
            var p2 = p1 + elen[i]
            var pn = p1
            var hash = 0
            deg = 0
            if aggressive:
                for p in range(p1, p2):
                    var e = iw[p]
                    var we = w[e]
                    if we != 0:
                        var dext = we - wflg
                        if dext > 0:
                            deg += dext
                            iw[pn] = e
                            pn += 1
                            hash += e
                        else:
                            pe[e] = _flip(me)
                            w[e] = 0
            else:
                for p in range(p1, p2):
                    var e = iw[p]
                    var we = w[e]
                    if we != 0:
                        deg += we - wflg
                        iw[pn] = e
                        pn += 1
                        hash += e
            elen[i] = pn - p1 + 1
            var p3 = pn
            var p4 = p1 + lens[i]
            for p in range(p2, p4):
                var j = iw[p]
                var nvj = nv[j]
                if nvj > 0:
                    deg += nvj
                    iw[pn] = j
                    pn += 1
                    hash += j
            if elen[i] == 1 and p3 == pn:
                # mass elimination: i is absorbed into the new element
                pe[i] = _flip(me)
                var nvi = -nv[i]
                degme -= nvi
                nvpiv += nvi
                nel += nvi
                nv[i] = 0
                elen[i] = _NONE
            else:
                degree[i] = min(degree[i], deg)
                iw[pn] = iw[p3]
                iw[p3] = iw[p1]
                iw[p1] = me
                lens[i] = pn - p1 + 1
                hash = hash % n
                var j = head[hash]
                if j <= _NONE:
                    next[i] = _flip(j)
                    head[hash] = _flip(i)
                else:
                    next[i] = last[j]
                    last[j] = i
                last[i] = hash
        degree[me] = degme
        lemax = max(lemax, degme)
        wflg += lemax
        wflg = _clear_flag(wflg, wbig, w)
        # supervariable detection: merge variables with equal patterns
        for pme in range(pme1, pme2 + 1):
            var i = iw[pme]
            if nv[i] < 0:
                var hash = last[i]
                var j = head[hash]
                if j == _NONE:
                    i = _NONE
                elif j < _NONE:
                    i = _flip(j)
                    head[hash] = _NONE
                else:
                    i = last[j]
                    last[j] = _NONE
                while i != _NONE and next[i] != _NONE:
                    var ln = lens[i]
                    var eln = elen[i]
                    for p in range(pe[i] + 1, pe[i] + ln):
                        w[iw[p]] = wflg
                    var jlast = i
                    var jj = next[i]
                    while jj != _NONE:
                        var ok = lens[jj] == ln and elen[jj] == eln
                        if ok:
                            for p in range(pe[jj] + 1, pe[jj] + ln):
                                if w[iw[p]] != wflg:
                                    ok = False
                                    break
                        if ok:
                            pe[jj] = _flip(i)
                            nv[i] += nv[jj]
                            nv[jj] = 0
                            elen[jj] = _NONE
                            jj = next[jj]
                            next[jlast] = jj
                        else:
                            jlast = jj
                            jj = next[jj]
                    wflg += 1
                    i = next[i]
        # restore degree lists, finalize the new element
        var p = pme1
        var nleft = n - nel
        for pme in range(pme1, pme2 + 1):
            var i = iw[pme]
            var nvi = -nv[i]
            if nvi > 0:
                nv[i] = nvi
                deg = degree[i] + degme - nvi
                deg = min(deg, nleft - nvi)
                var inx = head[deg]
                if inx != _NONE:
                    last[inx] = i
                next[i] = inx
                last[i] = _NONE
                head[deg] = i
                mindeg = min(mindeg, deg)
                degree[i] = deg
                iw[p] = i
                p += 1
        nv[me] = nvpiv
        lens[me] = p - pme1
        if lens[me] == 0:
            pe[me] = _NONE
            w[me] = 0
        if elenme != 0:
            pfree = p
    # the assembly tree: pe[i] the parent of element or variable i
    for i in range(n):
        pe[i] = _flip(pe[i])
        elen[i] = _flip(elen[i])
    for i in range(n):
        if nv[i] == 0:
            var j = pe[i]
            if j == _NONE:
                continue
            while nv[j] == 0:
                j = pe[j]
            var e = j
            j = i
            while nv[j] == 0:
                var jnext = pe[j]
                pe[j] = e
                j = jnext
    _postorder(w, pe, nv, elen)
    for i in range(n):
        head[i] = _NONE
        next[i] = _NONE
    for e in range(n):
        var k = w[e]
        if k != _NONE:
            head[k] = e
    nel = 0
    for k in range(n):
        var e = head[k]
        if e == _NONE:
            break
        next[e] = nel
        nel += nv[e]
    for i in range(n):
        if nv[i] == 0:
            var e = pe[i]
            if e != _NONE:
                next[i] = next[e]
                next[e] += 1
            else:
                next[i] = nel
                nel += 1
    for i in range(n):
        last[next[i]] = i
