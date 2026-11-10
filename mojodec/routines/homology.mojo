# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Homology (`mojodec.routines.homology`).
=======================================
Betti numbers of a complex, of a subcomplex, and of a pair; not in PyDEC.

The k-th Betti number `b_k = N_k - rank B_k - rank B_{k+1}` is the
dimension of the k-th homology with rational coefficients: `b_0` counts
connected pieces, `b_1` independent loops, `b_2` enclosed voids. The
relative Betti numbers of a pair `(K, L)` use the boundary operators with
the cells of `L` removed from both rows and columns; relative to the
boundary of a manifold they give the harmonic forms that vanish on the
boundary (Dirichlet conditions), and by Lefschetz duality
`b_k(K, ∂K) = b_{n-k}(K)` for an orientable n-manifold.

Ranks are computed exactly by sparse elimination modulo the prime
`p = 1073741789` (largest below 2^30). The rank modulo p equals the
rational rank unless p divides a torsion coefficient of the integral
homology, which needs a torsion group of order at least p; real meshes
have torsion of order 2 at most (e.g. the projective plane). Products of
residues fit in 64 bits, so no big integers are needed.

```mojo
var b = sc.betti_numbers()                 # [1, 2, 1] for a torus surface
var r = sc.relative_betti_numbers()        # relative to the boundary
var lk = sc.betti_numbers(sc.link(Simplex([v])))  # [1, 0, 1, 0] inside tets
```
"""

from std.math import floor

from mojodec.core.cell_set import CellSet
from mojodec.core.chain_complex import ChainComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.routines.topology import boundary_of, closure_of, link_of

comptime _PRIME = 1073741789
"""The modulus of the elimination: the largest prime below 2^30."""


def betti_numbers(cc: ChainComplex) raises -> List[Int]:
    """The Betti numbers of a complex, over the rationals.

    Args:
        cc: The complex.

    Returns:
        `b_0, ..., b_n`; their alternating sum is the Euler characteristic.

    Raises:
        If an operator entry is not an integer.
    """
    var keep = List[List[Bool]]()
    for k in range(cc.complex_dimension() + 1):
        keep.append(List[Bool](length=cc.num_cells(k), fill=True))
    return _betti(cc, keep)


def betti_numbers(cc: ChainComplex, cells: CellSet) raises -> List[Int]:
    """The Betti numbers of a subcomplex.

    Args:
        cc: The complex.
        cells: A subcomplex of `cc`: every face of a cell in it is in it
            (`closure_of` makes one).

    Returns:
        `b_0, ..., b_n` of the subcomplex, n the dimension of `cc`.

    Raises:
        If `cells` does not belong to `cc` or is not closed, or an operator
        entry is not an integer.
    """
    _require_subcomplex(cc, cells, "betti_numbers")
    var keep = List[List[Bool]]()
    for k in range(len(cells.levels)):
        keep.append(cells[k].mask())
    return _betti(cc, keep)


def relative_betti_numbers(cc: ChainComplex, sub: CellSet) raises -> List[Int]:
    """The Betti numbers of the pair `(cc, sub)`, over the rationals.

    The homology of the cells of `cc` not in `sub`, with chains on `sub`
    counted as zero.

    Args:
        cc: The complex.
        sub: A subcomplex of `cc`, such as its boundary
            (`boundary_of(cc)`).

    Returns:
        `b_0, ..., b_n` of the pair.

    Raises:
        If `sub` does not belong to `cc` or is not closed, or an operator
        entry is not an integer.
    """
    _require_subcomplex(cc, sub, "relative_betti_numbers")
    var keep = List[List[Bool]]()
    for k in range(len(sub.levels)):
        var mask = sub[k].mask()
        for i in range(len(mask)):
            mask[i] = not mask[i]
        keep.append(mask^)
    return _betti(cc, keep)


def non_manifold_cells(cc: ChainComplex) raises -> CellSet:
    """The cells around which a complex is not a homology manifold.

    A complex of dimension n is a homology n-manifold, possibly with
    boundary, when the link of every k-cell (k < n) has the rational
    homology of the (n-k-1)-sphere if the cell is inside, or of a point
    if it is on the boundary (`boundary_of(cc)`). Every triangulated
    manifold is one. The test covers cells of every dimension, not only
    vertices: from dimension 3 on, a vertex link can have the homology of
    a sphere while an edge link does not.

    Args:
        cc: The complex.

    Returns:
        The k-cells (k < n) whose links fail the test, such as the tip
        where two cones meet, an edge of three triangles, or a cell of a
        lower-dimensional piece. Empty for a homology manifold.

    Raises:
        If an operator entry is not an integer.
    """
    var n = cc.complex_dimension()
    var out = CellSet.empty(cc)
    if n == 0:
        return out^
    var bd = boundary_of(cc)
    var row_id = List[List[Int]]()
    for k in range(n + 1):
        row_id.append(List[Int](length=cc.num_cells(k), fill=-1))
    for k in range(n):
        var m = n - k - 1
        for i in range(cc.num_cells(k)):
            var lk = link_of(cc, CellSet.from_indices(cc, k, [i]))
            var b = _betti_of(cc, lk.indices(), row_id)
            var ok = True
            for j in range(n + 1):
                var want: Int
                if i in bd[k]:
                    want = 1 if j == 0 else 0  # a point
                elif m == 0:
                    want = 2 if j == 0 else 0  # two points
                else:
                    want = 1 if j == 0 or j == m else 0
                if b[j] != want:
                    ok = False
                    break
            if not ok:
                out[k].add(i)
    return out^


def _require_subcomplex(cc: ChainComplex, cells: CellSet, who: String) raises:
    cells.check_complex(cc)
    if closure_of(cc, cells) != cells:
        raise Error(
            who,
            ": the cells are not a subcomplex (a face of a cell is missing)",
        )


def _betti(cc: ChainComplex, keep: List[List[Bool]]) raises -> List[Int]:
    """Betti numbers of the chain complex restricted to the kept cells."""
    var cells = List[List[Int]]()
    var row_id = List[List[Int]]()
    for k in range(len(keep)):
        var idx = List[Int]()
        for i in range(len(keep[k])):
            if keep[k][i]:
                idx.append(i)
        cells.append(idx^)
        row_id.append(List[Int](length=len(keep[k]), fill=-1))
    return _betti_of(cc, cells, row_id)


def _betti_of(
    cc: ChainComplex, cells: List[List[Int]], mut row_id: List[List[Int]]
) raises -> List[Int]:
    """Betti numbers of the chain complex restricted to `cells` (indices
    per dimension), touching only those cells.

    `row_id[k]` is scratch, one entry per k-cell, all -1 on entry and exit.
    """
    var n = cc.complex_dimension()
    for k in range(n + 1):
        for i in range(len(cells[k])):
            row_id[k][cells[k][i]] = i
    # rank[k] = rank of B_k on the cells, k = 1..n; zero at 0 and n+1
    var rank = List[Int](length=n + 2, fill=0)
    for k in range(1, n + 1):
        rank[k] = _rank_mod_p(
            cc.coboundary(k - 1), cells[k], row_id[k - 1], len(cells[k - 1])
        )
    for k in range(n + 1):
        for c in cells[k]:
            row_id[k][c] = -1
    var out = List[Int](capacity=n + 1)
    for k in range(n + 1):
        out.append(len(cells[k]) - rank[k] - rank[k + 1])
    return out^


def _rank_mod_p(
    d: CSRMatrix, cells: List[Int], face_id: List[Int], num_faces: Int
) raises -> Int:
    """Rank modulo `_PRIME` of `B = d^T` on the given cells (columns of B)
    and the faces with `face_id >= 0` (rows of B, renumbered by it).

    Reduces the columns of B in order, each against the stored pivot
    columns by its last (largest) face, as in persistent homology. Stored
    columns are scaled so their last entry is 1.
    """
    comptime p = _PRIME
    var pivot = List[Int](length=num_faces, fill=-1)
    var piv_rows = List[List[Int]]()
    var piv_vals = List[List[Int]]()
    var rows = List[Int]()
    var vals = List[Int]()
    var tmp_rows = List[Int]()
    var tmp_vals = List[Int]()
    for c in cells:
        rows.clear()
        vals.clear()
        for q in range(d.indptr[c], d.indptr[c + 1]):
            var r = face_id[d.indices[q]]
            if r < 0:
                continue
            var x = d.data[q]
            if x != floor(x) or abs(x) >= Float64(p):
                raise Error(
                    "betti_numbers: boundary entries must be integers, got ",
                    x,
                )
            var v = Int(x) % p
            if v < 0:
                v += p
            if v == 0:
                continue
            # insert in order of r (rows of d are short)
            rows.append(r)
            vals.append(v)
            var j = len(rows) - 1
            while j > 0 and rows[j - 1] > rows[j]:
                rows.swap_elements(j - 1, j)
                vals.swap_elements(j - 1, j)
                j -= 1
        while len(rows) > 0:
            var low = rows[len(rows) - 1]
            var s = pivot[low]
            if s < 0:
                var inv = _inverse_mod(vals[len(vals) - 1], p)
                for j in range(len(vals)):
                    vals[j] = vals[j] * inv % p
                pivot[low] = len(piv_rows)
                piv_rows.append(rows.copy())
                piv_vals.append(vals.copy())
                break
            # column -= f * pivot column, which clears the last entry
            var f = vals[len(vals) - 1]
            ref pr = piv_rows[s]
            ref pv = piv_vals[s]
            tmp_rows.clear()
            tmp_vals.clear()
            var a = 0
            var b = 0
            while a < len(rows) or b < len(pr):
                if b == len(pr) or (a < len(rows) and rows[a] < pr[b]):
                    tmp_rows.append(rows[a])
                    tmp_vals.append(vals[a])
                    a += 1
                elif a == len(rows) or pr[b] < rows[a]:
                    tmp_rows.append(pr[b])
                    tmp_vals.append((p - f * pv[b] % p) % p)
                    b += 1
                else:
                    var v = (vals[a] - f * pv[b] % p + p) % p
                    if v != 0:
                        tmp_rows.append(rows[a])
                        tmp_vals.append(v)
                    a += 1
                    b += 1
            swap(rows, tmp_rows)
            swap(vals, tmp_vals)
    return len(piv_rows)


def _inverse_mod(a: Int, p: Int) -> Int:
    """`a^(p-2) mod p`, the inverse of `a` modulo the prime `p`."""
    var result = 1
    var base = a % p
    var e = p - 2
    while e > 0:
        if e & 1:
            result = result * base % p
        base = base * base % p
        e >>= 1
    return result
