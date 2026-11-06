# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Finite element inner products (`mojodec.routines.innerproduct`).
================================================================
Port of `pydec.fem.innerproduct`: mass matrices of Whitney k-forms on a
simplicial complex, and of the analogous tensor-product elements on a regular
cube complex.
"""

from std.math import sqrt

from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    factorial,
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
    var rc = massmatrix_rowcols(sc, k)

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

    # G is the Gram matrix of the barycentric gradients of a top simplex:
    # with the edge vectors E[r] = x_{r+1} - x_0 as rows, gradients 1..p
    # are the rows of (E E^T)^-1 E, and gradient 0 is minus their sum. The
    # determinant of pair (a, b) is that of the k x k block of G in the rows
    # of form a and the columns of form b (1 when k = 0).
    var dim = cols_of(sc.vertices())
    var edges = List[Float64](length=p * dim, fill=0)
    var gram = List[Float64](length=p * p, fill=0)
    var grads = List[Float64](length=(p + 1) * dim, fill=0)
    var G = List[Float64](length=(p + 1) * (p + 1), fill=0)
    var dets = List[Float64](length=num_forms * num_forms, fill=1)
    var work = List[Float64](length=k * k, fill=0)
    var ep = edges.unsafe_ptr()
    var gp = gram.unsafe_ptr()
    var Dp = grads.unsafe_ptr()
    var Gp = G.unsafe_ptr()
    var dp = dets.unsafe_ptr()
    var tsp = term_start.unsafe_ptr()
    var tpp = term_pair.unsafe_ptr()
    var tcp = term_coef.unsafe_ptr()

    var data = List[Float64](length=len(rc.rows), fill=0)
    var datap = data.unsafe_ptr()
    ref X = sc.vertices()
    ref top = sc[p]
    ref simplices = top.simplices
    for i in range(top.num_simplices):
        if k > 0:
            var v0 = Int(simplices.unsafe_get(i * (p + 1)))
            for r in range(p):
                var v = Int(simplices.unsafe_get(i * (p + 1) + r + 1))
                for c in range(dim):
                    ep[unsafe_offset=r * dim + c] = X.unsafe_get(
                        v * dim + c
                    ) - X.unsafe_get(v0 * dim + c)
            for r in range(p):
                for s in range(r + 1):
                    var t: Float64 = 0
                    for c in range(dim):
                        t += (
                            ep[unsafe_offset=r * dim + c]
                            * ep[unsafe_offset=s * dim + c]
                        )
                    gp[unsafe_offset=r * p + s] = t
            if not _cholesky_solve(gram, edges, p, dim):
                raise Error(
                    String(
                        "whitney_innerproduct: top simplex ",
                        i,
                        " is degenerate",
                    )
                )
            for c in range(dim):
                var t: Float64 = 0
                for r in range(p):
                    var g = ep[unsafe_offset=r * dim + c]
                    Dp[unsafe_offset=(r + 1) * dim + c] = g
                    t += g
                Dp[unsafe_offset=c] = -t
            for r in range(p + 1):
                for s in range(r + 1):
                    var t: Float64 = 0
                    for c in range(dim):
                        t += (
                            Dp[unsafe_offset=r * dim + c]
                            * Dp[unsafe_offset=s * dim + c]
                        )
                    Gp[unsafe_offset=r * (p + 1) + s] = t
                    Gp[unsafe_offset=s * (p + 1) + r] = t
            for a in range(num_forms):
                for b in range(a, num_forms):
                    var det = _minor_det(G, p + 1, forms, a * k, b * k, k, work)
                    dp[unsafe_offset=a * num_forms + b] = det
                    dp[unsafe_offset=b * num_forms + a] = det
        var weight = abs(top.primal_volume.unsafe_get(i)) * scale_integration
        for t in range(nfp):
            var s: Float64 = 0
            for q in range(tsp[unsafe_offset=t], tsp[unsafe_offset=t + 1]):
                s += (
                    tcp[unsafe_offset=q]
                    * dp[unsafe_offset=tpp[unsafe_offset=q]]
                )
            datap[unsafe_offset=i * nfp + t] = s * weight
    var N = sc[k].num_simplices
    return CSRMatrix.from_coo(N, N, rc.rows, rc.cols, data)


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


def _cholesky_solve(
    mut a: List[Float64], mut b: List[Float64], n: Int, m: Int
) -> Bool:
    """Overwrite the row-major `n x m` matrix `b` with `A^-1 b`, for the
    symmetric positive definite `n x n` matrix `A` whose lower triangle is in
    `a` (row-major).

    `a` is overwritten with its Cholesky factor. Returns False, leaving `b`
    undefined, if a pivot is not positive.
    """
    var L = a.unsafe_ptr()
    var B = b.unsafe_ptr()
    # A = L L^T, L in the lower triangle of a
    for j in range(n):
        var s = L[unsafe_offset=j * n + j]
        for c in range(j):
            s -= L[unsafe_offset=j * n + c] * L[unsafe_offset=j * n + c]
        if s <= 0:
            return False
        var d = sqrt(s)
        L[unsafe_offset=j * n + j] = d
        for i in range(j + 1, n):
            var t = L[unsafe_offset=i * n + j]
            for c in range(j):
                t -= L[unsafe_offset=i * n + c] * L[unsafe_offset=j * n + c]
            L[unsafe_offset=i * n + j] = t / d
    for col in range(m):
        # L y = b, then L^T x = y
        for i in range(n):
            var t = B[unsafe_offset=i * m + col]
            for c in range(i):
                t -= L[unsafe_offset=i * n + c] * B[unsafe_offset=c * m + col]
            B[unsafe_offset=i * m + col] = t / L[unsafe_offset=i * n + i]
        for i in range(n - 1, -1, -1):
            var t = B[unsafe_offset=i * m + col]
            for c in range(i + 1, n):
                t -= L[unsafe_offset=c * n + i] * B[unsafe_offset=c * m + col]
            B[unsafe_offset=i * m + col] = t / L[unsafe_offset=i * n + i]
    return True


def _minor_det(
    gram: List[Float64],
    n: Int,
    form_list: List[Int],
    rows: Int,
    cols: Int,
    k: Int,
    mut scratch: List[Float64],
) -> Float64:
    """The determinant of the k x k block of the row-major `n x n` matrix
    `gram` in rows `form_list[rows:rows + k]` and columns
    `form_list[cols:cols + k]` (1 when k = 0); `scratch` holds `k * k`
    values.
    """
    var G = gram.unsafe_ptr()
    var forms = form_list.unsafe_ptr()
    var work = scratch.unsafe_ptr()

    @always_inline
    def g(a: Int, b: Int) {G, n, forms, rows, cols} -> Float64:
        return G[
            unsafe_offset=forms[unsafe_offset=rows + a] * n
            + forms[unsafe_offset=cols + b]
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
