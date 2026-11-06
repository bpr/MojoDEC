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

import numojo as nm
from numojo.prelude import *

from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    RealArray,
    factorial,
    fget,
    iget,
    int_array,
    int_array_from_rows,
    real_array,
    row_list,
    rows_of,
    cols_of,
    take_rows,
)
from mojodec.routines import dense
from mojodec.routines.combinatorial import combinations
from mojodec.routines.cube_array import cube_array_boundary, cube_array_search
from mojodec.routines.geometry import barycentric_gradients
from mojodec.routines.simplex_array import simplex_array_searchsorted


def massmatrix_rowcols(
    sc: SimplicialComplex, k: Int
) raises -> Tuple[List[Int], List[Int]]:
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
        return (rows^, cols^)
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
    for s in range(num_simplices):
        for n in range(fps):
            for m in range(fps):
                rows.append(Int(indices.unsafe_get(s * fps + n)))
                cols.append(Int(indices.unsafe_get(s * fps + m)))
    return (rows^, cols^)


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
    var num_k_faces = len(k_faces)

    # unordered pairs of k-forms, then the diagonal pairs
    var pair_a = List[Int]()
    var pair_b = List[Int]()
    for i in range(len(k_forms)):
        for j in range(i + 1, len(k_forms)):
            pair_a.append(i)
            pair_b.append(j)
    for i in range(len(k_forms)):
        pair_a.append(i)
        pair_b.append(i)
    var num_pairs = len(pair_a)

    # dets_to_vals maps the vector of pair determinants to the local matrix
    var dr = List[Int]()
    var dc = List[Int]()
    var dv = List[Float64]()
    for f1 in range(num_k_faces):
        for f2 in range(num_k_faces):
            var row = f1 * num_k_faces + f2
            for n in range(k + 1):
                for m in range(k + 1):
                    var form1 = _without(k_faces[f1], n)
                    var form2 = _without(k_faces[f2], m)
                    var col = _pair_index(k_forms, pair_a, pair_b, form1, form2)
                    var same = 1 if k_faces[f1][n] == k_faces[f2][m] else 0
                    var sign: Float64 = 1.0 if (n + m) % 2 == 0 else -1.0
                    dr.append(row)
                    dc.append(col)
                    dv.append(sign * Float64(same + 1))
    var dets_to_vals = CSRMatrix.from_coo(
        num_k_faces * num_k_faces, num_pairs, dr, dc, dv
    )

    var nfp = num_k_faces * num_k_faces
    var data = List[Float64](length=len(rc[0]), fill=0)
    var dets = nm.ones[DType.float64](Shape(num_pairs))
    ref top = sc[p]
    for i in range(top.num_simplices):
        if k > 0:
            var pts = take_rows(sc.vertices(), row_list(top.simplices, i))
            var dl = barycentric_gradients(pts)
            var E = cols_of(dl)
            for t in range(num_pairs):
                ref A = k_forms[pair_a[t]]
                ref B = k_forms[pair_b[t]]
                var G = real_array(k, k)
                for a in range(k):
                    for b in range(k):
                        var s: Float64 = 0
                        for e in range(E):
                            s += fget(dl, A[a], e) * fget(dl, B[b], e)
                        G.unsafe_set(a * k + b, s)
                dets.unsafe_set(t, dense.det(G))
        var volume = abs(top.primal_volume.unsafe_get(i))
        var vals = dets_to_vals @ dets
        for t in range(nfp):
            data[i * nfp + t] = vals.unsafe_get(t) * volume * scale_integration
    var N = sc[k].num_simplices
    return CSRMatrix.from_coo(N, N, rc[0], rc[1], data)


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
        standard = cube_array_boundary(standard, i)[0].copy()
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


def _pair_index(
    forms: List[List[Int]],
    pair_a: List[Int],
    pair_b: List[Int],
    f1: List[Int],
    f2: List[Int],
) raises -> Int:
    var a = -1
    var b = -1
    for i in range(len(forms)):
        if forms[i] == f1:
            a = i
        if forms[i] == f2:
            b = i
    for t in range(len(pair_a)):
        if (pair_a[t] == a and pair_b[t] == b) or (
            pair_a[t] == b and pair_b[t] == a
        ):
            return t
    raise Error("whitney_innerproduct: k-form pair not found")
