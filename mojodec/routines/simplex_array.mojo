# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Simplex arrays (`mojodec.routines.simplex_array`).
==================================================
Port of `pydec.dec.simplex_array`. A simplex array is an `N x (k+1)` integer
array whose rows are the vertex lists of `N` k-simplices.
"""

from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    compare_rows,
    int_array,
    int_vector,
    lexsort_rows,
    rows_of,
    cols_of,
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
    var out = int_vector(nv)
    for i in range(nv):
        var lo = 0
        var hi = ns
        while lo < hi:
            var mid = (lo + hi) // 2
            if compare_rows(s, mid, v, i) < 0:
                lo = mid + 1
            else:
                hi = mid
        if lo >= ns or compare_rows(s, lo, v, i) != 0:
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
    var row = List[Int](length=N, fill=0)
    for i in range(M):
        for j in range(N):
            row[j] = Int(s.unsafe_get(i * N + j))
        # count transpositions of a selection sort
        var trans = 0
        for j in range(N - 1):
            var m = j
            for k in range(j + 1, N):
                if row[k] < row[m]:
                    m = k
            if m != j:
                row.swap_elements(j, m)
                trans += 1
        out.unsafe_set(i, Scalar[DType.int](trans % 2))
    return out^


def simplex_array_boundary(
    s: IntArray, parity: IntArray
) raises -> Tuple[IntArray, CSRMatrix]:
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

    var rows = List[Int](capacity=num_faces)
    var cols = List[Int](capacity=num_faces)
    var vals = List[Float64](capacity=num_faces)
    for t in range(num_faces):
        rows.append(face_index[t])
        cols.append(owner[perm[t]])
        vals.append(orient[perm[t]])
    var B = CSRMatrix.from_coo(
        len(unique_rows), num_simplices, rows, cols, vals
    )
    return (unique_faces^, B^)
