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

from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
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
    _require_pseudomanifold(faces, boundary, who)
    for f in range(boundary.nrows):
        var lo = boundary.indptr[f]
        if boundary.indptr[f + 1] - lo != 2:
            continue
        if boundary.data[lo] + boundary.data[lo + 1] == 0:
            continue
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
