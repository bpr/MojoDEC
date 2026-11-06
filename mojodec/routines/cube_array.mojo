# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Cube arrays (`mojodec.routines.cube_array`).
============================================
Port of `pydec.dec.cube_array`. In an N-dimensional regular grid a k-cube is
a row `[c_0, ..., c_{N-1}, a_0, ..., a_{k-1}]`: its lowest grid corner
followed by the (increasing) axes it spans.
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


def cube_array_search(
    k_face_array: IntArray, k_faces: IntArray
) raises -> IntArray:
    """Row indices in `k_face_array` of the cubes stored in the rows of `k_faces`.

    ```mojo
    # k_face_array = [[0,0,0],[0,0,1],[0,1,0],[1,0,1]], k_faces = [[0,1,0],[0,0,1]]
    # -> [2, 1]
    ```

    Args:
        k_face_array: The cubes to search, one per row.
        k_faces: The cubes to look up, with the same number of columns.

    Returns:
        For each row of `k_faces`, the index of the equal row of
        `k_face_array`.

    Raises:
        If either array is not 2-D, their column counts differ, or a cube
            of `k_faces` is not in `k_face_array`.
    """
    if k_face_array.ndim != 2 or k_faces.ndim != 2:
        raise Error("cube_array_search: expected rank 2 arrays")
    if cols_of(k_face_array) != cols_of(k_faces):
        raise Error("cube_array_search: number of columns must agree")
    var c = cols_of(k_face_array)
    # mixed-radix code of each row, radix = column maximum + 1
    var radix = List[Int](length=c, fill=1)
    for i in range(rows_of(k_face_array)):
        for j in range(c):
            radix[j] = max(
                radix[j], Int(k_face_array.unsafe_get(i * c + j)) + 1
            )
    var lookup = Dict[Int, Int]()
    for i in range(rows_of(k_face_array)):
        lookup[_code(k_face_array, i, radix)] = i
    var out = int_vector(rows_of(k_faces))
    for i in range(rows_of(k_faces)):
        for j in range(c):
            var x = Int(k_faces.unsafe_get(i * c + j))
            if x < 0 or x >= radix[j]:
                raise Error("cube_array_search: cube not found")
        var key = _code(k_faces, i, radix)
        if key not in lookup:
            raise Error("cube_array_search: cube not found")
        out.unsafe_set(i, Scalar[DType.int](lookup[key]))
    return out^


def cube_array_boundary(
    cubes: IntArray, dim: Int
) raises -> Tuple[IntArray, CSRMatrix]:
    """Faces and boundary operator of an array of `dim`-cubes in a regular grid.

    Each `dim`-cube has `2 * dim` faces: for its i-th axis, the face at its
    lowest corner with sign `(-1)^(i+1)` and the opposite face with sign
    `(-1)^i`.

    Args:
        cubes: `num_cubes x (N + dim)` cube rows (corner, then axes).
        dim: The dimension of the cubes.

    Returns:
        The unique (dim-1)-faces in lexicographic order, and the
        `num_faces x num_cubes` boundary matrix.

    Raises:
        Does not raise in practice.
    """
    var cube_dim = dim
    var width = cols_of(cubes)
    var top_dim = width - cube_dim
    var num_cubes = rows_of(cubes)
    var num_faces = 2 * num_cubes * cube_dim
    var fw = width - 1
    var faces = int_array(num_faces, fw)
    var owner = List[Int](length=num_faces, fill=0)
    var sign = List[Float64](length=num_faces, fill=0)
    for i in range(cube_dim):
        var drop = top_dim + i
        for side in range(2):
            for r in range(num_cubes):
                var row = (2 * i + side) * num_cubes + r
                var c = 0
                for j in range(width):
                    if j != drop:
                        faces.unsafe_set(
                            row * fw + c, cubes.unsafe_get(r * width + j)
                        )
                        c += 1
                owner[row] = r
                # faces at the cube's origin get (-1)^(i+1), the far ones (-1)^(i+2)
                sign[row] = 1.0 if (i + 1 + side) % 2 == 0 else -1.0
                if side == 1:
                    var axis = Int(cubes.unsafe_get(r * width + drop))
                    faces.unsafe_set(
                        row * fw + axis, faces.unsafe_get(row * fw + axis) + 1
                    )

    var perm = lexsort_rows(faces)
    var unique_rows = List[Int]()
    var rows = List[Int](capacity=num_faces)
    var cols = List[Int](capacity=num_faces)
    var vals = List[Float64](capacity=num_faces)
    for t in range(num_faces):
        var p = perm[t]
        if t == 0 or compare_rows(faces, p, faces, perm[t - 1]) != 0:
            unique_rows.append(p)
        rows.append(len(unique_rows) - 1)
        cols.append(owner[p])
        vals.append(sign[p])
    var unique_faces = int_array(len(unique_rows), fw)
    for i in range(len(unique_rows)):
        for j in range(fw):
            unique_faces.unsafe_set(
                i * fw + j, faces.unsafe_get(unique_rows[i] * fw + j)
            )
    var B = CSRMatrix.from_coo(len(unique_rows), num_cubes, rows, cols, vals)
    return (unique_faces^, B^)


def _code(a: IntArray, i: Int, radix: List[Int]) -> Int:
    var c = len(radix)
    var key = 0
    for j in range(c):
        key = key * radix[j] + Int(a.unsafe_get(i * c + j))
    return key
