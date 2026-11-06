# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Simplex geometry (`mojodec.routines.geometry`).
===============================================
Volumes and circumcenters of simplices, ports of `pydec.math.volume` and
`pydec.math.circumcenter`. A simplex is given as an `(M+1) x N` NuMojo array
whose rows are the coordinates of its `M+1` vertices in `N`-space; the dense
small dense solves use the pivoted LU of `mojodec.routines.dense`.
"""

import numojo as nm
from numojo.prelude import *

from mojodec.core.utils import (
    RealArray,
    real_array,
    real_vector,
    factorial,
    rows_of,
    cols_of,
)
from mojodec.routines import dense


def unsigned_volume(pts: RealArray) raises -> Float64:
    """Unsigned volume of an M-simplex embedded in N-space (M <= N).

    Zero-dimensional simplices (points) have unit volume.

    ```mojo
    unsigned_volume(nm.array[f64]("[[0, 0, 0], [0, 1, 0], [1, 0, 0]]"))  # 0.5
    ```

    Args:
        pts: `(M+1) x N` vertex coordinates.

    Returns:
        The M-dimensional volume, `sqrt(det(G)) / M!` for the Gram matrix
        `G` of the edge vectors from the first vertex.

    Raises:
        If `M < 0` or `M > N`.
    """
    var M = rows_of(pts) - 1
    var N = cols_of(pts)
    if M < 0 or M > N:
        raise Error("unsigned_volume: array has invalid shape")
    if M == 0:
        return 1.0
    var A = _edge_vectors(pts)
    var gram = real_array(M, M)
    for i in range(M):
        for j in range(M):
            var s: Float64 = 0
            for k in range(N):
                s += A.unsafe_get(i * N + k) * A.unsafe_get(j * N + k)
            gram.unsafe_set(i * M + j, s)
    return abs(_det(gram)) ** 0.5 / factorial(M)


def signed_volume(pts: RealArray) raises -> Float64:
    """Signed volume of an M-simplex embedded in M-space.

    ```mojo
    signed_volume(nm.array[f64]("[[0, 0], [1, 0], [0, 1]]"))  # 0.5
    ```

    Args:
        pts: `(M+1) x M` vertex coordinates.

    Returns:
        The volume, positive when the vertices are positively oriented.

    Raises:
        If `pts` is not `(M+1) x M`.
    """
    var M = rows_of(pts) - 1
    var N = cols_of(pts)
    if M != N:
        raise Error("signed_volume: array has invalid shape")
    return _det(_edge_vectors(pts)) / factorial(M)


def circumcenter_barycentric(pts: RealArray) raises -> RealArray:
    """Barycentric coordinates of the circumcenter of a simplex.

    `pts` is `N x K` with `1 <= N <= K + 1`; the result has length `N`.

    ```mojo
    circumcenter_barycentric(nm.array[f64]("[[0, 0], [4, 0], [0, 4]]"))
    # [0, 0.5, 0.5]
    ```

    Args:
        pts: `N x K` vertex coordinates.

    Returns:
        The `N` barycentric coordinates, summing to 1.

    Raises:
        If `N > K + 1`, or the simplex is degenerate (singular system).
    """
    var rows = rows_of(pts)
    var cols = cols_of(pts)
    if rows > cols + 1:
        raise Error("circumcenter_barycentric: too many points")
    # [[2 P P^T, 1], [1^T, 0]] [x; mu] = [|p_i|^2; 1]
    var n = rows + 1
    var A = real_array(n, n)
    var b = real_array(n, 1)
    for i in range(rows):
        for j in range(rows):
            var s: Float64 = 0
            for k in range(cols):
                s += pts.unsafe_get(i * cols + k) * pts.unsafe_get(j * cols + k)
            A.unsafe_set(i * n + j, 2 * s)
            if i == j:
                b.unsafe_set(i, s)
        A.unsafe_set(i * n + rows, 1.0)
        A.unsafe_set(rows * n + i, 1.0)
    b.unsafe_set(rows, 1.0)
    var x = _solve(A, b)
    var out = real_vector(rows)
    for i in range(rows):
        out.unsafe_set(i, x.unsafe_get(i))
    return out^


def circumcenter(pts: RealArray) raises -> Tuple[RealArray, Float64]:
    """Circumcenter and circumradius of a simplex.

    ```mojo
    var (c, r) = circumcenter(nm.array[f64]("[[0, 0], [1, 0], [0, 1]]"))
    # c == [0.5, 0.5], r == 0.7071...
    ```

    Args:
        pts: `N x K` vertex coordinates, `N <= K + 1`.

    Returns:
        The circumcenter (length `K`) and the circumradius.

    Raises:
        If `N > K + 1`, or the simplex is degenerate.
    """
    var bary = circumcenter_barycentric(pts)
    var rows = rows_of(pts)
    var cols = cols_of(pts)
    var center = real_vector(cols)
    for k in range(cols):
        var s: Float64 = 0
        for i in range(rows):
            s += bary.unsafe_get(i) * pts.unsafe_get(i * cols + k)
        center.unsafe_set(k, s)
    var r2: Float64 = 0
    for k in range(cols):
        var d = pts.unsafe_get(k) - center.unsafe_get(k)
        r2 += d * d
    return (center^, r2**0.5)


def is_wellcentered(pts: RealArray, tol: Float64 = 1e-8) raises -> Bool:
    """Whether the circumcenter lies strictly inside the simplex.

    Args:
        pts: `N x K` vertex coordinates, `N <= K + 1`.
        tol: Every barycentric coordinate of the circumcenter must exceed
            this.

    Returns:
        True if the simplex is well-centered.

    Raises:
        If `N > K + 1`, or the simplex is degenerate.
    """
    var bary = circumcenter_barycentric(pts)
    for i in range(bary.size):
        if bary.unsafe_get(i) <= tol:
            return False
    return True


def barycentric_gradients(pts: RealArray) raises -> RealArray:
    """Gradients of the barycentric basis functions over a simplex.

    For an M-simplex in N-space returns an `(M+1) x N` array whose row `i`
    is the (constant) gradient of the `i`-th barycentric coordinate.

    Args:
        pts: `(M+1) x N` vertex coordinates.

    Returns:
        The `(M+1) x N` gradients, tangent to the simplex.

    Raises:
        If the simplex is degenerate (singular Gram matrix).
    """
    var M = rows_of(pts) - 1
    var N = cols_of(pts)
    var V = _edge_vectors(pts)
    # grads[1:] = inv(V V^T) V
    var gram = real_array(M, M)
    for i in range(M):
        for j in range(M):
            var s: Float64 = 0
            for k in range(N):
                s += V.unsafe_get(i * N + k) * V.unsafe_get(j * N + k)
            gram.unsafe_set(i * M + j, s)
    var G = _solve(gram, V)
    var out = real_array(M + 1, N)
    for k in range(N):
        var s: Float64 = 0
        for i in range(M):
            var g = G.unsafe_get(i * N + k)
            out.unsafe_set((i + 1) * N + k, g)
            s += g
        out.unsafe_set(k, -s)
    return out^


def _edge_vectors(pts: RealArray) raises -> RealArray:
    """The `M x N` array `pts[1:] - pts[0]`."""
    var M = rows_of(pts) - 1
    var N = cols_of(pts)
    var A = real_array(M, N)
    for i in range(M):
        for k in range(N):
            A.unsafe_set(
                i * N + k, pts.unsafe_get((i + 1) * N + k) - pts.unsafe_get(k)
            )
    return A^


def _det(A: RealArray) raises -> Float64:
    return dense.det(A)


def _solve(A: RealArray, B: RealArray) raises -> RealArray:
    return dense.solve(A, B)
