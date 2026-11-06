# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Mesh quality (`mojodec.routines.quality`).
==========================================
Per-simplex diagnostics for the top simplices of a complex; not in PyDEC.

Circumcentric DEC is most reliable on well-centered meshes, whose
simplices contain their own circumcenters: there every dual volume, and so
every diagonal Hodge star, is positive. `well_centeredness` says how far
each simplex is from that.
"""

from std.math import sqrt

from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import (
    RealArray,
    cols_of,
    fget,
    real_vector,
    row_list,
    rows_of,
    take_rows,
)
from mojodec.routines.geometry import circumcenter_barycentric


def well_centeredness(sc: SimplicialComplex) raises -> RealArray:
    """The smallest barycentric coordinate of each top simplex's circumcenter.

    Positive for a well-centered simplex (circumcenter strictly inside),
    zero when the circumcenter lies on a face (a right triangle), and
    negative, increasingly so, as it moves outside (an obtuse triangle).

    Args:
        sc: A complex (geometry need not be computed).

    Returns:
        The value for each top simplex.

    Raises:
        If a top simplex is degenerate.
    """
    ref top = sc.simplices()
    var n = cols_of(top)
    var out = real_vector(rows_of(top))
    for t in range(out.size):
        var b = circumcenter_barycentric(
            take_rows(sc.vertices(), row_list(top, t))
        )
        var m = b.unsafe_get(0)
        for i in range(1, n):
            m = min(m, b.unsafe_get(i))
        out.unsafe_set(t, m)
    return out^


def well_centered_fraction(
    sc: SimplicialComplex, tol: Float64 = 0
) raises -> Float64:
    """The fraction of top simplices with `well_centeredness > tol`.

    Args:
        sc: A complex (geometry need not be computed).
        tol: The threshold on `well_centeredness`.

    Returns:
        A fraction in `[0, 1]`.

    Raises:
        If a top simplex is degenerate.
    """
    var w = well_centeredness(sc)
    var count = 0
    for i in range(w.size):
        if w.unsafe_get(i) > tol:
            count += 1
    return Float64(count) / Float64(max(w.size, 1))


def shape_regularity(sc: SimplicialComplex) raises -> RealArray:
    """`1 - std/mean` of the edge lengths of each top simplex.

    1 for a regular simplex (all edges equal), smaller as the edge lengths
    spread out. Top simplices of dimension below 2 score 1, and simplices
    whose edges all have zero length score 0.

    Args:
        sc: A complex (geometry need not be computed).

    Returns:
        The value for each top simplex.

    Raises:
        Does not raise in practice.
    """
    ref top = sc.simplices()
    ref V = sc.vertices()
    var n = cols_of(top)
    var E = cols_of(V)
    var out = real_vector(rows_of(top))
    for t in range(out.size):
        if n < 3:
            out.unsafe_set(t, 1)
            continue
        var verts = row_list(top, t)
        var lengths = List[Float64]()
        for i in range(n):
            for j in range(i + 1, n):
                var s: Float64 = 0
                for k in range(E):
                    s += (fget(V, verts[j], k) - fget(V, verts[i], k)) ** 2
                lengths.append(sqrt(s))
        var mean: Float64 = 0
        for x in lengths:
            mean += x
        mean /= Float64(len(lengths))
        if mean == 0:
            out.unsafe_set(t, 0)
            continue
        var var_: Float64 = 0
        for x in lengths:
            var_ += (x - mean) ** 2
        out.unsafe_set(t, 1 - sqrt(var_ / Float64(len(lengths))) / mean)
    return out^
