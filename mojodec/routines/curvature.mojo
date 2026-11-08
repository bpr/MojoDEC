# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Curvature of triangulated surfaces (`mojodec.routines.curvature`).
==================================================================
Discrete Gaussian and mean curvature at the vertices of a 2-complex
embedded in any dimension (usually a surface in 3D); not in PyDEC.

- The *angle defect* at an interior vertex is `2 pi` minus the angles of
  its triangles there; at a boundary vertex it is `pi` minus them. By the
  discrete Gauss-Bonnet theorem the defects add up to `2 pi` times the
  Euler characteristic.
- The *Gaussian curvature* is the angle defect per unit dual area,
  `defect / |*v|`.
- The *mean curvature normal* is `H n = -1/2 Laplace(x)`, the
  Laplace-Beltrami operator applied to the vertex positions; in DEC that
  is `1/2 delta d x`, the cotangent formula (M. Meyer, M. Desbrun,
  P. Schroeder, A. H. Barr, *Discrete differential-geometry operators for
  triangulated 2-manifolds*, 2003).

Circumcentric dual areas can be zero or negative on meshes that are not
well-centered, which makes the per-area quantities unreliable there; see
`mojodec.routines.quality.well_centeredness`.
"""

from std.math import atan2, pi, sqrt
from std.utils.numerics import nan

from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import (
    RealArray,
    cols_of,
    fget,
    fset,
    iget,
    real_array,
    real_vector,
    rows_of,
)


def angle_defect(sc: SimplicialComplex) raises -> RealArray:
    """The angle defect at each vertex of a 2-complex.

    `2 pi` minus the vertex's triangle angles at interior vertices, `pi`
    minus them at boundary vertices, and 0 at vertices no simplex uses, so
    the defects add up to `2 pi` times the Euler characteristic.

    Args:
        sc: A 2-complex with geometry.

    Returns:
        The defect at each vertex, in radians.

    Raises:
        If `sc` is not a 2-complex or was built without geometry.
    """
    _check_surface(sc, "angle_defect")
    ref V = sc.vertices()
    ref T = sc.simplices()
    var nv = rows_of(V)
    var E = cols_of(V)
    var defect = real_vector(nv)
    for v in range(nv):
        defect.unsafe_set(v, 2 * pi)
    for ref e in sc.boundary():
        for v in e.vertices:
            defect.unsafe_set(v, pi)
    for t in range(rows_of(T)):
        for c in range(3):
            var a = iget(T, t, c)
            var b = iget(T, t, (c + 1) % 3)
            var o = iget(T, t, (c + 2) % 3)
            # angle at a between u = b - a and w = o - a, as
            # 2 atan2(| |w| u - |u| w |, | |w| u + |u| w |) (Kahan), which
            # keeps small angles that sqrt(|u|^2 |w|^2 - (u.w)^2) loses
            var uu: Float64 = 0
            var ww: Float64 = 0
            for k in range(E):
                var u = fget(V, b, k) - fget(V, a, k)
                var w = fget(V, o, k) - fget(V, a, k)
                uu += u * u
                ww += w * w
            var nu = sqrt(uu)
            var nw = sqrt(ww)
            var diff: Float64 = 0
            var sum: Float64 = 0
            for k in range(E):
                var u = fget(V, b, k) - fget(V, a, k)
                var w = fget(V, o, k) - fget(V, a, k)
                diff += (nw * u - nu * w) ** 2
                sum += (nw * u + nu * w) ** 2
            var angle = 2 * atan2(sqrt(diff), sqrt(sum))
            defect.unsafe_set(a, defect.unsafe_get(a) - angle)
    # vertices no triangle uses
    var used = _used_vertices(sc)
    for v in range(nv):
        if not used[v]:
            defect.unsafe_set(v, 0.0)
    return defect^


def gaussian_curvature(sc: SimplicialComplex) raises -> RealArray:
    """The Gaussian curvature at each vertex: angle defect over dual area.

    A vertex with zero dual area gets an infinite or NaN value, and a
    vertex no simplex uses gets NaN.

    Args:
        sc: A 2-complex with geometry.

    Returns:
        The curvature at each vertex.

    Raises:
        If `sc` is not a 2-complex or was built without geometry.
    """
    var defect = angle_defect(sc)
    var K = real_vector(defect.size)
    for v in range(K.size):
        K.unsafe_set(v, nan[DType.float64]())
    # sc[0] holds only the vertices some simplex uses
    ref level = sc[0]
    for j in range(level.num_simplices):
        var v = iget(level.simplices, j, 0)
        K.unsafe_set(v, defect.unsafe_get(v) / level.dual_volume.unsafe_get(j))
    return K^


def mean_curvature_normal(sc: SimplicialComplex) raises -> RealArray:
    """The mean curvature normal `H n` at each vertex, `num_vertices x E`.

    It does not depend on how the triangles are oriented. On a sphere of
    radius `r` it is `(x - center) / r^2`: outward, with `|H n| = 1/r`. At
    boundary vertices the cotangent Laplacian also measures the boundary,
    so the value there is not a curvature. Vertices no simplex uses get
    NaN.

    Args:
        sc: A 2-complex with geometry.

    Returns:
        `num_vertices x E` mean curvature normals, `E` the embedding
        dimension.

    Raises:
        If `sc` is not a 2-complex or was built without geometry.
    """
    _check_surface(sc, "mean_curvature_normal")
    var L = sc[0].star_inv @ sc[0].d.T() @ sc[1].star @ sc[0].d
    ref V = sc.vertices()
    var nv = rows_of(V)
    var E = cols_of(V)
    var out = real_array(nv, E)
    for v in range(nv):
        for k in range(E):
            fset(out, v, k, nan[DType.float64]())
    # L acts on the rows of sc[0], which hold only the vertices some
    # simplex uses
    ref level = sc[0]
    var x = real_vector(level.num_simplices)
    for k in range(E):
        for j in range(level.num_simplices):
            x.unsafe_set(j, fget(V, iget(level.simplices, j, 0), k))
        var Lx = L @ x
        for j in range(level.num_simplices):
            fset(out, iget(level.simplices, j, 0), k, 0.5 * Lx.unsafe_get(j))
    return out^


def mean_curvature(sc: SimplicialComplex) raises -> RealArray:
    """The unsigned mean curvature `|H n|` at each vertex.

    Args:
        sc: A 2-complex with geometry.

    Returns:
        The length of `mean_curvature_normal` at each vertex.

    Raises:
        If `sc` is not a 2-complex or was built without geometry.
    """
    var Hn = mean_curvature_normal(sc)
    var E = cols_of(Hn)
    var out = real_vector(rows_of(Hn))
    for v in range(rows_of(Hn)):
        var s: Float64 = 0
        for k in range(E):
            s += fget(Hn, v, k) ** 2
        out.unsafe_set(v, sqrt(s))
    return out^


def _used_vertices(sc: SimplicialComplex) raises -> List[Bool]:
    var used = List[Bool](length=rows_of(sc.vertices()), fill=False)
    ref level = sc[0]
    for j in range(level.num_simplices):
        used[iget(level.simplices, j, 0)] = True
    return used^


def _check_surface(sc: SimplicialComplex, caller: String) raises:
    if sc.complex_dimension() != 2:
        raise Error(caller + ": needs a 2-complex (a triangulated surface)")
    if not sc.has_geometry():
        raise Error(caller + ": the complex was built without geometry")
