# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
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
    minus them at boundary vertices.

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
            # angle at a between ab and ao
            var uu: Float64 = 0
            var vv: Float64 = 0
            var uv: Float64 = 0
            for k in range(E):
                var u = fget(V, b, k) - fget(V, a, k)
                var w = fget(V, o, k) - fget(V, a, k)
                uu += u * u
                vv += w * w
                uv += u * w
            var cross = sqrt(max(uu * vv - uv * uv, 0.0))
            defect.unsafe_set(a, defect.unsafe_get(a) - atan2(cross, uv))
    return defect^


def gaussian_curvature(sc: SimplicialComplex) raises -> RealArray:
    """The Gaussian curvature at each vertex: angle defect over dual area.

    A vertex with zero dual area gets an infinite or NaN value.

    Args:
        sc: A 2-complex with geometry.

    Returns:
        The curvature at each vertex.

    Raises:
        If `sc` is not a 2-complex or was built without geometry.
    """
    var K = angle_defect(sc)
    ref area = sc[0].dual_volume
    for v in range(K.size):
        K.unsafe_set(v, K.unsafe_get(v) / area.unsafe_get(v))
    return K^


def mean_curvature_normal(sc: SimplicialComplex) raises -> RealArray:
    """The mean curvature normal `H n` at each vertex, `num_vertices x E`.

    It does not depend on how the triangles are oriented. On a sphere of
    radius `r` it is `(x - center) / r^2`: outward, with `|H n| = 1/r`. At
    boundary vertices the cotangent Laplacian also measures the boundary,
    so the value there is not a curvature.

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
    var x = real_vector(nv)
    for k in range(E):
        for v in range(nv):
            x.unsafe_set(v, fget(V, v, k))
        var Lx = L @ x
        for v in range(nv):
            fset(out, v, k, 0.5 * Lx.unsafe_get(v))
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


def _check_surface(sc: SimplicialComplex, caller: String) raises:
    if sc.complex_dimension() != 2:
        raise Error(caller + ": needs a 2-complex (a triangulated surface)")
    if not sc.has_geometry():
        raise Error(caller + ": the complex was built without geometry")
