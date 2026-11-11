# Compute a harmonic 1-cochain basis for a square with 4 holes.
#
# Run from the repository root:  pixi run examples HodgeDecomposition [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).
# Writes the four harmonic 1-cochains to harmonic.txt (one column each) and
# plots each one to harmonic_<n>.png. harmonic.vtu holds all four, sampled
# at the triangle barycenters, for ParaView.

from std.math import sqrt
import numojo as nm
from numojo.prelude import *

from mojodec import (
    SimplicialComplex,
    well_centered_fraction,
    VTKField,
    write_vtu,
    Cochain,
    d,
    delta,
    cg,
    lsqr,
    read_mesh,
)
from mojodec.core.utils import (
    RealArray,
    dot,
    norm2,
    real_array,
    rows_of,
    cols_of,
    take_int_rows,
    fget,
    fset,
)
from mojodec.vis import pyplot, show_requested, simplex_quivers, triplot


def hodge_decomposition(
    omega: Cochain,
) raises -> Tuple[Cochain, Cochain, Cochain]:
    """Split a p-cochain as omega = d(alpha) + delta(beta) + h, h harmonic.

    The mesh is not well-centered (some dual volumes are negative), so
    d(delta(.)) on 2-cochains is nonsymmetric and indefinite. PyDEC's
    driver uses CG there, which does not converge (SciPy reports
    info > 0 and a residual larger than the right-hand side); LSQR solves
    it. The alpha system is solved with CG as in PyDEC, but in the form CG
    needs: delta(d(.)) = star^-1 K is not a symmetric matrix (only
    self-adjoint in the star inner product), so it solves the symmetric
    K alpha = star (delta omega), preconditioned by star^-1, which is CG
    in the star inner product. PyDEC passes star^-1 K to CG directly.
    """
    var sc = omega.complex
    var p = omega.k
    var alpha = sc.get_cochain(p - 1)
    var beta = sc.get_cochain(p + 1)

    # solve for alpha
    ref S = sc[p - 1].star
    var A = S @ delta(d(sc.get_cochain_basis(p - 1))).v  # K = d^T star d
    var b = S @ delta(omega).v
    alpha.v = cg(A, b, M=sc[p - 1].star_inv, rtol=1e-8).take_x()

    # solve for beta
    A = d(delta(sc.get_cochain_basis(p + 1))).v.copy()
    b = d(omega).v.copy()
    beta.v = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=20 * A.ncols).take_x()

    # solve for h
    var h = omega - d(alpha) - delta(beta)
    return (alpha^, beta^, h^)


def ortho(var A: RealArray) raises -> RealArray:
    """Separate the harmonic forms stored in the rows of A with Householder reflections.
    """
    var m = rows_of(A)
    var n = cols_of(A)
    for i in range(m):
        var j = 0
        for c in range(n):
            if abs(fget(A, i, c)) > abs(fget(A, i, j)):
                j = c
        var v = List[Float64](capacity=m)
        var vn: Float64 = 0
        for r in range(m):
            v.append(fget(A, r, j))
            vn += v[r] * v[r]
        vn = sqrt(vn)
        v[i] += vn if fget(A, i, j) > 0 else -vn
        var vv: Float64 = 0
        for r in range(m):
            vv += v[r] * v[r]
        # A = (I - 2 v v^T / v^T v) A
        for c in range(n):
            var s: Float64 = 0
            for r in range(m):
                s += v[r] * fget(A, r, c)
            for r in range(m):
                fset(A, r, c, fget(A, r, c) - 2 * v[r] * s / vv)
    return A^


def main() raises:
    nm.random.seed(1)  # make results consistent

    # read in mesh data from file
    var mesh = read_mesh("mesh_example.xml")

    # remove some triangles from the mesh
    var removed: List[Int] = [30, 320, 21, 198]
    var keep = List[Int]()
    for i in range(rows_of(mesh.elements)):
        if i not in removed:
            keep.append(i)
    var sc = SimplicialComplex(
        mesh.vertices, take_int_rows(mesh.elements, keep)
    )
    print(sc)
    # circumcentric DEC prefers well-centered meshes; this one is not, and
    # an obtuse pair of triangles can give an edge a negative dual length
    var negative = 0
    for e in range(sc[1].num_simplices):
        if sc[1].dual_volume.unsafe_get(e) < 0:
            negative += 1
    print(
        "well-centered triangles:",
        well_centered_fraction(sc) * 100,
        "%; edges with negative dual length:",
        negative,
    )

    var H = List[RealArray]()  # harmonic forms
    # decompose 4 random 1-cochains
    for _ in range(4):
        var omega = sc.get_cochain(1)
        omega.v = nm.random.rand[f64](Shape(omega.v.size))
        var abh = hodge_decomposition(omega)
        var h = abh[2].v.copy()
        for ref v in H:
            var c = dot(v, h)
            for t in range(h.size):
                h.unsafe_set(t, h.unsafe_get(t) - c * v.unsafe_get(t))
        var hn = norm2(h)
        for t in range(h.size):
            h.unsafe_set(t, h.unsafe_get(t) / hn)
        H.append(h^)

    var stacked = real_array(4, H[0].size)
    for r in range(4):
        for c in range(H[0].size):
            fset(stacked, r, c, H[r].unsafe_get(c))
    stacked = ortho(stacked^)

    # each row is harmonic: d h = 0 and delta h = 0
    for r in range(4):
        var hc = sc.get_cochain(1)
        for c in range(hc.v.size):
            hc[c] = fget(stacked, r, c)
        print(
            "harmonic 1-cochain",
            r,
            ": |d h| =",
            norm2(d(hc).v),
            " |delta h| =",
            norm2(delta(hc).v),
        )

    with open("harmonic.txt", "w") as f:
        for c in range(cols_of(stacked)):
            for r in range(4):
                f.write(fget(stacked, r, c), " " if r < 3 else "\n")
    print("wrote harmonic.txt")

    # plot the results
    var show = show_requested()
    var plt = pyplot(interactive=show)
    var fields = List[VTKField]()
    for n in range(4):
        var h = sc.get_cochain(1)
        for c in range(h.v.size):
            h[c] = fget(stacked, n, c)
        plt.figure()
        plt.title("Harmonic 1-cochain #" + String(n))
        _ = triplot(sc.vertices(), sc.simplices())
        var q = simplex_quivers(sc, h.v)
        var bases = q.bases.to_numpy()
        var dirs = q.vectors.to_numpy()
        plt.quiver(bases.T[0], bases.T[1], dirs.T[0], dirs.T[1])
        plt.axis("equal")
        plt.savefig("harmonic_" + String(n) + ".png")
        fields.append(VTKField("harmonic_" + String(n), q.vectors))
    print("wrote harmonic_0.png ... harmonic_3.png")
    write_vtu("harmonic.vtu", sc.vertices(), sc.simplices(), cell_data=fields)
    print("wrote harmonic.vtu")
    if show:
        plt.show()  # one window per figure; returns when all are closed
