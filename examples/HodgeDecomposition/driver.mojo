# Compute a harmonic 1-cochain basis for a square with 4 holes.
#
# The basis comes from integer cocycles (cohomology_basis): one per hole,
# made harmonic by removing its exact part (harmonic_basis), so harmonic
# form i circulates once around hole i and not around the others. PyDEC's
# driver decomposes four random cochains instead; this one decomposes one
# and checks that its harmonic part lies in the span of the basis.
#
# Run from the repository root:  pixi run examples HodgeDecomposition [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).
# Writes the four harmonic 1-cochains to harmonic.txt (one column each) and
# plots each one to harmonic_<n>.png. harmonic.vtu holds all four, sampled
# at the triangle barycenters, for ParaView.

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
    cohomology_basis,
    harmonic_basis,
    lsqr,
    read_mesh,
)
from mojodec.core.utils import (
    RealArray,
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

    # one integer cocycle per hole, made harmonic in the star inner product
    var basis = cohomology_basis(sc, 1)
    var H = harmonic_basis(sc, 1)
    print(
        "b_1 =",
        len(basis),
        "(Betti numbers",
        sc.betti_numbers(),
        "); periods of each harmonic 1-cochain around the holes:",
    )
    for r in range(len(H)):
        var p = basis.periods(H[r])
        print(
            "harmonic 1-cochain",
            r,
            ": periods",
            p,
            " |d h| =",
            norm2(d(H[r]).v),
            " |delta h| =",
            norm2(delta(H[r]).v),
        )

    # PyDEC's way: the harmonic part of a random cochain. Being closed, it
    # is determined by its periods, so it should equal sum_j p_j H[j].
    var omega = sc.get_cochain(1)
    omega.v = nm.random.rand[f64](Shape(omega.v.size))
    var h = hodge_decomposition(omega)[2].copy()
    var p = basis.periods(h)
    var diff = h.v.copy()
    for j in range(len(H)):
        for t in range(diff.size):
            diff.unsafe_set(t, diff.unsafe_get(t) - p[j] * H[j].v.unsafe_get(t))
    print(
        "harmonic part of a random 1-cochain: periods",
        p,
        "; |h - sum p_j H_j| / |h| =",
        norm2(diff) / norm2(h.v),
    )

    var stacked = real_array(len(H), H[0].v.size)
    for r in range(len(H)):
        for c in range(H[0].v.size):
            fset(stacked, r, c, H[r].v.unsafe_get(c))

    with open("harmonic.txt", "w") as f:
        for c in range(cols_of(stacked)):
            for r in range(len(H)):
                f.write(fget(stacked, r, c), " " if r < len(H) - 1 else "\n")
    print("wrote harmonic.txt")

    # plot the results
    var show = show_requested()
    var plt = pyplot(interactive=show)
    var fields = List[VTKField]()
    for n in range(len(H)):
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
