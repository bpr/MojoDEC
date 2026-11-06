# Darcy flow in a triangle mesh of a planar domain, with constant velocity
# prescribed on the boundary.
#
# Reference:
#   Numerical method for Darcy flow derived using Discrete Exterior Calculus.
#   A. N. Hirani, K. B. Nakshatrala, J. H. Chaudhry. arXiv:0810.3434 [math.NA]
#
# Run from the repository root:  pixi run examples DarcyFlow [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).
# Writes pressures.txt: triangle circumcenter x, DEC pressure, exact pressure.
# Plots the pressures to pressures.png and the flux, as a velocity field, to
# flux.png, and writes both to darcy.vtu for ParaView.

from std.python import Python

import numojo as nm
from numojo.prelude import *

from mojodec import (
    SimplicialComplex,
    CSRMatrix,
    VTKField,
    loadtxt,
    spsolve,
    write_vtu,
)
from mojodec.core.utils import (
    fget,
    fset,
    iget,
    real_array,
    real_vector,
    rows_of,
    take_rows,
    row_list,
)
from mojodec.routines.dense import det
from mojodec.vis import pyplot, show_requested, simplex_quivers


def main() raises:
    var velocity: List[Float64] = [1.0, 0.0]
    # read the mesh (1-based indices in the file)
    var vertices = loadtxt("vertices.txt")
    var triangles = loadtxt[DType.int]("triangles.txt")
    for i in range(triangles.size):
        triangles.unsafe_set(i, triangles.unsafe_get(i) - 1)
    var sc = SimplicialComplex(vertices, triangles)
    # Nk is the number of k-simplices
    var N1 = sc[1].num_simplices
    var N2 = sc[2].num_simplices
    # permeability k > 0 and viscosity mu > 0
    var k: Float64 = 1
    var mu: Float64 = 1
    # The full linear system for Darcy flow in 2D, before boundary conditions:
    #   [-(mu/k) star1   d1^T ]
    #   [      d1         0   ]
    var blocks = List[List[Optional[CSRMatrix]]]()
    blocks.append(
        [Optional((sc[1].star * (-mu / k)).to_csr()), Optional(sc[1].d.T())]
    )
    blocks.append([Optional(sc[1].d.copy()), Optional[CSRMatrix](None)])
    var A = CSRMatrix.bmat(blocks)
    var all_fluxes = real_vector(N1)

    # boundary and internal edges
    var boundary_edges = sc.boundary()  # sorted, as returned
    var boundary_indices = List[Int]()
    var is_boundary = List[Bool](length=N1, fill=False)
    for ref e in boundary_edges:
        var i = sc[1].simplex_to_index(e)
        boundary_indices.append(i)
        is_boundary[i] = True
    var internal_indices = List[Int]()
    for i in range(N1):
        if not is_boundary[i]:
            internal_indices.append(i)

    # all triangles have the same orientation, so look at the first one
    var t0 = take_rows(vertices, row_list(triangles, 0))
    var edge_mat = nm.zeros[f64](Shape(2, 2))
    for r in range(2):
        for c in range(2):
            edge_mat.unsafe_set(r * 2 + c, fget(t0, r + 1, c) - fget(t0, 0, c))
    var s: Float64 = 1.0 if det(edge_mat) > 0 else -1.0
    for i in range(len(boundary_edges)):
        ref e = boundary_edges[i]
        var sign: Float64 = -1.0 if e.parity == 1 else 1.0
        var ev0 = sign * (fget(vertices, e[1], 0) - fget(vertices, e[0], 0))
        var ev1 = sign * (fget(vertices, e[1], 1) - fget(vertices, e[0], 1))
        var n0 = -ev1
        var n1 = ev0
        var len2 = ev0 * ev0 + ev1 * ev1
        var vdotn = velocity[0] * n0 + velocity[1] * n1
        var area = abs(n0 * ev1 - n1 * ev0)
        all_fluxes.unsafe_set(
            boundary_indices[i], -s * (1 / len2) * vdotn * area
        )

    var pressures = real_vector(N2)
    # adjust the right-hand side for the known fluxes and pressures
    var known = real_vector(N1 + N2)
    for i in range(N1):
        known.unsafe_set(i, all_fluxes.unsafe_get(i))
    var b = (A @ known) * -1.0
    # the pressure at the right-most triangle circumcenter is taken as known (0)
    var rightmost = 0
    for i in range(N2):
        if fget(sc[2].circumcenter, i, 0) > fget(
            sc[2].circumcenter, rightmost, 0
        ):
            rightmost = i
    var keep = internal_indices.copy()
    for i in range(N2):
        if i != rightmost:
            keep.append(N1 + i)
    var bk = real_vector(len(keep))
    for i in range(len(keep)):
        bk.unsafe_set(i, b.unsafe_get(keep[i]))
    var Ak = A.select(keep, keep)
    var u = spsolve(Ak, bk)
    var ni = len(internal_indices)
    for i in range(ni):
        all_fluxes.unsafe_set(internal_indices[i], u.unsafe_get(i))
    for i in range(ni, len(keep)):
        pressures.unsafe_set(keep[i] - N1, u.unsafe_get(i))

    # compare with the exact pressure: slope -1 through the known point
    var xmax = fget(sc[2].circumcenter, rightmost, 0)
    var max_err: Float64 = 0
    with open("pressures.txt", "w") as f:
        for i in range(N2):
            var x = fget(sc[2].circumcenter, i, 0)
            var exact = xmax - x
            max_err = max(max_err, abs(pressures.unsafe_get(i) - exact))
            f.write(x, " ", pressures.unsafe_get(i), " ", exact, "\n")
    print("edges:", N1, " triangles:", N2, " unknowns:", len(keep))
    print("max |pressure - exact| =", max_err)
    print("wrote pressures.txt")

    # plot the pressures, with the exact line of slope -1 through the
    # known pressure point
    var show = show_requested()
    var plt = pyplot(interactive=show)
    var cc = sc[2].circumcenter.to_numpy()
    plt.figure()
    var ax = plt.gca()
    ax.set_xlabel("x", fontsize=20)
    ax.set_ylabel("Pressure", fontsize=20)
    ax.plot(cc.T[0], pressures.to_numpy(), "ro", markersize=8, mec="r")
    var xmin = cc.T[0].min()
    ax.plot(
        Python.list(xmin, xmax), Python.list(xmax - xmin, 0), "k", linewidth=2
    )
    ax.legend(Python.list("DEC", "Analytical"), numpoints=1)
    plt.savefig("pressures.png")

    # plot the triangles
    plt.figure()
    ax = plt.gca()
    var V = vertices.to_numpy()
    ax.triplot(V.T[0], V.T[1], triangles.to_numpy())
    # Whitney interpolate the flux and sample it at the barycenters, then
    # rotate 90 degrees opposite to the triangles' orientation
    var q = simplex_quivers(sc, all_fluxes)
    var bases = q.bases.to_numpy()
    var arrows = q.vectors.to_numpy()
    ax.quiver(
        bases.T[0],
        bases.T[1],
        arrows.T[1] * s,
        arrows.T[0] * -s,
        units="dots",
        width=1,
        scale=1.0 / 30,
    )
    ax.axis("equal")
    ax.set_title(
        "Flux interpolated using Whitney map \n"
        " and visualized as velocity at barycenters\n"
    )
    plt.savefig("flux.png")
    print("wrote pressures.png and flux.png")

    var barycenter_velocity = real_array(N2, 2)
    for i in range(N2):
        fset(barycenter_velocity, i, 0, s * fget(q.vectors, i, 1))
        fset(barycenter_velocity, i, 1, -s * fget(q.vectors, i, 0))
    write_vtu(
        "darcy.vtu",
        vertices,
        triangles,
        cell_data=[
            VTKField("pressure", pressures),
            VTKField("velocity", barycenter_velocity),
        ],
    )
    print("wrote darcy.vtu")
    if show:
        plt.show()  # one window per figure; returns when all are closed
