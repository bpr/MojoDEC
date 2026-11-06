# Solve the resonant cavity problem with Whitney forms.
#
# Reference:
#   D. N. Arnold, R. S. Falk, R. Winther. "Finite element exterior calculus:
#   from Hodge theory to numerical stability." Bull. Amer. Math. Soc. (N.S.)
#   47(2), 281-354. DOI 10.1090/S0273-0979-10-01278-4
#
# Run from the repository root:  pixi run examples ResonantCavity [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).
# Writes the nonzero eigenvalues to eigenvalues.txt and the third mode
# (one value per edge) to mode3.txt, plots them to eigenvalues.png and
# mode3.png, and writes the mode, sampled at the triangle barycenters, to
# mode3.vtu for ParaView.

import numojo as nm
from numojo.prelude import *

from mojodec import (
    SimplicialComplex,
    loadtxt,
    whitney_innerproduct,
    eigh_generalized,
    VTKField,
    write_vtu,
)
from mojodec.core.utils import fget, real_vector, real_vector_from_list
from mojodec.vis import pyplot, show_requested, simplex_quivers


def main() raises:
    # read the mesh and construct the complex
    var vertices = loadtxt("vertices.txt")
    var triangles = loadtxt[DType.int]("triangles.txt")
    var sc = SimplicialComplex(vertices, triangles)

    # stiffness and mass matrices
    var K = sc[1].d.T() @ whitney_innerproduct(sc, 2) @ sc[1].d
    var M = whitney_innerproduct(sc, 1)

    # eliminate the boundary edges
    var is_boundary = List[Bool](length=sc[1].num_simplices, fill=False)
    for ref e in sc.boundary():
        is_boundary[sc[1].simplex_to_index(e)] = True
    var interior = List[Int]()
    for i in range(sc[1].num_simplices):
        if not is_boundary[i]:
            interior.append(i)
    K = K.select(interior, interior)
    M = M.select(interior, interior)

    # eigenvalues and eigenvectors of K x = lambda M x
    var wv = eigh_generalized(K.todense(), M.todense())
    ref eigenvalues = wv[0]
    ref eigenvectors = wv[1]

    var NUM_EIGS = 50  # number of eigenvalues to report
    var values = List[Float64]()
    var columns = List[Int]()
    for i in range(eigenvalues.size):
        if eigenvalues.unsafe_get(i) > 1e-10 and len(values) < NUM_EIGS:
            values.append(eigenvalues.unsafe_get(i))
            columns.append(i)
    print("interior edges:", len(interior))
    print("first", len(values), "nonzero eigenvalues:")
    for i in range(len(values)):
        print("  ", i, values[i])
    with open("eigenvalues.txt", "w") as f:
        for v in values:
            f.write(v, "\n")

    # the third nonzero mode as a 1-cochain on all edges
    var N = 2
    var all_values = real_vector(sc[1].num_simplices)
    for i in range(len(interior)):
        all_values.unsafe_set(interior[i], fget(eigenvectors, i, columns[N]))
    with open("mode3.txt", "w") as f:
        for i in range(all_values.size):
            f.write(all_values.unsafe_get(i), "\n")
    print("wrote eigenvalues.txt and mode3.txt")

    # plot the eigenvalues
    var show = show_requested()
    var plt = pyplot(interactive=show)
    var ax = plt.figure().gca()
    ax.set_title("First " + String(len(values)) + " Eigenvalues\n\n")
    ax.plot(real_vector_from_list(values).to_numpy(), "ko")
    plt.savefig("eigenvalues.png")

    # plot the eigenvector 1-cochain as a vector field
    var bd = simplex_quivers(sc, all_values)
    var bases = bd[0].to_numpy()
    var arrows = bd[1].to_numpy()
    var V = sc.vertices().to_numpy()
    ax = plt.figure().gca()
    ax.set_title("Mode #" + String(N + 1))
    ax.quiver(bases.T[0], bases.T[1], arrows.T[0], arrows.T[1])
    ax.triplot(V.T[0], V.T[1], sc.simplices().to_numpy())
    ax.axis("equal")
    plt.savefig("mode3.png")
    print("wrote eigenvalues.png and mode3.png")
    write_vtu(
        "mode3.vtu",
        sc.vertices(),
        sc.simplices(),
        cell_data=[VTKField("mode3", bd[1])],
    )
    print("wrote mode3.vtu")
    if show:
        plt.show()  # one window per figure; returns when all are closed
