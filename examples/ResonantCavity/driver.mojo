# Solve the resonant cavity problem with Whitney forms, using the typed
# layer: the stiffness and mass matrices are operators from primal to dual
# 1-cochains, and the eigenvectors come back as primal 1-cochains.
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

from mojodec import (
    PRIMAL,
    TypedCochain,
    TypedComplex,
    loadtxt,
    whitney_innerproduct,
    eigh_generalized,
    VTKField,
    write_vtu,
)
from mojodec.core.utils import real_vector_from_list
from mojodec.vis import pyplot, show_requested, simplex_quivers


def main() raises:
    # read the mesh and construct the complex
    var vertices = loadtxt("vertices.txt")
    var triangles = loadtxt[DType.int]("triangles.txt")
    var tc = TypedComplex[2](vertices, triangles)
    ref sc = tc.sc

    # stiffness and mass matrices, both from primal 1-cochains (the
    # electric field on the edges) to dual 1-cochains
    var d1 = tc.d[1, PRIMAL]()
    var K = d1.T() @ whitney_innerproduct[2](tc) @ d1
    var M = whitney_innerproduct[1](tc)

    # the field's tangential part vanishes on the boundary: P zeroes the
    # boundary edges and Q keeps only them. P^T K P and P^T M P are the
    # problem on the interior edges; Q^T M Q, with no stiffness, adds an
    # eigenvalue 0 per boundary edge, which the zero filter below drops
    var boundary = tc.boundary_cells[1]()
    var P = tc.identity[1, PRIMAL]().exclude_subset(boundary)
    var Q = tc.identity[1, PRIMAL]().restrict_to_subset(boundary)
    var K_int = P.T() @ K @ P
    var M_int = P.T() @ M @ P + Q.T() @ M @ Q

    # eigenvalues and eigenvectors of K x = lambda M x
    var eig = eigh_generalized(K_int, M_int)
    # how far each pair is from exact for K_int and M_int, relatively
    var eta = eig.residuals(K_int, M_int)
    var worst_eta: Float64 = 0
    for i in range(eta.size):
        worst_eta = max(worst_eta, eta.unsafe_get(i))
    print("largest backward error of any eigenpair:", worst_eta)
    if worst_eta > 1e-12:
        raise Error("the eigensolver lost accuracy")

    var NUM_EIGS = 50  # number of eigenvalues to report
    var values = List[Float64]()
    var columns = List[Int]()
    for i in range(len(eig)):
        var v = eig.values.unsafe_get(i)
        if v > 1e-10 and len(values) < NUM_EIGS:
            values.append(v)
            columns.append(i)
    print("interior edges:", sc[1].num_simplices - len(boundary))
    print("first", len(values), "nonzero eigenvalues:")
    for i in range(len(values)):
        print("  ", i, values[i])
    with open("eigenvalues.txt", "w") as f:
        for v in values:
            f.write(v, "\n")

    # the third nonzero mode, a primal 1-cochain (zero on the boundary)
    var N = 2
    var mode: TypedCochain[2, 1, PRIMAL] = eig.mode(columns[N])
    with open("mode3.txt", "w") as f:
        for i in range(len(mode)):
            f.write(mode[i], "\n")
    print("wrote eigenvalues.txt and mode3.txt")

    # plot the eigenvalues
    var show = show_requested()
    var plt = pyplot(interactive=show)
    var ax = plt.figure().gca()
    ax.set_title("First " + String(len(values)) + " Eigenvalues\n\n")
    ax.plot(real_vector_from_list(values).to_numpy(), "ko")
    plt.savefig("eigenvalues.png")

    # plot the eigenvector 1-cochain as a vector field
    var q = simplex_quivers(sc, mode.v)
    var bases = q.bases.to_numpy()
    var arrows = q.vectors.to_numpy()
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
        cell_data=[VTKField("mode3", q.vectors)],
    )
    print("wrote mode3.vtu")
    if show:
        plt.show()  # one window per figure; returns when all are closed
