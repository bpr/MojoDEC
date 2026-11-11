# Maxwell's equations in a 2D cavity, stepped in time with the explicit
# leapfrog (Yee's scheme), using the typed layer. The electric field e is a
# primal 1-cochain (on the edges), the magnetic flux b a primal 2-cochain
# (through the triangles):
#
#     star1 e' =  d1^T star2 b,     star2 b' = -star2 d1 e,
#
# with the tangential electric field zero on the walls. The cavity is an
# equilateral triangle of side 1 cut into equilateral triangles, so every
# Hodge star is positive and each step is explicit. Its lowest resonance
# is omega^2 = (4 pi / 3)^2, the first Neumann eigenvalue of the triangle.
#
# The run starts from the lowest discrete mode with b = 0, so
# e(t) = cos(omega_dt t) e(0) with omega_dt = (2/dt) asin(omega dt / 2),
# and the leapfrog's staggered energy stays constant to rounding.
#
# Not in PyDEC. Reference for the scheme:
#   E. Hairer, C. Lubich, G. Wanner. "Geometric numerical integration
#   illustrated by the Stormer-Verlet method." Acta Numerica 12 (2003),
#   399-450. DOI 10.1017/S0962492902000144
#
# Run from the repository root:  pixi run examples CavityWaves [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).
# Writes the amplitude and energies over time to waves.txt, plots them to
# amplitude.png and energy.png, and writes the initial electric field and
# the magnetic field a quarter period later to waves.vtu for ParaView.

from std.math import asin, ceil, cos, pi, sqrt

import numojo as nm
from numojo.prelude import *

from mojodec import (
    PRIMAL,
    SimplicialMesh,
    TypedCochain,
    TypedComplex,
    TypedLeapfrog,
    VTKField,
    eigh_generalized,
    loop_subdivision,
    write_vtu,
)
from mojodec.core.utils import dot, real_vector, real_vector_from_list
from mojodec.vis import pyplot, show_requested, simplex_quivers


def main() raises:
    # an equilateral triangle of side 1, subdivided into 256 triangles
    var mesh = SimplicialMesh(
        nm.fromstring[f64]("[[0, 0], [1, 0], [0.5, 0.8660254037844386]]"),
        nm.fromstring[DType.int]("[[0, 1, 2]]"),
    )
    for _ in range(4):
        mesh = loop_subdivision(mesh)
    var tc = TypedComplex[2](mesh)
    ref sc = tc.sc
    print("triangles:", sc[2].num_simplices, " edges:", sc[1].num_simplices)

    # P drops the boundary edges' rows: there the tangential field is zero
    var boundary = tc.boundary_cells[1]()
    var P = tc.identity[1, PRIMAL]().exclude_subset(boundary)
    var d1 = tc.d[1, PRIMAL]()
    var star1 = tc.star[1, PRIMAL]()
    var star2 = tc.star[2, PRIMAL]()
    var K = P.T() @ d1.T() @ star2.to_matrix_op()  # primal 2 -> dual 1

    # the resonances: K d1 P e = omega^2 star1 e
    var eig = eigh_generalized(K @ d1 @ P, star1)
    var first = 0
    while eig.values.unsafe_get(first) < 1e-8:
        first += 1
    var unit = (4 * pi / 3) ** 2
    print("lowest resonances omega^2 / (4 pi / 3)^2:")
    for i in range(first, first + 8):
        print("  ", eig.values.unsafe_get(i) / unit)
    var lam = eig.values.unsafe_get(first)

    # the scheme, at half the stability limit
    var limit = 2 / sqrt(eig.values.unsafe_get(len(eig.values) - 1))
    var lf = TypedLeapfrog(star1, star2, K, 0.5 * limit)
    var dt = lf.dt()
    print("dt:", dt, " stability limit:", lf.max_stable_dt(), "(", limit, ")")

    # start from the lowest mode, b = 0
    var e0: TypedCochain[2, 1, PRIMAL] = eig.mode(first)
    for i in boundary.indices():  # rounding leaves ~1e-59 there
        e0[i] = 0.0
    var e = e0.copy()
    var b = tc.cochain[2, PRIMAL]()
    var omega_dt = 2 / dt * asin(sqrt(lam) * dt / 2)
    var period = 2 * pi / omega_dt
    var steps = Int(ceil(5 * period / dt))
    var quarter = Int(period / 4 / dt + 0.5)

    var norm = dot(e0.v, e0.v)
    var E0 = lf.energy(e, b)
    var times = List[Float64]()
    var amplitude = List[Float64]()
    var exact = List[Float64]()
    var drift = List[Float64]()  # staggered energy, relative change
    var colocated = List[Float64]()  # 1/2 e star1 e + 1/2 b star2 b
    var b_quarter = b.copy()
    var worst_amplitude: Float64 = 0
    var worst_energy: Float64 = 0
    for n in range(steps + 1):
        if n > 0:
            lf.step(e, b)
        var t = Float64(n) * dt
        var c = dot(e.v, e0.v) / norm
        var w = cos(omega_dt * t)
        var s = (lf.energy(e, b) - E0) / E0
        var col = (
            0.5 * star1.diag.inner_product(e.v, e.v)
            + 0.5 * star2.diag.inner_product(b.v, b.v)
        ) / E0 - 1
        times.append(t)
        amplitude.append(c)
        exact.append(w)
        drift.append(s)
        colocated.append(col)
        worst_amplitude = max(worst_amplitude, abs(c - w))
        worst_energy = max(worst_energy, abs(s))
        if n == quarter:
            b_quarter = b.copy()
    print(
        "steps:",
        steps,
        " period:",
        period,
        " omega:",
        sqrt(lam),
        " omega_dt:",
        omega_dt,
        " exact omega:",
        4 * pi / 3,
    )
    print("max |amplitude - cos(omega_dt t)|:", worst_amplitude)
    print("max relative change of the staggered energy:", worst_energy)
    with open("waves.txt", "w") as f:
        f.write(
            "# t amplitude cos(omega_dt t) staggered_energy colocated_energy\n"
        )
        for i in range(len(times)):
            f.write(
                times[i],
                " ",
                amplitude[i],
                " ",
                exact[i],
                " ",
                drift[i],
                " ",
                colocated[i],
                "\n",
            )
    print("wrote waves.txt")

    # plots
    var show = show_requested()
    var plt = pyplot(interactive=show)
    var T = real_vector_from_list(times).to_numpy()
    var ax = plt.figure().gca()
    ax.set_title("Amplitude of the lowest mode")
    ax.plot(
        T,
        real_vector_from_list(exact).to_numpy(),
        "k-",
        label="cos(omega_dt t)",
    )
    ax.plot(
        T,
        real_vector_from_list(amplitude).to_numpy(),
        "r.",
        markersize=3,
        label="leapfrog",
    )
    ax.set_xlabel("t")
    ax.legend()
    plt.savefig("amplitude.png")
    ax = plt.figure().gca()
    ax.set_title("Relative change of the energy")
    ax.plot(
        T,
        real_vector_from_list(colocated).to_numpy(),
        label="1/2 e*e + 1/2 b*b",
    )
    ax.plot(
        T,
        real_vector_from_list(drift).to_numpy(),
        label="staggered (conserved)",
    )
    ax.set_xlabel("t")
    ax.legend()
    plt.savefig("energy.png")
    print("wrote amplitude.png and energy.png")

    # the initial electric field and B_z (flux over area) a quarter period on
    var q = simplex_quivers(sc, e0.v)
    var bz = real_vector(sc[2].num_simplices)
    for t in range(sc[2].num_simplices):
        bz.unsafe_set(t, b_quarter[t] / abs(sc[2].primal_volume.unsafe_get(t)))
    write_vtu(
        "waves.vtu",
        sc.vertices(),
        sc.simplices(),
        cell_data=[VTKField("E0", q.vectors), VTKField("Bz_quarter", bz)],
    )
    print("wrote waves.vtu")
    if show:
        plt.show()  # one window per figure; returns when all are closed
