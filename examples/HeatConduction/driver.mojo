# Steady heat conduction through a plate of two materials, on a mesh made
# by Gmsh.
#
# plate.geo describes a 2 x 1 plate: an insulator (conductivity 1) for
# x < 0.8 and a conductor (conductivity 5) beyond it. Its physical groups
# name the pieces: the "hot" edge x = 0 is held at temperature 1, the
# "cold" edge x = 2 at 0, and the "insulated" top and bottom let no heat
# through. read_gmsh turns the groups into Subsets of the complex: the
# boundary conditions are imposed on the vertices of "hot" and "cold", and
# the triangles of "conductor" get the higher conductivity.
#
# The temperature solves div(k grad u) = 0, discretized as
#   d0^T * K * star1 * d0 u = 0,
# where K * star1 weighs each dual edge by the conductivities of the two
# triangles it crosses: its part in triangle T, cot(angle opposite the
# edge in T) / 2, times k_T. With k = 1 these weights are the Hodge star.
#
# The program uses the typed layer: the temperature is a primal 0-cochain,
# the operator takes it to a dual 2-cochain (the net heat leaving each
# vertex's dual cell), and the compiler checks that the pieces compose.
# The exact temperature is piecewise linear in x, with a kink at the
# interface where the heat flux k du/dx is continuous; since the interface
# is a line of the mesh, DEC reproduces it to rounding error. The heat
# that enters through the hot edge leaves through the cold one.
#
# Run from the repository root:  pixi run examples HeatConduction [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).
# Writes temperature.txt: vertex x, y, DEC temperature, exact temperature.
# Plots the temperature to temperature.png and its profile along x to
# profile.png, and writes both the temperature and the conductivity to
# heat.vtu for ParaView.
#
# To change the plate, edit plate.geo and remesh it:
#   pixi exec --spec gmsh gmsh -2 plate.geo -format msh41 -o plate.msh

from std.python import Python

from mojodec import (
    DUAL,
    PRIMAL,
    Simplex,
    TypedCochain,
    TypedComplex,
    VTKField,
    read_gmsh,
    spsolve,
    write_vtu,
)
from mojodec.core.utils import fget, real_vector
from mojodec.vis import pyplot, show_requested

comptime INTERFACE_X = 0.8
comptime LENGTH = 2.0
comptime K_INSULATOR = 1.0
comptime K_CONDUCTOR = 5.0


def exact(x: Float64) -> Float64:
    """The temperature of the layered plate: linear in each material, with
    the same flux q = k du/dx in both."""
    var q = 1.0 / (
        INTERFACE_X / K_INSULATOR + (LENGTH - INTERFACE_X) / K_CONDUCTOR
    )
    if x <= INTERFACE_X:
        return 1.0 - q * x / K_INSULATOR
    return q * (LENGTH - x) / K_CONDUCTOR


def main() raises:
    # dim=2 makes sure the triangles were saved, not just the tagged curves
    var msh = read_gmsh("plate.msh", dim=2)
    var tc = TypedComplex[2](msh.mesh())
    ref sc = tc.sc
    print(sc)
    print("physical groups:", msh.group_names())
    var N0 = sc[0].num_simplices
    var N1 = sc[1].num_simplices
    var N2 = sc[2].num_simplices
    ref V = sc.vertices()

    # the named pieces of the plate, as sets of vertices and triangles
    var hot = msh.subset[0](tc, "hot")
    var cold = msh.subset[0](tc, "cold")
    var interface = msh.subset[0](tc, "interface")
    var conductor = msh.subset[2](tc, "conductor")
    print(
        "vertices: hot",
        len(hot),
        " cold",
        len(cold),
        " interface",
        len(interface),
        "   conductor triangles:",
        len(conductor),
        "of",
        N2,
    )

    # conductivity-weighted Hodge star on primal edges: each triangle adds
    # k_T cot(theta) / 2 to its three edges, theta the angle opposite one
    var k = real_vector(N2)
    var weights = List[Float64](length=N1, fill=0.0)
    var star1 = List[Float64](length=N1, fill=0.0)
    for t in range(N2):
        k.unsafe_set(t, K_CONDUCTOR if t in conductor else K_INSULATOR)
        var s = sc[2].index_to_simplex(t)
        for j in range(3):
            var a = s[(j + 1) % 3]
            var b = s[(j + 2) % 3]
            var c = s[j]
            var ux = fget(V, a, 0) - fget(V, c, 0)
            var uy = fget(V, a, 1) - fget(V, c, 1)
            var vx = fget(V, b, 0) - fget(V, c, 0)
            var vy = fget(V, b, 1) - fget(V, c, 1)
            var half_cot = 0.5 * (ux * vx + uy * vy) / abs(ux * vy - uy * vx)
            var e = sc[1].simplex_to_index(Simplex([a, b]))
            star1[e] += half_cot
            weights[e] += k.unsafe_get(t) * half_cot
    # with k = 1 the weights are the circumcentric Hodge star
    var star_err: Float64 = 0
    for e in range(N1):
        star_err = max(star_err, abs(star1[e] - sc[1].star[e]))
    print("max |cot weights - star1| =", star_err)

    # the operator d0^T K star1 d0, from primal 0- to dual 2-cochains; the
    # weights take primal edges to dual edges, as a Hodge star does
    var d0 = tc.d[0, PRIMAL]()
    var L = d0.T() @ tc.dual_diagonal[1, PRIMAL](weights^) @ d0
    # the temperature is fixed on the hot and cold vertices: the equations
    # of their dual cells become rows of the identity
    var fixed = (hot | cold).dual()
    var ones = List[Float64](length=N0, fill=1.0)
    var A = L.exclude_subset(fixed) + tc.dual_diagonal[0, PRIMAL](
        ones^
    ).restrict_to_subset(fixed)
    var b = tc.cochain[2, DUAL]()
    for i in hot.indices():
        b[i] = 1.0
    var u: TypedCochain[2, 0, PRIMAL] = spsolve(A, b)

    var max_err: Float64 = 0
    with open("temperature.txt", "w") as f:
        for i in range(N0):
            var x = fget(V, i, 0)
            var ex = exact(x)
            max_err = max(max_err, abs(u[i] - ex))
            f.write(x, " ", fget(V, i, 1), " ", u[i], " ", ex, "\n")
    print("max |temperature - exact| =", max_err)
    var at_interface: Float64 = 0
    for i in interface.indices():
        at_interface += u[i] / Float64(len(interface))
    print(
        "mean temperature on the interface:",
        at_interface,
        " exact:",
        exact(INTERFACE_X),
    )

    # heat balance: L u is a dual 2-cochain, the heat leaving each vertex's
    # dual cell, so it vanishes everywhere but the hot and cold edges
    var r = L @ u
    var q_in: Float64 = 0
    var q_out: Float64 = 0
    var leak: Float64 = 0
    for i in range(N0):
        if i in hot:
            q_in += r[i]
        elif i in cold:
            q_out -= r[i]
        else:
            leak = max(leak, abs(r[i]))
    print("heat in through 'hot':  ", q_in)
    print("heat out through 'cold':", q_out)
    print(
        "exact:                  ",
        1.0
        / (INTERFACE_X / K_INSULATOR + (LENGTH - INTERFACE_X) / K_CONDUCTOR),
    )
    print("max |heat lost| at any other vertex:", leak)
    print("wrote temperature.txt")

    # the temperature over the plate, with the interface marked
    var show = show_requested()
    var plt = pyplot(interactive=show)
    var Vn = V.to_numpy()
    var T = sc.simplices().to_numpy()
    plt.figure(figsize=Python.tuple(9, 4.5))
    var ax = plt.gca()
    var tp = ax.tripcolor(
        Vn.T[0], Vn.T[1], T, u.v.to_numpy(), shading="gouraud", cmap="inferno"
    )
    ax.triplot(Vn.T[0], Vn.T[1], T, color="w", linewidth=0.2, alpha=0.5)
    ax.plot(
        Python.list(INTERFACE_X, INTERFACE_X),
        Python.list(0, 1),
        "c--",
        linewidth=1.5,
    )
    plt.colorbar(tp, ax=ax, label="temperature")
    ax.set_aspect("equal")
    ax.set_title(
        "insulator (k = 1)  |  conductor (k = 5)\nhot edge at x = 0, cold"
        " edge at x = 2, top and bottom insulated"
    )
    plt.savefig("temperature.png", bbox_inches="tight")

    # the temperature at every vertex against x, and the exact profile
    plt.figure()
    ax = plt.gca()
    ax.plot(Vn.T[0], u.v.to_numpy(), "ro", markersize=4, mec="r")
    ax.plot(
        Python.list(0, INTERFACE_X, LENGTH),
        Python.list(1, exact(INTERFACE_X), 0),
        "k",
        linewidth=1.5,
    )
    ax.axvline(INTERFACE_X, color="c", linestyle="--")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("Temperature", fontsize=16)
    ax.legend(Python.list("DEC", "Exact", "interface"), numpoints=1)
    plt.savefig("profile.png")
    print("wrote temperature.png and profile.png")

    write_vtu(
        "heat.vtu",
        V,
        sc.simplices(),
        point_data=[VTKField("temperature", u.v)],
        cell_data=[VTKField("conductivity", k)],
    )
    print("wrote heat.vtu")
    if show:
        plt.show()  # one window per figure; returns when all are closed
