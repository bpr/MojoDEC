# Detection of holes in coverage in an idealized abstraction of a sensor
# network. Hodge decomposition of a random 1-chain of the Rips complex gives
# a harmonic 1-chain, which is concentrated around the holes.
#
# Run from the repository root:  pixi run examples RipsComplex [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).
# Writes the harmonic chain to harmonic.txt (one row per edge: i j |h|) and
# plots the nodes, the complex, and the harmonic chain to nodes.png,
# complex.png, and harmonic.png. harmonic.vtu holds the edges with h, for
# ParaView.

from std.python import PythonObject

import numojo as nm
from numojo.prelude import *

from mojodec import RipsComplex, VTKField, read_array, cg, write_vtu
from mojodec.core.utils import iget, norm2, real_vector, rows_of
from mojodec.vis import lineplot, pyplot, show_requested, triplot


def main() raises:
    var pts = read_array("300pts.mtx")  # 300 random points in 2D
    var rc = RipsComplex(pts, 0.15)
    ref cmplx = rc.chain_complex()  # boundary operators B_1, B_2
    ref b1 = cmplx.boundary(1)  # edge boundary operator
    ref b2 = cmplx.boundary(2)  # face boundary operator
    print(
        "vertices:",
        rows_of(rc.simplices[0]),
        " edges:",
        rows_of(rc.simplices[1]),
        " triangles:",
        rows_of(rc.simplices[2]),
    )

    nm.random.seed(0)
    var x = nm.random.rand[f64](Shape(b1.ncols))  # random 1-chain
    # decompose x using the discrete Hodge decomposition
    var alpha = cg(b1 @ b1.T(), b1 @ x, rtol=1e-8).take_x()
    var beta = cg(b2.T() @ b2, b2.T() @ x, rtol=1e-8).take_x()
    var h = x - b1.T() @ alpha - b2 @ beta  # harmonic component of x
    var hmax: Float64 = 0
    for i in range(h.size):
        hmax = max(hmax, abs(h.unsafe_get(i)))
    h /= hmax  # normalize h

    print("norm(h)        ", norm2(h))
    print("norm(b1 h)     ", norm2(b1 @ h), " (should be small)")
    print("norm(b2^T h)   ", norm2(b2.T() @ h), " (should be small)")

    with open("harmonic.txt", "w") as f:
        for e in range(h.size):
            f.write(
                iget(rc.simplices[1], e, 0),
                " ",
                iget(rc.simplices[1], e, 1),
                " ",
                abs(h.unsafe_get(e)),
                "\n",
            )
    print("wrote harmonic.txt")
    write_vtu(
        "harmonic.vtu",
        pts,
        rc.simplices[1],
        cell_data=[VTKField("h", h)],
        format=.APPENDED,
    )
    print("wrote harmonic.vtu")

    # plot the results
    var show = show_requested()
    var plt = pyplot(interactive=show)
    var P = pts.to_numpy()

    # nodes of the Rips complex
    plt.figure()
    plt.plot(P.T[0], P.T[1], "ko")
    _fiddle(plt, "Nodes of the Complex", "nodes.png")

    # the Rips complex (2D triangle mesh)
    plt.figure()
    _ = triplot(pts, rc.simplices[2])
    _fiddle(plt, "The Rips Complex", "complex.png")

    # the harmonic 1-chain, as line widths
    var widths = real_vector(h.size)
    for i in range(h.size):
        widths.unsafe_set(i, min(max(15 * abs(h.unsafe_get(i)), 1.0), 15.0))
    plt.figure()
    _ = lineplot(pts, rc.simplices[1], False, widths)
    _fiddle(plt, "Harmonic 1-chain of the Rips complex", "harmonic.png")
    print("wrote nodes.png, complex.png, and harmonic.png")
    if show:
        plt.show()  # one window per figure; returns when all are closed


def _fiddle(plt: PythonObject, title: String, path: String) raises:
    plt.title(title)
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.savefig(path)
