# Read ASCII vertex and element files, write a MojoDEC/PyDEC mesh, read it
# back, and plot it to square_8.png. The written files should match those
# in ExpectedOutput/.
#
# Run from the repository root:  pixi run examples MakeMesh [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build make_mesh.mojo [--show]
# --show also opens the figures in windows (see examples/README.md).

import numojo as nm
from numojo.prelude import *

from mojodec import SimplicialMesh, loadtxt, read_mesh, write_mesh
from mojodec.core.utils import rows_of
from mojodec.vis import pyplot, show_requested


def main() raises:
    var vertices = loadtxt("v.txt")
    var elements = loadtxt[DType.int]("s.txt")  # 1-based in the file
    for i in range(elements.size):
        elements.unsafe_set(i, elements.unsafe_get(i) - 1)
    var mymesh = SimplicialMesh(vertices, elements)
    write_mesh("square_8.xml", mymesh, format="basic")
    var rmesh = read_mesh("square_8.xml")
    print(rmesh)
    var same = True
    for i in range(mymesh.vertices.size):
        same = same and mymesh.vertices.unsafe_get(
            i
        ) == rmesh.vertices.unsafe_get(i)
    for i in range(mymesh.elements.size):
        same = same and mymesh.elements.unsafe_get(
            i
        ) == rmesh.elements.unsafe_get(i)
    var expected = read_mesh("ExpectedOutput/square_8.xml")
    for i in range(expected.elements.size):
        same = same and expected.elements.unsafe_get(
            i
        ) == rmesh.elements.unsafe_get(i)
    for i in range(expected.vertices.size):
        same = same and expected.vertices.unsafe_get(
            i
        ) == rmesh.vertices.unsafe_get(i)
    print("round trip matches input and ExpectedOutput:", same)

    var show = show_requested()
    var plt = pyplot(interactive=show)
    var V = rmesh.vertices.to_numpy()
    plt.triplot(V.T[0], V.T[1], rmesh.elements.to_numpy())
    plt.savefig("square_8.png")
    print("wrote square_8.png")
    if show:
        plt.show()  # one window per figure; returns when all are closed
