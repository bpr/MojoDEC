# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""Write the meshes every library in the comparison reads.

Each mesh is a Kuhn grid of the unit square or cube with its interior
vertices moved by up to 5% of the cell size (a fixed pseudo-random
sequence), so no dual volume is exactly zero and no two circumcenters tie.
The jitter keeps every element positively oriented, which is checked here.

Files go to `<dir>/<name>.vertices.arr` (float64) and
`<dir>/<name>.elements.arr` (int64), in PyDEC's binary array format, with
`<dir>/meshes.txt` listing `name dim num_vertices num_elements`.

```sh
pixi run mojo run -I . -I build benchmarks/compare/meshes.mojo build/compare/meshes
```
"""

from std.sys import argv
from std.os import makedirs
from std.os.path import exists

import numojo as nm
from numojo.prelude import *
from mojodec import *
from mojodec.core.utils import fget, fset, iget


struct Lcg(Movable):
    """A 64-bit linear congruential generator: the same jitter on every run
    and machine."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def uniform(mut self) -> Float64:
        """A value in [-1, 1)."""
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Float64(self.state >> 11) / Float64(1 << 53) * 2.0 - 1.0


def jittered_kuhn(dim: Int, n: Int) raises -> SimplicialMesh:
    var dims = List[Int]()
    for _ in range(dim):
        dims.append(n)
    var mesh = kuhn_grid(dims)
    var h = 1.0 / Float64(n)
    var rng = Lcg(UInt64(1000 * dim + n))
    var nv = mesh.vertices.shape[0]
    for i in range(nv):
        var interior = True
        for a in range(dim):
            var x = fget(mesh.vertices, i, a)
            if x < 0.5 * h or x > 1.0 - 0.5 * h:
                interior = False
        if interior:
            for a in range(dim):
                var x = fget(mesh.vertices, i, a)
                fset(mesh.vertices, i, a, x + 0.05 * h * rng.uniform())
    _require_positive(mesh, dim)
    return mesh^


def _require_positive(mesh: SimplicialMesh, dim: Int) raises:
    """Raise unless every element has a positive signed volume."""
    var V = mesh.vertices.copy()
    var E = mesh.elements.copy()
    for e in range(E.shape[0]):
        var p0 = iget(E, e, 0)
        var m = List[Float64]()
        for r in range(1, dim + 1):
            var p = iget(E, e, r)
            for a in range(dim):
                m.append(fget(V, p, a) - fget(V, p0, a))
        var det: Float64
        if dim == 2:
            det = m[0] * m[3] - m[1] * m[2]
        else:
            det = (
                m[0] * (m[4] * m[8] - m[5] * m[7])
                - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6])
            )
        if not det > 0:
            raise Error("element " + String(e) + " is not positively oriented")


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: meshes.mojo OUTPUT_DIR")
    var dir = String(args[1])
    if not exists(dir):
        makedirs(dir)
    var index = String()
    # (dimension, cells per axis): 20k/80k/320k triangles, 6k/48k/384k tets
    var specs: List[Tuple[Int, Int]] = [
        (2, 100),
        (2, 200),
        (2, 400),
        (3, 10),
        (3, 20),
        (3, 40),
    ]
    for spec in specs:
        var dim = spec[0]
        var n = spec[1]
        var name = ("tri_" if dim == 2 else "tet_") + String(n)
        var mesh = jittered_kuhn(dim, n)
        write_array(dir + "/" + name + ".vertices.arr", mesh.vertices)
        write_array(dir + "/" + name + ".elements.arr", mesh.elements)
        index += (
            name
            + " "
            + String(dim)
            + " "
            + String(mesh.vertices.shape[0])
            + " "
            + String(mesh.elements.shape[0])
            + "\n"
        )
        print(
            name,
            mesh.vertices.shape[0],
            "vertices",
            mesh.elements.shape[0],
            "elements",
        )
    with open(dir + "/meshes.txt", "w") as f:
        f.write(index)
