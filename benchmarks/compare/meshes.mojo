# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""Write the meshes every library in the comparison reads.

Each mesh is a Kuhn grid of the unit square or cube with its interior
vertices moved by up to 5% of the cell size (a fixed pseudo-random
sequence), so no dual volume is exactly zero and no two circumcenters tie.
The jitter keeps every element positively oriented, which is checked here.

A Kuhn grid numbers its vertices in spatial order. `tri_400_shuffled` and
`tet_40_shuffled` are `tri_400` and `tet_40` with their vertex numbers
randomly permuted (same geometry, same element order), as a mesh read
from a file with an incoherent numbering would be: MojoDEC's parallel
dual-volume walk splits work by blocks of vertex numbers, so this is its
worst case.

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
from mojodec.core.utils import fget, fset, iget, iset


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

    def below(mut self, n: Int) -> Int:
        """A value in [0, n)."""
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 11) % UInt64(n))


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


def shuffled(mesh: SimplicialMesh, seed: UInt64) raises -> SimplicialMesh:
    """`mesh` with its vertex numbers randomly permuted (Fisher-Yates):
    vertex `i` becomes `p[i]`. Element rows keep their order, and each
    row's vertices their positions, so orientations are unchanged."""
    var nv = mesh.vertices.shape[0]
    var d = mesh.vertices.shape[1]
    var p = List[Int](length=nv, fill=0)
    for i in range(nv):
        p[i] = i
    var rng = Lcg(seed)
    for i in range(nv - 1, 0, -1):
        var j = rng.below(i + 1)
        var t = p[i]
        p[i] = p[j]
        p[j] = t
    var V = mesh.vertices.copy()
    for i in range(nv):
        for a in range(d):
            fset(V, p[i], a, fget(mesh.vertices, i, a))
    var E = mesh.elements.copy()
    for e in range(E.shape[0]):
        for j in range(E.shape[1]):
            iset(E, e, j, p[iget(mesh.elements, e, j)])
    return SimplicialMesh(V, E)


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
    # (dimension, cells per axis, shuffled vertex numbers):
    # 20k/80k/320k triangles, 6k/48k/384k tets
    var specs: List[Tuple[Int, Int, Bool]] = [
        (2, 100, False),
        (2, 200, False),
        (2, 400, False),
        (2, 400, True),
        (3, 10, False),
        (3, 20, False),
        (3, 40, False),
        (3, 40, True),
    ]
    for spec in specs:
        var dim = spec[0]
        var n = spec[1]
        var name = ("tri_" if dim == 2 else "tet_") + String(n)
        var mesh = jittered_kuhn(dim, n)
        if spec[2]:
            name += "_shuffled"
            mesh = shuffled(mesh, UInt64(7000 + 1000 * dim + n))
            _require_positive(mesh, dim)
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
