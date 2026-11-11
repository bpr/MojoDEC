# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""MojoDEC's side of the cross-library comparison (see README.md here).

Prints one JSON object per line: `{"lib", "mesh", "phase", "ms",
"check"}`, `ms` the best of `repeats` runs, `check` a list of numbers every
library should agree on for that phase (see README.md). `run.py` sets the thread
count through `MOJODEC_NUM_THREADS`.

```sh
pixi run mojo run -D ASSERT=none -I . -I build \\
    benchmarks/compare/mojodec_bench.mojo build/compare/meshes 3 [MESH...]
```
"""

from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns

import numojo as nm
from numojo.prelude import *
from mojodec import *
from mojodec.routines.io.scan import write_float


def now() -> Int:
    return Int(perf_counter_ns())


def ms_since(t: Int) -> Float64:
    return Float64(now() - t) / 1e6


def emit(mesh: String, phase: String, ms: Float64, checks: List[Float64]):
    var s = String(
        '{"lib": "mojodec", "mesh": "',
        mesh,
        '", "phase": "',
        phase,
        '", "ms": ',
    )
    write_float(s, ms)
    s += ', "check": ['
    for i in range(len(checks)):
        if i > 0:
            s += ", "
        write_float(s, checks[i])
    s += "]}"
    print(s)


def frobenius(A: CSRMatrix) -> Float64:
    var s: Float64 = 0
    for v in A.data:
        s += v * v
    return sqrt(s)


def bench_mesh(dir: String, name: String, repeats: Int) raises:
    var V = read_array[DType.float64](dir + "/" + name + ".vertices.arr")
    var E = read_array[DType.int](dir + "/" + name + ".elements.arr")
    var n = E.shape[1] - 1

    # topology: every face, d and boundary at each level
    var best: Float64 = 1e300
    var checks = List[Float64]()
    for _ in range(repeats):
        var t = now()
        var sc = SimplicialComplex(V, E, compute_geometry=False)
        best = min(best, ms_since(t))
        checks = List[Float64]()
        for k in range(n + 1):
            checks.append(Float64(sc[k].num_simplices))
    emit(name, "topology", best, checks)

    # geometry: circumcenters, primal and dual volumes, Hodge stars
    best = 1e300
    var sc = SimplicialComplex(V, E, compute_geometry=False)
    for _ in range(repeats):
        sc = SimplicialComplex(V, E, compute_geometry=False)
        var t = now()
        sc.compute_geometry()
        best = min(best, ms_since(t))
    checks = List[Float64]()
    for k in range(n + 1):
        var total: Float64 = 0
        for v in sc[k].star.values:
            total += v
        checks.append(total)
    emit(name, "geometry", best, checks)

    # build: both at once, as `SimplicialComplex(V, E)` does by default
    best = 1e300
    for _ in range(repeats):
        var t = now()
        var full = SimplicialComplex(V, E)
        best = min(best, ms_since(t))
        _ = full^
    emit(name, "build", best, List[Float64]())

    for k in range(n + 1):
        best = 1e300
        for _ in range(repeats):
            var t = now()
            var M = whitney_innerproduct(sc, k)
            best = min(best, ms_since(t))
            checks = [frobenius(M)]
        emit(name, "whitney_" + String(k), best, checks)


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: mojodec_bench.mojo MESH_DIR REPEATS [MESH...]")
    var dir = String(args[1])
    var repeats = atol(String(args[2]))
    var names = List[String]()
    if len(args) > 3:
        for i in range(3, len(args)):
            names.append(String(args[i]))
    else:
        with open(dir + "/meshes.txt", "r") as f:
            for line in f.read().splitlines():
                var fields = line.split()
                if len(fields) > 0:
                    names.append(String(fields[0]))
    for name in names:
        bench_mesh(dir, name, repeats)
