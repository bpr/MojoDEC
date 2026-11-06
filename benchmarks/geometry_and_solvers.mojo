# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""Timings of the parallel loops: building a complex with its geometry,
Whitney mass matrices, sparse matrix-vector products and `cg`.

Each line prints a checksum too, so runs before and after a change can be
compared for the same results.

```sh
pixi run bench
```
"""

from std.time import perf_counter_ns
import numojo as nm
from numojo.prelude import *
from mojodec import *
from mojodec.core.utils import real_vector

def now() -> Int:
    return Int(perf_counter_ns())

def ms(t: Int) -> Float64:
    return Float64(now() - t) / 1e6

def main() raises:
    print("best of several runs, in ms")
    var g320 = simplicial_grid_2d(400)
    var best_build = 1e9
    var best_cc = 1e9
    var best_vol = 1e9
    var sc = SimplicialComplex(g320.vertices, g320.elements)
    for _ in range(5):
        var t = now()
        var s2 = SimplicialComplex(g320.vertices, g320.elements)
        best_build = min(best_build, ms(t))
        t = now()
        for d in range(3):
            s2.compute_circumcenters(d)
        best_cc = min(best_cc, ms(t))
        t = now()
        for d in range(3):
            s2.compute_primal_volume(d)
        best_vol = min(best_vol, ms(t))
    print("320k build+geometry", best_build, " circumcenters", best_cc, " primal volumes", best_vol)

    var g80 = simplicial_grid_2d(200)
    var sc80 = SimplicialComplex(g80.vertices, g80.elements)
    var chk: Float64 = 0
    for k in range(1, 3):
        var best = 1e9
        for _ in range(5):
            var t = now()
            var M = whitney_innerproduct(sc80, k)
            best = min(best, ms(t))
            chk = M.data[len(M.data) // 3] + M.data[7]
        print("80k whitney k=", k, best, "ms  check", chk)

    var g90 = simplicial_grid_2d(300)
    var sc90 = SimplicialComplex(g90.vertices, g90.elements)
    var A = sc90[0].d.T() @ (sc90[1].star @ sc90[0].d)
    var x = real_vector(A.ncols)
    for i in range(x.size):
        x.unsafe_set(i, Float64((i * 7) % 13) - 6.0)
    var y = real_vector(A.nrows)
    var bestmv = 1e9
    for _ in range(20):
        var t = now()
        A.matvec_into(x, y)
        bestmv = min(bestmv, ms(t))
    print("matvec n=", A.nrows, "nnz=", A.nnz(), bestmv, "ms  y[5]", y.unsafe_get(5))
    var b = A @ x
    var bestcg = 1e9
    var iters = 0
    var x5: Float64 = 0
    for _ in range(3):
        var t = now()
        var res = cg(A, b, rtol=1e-8)
        bestcg = min(bestcg, ms(t))
        iters = res.iterations
        x5 = res^.take_x().unsafe_get(5)
    print("cg", bestcg, "ms, iterations", iters, " x[5]", x5)
    var g800 = simplicial_grid_2d(800)
    var s8 = SimplicialComplex(g800.vertices, g800.elements)
    var A8 = s8[0].d.T() @ (s8[1].star @ s8[0].d)
    var x8 = real_vector(A8.ncols)
    for i in range(x8.size):
        x8.unsafe_set(i, Float64((i * 7) % 13) - 6.0)
    var b8 = A8 @ x8
    var t = now()
    var r8 = cg(A8, b8, rtol=1e-6)
    var el = ms(t)
    var it8 = r8.iterations
    print("cg n=", A8.nrows, "nnz=", A8.nnz(), el, "ms, iterations", it8, " x[5]", r8^.take_x().unsafe_get(5))
