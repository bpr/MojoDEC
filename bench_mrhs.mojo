from mojodec import SimplicialComplex, SparseLDLT, FactorLayout, kuhn_grid, whitney_innerproduct, amd_ordering
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import probe_vector, real_array, fget, fset, real_vector
from std.time import perf_counter_ns

def lap(sc: SimplicialComplex, k: Int) raises -> CSRMatrix:
    var A = whitney_innerproduct(sc, k)
    if k < sc.complex_dimension():
        A = A + sc[k].d.transpose() @ whitney_innerproduct(sc, k + 1) @ sc[k].d
    return A^

def run(A: CSRMatrix, name: String, lay: FactorLayout, steps: Int) raises:
    var n = A.nrows
    var F = SparseLDLT(A, layout=lay, refinement_steps=steps)
    for m in [1, 4, 16, 64]:
        var B = real_array(n, m)
        for c in range(m):
            var b = probe_vector(n, UInt64(c + 3))
            for i in range(n):
                fset(B, i, c, b.unsafe_get(i))
        var best: Float64 = 1e30
        var X = F.solve(B)
        for _ in range(3):
            var t0 = perf_counter_ns()
            X = F.solve(B)
            best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
        var t1 = perf_counter_ns()
        var bad = 0
        for c in range(m):
            var b = real_vector(n)
            for i in range(n):
                b.unsafe_set(i, fget(B, i, c))
            var x = F.solve(b)
            for i in range(n):
                if x.unsafe_get(i) != fget(X, i, c):
                    bad += 1
        var singles = Float64(perf_counter_ns() - t1) / 1e6
        print(name, lay, "refine", steps, "m", m, "block ms", best, "m singles ms", singles, "mismatches", bad)

def main() raises:
    var A2 = lap(SimplicialComplex(kuhn_grid([300, 300])), 0)
    run(A2, "2d 300", FactorLayout.SUPERNODAL, 0)
    run(A2, "2d 300", FactorLayout.SIMPLICIAL, 1)
    var A3 = lap(SimplicialComplex(kuhn_grid([30, 30, 30])), 0)
    run(A3, "3d 30", FactorLayout.SUPERNODAL, 0)
    run(A3, "3d 30", FactorLayout.SUPERNODAL, 1)
