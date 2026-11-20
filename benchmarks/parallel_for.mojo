# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""The thread pool of molina (`parallel_ranges`) timed against MAX's
`parallelize`, which it replaced: the cost of a call that does no work,
on workers still polling and on workers asleep, and a loop with work.

```sh
pixi run mojo run -D ASSERT=none -I . benchmarks/parallel_for.mojo
```
"""

from std.sys import num_logical_cores
from std.time import perf_counter_ns

from max.algorithm import parallelize

from molina.parallel import parallel_ranges


def _idle(ns: Int):
    """Spin for `ns` nanoseconds (long enough for idle workers to sleep)."""
    var s = perf_counter_ns()
    while perf_counter_ns() - s < ns:
        pass


def main():
    var cores = num_logical_cores()
    var n = 1 << 22
    var out = List[Float64](length=n, fill=0)
    var po = out.unsafe_ptr()

    def mark(c: Int, lo: Int, hi: Int) {po}:
        po[unsafe_offset=c] = 1.0

    def mark1(c: Int) {po}:
        po[unsafe_offset=c] = 1.0

    print("empty call, back to back (us):")
    for w in [2, 8, cores]:
        var t0 = perf_counter_ns()
        for _ in range(1000):
            parallel_ranges(w, 0, mark)
        var t1 = perf_counter_ns()
        for _ in range(1000):
            parallelize(mark1, w)
        var t2 = perf_counter_ns()
        print(
            "  ",
            w,
            "chunks: molina",
            Float64(t1 - t0) / 1e6,
            " MAX",
            Float64(t2 - t1) / 1e6,
        )

    var pool = 0
    var mx = 0
    for _ in range(20):
        _idle(2_000_000)
        var t = perf_counter_ns()
        parallel_ranges(cores, 0, mark)
        pool += perf_counter_ns() - t
        _idle(2_000_000)
        t = perf_counter_ns()
        parallelize(mark1, cores)
        mx += perf_counter_ns() - t
    print(
        "empty call after 2 ms idle (us): molina",
        Float64(pool) / 2e4,
        " MAX",
        Float64(mx) / 2e4,
    )

    def work(c: Int, lo: Int, hi: Int) {po}:
        for r in range(lo, hi):
            var x = Float64(r)
            for _ in range(20):
                x = x * 0.999 + 1.0
            po[unsafe_offset=r] = x

    var chunk = n // cores

    def work1(c: Int) {po, chunk}:
        for r in range(c * chunk, (c + 1) * chunk):
            var x = Float64(r)
            for _ in range(20):
                x = x * 0.999 + 1.0
            po[unsafe_offset=r] = x

    var t = perf_counter_ns()
    parallel_ranges(n, 0, work)
    var t1 = perf_counter_ns()
    parallelize(work1, cores)
    var t2 = perf_counter_ns()
    print(
        "4M updates of 20 steps (ms): molina",
        Float64(t1 - t) / 1e6,
        " MAX",
        Float64(t2 - t1) / 1e6,
    )
