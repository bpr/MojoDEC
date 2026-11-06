# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""A pure-Mojo `parallel_for` on POSIX threads, timed against MAX's
`parallelize`.

Kept as a baseline for benchmarking MojoDEC's parallel loops. Threads are
created and joined on every call, so it suits loops of at least tens of
microseconds per thread.

```sh
pixi run mojo run benchmarks/parallel_for.mojo
```
"""

from std.ffi import external_call
from std.time import perf_counter_ns
from max.algorithm import parallelize


@fieldwise_init
struct _Job(Copyable):
    """One thread's share: the body's address and its index range."""

    var body: Int
    var lo: Int
    var hi: Int


def _run[
    F: def(Int) -> None
](arg: OpaquePointer[MutAnyOrigin]) -> OpaquePointer[MutAnyOrigin]:
    """The thread entry point: run the body over the job's range."""
    var job = arg.unsafe_bitcast[_Job]()
    var body = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=job[].body
    ).unsafe_bitcast[F]()
    for i in range(job[].lo, job[].hi):
        body[](i)
    return arg


def parallel_for[F: def(Int) -> None](n: Int, body: F, workers: Int):
    """Run `body(i)` for `i` in `0..n`, split into `workers` contiguous
    ranges, one per thread (the calling thread takes the first).

    The body must not raise and must not write where another index writes.
    """
    var jobs = List[_Job](capacity=workers)
    var tids = List[UInt64](length=workers, fill=0)
    var addr = Int(Pointer(to=body))
    for w in range(workers):
        jobs.append(_Job(addr, n * w // workers, n * (w + 1) // workers))
    var entry = _run[F]
    for w in range(1, workers):
        _ = external_call["pthread_create", Int32](
            Pointer(to=tids[w]),
            Int(0),
            entry,
            Pointer(to=jobs[w]).unsafe_bitcast[NoneType](),
        )
    _ = _run[F](Pointer(to=jobs[0]).unsafe_bitcast[NoneType]())
    for w in range(1, workers):
        _ = external_call["pthread_join", Int32](tids[w], Int(0))


def main():
    var n = 1 << 24
    var out = List[Float64](length=n, fill=0)
    var po = out.unsafe_ptr()

    def work(r: Int) {po}:
        var x = Float64(r)
        for _ in range(20):
            x = x * 0.999 + 1.0
        po[unsafe_offset=r] = x

    var chunk = n // 64

    def work_chunk(c: Int) {po, chunk}:
        for r in range(c * chunk, (c + 1) * chunk):
            var x = Float64(r)
            for _ in range(20):
                x = x * 0.999 + 1.0
            po[unsafe_offset=r] = x

    for w in [1, 2, 4, 8, 16]:
        var t = perf_counter_ns()
        parallel_for(n, work, w)
        print(
            "pthreads",
            w,
            "threads:",
            Float64(perf_counter_ns() - t) / 1e6,
            "ms",
        )
    var check = out[n - 1]
    var t = perf_counter_ns()
    parallelize(work_chunk, 64)
    print("max parallelize:", Float64(perf_counter_ns() - t) / 1e6, "ms")
    print("same result:", out[n - 1] == check)
