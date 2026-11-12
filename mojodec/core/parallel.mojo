# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Parallel loops (`mojodec.core.parallel`).
=========================================
`parallel_ranges` splits a loop into one chunk per logical core (per
physical core for loops marked `memory_bound`, such as sparse
matrix-vector products) and runs them on MAX's thread pool (`max.algorithm.parallelize`), the only use of
MAX in MojoDEC. A call costs about 200 microseconds however little work it
does, so loops below a caller-chosen size run serially on the calling
thread.

```mojo
var out = List[Float64](length=n, fill=0)
var op = out.unsafe_ptr()

def body(chunk: Int, lo: Int, hi: Int) {op}:
    for i in range(lo, hi):
        op[unsafe_offset=i] = Float64(i) * 2

parallel_ranges(n, 50_000, body)
```

Setting the environment variable `MOJODEC_NUM_THREADS` to a positive
integer caps the number of chunks (so `MOJODEC_NUM_THREADS=1` runs every
loop serially, as single-threaded benchmarks need); unset, or not a
positive integer, it leaves one chunk per logical (or physical) core.

The body must not raise, and chunks must write to disjoint memory. A loop
that can fail records it at its chunk index (there are `num_chunks(n,
serial_below)` of them) and raises after `parallel_ranges` returns.

Disjoint is not enough for scratch space a body writes in its inner loop:
two chunks' small lists can share a cache line, and every write then
takes the line from the other core (false sharing; it made the 3D dual
walk slower on two threads than on one). Allocate it with
`chunk_scratch`, which leaves room after the list.
"""

from std.os import getenv
from std.sys import num_logical_cores, num_physical_cores, size_of

from max.algorithm import parallelize


def num_chunks(n: Int, serial_below: Int, *, memory_bound: Bool = False) -> Int:
    """The number of chunks `parallel_ranges` splits a loop into.

    Args:
        n: The number of iterations.
        serial_below: Loops of fewer iterations run as one chunk.
        memory_bound: Whether the loop is limited by memory bandwidth
            (one chunk per physical core) rather than by arithmetic (one
            per logical core).

    Returns:
        0 when `n <= 0`; 1 when `n < serial_below`; else
        `max_threads(memory_bound=memory_bound)`, at most `n`.
    """
    if n <= 0:
        return 0
    if n < serial_below:
        return 1
    return max(1, min(n, max_threads(memory_bound=memory_bound)))


def max_threads(*, memory_bound: Bool = False) -> Int:
    """The most chunks a parallel loop is split into.

    Args:
        memory_bound: Count physical cores instead of logical ones: a
            second hardware thread on a core speeds up arithmetic but not
            streaming through memory, and its chunk costs a thread.

    Returns:
        The number of logical cores (physical cores when `memory_bound`),
        or the value of the environment variable `MOJODEC_NUM_THREADS`
        when it is a smaller positive integer.
    """
    var cores = num_physical_cores() if memory_bound else num_logical_cores()
    var s = getenv("MOJODEC_NUM_THREADS")
    if not s:
        return cores
    try:
        var cap = atol(s)
        if cap > 0:
            return min(cap, cores)
    except:
        pass
    return cores


comptime _SCRATCH_PAD_BYTES = 128
"""Unused bytes after a chunk's scratch list: a cache line, and the
second line Intel's spatial prefetcher fetches with it."""


def chunk_scratch[T: Copyable](length: Int, fill: T) -> List[T]:
    """Scratch space for one chunk of a parallel loop.

    The list holds `length` copies of `fill` and has at least 128 unused
    bytes of capacity after them. When every chunk's scratch has that
    room, no two chunks' elements share a cache line, so writes in one
    chunk's inner loop do not slow down another's.

    Parameters:
        T: The element type.

    Args:
        length: The number of elements.
        fill: The initial value of each.

    Returns:
        The list (do not grow it past `length`).
    """
    var out = List[T](capacity=length + _SCRATCH_PAD_BYTES // size_of[T]() + 1)
    for _ in range(length):
        out.append(fill.copy())
    return out^


def parallel_ranges[
    F: def(Int, Int, Int) -> None
](n: Int, serial_below: Int, body: F, *, memory_bound: Bool = False):
    """Run `body(chunk, lo, hi)` over `0..n` split into `num_chunks(n,
    serial_below)` contiguous chunks, in parallel when there is more than
    one.

    Parameters:
        F: The body's closure type.

    Args:
        n: The number of iterations.
        serial_below: Loops of fewer iterations run serially; choose it so
            that they take less than a few hundred microseconds.
        body: Called with each chunk's index and half-open range; must not
            raise.
        memory_bound: Whether the loop is limited by memory bandwidth, as
            for `num_chunks`.
    """
    var chunks = num_chunks(n, serial_below, memory_bound=memory_bound)
    if chunks == 0:
        return
    if chunks == 1:
        body(0, 0, n)
        return

    def chunk(c: Int) {imm body, n, chunks}:
        body(c, n * c // chunks, n * (c + 1) // chunks)

    parallelize(chunk, chunks)
