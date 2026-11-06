# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Parallel loops (`mojodec.core.parallel`).
=========================================
`parallel_ranges` splits a loop into one chunk per physical core and runs
them on MAX's thread pool (`max.algorithm.parallelize`), the only use of
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

The body must not raise, and chunks must write to disjoint memory. A loop
that can fail records it at its chunk index (there are `num_chunks(n,
serial_below)` of them) and raises after `parallel_ranges` returns.
"""

from std.sys import num_physical_cores

from max.algorithm import parallelize


def num_chunks(n: Int, serial_below: Int) -> Int:
    """The number of chunks `parallel_ranges` splits a loop into.

    Args:
        n: The number of iterations.
        serial_below: Loops of fewer iterations run as one chunk.

    Returns:
        0 when `n <= 0`; 1 when `n < serial_below`; else the number of
        physical cores, at most `n`.
    """
    if n <= 0:
        return 0
    if n < serial_below:
        return 1
    return max(1, min(n, num_physical_cores()))


def parallel_ranges[
    F: def(Int, Int, Int) -> None
](n: Int, serial_below: Int, body: F):
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
    """
    var chunks = num_chunks(n, serial_below)
    if chunks == 0:
        return
    if chunks == 1:
        body(0, 0, n)
        return

    def chunk(c: Int) {imm body, n, chunks}:
        body(c, n * c // chunks, n * (c + 1) // chunks)

    parallelize(chunk, chunks)
