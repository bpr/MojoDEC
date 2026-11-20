# ===----------------------------------------------------------------------=== #
# molina: linear algebra in Mojo (sparse and dense matrices, direct and
# iterative solvers, eigensolvers, multigrid), split from MojoDEC.
# ===----------------------------------------------------------------------=== #
"""
Parallel loops (`molina.parallel`).
===================================
`parallel_ranges` splits a loop into one chunk per logical core (per
physical core for loops marked `memory_bound`, such as sparse
matrix-vector products) and runs them on molina's own thread pool:
POSIX threads, started on first use (one fewer than the chunks; the
calling thread runs chunks too) and kept for the life of the process.
Idle workers poll for the next loop for 100 microseconds, then sleep on
a futex. A call on warm workers costs 1 to 3 microseconds, on sleeping
ones about 8 (MAX's `parallelize`, used before, 12 to 130), so loops of
a few microseconds' work can run in parallel; loops below a
caller-chosen size still run serially on the calling thread. Linux on
x86-64 only (futexes, `pause`).

```mojo
var out = List[Float64](length=n, fill=0)
var op = out.unsafe_ptr()

def body(chunk: Int, lo: Int, hi: Int) {op}:
    for i in range(lo, hi):
        op[unsafe_offset=i] = Float64(i) * 2

parallel_ranges(n, 50_000, body)
```

Setting the environment variable `MOLINA_NUM_THREADS` to a positive
integer caps the number of chunks (so `MOLINA_NUM_THREADS=1` runs every
loop serially, as single-threaded benchmarks need); unset, or not a
positive integer, it leaves one chunk per logical (or physical) core.

The body must not raise, and chunks must write to disjoint memory. A loop
that can fail records it at its chunk index (there are `num_chunks(n,
serial_below)` of them) and raises after `parallel_ranges` returns.
Chunks are claimed in index order by whichever thread is free, so a body
may wait for a lower chunk's work but never for a higher one's. A loop
started while another runs (from inside a body, or from a second
thread) runs its chunks serially on the calling thread. Workers keep
the default floating-point environment, as the calling thread does
(subnormals are not flushed to zero), so a loop whose chunks own their
outputs gives the serial loop's bits.

Disjoint is not enough for scratch space a body writes in its inner loop:
two chunks' small lists can share a cache line, and every write then
takes the line from the other core (false sharing; it made the 3D dual
walk slower on two threads than on one). Allocate it with
`chunk_scratch`, which leaves room after the list.
"""

from std.atomic import Atomic
from std.ffi import _get_global, external_call
from std.os import getenv
from std.sys import num_logical_cores, num_physical_cores, size_of
from std.sys.intrinsics import llvm_intrinsic
from std.time import perf_counter_ns


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
        or the value of the environment variable `MOLINA_NUM_THREADS`
        when it is a smaller positive integer.
    """
    var cores = num_physical_cores() if memory_bound else num_logical_cores()
    var s = getenv("MOLINA_NUM_THREADS")
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

    _run_chunks(n, chunks, body)


# ---- the thread pool
#
# One block of words per process, found by name through the Mojo
# runtime's globals (`_get_global`), whose teardown after `main` stops
# and joins the workers before `mojo run` frees the code they run. A
# loop's chunks are claimed by compare-and-swap on one 64-bit ticket:
# the loop's generation in the high half (the futex word the workers
# sleep on) and the next chunk in the low half. A claim that succeeds
# saw the ticket unchanged since it read the loop's fields, and the
# caller rewrites them only after closing the last loop's ticket, so no
# straggler from an earlier loop can run a chunk of the next.


comptime _SPIN_NS = 100_000
"""How long an idle worker polls for the next loop before sleeping."""
comptime _SYS_FUTEX = 202
"""`futex(2)`'s number on x86-64 Linux."""
comptime _FUTEX_WAIT_PRIVATE = 128
comptime _FUTEX_WAKE_PRIVATE = 129

comptime _TICKET = 0
"""The loop's generation (high 32 bits, the workers' futex word) and its
next unclaimed chunk (low 32 bits)."""
comptime _DONE = 128
"""Chunks of the current loop finished."""
comptime _SLEEPERS = 256
"""Workers in (or about to enter) `futex_wait`."""
comptime _BUSY = 264
"""1 while a loop runs; a loop started meanwhile runs serially."""
comptime _WORKERS = 272
"""Threads started."""
comptime _STOP = 280
"""1 once the process is exiting: workers return."""
comptime _ENTRY = 384
"""The loop: `_run_chunk[F]`, which runs chunk `c` of the `_Loop` at
`_CTX`, and the number of chunks. Written only while the ticket is
closed."""
comptime _CTX = 392
comptime _CHUNKS = 400
comptime _TIDS = 512
"""The workers' thread ids, one word each (at most one per logical core)."""

comptime _Entry = def(Int, Int) thin -> None
comptime _Handle = Optional[Pointer[NoneType, MutUntrackedOrigin]]
"""The pool as the runtime's globals hold it."""


def _new_pool() -> _Handle:
    """The pool's shared words, zeroed: the hot words on cache lines (and
    prefetched pairs) of their own, then room for a thread id per logical
    core."""
    var bytes = _TIDS + 8 * num_logical_cores()
    var addr = external_call["aligned_alloc", Int](128, bytes)
    _ = external_call["memset", Int](addr, 0, bytes)
    return _at[NoneType](addr).unsafe_origin_cast[MutUntrackedOrigin]()


def _stop_pool(pool: _Handle):
    """Stop and join the workers (the runtime calls this after `main`
    returns, before `mojo run` frees the compiled code they run)."""
    if not pool:
        return
    var p = Int(pool.value())
    Atomic[Int].store(_at[Int](p + _STOP), 1)
    var t = _at[UInt64](p + _TICKET)
    var gen = ((Atomic[UInt64].load(t) >> 32) + 1) & 0xFFFFFFFF
    Atomic[UInt64].store(t, gen << 32)
    _futex(p + _TICKET + 4, _FUTEX_WAKE_PRIVATE, Int(Int32.MAX))
    for w in range(_at[Int](p + _WORKERS)[]):
        _ = external_call["pthread_join", Int32](
            _at[UInt64](p + _TIDS + 8 * w)[], Int(0)
        )
    external_call["free", NoneType](p)


def _pool() -> Int:
    """The pool's address (0 if the runtime could not make it)."""
    var g = _get_global["molina.parallel.pool.1", _new_pool, _stop_pool]()
    return Int(g.value()) if g else 0


@always_inline
def _at[T: AnyType](addr: Int) -> Pointer[T, MutAnyOrigin]:
    return OpaquePointer[MutAnyOrigin](unsafe_from_address=addr).unsafe_bitcast[
        T
    ]()


@always_inline
def _pause():
    llvm_intrinsic["llvm.x86.sse2.pause", NoneType]()


def _futex(addr: Int, op: Int, val: Int):
    _ = external_call["syscall", Int](_SYS_FUTEX, addr, op, val, 0, 0, 0)


def _claim_and_run(p: Int, gen: UInt64):
    """Run chunks of loop `gen` until none is left unclaimed (or a later
    loop has started)."""
    var t = _at[UInt64](p + _TICKET)
    while True:
        var cur = Atomic[UInt64].load(t)
        if cur >> 32 != gen:
            return
        var entry = _at[_Entry](p + _ENTRY)[]
        var ctx = _at[Int](p + _CTX)[]
        var c = Int(cur & 0xFFFFFFFF)
        if c >= _at[Int](p + _CHUNKS)[]:
            return
        if Atomic[UInt64].compare_exchange(t, cur, cur + 1):
            entry(ctx, c)
            _ = Atomic[Int].fetch_add(_at[Int](p + _DONE), 1)


def _worker(p: Int) abi("C") -> Int:
    """A worker thread: wait for a loop (polling, then asleep on the
    futex), claim its chunks, repeat; return once the pool stops."""
    var t = _at[UInt64](p + _TICKET)
    var stop = _at[Int](p + _STOP)
    var sleepers = _at[Int](p + _SLEEPERS)
    var seen = UInt64.MAX  # no generation: join the loop that started us
    while True:
        var start = perf_counter_ns()
        while (
            Atomic[UInt64].load(t) >> 32 == seen and Atomic[Int].load(stop) == 0
        ):
            if perf_counter_ns() - start < _SPIN_NS:
                for _ in range(16):
                    _pause()
                continue
            _ = Atomic[Int].fetch_add(sleepers, 1)
            if (
                Atomic[UInt64].load(t) >> 32 == seen
                and Atomic[Int].load(stop) == 0
            ):
                _futex(p + _TICKET + 4, _FUTEX_WAIT_PRIVATE, Int(seen))
            _ = Atomic[Int].fetch_add(sleepers, -1)
        if Atomic[Int].load(stop) != 0:
            return 0
        seen = Atomic[UInt64].load(t) >> 32
        _claim_and_run(p, seen)


@fieldwise_init
struct _Loop(Movable):
    """What `_run_chunk` needs: the body's address, `n`, the chunks."""

    var body: Int
    var n: Int
    var chunks: Int


def _run_chunk[F: def(Int, Int, Int) -> None](ctx: Int, c: Int):
    """Chunk `c` of the loop whose `_Loop` is at `ctx`."""
    var loop = _at[_Loop](ctx)
    var n = loop[].n
    var k = loop[].chunks
    _at[F](loop[].body)[](c, n * c // k, n * (c + 1) // k)


def _run_chunks[F: def(Int, Int, Int) -> None](n: Int, chunks: Int, body: F):
    """Run `body(c, lo, hi)` for each of `chunks` (2 or more) chunks of
    `0..n` on the pool, the calling thread included."""
    var p = _pool()
    var idle = 0
    if p == 0 or not Atomic[Int].compare_exchange(_at[Int](p + _BUSY), idle, 1):
        for c in range(chunks):
            body(c, n * c // chunks, n * (c + 1) // chunks)
        return
    var workers = _at[Int](p + _WORKERS)
    while workers[] < min(chunks, num_logical_cores()) - 1:
        var entry = _worker
        if (
            external_call["pthread_create", Int32](
                _at[UInt64](p + _TIDS + 8 * workers[]), Int(0), entry, p
            )
            != 0
        ):
            break
        workers[] += 1
    # Close the last loop to stragglers before rewriting its fields: a
    # claim needs the ticket unchanged since it read them.
    var t = _at[UInt64](p + _TICKET)
    var gen = Atomic[UInt64].load(t) >> 32
    Atomic[UInt64].store(t, (gen << 32) | 0xFFFFFFFF)
    gen = (gen + 1) & 0xFFFFFFFF
    var loop = _Loop(Int(Pointer(to=body)), n, chunks)
    _at[_Entry](p + _ENTRY)[] = _run_chunk[F]
    _at[Int](p + _CTX)[] = Int(Pointer(to=loop))
    _at[Int](p + _CHUNKS)[] = chunks
    var done = _at[Int](p + _DONE)
    done[] = 0
    Atomic[UInt64].store(t, gen << 32)
    if Atomic[Int].load(_at[Int](p + _SLEEPERS)) > 0:
        _futex(p + _TICKET + 4, _FUTEX_WAKE_PRIVATE, Int(Int32.MAX))
    _claim_and_run(p, gen)
    var spins = 0
    while Atomic[Int].load(done) < chunks:
        _pause()
        spins += 1
        if spins % 4096 == 0:
            _ = external_call["sched_yield", Int32]()
    _ = loop^
    Atomic[Int].store(_at[Int](p + _BUSY), 0)
