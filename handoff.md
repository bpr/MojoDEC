# Handoff: sparse direct solver work (2026-11-17)

Prompt: "Read handoff.md and continue." Everything below is committed
through 30d598b; nothing is in progress in the working tree. Untracked
root notes (announcement.md, dexterior.md, ...) stay uncommitted, as
does this file and bench_mrhs.mojo.

## Context

Recent commits (sparse direct, `mojodec/routines/sparse_ldlt.mojo`,
`mojodec/routines/supernodal.mojo`):

- 8437294 parallel supernodal LDL^T (one task graph, serial bits)
- 6e6ac09 Gilbert-Ng-Peyton column counts
- 2b575d6 `SparseLDLT.solve(B)`, n x m right-hand sides
- 8cccf79 typed `solve(List[...])` overloads
- 2d4eb48 refinement residuals per column block, in parallel
  (`_solve_many` -> `_solve_block`, one `parallel_ranges` call)
- 30d598b GEMM-style block solve: supernodal `solve_columns` gathers the
  rows below each triangle for all columns (`_gather`), then
  `_sub_product` going down and `_sub_dots` (2 x 2 tiles) going up.
  Single solves are the one-column case, so columns stay bit-equal to
  single solves. Single-solve bits may differ from before only in the
  sign of a zero.

The last two were verified (tests, examples, doc-check, docs-site) but
NOT benchmarked: the old machine was loaded (load average 10-24).

## Tasks to run on the quiet machine

1. Build the benchmark (`bench_mrhs.mojo`, 2D 300^2 and 3D 30^3
   Laplacians, m = 1, 4, 16, 64, refinement 0 and 1; it also checks
   columns against single solves, "mismatches" must be 0) at three
   commits, release mode:

   ```sh
   pixi run numojo
   for c in 8cccf79 2d4eb48 30d598b; do
     git checkout -q $c
     pixi run mojo build -D ASSERT=none -I . -I build \
       bench_mrhs.mojo -o build/mrhs-$c
   done
   git checkout -q master
   ```

   (bench_mrhs.mojo is untracked, so it survives the checkouts; the
   vendored NuMojo did not change in these commits.)

2. Run each with `MOJODEC_NUM_THREADS=1` and with all cores, best of a
   few runs. Compare:
   - 8cccf79 vs 2d4eb48: refinement residuals in parallel (look at
     `refine 1`, and the 2D rows, where gathers dominated).
   - 2d4eb48 vs 30d598b: the GEMM-style kernel (block ms, 1 thread and
     8 cores; also check m = 1 did not get slower).
3. Update the table in `docs/guide/sparse-direct.md` ("Several
   right-hand sides": m solves / block 1 thread / block 8 cores for 30^3
   m = 16, 64 and 2D 300^2 m = 64; the text says "supernodal, no
   refinement") and the 30^3 m = 64 numbers in AGENTS.md (Sparse direct
   bullet: "344 ms as single solves, 156 ms one thread, 41 ms on 8").
   Consider adding a refinement row. If the 2D one-thread remark ("the
   gathers and scatters each column still does dominate") no longer
   holds, revise it.
4. If the GEMM kernel is not a win somewhere (e.g. m = 1), report that
   before changing anything; the bit-equality design (single solve =
   one-column case) must stay.
5. `pixi run test` once more on the new machine, then write
   commit_msg.txt for the doc update.

## Open ideas (not requested yet)

- Tile the triangular parts of the block solve too (still column by
  column).
- Saddle-point constraint ordering (dense Schur complement).
- GMG/AMG items and older review items (see docs/plan.md,
  mojodec-audit.md, linear-algebra-roadmap.md).
