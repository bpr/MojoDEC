# CLAUDE.md

> **Note:** Core project architecture and coding standards are defined in [AGENTS.md](./AGENTS.md).

@AGENTS.md

This file holds what is specific to Claude Code: how to work in this
repository. Commands and everything about the code itself are in
AGENTS.md.

## Workflow

- Run single test files while iterating; run `pixi run test` before
  calling a change done.
- After touching `third_party/numojo`, run `pixi run numojo` and update
  `third_party/numojo-mojo-1.3.patch`.
- After changing doc strings, run `pixi run doc-check`; if it improved,
  `pixi run doc-check --update` lowers the baseline.
- After changing geometry, solvers or I/O, run the examples and compare
  with README.md's Examples table.
- `--show` opens matplotlib windows: don't run it from inside Claude Code.
  Give the user the command to run in their own terminal.

## Where to look

| Question                                   | Where                                   |
| ------------------------------------------ | --------------------------------------- |
| Commands, architecture, conventions, traps | AGENTS.md                               |
| MojoDEC ↔ PyDEC file mapping, differences  | README.md                               |
| Expected example results                   | README.md, Examples table               |
| Doc string style                           | `docs/writing-docs.md`                  |
| Vendored NuMojo patch                      | `third_party/README.md`                 |
