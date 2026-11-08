# Submitting MojoDEC to modular-community

Wait until both of these are available:

- **Mojo 1.3.0 stable.** The modular-community CI builds only against the
  stable channel (`https://conda.modular.com/max`), never `max-nightly`,
  and MojoDEC needs Mojo 1.3.
- **NuMojo for Mojo 1.3.** MojoDEC's API uses NuMojo arrays, and the
  published `numojo` 0.12.0 requires `mojo-compiler >=1.2.0,<1.3.0`.

A recipe that fails CI is deleted automatically after four weeks.

To check:

```sh
pixi search -c https://conda.modular.com/max mojo-compiler
pixi search -c https://conda.modular.com/max max-core
pixi search -c https://prefix.dev/modular-community -c https://conda.modular.com/max numojo
```

Status as of 2026-11-08: the stable channel has Mojo 1.2.0, and the newest
`numojo` is 0.12.0 (Mojo 1.2). The recipe builds and passes its test from
the working tree with the 1.3.0.dev2026110305 nightly and the vendored
NuMojo (`numojo: vendored`).

## Checklist

### 1. Move MojoDEC off the vendored NuMojo

- [ ] In `pixi.toml`, add `numojo` at the new release under
      `[dependencies]`, and add `https://repo.prefix.dev/modular-community`
      to `channels`.
- [ ] Delete `third_party/` and the `numojo` task. Drop `depends-on =
      ["numojo"]` and `-I build` from the other tasks and from
      `tests/run_all.sh`, `examples/run.sh` and `docs/*.py`.
- [ ] Fix anything the new NuMojo release changed relative to the patched
      0.12.0.
- [ ] Update the README installation notes, the Toolchain section of
      `AGENTS.md`, and `docs/publishing.md`.
- [ ] In `conda.recipe/recipe.yaml`, set `numojo: package` and
      `numojo_version`, then remove the `vendored` branches.

### 2. Update for Mojo 1.3.0

- [ ] In `pixi.toml`, raise the floors to the releases:
      `mojo = ">=1.3.0,<2"` and `max-core = ">=26.8.0,<27"` (check the
      MAX version that pairs with Mojo 1.3.0), and move `channels` to
      `https://conda.modular.com/max` (keep `max-nightly` if you still
      want nightlies).
- [ ] Run `pixi install`, `pixi run test`, and `pixi run examples`, and fix
      anything 1.3.0 broke. The property tests need a proptest checkout
      (or the proptest package, once it's published).
- [ ] In `conda.recipe/recipe.yaml`, set `mojo_version: "1.3.0"` and the
      matching `max_version`. Bump `version` too if you're releasing more
      than 0.1.0, and keep it in step with `version` in `pixi.toml`.
- [ ] Build and test the package locally against the stable channel:
      ```sh
      rattler-build build --recipe conda.recipe/recipe.yaml \
        -c conda-forge -c https://conda.modular.com/max \
        -c https://prefix.dev/modular-community
      ```
      The test should print `mojodec package OK` and `all tests passed!`.
- [ ] Commit and push. Note the full 40-character hash: `git rev-parse HEAD`.
- [ ] Optional: tag it, `git tag v0.1.0 && git push --tags`.

### 3. Add the recipe to your modular-community fork

- [ ] Sync the fork with upstream and create a branch, e.g. `add-mojodec`.
- [ ] Create `recipes/mojodec/` with copies of
      `conda.recipe/recipe.yaml` and `conda.recipe/test_package.mojo`.
- [ ] In `recipes/mojodec/recipe.yaml`:
  - [ ] Replace the `source` entry (`path: ..`) with the pushed commit:
        ```yaml
        source:
          - git: https://github.com/bpr/MojoDEC.git
            rev: <full 40-character commit hash>
        ```
  - [ ] Delete the header comment, which refers to files in the MojoDEC repo.
- [ ] Build it with the channels the CI uses:
      ```sh
      rattler-build build --recipe recipes/mojodec/recipe.yaml \
        -c conda-forge -c https://conda.modular.com/max \
        -c https://prefix.dev/modular-community
      ```

### 4. Open the pull request

- [ ] Commit `recipes/mojodec/` and open a PR against
      `modular/modular-community`.
- [ ] Watch the CI. It also builds on linux-aarch64 and osx-arm64, which
      have only been tested here on linux-64. Check that `matplotlib-base`
      and `numpy` resolve there too.
- [ ] CodeQL: the package is all Mojo (`mojodec.vis` calls matplotlib
      through Python interop but ships no Python files), so it shouldn't
      be needed. If reviewers count the `docs/*.py` build scripts, set up
      CodeQL for the repository.

### 5. After it's merged

- [ ] In `README.md`, replace the installation notes that build from a
      clone with install instructions: add
      `https://repo.prefix.dev/modular-community` to `channels` in
      `pixi.toml`, then `pixi add mojodec`.

## Later releases

- New MojoDEC version: bump `version` (recipe and `pixi.toml`), set
  `build.number: 0`, push, then update `rev` (and `version`) in
  `recipes/mojodec/recipe.yaml` and open a new PR.
- New Mojo release, same MojoDEC: change `mojo_version` and `max_version`
  (and `numojo_version` if NuMojo needed a new build), increment
  `build.number`, and open a PR. The compiler and MAX are pinned exactly
  because a `.mojoc` only loads in the compiler that built it.

See `docs/publishing.md` for how the recipe works.
