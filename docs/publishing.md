# Publishing MojoDEC

MojoDEC is packaged as a conda package with
[rattler-build](https://rattler-build.prefix.dev/). The recipe is
`conda.recipe/recipe.yaml`. The package installs
`lib/mojo/mojodec.mojoc`, which Mojo finds without any `-I` flag.

## Dependencies

- `mojo-compiler` and `max-core`, pinned exactly. A `.mojoc` only loads in
  the compiler that built it, and MojoDEC imports `max.algorithm` (in
  `core/parallel.mojo`), so `max-core` must be the release that goes with
  the compiler (e.g. Mojo `1.3.0.dev2026110305` with MAX
  `26.8.0.dev2026110305`).
- `numojo`. MojoDEC's API takes and returns NuMojo arrays.
- `python`, `numpy` and `matplotlib-base`, for `mojodec.vis`, which plots
  through Python interop. Code that doesn't plot never loads Python.

Both `max-core` and `numojo` are listed under `build` as well as `host`
because the compiler only searches its own environment for imported
`.mojoc` files.

### NuMojo

The published `numojo` 0.12.0 package requires Mojo < 1.3, so MojoDEC builds
against the patched copy in `third_party/numojo` (see
`third_party/README.md`). The recipe's `numojo` context variable chooses:

- `vendored` (the default in `conda.recipe/`): precompile
  `third_party/numojo` and install its `numojo.mojoc` in the MojoDEC
  package too. This is for local builds only. It would clash with an
  installed `numojo` package and isn't acceptable on modular-community.
- `package`: depend on the `numojo` conda package at `numojo_version`.
  This is what gets submitted, once NuMojo publishes a release for Mojo 1.3.
  `numojo_version` is a placeholder until then.

## Building locally

```sh
rattler-build build --recipe conda.recipe/recipe.yaml \
  -c conda-forge -c https://conda.modular.com/max-nightly
```

This builds the working tree, runs `conda.recipe/test_package.mojo` against
the installed package, and writes `output/linux-64/mojodec-<version>-*.conda`.

## The modular-community channel

Community packages live in
[modular/modular-community](https://github.com/modular/modular-community).
Its CI builds every recipe with the channels `conda-forge`,
`https://conda.modular.com/max` and `https://prefix.dev/modular-community`.
It does **not** use `max-nightly`. A recipe that needs a nightly compiler
can't resolve its dependencies there. The CI records the failure, and a
recipe still failing after four weeks is deleted automatically.

So MojoDEC can be submitted once both of these hold:

1. `mojo-compiler 1.3.0` (and its `max-core`) is on the stable `max` channel.
2. A `numojo` release supporting Mojo 1.3 is on modular-community.

proptest, used by `tests/test_properties*.mojo`, is a test-only dependency
and isn't needed by the package.

### Submitting

1. Push the release commit to https://github.com/bpr/MojoDEC.
2. In a fork of modular-community, copy `conda.recipe/recipe.yaml` and
   `conda.recipe/test_package.mojo` to `recipes/mojodec/`.
3. In `recipes/mojodec/recipe.yaml`:
   - replace the `source` entry with
     ```yaml
     source:
       - git: https://github.com/bpr/MojoDEC.git
         rev: <full 40-character commit SHA>
     ```
   - set `numojo: package`, and `numojo_version` to the NuMojo release;
   - set `mojo_version` and `max_version` to the stable releases, e.g.
     `"1.3.0"` and `"26.8.0"`.
4. Build it there with the same channels the CI uses:
   ```sh
   rattler-build build --recipe recipes/mojodec/recipe.yaml \
     -c conda-forge -c https://conda.modular.com/max \
     -c https://prefix.dev/modular-community
   ```
5. Open a pull request.

MojoDEC's package is all Mojo. The Python scripts in `docs/` are build
tools that aren't packaged, so the CodeQL requirement for packages that
contain Python or other languages shouldn't apply.

### Releasing a new version

Bump `version` in `conda.recipe/recipe.yaml` and `pixi.toml`, set
`build.number` back to 0, and update `rev` in the modular-community recipe.
To rebuild the same version for a new Mojo release, change `mojo_version`
and `max_version` and increment `build.number`.
