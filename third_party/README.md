# Vendored NuMojo

`numojo/` is [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo)
v0.12.0 (commit `c9b3f07`, Apache-2.0, see `numojo/LICENSE`), patched to
build with Mojo 1.3.

The vendored copy is needed because the published `numojo` 0.12.0 conda
package requires `mojo-compiler >=1.2.0,<1.3.0`, and NuMojo has no release
or branch for Mojo 1.3 yet. `pixi run numojo` precompiles it to
`build/numojo.mojoc`, and every other task builds against that file.

## The patch

`numojo-mojo-1.3.patch` is the complete diff against v0.12.0: 144 lines in
21 files.

- **Legacy closures** (135 sites). Mojo 1.3 removed the `capturing`
  function effect. In compile-time function parameters,
  `def[...](...) capturing -> T` is now `def[...](...) thin -> T`. On the
  nested kernel definitions passed to them, the effect is simply dropped.
  None of those kernels capture anything, so they are thin as written.
- **`default_accelerator`** moved from `std.sys.info` to `max.gpu.host`.
- **`DeviceContext`** is now parametric on its compilation target. Stored
  fields, constructor arguments, and returns use `DeviceContext[]`, which
  takes the default target.
- **`FileHandle.close()`** consumes the handle: `f.close()` becomes
  `f^.close()` in `routines/io/npy.mojo` and `routines/io/torch.mojo`.

## Verification

NuMojo's own v0.12.0 test suite (`tests/core`, `tests/routines`) was run
against the patched build on Mojo `1.3.0.dev2026110305`, with NumPy and
SciPy available for its reference comparisons. Every test passes except:

- `test_fmax_fmin_against_numpy`, which fails identically with the official
  0.12.0 package on Mojo 1.2 (an existing NuMojo bug in signed-zero
  handling);
- `test_linalg.mojo`, which does not compile on any Mojo version because it
  calls `nm.math.linalg`, a path that does not exist in v0.12.0.

`tests/core/test_device.mojo` imports `default_accelerator` from its old
location; with that import updated, all 43 of its tests pass.

When NuMojo publishes a Mojo 1.3 release, delete this directory and depend
on the package again.
