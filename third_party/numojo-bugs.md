# NuMojo bugs found by MojoDEC

Bugs in NuMojo v0.12.0 (commit `c9b3f07`, the vendored copy in
`numojo/`) that MojoDEC hit or worked around, for reports to the NuMojo
maintainers. Each repro is a complete program; run it from the MojoDEC
root with `pixi run mojo run -I build repro.mojo` (after `pixi run
numojo`). All were confirmed on Mojo `1.3.0.dev2026110305`; none is caused
by the Mojo 1.3 patch, which changes only closures, imports, and
`DeviceContext`.

## 1. `fromstring` drops the `e` of an exponent

`"1e4"` reads as 14 and `"1e+2"` as 12; `"2.5e-3"` raises
`String is not convertible to float: '2.5-3'`.

```mojo
import numojo as nm
from numojo.prelude import *

def main() raises:
    var a = nm.fromstring[f64]("[1e4, 1e+2]")
    print(a.unsafe_get(0), a.unsafe_get(1))   # 14.0 12.0; want 10000.0 100.0
    _ = nm.fromstring[f64]("[2.5e-3]")        # raises
```

MojoDEC writes test literals without exponents. Minor, same function: the
doc string's example output shows `[0.1, -2.3, 41.5, 19.29145, -199]`
printing without its minus signs, which the code does not actually do.

## 2. `linalg.solve` does not pivot

`solve` (through `lu_decomposition`) factors without row exchanges, so a
nonsingular matrix with a zero leading pivot gives NaN instead of the
solution. NumPy's `solve` (LAPACK `getrf`) pivots.

```mojo
import numojo as nm
from numojo.prelude import *

def main() raises:
    var A = nm.fromstring[f64]("[[0, 1], [1, 0]]")
    var y = nm.fromstring[f64]("[[2], [3]]")
    var x = nm.linalg.solve(A, y)
    print(x.unsafe_get(0), x.unsafe_get(1))   # nan nan; want 3.0 2.0
```

Without pivoting, small pivots also make the result inaccurate when they
are not exactly zero. MojoDEC uses its own pivoted LU
(`mojodec/routines/dense.mojo`).

## 3. A new axis is dropped when every dimension gets an integer

NumPy's `a[np.newaxis, 0, 0, 0]` on a 3-D array has shape `(1,)`; NuMojo
returns a 0-D array.

```mojo
import numojo as nm
from numojo.prelude import *

def main() raises:
    var a = nm.zeros[DType.int32](Shape(3, 4, 2))
    print(a[nm.newaxis, 0, 0, 0].ndim)       # 0; want 1
```

## 4. An ellipsis that stands for no dimensions raises

NumPy accepts `a[..., 0, 0, 0]` on a 3-D array (the ellipsis matches zero
dimensions) and gives a 0-D result. NuMojo counts the ellipsis as an index:

```mojo
import numojo as nm
from numojo.prelude import *

def main() raises:
    var a = nm.zeros[DType.int32](Shape(3, 4, 2))
    _ = a[..., 0, 0, 0]
    # IndexError: Too many indices: 4 consume array dimensions but array
    # only has 3. [at NDArray._getitem_from_index_list]
```

Bugs 3 and 4 were found by the NuMojo index property test in proptest
(`../proptest/extras/tests/test_numojo.mojo`), which `assume`s them away
with a comment.

## 5. `fmax`/`fmin` and signed zeros (from NuMojo's own tests)

NuMojo's own test `test_fmax_fmin_against_numpy` fails on signed zeros,
with both the vendored copy and the official 0.12.0 package on Mojo 1.2. Not investigated further here; the failing test is
the repro.

## 6. Test suite: `test_linalg.mojo` does not compile

NuMojo's `test_linalg.mojo` calls `nm.math.linalg`, which does not
exist in v0.12.0 (the functions are under `nm.linalg`).

## Not bugs, but worth mentioning

- `linalg.eigh` uses cyclic Jacobi sweeps: correct, but much slower than
  Householder tridiagonalization plus QL for a few hundred unknowns.
  MojoDEC uses its own (`dense.eigh`).

## Not NuMojo (Mojo standard library)

These look like NuMojo problems from MojoDEC's side but come from Mojo
itself, so they belong in a Mojo report instead:

- `String(Float64)` is not round-trip for about one in five integral
  doubles between ~1.8e16 and 1e23 (one digit too few), and for some
  `Float32` values.
- `x ** 0.5` is off by up to ~1.1e-12 relative to `sqrt(x)`, for `Float64`
  scalars and `NDArray`s alike (NuMojo's `**` inherits it).
