# MojoDEC against PyDEC, dexterior, and formoniq

Times the work every DEC library does before it solves anything: building
the complex, its geometry, and the Whitney mass matrices. Each library
reads the same meshes and prints a check value next to each timing, so a
fast library that gets a different answer shows up as different, not as
faster.

```sh
pixi run bench-compare                       # all but PyDEC, best of 3
pixi run bench-compare --libs mojodec,pydec,dexterior,formoniq  # with PyDEC (~12 min)
pixi run bench-compare --meshes tri_100,tet_10 --libs mojodec,formoniq
pixi run bench-compare --report build/compare/results/<time>.jsonl
```

The first run writes the meshes, builds MojoDEC's bench (release, `-D
ASSERT=none`) and the Rust crate, and clones PyDEC. Results go to
`build/compare/results/<time>.jsonl` (every measurement, one JSON object
per line, after a line with the machine and versions) and `<time>.md`
(the tables). The guide's
[performance page](../../docs/guide/performance.md) has the latest
results and what they say.

## Libraries and versions

| Library | Version | Built with |
|---|---|---|
| MojoDEC | this checkout | `mojo build -D ASSERT=none` |
| [PyDEC](https://github.com/hirani/pydec) | `c181202`, plus `pydec-numpy2.patch` | CPython 3.13, NumPy 2, SciPy, from the `bench` pixi environment |
| [dexterior](https://codeberg.org/molentum/dexterior) | `9b0f379` | `cargo build --release`, `target-cpu=native`, one codegen unit |
| [formoniq](https://github.com/luiswirth/formoniq) | `710139e` | the same |

`pydec-numpy2.patch` is one line: for `k = 1`, `whitney_innerproduct`
stores a 1×1 array as a determinant, which NumPy 2 refuses to convert to
a scalar. To use another PyDEC checkout, set `PYDEC_PATH` to the
directory that contains its `pydec` package (in PyDEC's repository, the
top-level `pydec/` directory).

## The meshes

`meshes.mojo` writes Kuhn grids of the unit square and cube with interior
vertices moved by up to 5% of the cell size (fixed pseudo-random
sequence; every element checked positively oriented):

| Mesh | Vertices | Elements |
|---|---:|---:|
| `tri_100`, `tri_200`, `tri_400` | 10k, 40k, 161k | 20k, 80k, 320k triangles |
| `tet_10`, `tet_20`, `tet_40` | 1.3k, 9.3k, 69k | 6k, 48k, 384k tetrahedra |
| `tri_400_shuffled`, `tet_40_shuffled` | as `tri_400`, `tet_40` | as `tri_400`, `tet_40` |

The jitter matters: an exact Kuhn grid has dual volumes that are exactly
zero and circumcenters that coincide, special cases a real mesh never
hits. The jittered meshes are not well-centered, so some signed dual
volumes are negative.

A Kuhn grid numbers its vertices in spatial order, which every loop over
vertices or faces benefits from (neighbouring numbers, neighbouring
memory), and which MojoDEC's parallel dual-volume walk relies on: it
splits the work by blocks of vertex numbers, and walks each top simplex
once per block it touches. The `_shuffled` meshes are the same meshes
with their vertex numbers randomly permuted (element order kept), as a
mesh file with an incoherent numbering would be. Their checks differ
from the unshuffled ones' only in the last bits (sums in another order).

## Phases

| Phase | MojoDEC | PyDEC | dexterior | formoniq |
|---|---|---|---|---|
| `topology`: every face, `d` and boundary per level | `SimplicialComplex(V, E, compute_geometry=False)` | `simplicial_complex((V, E))` | — | `Complex::from_cells` (sorting each element included) |
| `geometry`: circumcenters, primal and dual volumes, diagonal stars | `sc.compute_geometry()` | `sc.construct_hodge()` | — | squared edge lengths only |
| `build`: both | `SimplicialComplex(V, E)` | both of the above | `SimplicialMesh::new` | — |
| `whitney_k`: Whitney mass matrix of k-forms | `whitney_innerproduct(sc, k)` | `whitney_innerproduct(sc, k)` | — | `WhitneyComplex::mass(k)` |

dexterior builds topology and geometry in one call, so it has only a
`build` time; its checks are taken afterwards. formoniq computes metric
quantities per element from edge lengths when it assembles, and has no
circumcentric dual, so its `geometry` time is not comparable with the
others' and it has no `build` row.

## Checks

- `topology`: the number of simplices of each dimension.
- `geometry`: the sum of each primal star's diagonal, `sum(dual / primal)`
  per dimension (signed, so negative dual volumes count).
- `whitney_k`: the Frobenius norm of the assembled matrix (independent of
  simplex numbering and orientation; duplicates summed).

The tables mark a check that differs from MojoDEC's by more than 1e-9
relative with `†`, and one containing NaN with `NaN`.

## Threads

MojoDEC (`MOJODEC_NUM_THREADS`, see `mojodec/core/parallel.mojo`) and
formoniq (`RAYON_NUM_THREADS`) run once with one thread and once on all
cores. PyDEC and dexterior are single-threaded and run once, with
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for NumPy. MojoDEC's
loops go parallel only above fixed sizes (50k simplices for circumcenters
and volumes, 4096 top simplices for Whitney), so its two columns match on
small meshes.

## What is not compared

- Solvers (`cg`, sparse LU, eigensolvers): those timings mostly compare
  SciPy, faer, and nalgebra, not DEC code.
- Time stepping, wedge products, I/O.
- Memory use.
