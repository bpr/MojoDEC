# MojoDEC examples

Ports of PyDEC's six example programs, plus `Sphere`, a spectral check
taken from [Dxtr](https://gitlab.inria.fr/oali/dxtr), `Neighborhoods`
and `Neighborhoods3D`, tours of stars, links, and boundaries in 2D and 3D,
and `HeatConduction`, which solves a problem set up in Gmsh. Each one computes
something on a complex, prints a summary, writes its
results to text files or the terminal, and draws its figures with matplotlib. Eight of them
also write VTK files for ParaView.

## Running them

From the repository root, after `pixi install`:

```sh
pixi run examples                      # all ten, figures saved as PNG files
pixi run examples DarcyFlow            # just one
pixi run examples DarcyFlow --show     # ... and open its figures in windows
pixi run examples --show               # all ten, showing every figure
```

The examples are `Ranking`, `RipsComplex`, `HodgeDecomposition`,
`DarcyFlow`, `HeatConduction`, `ResonantCavity`, `Sphere`, `Neighborhoods`, `Neighborhoods3D`, and `MakeMesh`. Each runs in its own
directory and writes its output there. The task builds the vendored NuMojo
first, so the first run takes a minute longer.

To run one by hand, `cd` into its directory and give `mojo` the two import
paths. Arguments after the file go to the program:

```sh
pixi run numojo                        # once: builds build/numojo.mojoc
cd examples/DarcyFlow
pixi run mojo run -I ../.. -I ../../build driver.mojo
pixi run mojo run -I ../.. -I ../../build driver.mojo --show
```

(MakeMesh's program is `make_mesh.mojo`; the others are `driver.mojo`.)

## Looking at the figures

**PNG files.** By default the figures go to PNG files next to the program,
using matplotlib's Agg backend, so the examples also run over SSH, in CI,
or anywhere else without a display. Open them with any image viewer.

**Windows (`--show`).** With `--show`, the program still writes the PNG
files, then opens one matplotlib window per figure, which you can pan,
zoom, and save from. The program finishes when you close the last window.
This needs a display; the pixi environment provides Qt and Tk, and
matplotlib picks one. Choose a backend with `MPLBACKEND`:

```sh
MPLBACKEND=TkAgg pixi run examples ResonantCavity --show
```

Without a display, matplotlib falls back to Agg, warns that it cannot
show the figures, and the PNG files are still written.

## What each example produces

| Example | Computes | Text | Figures | VTK |
|---|---|---|---|---|
| Ranking | least-squares ranking of a graph's vertices from edge data | printed | — | — |
| RipsComplex | Rips complex of 300 random points and a harmonic 1-chain, which outlines the holes in coverage | `harmonic.txt` | `nodes.png`, `complex.png`, `harmonic.png` (edge width ∝ \|h\|) | `harmonic.vtu` |
| HodgeDecomposition | four harmonic 1-cochains on a square with four holes | `harmonic.txt` | `harmonic_0.png` … `harmonic_3.png` | `harmonic.vtu` |
| DarcyFlow | Darcy flow through a square: pressures and fluxes | `pressures.txt` | `pressures.png` (vs. exact), `flux.png` (velocity field) | `darcy.vtu` |
| HeatConduction | steady heat flow through a plate of two materials, meshed by Gmsh: its physical groups give the hot, cold, and insulated edges and the material regions | `temperature.txt` | `temperature.png` (over the plate), `profile.png` (vs. exact, along x) | `heat.vtu` |
| ResonantCavity | eigenvalues of the curl-curl operator on a square cavity, and one eigenmode | `eigenvalues.txt`, `mode3.txt` | `eigenvalues.png`, `mode3.png` | `mode3.vtu` |
| Sphere | Hodge-Laplacian spectra of 0-, 1-, and 2-forms on an icosphere vs. the spherical harmonics' n(n+1); Gauss-Bonnet and curvature checks | `spectrum.txt` | `spectrum.png` | `sphere.vtu` |
| Neighborhoods | stars, links, and boundaries of vertices, an edge, growing rings of triangles, and the whole mesh on a grid of the square, checking that links are polygons or paths and that Bd(Cl(St(v))) = Lk(v) | printed | `neighborhoods.png` | `neighborhoods.vtu` |
| Neighborhoods3D | the same on PyDEC's tetrahedral solid torus: vertex links are spheres or disks, and the growing rings are balls until they wrap around (pinched once) into the solid torus, whose boundary is a torus | printed | `neighborhoods3d.png` (vertex links, Euler characteristics of the rings) | `neighborhoods3d.vtu`, `links.vtu` |
| MakeMesh | writes a mesh in PyDEC's format and reads it back | `square_8.*` | `square_8.png` | — |

The vector fields in the figures and VTK files are 1-cochains turned into
vectors with Whitney interpolation (`simplex_quivers`), one vector per
triangle at its barycenter.

## Viewing the VTK files in ParaView

The `.vtu` files open in [ParaView](https://www.paraview.org) (or VisIt):
**File → Open**, choose the file, then **Apply**. The fields are cell
data, one value or vector per cell, except in `sphere.vtu`, whose
curvatures and eigenfunction are point data, one per vertex.

**Color by a field.** In the toolbar's coloring menu, pick the array:
`pressure` in `darcy.vtu`, `temperature` or `conductivity` in `heat.vtu`, `h` in RipsComplex's `harmonic.vtu`, or `ring`
in `neighborhoods.vtu` and `neighborhoods3d.vtu` (each cell's ring around
the starting vertex). In `neighborhoods3d.vtu`, a **Threshold** on `ring`
shows the region growing around the torus; in `links.vtu`, color by `part`
(0: the sphere around an interior vertex, 1: a boundary vertex's disk, 2:
the torus surface). For
vectors, ParaView colors by magnitude.

**Arrows for a vector field** (`velocity` in `darcy.vtu`, `mode3`, or
`harmonic_0` … `harmonic_3`). Glyphs are placed at points, so first move the
cell data to the triangle centers:

1. Select the file in the Pipeline Browser, then **Filters → Alphabetical →
   Cell Centers**, and **Apply**.
2. With **CellCenters1** selected, add **Filters → Common → Glyph**. Set
   **Glyph Type** to *Arrow*, **Orientation Array** to the field (for
   example `velocity`), and **Scale Array** to the same field or to *No
   scale array*. Adjust **Scale Factor** until the arrows are readable,
   then **Apply**.
3. Turn the original file's visibility back on (the eye icon) to see the
   mesh under the arrows; *Surface With Edges* shows the triangles.

**The sphere.** In `sphere.vtu`, color by `gaussian_curvature` or
`eigenfunction_l1` (a spherical harmonic of degree 1: one hemisphere
positive, the other negative). The `mean_curvature_normal` vectors are
already at the vertices, so a **Glyph** filter applies directly, without
Cell Centers; they point outward with length about 1. Run
`pixi run examples Sphere --subdivisions 3` for a finer mesh (642
vertices, about a minute) and eigenvalues closer to n(n+1).

**RipsComplex's edges.** `harmonic.vtu` holds the 2831 edges of the
complex with the harmonic chain `h` on them. Color by `h` with a diverging
color map (the harmonic chain is signed), and raise **Line Width** in the
Properties panel's advanced settings (the gear icon). A **Threshold**
filter on `h` keeps only the edges where it is large, which outlines the
holes.

RipsComplex writes its VTK file in the compact `"appended"` binary
format; the others write ASCII, which you can read in a text editor. Both
open the same way.

## Plotting from your own program

`mojodec.vis` and `write_vtu` work the same way outside the examples.
This program plots the 1-cochain `dx` on a 4×4 grid. Whitney
interpolation reproduces the constant field `(1, 0)` exactly, so every
arrow should point right with the same length.

```mojo
import numojo as nm
from numojo.prelude import *

from mojodec import SimplicialComplex, VTKField, simplicial_grid_2d, write_vtu
from mojodec.core.utils import fget
from mojodec.vis import pyplot, show_requested, simplex_quivers, triplot


def main() raises:
    # a 4x4 grid of the unit square and the 1-cochain dx (edge x-extents)
    var vt = simplicial_grid_2d(4)
    var sc = SimplicialComplex(vt[0], vt[1])
    var dx = sc.get_cochain(1)
    for e in range(sc[1].num_simplices):
        var a = sc[1].index_to_simplex[e][0]
        var b = sc[1].index_to_simplex[e][1]
        dx[e] = fget(sc.vertices(), b, 0) - fget(sc.vertices(), a, 0)

    # sample it at the triangle barycenters and plot it over the mesh
    var bd = simplex_quivers(sc, dx.v)
    var b = bd[0].to_numpy()
    var v = bd[1].to_numpy()
    var show = show_requested()
    var plt = pyplot(interactive=show)
    _ = triplot(sc.vertices(), sc.simplices())
    plt.quiver(b.T[0], b.T[1], v.T[0], v.T[1])
    plt.gca().set_aspect("equal")
    plt.savefig("dx.png")

    # the same field for ParaView
    write_vtu(
        "dx.vtu",
        sc.vertices(),
        sc.simplices(),
        cell_data=[VTKField("dx", bd[1])],
        format="appended",
    )
    print("wrote dx.png and dx.vtu")
    if show:
        plt.show()
```

Save it as `plot_form.mojo` anywhere and run it with the MojoDEC checkout
on the import path, through MojoDEC's pixi environment:

```sh
pixi run --manifest-path /path/to/MojoDEC/pixi.toml \
    mojo run -I /path/to/MojoDEC -I /path/to/MojoDEC/build plot_form.mojo --show
```

`pyplot(interactive=...)` returns `matplotlib.pyplot`, set up with the
Agg backend unless `interactive` is true; `show_requested()` is true when
the program was given `--show`. Anything else in matplotlib is available
through the returned object, as in Python.
