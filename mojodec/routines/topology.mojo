# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Neighborhoods in a complex (`mojodec.routines.topology`).
=========================================================
The star, closure, link, boundary, and interior of a set of cells; not in
PyDEC.

- The *star* `St(S)` is every cell that has a cell of `S` as a face,
  including the cells of `S` themselves. It is usually not a subcomplex.
- The *closure* `Cl(S)` is every face of every cell of `S`: the smallest
  subcomplex containing `S`.
- The *link* `Lk(S) = Cl(St(S)) - St(Cl(S))` is the boundary of the
  neighborhood: the faces of the cells around `S` that do not touch `S`.
  The link of a vertex inside a triangulated surface is the ring of edges
  and vertices around it; inside a tetrahedral mesh, a triangulated
  sphere.
- The *boundary* `Bd(S)` of a pure set, whose cells are all faces of its
  k-cells (k its highest dimension), is the closure of the (k-1)-faces
  that bound exactly one k-cell of `S`. A set that is not pure raises.
- The *interior* `Int(S) = Cl(S) - Bd(S)`.

The functions work on any `CellComplex` (`SimplicialComplex`,
`RegularCubeComplex`, `AbstractSimplicialComplex`, `RipsComplex`, or a
`ChainComplex`) and take and return a `CellSet`, one bitset of cells per
dimension. Each step walks only the incidence rows of cells in the set,
so a query costs about the size of its answer, not of the complex.

`boundary_of` and `interior_of` without a set of cells mean the whole
complex: its top-dimensional cells and their faces.

The complexes have `star`, `closure`, `link`, `boundary`, and `interior`
methods that call these, and `boundary_cells` and `interior_cells` for
the whole complex; `SimplicialComplex`'s also take a `Simplex`.
(`SimplicialComplex.boundary()` with no arguments is PyDEC's: the boundary
faces as a list of simplices.)

```mojo
var ring = sc.link(Simplex([v]))       # vertices and edges around v
var same = link_of(sc, CellSet.from_indices(sc, 0, [v]))
```
"""

from mojodec.core.cell_set import CellComplex, CellSet


def star_of[C: CellComplex](cx: C, cells: CellSet) raises -> CellSet:
    """Every cell that has a cell of `cells` as a face, `cells` included.

    Parameters:
        C: The complex type.

    Args:
        cx: The complex.
        cells: Cells of `cx`.

    Returns:
        The star.

    Raises:
        If `cells` does not belong to `cx`.
    """
    cells.check_complex(cx)
    var out = cells.copy()
    for k in range(len(out.levels) - 1):
        if not out.levels[k].is_empty():
            out.levels[k + 1] = out.levels[k + 1] | cx.cofaces_of(out.levels[k])
    return out^


def closure_of[C: CellComplex](cx: C, cells: CellSet) raises -> CellSet:
    """Every face of every cell of `cells`: the subcomplex they generate.

    Parameters:
        C: The complex type.

    Args:
        cx: The complex.
        cells: Cells of `cx`.

    Returns:
        The closure.

    Raises:
        If `cells` does not belong to `cx`.
    """
    cells.check_complex(cx)
    var out = cells.copy()
    for k in range(len(out.levels) - 1, 0, -1):
        if not out.levels[k].is_empty():
            out.levels[k - 1] = out.levels[k - 1] | cx.faces_of(out.levels[k])
    return out^


def link_of[C: CellComplex](cx: C, cells: CellSet) raises -> CellSet:
    """`Cl(St(S)) - St(Cl(S))`: the faces around `cells` that do not touch them.

    Parameters:
        C: The complex type.

    Args:
        cx: The complex.
        cells: Cells of `cx`.

    Returns:
        The link.

    Raises:
        If `cells` does not belong to `cx`.
    """
    return closure_of(cx, star_of(cx, cells)) - star_of(
        cx, closure_of(cx, cells)
    )


def boundary_of[C: CellComplex](cx: C, cells: CellSet) raises -> CellSet:
    """The closure of the (k-1)-faces that bound exactly one k-cell of `cells`.

    k is the highest dimension in `cells`, and every other cell must be a
    face of one of those k-cells. An empty set has an empty boundary, and
    so does a set of vertices.

    Parameters:
        C: The complex type.

    Args:
        cx: The complex.
        cells: Cells of `cx`.

    Returns:
        The boundary, a closed set.

    Raises:
        If `cells` does not belong to `cx` or is not pure.
    """
    cells.check_complex(cx)
    var k = cells.dimension()
    var faces = CellSet.empty(cx)
    if k < 0:
        return faces^
    var top = CellSet.from_subset(cx, cells.levels[k])
    var spanned = closure_of(cx, top)
    for d in range(k):
        var stray = cells.levels[d] - spanned.levels[d]
        if not stray.is_empty():
            raise Error(
                String(
                    "boundary: the cells are not pure: ",
                    d,
                    "-cell ",
                    stray.indices()[0],
                    " is not a face of a ",
                    k,
                    "-cell",
                )
            )
    if k == 0:
        return faces^
    faces.levels[k - 1] = cx.boundary_faces_of(cells.levels[k])
    return closure_of(cx, faces)


def boundary_of[C: CellComplex](cx: C) raises -> CellSet:
    """The boundary of the whole complex, a closed set.

    For a mesh of a manifold with boundary, its boundary mesh: the faces
    of the boundary (n-1)-cells.

    Parameters:
        C: The complex type.

    Args:
        cx: The complex.

    Returns:
        The boundary.

    Raises:
        Never in practice; the top cells form a pure set.
    """
    return boundary_of(cx, CellSet.top_cells(cx))


def interior_of[C: CellComplex](cx: C, cells: CellSet) raises -> CellSet:
    """`Cl(S) - Bd(S)`: the cells of the closure not on the boundary.

    Parameters:
        C: The complex type.

    Args:
        cx: The complex.
        cells: Cells of `cx`.

    Returns:
        The interior.

    Raises:
        If `cells` does not belong to `cx` or is not pure.
    """
    var bd = boundary_of(cx, cells)
    return closure_of(cx, cells) - bd


def interior_of[C: CellComplex](cx: C) raises -> CellSet:
    """The cells of the whole complex not on its boundary.

    Parameters:
        C: The complex type.

    Args:
        cx: The complex.

    Returns:
        The interior.

    Raises:
        Never in practice; the top cells form a pure set.
    """
    return interior_of(cx, CellSet.top_cells(cx))
