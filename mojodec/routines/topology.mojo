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

`star_of`, `closure_of`, and `link_of` work on any complex through its
chain complex, the boundary matrices `B_k` that `chain_complex()` returns.
A set of cells is a `List[List[Int]]` with one entry per dimension
`k = 0, ..., n`, holding indices of k-cells in increasing order; a shorter
list leaves the higher dimensions empty, so `[[v]]` is the vertex `v`.

`boundary_of` and `interior_of` without a set of cells mean the whole
complex: its top-dimensional cells and their faces.

`SimplicialComplex`, `AbstractSimplicialComplex`, `RipsComplex`, and
`RegularCubeComplex` have `star`, `closure`, `link`, `boundary`, and
`interior` methods that call these, and `boundary_cells` and
`interior_cells` for the whole complex; `SimplicialComplex`'s also take a
`Simplex`. (`SimplicialComplex.boundary()` with no arguments is PyDEC's:
the boundary faces as a list of simplices.)

```mojo
var ring = sc.link(Simplex([v]))       # [[neighbors], [ring edges], []]
var same = link_of(sc.chain_complex(), [[v]])
```
"""

from mojodec.core.sparse import CSRMatrix


def star_of(
    chain_complex: List[CSRMatrix], cells: List[List[Int]]
) raises -> List[List[Int]]:
    """Every cell that has a cell of `cells` as a face, `cells` included.

    Args:
        chain_complex: The boundary operators `B_0, ..., B_n`, as
            `chain_complex()` returns them.
        cells: Indices of cells by dimension: `cells[k]` lists k-cells,
            and may stop short of the top dimension.

    Returns:
        Indices per dimension: entry k lists k-cells, sorted.

    Raises:
        If `cells` has more dimensions than `chain_complex` or an index is
        out of range.
    """
    var marked = _marked(chain_complex, cells)
    for k in range(len(chain_complex) - 1):
        # row i of B_{k+1} lists the (k+1)-cells that have k-cell i as a face
        ref B = chain_complex[k + 1]
        for i in range(len(marked[k])):
            if marked[k][i]:
                for p in range(B.indptr[i], B.indptr[i + 1]):
                    if B.data[p] != 0:
                        marked[k + 1][B.indices[p]] = True
    return _indices(marked)


def closure_of(
    chain_complex: List[CSRMatrix], cells: List[List[Int]]
) raises -> List[List[Int]]:
    """Every face of every cell of `cells`: the subcomplex they generate.

    Args:
        chain_complex: The boundary operators `B_0, ..., B_n`, as
            `chain_complex()` returns them.
        cells: Indices of cells by dimension: `cells[k]` lists k-cells,
            and may stop short of the top dimension.

    Returns:
        Indices per dimension: entry k lists k-cells, sorted.

    Raises:
        If `cells` has more dimensions than `chain_complex` or an index is
        out of range.
    """
    var marked = _marked(chain_complex, cells)
    for k in range(len(chain_complex) - 1, 0, -1):
        # row j of B_k^T lists the (k-1)-faces of k-cell j
        var Bt = chain_complex[k].transpose()
        for j in range(len(marked[k])):
            if marked[k][j]:
                for p in range(Bt.indptr[j], Bt.indptr[j + 1]):
                    if Bt.data[p] != 0:
                        marked[k - 1][Bt.indices[p]] = True
    return _indices(marked)


def link_of(
    chain_complex: List[CSRMatrix], cells: List[List[Int]]
) raises -> List[List[Int]]:
    """`Cl(St(S)) - St(Cl(S))`: the faces around `cells` that do not touch them.

    Args:
        chain_complex: The boundary operators `B_0, ..., B_n`, as
            `chain_complex()` returns them.
        cells: Indices of cells by dimension: `cells[k]` lists k-cells,
            and may stop short of the top dimension.

    Returns:
        Indices per dimension: entry k lists k-cells, sorted.

    Raises:
        If `cells` has more dimensions than `chain_complex` or an index is
        out of range.
    """
    return _minus(
        closure_of(chain_complex, star_of(chain_complex, cells)),
        star_of(chain_complex, closure_of(chain_complex, cells)),
    )


def boundary_of(
    chain_complex: List[CSRMatrix], cells: List[List[Int]]
) raises -> List[List[Int]]:
    """The closure of the (k-1)-faces that bound exactly one k-cell of `cells`.

    k is the highest dimension in `cells`, and every other cell must be a
    face of one of those k-cells. An empty set has an empty boundary, and
    so does a set of vertices.

    Args:
        chain_complex: The boundary operators `B_0, ..., B_n`, as
            `chain_complex()` returns them.
        cells: Indices of cells by dimension: `cells[k]` lists k-cells,
            and may stop short of the top dimension.

    Returns:
        Indices per dimension: entry k lists k-cells, sorted.

    Raises:
        If `cells` has more dimensions than `chain_complex`, an index is out
        of range, or `cells` is not pure.
    """
    var marked = _marked(chain_complex, cells)
    var k = len(marked) - 1
    while k >= 0 and True not in marked[k]:
        k -= 1
    var faces = List[List[Int]](length=len(marked), fill=List[Int]())
    if k < 0:
        return faces^
    var top = faces.copy()
    for i in range(len(marked[k])):
        if marked[k][i]:
            top[k].append(i)
    var spanned = closure_of(chain_complex, top)
    for d in range(k):
        for i in range(len(marked[d])):
            if marked[d][i] and i not in spanned[d]:
                raise Error(
                    String(
                        "boundary: the cells are not pure: ",
                        d,
                        "-cell ",
                        i,
                        " is not a face of a ",
                        k,
                        "-cell",
                    )
                )
    if k == 0:
        return faces^
    # row j of B_k lists the k-cells that have (k-1)-cell j as a face
    ref B = chain_complex[k]
    for j in range(B.nrows):
        var count = 0
        for p in range(B.indptr[j], B.indptr[j + 1]):
            if B.data[p] != 0 and marked[k][B.indices[p]]:
                count += 1
        if count == 1:
            faces[k - 1].append(j)
    return closure_of(chain_complex, faces)


def boundary_of(chain_complex: List[CSRMatrix]) raises -> List[List[Int]]:
    """The boundary of the whole complex, a closed subcomplex.

    For a mesh of a manifold with boundary, its boundary mesh: the faces
    of the boundary (n-1)-cells.

    Args:
        chain_complex: The boundary operators `B_0, ..., B_n`, as
            `chain_complex()` returns them.

    Returns:
        Indices per dimension: entry k lists k-cells, sorted.

    Raises:
        Never in practice; the whole complex is pure.
    """
    return boundary_of(chain_complex, _top_cells(chain_complex))


def interior_of(
    chain_complex: List[CSRMatrix], cells: List[List[Int]]
) raises -> List[List[Int]]:
    """`Cl(S) - Bd(S)`: the cells of the closure not on the boundary.

    Args:
        chain_complex: The boundary operators `B_0, ..., B_n`, as
            `chain_complex()` returns them.
        cells: Indices of cells by dimension: `cells[k]` lists k-cells,
            and may stop short of the top dimension.

    Returns:
        Indices per dimension: entry k lists k-cells, sorted.

    Raises:
        If `cells` has more dimensions than `chain_complex`, an index is out
        of range, or `cells` is not pure.
    """
    var bd = boundary_of(chain_complex, cells)
    return _minus(closure_of(chain_complex, cells), bd)


def interior_of(chain_complex: List[CSRMatrix]) raises -> List[List[Int]]:
    """The cells of the whole complex not on its boundary.

    Args:
        chain_complex: The boundary operators `B_0, ..., B_n`, as
            `chain_complex()` returns them.

    Returns:
        Indices per dimension: entry k lists k-cells, sorted.

    Raises:
        Never in practice; the whole complex is pure.
    """
    return interior_of(chain_complex, _top_cells(chain_complex))


def _marked(
    chain_complex: List[CSRMatrix], cells: List[List[Int]]
) raises -> List[List[Bool]]:
    """One flag per cell of each dimension, set for the cells in `cells`."""
    if len(cells) > len(chain_complex):
        raise Error(
            String(
                "cells: ",
                len(cells),
                " dimensions given, but the complex has ",
                len(chain_complex),
            )
        )
    var marked = List[List[Bool]]()
    for k in range(len(chain_complex)):
        marked.append(List[Bool](length=chain_complex[k].ncols, fill=False))
    for k in range(len(cells)):
        for i in cells[k]:
            if i < 0 or i >= len(marked[k]):
                raise Error(String("cells: no ", k, "-cell with index ", i))
            marked[k][i] = True
    return marked^


def _top_cells(chain_complex: List[CSRMatrix]) -> List[List[Int]]:
    """Every cell of the highest dimension."""
    var n = len(chain_complex)
    var cells = List[List[Int]](length=n, fill=List[Int]())
    if n > 0:
        for i in range(chain_complex[n - 1].ncols):
            cells[n - 1].append(i)
    return cells^


def _minus(a: List[List[Int]], b: List[List[Int]]) -> List[List[Int]]:
    """The cells of `a` not in `b`; both sorted per dimension."""
    var out = List[List[Int]]()
    for k in range(len(a)):
        var level = List[Int]()
        var j = 0
        for i in a[k]:
            while j < len(b[k]) and b[k][j] < i:
                j += 1
            if j == len(b[k]) or b[k][j] != i:
                level.append(i)
        out.append(level^)
    return out^


def _indices(marked: List[List[Bool]]) -> List[List[Int]]:
    var out = List[List[Int]]()
    for ref level in marked:
        var ids = List[Int]()
        for i in range(len(level)):
            if level[i]:
                ids.append(i)
        out.append(ids^)
    return out^
