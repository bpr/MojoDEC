# Stars, links, and boundaries of cells in a triangle mesh.
#
# The star St(S) of a set of cells S is every cell that has a cell of S as
# a face; the link Lk(S) is the ring of faces around S that do not touch
# it; the boundary Bd(S) of a pure set is the closure of the faces that
# bound exactly one of its top cells. This program computes them on a grid
# of the unit square and checks what they should be:
#
# - the link of an interior vertex is a closed polygon (Euler
#   characteristic 0), and of a boundary vertex an open path (1);
# - the link of an interior edge is its two opposite vertices;
# - the closed star Cl(St(v)) of an interior vertex has the link as its
#   boundary and the open star as its interior;
# - growing a region by repeated closed stars, Cl(St(R)), gives disks, so
#   every boundary is a polygon, even once it runs along the square's edge;
# - the boundary of the whole mesh is a polygon, and its boundary is empty.
#
# Run from the repository root:  pixi run examples Neighborhoods [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figure in a window (see examples/README.md).
# Writes neighborhoods.png and neighborhoods.vtu (each triangle's ring
# number around the center vertex, for ParaView).

from std.python import Python, PythonObject

from mojodec import (
    Simplex,
    SimplicialComplex,
    VTKField,
    simplicial_grid_2d,
    write_vtu,
)
from mojodec.core.utils import real_vector, take_int_rows, take_rows
from mojodec.vis import lineplot, pyplot, show_requested, triplot


def main() raises:
    var n = 8
    var vt = simplicial_grid_2d(n)
    var sc = SimplicialComplex(vt[0], vt[1])
    print(sc)

    # (n+1)^2 vertices numbered row by row, x fastest
    var center = (n // 2) * (n + 1) + n // 2
    var side = n // 2  # middle of the bottom edge
    var corner = n * (n + 1)  # top left
    var names: List[String] = ["interior", "boundary", "corner"]
    var vertices: List[Int] = [center, side, corner]

    print("\n  vertex        star (v/e/t)  link (v/e)  chi(link)")
    for i in range(3):
        var v = vertices[i]
        var st = sc.star(Simplex([v]))
        var lk = sc.link(Simplex([v]))
        print(
            "  "
            + _pad(names[i] + " " + String(v), 14)
            + _pad(String(len(st[0]), "/", len(st[1]), "/", len(st[2])), 14)
            + _pad(String(len(lk[0]), "/", len(lk[1])), 12)
            + String(_euler(lk))
        )
    _check(_euler(sc.link(Simplex([center]))) == 0, "interior link is closed")
    _check(_euler(sc.link(Simplex([side]))) == 1, "boundary link is a path")
    print(
        "\nlink of vertex",
        center,
        ":",
        _join(sc.simplices_in(sc.link(Simplex([center])))),
    )

    # an edge from the center vertex: its two triangles' opposite vertices
    var e = sc[1].simplex_to_index(Simplex([center, center + 1]))
    var lk_e = sc.link([[], [e]])
    print("link of edge", center, center + 1, ":", _join(sc.simplices_in(lk_e)))
    _check(len(lk_e[0]) == 2 and len(lk_e[1]) == 0, "edge link is two vertices")

    # the closed star is a disk with the link as its boundary
    var st = sc.star(Simplex([center]))
    var closed = sc.closure(st)
    _check(
        _same(sc.boundary(closed), sc.link(Simplex([center]))),
        "Bd(Cl(St(v))) = Lk(v)",
    )
    _check(_same(sc.interior(closed), st), "Int(Cl(St(v))) = St(v)")

    # grow rings around the center: R_0 = Cl(St(v)), R_k = Cl(St(R_{k-1}))
    var ring = real_vector(sc[2].num_simplices)
    for t in range(ring.size):
        ring.unsafe_set(t, -1)
    var on_edge = sc.boundary_cells()
    var regions = List[List[List[Int]]]()
    var region = closed.copy()
    print(
        "\n  ring  triangles  Bd edges  chi(Bd)  Bd edges on the square's edge"
    )
    for k in range(n):
        for t in region[2]:
            if ring.unsafe_get(t) < 0:
                ring.unsafe_set(t, Float64(k))
        var bd = sc.boundary(region)
        var shared = 0
        for b in bd[1]:
            if b in on_edge[1]:
                shared += 1
        print(
            "  "
            + _pad(String(k), 6)
            + _pad(String(len(region[2])), 11)
            + _pad(String(len(bd[1])), 10)
            + _pad(String(_euler(bd)), 9)
            + String(shared)
        )
        _check(_euler(bd) == 0, "ring boundary is a polygon")
        regions.append(region.copy())
        if len(region[2]) == sc[2].num_simplices:
            break
        region = sc.closure(sc.star(region))

    # the boundary of the square, and the boundary of that
    var interior = sc.interior_cells()
    print(
        "\nboundary of the mesh:",
        len(on_edge[0]),
        "vertices,",
        len(on_edge[1]),
        "edges; interior:",
        len(interior[0]),
        "vertices,",
        len(interior[1]),
        "edges,",
        len(interior[2]),
        "triangles",
    )
    _check(len(on_edge[1]) == 4 * n, "the boundary has 4n edges")
    _check(_same(sc.boundary(on_edge), [[], [], []]), "Bd(Bd(M)) is empty")

    var show = show_requested()
    var plt = pyplot(interactive=show)
    var fig_axes = plt.subplots(1, 3, figsize=Python.tuple(15.0, 5.0))
    var colors: List[String] = ["tab:orange", "tab:green", "tab:purple"]

    # stars and links of the three vertices
    plt.sca(fig_axes[1][0])
    _ = triplot(sc.vertices(), sc.simplices())
    for i in range(3):
        var v = vertices[i]
        _fill(sc, sc.star(Simplex([v]))[2], colors[i])
        _edges(sc, sc.link(Simplex([v]))[1], "r", 3.0)
        _dots(plt, sc, [v], "k")
    plt.title("stars (filled) and links (red)")

    # rings around the center and their boundaries
    plt.sca(fig_axes[1][1])
    _ = triplot(sc.vertices(), sc.simplices())
    var cmap = plt.get_cmap("viridis")
    for k in range(len(regions) - 1, -1, -1):
        var col = triplot(
            sc.vertices(), take_int_rows(sc[2].simplices, regions[k][2])
        )
        col.set_facecolor(cmap(Float64(k) / Float64(max(len(regions) - 1, 1))))
        col.set_alpha(1.0)
        _edges(sc, sc.boundary(regions[k])[1], "w", 2.0)
    _dots(plt, sc, [center], "r")
    plt.title("rings R_k = Cl(St(R_k-1)), boundaries in white")

    # the boundary and interior of the whole mesh
    plt.sca(fig_axes[1][2])
    _ = triplot(sc.vertices(), sc.simplices())
    _edges(sc, on_edge[1], "tab:blue", 3.0)
    _dots(plt, sc, on_edge[0], "tab:blue")
    _dots(plt, sc, interior[0], "w")
    plt.title("boundary (blue) and interior vertices (white)")

    for i in range(3):
        fig_axes[1][i].set_aspect("equal")
    plt.savefig("neighborhoods.png", bbox_inches="tight")
    print("wrote neighborhoods.png")

    write_vtu(
        "neighborhoods.vtu",
        sc.vertices(),
        sc.simplices(),
        cell_data=[VTKField("ring", ring)],
    )
    print("wrote neighborhoods.vtu")
    if show:
        plt.show()


def _euler(cells: List[List[Int]]) -> Int:
    """The alternating sum of the numbers of k-cells."""
    var chi = 0
    for k in range(len(cells)):
        chi += -len(cells[k]) if k % 2 == 1 else len(cells[k])
    return chi


def _same(a: List[List[Int]], b: List[List[Int]]) -> Bool:
    if len(a) != len(b):
        return False
    for k in range(len(a)):
        if a[k] != b[k]:
            return False
    return True


def _check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("check failed: " + what)


def _join(simplices: List[Simplex]) -> String:
    """The simplices' vertex lists, separated by spaces."""
    var out = String()
    for i in range(len(simplices)):
        if i > 0:
            out += " "
        out += String(simplices[i].vertices)
    return out


def _pad(s: String, width: Int) -> String:
    """`s` left-aligned in `width` characters."""
    var out = s
    while out.byte_length() < width:
        out += " "
    return out


def _fill(sc: SimplicialComplex, triangles: List[Int], color: String) raises:
    var col = triplot(sc.vertices(), take_int_rows(sc[2].simplices, triangles))
    col.set_facecolor(color)
    col.set_alpha(0.8)


def _edges(
    sc: SimplicialComplex, edges: List[Int], color: String, width: Float64
) raises:
    if len(edges) == 0:
        return
    var col = lineplot(
        sc.vertices(), take_int_rows(sc[1].simplices, edges), linewidths=width
    )
    col.set_color(color)


def _dots(
    plt: PythonObject, sc: SimplicialComplex, vertices: List[Int], color: String
) raises:
    if len(vertices) == 0:
        return
    var p = take_rows(sc.vertices(), vertices).to_numpy()
    plt.scatter(p.T[0], p.T[1], s=30, c=color, edgecolors="k", zorder=3)
