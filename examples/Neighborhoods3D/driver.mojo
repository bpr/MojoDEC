# Stars, links, and boundaries of cells in a tetrahedral mesh of a solid
# torus: the 3D companion of the Neighborhoods example.
#
# In a mesh of a 3-manifold with boundary, the link of a cell is a sphere
# around it when the cell is inside, and a disk when it is on the
# boundary. This program checks, on PyDEC's 592-tetrahedron solid torus:
#
# - the link of an interior vertex is a triangulated sphere (Euler
#   characteristic 2), and of a boundary vertex a disk (1);
# - the link of an interior edge is a polygon (0), and of an interior
#   triangle its two opposite vertices; a boundary triangle has one;
# - the closed star Cl(St(v)) of an interior vertex has the link as its
#   boundary and the open star as its interior;
# - growing a region by repeated closed stars, Cl(St(R)), gives balls
#   (chi 1, bounded by spheres, chi 2) until the region reaches around the
#   torus and meets itself. On that step its boundary is pinched where
#   the two ends touch, so it is not a surface; after it, the region is a
#   solid torus (chi 0) bounded by a torus (chi 0). Whenever Bd R is a
#   surface, chi(Bd R) = 2 chi(R);
# - the boundary of the whole mesh is a closed torus, with empty boundary.
#
# Run from the repository root:  pixi run examples Neighborhoods3D [--show]
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo [--show]
# --show also opens the figure in a window (see examples/README.md).
# Writes neighborhoods3d.png; neighborhoods3d.vtu, each tetrahedron's ring
# number around an interior vertex; and links.vtu, the two vertex links
# and the torus surface, for ParaView.

from std.python import Python, PythonObject

from mojodec import (
    CellSet,
    Simplex,
    Subset,
    SimplicialComplex,
    VTKField,
    read_mesh,
    write_vtu,
)
from mojodec.core.utils import (
    int_array,
    iset,
    real_vector,
    real_vector_from_list,
    take_int_rows,
)
from mojodec.vis import pyplot, show_requested


def main() raises:
    var mesh = read_mesh("torus_tet.xml")
    var sc = SimplicialComplex(mesh.vertices, mesh.elements)
    print(sc)

    var surface = sc.boundary_cells()
    var interior = sc.interior_cells()
    var inner = interior[0].indices()[0]
    var outer = surface[0].indices()[0]

    print("\n  cell              star (v/e/t/T)  link (v/e/t)  chi(link)")
    var v_in = CellSet.from_indices(sc, 0, [inner])
    var v_out = CellSet.from_indices(sc, 0, [outer])
    var edge = CellSet.from_indices(sc, 1, [interior[1].indices()[0]])
    var face = CellSet.from_indices(sc, 2, [interior[2].indices()[0]])
    var bface = CellSet.from_indices(sc, 2, [surface[2].indices()[0]])
    _row(sc, "interior vertex", v_in)
    _row(sc, "boundary vertex", v_out)
    _row(sc, "interior edge", edge)
    _row(sc, "interior face", face)
    _row(sc, "boundary face", bface)

    var lk_in = sc.link(v_in)
    var lk_out = sc.link(v_out)
    _check(
        lk_in.euler_characteristic() == 2, "interior vertex link is a sphere"
    )
    _check(_closed(sc, lk_in[2]), "interior vertex link has no boundary")
    _check(lk_out.euler_characteristic() == 1, "boundary vertex link is a disk")
    _check(not _closed(sc, lk_out[2]), "boundary vertex link has a boundary")
    _check(
        sc.link(edge).euler_characteristic() == 0,
        "interior edge link is a circle",
    )
    _check(len(sc.link(face)[0]) == 2, "interior face link")
    _check(len(sc.link(bface)[0]) == 1, "boundary face link")

    # the closed star is a ball with the link as its boundary
    var st = sc.star(v_in)
    var closed = sc.closure(st)
    _check(sc.boundary(closed) == lk_in, "Bd(Cl(St(v))) = Lk(v)")
    _check(sc.interior(closed) == st, "Int(Cl(St(v))) = St(v)")

    # grow rings around the vertex: R_0 = Cl(St(v)), R_k = Cl(St(R_{k-1}))
    var ring = real_vector(sc[3].num_simplices)
    for t in range(ring.size):
        ring.unsafe_set(t, -1)
    var chi_region = List[Float64]()
    var chi_boundary = List[Float64]()
    var pinched = List[Int]()
    var region = closed.copy()
    print(
        "\n  ring  tetrahedra  Bd triangles  chi(R)  chi(Bd R)  pinches  on the"
        " torus"
    )
    var k = 0
    while True:
        for t in region[3].indices():
            if ring.unsafe_get(t) < 0:
                ring.unsafe_set(t, Float64(k))
        var bd = sc.boundary(region)
        var pinches = _pinches(sc, bd[2])
        var shared = len(bd[2] & surface[2])
        print(
            "  "
            + _pad(String(k), 6)
            + _pad(String(len(region[3])), 12)
            + _pad(String(len(bd[2])), 14)
            + _pad(String(region.euler_characteristic()), 8)
            + _pad(String(bd.euler_characteristic()), 11)
            + _pad(String(pinches), 9)
            + String(shared)
        )
        _check(_closed(sc, bd[2]), "a ring boundary has no boundary")
        if pinches == 0:
            _check(
                bd.euler_characteristic() == 2 * region.euler_characteristic(),
                "chi(Bd R) = 2 chi(R)",
            )
        chi_region.append(Float64(region.euler_characteristic()))
        chi_boundary.append(Float64(bd.euler_characteristic()))
        if pinches > 0:
            pinched.append(k)
        if len(region[3]) == sc[3].num_simplices:
            break
        region = sc.closure(sc.star(region))
        k += 1
    _check(chi_boundary[0] == 2, "the first ring is a ball")
    _check(chi_boundary[len(chi_boundary) - 1] == 0, "the last is a torus")

    # the boundary of the solid torus, and the boundary of that
    print(
        "\nboundary of the mesh:",
        len(surface[0]),
        "vertices,",
        len(surface[1]),
        "edges,",
        len(surface[2]),
        "triangles; chi =",
        surface.euler_characteristic(),
    )
    print(
        "interior:",
        len(interior[0]),
        "vertices,",
        len(interior[1]),
        "edges,",
        len(interior[2]),
        "triangles,",
        len(interior[3]),
        "tetrahedra",
    )
    _check(surface.euler_characteristic() == 0, "the boundary is a torus")
    _check(sc.boundary(surface).is_empty(), "Bd(Bd(M)) is empty")

    var show = show_requested()
    var plt = pyplot(interactive=show)
    var fig = plt.figure(figsize=Python.tuple(14.0, 6.0))

    # the two vertex links inside the transparent torus surface
    var ax = fig.add_subplot(1, 2, 1, projection="3d")
    _surface(ax, sc, surface[2], "grey", 0.08)
    _surface(ax, sc, lk_in[2], "tab:red", 0.9)
    _surface(ax, sc, lk_out[2], "tab:blue", 0.9)
    var p = sc.vertices().to_numpy()
    ax.scatter(
        Python.list(p[inner][0], p[outer][0]),
        Python.list(p[inner][1], p[outer][1]),
        Python.list(p[inner][2], p[outer][2]),
        c="k",
        s=20,
    )
    _fit(ax, p)
    ax.set_axis_off()
    ax.set_title(
        "links of an interior vertex (sphere, red)\nand a boundary vertex"
        " (disk, blue)"
    )

    # the Euler characteristics of the growing rings
    var ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(
        real_vector_from_list(chi_boundary).to_numpy(),
        "ro-",
        label="chi(Bd R_k)",
    )
    ax2.plot(
        real_vector_from_list(chi_region).to_numpy(), "ks--", label="chi(R_k)"
    )
    ax2.set_xlabel("ring k")
    ax2.set_ylabel("Euler characteristic")
    ax2.set_title("R_k = Cl(St(R_k-1)): balls, then the solid torus")
    for p in pinched:
        ax2.annotate(
            "pinched",
            Python.tuple(Float64(p), chi_boundary[p]),
            xytext=Python.tuple(Float64(p) + 0.3, chi_boundary[p] + 0.3),
            arrowprops=Python.dict(arrowstyle="->"),
        )
    ax2.set_yticks(Python.list(0, 1, 2))
    ax2.legend()
    plt.savefig("neighborhoods3d.png", bbox_inches="tight")
    print("wrote neighborhoods3d.png")

    write_vtu(
        "neighborhoods3d.vtu",
        sc.vertices(),
        sc.simplices(),
        cell_data=[VTKField("ring", ring)],
        format=.APPENDED,
    )
    # 0: interior vertex link, 1: boundary vertex link, 2: torus surface
    var parts: List[List[Int]] = [
        lk_in[2].indices(),
        lk_out[2].indices(),
        surface[2].indices(),
    ]
    var count = len(parts[0]) + len(parts[1]) + len(parts[2])
    var triangles = int_array(count, 3)
    var which = real_vector(count)
    var r = 0
    for i in range(3):
        for t in parts[i]:
            for j in range(3):
                iset(triangles, r, j, sc[2].index_to_simplex(t)[j])
            which.unsafe_set(r, Float64(i))
            r += 1
    write_vtu(
        "links.vtu",
        sc.vertices(),
        triangles,
        cell_data=[VTKField("part", which)],
        format=.APPENDED,
    )
    print("wrote neighborhoods3d.vtu and links.vtu")
    if show:
        plt.show()


def _row(sc: SimplicialComplex, name: String, cells: CellSet) raises:
    var st = sc.star(cells)
    var lk = sc.link(cells)
    print(
        "  "
        + _pad(name, 18)
        + _pad(
            String(
                len(st[0]), "/", len(st[1]), "/", len(st[2]), "/", len(st[3])
            ),
            16,
        )
        + _pad(String(len(lk[0]), "/", len(lk[1]), "/", len(lk[2])), 14)
        + String(lk.euler_characteristic())
    )


def _closed(sc: SimplicialComplex, triangles: Subset) raises -> Bool:
    """Whether every edge of the triangles is on exactly two of them."""
    return sc.boundary(CellSet.from_subset(sc, triangles)).is_empty()


def _pinches(sc: SimplicialComplex, triangles: Subset) raises -> Int:
    """The edges and vertices where the triangles are not a surface.

    On a closed surface every edge is on two triangles, and the triangles
    around each vertex form one fan: their edges opposite the vertex make
    a single cycle.
    """
    var on_edge = List[Int](length=sc[1].num_simplices, fill=0)
    # for each vertex, the edges opposite it, as vertex pairs
    var fans = List[List[Tuple[Int, Int]]](
        length=sc[0].num_simplices, fill=List[Tuple[Int, Int]]()
    )
    for t in triangles.indices():
        var v = sc[2].index_to_simplex(t).vertices.copy()
        for j in range(3):
            var a = v[(j + 1) % 3]
            var b = v[(j + 2) % 3]
            on_edge[sc[1].simplex_to_index(Simplex([a, b]))] += 1
            fans[v[j]].append((a, b))
    var count = 0
    for e in range(len(on_edge)):
        if on_edge[e] != 0 and on_edge[e] != 2:
            count += 1
    for ref fan in fans:
        if len(fan) > 0 and _cycle_length(fan) != len(fan):
            count += 1
    return count


def _cycle_length(edges: List[Tuple[Int, Int]]) -> Int:
    """The number of edges on the cycle through the first edge."""
    var used = List[Bool](length=len(edges), fill=False)
    used[0] = True
    var start = edges[0][0]
    var at = edges[0][1]
    var length = 1
    while at != start:
        var found = False
        for i in range(len(edges)):
            if not used[i] and (edges[i][0] == at or edges[i][1] == at):
                used[i] = True
                at = edges[i][1] if edges[i][0] == at else edges[i][0]
                length += 1
                found = True
                break
        if not found:
            return -1
    return length


def _check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("check failed: " + what)


def _pad(s: String, width: Int) -> String:
    """`s` left-aligned in `width` characters."""
    var out = s
    while out.byte_length() < width:
        out += " "
    return out


def _surface(
    ax: PythonObject,
    sc: SimplicialComplex,
    triangles: Subset,
    color: String,
    alpha: Float64,
) raises:
    var art3d = Python.import_module("mpl_toolkits.mplot3d.art3d")
    var polys = sc.vertices().to_numpy()[
        take_int_rows(sc[2].simplices, triangles.indices()).to_numpy()
    ]
    var col = art3d.Poly3DCollection(polys)
    col.set_facecolor(color)
    col.set_edgecolor("k")
    col.set_linewidth(0.2)
    col.set_alpha(alpha)
    ax.add_collection3d(col)


def _fit(ax: PythonObject, p: PythonObject) raises:
    """Fit 3D axes to the points, with equal scales on x, y, and z."""
    var np = Python.import_module("numpy")
    var lo = np.min(p, axis=0)
    var hi = np.max(p, axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    var e = hi - lo
    ax.set_box_aspect(Python.tuple(e[0], e[1], e[2]))
