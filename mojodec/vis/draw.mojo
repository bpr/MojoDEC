# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Drawing (`mojodec.vis.draw`).
=============================
Largely inspired by `pydec.vis.draw`. `triplot`, `lineplot`, and `lineplot2`
draw 2D meshes and line segments with matplotlib, through Mojo's Python
interop; like PyDEC's, they add to the current axes (`pyplot.gca()`) and return
the matplotlib collection they added. They need a Python environment with NumPy
and matplotlib, which is loaded on first use.

`simplex_quivers` and `cube_quivers` are pure Mojo: they sample a 1-cochain
as a vector field, ready for `quiver`. `pyplot` and `show_requested` set
up matplotlib for scripts that save figures, or show them on request.

```mojo
from mojodec.vis.draw import pyplot, show_requested, triplot, simplex_quivers

var plt = pyplot(interactive=show_requested())
_ = triplot(sc.vertices(), sc.simplices())
var q = simplex_quivers(sc, form)
var b = q.bases.to_numpy()
var v = q.vectors.to_numpy()
plt.gca().quiver(b.T[0], b.T[1], v.T[0], v.T[1])
plt.savefig("form.png")
if show_requested():
    plt.show()
```
"""

from std.python import Python, PythonObject
from std.sys import argv

from mojodec.core.regular_cube_complex import RegularCubeComplex
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.utils import (
    IntArray,
    RealArray,
    cols_of,
    fget,
    fset,
    iget,
    real_array,
    row_list,
    rows_of,
    take_rows,
)
from mojodec.routines.geometry import barycentric_gradients


def pyplot(interactive: Bool = False) raises -> PythonObject:
    """`matplotlib.pyplot`, set up for a script.

    Unless `interactive`, this selects the non-interactive Agg backend, so
    figures can be saved with `savefig` without a display. With
    `interactive`, matplotlib keeps its default GUI backend and
    `pyplot.show()` opens a window per figure.

    Args:
        interactive: Keep matplotlib's default (GUI) backend.

    Returns:
        The `matplotlib.pyplot` module.

    Raises:
        If matplotlib cannot be imported, or the backend cannot be set.
    """
    if not interactive:
        Python.import_module("matplotlib").use("Agg")
    return Python.import_module("matplotlib.pyplot")


def show_requested() -> Bool:
    """Whether the program was run with a `--show` argument.

    Returns:
        True if any command-line argument is `--show`.
    """
    for arg in argv():
        if arg == "--show":
            return True
    return False


def triplot(
    vertices: RealArray, indices: IntArray, labels: Bool = False
) raises -> PythonObject:
    """Draw a 2D triangle mesh as gray, half-transparent triangles.

    The edges are drawn in black: PyDEC relied on matplotlib's pre-2.0
    default edge color, and current matplotlib draws no edges.

    With `labels`, each triangle's index is written at its barycenter.

    Args:
        vertices: Vertex coordinates, `N x 2`.
        indices: The triangles, one row of 3 vertex indices each.
        labels: Write each triangle's index at its barycenter.

    Returns:
        The `PolyCollection` added to the current axes.

    Raises:
        If `vertices` is not `N x 2`, or on a Python error (NumPy or
        matplotlib missing, or a bad index).
    """
    _check_planar(vertices, "triplot")
    var collections = Python.import_module("matplotlib.collections")
    # [triangle][vertex][x/y]
    var triangles = vertices.to_numpy()[indices.to_numpy()]
    var col = collections.PolyCollection(triangles)
    col.set_facecolor("gray")
    col.set_edgecolor("k")
    col.set_alpha(0.5)
    col.set_linewidth(1)
    _add_collection(col)
    if labels:
        _label(triangles)
    return col


def lineplot(
    vertices: RealArray,
    indices: IntArray,
    labels: Bool = False,
    linewidths: Float64 = 1,
) raises -> PythonObject:
    """Draw the 2D line segments `indices` (pairs of vertex indices) in black.

    With `labels`, each segment's index is written at its midpoint.

    Args:
        vertices: Vertex coordinates, `N x 2`.
        indices: The segments, `M x 2` vertex indices.
        labels: Write each segment's index at its midpoint.
        linewidths: The width of every segment, in points.

    Returns:
        The `LineCollection` added to the current axes.

    Raises:
        If `vertices` is not `N x 2` or `indices` is not `M x 2`, or on a
        Python error (NumPy or matplotlib missing, or a bad index).
    """
    return _lineplot(vertices, indices, labels, PythonObject(linewidths))


def lineplot(
    vertices: RealArray,
    indices: IntArray,
    labels: Bool,
    linewidths: RealArray,
) raises -> PythonObject:
    """Draw 2D line segments, each with its own width from `linewidths`.

    Args:
        vertices: Vertex coordinates, `N x 2`.
        indices: The segments, `M x 2` vertex indices.
        labels: Write each segment's index at its midpoint.
        linewidths: The width of each segment, in points.

    Returns:
        The `LineCollection` added to the current axes.

    Raises:
        If `vertices` is not `N x 2` or `indices` is not `M x 2`, or on a
        Python error (NumPy or matplotlib missing, or a bad index).
    """
    return _lineplot(vertices, indices, labels, linewidths.to_numpy())


def lineplot2(
    tails: RealArray,
    heads: RealArray,
    labels: Bool = False,
    linewidths: Float64 = 1,
) raises -> PythonObject:
    """Draw the 2D segments from `tails[i]` to `heads[i]` in black.

    With `labels`, each segment's index is written at its midpoint.

    Args:
        tails: The segments' start points, `M x 2`.
        heads: The segments' end points, `M x 2`.
        labels: Write each segment's index at its midpoint.
        linewidths: The width of every segment, in points.

    Returns:
        The `LineCollection` added to the current axes.

    Raises:
        If `tails` or `heads` is not `M x 2`, they differ in length, or on a
        Python error (NumPy or matplotlib missing).
    """
    _check_planar(tails, "lineplot2")
    _check_planar(heads, "lineplot2")
    if rows_of(tails) != rows_of(heads):
        raise Error("lineplot2: tails and heads must have the same length")
    var np = Python.import_module("numpy")
    var lines = np.stack(
        Python.tuple(tails.to_numpy(), heads.to_numpy()), axis=1
    )
    return _draw_lines(lines, labels, PythonObject(linewidths))


struct Quivers(Copyable, Movable):
    """Arrows for `quiver`: base points and the vectors drawn from them."""

    var bases: RealArray
    """The base points, one row each."""
    var vectors: RealArray
    """The vectors, one row each, matching `bases`."""

    def __init__(out self, var bases: RealArray, var vectors: RealArray):
        """Arrows from their bases and vectors.

        Args:
            bases: The base points.
            vectors: The vectors.
        """
        self.bases = bases^
        self.vectors = vectors^


def simplex_quivers(sc: SimplicialComplex, form: RealArray) raises -> Quivers:
    """Sample the Whitney interpolant of a 1-cochain at top-simplex barycenters.

    Args:
        sc: The complex, of dimension at least 1.
        form: The 1-cochain, one value per edge of `sc`.

    Returns:
        The barycenters and the interpolated vectors there, both
        `num_top_simplices x embedding_dimension`.

    Raises:
        If `form` does not have one value per edge.
    """
    var N = sc.complex_dimension()
    var E = sc.embedding_dimension()
    if form.size != sc[1].num_simplices:
        raise Error("simplex_quivers: form must have one value per edge")
    ref top = sc[N]
    ref edges = sc[1]
    var bases = real_array(top.num_simplices, E)
    var dirs = real_array(top.num_simplices, E)
    for n in range(top.num_simplices):
        # the level's simplices are sorted, so its edges have parity 0
        var verts = row_list(top.simplices, n)
        var pts = take_rows(sc.vertices(), verts)
        var d_lambda = barycentric_gradients(pts)
        for e in range(E):
            var s: Float64 = 0
            for i in range(N + 1):
                s += fget(pts, i, e)
            fset(bases, n, e, s / Float64(N + 1))
        for i in range(N + 1):
            for j in range(i + 1, N + 1):
                var v = form.unsafe_get(
                    edges.simplex_to_index(Simplex([verts[i], verts[j]]))
                )
                for e in range(E):
                    var g = fget(d_lambda, j, e) - fget(d_lambda, i, e)
                    fset(dirs, n, e, fget(dirs, n, e) + v * g)
    for k in range(dirs.size):
        dirs.unsafe_set(k, dirs.unsafe_get(k) / Float64(N + 1))
    return Quivers(bases^, dirs^)


def cube_quivers(cmplx: RegularCubeComplex, vals: RealArray) raises -> Quivers:
    """Sample a 1-cochain on a regular cube complex at the 2-cube centers.

    Each 2-cube's vector has, along axis `i`, the mean of the values on its
    edges parallel to axis `i`.

    Args:
        cmplx: The cube complex, of dimension at least 2.
        vals: The 1-cochain, one value per edge (1-cube).

    Returns:
        The 2-cube centers and the vectors there, both
        `num_2_cubes x complex_dimension`.

    Raises:
        If the complex has dimension below 2, or `vals` does not have one
        value per edge.
    """
    var N = cmplx.complex_dimension()
    if N < 2:
        raise Error("cube_quivers: the complex has no 2-cubes")
    ref edges = cmplx[1].cube_array
    ref faces = cmplx[2].cube_array
    if vals.size != rows_of(edges):
        raise Error("cube_quivers: vals must have one value per edge")
    var num_faces = rows_of(faces)
    var axis_col = cols_of(edges) - 1
    var bases = real_array(num_faces, N)
    for f in range(num_faces):
        for i in range(N):
            fset(bases, f, i, Float64(iget(faces, f, i)) + 0.5)
    var dirs = real_array(num_faces, N)
    # boundary of the 2-cubes: rows are edges, columns are faces
    var coo = cmplx[2].boundary.to_coo()
    for k in range(len(coo.rows)):
        var edge = coo.rows[k]
        var face = coo.cols[k]
        var axis = iget(edges, edge, axis_col)
        var w = 0.5 * abs(coo.values[k]) * vals.unsafe_get(edge)
        fset(dirs, face, axis, fget(dirs, face, axis) + w)
    return Quivers(bases^, dirs^)


def _check_planar(points: RealArray, caller: String) raises:
    if points.ndim != 2 or cols_of(points) != 2:
        raise Error(caller + ": expected an N x 2 array of 2D points")


def _lineplot(
    vertices: RealArray,
    indices: IntArray,
    labels: Bool,
    linewidths: PythonObject,
) raises -> PythonObject:
    _check_planar(vertices, "lineplot")
    if indices.ndim != 2 or cols_of(indices) != 2:
        raise Error("lineplot: indices must be an N x 2 array of segments")
    # [segment][vertex][x/y]
    var lines = vertices.to_numpy()[indices.to_numpy()]
    return _draw_lines(lines, labels, linewidths)


def _draw_lines(
    lines: PythonObject, labels: Bool, linewidths: PythonObject
) raises -> PythonObject:
    var collections = Python.import_module("matplotlib.collections")
    var col = collections.LineCollection(lines)
    col.set_color("k")
    col.set_linewidth(linewidths)
    _add_collection(col)
    if labels:
        _label(lines)
    return col


def _add_collection(col: PythonObject) raises:
    var plt = Python.import_module("matplotlib.pyplot")
    var ax = plt.gca()
    ax.add_collection(col, autolim=True)
    plt.axis("off")
    ax.autoscale_view()


def _label(cells: PythonObject) raises:
    """Write each cell's index at the mean of its points."""
    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")
    var centers = np.average(cells, axis=1)
    for n in range(Int(py=centers.shape[0])):
        var c = centers[n]
        plt.text(
            c[0],
            c[1],
            String(n),
            color="k",
            fontsize=8,
            horizontalalignment="center",
            verticalalignment="center",
        )
