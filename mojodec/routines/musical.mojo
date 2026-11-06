# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Musical isomorphisms (`mojodec.routines.musical`).
==================================================
The discrete flat, which turns a vector field into a 1-cochain; PyDEC has
no flat. The opposite direction, a 1-cochain to vectors, is
`mojodec.vis.simplex_quivers` (Whitney interpolation).

The flat is Hirani's dual-primal-primal ("DPP") flat (A. N. Hirani,
*Discrete Exterior Calculus*, PhD thesis, Caltech 2003, sec. 5.5). A vector
field `X` constant on each top simplex gives, on an edge `e`,

    flat(X)[e] = sum over top simplices t containing e of
                 |*e ^ t| / |*e|  X(t) . (e_1 - e_0)

where `|*e ^ t|` is the signed length (area, ...) of the part of e's
circumcentric dual cell inside `t`. The weights add up to 1, so a constant
field is flattened exactly: `flat(X)[e] = X . (e_1 - e_0)`.
"""

from std.math import abs

from mojodec.core.cochain import Cochain
from mojodec.core.primality import Primality
from mojodec.core.typed import PRIMAL, TypedCochain, TypedComplex
from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import (
    SimplicialComplex,
    _is_negative_zero,
)
from mojodec.core.utils import (
    IntArray,
    RealArray,
    cols_of,
    fget,
    int_array,
    real_array,
    real_vector,
    row_list,
    rows_of,
)
from mojodec.routines.combinatorial import combinations, permutations
from mojodec.routines.geometry import unsigned_volume


def flat[
    n: Int
](tc: TypedComplex[n], X: RealArray) raises -> TypedCochain[n, 1, PRIMAL]:
    """The DPP flat of a vector field, as a typed primal 1-cochain.

    See the untyped overload.

    Parameters:
        n: The complex dimension, at least 1.

    Args:
        tc: The complex, with geometry.
        X: `num_top_simplices x embedding_dimension` vectors.

    Returns:
        The primal 1-cochain `flat(X)`.

    Raises:
        If the complex was built without geometry, or `X` has the wrong
            shape.
    """
    comptime assert n >= 1, "flat: the complex has no edges"
    return TypedCochain[n, 1, PRIMAL](flat(tc.sc, X))


def flat(sc: SimplicialComplex, X: RealArray) raises -> Cochain:
    """The primal 1-cochain of the vector field `X`, by the DPP flat.

    `X` is `num_top_simplices x embedding_dimension`: one vector per top
    simplex, in the order of `sc.simplices()`. An edge whose dual cell has
    (numerically) zero signed size, where the weights are undefined, takes
    the mean of its top simplices' contributions instead.

    ```mojo
    var omega = flat(sc, X)
    ```

    Args:
        sc: A complex of dimension at least 1, with geometry.
        X: `num_top_simplices x embedding_dimension` vectors.

    Returns:
        The primal 1-cochain `flat(X)`.

    Raises:
        If `sc` has no edges or was built without geometry, or `X` has the
            wrong shape.
    """
    var n = sc.complex_dimension()
    var E = sc.embedding_dimension()
    if n < 1:
        raise Error("flat: the complex has no edges")
    if not sc.has_geometry():
        raise Error("flat: the complex was built without geometry")
    ref top = sc[n]
    if X.ndim != 2 or rows_of(X) != top.num_simplices or cols_of(X) != E:
        raise Error(
            String(
                "flat: X must be ",
                top.num_simplices,
                " x ",
                E,
                " (one vector per top simplex)",
            )
        )
    ref edges = sc[1]
    var weighted = real_vector(edges.num_simplices)
    var plain = real_vector(edges.num_simplices)
    var count = List[Int](length=edges.num_simplices, fill=0)
    var size = real_vector(edges.num_simplices)
    for t in range(top.num_simplices):
        var verts = row_list(top.simplices, t)
        for pair in combinations(verts, 2):
            var e = edges.simplex_to_index(Simplex(pair))
            var along: Float64 = 0
            for c in range(E):
                along += fget(X, t, c) * (
                    fget(sc.vertices(), pair[1], c)
                    - fget(sc.vertices(), pair[0], c)
                )
            var piece = _dual_piece(sc, verts, pair)
            weighted.unsafe_set(e, weighted.unsafe_get(e) + piece * along)
            plain.unsafe_set(e, plain.unsafe_get(e) + along)
            size.unsafe_set(e, size.unsafe_get(e) + abs(piece))
            count[e] += 1
    var out = real_vector(edges.num_simplices)
    for e in range(edges.num_simplices):
        var dual = edges.dual_volume.unsafe_get(e)
        if abs(dual) > 1e-12 * size.unsafe_get(e):
            out.unsafe_set(e, weighted.unsafe_get(e) / dual)
        else:
            out.unsafe_set(e, plain.unsafe_get(e) / Float64(count[e]))
    return Cochain(sc, 1, Primality.PRIMAL, out^)


struct DualEdgePieces(Copyable, Movable):
    """The signed sizes `|*e ^ t|` of each edge's dual cell inside each top
    simplex, as `dual_edge_pieces` computes them."""

    var values: RealArray
    """`num_top_simplices x C(n+1, 2)`: entry `(t, j)` is the piece of the
    `j`-th edge of top simplex `t` (vertex pairs of its sorted vertices, in
    `combinations` order)."""
    var edges: IntArray
    """The same shape as `values`: entry `(t, j)` is that edge's index in
    `sc[1]`."""

    def __init__(out self, var values: RealArray, var edges: IntArray):
        """Pieces from their values and edge indices.

        Args:
            values: The pieces.
            edges: The edge of each piece.
        """
        self.values = values^
        self.edges = edges^

    def edge(self, t: Int, j: Int) -> Int:
        """The index in `sc[1]` of the `j`-th edge of top simplex `t`.

        Args:
            t: A top simplex.
            j: A local edge, `0 <= j < C(n+1, 2)`.

        Returns:
            The edge's index.
        """
        return Int(self.edges.unsafe_get(t * cols_of(self.edges) + j))

    def piece(self, t: Int, j: Int) -> Float64:
        """The piece of the `j`-th edge's dual cell inside top simplex `t`.

        Args:
            t: A top simplex.
            j: A local edge, `0 <= j < C(n+1, 2)`.

        Returns:
            `|*e ^ t|`, signed as the dual volumes are.
        """
        return self.values.unsafe_get(t * cols_of(self.values) + j)


def dual_edge_pieces(sc: SimplicialComplex) raises -> DualEdgePieces:
    """The signed sizes `|*e ^ t|` of each edge's dual cell inside each top simplex.

    Summing an edge's pieces over its top simplices gives its dual volume,
    `sc[1].dual_volume`:

    ```mojo
    var P = dual_edge_pieces(sc)
    for t in range(sc[n].num_simplices):
        for j in range((n + 1) * n // 2):
            total[P.edge(t, j)] += P.piece(t, j)
    ```

    Args:
        sc: A complex of dimension `n >= 1`, with geometry.

    Returns:
        The pieces, with the edge each one belongs to.

    Raises:
        If `sc` has no edges or was built without geometry.
    """
    var n = sc.complex_dimension()
    if n < 1:
        raise Error("dual_edge_pieces: the complex has no edges")
    if not sc.has_geometry():
        raise Error("dual_edge_pieces: the complex was built without geometry")
    ref top = sc[n]
    var pairs = (n + 1) * n // 2
    var values = real_array(top.num_simplices, pairs)
    var edges = int_array(top.num_simplices, pairs)
    for t in range(top.num_simplices):
        var verts = row_list(top.simplices, t)
        var j = 0
        for pair in combinations(verts, 2):
            values.unsafe_set(t * pairs + j, _dual_piece(sc, verts, pair))
            edges.unsafe_set(
                t * pairs + j,
                Scalar[DType.int](sc[1].simplex_to_index(Simplex(pair))),
            )
            j += 1
    return DualEdgePieces(values^, edges^)


def _dual_piece(
    sc: SimplicialComplex, top: List[Int], edge: List[Int]
) raises -> Float64:
    """`|*edge ^ top|`, summed over the flags edge < ... < top.

    Each flag contributes the volume of the simplex spanned by its
    circumcenters, signed as in the dual volumes: a step to a face is
    negative when the parent's circumcenter lies across that face from
    the parent's opposite vertex.
    """
    var n = sc.complex_dimension()
    var E = sc.embedding_dimension()
    var rest = List[Int]()
    for v in top:
        if v not in edge:
            rest.append(v)
    var orders = permutations(rest)
    if len(orders) == 0:
        orders.append(List[Int]())  # the edge is a top simplex
    var total: Float64 = 0
    for ref order in orders:
        var flag = edge.copy()
        var pts = real_array(n, E)
        var sign: Float64 = 1
        _set_circumcenter(sc, flag, pts, 0)
        for j in range(len(order)):
            var parent = flag.copy()
            parent.append(order[j])
            sort(parent)
            # barycentric weight of the parent's circumcenter on the vertex
            # not in the face below it
            var p = sc[len(parent) - 1].simplex_to_index(Simplex(parent))
            var b = fget(
                sc[len(parent) - 1].bary_circumcenter,
                p,
                parent.index(order[j]),
            )
            if b < 0 or (b == 0 and _is_negative_zero(b)):
                sign = -sign
            flag = parent^
            _set_circumcenter(sc, flag, pts, j + 1)
        total += sign * unsigned_volume(pts)
    return total


def _set_circumcenter(
    sc: SimplicialComplex, simplex: List[Int], mut pts: RealArray, row: Int
) raises:
    ref level = sc[len(simplex) - 1]
    var i = level.simplex_to_index(Simplex(simplex))
    var E = cols_of(pts)
    for c in range(E):
        pts.unsafe_set(row * E + c, fget(level.circumcenter, i, c))
