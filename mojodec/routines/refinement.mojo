# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Uniform refinement and prolongation (`mojodec.routines.refinement`).
====================================================================
Nested meshes and the maps between their cochains; not in PyDEC.

`refine(sc)` splits every n-simplex into 2^n children by Freudenthal's
subdivision: each edge gains its midpoint, and the children are the
simplices of the Kuhn triangulation of the doubled simplex. It works in
every dimension, the children of a simplex all have half its size in
every direction and `2^-n` of its volume, and repeated refinement gives
finitely many shapes (it does not degenerate). Shared faces are split the
same way from both sides, because each element is split in the order of
its global vertex numbers.

`Refinement.prolongation(k)` is the matrix `P_k` taking a coarse
k-cochain to the fine cochain with the same Whitney form:
`P_k = R_fine W_coarse`. The coarse Whitney forms are fine Whitney forms,
so `P_k` is exact; it commutes with `d` (`P_{k+1} d = d P_k`) and
preserves the Whitney inner products (`P^T M_fine P = M_coarse`). Its
entries depend only on barycentric coordinates, so they are the same
dyadic numbers on every mesh.

```mojo
var r = refine(sc)                   # r.coarse, r.fine, r.parent
var P1 = r.prolongation(1)           # fine edges x coarse edges
var fine_c = P1 @ c.v                # the same Whitney 1-form, finer
```
"""

from mojodec.core.mesh import SimplicialMesh
from mojodec.core.simplicial_complex import SimplexLevel, SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.utils import (
    IntArray,
    cols_of,
    fget,
    fset,
    int_array,
    iset,
    real_array,
    rows_of,
)
from mojodec.routines.combinatorial import combinations
from mojodec.routines.mesh_generation import _permutations


struct Refinement(Movable):
    """A complex, its uniform refinement, and the map between them.

    The fine vertices are the coarse vertices and the midpoints of the
    coarse edges, numbered in lexicographic order of `(a, a)` for coarse
    vertex `a` and `(a, b)` for edge `(a, b)`, `a < b`: vertex 0, the
    midpoints of the edges at vertex 0 toward higher numbers, vertex 1,
    and so on. That order makes the next refinement split every child the
    way its parent was split. The fine elements are the children of the
    coarse elements, in order, `2^n` per parent.
    """

    var coarse: SimplicialComplex
    """The complex that was refined."""
    var fine: SimplicialComplex
    """The refined complex."""
    var parent: List[Int]
    """For each fine element, the coarse element that contains it."""

    def __init__(
        out self,
        coarse: SimplicialComplex,
        fine: SimplicialComplex,
        var parent: List[Int],
    ):
        """Assemble a refinement; `refine` builds them.

        Args:
            coarse: The coarse complex.
            fine: The refined complex.
            parent: Each fine element's coarse element.
        """
        self.coarse = coarse
        self.fine = fine
        self.parent = parent^

    def prolongation(self, k: Int) raises -> CSRMatrix:
        """The prolongation of k-cochains, `P_k = R_fine W_coarse`.

        Entry `(tau, sigma)` is the integral of the coarse Whitney form
        of `sigma` over the fine k-simplex `tau`, both oriented as the
        cochains of their complexes are. A fine simplex inside a coarse
        face gets the same value from every element around that face,
        since Whitney forms have continuous tangential traces.

        ```mojo
        var c_fine = r.prolongation(1) @ c.v   # P d = d P, exactly
        ```

        Args:
            k: The degree, `0..n`.

        Returns:
            A `fine[k].num_simplices x coarse[k].num_simplices` matrix.
            Row `tau` has the coarse k-simplices of an element containing
            `tau`; for `k = 0` an old vertex has one entry 1 and a
            midpoint two entries 1/2.

        Raises:
            If `k` is outside `0..n`.
        """
        var n = self.coarse.complex_dimension()
        if k < 0 or k > n:
            raise Error(String("prolongation: k = ", k, " is outside 0..", n))
        var tpl = _template(n)
        ref C = self.coarse[k]
        ref F = self.fine[k]
        var done = List[Bool](length=F.num_simplices, fill=False)
        var local = List[Int](capacity=n + 1)
        for i in range(n + 1):
            local.append(i)
        var faces = combinations(local, k + 1)  # sorted local = sorted global
        var rows = List[Int]()
        var cols = List[Int]()
        var vals = List[Float64]()
        ref top = self.coarse[n]
        var num = _Numbering(self.coarse)
        for t in range(top.num_simplices):
            var g = _row(top.simplices, t, n + 1)
            var t_sign = -1.0 if k == n and _odd(top, t) else 1.0
            var coarse_faces = List[Int](capacity=len(faces))
            for ref f in faces:
                var ids = List[Int](capacity=k + 1)
                for i in f:
                    ids.append(g[i])
                coarse_faces.append(C.find_vertices(ids))
            for ref child in tpl.children:
                for ref sub in combinations(local, k + 1):
                    # the fine face: its pairs, ordered by global vertex
                    var pairs = List[Int](capacity=k + 1)
                    var ids = List[Int](capacity=k + 1)
                    for m in sub:
                        var code = child[m]
                        var gid = _fine_vertex(self.coarse, g, code, n, num)
                        var at = len(ids)
                        ids.append(gid)
                        pairs.append(code)
                        while at > 0 and ids[at - 1] > ids[at]:
                            ids.swap_elements(at - 1, at)
                            pairs.swap_elements(at - 1, at)
                            at -= 1
                    var tau = F.find_vertices(ids)
                    if tau < 0:
                        raise Error(
                            "prolongation: the fine complex lacks a face"
                        )
                    if done[tau]:
                        continue
                    done[tau] = True
                    var sign = t_sign
                    if k == n and _odd(F, tau):
                        sign = -sign
                    var beta = _doubled_barycentric(pairs, n)
                    for s in range(len(faces)):
                        var v = _whitney_integral(beta, faces[s]) * sign
                        if v != 0:
                            rows.append(tau)
                            cols.append(coarse_faces[s])
                            vals.append(v)
        return CSRMatrix.from_coo(
            F.num_simplices, C.num_simplices, rows, cols, vals
        )


struct _Template(Movable):
    """Freudenthal's subdivision of the simplex `[0, ..., n]`: each child
    as `n + 1` vertex codes `i * (n + 1) + j` (`i <= j`, the midpoint of
    vertices i and j, or vertex i when equal), with whether listing them
    in that order reverses the orientation."""

    var children: List[List[Int]]
    var odd: List[Bool]

    def __init__(out self):
        self.children = List[List[Int]]()
        self.odd = List[Bool]()


def _template(n: Int) -> _Template:
    """The children of the doubled simplex `2 S`,
    `S = {1 >= x_1 >= ... >= x_n >= 0}` with vertices `p_i` (i ones then
    zeros): the Kuhn simplices of the unit cubes at the corners
    `c in {0, 1}^n` that lie in `2 S`. A lattice point of `2 S` is
    `p_i + p_j`, the midpoint of vertices i and j, and a Kuhn simplex's
    orientation is the sign of its permutation."""
    var out = _Template()
    var perms = List[List[Int]]()
    var odd = List[Bool]()
    _permutations(n, perms, odd)
    for c in range(1 << n):
        for q in range(len(perms)):
            var y = List[Int](capacity=n)
            for a in range(n):
                y.append((c >> a) & 1)
            var codes = List[Int](capacity=n + 1)
            var inside = True
            for m in range(n + 1):
                if m > 0:
                    y[perms[q][m - 1]] += 1
                var twos = 0
                var ones = 0
                for a in range(n):
                    if a > 0 and y[a] > y[a - 1]:
                        inside = False
                    if y[a] == 2:
                        twos += 1
                    elif y[a] == 1:
                        ones += 1
                    elif y[a] != 0:
                        inside = False
                codes.append(twos * (n + 1) + twos + ones)
            if inside:
                out.children.append(codes^)
                out.odd.append(odd[q])
    return out^


def _row(A: IntArray, i: Int, w: Int) -> List[Int]:
    """Row i of a `* x w` integer array."""
    var out = List[Int](capacity=w)
    for j in range(w):
        out.append(Int(A.unsafe_get(i * w + j)))
    return out^


def _odd(level: SimplexLevel, i: Int) -> Bool:
    """Whether simplex i is oriented against its sorted vertex list."""
    return Int(level.simplex_parity.unsafe_get(i)) == 1


struct _Numbering(Movable):
    """The fine vertex numbers: vertex `a` of the coarse complex and the
    midpoint of edge `(a, b)`, `a < b`, numbered in lexicographic order of
    `(a, a)` and `(a, b)`. This order extends the componentwise order of
    the pairs, so on every child it is the order of its path in the
    doubled simplex: the next refinement splits each child as its parent
    was split (Bey's inherited order), which keeps the number of shapes
    bounded."""

    var vertex: List[Int]
    var edge: List[Int]

    def __init__(out self, sc: SimplicialComplex) raises:
        var nv = rows_of(sc.vertices())
        var n = sc.complex_dimension()
        var ne = sc[1].num_simplices if n >= 1 else 0
        self.vertex = List[Int](length=nv, fill=-1)
        self.edge = List[Int](length=ne, fill=-1)
        var next = 0
        var e = 0
        for a in range(nv):
            self.vertex[a] = next
            next += 1
            # the edges are sorted, so those starting at a come next
            while e < ne and Int(sc[1].simplices.unsafe_get(2 * e)) == a:
                self.edge[e] = next
                next += 1
                e += 1


def _fine_vertex(
    coarse: SimplicialComplex, g: List[Int], code: Int, n: Int, num: _Numbering
) raises -> Int:
    """The fine vertex of a code on the element with sorted vertices `g`."""
    var i = code // (n + 1)
    var j = code % (n + 1)
    if i == j:
        return num.vertex[g[i]]
    var e = coarse[1].find_vertices([g[i], g[j]])
    if e < 0:
        raise Error("refine: an element's edge is missing")
    return num.edge[e]


def _doubled_barycentric(codes: List[Int], n: Int) -> List[List[Int]]:
    """Twice the barycentric coordinates of each coded point: integers."""
    var out = List[List[Int]](capacity=len(codes))
    for code in codes:
        var b = List[Int](length=n + 1, fill=0)
        b[code // (n + 1)] += 1
        b[code % (n + 1)] += 1
        out.append(b^)
    return out^


def _whitney_integral(beta2: List[List[Int]], s: List[Int]) -> Float64:
    """The integral of `W_s` over the simplex whose vertices have twice
    the barycentric coordinates `beta2`, in that order. `W_s` is affine,
    so it is its value at the centroid `b` on the edge vectors over `k!`:
    `sum_i (-1)^i b[s_i] det(beta_m[s_j] - beta_0[s_j])`, j over `s`
    without `s_i`, m over `1..k`. In integers, with one rounding at the
    end: `b = B / (2 (k + 1))` and the determinant is `D / 2^k`."""
    var k = len(s) - 1
    var total = 0
    for i in range(k + 1):
        var B = 0
        for ref p in beta2:
            B += p[s[i]]
        if B == 0:
            continue
        var A = List[Int](capacity=k * k)
        for j in range(k + 1):
            if j == i:
                continue
            for m in range(1, k + 1):
                A.append(beta2[m][s[j]] - beta2[0][s[j]])
        var term = B * _det(A, k)
        total += -term if i % 2 == 1 else term
    return Float64(total) / Float64(2 * (k + 1) * (1 << k))


def _det(A: List[Int], k: Int) -> Int:
    """The determinant of a row-major `k x k` integer matrix, by cofactor
    expansion along the first row (k is at most the dimension)."""
    if k == 0:
        return 1
    if k == 1:
        return A[0]
    var det = 0
    for c in range(k):
        if A[c] == 0:
            continue
        var minor = List[Int](capacity=(k - 1) * (k - 1))
        for r in range(1, k):
            for j in range(k):
                if j != c:
                    minor.append(A[r * k + j])
        var term = A[c] * _det(minor, k - 1)
        det += -term if c % 2 == 1 else term
    return det


def refine(sc: SimplicialComplex) raises -> Refinement:
    """Split every element into `2^n` children (Freudenthal subdivision).

    The vertices of each child are vertices and edge midpoints of its
    parent: the lattice points of the doubled simplex, cut along the
    Kuhn triangulation of its unit cubes. Each element is split in the
    order of its sorted global vertex numbers, so a face shared by two
    elements is split the same way from both sides and the result is
    conforming. Children are oriented as their parent, so a coherently
    oriented complex stays one. In 2D this is the usual split into four
    triangles; in 3D, eight tetrahedra, with the interior octahedron cut
    along the diagonal between the midpoints of edges `(0, 2)` and
    `(1, 3)` in sorted vertex order.

    Args:
        sc: The complex. Its geometry is not needed, only its vertices.

    Returns:
        The coarse and fine complexes and each fine element's parent.

    Raises:
        If the fine complex fails `SimplicialComplex`'s checks (it does
        not for a valid coarse complex).
    """
    var n = sc.complex_dimension()
    var tpl = _template(n)
    var nchild = len(tpl.children)
    ref V = sc.vertices()
    var N = cols_of(V)
    var nv = rows_of(V)
    var ne = sc[1].num_simplices if n >= 1 else 0
    var num = _Numbering(sc)
    var vertices = real_array(nv + ne, N)
    for v in range(nv):
        for x in range(N):
            fset(vertices, num.vertex[v], x, fget(V, v, x))
    if n >= 1:
        ref edges = sc[1].simplices
        for e in range(ne):
            var a = Int(edges.unsafe_get(2 * e))
            var b = Int(edges.unsafe_get(2 * e + 1))
            for x in range(N):
                fset(
                    vertices,
                    num.edge[e],
                    x,
                    0.5 * (fget(V, a, x) + fget(V, b, x)),
                )
    ref top = sc[n]
    var elements = int_array(top.num_simplices * nchild, n + 1)
    var parent = List[Int](capacity=top.num_simplices * nchild)
    for t in range(top.num_simplices):
        var g = _row(top.simplices, t, n + 1)
        var flip = _odd(top, t)
        for c in range(nchild):
            var ids = List[Int](capacity=n + 1)
            for code in tpl.children[c]:
                ids.append(_fine_vertex(sc, g, code, n, num))
            if n >= 1 and tpl.odd[c] != flip:
                ids.swap_elements(n - 1, n)
            var row = t * nchild + c
            for j in range(n + 1):
                iset(elements, row, j, ids[j])
            parent.append(t)
    var fine = SimplicialComplex(SimplicialMesh(vertices, elements))
    return Refinement(sc, fine, parent^)
