# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Oriented simplices (`mojodec.core.simplex`).
============================================
Port of `pydec.mesh.simplex.simplex`: a simplex is stored as its sorted
vertex list plus a parity bit giving its orientation relative to that sorted
order. Equality and hashing ignore the parity (as PyDEC's tuple subclass
does), so oppositely oriented copies of a simplex find the same dictionary
entry.
"""

from std.hash import Hasher

from mojodec.routines.combinatorial import relative_parity


struct Simplex(Copyable, Equatable, Hashable, Movable, Sized, Writable):
    """An oriented simplex: sorted vertices plus orientation parity."""

    var vertices: List[Int]
    """The vertex indices in increasing order."""
    var parity: Int
    """0 if the simplex has the orientation of `vertices`, 1 otherwise."""

    def __init__(out self, s: List[Int], parity: Int = 0) raises:
        """The simplex spanned by `s`, oriented as `s` is ordered.

        `parity` flips the orientation once more when it is 1.

        Args:
            s: The vertex indices, in the order giving the orientation.
            parity: 1 to reverse the orientation of `s`, else 0.

        Raises:
            If `s` contains a repeated vertex.
        """
        var sorted_s = s.copy()
        sort(sorted_s)
        self.parity = relative_parity(sorted_s, s) ^ parity
        self.vertices = sorted_s^

    def __len__(self) -> Int:
        """Number of vertices (dimension + 1).

        Returns:
            The number of vertices.
        """
        return len(self.vertices)

    def __getitem__(self, i: Int) -> Int:
        """The `i`-th vertex in sorted order.

        Args:
            i: A position in `0 ..< len(self)`.

        Returns:
            The vertex index at position `i` of `vertices`.
        """
        return self.vertices[i]

    def dimension(self) -> Int:
        """The simplex dimension (number of vertices - 1).

        Returns:
            The dimension.
        """
        return len(self.vertices) - 1

    def boundary(self) raises -> List[Simplex]:
        """The oriented faces in the boundary of this simplex.

        Face `n` omits vertex `n` and has parity `(parity + n) % 2`.

        Returns:
            The `dimension() + 1` faces, face `n` omitting `vertices[n]`.

        Raises:
            Never in practice: the faces of a valid simplex have distinct
            vertices.
        """
        var out = List[Simplex](capacity=len(self.vertices))
        for n in range(len(self.vertices)):
            var face = List[Int](capacity=len(self.vertices) - 1)
            for i in range(len(self.vertices)):
                if i != n:
                    face.append(self.vertices[i])
            out.append(Simplex(face, (self.parity + n) % 2))
        return out^

    def __eq__(self, other: Self) -> Bool:
        """Same vertex set (orientation is ignored).

        Args:
            other: The simplex to compare with.

        Returns:
            True if both simplices have the same vertices.
        """
        return self.vertices == other.vertices

    def __ne__(self, other: Self) -> Bool:
        """Different vertex sets (orientation is ignored).

        Args:
            other: The simplex to compare with.

        Returns:
            True if the simplices have different vertices.
        """
        return not self == other

    def __hash__[H: Hasher](self, mut hasher: H):
        """Hashes the vertex list, ignoring the parity.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher to update.
        """
        len(self.vertices).__hash__(hasher)
        for v in self.vertices:
            v.__hash__(hasher)

    def write_to(self, mut writer: Some[Writer]):
        """Writes `simplex([v0, v1, ...],parity=p)`.

        Args:
            writer: The writer to write to.
        """
        writer.write("simplex(", self.vertices, ",parity=", self.parity, ")")
