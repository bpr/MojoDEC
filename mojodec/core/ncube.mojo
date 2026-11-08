# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Oriented n-cubes (`mojodec.core.ncube`).
========================================
Port of `pydec.mesh.ncube.nCube`: an n-cube is given by its `2^n` vertex
indices arranged in an array of shape `(2, ..., 2)`. Its orientation is that
of its corner simplex: the smallest vertex followed by its neighbors along
each axis, with parity flipped when that vertex sits at an odd corner.
"""

from mojodec.core.simplex import Simplex
from mojodec.core.utils import IntArray


struct NCube(Copyable, Equatable, Movable, Writable):
    """An oriented n-cube.

    ```mojo
    var a = NCube(nm.fromstring[DType.int]("[[0,1],[2,3]]"))
    var b = NCube(nm.fromstring[DType.int]("[[2,3],[0,1]]"))
    a.relative_orientation(b)   # True: opposite orientations
    ```
    """

    var indices: IntArray
    """Vertex indices, shape `(2,)*n` (or `(1,)` for a point)."""
    var corner_simplex: Simplex
    """The oriented simplex determining the cube's orientation."""

    def __init__(out self, indices: IntArray) raises:
        """The cube with vertex array `indices`.

        Args:
            indices: Vertex indices, shape `(2,)*n`, or `(1,)` for a point.

        Raises:
            If `indices` has another shape or a repeated vertex.
        """
        var n = indices.ndim
        for d in range(n):
            var len_d = Int(indices.shape.unsafe_get(d))
            if not (len_d == 2 or (n == 1 and len_d == 1)):
                raise Error("NCube: expected shape (1,) or (2,)*n")
        self.indices = indices.copy()
        if n < 2:
            var vs = List[Int]()
            for i in range(indices.size):
                vs.append(Int(indices.unsafe_get(i)))
            self.corner_simplex = Simplex(vs)
            return
        var corner = 0
        for i in range(1, indices.size):
            if indices.unsafe_get(i) < indices.unsafe_get(corner):
                corner = i
        # flat C-order position: bit (n-1-d) is the coordinate along axis d
        var vs = List[Int]()
        vs.append(Int(indices.unsafe_get(corner)))
        var coord_sum = 0
        for d in range(n):
            var bit = 1 << (n - 1 - d)
            vs.append(Int(indices.unsafe_get(corner ^ bit)))
            if corner & bit != 0:
                coord_sum += 1
        self.corner_simplex = Simplex(vs, coord_sum % 2)

    def relative_orientation(self, other: Self) raises -> Bool:
        """Whether two cubes on the same vertices have opposite orientations.

        Args:
            other: A cube on the same vertex set.

        Returns:
            True if the orientations are opposite, False if they agree.

        Raises:
            If the cubes do not have the same vertices.
        """
        if self.corner_simplex != other.corner_simplex:
            raise Error("NCube: cubes do not share the same vertices")
        return (self.corner_simplex.parity ^ other.corner_simplex.parity) == 1

    def __eq__(self, other: Self) -> Bool:
        """Same vertex set (orientation is ignored).

        Args:
            other: The cube to compare with.

        Returns:
            True if both cubes have the same corner simplex vertices.
        """
        return self.corner_simplex == other.corner_simplex

    def __ne__(self, other: Self) -> Bool:
        """Different vertex sets (orientation is ignored).

        Args:
            other: The cube to compare with.

        Returns:
            True if the cubes have different corner simplex vertices.
        """
        return not self == other

    def write_to(self, mut writer: Some[Writer]):
        """Writes `nCube(` followed by the index array and `)`.

        Args:
            writer: The writer to write to.
        """
        writer.write("nCube(", self.indices, ")")
