# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
K-d trees (`mojodec.core.kd_tree`).
===================================
Port of `pydec.math.kd_tree`: a k-d tree over the rows of a point array,
answering nearest-neighbor and fixed-radius queries. Each point carries an
integer value, by default its row index.
"""

from std.math import sqrt
from mojodec.core.utils import RealArray, rows_of, cols_of


struct KDTree(Copyable, Movable, Writable):
    """A k-d tree over `N` points in `k` dimensions.

    ```mojo
    var kd = KDTree(nm.fromstring[f64]("[[0,0],[1,0],[0,1],[1,1]]"))
    kd.nearest([2.0, 0.0])            # 1
    kd.nearest_n([2.0, 0.0], 2)       # [1, 3]
    kd.in_sphere([0.1, 0.2], 1.1)     # [0, 2, 1]
    ```
    """

    var k: Int
    """Dimension of the points."""
    var num_points: Int
    """Number of points."""
    var _points: List[List[Float64]]
    var _values: List[Int]
    # node i stores point _node_point[i] split on _node_axis[i]; -1 is no child
    var _node_point: List[Int]
    var _node_axis: List[Int]
    var _left: List[Int]
    var _right: List[Int]
    var _root: Int

    def __init__(out self, points: RealArray) raises:
        """A tree over the rows of `points`, valued by row index.

        Args:
            points: `N x k` point coordinates.

        Raises:
            If the points have no coordinates.
        """
        var values = List[Int](capacity=rows_of(points))
        for i in range(rows_of(points)):
            values.append(i)
        self = Self(points, values)

    def __init__(out self, points: RealArray, values: List[Int]) raises:
        """A tree over the rows of `points`, with one value per point.

        Args:
            points: `N x k` point coordinates.
            values: `N` values, returned by queries in place of row indices.

        Raises:
            If `values` does not have one entry per row of `points`, or the
            points have no coordinates.
        """
        if len(values) != rows_of(points):
            raise Error("KDTree: points and values must have the same lengths")
        if cols_of(points) == 0 and rows_of(points) > 0:
            raise Error("KDTree: points need at least one coordinate")
        self.k = cols_of(points)
        self.num_points = rows_of(points)
        self._points = List[List[Float64]](capacity=self.num_points)
        for i in range(self.num_points):
            var p = List[Float64](capacity=self.k)
            for j in range(self.k):
                p.append(points.unsafe_get(i * self.k + j))
            self._points.append(p^)
        self._values = values.copy()
        self._node_point = List[Int]()
        self._node_axis = List[Int]()
        self._left = List[Int]()
        self._right = List[Int]()
        var ids = List[Int](capacity=self.num_points)
        for i in range(self.num_points):
            ids.append(i)
        self._root = -1
        self._root = self._build(ids^, 0)

    def nearest(
        self, point: List[Float64], max_dist: Float64 = Float64.MAX
    ) -> Optional[Int]:
        """The value of the nearest point, or `None` if none is within `max_dist`.

        Args:
            point: The query point, `k` coordinates.
            max_dist: Only points strictly closer than this are considered.

        Returns:
            The nearest point's value, or `None`.
        """
        var x = self.nearest_n(point, 1, max_dist)
        if len(x) == 0:
            return None
        return x[0]

    def in_sphere(
        self, point: List[Float64], radius: Float64, max_points: Int = Int.MAX
    ) -> List[Int]:
        """Values of all points strictly closer than `radius`, nearest first.

        Args:
            point: The center of the sphere, `k` coordinates.
            radius: The sphere's radius.
            max_points: Return at most this many values.

        Returns:
            The values of the points in the sphere, nearest first.
        """
        return self.nearest_n(point, max_points, radius)

    def nearest_n(
        self, point: List[Float64], n: Int, max_dist: Float64 = Float64.MAX
    ) -> List[Int]:
        """Values of the (at most) `n` nearest points within `max_dist`, nearest first.

        Args:
            point: The query point, `k` coordinates.
            n: The maximum number of values to return (none if `n <= 0`).
            max_dist: Only points strictly closer than this are considered.

        Returns:
            Up to `n` values, nearest first.
        """
        var dists = List[Float64]()
        var ids = List[Int]()
        if n <= 0:
            return List[Int]()
        _ = self._nearest_n(point, n, max_dist, self._root, dists, ids)
        var out = List[Int](capacity=len(ids))
        for i in ids:
            out.append(self._values[i])
        return out^

    def write_to(self, mut writer: Some[Writer]):
        """Writes `kd_tree< N points in k-dimensions >`.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "kd_tree< ", self.num_points, " points in ", self.k, "-dimensions >"
        )

    def _build(mut self, var ids: List[Int], depth: Int) -> Int:
        if len(ids) == 0:
            return -1
        var axis = depth % self.k
        var keys = List[Float64](capacity=len(ids))
        for i in ids:
            keys.append(self._points[i][axis])
        ids = _sorted_by(ids, keys)
        var mid = len(ids) // 2
        var node = len(self._node_point)
        self._node_point.append(ids[mid])
        self._node_axis.append(axis)
        self._left.append(-1)
        self._right.append(-1)
        var lo = List[Int](capacity=mid)
        var hi = List[Int](capacity=len(ids) - mid - 1)
        for i in range(mid):
            lo.append(ids[i])
        for i in range(mid + 1, len(ids)):
            hi.append(ids[i])
        var l = self._build(lo^, depth + 1)
        var r = self._build(hi^, depth + 1)
        self._left[node] = l
        self._right[node] = r
        return node

    def _nearest_n(
        self,
        point: List[Float64],
        n: Int,
        max_dist_in: Float64,
        current: Int,
        mut dists: List[Float64],
        mut ids: List[Int],
    ) -> Float64:
        var max_dist = max_dist_in
        if current < 0:
            return max_dist
        var p = self._node_point[current]
        var axis = self._node_axis[current]
        var pt_dist: Float64 = 0
        for j in range(self.k):
            var t = point[j] - self._points[p][j]
            pt_dist += t * t
        pt_dist = sqrt(pt_dist)
        var sep_dist = point[axis] - self._points[p][axis]

        if pt_dist < max_dist:
            # keep (dists, ids) sorted by distance, at most n long
            var pos = len(dists)
            while pos > 0 and dists[pos - 1] > pt_dist:
                pos -= 1
            dists.insert(pos, pt_dist)
            ids.insert(pos, p)
            if len(dists) > n:
                _ = dists.pop()
                _ = ids.pop()
            if len(dists) == n:
                max_dist = min(dists[len(dists) - 1], max_dist)

        var near = self._left[current] if sep_dist < 0 else self._right[current]
        var far = self._right[current] if sep_dist < 0 else self._left[current]
        max_dist = self._nearest_n(point, n, max_dist, near, dists, ids)
        if abs(sep_dist) < max_dist:
            return self._nearest_n(point, n, max_dist, far, dists, ids)
        return max_dist


def _sorted_by(ids: List[Int], keys: List[Float64]) -> List[Int]:
    """`ids` stably sorted by the matching `keys`."""
    var order = List[Int](capacity=len(ids))
    for i in range(len(ids)):
        order.append(i)

    def key_less(a: Int, b: Int) {imm keys} -> Bool:
        return keys[a] < keys[b]

    sort[stable=True](Span(order), key_less)
    var out = List[Int](capacity=len(ids))
    for i in order:
        out.append(ids[i])
    return out^
