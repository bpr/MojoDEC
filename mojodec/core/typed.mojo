# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Typed cochains and operators (`mojodec.core.typed`).
====================================================
A layer over `SimplicialComplex`, `CSRMatrix`, and `DiagonalOperator` that
carries the degree and primality of cochains in their types, so the compiler
checks that operators compose. `TypedCochain[n, k, primal]` is a k-cochain on
an n-complex, primal or dual. `MatrixOp` and `DiagonalOp`, parameterized by
the kinds of cochain they take and give, are a sparse and a diagonal matrix.
`A @ B` compiles only when `B` gives what `A` takes, and `A @ c` only when
`c` is what `A` takes.

```mojo
var tc = TypedComplex[2](vertices, triangles)
# the Laplacian on primal 0-cochains: dual 2 <- dual 1 <- primal 1 <- primal 0
var L = tc.star[2, DUAL]() @ tc.d[1, DUAL]() @ tc.star[1, PRIMAL]() @ tc.d[0, PRIMAL]()
var u = tc.cochain[0, PRIMAL]()
var f = L @ u                      # TypedCochain[2, 0, PRIMAL]
# tc.d[0, PRIMAL]() @ tc.star[1, PRIMAL]()  # does not compile
```

The degree and primality are compile-time parameters, so the complex
dimension `n` must be one too: `TypedComplex[n]` checks it when built. For
a mesh whose dimension is known only at run time, such as one read from a
file, `visit_typed` calls a `TypedComplexVisitor` with the `TypedComplex`
of the right dimension:

```mojo
struct Report(TypedComplexVisitor):
    def visit[n: Int](mut self, tc: TypedComplex[n]) raises:
        print(n, tc.num_cells[n, PRIMAL]())

var r = Report()
visit_typed[min_n=2](SimplicialComplex(read_mesh(path)), r)
```

The untyped core remains available through `complex()`, `untyped()`,
`.mat`, and `.diag`.
"""

from mojodec.core.cochain import Cochain
from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.mesh import SimplicialMesh
from mojodec.core.operator import LinearOperator, Preconditioner
from mojodec.core.primality import Primality
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset
from mojodec.core.utils import IntArray, RealArray, real_vector


comptime PRIMAL = True
"""Primality parameter of primal cochains.

The parameter is a `Bool` rather than a `Primality`: Mojo 1.3 does not fold
`p.dual()` in a parameter, so `star`'s result type would not match `DUAL`.
"""
comptime DUAL = False
"""Primality parameter of dual cochains."""


def _level(n: Int, k: Int, primal: Bool) -> Int:
    """The level of `sc` indexing primal k-simplices or dual k-cells."""
    return k if primal else n - k


def _sign(e: Int) -> Float64:
    """`(-1)^e`."""
    return -1.0 if e % 2 != 0 else 1.0


struct TypedComplex[n: Int](ImplicitlyCopyable, Movable):
    """A `SimplicialComplex` of dimension `n` known at compile time, giving
    typed cochains and operators.

    Parameters:
        n: The dimension of the complex.
    """

    var sc: SimplicialComplex
    """The untyped complex."""

    def __init__(out self, sc: SimplicialComplex) raises:
        """View `sc` as typed.

        Args:
            sc: The complex; it shares its data with `self`.

        Raises:
            If `sc` is not of dimension `n`.
        """
        if sc.complex_dimension() != Self.n:
            raise Error(
                String(
                    "TypedComplex[",
                    Self.n,
                    "]: the complex has dimension ",
                    sc.complex_dimension(),
                )
            )
        self.sc = sc

    def __init__(
        out self,
        vertices: RealArray,
        simplices: IntArray,
        compute_geometry: Bool = True,
    ) raises:
        """Build the complex of the mesh `(vertices, simplices)`.

        Args:
            vertices: `N x embedding_dimension` vertex coordinates.
            simplices: `M x (n+1)` vertex indices of the top simplices.
            compute_geometry: Whether to compute circumcenters, volumes, and
                Hodge stars; `star` and the operators built on it need them.

        Raises:
            If the `SimplicialComplex` cannot be built, or its dimension is
            not `n`.
        """
        self = Self(SimplicialComplex(vertices, simplices, compute_geometry))

    def __init__(
        out self, mesh: SimplicialMesh, compute_geometry: Bool = True
    ) raises:
        """Build the complex of a mesh, e.g. one from `read_mesh` or
        `GmshMesh.mesh()`.

        Args:
            mesh: The vertices and top simplices.
            compute_geometry: Whether to compute circumcenters, volumes, and
                Hodge stars; `star` and the operators built on it need them.

        Raises:
            If the `SimplicialComplex` cannot be built, or its dimension is
            not `n`.
        """
        self = Self(SimplicialComplex(mesh, compute_geometry))

    def complex(self) -> SimplicialComplex:
        """The untyped complex.

        Returns:
            `self.sc`, sharing its data.
        """
        return self.sc

    def num_cells[k: Int, primal: Bool](self) raises -> Int:
        """The number of primal k-simplices or dual k-cells.

        Parameters:
            k: The degree, `0..n`.
            primal: `PRIMAL` or `DUAL`.

        Returns:
            The number of cells, i.e. of values of a k-cochain.

        Raises:
            Only if the level lookup fails, which the degree check rules out.
        """
        comptime assert 0 <= k and k <= Self.n, "degree out of range"
        return self.sc[_level(Self.n, k, primal)].num_simplices

    # ===------------------------------------------------------------------=== #
    # Cochains
    # ===------------------------------------------------------------------=== #

    def cochain[
        k: Int, primal: Bool
    ](self) raises -> TypedCochain[Self.n, k, primal]:
        """The zero k-cochain.

        Parameters:
            k: The degree, `0..n`.
            primal: `PRIMAL` or `DUAL`.

        Returns:
            A k-cochain of zeros.

        Raises:
            Only if the level lookup fails, which the degree check rules out.
        """
        return TypedCochain[Self.n, k, primal](
            self.sc, real_vector(self.num_cells[k, primal]())
        )

    def cochain[
        k: Int, primal: Bool
    ](self, var values: RealArray) raises -> TypedCochain[Self.n, k, primal]:
        """The k-cochain with the given values, one per cell.

        Parameters:
            k: The degree, `0..n`.
            primal: `PRIMAL` or `DUAL`.

        Args:
            values: A vector of `num_cells[k, primal]()` values.

        Returns:
            The k-cochain holding `values`.

        Raises:
            If `values` has the wrong length.
        """
        if values.size != self.num_cells[k, primal]():
            raise Error("TypedComplex.cochain: wrong number of values")
        return TypedCochain[Self.n, k, primal](self.sc, values^)

    # ===------------------------------------------------------------------=== #
    # Operators
    # ===------------------------------------------------------------------=== #

    def d[
        k: Int, primal: Bool
    ](self) raises -> MatrixOp[Self.n, k, primal, k + 1, primal]:
        """The exterior derivative on k-cochains: `sc[k].d` on primal ones,
        `(-1)^(n-k) sc[n-k].boundary` on dual ones (as `mojodec.d`).

        Parameters:
            k: The degree, `0..n-1`.
            primal: `PRIMAL` or `DUAL`.

        Returns:
            The operator from k-cochains to (k+1)-cochains of the same kind.

        Raises:
            Only if the level lookup fails, which the degree check rules out.
        """
        comptime assert 0 <= k and k < Self.n, "d: degree out of range"
        comptime if primal:
            return MatrixOp[Self.n, k, primal, k + 1, primal](
                self.sc[k].d.copy(), self.sc
            )
        else:
            return MatrixOp[Self.n, k, primal, k + 1, primal](
                self.sc[Self.n - k].boundary * _sign(Self.n - k), self.sc
            )

    def star[
        k: Int, primal: Bool
    ](self) raises -> DiagonalOp[Self.n, k, primal, Self.n - k, not primal]:
        """The Hodge star on k-cochains: `sc[k].star` from primal to dual,
        `sc[n-k].star_inv` from dual to primal.

        Parameters:
            k: The degree, `0..n`.
            primal: `PRIMAL` or `DUAL`.

        Returns:
            The diagonal operator from k-cochains to (n-k)-cochains of the
            other kind.

        Raises:
            If the complex was built without geometry, or (from dual to
            primal) a dual cell has zero volume, so the inverse star is
            infinite.
        """
        comptime assert 0 <= k and k <= Self.n, "star: degree out of range"
        if not self.sc.has_geometry():
            raise Error("star: the complex was built without geometry")
        comptime if primal:
            return DiagonalOp[Self.n, k, primal, Self.n - k, not primal](
                self.sc[k].star.copy(), self.sc
            )
        else:
            self.sc[Self.n - k].require_finite_star_inv()
            return DiagonalOp[Self.n, k, primal, Self.n - k, not primal](
                self.sc[Self.n - k].star_inv.copy(), self.sc
            )

    def delta[
        k: Int, primal: Bool
    ](self) raises -> MatrixOp[Self.n, k, primal, k - 1, primal]:
        """The codifferential `(-1)^(n(k-1)+1) * d *` on k-cochains.

        Parameters:
            k: The degree, `1..n`.
            primal: `PRIMAL` or `DUAL`.

        Returns:
            The operator from k-cochains to (k-1)-cochains of the same kind.

        Raises:
            If the complex was built without geometry.
        """
        comptime assert 1 <= k and k <= Self.n, "delta: degree out of range"
        var sds = (
            self.star[Self.n - k + 1, not primal]()
            @ self.d[Self.n - k, not primal]()
            @ self.star[k, primal]()
        )
        return sds * _sign(Self.n * (k - 1) + 1)

    def laplacian[
        k: Int, primal: Bool
    ](self) raises -> MatrixOp[Self.n, k, primal, k, primal]:
        """The Laplace-de Rham operator `d delta + delta d` on k-cochains
        (one term at `k = 0` and at `k = n`).

        Parameters:
            k: The degree, `0..n`; `n` must be positive.
            primal: `PRIMAL` or `DUAL`.

        Returns:
            The operator from k-cochains to k-cochains of the same kind.

        Raises:
            If the complex was built without geometry.
        """
        comptime assert (
            0 <= k and k <= Self.n and Self.n > 0
        ), "laplacian: degree out of range"
        comptime if k == 0:
            return self.delta[k + 1, primal]() @ self.d[k, primal]()
        elif k == Self.n:
            return self.d[k - 1, primal]() @ self.delta[k, primal]()
        else:
            return (self.d[k - 1, primal]() @ self.delta[k, primal]()) + (
                self.delta[k + 1, primal]() @ self.d[k, primal]()
            )

    def laplace_beltrami[
        k: Int, primal: Bool
    ](self) raises -> MatrixOp[Self.n, k, primal, k, primal]:
        """The Laplace-Beltrami operator `delta d` on k-cochains.

        On 0-cochains it equals `laplacian[0, primal]`.

        Parameters:
            k: The degree, `0..n-1`.
            primal: `PRIMAL` or `DUAL`.

        Returns:
            The operator from k-cochains to k-cochains of the same kind.

        Raises:
            If the complex was built without geometry.
        """
        comptime assert (
            0 <= k and k < Self.n
        ), "laplace_beltrami: degree out of range"
        return self.delta[k + 1, primal]() @ self.d[k, primal]()

    def zero[
        in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
    ](self) raises -> MatrixOp[Self.n, in_k, in_primal, out_k, out_primal]:
        """The zero operator between two kinds of cochain, e.g. an empty
        block of a `BlockOp`.

        Parameters:
            in_k: The degree of the cochains it takes, `0..n`.
            in_primal: Whether the cochains it takes are primal.
            out_k: The degree of the cochains it gives, `0..n`.
            out_primal: Whether the cochains it gives are primal.

        Returns:
            An operator with no stored entries.

        Raises:
            Only if the level lookup fails, which the degree check rules out.
        """
        return MatrixOp[Self.n, in_k, in_primal, out_k, out_primal](
            CSRMatrix(
                self.num_cells[out_k, out_primal](),
                self.num_cells[in_k, in_primal](),
            ),
            self.sc,
        )

    def identity[
        k: Int, primal: Bool
    ](self) raises -> DiagonalOp[Self.n, k, primal, k, primal]:
        """The identity on k-cochains.

        Parameters:
            k: The degree, `0..n`.
            primal: `PRIMAL` or `DUAL`.

        Returns:
            The diagonal operator of ones.

        Raises:
            Only if the level lookup fails, which the degree check rules out.
        """
        return DiagonalOp[Self.n, k, primal, k, primal](
            DiagonalOperator.identity(self.num_cells[k, primal]()), self.sc
        )

    def diagonal[
        k: Int, primal: Bool
    ](self, var values: List[Float64]) raises -> DiagonalOp[
        Self.n, k, primal, k, primal
    ]:
        """Multiplication by a value per cell, e.g. a material coefficient.

        Parameters:
            k: The degree, `0..n`.
            primal: `PRIMAL` or `DUAL`.

        Args:
            values: One value per cell, `num_cells[k, primal]()` of them.

        Returns:
            The diagonal operator with `values` on the diagonal.

        Raises:
            If `values` has the wrong length.
        """
        if len(values) != self.num_cells[k, primal]():
            raise Error("TypedComplex.diagonal: wrong number of values")
        return DiagonalOp[Self.n, k, primal, k, primal](
            DiagonalOperator(values^), self.sc
        )

    def dual_diagonal[
        k: Int, primal: Bool
    ](self, var values: List[Float64]) raises -> DiagonalOp[
        Self.n, k, primal, Self.n - k, not primal
    ]:
        """A diagonal operator from k-cochains to the dual (n-k)-cochains on
        the same cells, the kind of a Hodge star: e.g. a star weighted by a
        material coefficient, or (with ones on some cells) the rows that
        impose Dirichlet values in a system whose equations are dual.

        ```mojo
        # rows of the identity on the fixed vertices of d0^T *1 d0
        var ones = List[Float64](length=tc.num_cells[0, PRIMAL](), fill=1.0)
        var I = tc.dual_diagonal[0, PRIMAL](ones^).restrict_to_subset(fixed.dual())
        ```

        Parameters:
            k: The degree, `0..n`.
            primal: `PRIMAL` or `DUAL`.

        Args:
            values: One value per cell, `num_cells[k, primal]()` of them.

        Returns:
            The diagonal operator with `values` on the diagonal.

        Raises:
            If `values` has the wrong length.
        """
        if len(values) != self.num_cells[k, primal]():
            raise Error("TypedComplex.dual_diagonal: wrong number of values")
        return DiagonalOp[Self.n, k, primal, Self.n - k, not primal](
            DiagonalOperator(values^), self.sc
        )

    # ===------------------------------------------------------------------=== #
    # Subsets
    # ===------------------------------------------------------------------=== #

    def boundary_cells[k: Int](self) raises -> TypedSubset[Self.n, k, PRIMAL]:
        """The primal k-simplices on the mesh boundary.

        Parameters:
            k: The dimension of the simplices, `0..n`.

        Returns:
            The k-simplices of `sc.boundary_cells()` (none for `k = n`).

        Raises:
            Only if `SimplicialComplex.boundary_cells` fails, which a valid
            complex rules out.
        """
        comptime assert 0 <= k and k <= Self.n, "degree out of range"
        return TypedSubset[Self.n, k, PRIMAL](
            self.sc, self.sc.boundary_cells()[k].copy()
        )

    def subset[
        k: Int, primal: Bool
    ](self, indices: List[Int]) raises -> TypedSubset[Self.n, k, primal]:
        """The primal k-simplices or dual k-cells with the given indices.

        Parameters:
            k: The cell dimension, `0..n`.
            primal: `PRIMAL` or `DUAL`.

        Args:
            indices: Cell indices, `0 <= i < num_cells[k, primal]()`.

        Returns:
            The subset.

        Raises:
            If an index is out of range.
        """
        comptime assert 0 <= k and k <= Self.n, "degree out of range"
        return TypedSubset[Self.n, k, primal](
            self.sc,
            Subset.from_indices(self.sc, k, indices, Primality(primal)),
        )


trait TypedComplexVisitor:
    """Code generic over the complex dimension, run by `visit_typed` on a
    complex whose dimension is known only at run time.

    ```mojo
    struct Poisson(TypedComplexVisitor):
        var energy: Float64

        def visit[n: Int](mut self, tc: TypedComplex[n]) raises:
            var L = tc.laplacian[0, PRIMAL]()
            ...
    ```
    """

    def visit[n: Int](mut self, tc: TypedComplex[n]) raises:
        """Run on the complex, its dimension `n` now a parameter.

        Parameters:
            n: The complex dimension.

        Args:
            tc: The complex.

        Raises:
            As the implementation does.
        """
        ...


def visit_typed[
    V: TypedComplexVisitor, //, min_n: Int = 1, max_n: Int = 3
](sc: SimplicialComplex, mut visitor: V) raises:
    """Call `visitor.visit[n]` with `TypedComplex[n](sc)`, `n` the dimension
    of `sc`.

    `visit` is compiled for every `n` in `min_n..max_n`, so raise `min_n`
    when it uses operators that need a larger complex (`d[1]` needs
    `n >= 2`, and a `comptime assert` in it fails to compile otherwise).

    Parameters:
        V: The visitor type.
        min_n: The smallest dimension accepted.
        max_n: The largest dimension accepted.

    Args:
        sc: The complex.
        visitor: The code to run; it may keep results in its fields.

    Raises:
        If the dimension of `sc` is outside `min_n..max_n`, or as `visit`
        does.
    """
    comptime assert 0 <= min_n and min_n <= max_n, "visit_typed: empty range"
    var dim = sc.complex_dimension()
    comptime for n in range(min_n, max_n + 1):
        if dim == n:
            visitor.visit[n](TypedComplex[n](sc))
            return
    raise Error(
        String(
            "visit_typed: the complex has dimension ",
            dim,
            ", outside ",
            min_n,
            "..",
            max_n,
        )
    )


struct TypedSubset[n: Int, k: Int, primal: Bool](
    Copyable, Movable, Sized, Writable
):
    """A set of primal k-simplices or dual k-cells of an n-complex, with its
    kind in its type.

    `boundary`, `coboundary`, and `dual` give the kind of their result in
    its type, and the typed operators and cochains take only subsets of
    the cells they act on:

    ```mojo
    var fixed = tc.boundary_cells[0]()        # TypedSubset[2, 0, PRIMAL]
    var A = tc.laplacian[0, PRIMAL]().exclude_subset(fixed)
    # tc.d[0, PRIMAL]().exclude_subset(fixed)  # error: d gives 1-cochains
    ```

    Parameters:
        n: The dimension of the complex.
        k: The dimension of the cells.
        primal: `PRIMAL` for k-simplices, `DUAL` for dual k-cells.
    """

    var cells: Subset
    """The untyped set."""
    var complex: SimplicialComplex
    """The complex the cells belong to."""

    def __init__(out self, sc: SimplicialComplex, var cells: Subset) raises:
        """The typed view of an untyped subset of `sc`.

        Args:
            sc: The complex.
            cells: A subset of `sc`'s cells of this kind.

        Raises:
            If `sc` does not have dimension `n`, or `cells` is not a set of
            `k`-cells of this primality of `sc`.
        """
        comptime assert 0 <= Self.k and Self.k <= Self.n, "degree out of range"
        if sc.complex_dimension() != Self.n:
            raise Error("TypedSubset: the complex has another dimension")
        var level = _level(Self.n, Self.k, Self.primal)
        if (
            cells.n != Self.n
            or cells.dim != Self.k
            or cells.primality().is_primal() != Self.primal
            or cells.size != sc[level].num_simplices
        ):
            raise Error("TypedSubset: the subset is of other cells")
        self.cells = cells^
        self.complex = sc

    @staticmethod
    def empty(sc: SimplicialComplex) raises -> Self:
        """No cells of `sc`.

        Args:
            sc: An n-complex.

        Returns:
            The empty set.

        Raises:
            If `sc` does not have dimension `n`.
        """
        return Self(sc, Subset.empty(sc, Self.k, Primality(Self.primal)))

    @staticmethod
    def full(sc: SimplicialComplex) raises -> Self:
        """Every cell of this kind of `sc`.

        Args:
            sc: An n-complex.

        Returns:
            The full set.

        Raises:
            If `sc` does not have dimension `n`.
        """
        return Self(sc, Subset.full(sc, Self.k, Primality(Self.primal)))

    def untyped(self) -> Subset:
        """The cells as an untyped `Subset`.

        Returns:
            A copy of `cells`.
        """
        return self.cells.copy()

    def __len__(self) -> Int:
        """The number of cells in the set.

        Returns:
            The cell count.
        """
        return len(self.cells)

    def __contains__(self, i: Int) -> Bool:
        """Whether cell `i` is in the set.

        Args:
            i: A cell index.

        Returns:
            True if it is in the set.
        """
        return i in self.cells

    def indices(self) -> List[Int]:
        """The cells' indices, ascending.

        Returns:
            The indices.
        """
        return self.cells.indices()

    def is_empty(self) -> Bool:
        """Whether the set has no cells.

        Returns:
            True if empty.
        """
        return self.cells.is_empty()

    def __or__(self, other: Self) raises -> Self:
        """The union.

        Args:
            other: A set of the same kind on the same complex.

        Returns:
            `self | other`.

        Raises:
            If `other` belongs to another complex.
        """
        self._check_same(other)
        return Self(self.complex, self.cells | other.cells)

    def __and__(self, other: Self) raises -> Self:
        """The intersection.

        Args:
            other: A set of the same kind on the same complex.

        Returns:
            `self & other`.

        Raises:
            If `other` belongs to another complex.
        """
        self._check_same(other)
        return Self(self.complex, self.cells & other.cells)

    def __sub__(self, other: Self) raises -> Self:
        """The difference.

        Args:
            other: A set of the same kind on the same complex.

        Returns:
            The cells of `self` not in `other`.

        Raises:
            If `other` belongs to another complex.
        """
        self._check_same(other)
        return Self(self.complex, self.cells - other.cells)

    def complement(self) raises -> Self:
        """The cells of this kind not in the set.

        Returns:
            The complement.

        Raises:
            Does not raise in practice.
        """
        return Self(self.complex, self.cells.complement())

    def dual(
        self,
    ) raises -> TypedSubset[Self.n, Self.n - Self.k, not Self.primal]:
        """The same cells seen from the other side: primal k-simplices as
        dual (n-k)-cells, and back.

        Returns:
            The subset of the other kind.

        Raises:
            Does not raise in practice.
        """
        return TypedSubset[Self.n, Self.n - Self.k, not Self.primal](
            self.complex, self.cells.dual()
        )

    def boundary(self) raises -> TypedSubset[Self.n, Self.k - 1, Self.primal]:
        """The (k-1)-cells on the boundary of some cell of the set.

        Returns:
            The faces, of the same kind.

        Raises:
            Does not raise in practice.
        """
        comptime assert Self.k >= 1, "boundary of 0-cells"
        return TypedSubset[Self.n, Self.k - 1, Self.primal](
            self.complex, self.cells.boundary(self.complex)
        )

    def coboundary(
        self,
    ) raises -> TypedSubset[Self.n, Self.k + 1, Self.primal]:
        """The (k+1)-cells with some cell of the set on their boundary.

        Returns:
            The cofaces, of the same kind.

        Raises:
            Does not raise in practice.
        """
        comptime assert Self.k < Self.n, "coboundary of n-cells"
        return TypedSubset[Self.n, Self.k + 1, Self.primal](
            self.complex, self.cells.coboundary(self.complex)
        )

    def write_to(self, mut writer: Some[Writer]):
        """Write the type and the cells, e.g. `TypedSubset[2, 0, PRIMAL](...)`.

        Args:
            writer: The destination.
        """
        writer.write(
            "TypedSubset[",
            Self.n,
            ", ",
            Self.k,
            ", ",
            "PRIMAL" if Self.primal else "DUAL",
            "](",
            self.cells,
            ")",
        )

    def _check_same(self, other: Self) raises:
        if not self.complex.is_same(other.complex):
            raise Error("TypedSubset: subsets of different complexes")


struct TypedCochain[n: Int, k: Int, primal: Bool](
    Copyable, Movable, Sized, Writable
):
    """A k-cochain, primal or dual, on an n-complex.

    Parameters:
        n: The dimension of the complex.
        k: The degree of the cochain.
        primal: `PRIMAL` for a cochain on primal k-simplices, `DUAL` for one
            on dual k-cells.
    """

    var complex: SimplicialComplex
    """The complex the cochain lives on."""
    var v: RealArray
    """One value per primal k-simplex or dual k-cell."""

    def __init__(out self, complex: SimplicialComplex, var v: RealArray):
        """A cochain with the given values (sizes are not checked; see
        `TypedComplex.cochain`).

        Args:
            complex: The complex the cochain lives on.
            v: One value per primal k-simplex or dual k-cell.
        """
        self.complex = complex
        self.v = v^

    def __init__(out self, c: Cochain) raises:
        """The typed view of an untyped cochain.

        The values are copied.

        Args:
            c: A `Cochain` on an `n`-complex.

        Raises:
            Unless `c` is a `k`-cochain of the right primality on an
            `n`-complex.
        """
        if (
            c.n != Self.n
            or c.k != Self.k
            or c.primality().is_primal() != Self.primal
        ):
            raise Error("TypedCochain: the cochain is of another kind")
        self.complex = c.complex
        self.v = c.v.copy()

    def untyped(self) -> Cochain:
        """The same cochain as a `Cochain`.

        Returns:
            A `Cochain` with a copy of the values.
        """
        return Cochain(
            self.complex, Self.k, Primality(Self.primal), self.v.copy()
        )

    def __len__(self) -> Int:
        """The number of values.

        Returns:
            The number of primal k-simplices or dual k-cells.
        """
        return self.v.size

    def __getitem__(self, i: Int) -> Float64:
        """The value at index `i` (not bounds-checked).

        Args:
            i: A cell index, `0 <= i < len(self)`.

        Returns:
            The value on cell `i`.
        """
        return self.v.unsafe_get(i)

    def __setitem__(mut self, i: Int, value: Float64):
        """Set the value at index `i` (not bounds-checked).

        Args:
            i: A cell index, `0 <= i < len(self)`.
            value: The new value.
        """
        self.v.unsafe_set(i, value)

    def overwrite(
        mut self,
        subset: TypedSubset[Self.n, Self.k, Self.primal],
        other: Self,
    ) raises:
        """Copy the values of `other` on the cells of `subset`.

        Args:
            subset: Cells of the same kind as the cochain.
            other: A cochain of the same kind on the same complex.

        Raises:
            If `other` or `subset` belongs to another complex.
        """
        self._check_same(other)
        if not subset.complex.is_same(self.complex):
            raise Error("TypedCochain.overwrite: subset of another complex")
        for i in subset.indices():
            self.v.unsafe_set(i, other.v.unsafe_get(i))

    def __add__(self, other: Self) raises -> Self:
        """The sum of two cochains.

        Args:
            other: A cochain of the same kind on the same complex.

        Returns:
            `self + other`.

        Raises:
            If `other` lives on another complex.
        """
        self._check_same(other)
        return Self(self.complex, self.v + other.v)

    def __sub__(self, other: Self) raises -> Self:
        """The difference of two cochains.

        Args:
            other: A cochain of the same kind on the same complex.

        Returns:
            `self - other`.

        Raises:
            If `other` lives on another complex.
        """
        self._check_same(other)
        return Self(self.complex, self.v - other.v)

    def __mul__(self, s: Float64) raises -> Self:
        """The cochain scaled by `s`.

        Args:
            s: The scale factor.

        Returns:
            `s * self`.

        Raises:
            Only if the NuMojo array arithmetic fails.
        """
        return Self(self.complex, self.v * s)

    def __rmul__(self, s: Float64) raises -> Self:
        """The cochain scaled by `s`, for `s * c`.

        Args:
            s: The scale factor.

        Returns:
            `s * self`.

        Raises:
            Only if the NuMojo array arithmetic fails.
        """
        return self * s

    def __neg__(self) raises -> Self:
        """The negated cochain.

        Returns:
            `-self`.

        Raises:
            Only if the NuMojo array arithmetic fails.
        """
        return self * -1.0

    def write_to(self, mut writer: Some[Writer]):
        """Write the type and values, e.g. `TypedCochain[2, 0, PRIMAL](...)`.

        Args:
            writer: The destination.
        """
        writer.write(
            "TypedCochain[",
            Self.n,
            ", ",
            Self.k,
            ", ",
            "PRIMAL" if Self.primal else "DUAL",
            "](",
            self.v,
            ")",
        )

    def _check_same(self, other: Self) raises:
        if not self.complex.is_same(other.complex):
            raise Error("TypedCochain: cochains on different complexes")


def _join(
    a: Optional[SimplicialComplex], b: Optional[SimplicialComplex]
) raises -> Optional[SimplicialComplex]:
    """The complex of an operator combining operators on `a` and `b`.

    An operator wrapped from a bare matrix has no complex and combines with
    any other.
    """
    if not a:
        return b
    if b and not a.value().is_same(b.value()):
        raise Error("typed operators: operands on different complexes")
    return a


struct MatrixOp[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](Copyable, LinearOperator, Movable, Writable):
    """A sparse operator from `in_k`-cochains to `out_k`-cochains.

    Parameters:
        n: The dimension of the complex.
        in_k: The degree of the cochains it takes.
        in_primal: Whether the cochains it takes are primal.
        out_k: The degree of the cochains it gives.
        out_primal: Whether the cochains it gives are primal.
    """

    var mat: CSRMatrix
    """The matrix: one row per output cell, one column per input cell."""
    var complex: Optional[SimplicialComplex]
    """The complex the operator was built on, or `None` for one wrapped from
    a bare matrix."""

    def __init__(
        out self,
        var mat: CSRMatrix,
        complex: Optional[SimplicialComplex] = None,
    ):
        """Wrap a matrix (its shape is not checked).

        Args:
            mat: The matrix, `N_out x N_in`.
            complex: The complex the operator acts on, if any. When given,
                the operator refuses cochains and operators on other
                complexes.
        """
        self.mat = mat^
        self.complex = complex

    def num_rows(self) -> Int:
        """The number of output cells, for `LinearOperator`.

        Returns:
            The number of rows of `mat`.
        """
        return self.mat.num_rows()

    def num_cols(self) -> Int:
        """The number of input cells, for `LinearOperator`.

        Returns:
            The number of columns of `mat`.
        """
        return self.mat.num_cols()

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """`mat` applied to raw values, for `LinearOperator` (the
        untyped solvers); `@` applies it to cochains.

        Args:
            x: One value per input cell.
            y: One value per output cell, overwritten.

        Raises:
            If a length does not match.
        """
        self.mat.apply_into(x, y)

    def __matmul__(
        self, c: TypedCochain[Self.n, Self.in_k, Self.in_primal]
    ) raises -> TypedCochain[Self.n, Self.out_k, Self.out_primal]:
        """Apply the operator.

        Args:
            c: A cochain of the input kind.

        Returns:
            The output cochain, on `c`'s complex.

        Raises:
            If `c` is on another complex than the operator, or has not
            as many values as `mat` has columns.
        """
        _ = _join(self.complex, c.complex)
        return TypedCochain[Self.n, Self.out_k, Self.out_primal](
            c.complex, self.mat @ c.v
        )

    def __matmul__[
        k: Int, primal: Bool
    ](
        self, other: MatrixOp[Self.n, k, primal, Self.in_k, Self.in_primal]
    ) raises -> MatrixOp[Self.n, k, primal, Self.out_k, Self.out_primal]:
        """The composition: `other`, then `self`.

        Parameters:
            k: The degree of the cochains `other` takes.
            primal: Whether the cochains `other` takes are primal.

        Args:
            other: An operator giving the cochains `self` takes.

        Returns:
            The product `self.mat @ other.mat`.

        Raises:
            If the operators are on different complexes, or the matrix
            shapes do not match.
        """
        return MatrixOp[Self.n, k, primal, Self.out_k, Self.out_primal](
            self.mat @ other.mat, _join(self.complex, other.complex)
        )

    def __matmul__[
        k: Int, primal: Bool
    ](
        self, other: DiagonalOp[Self.n, k, primal, Self.in_k, Self.in_primal]
    ) raises -> MatrixOp[Self.n, k, primal, Self.out_k, Self.out_primal]:
        """The composition with a diagonal operator: a column scaling.

        Parameters:
            k: The degree of the cochains `other` takes.
            primal: Whether the cochains `other` takes are primal.

        Args:
            other: A diagonal operator giving the cochains `self` takes.

        Returns:
            `self.mat` with column `j` scaled by `other`'s `j`-th entry.

        Raises:
            If the operators are on different complexes, or the sizes do not
            match.
        """
        return MatrixOp[Self.n, k, primal, Self.out_k, Self.out_primal](
            self.mat @ other.diag, _join(self.complex, other.complex)
        )

    def __add__(self, other: Self) raises -> Self:
        """The sum of two operators of the same kind.

        Args:
            other: The other operator.

        Returns:
            `self + other`.

        Raises:
            If the operators are on different complexes, or the matrix shapes
            differ.
        """
        return Self(self.mat + other.mat, _join(self.complex, other.complex))

    def __add__(
        self,
        other: DiagonalOp[
            Self.n, Self.in_k, Self.in_primal, Self.out_k, Self.out_primal
        ],
    ) raises -> Self:
        """The sum with a diagonal operator of the same kind.

        Args:
            other: The diagonal operator.

        Returns:
            `self + other`, sparse.

        Raises:
            If the operators are on different complexes, or the shapes differ.
        """
        return Self(
            self.mat + other.diag.to_csr(), _join(self.complex, other.complex)
        )

    def __sub__(self, other: Self) raises -> Self:
        """The difference of two operators of the same kind.

        Args:
            other: The other operator.

        Returns:
            `self - other`.

        Raises:
            If the operators are on different complexes, or the matrix shapes
            differ.
        """
        return Self(self.mat - other.mat, _join(self.complex, other.complex))

    def __sub__(
        self,
        other: DiagonalOp[
            Self.n, Self.in_k, Self.in_primal, Self.out_k, Self.out_primal
        ],
    ) raises -> Self:
        """The difference with a diagonal operator of the same kind.

        Args:
            other: The diagonal operator.

        Returns:
            `self - other`, sparse.

        Raises:
            If the operators are on different complexes, or the shapes differ.
        """
        return Self(
            self.mat - other.diag.to_csr(), _join(self.complex, other.complex)
        )

    def __mul__(self, s: Float64) -> Self:
        """The operator scaled by `s`.

        Args:
            s: The scale factor.

        Returns:
            `s * self`.
        """
        return Self(self.mat * s, self.complex)

    def __rmul__(self, s: Float64) -> Self:
        """The operator scaled by `s`, for `s * A`.

        Args:
            s: The scale factor.

        Returns:
            `s * self`.
        """
        return self * s

    def __neg__(self) -> Self:
        """The negated operator.

        Returns:
            `-self`.
        """
        return self * -1.0

    def transpose(
        self,
    ) raises -> MatrixOp[
        Self.n,
        Self.n - Self.out_k,
        not Self.out_primal,
        Self.n - Self.in_k,
        not Self.in_primal,
    ]:
        """The transpose, the adjoint under the pairing of primal k-cochains
        with dual (n-k)-cochains (one value per cell on each side).

        An operator from k-cochains to l-cochains transposes to one from
        the duals of l-cochains to the duals of k-cochains: `d[k, PRIMAL]`
        to `(-1)^(k+1) d[n-k-1, DUAL]`, so
        `tc.d[0, PRIMAL]().T() @ tc.star[1, PRIMAL]() @ tc.d[0, PRIMAL]()`
        is the positive semidefinite stiffness matrix from primal
        0-cochains to dual n-cochains.

        Returns:
            The operator with the transposed matrix.

        Raises:
            Does not raise in practice.
        """
        return MatrixOp[
            Self.n,
            Self.n - Self.out_k,
            not Self.out_primal,
            Self.n - Self.in_k,
            not Self.in_primal,
        ](self.mat.transpose(), self.complex)

    def T(
        self,
    ) raises -> MatrixOp[
        Self.n,
        Self.n - Self.out_k,
        not Self.out_primal,
        Self.n - Self.in_k,
        not Self.in_primal,
    ]:
        """The transpose (alias of `transpose`).

        Returns:
            The operator with the transposed matrix.

        Raises:
            Does not raise in practice.
        """
        return self.transpose()

    def exclude_subset(
        self, subset: TypedSubset[Self.n, Self.out_k, Self.out_primal]
    ) raises -> Self:
        """The operator with the output rows (entries, for a diagonal) of
        `subset` emptied.

        Args:
            subset: Cells of the output kind.

        Returns:
            A copy with the rows of `subset` emptied.

        Raises:
            If `subset` belongs to another complex, or (for an operator
            wrapped from a bare matrix) has another number of cells.
        """
        var complex = _join(self.complex, subset.complex)
        return Self(self.mat.exclude_subset(subset.cells), complex)

    def restrict_to_subset(
        self, subset: TypedSubset[Self.n, Self.out_k, Self.out_primal]
    ) raises -> Self:
        """The operator with only the output rows (entries, for a diagonal)
        of `subset` kept.

        Args:
            subset: Cells of the output kind.

        Returns:
            A copy with the rows outside `subset` emptied.

        Raises:
            If `subset` belongs to another complex, or (for an operator
            wrapped from a bare matrix) has another number of cells.
        """
        var complex = _join(self.complex, subset.complex)
        return Self(self.mat.restrict_to_subset(subset.cells), complex)

    def write_to(self, mut writer: Some[Writer]):
        """Write the kind, e.g. `MatrixOp[2]: primal 0 -> primal 1`, then the
        matrix on the following lines.

        Args:
            writer: The destination.
        """
        writer.write(
            _op_name(
                "MatrixOp",
                Self.n,
                Self.in_k,
                Self.in_primal,
                Self.out_k,
                Self.out_primal,
            ),
            "\n",
            self.mat,
        )


struct DiagonalOp[
    n: Int, in_k: Int, in_primal: Bool, out_k: Int, out_primal: Bool
](Copyable, Movable, Preconditioner, Writable):
    """A diagonal operator from `in_k`-cochains to `out_k`-cochains with as
    many cells, such as a Hodge star.

    Parameters:
        n: The dimension of the complex.
        in_k: The degree of the cochains it takes.
        in_primal: Whether the cochains it takes are primal.
        out_k: The degree of the cochains it gives.
        out_primal: Whether the cochains it gives are primal.
    """

    var diag: DiagonalOperator
    """The diagonal matrix."""
    var complex: Optional[SimplicialComplex]
    """The complex the operator was built on, or `None` for one wrapped from
    a bare diagonal."""

    def __init__(
        out self,
        var diag: DiagonalOperator,
        complex: Optional[SimplicialComplex] = None,
    ):
        """Wrap a diagonal matrix (its size is not checked).

        Args:
            diag: The diagonal, one entry per cell.
            complex: The complex the operator acts on, if any. When given,
                the operator refuses cochains and operators on other
                complexes.
        """
        self.diag = diag^
        self.complex = complex

    def num_rows(self) -> Int:
        """The number of output cells, for `LinearOperator`.

        Returns:
            The number of rows of `diag`.
        """
        return self.diag.num_rows()

    def num_cols(self) -> Int:
        """The number of input cells, for `LinearOperator`.

        Returns:
            The number of columns of `diag`.
        """
        return self.diag.num_cols()

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        """`diag` applied to raw values, for `LinearOperator` (the
        untyped solvers); `@` applies it to cochains.

        Args:
            x: One value per input cell.
            y: One value per output cell, overwritten.

        Raises:
            If a length does not match.
        """
        self.diag.apply_into(x, y)

    def __matmul__(
        self, c: TypedCochain[Self.n, Self.in_k, Self.in_primal]
    ) raises -> TypedCochain[Self.n, Self.out_k, Self.out_primal]:
        """Apply the operator.

        Args:
            c: A cochain of the input kind.

        Returns:
            The output cochain, on `c`'s complex.

        Raises:
            If `c` is on another complex than the operator, or has not
            as many values as the diagonal.
        """
        _ = _join(self.complex, c.complex)
        return TypedCochain[Self.n, Self.out_k, Self.out_primal](
            c.complex, self.diag @ c.v
        )

    def __matmul__[
        k: Int, primal: Bool
    ](
        self, other: DiagonalOp[Self.n, k, primal, Self.in_k, Self.in_primal]
    ) raises -> DiagonalOp[Self.n, k, primal, Self.out_k, Self.out_primal]:
        """The composition, itself diagonal.

        Parameters:
            k: The degree of the cochains `other` takes.
            primal: Whether the cochains `other` takes are primal.

        Args:
            other: A diagonal operator giving the cochains `self` takes.

        Returns:
            The entrywise product of the diagonals.

        Raises:
            If the operators are on different complexes, or the sizes differ.
        """
        return DiagonalOp[Self.n, k, primal, Self.out_k, Self.out_primal](
            self.diag @ other.diag, _join(self.complex, other.complex)
        )

    def __matmul__[
        k: Int, primal: Bool
    ](
        self, other: MatrixOp[Self.n, k, primal, Self.in_k, Self.in_primal]
    ) raises -> MatrixOp[Self.n, k, primal, Self.out_k, Self.out_primal]:
        """The composition with a sparse operator: a row scaling.

        Parameters:
            k: The degree of the cochains `other` takes.
            primal: Whether the cochains `other` takes are primal.

        Args:
            other: A sparse operator giving the cochains `self` takes.

        Returns:
            `other.mat` with row `i` scaled by the `i`-th entry of `self`.

        Raises:
            If the operators are on different complexes, or the sizes do not
            match.
        """
        return MatrixOp[Self.n, k, primal, Self.out_k, Self.out_primal](
            self.diag @ other.mat, _join(self.complex, other.complex)
        )

    def inverse(
        self,
    ) raises -> DiagonalOp[
        Self.n, Self.out_k, Self.out_primal, Self.in_k, Self.in_primal
    ]:
        """The inverse, from the output kind back to the input kind. For the
        Hodge star this differs from the star on the output kind by the
        sign `(-1)^(k(n-k))`.

        Returns:
            The diagonal operator of reciprocals.

        Raises:
            If an entry is zero.
        """
        return DiagonalOp[
            Self.n, Self.out_k, Self.out_primal, Self.in_k, Self.in_primal
        ](self.diag.inverse(), self.complex)

    def transpose(
        self,
    ) -> DiagonalOp[
        Self.n,
        Self.n - Self.out_k,
        not Self.out_primal,
        Self.n - Self.in_k,
        not Self.in_primal,
    ]:
        """The transpose: the same diagonal between the dual kinds, as for
        `MatrixOp.transpose`. A Hodge star from primal k-cochains keeps its
        kind; one from dual k-cochains does too.

        Returns:
            The operator with the same diagonal.
        """
        return DiagonalOp[
            Self.n,
            Self.n - Self.out_k,
            not Self.out_primal,
            Self.n - Self.in_k,
            not Self.in_primal,
        ](self.diag.copy(), self.complex)

    def T(
        self,
    ) -> DiagonalOp[
        Self.n,
        Self.n - Self.out_k,
        not Self.out_primal,
        Self.n - Self.in_k,
        not Self.in_primal,
    ]:
        """The transpose (alias of `transpose`).

        Returns:
            The operator with the same diagonal.
        """
        return self.transpose()

    def to_matrix_op(
        self,
    ) raises -> MatrixOp[
        Self.n, Self.in_k, Self.in_primal, Self.out_k, Self.out_primal
    ]:
        """The same operator as a sparse matrix.

        Returns:
            A `MatrixOp` of the same kind.

        Raises:
            Only if building the `CSRMatrix` fails.
        """
        return MatrixOp[
            Self.n, Self.in_k, Self.in_primal, Self.out_k, Self.out_primal
        ](self.diag.to_csr(), self.complex)

    def inner_product(
        self,
        a: TypedCochain[Self.n, Self.in_k, Self.in_primal],
        b: TypedCochain[Self.n, Self.in_k, Self.in_primal],
    ) raises -> Float64:
        """`a^T D b`: for the primal star, the discrete L2 inner product.

        Args:
            a: A cochain of the input kind.
            b: A cochain of the input kind.

        Returns:
            The weighted inner product.

        Raises:
            If `a` or `b` is on another complex than the operator, or has
            not as many values as the diagonal.
        """
        _ = _join(_join(self.complex, a.complex), b.complex)
        return self.diag.inner_product(a.v, b.v)

    def norm(
        self, a: TypedCochain[Self.n, Self.in_k, Self.in_primal]
    ) raises -> Float64:
        """`sqrt(a^T D a)`.

        Args:
            a: A cochain of the input kind.

        Returns:
            The weighted norm.

        Raises:
            If `a` is on another complex than the operator, or has not as
            many values as the diagonal.
        """
        _ = _join(self.complex, a.complex)
        return self.diag.norm(a.v)

    def __add__(self, other: Self) raises -> Self:
        """The sum of two diagonal operators of the same kind.

        Args:
            other: The other operator.

        Returns:
            `self + other`.

        Raises:
            If the operators are on different complexes, or the sizes differ.
        """
        return Self(self.diag + other.diag, _join(self.complex, other.complex))

    def __sub__(self, other: Self) raises -> Self:
        """The difference of two diagonal operators of the same kind.

        Args:
            other: The other operator.

        Returns:
            `self - other`.

        Raises:
            If the operators are on different complexes, or the sizes differ.
        """
        return Self(self.diag - other.diag, _join(self.complex, other.complex))

    def __mul__(self, s: Float64) -> Self:
        """The operator scaled by `s`.

        Args:
            s: The scale factor.

        Returns:
            `s * self`.
        """
        return Self(self.diag * s, self.complex)

    def __rmul__(self, s: Float64) -> Self:
        """The operator scaled by `s`, for `s * D`.

        Args:
            s: The scale factor.

        Returns:
            `s * self`.
        """
        return self * s

    def __neg__(self) -> Self:
        """The negated operator.

        Returns:
            `-self`.
        """
        return self * -1.0

    def exclude_subset(
        self, subset: TypedSubset[Self.n, Self.out_k, Self.out_primal]
    ) raises -> Self:
        """The operator with the output rows (entries, for a diagonal) of
        `subset` emptied.

        Args:
            subset: Cells of the output kind.

        Returns:
            A copy with the rows of `subset` emptied.

        Raises:
            If `subset` belongs to another complex, or (for an operator
            wrapped from a bare diagonal) has another number of cells.
        """
        var complex = _join(self.complex, subset.complex)
        return Self(self.diag.exclude_subset(subset.cells), complex)

    def restrict_to_subset(
        self, subset: TypedSubset[Self.n, Self.out_k, Self.out_primal]
    ) raises -> Self:
        """The operator with only the output rows (entries, for a diagonal)
        of `subset` kept.

        Args:
            subset: Cells of the output kind.

        Returns:
            A copy with the rows outside `subset` emptied.

        Raises:
            If `subset` belongs to another complex, or (for an operator
            wrapped from a bare diagonal) has another number of cells.
        """
        var complex = _join(self.complex, subset.complex)
        return Self(self.diag.restrict_to_subset(subset.cells), complex)

    def write_to(self, mut writer: Some[Writer]):
        """Write the kind, e.g. `DiagonalOp[2]: primal 1 -> dual 1`, then the
        diagonal on the following line.

        Args:
            writer: The destination.
        """
        writer.write(
            _op_name(
                "DiagonalOp",
                Self.n,
                Self.in_k,
                Self.in_primal,
                Self.out_k,
                Self.out_primal,
            ),
            "\n",
            self.diag,
        )


def _op_name(
    name: String,
    n: Int,
    in_k: Int,
    in_primal: Bool,
    out_k: Int,
    out_primal: Bool,
) -> String:
    return String(
        name,
        "[",
        n,
        "]: ",
        "primal " if in_primal else "dual ",
        in_k,
        " -> ",
        "primal " if out_primal else "dual ",
        out_k,
    )
