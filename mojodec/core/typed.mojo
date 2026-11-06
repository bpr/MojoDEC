# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
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
dimension `n` must be one too: `TypedComplex[n]` checks it when built. The
untyped core remains available through `complex()`, `untyped()`, `.mat`,
and `.diag`.
"""

from mojodec.core.cochain import Cochain
from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset
from mojodec.core.utils import IntArray, RealArray, real_vector


comptime PRIMAL = True
"""Primality parameter of primal cochains."""
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
        `(-1)^k sc[n-k].boundary` on dual ones (as `mojodec.d`).

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
                self.sc[Self.n - k].boundary * _sign(k), self.sc
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
            If the complex was built without geometry.
        """
        comptime assert 0 <= k and k <= Self.n, "star: degree out of range"
        if not self.sc.has_geometry():
            raise Error("star: the complex was built without geometry")
        comptime if primal:
            return DiagonalOp[Self.n, k, primal, Self.n - k, not primal](
                self.sc[k].star.copy(), self.sc
            )
        else:
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

    # ===------------------------------------------------------------------=== #
    # Subsets
    # ===------------------------------------------------------------------=== #

    def boundary_subset[k: Int](self) raises -> Subset:
        """The primal k-simplices on the mesh boundary.

        Parameters:
            k: The dimension of the simplices, `0..n`.

        Returns:
            A primal `Subset` of the k-simplices (empty for `k = n`).

        Raises:
            Only if `SimplicialComplex.boundary_subset` fails, which the
            degree check rules out.
        """
        comptime assert 0 <= k and k <= Self.n, "degree out of range"
        return self.sc.boundary_subset(k)


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
        if c.n != Self.n or c.k != Self.k or c.is_primal != Self.primal:
            raise Error("TypedCochain: the cochain is of another kind")
        self.complex = c.complex
        self.v = c.v.copy()

    def untyped(self) -> Cochain:
        """The same cochain as a `Cochain`.

        Returns:
            A `Cochain` with a copy of the values.
        """
        return Cochain(self.complex, Self.k, Self.primal, self.v.copy())

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

    def overwrite(mut self, subset: Subset, other: Self) raises:
        """Copy the values of `other` on the cells of `subset`.

        Args:
            subset: Cells of the same kind as the cochain.
            other: A cochain of the same kind on the same complex.

        Raises:
            If `other` lives on another complex, or `subset` is not of
            k-cells of this primality and size.
        """
        self._check_same(other)
        _check_subset[Self.k, Self.primal](subset, self.v.size)
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


def _check_subset[k: Int, primal: Bool](subset: Subset, size: Int) raises:
    if subset.dim != k or subset.is_primal != primal or subset.size != size:
        raise Error("subset is of other cells than the cochain or operator")


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
](Copyable, Movable, Writable):
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

    def exclude_subset(self, subset: Subset) raises -> Self:
        """The operator with the output rows of `subset` emptied.

        Args:
            subset: Cells of the output kind.

        Returns:
            A copy with the rows of `subset` emptied.

        Raises:
            If `subset` is not of `out_k`-cells of the output primality, one
            per row.
        """
        _check_subset[Self.out_k, Self.out_primal](subset, self.mat.nrows)
        return Self(self.mat.exclude_subset(subset), self.complex)

    def restrict_to_subset(self, subset: Subset) raises -> Self:
        """The operator with only the output rows of `subset` kept.

        Args:
            subset: Cells of the output kind.

        Returns:
            A copy with the rows outside `subset` emptied.

        Raises:
            If `subset` is not of `out_k`-cells of the output primality, one
            per row.
        """
        _check_subset[Self.out_k, Self.out_primal](subset, self.mat.nrows)
        return Self(self.mat.restrict_to_subset(subset), self.complex)

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
](Copyable, Movable, Writable):
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

    def exclude_subset(self, subset: Subset) raises -> Self:
        """The operator with the entries of `subset` zeroed.

        Args:
            subset: Cells of the output kind.

        Returns:
            A copy with the entries of `subset` set to zero.

        Raises:
            If `subset` is not of `out_k`-cells of the output primality, one
            per entry.
        """
        _check_subset[Self.out_k, Self.out_primal](subset, len(self.diag))
        return Self(self.diag.exclude_subset(subset), self.complex)

    def restrict_to_subset(self, subset: Subset) raises -> Self:
        """The operator with the entries outside `subset` zeroed.

        Args:
            subset: Cells of the output kind.

        Returns:
            A copy with the entries outside `subset` set to zero.

        Raises:
            If `subset` is not of `out_k`-cells of the output primality, one
            per entry.
        """
        _check_subset[Self.out_k, Self.out_primal](subset, len(self.diag))
        return Self(self.diag.restrict_to_subset(subset), self.complex)

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
