# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Cochains and the DEC operators (`mojodec.core.cochain`).
========================================================
Port of `pydec.dec.cochain`. PyDEC's `cochain.v` holds either a vector of
values or, for the basis returned by `get_cochain_basis`, a sparse identity
whose columns are the basis cochains; applying operators to a basis composes
their matrices. Mojo's static types split these into two structs.

- `Cochain`: values in a 1-D NuMojo array.
- `CochainBasis`: values in the columns of a `CSRMatrix`.

The operators `d`, `star`, `delta`, `laplace_derham`, and `laplace_beltrami`
are generic over both through the `CochainLike` trait.
"""

import numojo as nm
from numojo.prelude import *

from mojodec.core.simplex import Simplex
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.sparse import CSRMatrix
from mojodec.core.subset import Subset
from mojodec.core.utils import RealArray, real_vector, rows_of


trait CochainLike(Copyable, Deinitable, Movable, Writable):
    """What the DEC operators need from a cochain."""

    def get_complex(self) -> SimplicialComplex:
        """The complex the cochain lives on.

        Returns:
            The (shared) complex.
        """
        ...

    def degree(self) -> Int:
        """The cochain dimension k.

        Returns:
            The dimension k.
        """
        ...

    def complex_dim(self) -> Int:
        """The complex dimension n.

        Returns:
            The dimension n.
        """
        ...

    def primal(self) -> Bool:
        """Whether this is a primal (vs dual) cochain.

        Returns:
            True for a primal cochain, false for a dual one.
        """
        ...

    def value_rows(self) -> Int:
        """Number of rows of the values (one per simplex).

        Returns:
            The number of rows of `v`.
        """
        ...

    def mapped(self, A: CSRMatrix, k: Int, is_primal: Bool) raises -> Self:
        """A cochain of dimension `k` whose values are `A @ self.v`.

        Args:
            A: The operator, with `self.value_rows()` columns.
            k: The dimension of the result.
            is_primal: Whether the result is primal.

        Returns:
            A cochain on the same complex.

        Raises:
            If the shapes of `A` and the values do not match.
        """
        ...

    def mapped(
        self, D: DiagonalOperator, k: Int, is_primal: Bool
    ) raises -> Self:
        """A cochain of dimension `k` whose values are `D @ self.v`.

        Args:
            D: The diagonal operator, of size `self.value_rows()`.
            k: The dimension of the result.
            is_primal: Whether the result is primal.

        Returns:
            A cochain on the same complex.

        Raises:
            If the size of `D` and the values do not match.
        """
        ...

    def relabeled(self, k: Int, is_primal: Bool) -> Self:
        """The same values viewed as a cochain of dimension `k`.

        Args:
            k: The new dimension.
            is_primal: Whether the result is primal.

        Returns:
            A copy of the cochain with the new dimension and kind.
        """
        ...

    def scaled(self, s: Float64) -> Self:
        """The values multiplied by `s`.

        Args:
            s: The scale factor.

        Returns:
            A new cochain.
        """
        ...

    def __add__(self, other: Self) raises -> Self:
        """The sum of two cochains.

        Args:
            other: A cochain of the same dimension on the same complex.

        Returns:
            The sum, with `self`'s dimension and kind.

        Raises:
            If the cochains differ in dimension or complex.
        """
        ...

    def __sub__(self, other: Self) raises -> Self:
        """The difference of two cochains.

        Args:
            other: A cochain of the same dimension on the same complex.

        Returns:
            `self - other`, with `self`'s dimension and kind.

        Raises:
            If the cochains differ in dimension or complex.
        """
        ...


struct Cochain(CochainLike):
    """A cochain with its values in a 1-D NuMojo array.

    ```mojo
    var omega = sc.get_cochain(1)
    omega.v.unsafe_set(0, 1.0)
    var domega = d(omega)
    ```
    """

    var complex: SimplicialComplex
    """The complex the cochain lives on."""
    var k: Int
    """Cochain dimension."""
    var n: Int
    """Complex dimension."""
    var is_primal: Bool
    """Primal or dual."""
    var v: RealArray
    """One value per simplex (primal k-simplex or dual of an (n-k)-simplex)."""

    def __init__(
        out self,
        complex: SimplicialComplex,
        dimension: Int,
        is_primal: Bool,
        var v: RealArray,
    ):
        """Wrap values as a cochain on `complex`.

        Args:
            complex: The complex.
            dimension: The cochain dimension k.
            is_primal: Whether the cochain is primal.
            v: One value per cell. The values are not checked against
                the number of cells.
        """
        self.complex = complex
        self.k = dimension
        self.n = complex.complex_dimension()
        self.is_primal = is_primal
        self.v = v^

    def __getitem__(self, i: Int) -> Float64:
        """The value at index `i` (not bounds checked).

        Args:
            i: A cell index.

        Returns:
            `v[i]`.
        """
        return self.v.unsafe_get(i)

    def __setitem__(mut self, i: Int, value: Float64):
        """Set the value at index `i` (not bounds checked).

        Args:
            i: A cell index.
            value: The new value.
        """
        self.v.unsafe_set(i, value)

    def __getitem__(self, s: Simplex) raises -> Float64:
        """The value on an oriented simplex (negated for opposite orientation).

        For a dual k-cochain, pass the primal (n-k)-simplex dual to the
        cell.

        Args:
            s: A k-simplex of the complex, or an (n-k)-simplex for a dual
                cochain.

        Returns:
            The value on `s`, signed by its orientation.

        Raises:
            If `s` has the wrong dimension or is not a simplex of the
            complex.
        """
        ref data = self.complex[self._level_of(s)]
        var index = data.simplex_to_index(s)
        var value = self.v.unsafe_get(index)
        if s.parity == Int(data.simplex_parity.unsafe_get(index)):
            return value
        return -value

    def __setitem__(mut self, s: Simplex, value: Float64) raises:
        """Set the value on an oriented simplex.

        The value is negated when `s` has the opposite orientation to the
        stored simplex.

        Args:
            s: A k-simplex of the complex, or an (n-k)-simplex for a dual
                cochain.
            value: The value on `s` as oriented.

        Raises:
            If `s` has the wrong dimension or is not a simplex of the
            complex.
        """
        ref data = self.complex[self._level_of(s)]
        var index = data.simplex_to_index(s)
        if s.parity == Int(data.simplex_parity.unsafe_get(index)):
            self.v.unsafe_set(index, value)
        else:
            self.v.unsafe_set(index, -value)

    def overwrite(mut self, subset: Subset, other: Self) raises:
        """Copy the values of `other` on the cells of `subset`, e.g. to
        impose boundary values.

        Both cochains and the subset must be of the same kind of cells.

        Args:
            subset: The cells to overwrite.
            other: The cochain to copy values from.

        Raises:
            If the cochains differ in dimension, primal/dual, complex, or
            number of values, or `subset` is not of the cochain's cells
            (dimension, primal/dual, or size).
        """
        _check_compatible(self, other)
        _check_same_size(self, other, "Cochain.overwrite")
        if (
            subset.dim != self.k
            or subset.is_primal != self.is_primal
            or subset.size != self.v.size
        ):
            raise Error("Cochain.overwrite: subset is of other cells")
        for i in subset.indices():
            self.v.unsafe_set(i, other.v.unsafe_get(i))

    def _level_of(self, s: Simplex) raises -> Int:
        """The level of the complex holding this cochain's cells, checked
        against the dimension of `s`."""
        var level = self.k if self.is_primal else self.n - self.k
        if len(s) - 1 != level:
            raise Error(
                String(
                    "Cochain: a ",
                    len(s) - 1,
                    "-simplex does not index a ",
                    "primal " if self.is_primal else "dual ",
                    self.k,
                    "-cochain (expected a ",
                    level,
                    "-simplex)",
                )
            )
        return level

    def get_complex(self) -> SimplicialComplex:
        """The complex the cochain lives on.

        Returns:
            The (shared) complex.
        """
        return self.complex

    def degree(self) -> Int:
        """The cochain dimension k.

        Returns:
            `self.k`.
        """
        return self.k

    def complex_dim(self) -> Int:
        """The complex dimension n.

        Returns:
            `self.n`.
        """
        return self.n

    def primal(self) -> Bool:
        """Whether this is a primal (vs dual) cochain.

        Returns:
            `self.is_primal`.
        """
        return self.is_primal

    def value_rows(self) -> Int:
        """Number of values (one per cell).

        Returns:
            The length of `v`.
        """
        return self.v.size

    def mapped(self, A: CSRMatrix, k: Int, is_primal: Bool) raises -> Self:
        """A cochain of dimension `k` whose values are `A @ self.v`.

        Args:
            A: The operator, with `self.value_rows()` columns.
            k: The dimension of the result.
            is_primal: Whether the result is primal.

        Returns:
            A cochain on the same complex.

        Raises:
            If `A` has a different number of columns than `v` has rows.
        """
        return Self(self.complex, k, is_primal, A @ self.v)

    def mapped(
        self, D: DiagonalOperator, k: Int, is_primal: Bool
    ) raises -> Self:
        """A cochain of dimension `k` whose values are `D @ self.v`.

        Args:
            D: The diagonal operator, of size `self.value_rows()`.
            k: The dimension of the result.
            is_primal: Whether the result is primal.

        Returns:
            A cochain on the same complex.

        Raises:
            If `D` has a different size than `v` has rows.
        """
        return Self(self.complex, k, is_primal, D @ self.v)

    def relabeled(self, k: Int, is_primal: Bool) -> Self:
        """The same values viewed as a cochain of dimension `k`.

        Args:
            k: The new dimension.
            is_primal: Whether the result is primal.

        Returns:
            A cochain with a copy of the values.
        """
        return Self(self.complex, k, is_primal, self.v.copy())

    def scaled(self, s: Float64) -> Self:
        """The values multiplied by `s`.

        Args:
            s: The scale factor.

        Returns:
            A new cochain.
        """
        var out = self.copy()
        for i in range(out.v.size):
            out.v.unsafe_set(i, s * out.v.unsafe_get(i))
        return out^

    def __add__(self, other: Self) raises -> Self:
        """The sum of two cochains.

        Args:
            other: A cochain of the same dimension and kind on the same
                complex.

        Returns:
            The sum, with `self`'s dimension and kind.

        Raises:
            If the cochains differ in dimension, primal/dual, complex, or
            number of values.
        """
        _check_compatible(self, other)
        _check_same_size(self, other, "Cochain +")
        var out = self.copy()
        for i in range(out.v.size):
            out.v.unsafe_set(i, out.v.unsafe_get(i) + other.v.unsafe_get(i))
        return out^

    def __sub__(self, other: Self) raises -> Self:
        """The difference of two cochains.

        Args:
            other: A cochain of the same dimension and kind on the same
                complex.

        Returns:
            `self - other`, with `self`'s dimension and kind.

        Raises:
            If the cochains differ in dimension, primal/dual, complex, or
            number of values.
        """
        _check_compatible(self, other)
        _check_same_size(self, other, "Cochain -")
        var out = self.copy()
        for i in range(out.v.size):
            out.v.unsafe_set(i, out.v.unsafe_get(i) - other.v.unsafe_get(i))
        return out^

    def write_to(self, mut writer: Some[Writer]):
        """Write the dimensions, kind, and values, as PyDEC prints a cochain.

        Args:
            writer: The destination.
        """
        writer.write(
            "cochain(k=",
            self.k,
            ",n=",
            self.n,
            ",is_primal=",
            self.is_primal,
            "\n",
            self.v,
            ")",
        )


struct CochainBasis(CochainLike):
    """A family of cochains stored as the columns of a sparse matrix.

    Operators applied to `sc.get_cochain_basis(k)` yield their matrices:

    ```mojo
    var L = delta(d(sc.get_cochain_basis(0))).v   # the 0-form Laplacian
    ```
    """

    var complex: SimplicialComplex
    """The complex the cochains live on."""
    var k: Int
    """Cochain dimension."""
    var n: Int
    """Complex dimension."""
    var is_primal: Bool
    """Primal or dual."""
    var v: CSRMatrix
    """One column per cochain, one row per simplex."""

    def __init__(
        out self,
        complex: SimplicialComplex,
        dimension: Int,
        is_primal: Bool,
        var v: CSRMatrix,
    ):
        """Wrap values as a cochain on `complex`.

        Args:
            complex: The complex.
            dimension: The cochain dimension k.
            is_primal: Whether the cochain is primal.
            v: One column per cochain, one row per cell. The values are not checked against
                the number of cells.
        """
        self.complex = complex
        self.k = dimension
        self.n = complex.complex_dimension()
        self.is_primal = is_primal
        self.v = v^

    def get_complex(self) -> SimplicialComplex:
        """The complex the cochain lives on.

        Returns:
            The (shared) complex.
        """
        return self.complex

    def degree(self) -> Int:
        """The cochain dimension k.

        Returns:
            `self.k`.
        """
        return self.k

    def complex_dim(self) -> Int:
        """The complex dimension n.

        Returns:
            `self.n`.
        """
        return self.n

    def primal(self) -> Bool:
        """Whether this is a primal (vs dual) cochain.

        Returns:
            `self.is_primal`.
        """
        return self.is_primal

    def value_rows(self) -> Int:
        """Number of rows of the values (one per cell).

        Returns:
            The number of rows of `v`.
        """
        return self.v.nrows

    def mapped(self, A: CSRMatrix, k: Int, is_primal: Bool) raises -> Self:
        """A cochain of dimension `k` whose values are `A @ self.v`.

        Args:
            A: The operator, with `self.value_rows()` columns.
            k: The dimension of the result.
            is_primal: Whether the result is primal.

        Returns:
            A cochain on the same complex.

        Raises:
            If `A` has a different number of columns than `v` has rows.
        """
        return Self(self.complex, k, is_primal, A @ self.v)

    def mapped(
        self, D: DiagonalOperator, k: Int, is_primal: Bool
    ) raises -> Self:
        """A cochain of dimension `k` whose values are `D @ self.v`.

        Args:
            D: The diagonal operator, of size `self.value_rows()`.
            k: The dimension of the result.
            is_primal: Whether the result is primal.

        Returns:
            A cochain on the same complex.

        Raises:
            If `D` has a different size than `v` has rows.
        """
        return Self(self.complex, k, is_primal, D @ self.v)

    def relabeled(self, k: Int, is_primal: Bool) -> Self:
        """The same values viewed as a cochain of dimension `k`.

        Args:
            k: The new dimension.
            is_primal: Whether the result is primal.

        Returns:
            A cochain with a copy of the values.
        """
        return Self(self.complex, k, is_primal, self.v.copy())

    def scaled(self, s: Float64) -> Self:
        """The values multiplied by `s`.

        Args:
            s: The scale factor.

        Returns:
            A new cochain.
        """
        return Self(self.complex, self.k, self.is_primal, self.v * s)

    def __add__(self, other: Self) raises -> Self:
        """The sum of two cochain bases.

        Args:
            other: A basis of the same dimension on the same complex.

        Returns:
            The sum, with `self`'s dimension and kind.

        Raises:
            If the cochains differ in dimension or complex, or their
            matrices differ in shape.
        """
        _check_compatible(self, other)
        return Self(self.complex, self.k, self.is_primal, self.v + other.v)

    def __sub__(self, other: Self) raises -> Self:
        """The difference of two cochain bases.

        Args:
            other: A basis of the same dimension on the same complex.

        Returns:
            `self - other`, with `self`'s dimension and kind.

        Raises:
            If the cochains differ in dimension or complex, or their
            matrices differ in shape.
        """
        _check_compatible(self, other)
        return Self(self.complex, self.k, self.is_primal, self.v - other.v)

    def write_to(self, mut writer: Some[Writer]):
        """Write the dimensions, kind, and values, as PyDEC prints a cochain.

        Args:
            writer: The destination.
        """
        writer.write(
            "cochain(k=",
            self.k,
            ",n=",
            self.n,
            ",is_primal=",
            self.is_primal,
            "\n",
            self.v,
            ")",
        )


def d[C: CochainLike](f: C) raises -> C:
    """The discrete exterior derivative.

    On primal cochains this is `sc[k].d`; on dual cochains it is the signed
    boundary operator `(-1)^k sc[n-k].boundary`. Out-of-range degrees
    give zero cochains: k = -1 maps to the zero 0-cochain, and k >= n (or
    k < -1) to a single zero value.

    Parameters:
        C: `Cochain` or `CochainBasis`.

    Args:
        f: A k-cochain.

    Returns:
        The (k+1)-cochain `d f`, of the same kind as `f`.

    Raises:
        If the values of `f` do not match the operator's shape.
    """
    var k = f.degree()
    var n = f.complex_dim()
    var sc = f.get_complex()
    if f.primal():
        if k == -1:
            return f.mapped(
                CSRMatrix(sc[0].num_simplices, f.value_rows()), k + 1, True
            )
        if k < -1 or k >= n:
            return f.mapped(CSRMatrix(1, f.value_rows()), k + 1, True)
        return f.mapped(sc[k].d, k + 1, True)
    if k == -1:
        return f.mapped(
            CSRMatrix(sc[n].num_simplices, f.value_rows()), k + 1, False
        )
    if k < -1 or k >= n:
        return f.mapped(CSRMatrix(1, f.value_rows()), k + 1, False)
    var sign: Float64 = -1.0 if k % 2 == 1 else 1.0
    return f.mapped(sc[n - k].boundary * sign, k + 1, False)


def star[C: CochainLike](f: C) raises -> C:
    """The discrete (diagonal, circumcentric) Hodge star.

    Maps primal k-cochains to dual (n-k)-cochains and back. A cochain of
    degree -1 or n+1 (e.g. from `d` or `delta` out of range) is relabeled
    without change.

    Parameters:
        C: `Cochain` or `CochainBasis`.

    Args:
        f: A k-cochain.

    Returns:
        The (n-k)-cochain `*f`, of the other kind.

    Raises:
        If the complex was built without geometry, or the values of `f` do
        not match the star's size.
    """
    var k = f.degree()
    var n = f.complex_dim()
    if k == -1 or k == n + 1:
        return f.relabeled(n - k, not f.primal())
    var sc = f.get_complex()
    if not sc.has_geometry():
        raise Error("star: the complex was built without geometry")
    if f.primal():
        return f.mapped(sc[k].star, n - k, False)
    return f.mapped(sc[n - k].star_inv, n - k, True)


def delta[C: CochainLike](f: C) raises -> C:
    """The discrete codifferential `(-1)^(n(k-1)+1) * d *`.

    Parameters:
        C: `Cochain` or `CochainBasis`.

    Args:
        f: A k-cochain.

    Returns:
        The (k-1)-cochain `delta f`, of the same kind as `f`.

    Raises:
        As `star` and `d` do.
    """
    var sdsf = star(d(star(f)))
    var e = f.complex_dim() * (f.degree() - 1) + 1
    return sdsf.scaled(-1.0 if e % 2 != 0 else 1.0)


def laplace_derham[C: CochainLike](f: C) raises -> C:
    """The discrete Laplace-de Rham operator `d delta + delta d`.

    Parameters:
        C: `Cochain` or `CochainBasis`.

    Args:
        f: A k-cochain.

    Returns:
        The k-cochain `(d delta + delta d) f`.

    Raises:
        As `star` and `d` do.
    """
    return d(delta(f)) + delta(d(f))


def laplace_beltrami[C: CochainLike](f: C) raises -> C:
    """The discrete Laplace-Beltrami operator `delta d`.

    On 0-forms it coincides with the Laplace-de Rham operator.

    Parameters:
        C: `Cochain` or `CochainBasis`.

    Args:
        f: A k-cochain.

    Returns:
        The k-cochain `delta d f`.

    Raises:
        As `star` and `d` do.
    """
    return delta(d(f))


def _check_compatible[C: CochainLike](a: C, b: C) raises:
    """Raise unless `a` and `b` are cochains of one kind on one complex."""
    if (
        a.degree() != b.degree()
        or a.primal() != b.primal()
        or not a.get_complex().is_same(b.get_complex())
    ):
        raise Error(
            "cochain arithmetic: incompatible cochains (degree, primal/dual,"
            " or complex differ)"
        )


def _check_same_size(a: Cochain, b: Cochain, what: String) raises:
    """Raise unless `a` and `b` have as many values as each other."""
    if a.v.size != b.v.size:
        raise Error(String(what, ": ", a.v.size, " values against ", b.v.size))
