# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Primal or dual (`mojodec.core.primality`).
==========================================
Whether a cochain or a set of cells lives on the primal complex (on
k-simplices) or on the dual one (on the dual cells of (n-k)-simplices).
PyDEC passes this as a `bool`; `Primality` names it at the call site.

```mojo
var c = sc.get_cochain(1, .DUAL)
var s = Subset.empty(sc, 0, .DUAL)
```
"""


struct Primality(Equatable, ImplicitlyCopyable, Writable):
    """Primal or dual."""

    var _primal: Bool

    comptime PRIMAL = Primality(True)
    """On the primal complex: k-cochains on k-simplices."""
    comptime DUAL = Primality(False)
    """On the dual complex: k-cochains on the duals of (n-k)-simplices."""

    def __init__(out self, primal: Bool):
        """`PRIMAL` if `primal`, else `DUAL`.

        Args:
            primal: Whether it is primal.
        """
        self._primal = primal

    def is_primal(self) -> Bool:
        """Whether this is `PRIMAL`.

        Returns:
            True for `PRIMAL`.
        """
        return self._primal

    def is_dual(self) -> Bool:
        """Whether this is `DUAL`.

        Returns:
            True for `DUAL`.
        """
        return not self._primal

    def dual(self) -> Self:
        """The other kind, as the Hodge star maps it.

        Returns:
            `DUAL` for `PRIMAL`, and `PRIMAL` for `DUAL`.
        """
        return Self(not self._primal)

    def level(self, n: Int, k: Int) -> Int:
        """The level of an n-dimensional complex that holds k-cells of this
        kind: the k-simplices if primal, the (n-k)-simplices if dual.

        Args:
            n: The complex dimension.
            k: The cell (or cochain) dimension.

        Returns:
            `k` if primal, else `n - k`.
        """
        return k if self._primal else n - k

    def __eq__(self, other: Self) -> Bool:
        """Whether both are primal or both dual.

        Args:
            other: The other value.

        Returns:
            True if they are equal.
        """
        return self._primal == other._primal

    def write_to(self, mut writer: Some[Writer]):
        """Write `primal` or `dual`.

        Args:
            writer: The destination.
        """
        writer.write("primal" if self._primal else "dual")
