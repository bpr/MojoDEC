# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Index sets (`mojodec.core.index_set`).
======================================
`IndexSet`: what `CSRMatrix.exclude_subset` and `restrict_to_subset` (and
`DiagonalOperator`'s) need of a set of row indices. `Subset`, a set of
cells of a complex, implements it.
"""


trait IndexSet:
    """A set of indices drawn from `0..universe_size()`."""

    def universe_size(self) -> Int:
        """The number of indices the set is drawn from.

        Returns:
            The size of the index range, members or not.
        """
        ...

    def __contains__(self, i: Int) -> Bool:
        """Whether `i` is in the set.

        Args:
            i: An index.

        Returns:
            True if `i` is a member.
        """
        ...
