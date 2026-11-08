# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Combinatorics and permutation parity (`mojodec.routines.combinatorial`).
========================================================================
Ports of `pydec.math.combinatorial` and `pydec.math.parity`. PyDEC's
generators become functions returning lists, in the same order.
"""


def combinations(L: List[Int], n: Int) -> List[List[Int]]:
    """All combinations of `n` elements of `L`, preserving element order.

    As in PyDEC, `n == 0` or `n > len(L)` yields a single empty
    combination; so does a negative `n`.

    ```mojo
    combinations([1, 2, 3, 4], 2)
    # [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    ```

    Args:
        L: The elements to choose from.
        n: The number of elements in each combination.

    Returns:
        The combinations, in lexicographic order of positions in `L`.
    """
    var out = List[List[Int]]()
    if n <= 0 or n > len(L):
        out.append(List[Int]())
        return out^
    var prefix = List[Int]()
    _combinations(L, n, 0, prefix, out)
    return out^


def permutations(L: List[Int]) -> List[List[Int]]:
    """All permutations of `L`, in PyDEC's order.

    ```mojo
    permutations([1, 2, 3])
    # [[1, 2, 3], [2, 1, 3], [2, 3, 1], [1, 3, 2], [3, 1, 2], [3, 2, 1]]
    ```

    Args:
        L: The elements to permute.

    Returns:
        The `len(L)!` permutations (none when `L` is empty).
    """
    var out = List[List[Int]]()
    if len(L) == 1:
        out.append(L.copy())
    elif len(L) >= 2:
        var head = L[0]
        var tail = List[Int](capacity=len(L) - 1)
        for i in range(1, len(L)):
            tail.append(L[i])
        for p in permutations(tail):
            for i in range(len(p) + 1):
                var q = List[Int](capacity=len(p) + 1)
                for j in range(i):
                    q.append(p[j])
                q.append(head)
                for j in range(i, len(p)):
                    q.append(p[j])
                out.append(q^)
    return out^


def permutation_parity(perm: List[Int], check_input: Bool = True) raises -> Int:
    """Parity of a permutation of `0..N-1`: 0 if even, 1 if odd.

    ```mojo
    permutation_parity([0, 2, 1])  # 1
    permutation_parity([1, 2, 0])  # 0
    ```

    Args:
        perm: A permutation of `0..N-1`.
        check_input: Whether to verify that `perm` is a permutation.

    Returns:
        0 for an even permutation, 1 for an odd one.

    Raises:
        If `check_input` is true and `perm` is not a permutation of
            `0..N-1`.
    """
    var n = len(perm)
    if check_input:
        var seen = List[Bool](length=n, fill=False)
        for p in perm:
            if p < 0 or p >= n or seen[p]:
                raise Error("permutation_parity: invalid input")
            seen[p] = True
    # count disjoint cycles; parity = (n - cycles) mod 2
    var num_cycles = 0
    var visited = List[Bool](length=n, fill=False)
    for i in range(n):
        if visited[i]:
            continue
        num_cycles += 1
        var j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
    return (n - num_cycles) % 2


def relative_parity(A: List[Int], B: List[Int]) raises -> Int:
    """Relative parity of two orderings of the same distinct elements.

    0 if `B` differs from `A` by an even number of transpositions, else 1.

    ```mojo
    relative_parity([0, 1, 2], [0, 2, 1])  # 1
    ```

    Args:
        A: The reference ordering, with distinct elements.
        B: A reordering of `A`.

    Returns:
        0 if the orderings have the same parity, 1 otherwise.

    Raises:
        If `A` has duplicate values or `B` is not a permutation of `A`.
    """
    if len(A) != len(B):
        raise Error("relative_parity: B is not a permutation of A")
    var perm = List[Int](capacity=len(B))
    for b in B:
        var found = -1
        for i in range(len(A)):
            if A[i] == b:
                if found != -1:
                    raise Error("relative_parity: A contains duplicate values")
                found = i
        if found == -1:
            raise Error("relative_parity: B is not a permutation of A")
        perm.append(found)
    return permutation_parity(perm, check_input=True)


def _combinations(
    L: List[Int],
    n: Int,
    start: Int,
    mut prefix: List[Int],
    mut out: List[List[Int]],
):
    if len(prefix) == n:
        out.append(prefix.copy())
        return
    var remaining = n - len(prefix)
    for i in range(start, len(L) - remaining + 1):
        prefix.append(L[i])
        _combinations(L, n, i + 1, prefix, out)
        _ = prefix.pop()
