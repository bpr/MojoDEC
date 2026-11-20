# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Sparse symmetric eigenproblems on cochains (`mojodec.routines.lanczos`).
=======================================================================
`eigsh` on typed operators, giving k-cochains as eigenvectors. The
untyped `eigsh` is `molina.lanczos`'s, which documents the method; its
overloads are defined here too, forwarding to it, because
Mojo does not merge overloads of one name imported from two modules.
"""

from molina.diagonal import DiagonalOperator
from mojodec.core.simplicial_complex import SimplicialComplex
from molina.sparse import CSRMatrix
from mojodec.core.typed import MatrixOp
from molina import lanczos as molina_lanczos
from molina.dense import EighResult
from mojodec.routines.linalg import TypedEighResult


def eigsh(
    K: CSRMatrix,
    M: CSRMatrix,
    nev: Int,
    *,
    sigma: Float64,
    tol: Float64 = 1e-12,
    ncv: Optional[Int] = None,
    max_restarts: Int = 300,
) raises -> EighResult:
    """The `nev` eigenpairs of `K x = lambda M x` nearest `sigma`:
    `molina.lanczos.eigsh`, which documents it.

    Args:
        K: The `n x n` symmetric matrix.
        M: The `n x n` symmetric positive definite mass matrix.
        nev: The number of eigenpairs.
        sigma: The shift: the eigenvalues nearest it are found.
        tol: The relative Ritz residual at which a pair is accepted.
        ncv: The Lanczos basis size.
        max_restarts: Restarts before giving up.

    Returns:
        The eigenvalues ascending and the eigenvectors as columns.

    Raises:
        As `molina.lanczos.eigsh`.
    """
    return molina_lanczos.eigsh(
        K, M, nev, sigma=sigma, tol=tol, ncv=ncv, max_restarts=max_restarts
    )


def eigsh(
    K: CSRMatrix,
    M: CSRMatrix,
    nev: Int,
    *,
    sigma: Float64,
    deflate: CSRMatrix,
    tol: Float64 = 1e-12,
    ncv: Optional[Int] = None,
    max_restarts: Int = 300,
) raises -> EighResult:
    """`eigsh` with eigenvectors `M`-orthogonal to `range(deflate)`:
    `molina.lanczos.eigsh`, which documents it.

    Args:
        K: The `n x n` symmetric matrix.
        M: The `n x n` symmetric positive definite mass matrix.
        nev: The number of eigenpairs.
        sigma: The shift: the eigenvalues nearest it are found.
        deflate: `G`, `n x p` of full column rank.
        tol: The relative Ritz residual at which a pair is accepted.
        ncv: The Lanczos basis size.
        max_restarts: Restarts before giving up.

    Returns:
        The eigenvalues ascending and the eigenvectors as columns.

    Raises:
        As `molina.lanczos.eigsh`.
    """
    return molina_lanczos.eigsh(
        K,
        M,
        nev,
        sigma=sigma,
        deflate=deflate,
        tol=tol,
        ncv=ncv,
        max_restarts=max_restarts,
    )


def eigsh(
    K: CSRMatrix,
    M: DiagonalOperator,
    nev: Int,
    *,
    sigma: Float64,
    tol: Float64 = 1e-12,
    ncv: Optional[Int] = None,
    max_restarts: Int = 300,
) raises -> EighResult:
    """`eigsh` with a diagonal mass matrix: `molina.lanczos.eigsh`, which
    documents it.

    Args:
        K: The `n x n` symmetric matrix.
        M: The diagonal, positive mass matrix.
        nev: The number of eigenpairs.
        sigma: The shift: the eigenvalues nearest it are found.
        tol: The relative Ritz residual at which a pair is accepted.
        ncv: The Lanczos basis size.
        max_restarts: Restarts before giving up.

    Returns:
        The eigenvalues ascending and the eigenvectors as columns.

    Raises:
        As `molina.lanczos.eigsh`.
    """
    return molina_lanczos.eigsh(
        K, M, nev, sigma=sigma, tol=tol, ncv=ncv, max_restarts=max_restarts
    )


def eigsh[
    n: Int, k: Int, primal: Bool, out_k: Int, out_primal: Bool
](
    K: MatrixOp[n, k, primal, out_k, out_primal],
    M: MatrixOp[n, k, primal, out_k, out_primal],
    nev: Int,
    *,
    sigma: Float64,
    tol: Float64 = 1e-12,
    ncv: Optional[Int] = None,
    max_restarts: Int = 300,
) raises -> TypedEighResult[n, k, primal]:
    """`eigsh` on cochains: the eigenvectors are k-cochains of `K`'s input
    kind. For a diagonal mass, pass `star.to_matrix_op()`.

    ```mojo
    var d0 = tc.d[0, PRIMAL]()
    var K = d0.T() @ whitney_innerproduct[1](tc) @ d0   # primal 0 -> dual n
    var eig = eigsh(K, whitney_innerproduct[0](tc), 6, sigma=-0.1)
    var mode = eig.mode(1)                   # TypedCochain[n, 0, PRIMAL]
    ```

    Parameters:
        n: The complex dimension.
        k: The degree of the eigenvectors.
        primal: Whether the eigenvectors are primal.
        out_k: The degree of the cochains `K` and `M` give, `n - k`.
        out_primal: Whether they are primal, `not primal`.

    Args:
        K: The symmetric operator.
        M: The symmetric positive definite mass, of the same kinds.
        nev: The number of eigenpairs.
        sigma: The shift: the eigenvalues nearest it are found.
        tol: The relative Ritz residual at which a pair is accepted.
        ncv: The Lanczos basis size.
        max_restarts: Restarts before giving up.

    Returns:
        The eigenpairs, eigenvectors of `K`'s input kind.

    Raises:
        If `K` and `M` are on different complexes, or as the untyped
        `eigsh`.
    """
    comptime assert out_k == n - k and out_primal == (
        not primal
    ), "eigsh: K and M must map a kind to its dual kind"
    var complex = _complex_of(K.complex, M.complex)
    return TypedEighResult[n, k, primal](
        eigsh(
            K.mat,
            M.mat,
            nev,
            sigma=sigma,
            tol=tol,
            ncv=ncv,
            max_restarts=max_restarts,
        ),
        complex,
    )


def eigsh[
    n: Int,
    k: Int,
    primal: Bool,
    out_k: Int,
    out_primal: Bool,
    g_k: Int,
    g_primal: Bool,
](
    K: MatrixOp[n, k, primal, out_k, out_primal],
    M: MatrixOp[n, k, primal, out_k, out_primal],
    nev: Int,
    *,
    sigma: Float64,
    deflate: MatrixOp[n, g_k, g_primal, k, primal],
    tol: Float64 = 1e-12,
    ncv: Optional[Int] = None,
    max_restarts: Int = 300,
) raises -> TypedEighResult[n, k, primal]:
    """`eigsh` on cochains with a deflated range, such as the gradients
    `d0` for the curl-curl operator on 1-forms; see the untyped overload.

    ```mojo
    # G: d0 with the constants removed (a vertex dropped, or the boundary
    # vertices for a Dirichlet problem), so it has full column rank
    var eig = eigsh(K, M, 6, sigma=-1.0, deflate=G)
    ```

    Parameters:
        n: The complex dimension.
        k: The degree of the eigenvectors.
        primal: Whether the eigenvectors are primal.
        out_k: The degree of the cochains `K` and `M` give, `n - k`.
        out_primal: Whether they are primal, `not primal`.
        g_k: The degree of the cochains `deflate` takes.
        g_primal: Whether the cochains `deflate` takes are primal.

    Args:
        K: The symmetric operator.
        M: The symmetric positive definite mass, of the same kinds.
        nev: The number of eigenpairs.
        sigma: The shift: the eigenvalues nearest it are found.
        deflate: `G`, into `K`'s input kind, of full column rank:
            eigenvectors are `M`-orthogonal to its range.
        tol: The relative Ritz residual at which a pair is accepted.
        ncv: The Lanczos basis size.
        max_restarts: Restarts before giving up.

    Returns:
        The eigenpairs, eigenvectors of `K`'s input kind.

    Raises:
        If the operators are on different complexes, or as the untyped
        `eigsh`.
    """
    comptime assert out_k == n - k and out_primal == (
        not primal
    ), "eigsh: K and M must map a kind to its dual kind"
    var complex = _complex_of(K.complex, M.complex)
    if deflate.complex and not deflate.complex.value().is_same(complex):
        raise Error("eigsh: operands on different complexes")
    return TypedEighResult[n, k, primal](
        eigsh(
            K.mat,
            M.mat,
            nev,
            sigma=sigma,
            deflate=deflate.mat,
            tol=tol,
            ncv=ncv,
            max_restarts=max_restarts,
        ),
        complex,
    )


def _complex_of(
    a: Optional[SimplicialComplex], b: Optional[SimplicialComplex]
) raises -> SimplicialComplex:
    """The complex both operators were built on."""
    if a and b and not a.value().is_same(b.value()):
        raise Error("eigsh: operands on different complexes")
    if a:
        return a.value()
    if b:
        return b.value()
    raise Error("eigsh: neither operator records its complex")
