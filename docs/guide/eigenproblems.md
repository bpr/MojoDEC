# Eigenproblems and backward errors

DEC turns spectral questions into eigenvalue problems: the vibration modes
of a membrane, the resonances of a cavity, the spectrum of the Hodge
Laplacian on a surface. This page covers how `eigh_generalized` solves
them densely, how `eigsh` finds a few eigenpairs of a large sparse
problem, and how to check the answer with `EighResult.residuals`.

## The symmetric-definite problem

A DEC eigenproblem is almost always

$$
K x = \lambda M x,
$$

where $K$ is symmetric (a stiffness matrix such as $d^T \star d$) and $M$
is symmetric positive definite (a Hodge star, or a Whitney mass matrix).
Such a problem has $n$ real eigenvalues, and eigenvectors that can be
chosen $M$-orthonormal: $X^T M X = I$ and $X^T K X = \Lambda$.

The Laplacian `star_inv @ d.T() @ star @ d` is not symmetric (see
[Linear solvers](linear-solvers.md#dec-systems-in-symmetric-form)), but
$\star^{-1} K x = \lambda x$ is the same problem as $K x = \lambda \star x$.
Pass `K` and the star:

```mojo
ref S = sc[k].star
var K = S @ laplace_derham(sc.get_cochain_basis(k)).v   # symmetric, sparse
var eig = eigh_generalized(K.todense(), S)
```

`eigh_generalized` reduces the problem to a standard one and solves that
with `dense.eigh`, which uses Householder tridiagonalization and the
implicit QL method.
- **Diagonal M** (a Hodge star): with $M = D^2$, it solves
  $D^{-1} K D^{-1} y = \lambda y$ and sets $x = D^{-1} y$. That is a
  scaling of $K$, with no factorization.
- **Dense M** (a Whitney mass matrix): with the Cholesky factor
  $M = L L^T$, it solves $L^{-1} K L^{-T} y = \lambda y$ and sets
  $x = L^{-T} y$.

Both are dense and $O(n^3)$, which suits a few thousand unknowns. The
typed overloads take a `MatrixOp` or `DiagonalOp` for $M$, and return a
`TypedEighResult` whose `mode(i)` is a cochain.

## Large problems: shift-invert Lanczos

`eigsh(K, M, nev, sigma=s)` finds the `nev` eigenvalues nearest $s$ and
their eigenvectors, for sparse $K$ and $M$ of any size the factorization
of $K - sM$ fits in memory:

```mojo
var eig = eigsh(K, M, 20, sigma=-0.1)    # the 20 smallest, 0 included
var eta = eig.residuals(K, M)            # one backward error per pair
```

**Spectral transformation.** The operator

$$
\mathrm{OP} = (K - sM)^{-1} M
$$

has the eigenvectors of $K x = \lambda M x$ with eigenvalues
$\theta = 1/(\lambda - s)$, and is self-adjoint in the $M$ inner product
$\langle x, y \rangle_M = x^T M y$. The eigenvalues nearest $s$ become
the largest $|\theta|$, well separated from the rest, and Lanczos finds
the extreme eigenvalues of a self-adjoint operator first (Ericsson and
Ruhe, 1980). $K - sM$ is factored once by `SparseLDLT` (AMD ordering,
see [Sparse direct](sparse-direct.md)); each Lanczos step is then two
triangular solves and one product with $M$. Shifts inside the spectrum
work: $K - sM$ is then indefinite, which $LDL^T$ handles.

**Lanczos in the $M$ inner product.** The basis $V$ is kept
$M$-orthonormal ($V^T M V = I$) with $MV$ stored beside it, so each
inner product is one dot product. Each new vector is orthogonalized
against the whole basis twice (classical Gram–Schmidt with
reorthogonalization), which keeps $V$ orthonormal to rounding and avoids
the spurious copies of plain Lanczos. The projection $H = V^T M\,
\mathrm{OP}\, V$ is tridiagonal, and its eigenpairs (the Ritz pairs,
from `dense.eigh`) approximate those of $\mathrm{OP}$. A pair is accepted
when its residual $\|\mathrm{OP}\,x - \theta x\|_M$, read off the last
row of $H$'s eigenvectors, is at most `tol` $|\theta|$ (default
$10^{-12}$).

**Thick restart.** The basis holds `ncv` vectors (default
$\max(2\,\mathrm{nev} + 1, 20)$). When it is full and the wanted pairs
have not converged, the method keeps the $\mathrm{nev} + (\mathrm{ncv} -
\mathrm{nev})/2$ Ritz vectors of largest $|\theta|$ and the last
residual vector, and continues (Wu and Simon, 2000). $H$ is then an arrow
(the kept Ritz values on its diagonal, their couplings to the residual in
one row and column) followed by a tridiagonal. Convergence is linear in
the ratio of the wanted $|\theta|$ to the nearest unwanted one, so a
basis of `ncv` below about $2\,$`nev` can take thousands of restarts when
eigenvalues cluster.

**Multiple eigenvalues.** Lanczos from one start vector sees one copy of
each eigenvalue. Further copies of a multiple eigenvalue enter only
through rounding, and the wanted pairs can converge before they do. On an
icosphere, whose symmetry makes the $l(l+1)$ eigenvalues exactly
multiple, plain Lanczos returned five copies of a six-fold eigenvalue and
the next cluster in place of the sixth. So `eigsh` locks the converged
pairs and runs Lanczos again from a fresh vector $M$-orthogonal to them.
$\mathrm{OP}$ maps that complement to itself, so a missing copy is the
nearest eigenvalue there. Anything found nearer $s$ than the farthest
pair kept is swapped in, and the search repeats until nothing is.

**Deflating a kernel.** Curl–curl $K = d_1^T M_2 d_1$ on edges vanishes on
every gradient: its eigenvalue 0 has the multiplicity of the number of
interior vertices. The nonzero eigenvalues are the wanted ones, and
`deflate=G` with $G = d_0$ (restricted to the free cells, of full column
rank) seeks the eigenvectors $M$-orthogonal to $\mathrm{range}(G)$. Each
new vector is projected,

$$
x \leftarrow x - G\,(G^T M G)^{-1} G^T M x ,
$$

with $G^T M G$ factored once (`SparseCholesky`). Both $\mathrm{range}(G)$
and its $M$-orthogonal complement are invariant under $\mathrm{OP}$, so
the kernel never reappears.

```mojo
var G = sc[0].d.select(interior_edges, interior_vertices)
var eig = eigsh(K, M, 20, sigma=-1.0, deflate=G)   # first 20 nonzero
```

The shift still must not be an eigenvalue of the whole pencil, kernel
included: $K - 0 M$ is singular here even though the iteration never
sees the kernel, so shift below 0.

**Measured** (one thread, release build, 20 eigenpairs, Kuhn grids of the
unit square):

| problem | unknowns | time | backward error | eigenvalues / $\pi^2$ |
|---|---|---|---|---|
| P1 Neumann Laplacian, $100^2$ | 10,201 | 66 ms | 1e-14 | 1.00008, …, 18.04 |
| P1 Neumann Laplacian, $300^2$ | 90,601 | 0.98 s | 3e-14 | 1.00001, …, 18.004 |
| cavity, curl–curl deflated, $100^2$ | 29,800 | 0.19 s | 5e-16 | 0.99995, …, 19.9996 |
| cavity, curl–curl deflated, $200^2$ | 119,600 | 1.1 s | 4e-16 | 0.99999, …, 19.9999 |

The Sphere example uses `eigsh`; `--subdivisions 6` (40,962 vertices,
122,880 edges) runs in 11 s, compilation included, with backward errors
below $10^{-15}$.

**Typed.** `eigsh(K, M, nev, sigma=s)` on `MatrixOp`s returns a
`TypedEighResult` (pass a star as `star.to_matrix_op()`); with
`deflate=G`, $G$ maps into $K$'s input kind. Both check at compile time
that $K$ and $M$ map a kind to its dual kind.

## Backward error

A computed eigenpair $(\tilde\lambda, \tilde x)$ is never exact. Its
**backward error** asks how much the problem would have to change for it
to be exact:

$$
\eta(\tilde\lambda, \tilde x) =
\min \left\{ \epsilon : (K + \Delta K) \tilde x = \tilde\lambda (M + \Delta M) \tilde x,\;
\|\Delta K\| \le \epsilon \|K\|,\; \|\Delta M\| \le \epsilon \|M\| \right\}.
$$

In the 2-norm it has a closed form (Higham and Higham, 1998), the scaled
residual:

$$
\eta = \frac{\| K \tilde x - \tilde\lambda M \tilde x \|}
            {\left( \|K\| + |\tilde\lambda| \, \|M\| \right) \| \tilde x \|}.
$$

`eig.residuals(K, M)` returns $\eta$ for every eigenpair, in the order of
`eig.values`:

```mojo
var eta = eig.residuals(K, S)
for i in range(eta.size):
    if eta.unsafe_get(i) > 1e-12:
        print("pair", i, "is not to be trusted")
```

**What the numbers mean.** The matrices themselves carry a relative
rounding error of order the machine epsilon,
$u = 2.2 \times 10^{-16}$. A backward-stable solver produces $\eta$ of a
modest multiple of $n u$, and then each pair is the exact answer to a
problem as close to yours as rounding already put it. `eigh_generalized`
gives about $10^{-15}$ on the examples (Sphere $4.9 \times 10^{-15}$,
ResonantCavity $6.6 \times 10^{-15}$, each with a few hundred
unknowns). A value orders of magnitude larger means one of these:
- a bug;
- an ill-conditioned mass matrix (the Cholesky reduction loses accuracy
  when $M$ is nearly singular);
- the wrong matrices passed to `residuals`.

**One threshold for every problem.** Unlike a check of
$K x - \lambda M x$ against a tolerance chosen per example, $\eta$ is
scale-free. Scaling $K$, $M$, or $x$ leaves it unchanged, so one threshold
serves every problem. Sphere and ResonantCavity raise if any pair exceeds
$10^{-12}$.

### From backward error to accuracy

The backward error is about the solver. How close $\tilde\lambda$ is to a
true eigenvalue also depends on the problem. For symmetric-definite
problems the bound is clean. Some eigenvalue $\lambda$ satisfies

$$
|\tilde\lambda - \lambda| \le
\frac{\| r \|_{M^{-1}}}{\| \tilde x \|_M}
\le \eta \, \frac{\|K\| + |\tilde\lambda| \|M\|}{\lambda_{\min}(M)},
$$

where $r = K \tilde x - \tilde\lambda M \tilde x$. Eigenvalues are therefore
accurate relative to $\|K\|$: the largest ones to nearly full precision,
the smallest ones to an absolute error near $u \, \|K\|$. On a mesh of
size $h$, $\|K\| / \lambda_{\min}(M) \sim h^{-2}$, so the low end of the
spectrum loses digits as the mesh is refined. That is why the zero
eigenvalues of a Laplacian come out as $\pm 10^{-12}$ rather than 0, and
why ResonantCavity treats values below $10^{-10}$ as zero (Sphere's
constant 0-form comes out as $7 \times 10^{-14}$).

Eigenvectors are sensitive to the **gap**: the angle between $\tilde x$
and the true eigenvector is at most about
$\|r\|_{M^{-1}} / (\|\tilde x\|_M \cdot \mathrm{gap})$, with gap the
distance from $\tilde\lambda$ to the rest of the spectrum. Within a cluster
of equal eigenvalues (the $2l+1$ spherical harmonics of degree $l$, the
pairs $m^2 + n^2 = n^2 + m^2$ of a square cavity), individual eigenvectors
are not determined at all; only the space they span is.

### Solver error is not discretization error

$\eta$ says nothing about how well the discrete spectrum approximates the
continuous one. On the Sphere example's icosphere, the computed 0-form eigenvalues
are below $l(l+1)$ by 0.0005% for $l = 1$, 2% for $l = 2$, and up to 6%
for $l = 3$. That is discretization error. Meanwhile every pair has
$\eta < 5 \times 10^{-15}$, which is the solver. The $l = 3$ cluster
also splits into a group of 3 and a group of 4, because the icosahedral
mesh breaks the sphere's symmetry. Check the two separately: $\eta$
for the solver, and a refinement study or a known exact spectrum for the
discretization.

### How it is computed

- **Cost.** For each pair, one product with $K$ and one with $M$, so
  $n$ products in all. With sparse operators that is $O(n \cdot
  \text{nnz})$, far below the $O(n^3)$ of the solve. Pass the sparse $K$
  even if you solved with its dense copy, as the Sphere example does.
  `residuals` takes any `LinearOperator` (`CSRMatrix`, `DiagonalOperator`,
  the typed ops) as well as dense arrays.
- **Norms.** $\|K\|$ and $\|M\|$ are 2-norms, estimated by 30 steps of
  power iteration from a fixed start. Each step's $\|A v\| / \|v\|$ is a
  lower bound on $\|A\|$, so the estimate is too small, if anything, and
  $\eta$ too large. The answer errs on the safe side, and for symmetric
  matrices it is within a few percent.
- **The standard problem.** `dense.eigh(A)` results take
  `residuals(A)`: $M = I$, $\eta = \|A x - \lambda x\| / ((\|A\| + |\lambda|)\|x\|)$.
- **Checks.** `residuals` raises unless $K$ and $M$ are $n \times n$, $n$
  the eigenvectors' length (all $n$ pairs from `eigh_generalized`, a few
  from `eigsh`). The typed overloads raise if either operator is on
  another complex than the eigenvectors, and an operator of the wrong
  kinds does not compile.

## Pitfalls

- **Pass the problem you solved.** For a cavity with boundary conditions,
  that is the reduced `K_int`, `M_int`, not the full matrices.
- **Zero eigenvalues.** A zero eigenvalue's backward error is
  $\|K x\| / (\|K\| \|x\|)$: small when $x$ is in the kernel to rounding,
  as it should be.
- **Shift at an eigenvalue.** `eigsh` raises when $K - sM$ has a zero
  pivot. A shift very near an eigenvalue gives a tiny pivot instead, which
  is harmless: that eigenvalue's $\theta$ is huge and converges at once.
- **Shift far from the spectrum.** The accepted residual is relative to
  $K - sM$, so the backward error grows like $|s|\,\|M\| / \|K\|$ times
  `tol`. Shift near the wanted eigenvalues.
- **Non-symmetric input.** `eigh_generalized` assumes $K$ is symmetric
  and does not check. A non-symmetric $K$ (for example
  $\star^{-1} K$ passed by mistake) gives eigenpairs of a different
  matrix, and their backward errors against the matrix you meant come out
  large. That makes `residuals` a cheap way to catch the mistake.

## References

- T. Ericsson, A. Ruhe, "The spectral transformation Lanczos method for
  the numerical solution of large sparse generalized symmetric eigenvalue
  problems", Math. Comp. 35(152), 1251–1268, 1980.
- K. Wu, H. Simon, "Thick-restart Lanczos method for large symmetric
  eigenvalue problems", SIAM J. Matrix Anal. Appl. 22(2), 602–616, 2000.

## Tests and examples to read

- `tests/test_linalg.mojo`, `test_eigen_residuals`:
  - the standard, dense-mass, and diagonal-mass solvers on random
    symmetric matrices, all below $n u$;
  - the formula on a hand-made inexact pair ($\eta = 1/9$);
  - size errors.
- `tests/test_typed.mojo`, `test_whitney_and_eigenproblems`: the typed
  overloads with a Whitney mass matrix and with a Hodge star.
- `tests/test_lanczos.mojo`: the smallest eigenvalues of a Neumann
  Laplacian against `eigh_generalized`, with $M$-orthonormal vectors;
  shifts inside the spectrum, a diagonal mass, and the standard problem;
  curl–curl with the gradients deflated (and their $M$-orthogonality);
  the typed overloads; whole and exhausted spaces; the errors.
- `tests/test_properties_robust.mojo`,
  `test_eigsh_finds_every_copy_of_multiple_eigenvalues`: pencils made of
  $r$ disjoint copies, every eigenvalue exactly $r$-fold, any shift and
  basis size. Without the locked search it fails at once.
- The Sphere and ResonantCavity examples print the largest backward error,
  and raise above $10^{-12}$.
