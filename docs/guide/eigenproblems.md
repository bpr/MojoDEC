# Eigenproblems and backward errors

DEC turns spectral questions into eigenvalue problems: the vibration modes
of a membrane, the resonances of a cavity, the spectrum of the Hodge
Laplacian on a surface. This page covers how `eigh_generalized` solves
them, and how to check the answer with `EighResult.residuals`.

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
- **Checks.** `residuals` raises unless $K$ and $M$ are $n \times n$, one
  row per eigenpair. The typed overloads raise if either operator is on
  another complex than the eigenvectors, and an operator of the wrong
  kinds does not compile.

## Pitfalls

- **Pass the problem you solved.** For a cavity with boundary conditions,
  that is the reduced `K_int`, `M_int`, not the full matrices.
- **Zero eigenvalues.** A zero eigenvalue's backward error is
  $\|K x\| / (\|K\| \|x\|)$: small when $x$ is in the kernel to rounding,
  as it should be.
- **Non-symmetric input.** `eigh_generalized` assumes $K$ is symmetric
  and does not check. A non-symmetric $K$ (for example
  $\star^{-1} K$ passed by mistake) gives eigenpairs of a different
  matrix, and their backward errors against the matrix you meant come out
  large. That makes `residuals` a cheap way to catch the mistake.

## Tests and examples to read

- `tests/test_linalg.mojo`, `test_eigen_residuals`:
  - the standard, dense-mass, and diagonal-mass solvers on random
    symmetric matrices, all below $n u$;
  - the formula on a hand-made inexact pair ($\eta = 1/9$);
  - size errors.
- `tests/test_typed.mojo`, `test_whitney_and_eigenproblems`: the typed
  overloads with a Whitney mass matrix and with a Hodge star.
- The Sphere and ResonantCavity examples print the largest backward error,
  and raise above $10^{-12}$.
