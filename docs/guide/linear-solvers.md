# Linear solvers: operators, CG, and MINRES

DEC turns differential equations into sparse linear systems. This page covers
the iterative solvers in `mojodec.routines.linalg`:
- conjugate gradients (`cg`) with or without a preconditioner;
- MINRES (`minres`) for symmetric systems that are not positive definite;
- the operator traits both solvers are generic over.

It also covers how to put a DEC system into the symmetric form these
solvers need.

## Which solver

| System | Solver | Notes |
|---|---|---|
| symmetric positive (semi)definite: Poisson, a Hodge Laplacian, a stiffness matrix | `cg` | the cheapest per step; add `M=` to precondition |
| symmetric indefinite: a mixed formulation's saddle point, a shifted Laplacian | `minres` | takes a `BlockOp` whole |
| rectangular, or a least-squares problem | `lsqr` | |
| non-symmetric, or small | `spsolve` | dense pivoted LU; for moderate sizes |

`cg` and `minres` both raise on an operator that is not symmetric (see
[the symmetry check](#the-symmetry-check)).

## DEC systems in symmetric form

The operators DEC builds are symmetric when they are written as a product
$K = d^T \star\, d$ ("stiffness" form), and usually not otherwise. The
Laplacian on k-cochains, `delta(d(.))`, is

$$
\delta d = \star_k^{-1} \, d_k^T \star_{k+1} d_k = \star_k^{-1} K .
$$

$K$ is symmetric. $\star_k^{-1} K$ is not, unless $\star_k$ is a multiple
of the identity: entry $(i, j)$ is $K_{ij} / \star_k[i]$, and entry $(j, i)$
is $K_{ij} / \star_k[j]$. What $\star_k^{-1} K$ is, is **self-adjoint in the
star inner product** $\langle x, y\rangle_\star = x^T \star_k\, y$:

$$
\langle x, \star^{-1} K y \rangle_\star = x^T K y = \langle \star^{-1} K x, y \rangle_\star .
$$

CG needs symmetry in the inner product it computes with. Plain CG uses the
Euclidean one, so it must not be given $\star^{-1} K$. Two equivalent
fixes:

1. **Multiply through by the star.** Solve $K x = \star_k b$ in place of
   $\star_k^{-1} K x = b$. The matrix is now symmetric.
2. **Precondition with the inverse star.** Preconditioned CG on
   $K x = \star_k b$ with $M = \star_k^{-1}$ is exactly CG on
   $\star_k^{-1} K$ in the star inner product. The residual of the
   multiplied system is $R = \star b - K x = \star r$, where
   $r = b - \star^{-1} K x$. So the preconditioned residual is
   $z = M R = r$, and the inner products PCG forms are
   $R^T z = \langle r, r \rangle_\star$ and
   $p^T K p = \langle p, \star^{-1} K p \rangle_\star$.

Both fixes together are what the HodgeDecomposition example does:

```mojo
ref S = sc[0].star
var A = S @ delta(d(sc.get_cochain_basis(0))).v   # K = d0^T star1 d0, symmetric
var b = S @ delta(omega).v                         # star0 (delta omega)
alpha.v = cg(A, b, M=sc[0].star_inv, rtol=1e-8).take_x()
```

PyDEC's driver hands $\star^{-1}K$ to SciPy's `cg`, which does not check
symmetry. MojoDEC refuses it. The symmetric solve gives the same harmonic
space (the largest principal angle between the two is $3 \times 10^{-6}$,
within the solve tolerance).

In the typed layer the symmetric form is the natural one. A stiffness
operator such as `d0.T() @ tc.star[1, PRIMAL]() @ d0` maps primal
0-cochains to dual n-cochains, and is symmetric as a matrix.

## Operators

`mojodec.core.operator` defines what a solver needs of a matrix.

```mojo
trait LinearOperator:
    def num_rows(self) -> Int: ...
    def num_cols(self) -> Int: ...
    def apply_into(self, x: RealArray, mut y: RealArray) raises: ...   # y = A x

trait Preconditioner(LinearOperator):
    pass                                         # a promise: symmetric positive definite
```

| Type | `LinearOperator` | `Preconditioner` |
|---|---|---|
| `CSRMatrix` | yes | no |
| `DiagonalOperator` (a Hodge star, `jacobi(A)`) | yes | yes |
| `Identity(n)` | yes | yes |
| `MatrixOp` | yes | no |
| `DiagonalOp` | yes | yes |
| `BlockOp` | yes, block by block, never assembled | no |

`Preconditioner` has no methods of its own. It marks the types whose
values can be symmetric positive definite, and a type takes on that promise
by conforming. A general `CSRMatrix` cannot conform, because nothing about
its type says it is symmetric, so a one-sided Gauss–Seidel sweep or an
arbitrary matrix cannot be passed as `M` by mistake. A `DiagonalOperator`
is always symmetric but can have negative entries (a Hodge star on a mesh
that is not well-centered). The solvers catch that case at run time (see
[breakdown](#stopping-and-breakdown)).

Your own operator only needs the three methods. For example, a product
applied without forming it:

```mojo
struct Stiffness(LinearOperator):
    var d: CSRMatrix                 # d0
    var dt: CSRMatrix                # d0^T, stored once
    var star: DiagonalOperator       # star1

    def __init__(out self, sc: SimplicialComplex) raises:
        self.d = sc[0].d.copy()
        self.dt = sc[0].d.transpose()
        self.star = sc[1].star.copy()

    def num_rows(self) -> Int:
        return self.d.ncols

    def num_cols(self) -> Int:
        return self.d.ncols

    def apply_into(self, x: RealArray, mut y: RealArray) raises:
        var t = self.star @ (self.d @ x)   # two allocations per product
        self.dt.apply_into(t, y)

var res = cg(Stiffness(sc), b, rtol=1e-12)
```

`apply_into(x, y)` overwrites `y`. The solvers call it with the same `y`
every iteration. This one allocates its intermediate vectors; storing the
matrix product once is usually faster anyway, but the pattern applies to
operators that are never stored, such as a matrix-free stencil.

## Conjugate gradients

For symmetric positive definite $A$, the k-th CG iterate minimizes the
energy norm of the error over the Krylov space:

$$
x_k = \operatorname*{arg\,min}_{x \in \mathcal K_k} \| x - x_* \|_A,
\qquad
\mathcal K_k = \operatorname{span}\{ b, A b, \dots, A^{k-1} b \},
$$

with $\|e\|_A^2 = e^T A e$. It needs one product with $A$ per step and three
vectors of storage. The classic bound in terms of the condition number
$\kappa = \lambda_{\max} / \lambda_{\min}$ is

$$
\| x_k - x_* \|_A \le 2 \left( \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} \right)^k \| x_0 - x_* \|_A .
$$

On a mesh of size $h$, the Laplacian has $\kappa \sim h^{-2}$, so the
iteration count grows like $h^{-1}$: doubling the resolution doubles the
iterations.

**Singular systems.** A Laplacian with Neumann conditions, or on a closed
surface, has the constants in its kernel. CG still converges when the
system is consistent ($b$ orthogonal to the kernel, as for $b = A f$).
Starting from $x_0 = 0$, every iterate lies in the range of $A$, so the
limit is the solution of least norm. The constant is not pinned by
anything except that start, so add or remove it yourself if you need a
particular one.

## Preconditioning

A preconditioner $M \approx A^{-1}$, symmetric positive definite, replaces
the spectrum of $A$ by that of $M^{1/2} A M^{1/2}$. In exact arithmetic,
preconditioned CG is CG on that operator, written so that $M^{1/2}$ is never
formed. Each step costs one product with $A$ and one with $M$.

```mojo
var res = cg(A, b, M=jacobi(A), rtol=1e-10)
```

`jacobi(A)` is the diagonal $\operatorname{diag}(1/|a_{ii}|)$. It undoes
bad scaling between rows, which comes from strongly graded meshes, material
coefficients that differ by orders of magnitude, or mixing units. On
`tests/test_linalg.mojo`'s 1-D Laplacian scaled by a diagonal from 1 to
1000, it brings CG from 173 iterations to 60 (the dimension). Jacobi does
nothing for the $h^{-2}$ growth of $\kappa$, which needs multigrid or an
incomplete factorization. Neither is in MojoDEC yet.

`jacobi` raises if a diagonal entry is zero or not finite. A saddle-point
matrix has zeros on its diagonal, in the constraint block.

**The kinds of a typed preconditioner.** The preconditioner approximates
$A^{-1}$, so it maps the kind of cochain $A$ produces back to the kind $A$
takes. For a stiffness operator from primal 0-cochains to dual n-cochains,
`M` is a `DiagonalOp` from dual n to primal 0:

```mojo
var A = d0.T() @ tc.star[1, PRIMAL]() @ d0          # primal 0 -> dual 2
var J: DiagonalOp[2, 2, DUAL, 0, PRIMAL] = jacobi(A)
var u = cg(A, b, M=J, rtol=1e-12).take_x()          # TypedCochain[2, 0, PRIMAL]
```

A preconditioner of the wrong kinds does not compile.

## MINRES

When $A$ is symmetric but indefinite, $\|e\|_A$ is not a norm and CG can
break down: the step length $r^T r / p^T A p$ divides by a $p^T A p$ that
changes sign. MINRES (Paige and Saunders, 1975) instead minimizes the
residual over the same Krylov space:

$$
x_k = \operatorname*{arg\,min}_{x \in \mathcal K_k} \| b - A x \|_2 .
$$

The method:
1. The Lanczos process builds an orthonormal basis $V_k$ of
   $\mathcal K_k$ with a three-term recurrence,
   $A V_k = V_{k+1} \bar T_k$. Here $\bar T_k$ is $(k+1) \times k$ and
   tridiagonal, and $V_k$'s first column is $b / \beta_1$, with
   $\beta_1 = \|b\|$.
2. Writing $x = V_k y$, the residual norm is
   $\| \beta_1 e_1 - \bar T_k y \|$, a small least-squares problem.
3. That problem is solved by a QR factorization of $\bar T_k$, which grows
   by one Givens rotation per step. The rotation's sine $s_k$ updates the
   residual norm, $\phi_k = s_k \, \phi_{k-1}$, so it is known without
   forming the residual.
4. The solution is updated through short recurrences, so storage is a
   handful of vectors, like CG's, and each step costs one product with $A$.

The residual norm never increases from one step to the next. It is
reported in `residual_norm`, recomputed as $\|b - A x\|$ from the final
$x$ rather than taken from the recurrence.

**Convergence** depends on the eigenvalues on both sides of zero. For a
spectrum in $[-d, -c] \cup [c, d]$, with $\kappa = d / c$,

$$
\frac{\| r_k \|}{\| r_0 \|} \le 2 \left( \frac{\kappa - 1}{\kappa + 1} \right)^{\lfloor k/2 \rfloor},
$$

which behaves like CG with the condition number squared. An indefinite
system takes more iterations than a definite one of the same $\kappa$.

**Singular systems.** On a consistent singular system MINRES converges to
the solution of least norm, as CG does. On an inconsistent one (b has a
component in the kernel), the residual stalls at the least-squares
residual and the iteration runs to `maxiter`. Make the system consistent
first (for a Neumann problem, remove the mean of the right-hand side).

### Saddle points

The mixed formulation of Darcy flow solves for a flux $u$ (primal
1-cochain) and a pressure $p$ (dual 0-cochain) at once:

$$
\begin{bmatrix} -\star_1 & d_1^T \\ d_1 & 0 \end{bmatrix}
\begin{bmatrix} u \\ p \end{bmatrix} =
\begin{bmatrix} f \\ g \end{bmatrix}.
$$

The matrix is symmetric and indefinite: $n_1$ negative eigenvalues from the
$-\star_1$ block, and $n_2$ positive ones from the constraint. It is
non-singular when $\star_1$ is definite and $d_1$ is onto (no dual
0-cochain is in the kernel of $d_1^T$), which holds on a disk. `minres`
takes the `BlockOp` and applies its four blocks to the stacked vector one
at a time, without assembling them:

```mojo
var A = BlockOp(tc.star[1, PRIMAL]().to_matrix_op() * -1.0, d1.T(), d1,
                tc.zero[0, DUAL, 2, PRIMAL]())
var res = minres(A, b, rtol=1e-12)            # b: CochainPair[2, 1, DUAL, 2, PRIMAL]
var x = res^.take_x()                         # CochainPair[2, 1, PRIMAL, 0, DUAL]
```

The whole `BlockOp` must be symmetric: $a_{11}$ and $a_{22}$ symmetric,
and $a_{21} = a_{12}^T$. Scaling one off-diagonal block breaks that, and
`minres` raises. The DarcyFlow example prescribes some unknowns (boundary
fluxes, one pressure). That reduced system is solved by the
`spsolve(..., values=, fixed_first=, fixed_second=)` overload; `minres` has
no such overload yet.

### Preconditioned MINRES

A MINRES preconditioner must be symmetric positive definite even though
$A$ is not: it defines the norm being minimized, $\|r\|_M = \sqrt{r^T M r}$.
It should approximate $|A|^{-1}$ in size, not in sign. That is why
`jacobi` takes $1/|a_{ii}|$. On the indefinite test matrix (the shifted
Laplacian, scaled by a diagonal from 1 to 100), Jacobi brings MINRES from
159 iterations to 60.

With `M=`, the stopping test is in the M-norm:
$\|r\|_M \le \max(\texttt{rtol} \cdot \|b\|_M, \texttt{atol})$. Preconditioned
CG keeps the Euclidean test.

For saddle points, the effective preconditioners are block-diagonal, with
the exact inverse star on the flux block and a Schur-complement
approximation $(d_1 \star_1^{-1} d_1^T)^{-1}$ on the pressure block. That
needs an inner solve per application, and is not in MojoDEC yet.

## Stopping and breakdown

| | `cg` | `minres` | `minres` with `M=` |
|---|---|---|---|
| stops when | $\|r\| \le \max(\texttt{rtol}\,\|b\|, \texttt{atol})$ | the same, from the recurrence | $\|r\|_M \le \max(\texttt{rtol}\,\|b\|_M, \texttt{atol})$ |
| `residual_norm` | from the recurrence | $\|b - A x\|$, recomputed | $\|b - A x\|$, recomputed |
| `maxiter` default | $10 n$ | $10 n$ | $10 n$ |

The result's `stop` says why the iteration ended:

- `StopReason.CONVERGED`: the test passed.
- `StopReason.MAX_ITER`: the limit came first. The `x` is the last iterate.
- `StopReason.BREAKDOWN`: the iteration cannot go on.
  - In `cg`, $p^T A p \le 0$: $A$ is not positive definite on the Krylov
    space. A stiffness matrix with a negative Hodge-star entry (a mesh that
    is not well-centered) can do this; use `minres`.
  - In either solver, $r^T M r < 0$: the preconditioner is not positive
    definite.
  - Any NaN reports a breakdown too, rather than running to `maxiter`.

Always check `res.converged()` (or `res.stop`) before trusting `x`.

## The symmetry check

Before iterating, `cg` and `minres` apply $A$ to two fixed pseudorandom
vectors $u$ and $w$ with entries in $[-1, 1)$, and raise unless

$$
| u^T A w - w^T A u | \le 10^{-8} \left( \|u\| \|A w\| + \|w\| \|A u\| \right).
$$

The left side is $u^T (A - A^T) w$. For a non-symmetric $A$ it vanishes
only for $(u, w)$ on a set of measure zero, so the check is reliable while
costing two products. The tolerance passes matrices that are symmetric up
to rounding, such as `B.T() @ S @ B` assembled in floating point, by a wide
margin. A NaN in either product passes the check and is reported as a
breakdown.

The message gives both values:

```
cg: the operator is not symmetric (u^T A w = -484.22, w^T A u = -400.53)
```

## Pitfalls

- **`delta(d(.))` is not symmetric.** Multiply by the star, or use the
  star inner product through `M=` (see above).
- **Shapes.** A stiffness operator from primal 0 to dual n is square, but
  an operator between different degrees (`d0`, for example) is not. Use
  `lsqr` for those.
- **Negative stars.** On a mesh that is not well-centered, some
  $\star_1$ entries are negative. Then $d^T \star d$ can be indefinite,
  `cg` can break down, and an inverse star is no longer a valid
  preconditioner.
- **`rtol` is relative to $\|b\|$.** For a right-hand side that is mostly
  cancellation, pass `atol` too.
- **Reproducibility.** Without `M=`, both solvers run with `Identity`,
  and `cg` produces exactly the iterates it did before preconditioning
  existed.

## Tests to read

- `tests/test_linalg.mojo`:
  - `test_pcg`: Jacobi on a badly scaled Laplacian; Identity giving
    exactly the plain iterates; a `DiagonalOperator` as `A`.
  - `test_minres_indefinite`: a shifted Laplacian, where `cg` breaks
    down; monotone residuals.
  - `test_minres_singular_consistent`: the least-norm solution.
  - `test_preconditioned_minres`.
  - `test_solver_input_checks`: non-symmetric operators, indefinite
    preconditioners, `jacobi`'s errors.
- `tests/test_typed.mojo`, `test_solvers`: typed `cg` and `minres` with a
  `jacobi` preconditioner of the right kinds.
- `tests/test_typed_block.mojo`, `test_minres_on_blocks`: the Darcy
  saddle point against `spsolve`, and a non-symmetric block operator
  refused.
