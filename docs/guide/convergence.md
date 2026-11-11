# Convergence of Whitney-form discretizations

A discretization is only as good as its rate of convergence: how fast the
error falls as the mesh is refined. `tests/test_convergence.mojo` measures
this for MojoDEC's Whitney forms:
- the L² projection and the interpolant $W R$ of smooth forms of every
  degree, in 1, 2, and 3 dimensions;
- the Galerkin solution of Poisson's equation.

This page explains what rates to expect and why, and shows the measured
numbers.

## What to expect

On a shape-regular mesh with element size $h$, an approximation from a
space that contains all polynomials of degree $p$ has an $L^2$ error of
order $h^{p+1}$ for a smooth target.
- **Whitney 0-forms** are the continuous piecewise linear functions,
  which contain all linear polynomials: $p = 1$, error $O(h^2)$.
- **Whitney k-forms, $k \ge 1$**, contain all constant k-forms but not
  all linear ones (the lowest-order Nédélec and Raviart–Thomas elements
  and their relatives): $p = 0$, error $O(h)$.

Both the projection (the best approximation) and the interpolant $W R$
reach these rates. The projection's constant is smaller, and its error is
never larger.

For **Poisson's equation** with Whitney 0-forms (linear finite elements),
Céa's lemma bounds the error in the energy norm by the best
approximation, $O(h)$ in the gradient. The Aubin–Nitsche duality argument
gains one more order in $L^2$: $\|u_h - u\| = O(h^2)$, within a constant
factor of the best approximation.

## Measured

Kuhn grids of the unit cube (see [Kuhn grids](kuhn-grids-and-tori.md)),
halving the cells. Each form's components are smooth, non-polynomial
functions. The rate is $\log_2$ of the error ratio between the last two
meshes. Errors are $\|W c - \omega\|$ from `l2_error` with a degree-7 rule.

| $n$ | $k$ | cells per axis | projection | interpolation $W R$ | rate |
|---|---|---|---|---|---|
| 1 | 0 | 8, 16, 32 | 3.3e-3, 8.2e-4, 2.0e-4 | 8.0e-3, 2.0e-3, 5.0e-4 | 2.0 |
| 1 | 1 | 8, 16, 32 | 6.3e-2, 3.2e-2, 1.6e-2 | equal | 1.0 |
| 2 | 0 | 4, 8, 16 | 8.7e-3, 2.1e-3, 5.3e-4 | 1.7e-2, 4.2e-3, 1.0e-3 | 2.0 |
| 2 | 1 | 4, 8, 16 | 0.17, 0.091, 0.046 | 0.19, 0.095, 0.048 | 0.97 |
| 2 | 2 | 4, 8, 16 | 7.2e-2, 3.6e-2, 1.8e-2 | equal | 1.0 |
| 3 | 0 | 2, 4, 8 | 5.4e-2, 1.3e-2, 3.3e-3 | 0.11, 2.7e-2, 6.9e-3 | 2.0 |
| 3 | 1 | 2, 4, 8 | 0.52, 0.30, 0.16 | 0.66, 0.34, 0.17 | 0.90 |
| 3 | 2 | 2, 4, 8 | 0.50, 0.26, 0.13 | 0.55, 0.28, 0.14 | 0.97 |
| 3 | 3 | 2, 4, 8 | 0.16, 8.2e-2, 4.1e-2 | equal | 1.0 |

For $k = n$, and for $k = 1$ in 1D, both methods give each element's mean,
so they coincide. The 3D 1-form rate, 0.90, is still approaching 1 at
eight cells per axis.

**Poisson's equation.** Solve $-\Delta u = f$ on the unit cube with
$u = \prod_a \sin(\pi x_a)$ and zero boundary values:
- the stiffness matrix is $d_0^T W_1 d_0$;
- the load is `whitney_load(sc, 0, 5, f)`;
- the boundary vertices are eliminated, and CG solves the rest.

| $n$ | Galerkin error | best (projection) | ratio | rate |
|---|---|---|---|---|
| 1 | 9.9e-3, 2.5e-3, 6.2e-4 | 4.1e-3, 1.0e-3, 2.5e-4 | 2.4 | 2.0 |
| 2 | 7.9e-2, 2.1e-2, 5.4e-3 | 2.8e-2, 6.6e-3, 1.6e-3 | 2.9–3.3 | 1.97 |
| 3 | 0.24, 8.7e-2, 2.5e-2 | 8.7e-2, 3.1e-2, 7.3e-3 | 2.7–3.4 | 1.83 |

## What the tests assert

The thresholds sit below the measured values, with room for the meshes'
pre-asymptotic behavior:
- **Projection never worse.** $\|W P\omega - \omega\| \le (1 + 10^{-9})
  \|W R\omega - \omega\|$ on every mesh, for every $n$ and $k$. The slack
  covers the load vector's quadrature error, which is about $10^{-11}$
  relative for these non-polynomial forms at degree 5.
- **Rates.** At least 1.9 for $k = 0$, at least 0.85 for $k \ge 1$, for
  both the projection and the interpolant.
- **Galerkin.** Within 5 times the best approximation on every mesh, and
  a rate of at least 1.75.

A regression that loses an order (a wrong sign in a Whitney form, a
quadrature rule that is not exact to its degree, a wrong orientation)
fails these tests by a wide margin. A change that only shifts the
constants does not.

## Why Kuhn grids, and what is not covered

These tests use Kuhn grids because they are structured in every
dimension, so halving $h$ is one argument. On them the circumcentric dual
degenerates (see [Kuhn grids](kuhn-grids-and-tori.md)), which is no
problem for Whitney forms but rules out the diagonal Hodge star.

Not yet covered:
- **DEC's diagonal Hodge star.** Its convergence needs well-centered mesh
  families, where the circumcentric dual is non-degenerate.
- **Nested meshes.** The meshes here are generated independently at each
  size, not refined from each other. Refinement with prolongation is a
  separate item.
- **Mixed problems for $k \ge 1$.** The Hodge Laplacian on 1-forms, and
  Darcy flow measured against an exact solution, test the pairing of
  spaces rather than a single space.

## Tests to read

- `tests/test_convergence.mojo`: `test_projection_and_interpolation` and
  `test_poisson_galerkin`, about 0.4 s together.
- `tests/test_derham.mojo`, `test_load_projection_and_error`: the exact
  cases. For constant forms, the load vector is $M R\omega$, the
  projection is $R\omega$, and the errors vanish.
