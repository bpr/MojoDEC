# User guide

Topics that go beyond the API reference: the mathematics behind a feature,
how to use it, and what can go wrong.

- **[Orientation and pseudomanifolds](orientation.md)**: what a coherent
  orientation is, why DEC's dual operators need it, what the
  `SimplicialComplex` constructor checks, and fixing a mesh with
  `orient_coherently`.
- **[Homology](homology.md)**: Betti numbers of complexes, subcomplexes,
  and pairs; what they say about harmonic forms; rational coefficients
  and how the ranks are computed; homology manifolds and
  `non_manifold_cells`.
- **[Cohomology bases and harmonic forms](cohomology.md)**: integer
  cocycles and dual cycles by exact elimination (`cohomology_basis`),
  periods, and a harmonic basis with period 1 around each hole
  (`harmonic_basis`) in the Hodge star or the Whitney inner product.
- **[Kuhn grids and flat tori](kuhn-grids-and-tori.md)**: boxes and
  flat tori in any dimension, why the torus in R^(2d) is exactly flat,
  the circumcentric dual being the dual cube grid, the Whitney stiffness
  equal to finite differences, and closed-form spectra.
- **[Quadrature](quadrature.md)**: Grundmann–Möller rules on simplices
  of any dimension and odd degree, their negative weights and the degree
  limit, and integrating functions over the simplices of a mesh.
- **[The de Rham map and Whitney forms](derham-and-whitney.md)**: forms
  as component lists, integrating a form into a cochain (`derham_map`),
  evaluating a cochain's Whitney form (`whitney_form`), the L²
  projection (`l2_projection`, `whitney_load`, `l2_error`), orientation,
  and why `R d = d R`, `R W = id`, and `d W = W d`.
- **[Convergence](convergence.md)**: the rates Whitney forms reach (and
  why), measured for projection, interpolation, and Poisson's equation in
  1–3D, and what the convergence tests assert.
- **[Linear solvers](linear-solvers.md)**: putting DEC systems in
  symmetric form, the `LinearOperator` and `Preconditioner` traits,
  conjugate gradients and preconditioning, MINRES for saddle points,
  stopping tests, and the symmetry check.
- **[Eigenproblems](eigenproblems.md)**: the symmetric-definite problem
  `K x = lambda M x`, how `eigh_generalized` reduces it, and checking
  eigenpairs with their backward errors (`EighResult.residuals`).
- **[Time stepping](time-stepping.md)**: the leapfrog for Maxwell's
  equations and other wave systems, its conserved energy, the stability
  limit, dispersion, mesh requirements, and boundary conditions.

The repository's `README.md` covers installation, a tour of complexes,
cochains, and operators, the differences from PyDEC, and the examples.
