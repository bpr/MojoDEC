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
- **[Linear solvers](linear-solvers.md)**: putting DEC systems in
  symmetric form, the `LinearOperator` and `Preconditioner` traits,
  conjugate gradients and preconditioning, MINRES for saddle points,
  stopping tests, and the symmetry check.
- **[Time stepping](time-stepping.md)**: the leapfrog for Maxwell's
  equations and other wave systems, its conserved energy, the stability
  limit, dispersion, mesh requirements, and boundary conditions.

The repository's `README.md` covers installation, a tour of complexes,
cochains, and operators, the differences from PyDEC, and the examples.
