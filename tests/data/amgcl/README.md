# amgcl's aggregates and smoothed prolongations

The `.out` files are what amgcl (commit `c68350e`, MIT) computes for
three matrices that `tests/test_amg.mojo`, `test_matches_amgcl`,
rebuilds:

- `grid2`: the P1 Laplacian `d0^T M1 d0` on the interior vertices of
  `kuhn_grid([20, 20])`;
- `edges2`: the 1-form Hodge stiffness plus mass on the interior edges of
  `kuhn_grid([8, 8])` (off-diagonal entries of both signs);
- `random`: a 300 x 300 weighted graph Laplacian plus a diagonal, with
  every 37th row decoupled (amgcl leaves those rows out).

Each file holds the number of aggregates, the aggregate id of each row
(`-2`: left out), and the smoothed prolongation `P` with amgcl's default
`omega = 2/3`, as `nrows ncols`, row pointers, columns, and values
(`%.17g`, exact). MojoDEC's `_aggregates` and `_smoothed_prolongation`
agree with them bit for bit.

To regenerate: write the matrix as `n nnz`, the row pointers, the
columns, and the values (17 significant digits), build `amgcl_sa.cpp`
with `g++ -O2 -std=c++17 -I <amgcl checkout> amgcl_sa.cpp -o amgcl_sa`
(header only, no Boost), and run `./amgcl_sa matrix.csr > name.out`.
