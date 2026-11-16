# faer's AMD permutations

The `.perm` files are the permutations faer's `sparse::linalg::amd::order`
(faer 0.24.4, commit `0ecd846`, default `Control`) gives for three
matrices that `tests/test_sparse_ldlt.mojo`, `test_amd_matches_faer`,
rebuilds:

- `grid2_1`: the 1-form Hodge stiffness plus mass on `kuhn_grid([20, 20])`;
- `grid3big_0`: the 0-form one on `kuhn_grid([14, 14, 14])`, which makes
  AMD compress its workspace;
- `random`: a random 400 x 400 pattern with a dense row and column.

MojoDEC's `amd_ordering` builds the quotient graph in faer's order, so
ties break the same way and the permutations agree entry for entry.

To regenerate: write the pattern of `A + A^T` as `n nnz`, the column
pointers, and the row indices (whitespace separated), and pipe it to
`faer_amd.rs` built as a binary crate with `faer = "0.24"` and
`dyn-stack` as dependencies; it prints the permutation.
