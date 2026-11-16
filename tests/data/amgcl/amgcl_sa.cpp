// Reads a CSR matrix (n nnz; indptr; indices; values), runs amgcl's plain
// aggregation (eps 0.08) and smoothed-aggregation prolongation (omega 2/3),
// prints the aggregate ids and P in CSR.
#define AMGCL_NO_BOOST
#include <cstdio>
#include <iostream>
#include <vector>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>

int main(int argc, char **argv) {
    typedef amgcl::backend::builtin<double> Backend;
    typedef amgcl::backend::crs<double> Matrix;
    FILE *f = fopen(argv[1], "r");
    long n, nnz;
    if (fscanf(f, "%ld %ld", &n, &nnz) != 2) return 1;
    std::vector<ptrdiff_t> ptr(n + 1), col(nnz);
    std::vector<double> val(nnz);
    for (auto &p : ptr) if (fscanf(f, "%td", &p) != 1) return 1;
    for (auto &c : col) if (fscanf(f, "%td", &c) != 1) return 1;
    for (auto &v : val) if (fscanf(f, "%lf", &v) != 1) return 1;
    fclose(f);
    auto A = std::make_shared<Matrix>(n, n, ptr, col, val);
    amgcl::coarsening::plain_aggregates::params ap;
    amgcl::coarsening::plain_aggregates aggr(*A, ap);
    printf("%zu\n", aggr.count);
    for (long i = 0; i < n; ++i) printf("%td ", aggr.id[i]);
    printf("\n");
    amgcl::coarsening::smoothed_aggregation<Backend> sa;
    auto PR = sa.transfer_operators(*A);
    auto P = std::get<0>(PR);
    printf("%zu %zu\n", P->nrows, P->ncols);
    for (size_t i = 0; i <= P->nrows; ++i) printf("%td ", P->ptr[i]);
    printf("\n");
    for (ptrdiff_t k = 0; k < P->ptr[P->nrows]; ++k) printf("%td ", P->col[k]);
    printf("\n");
    for (ptrdiff_t k = 0; k < P->ptr[P->nrows]; ++k) printf("%.17g ", P->val[k]);
    printf("\n");
}
