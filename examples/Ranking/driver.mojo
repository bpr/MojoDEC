# Least squares ranking on graphs.
#
# Reference:
#   Least Squares Ranking on Graphs, Hodge Laplacians, Time Optimality,
#   and Iterative Methods. A. N. Hirani, K. Kalyanaraman, S. Watts.
#   arXiv:1011.1716 [cs.NA]
#
# Run from the repository root:  pixi run examples Ranking
# or from this directory:  pixi run mojo run -I ../.. -I ../../build driver.mojo

import numojo as nm
from numojo.prelude import *

from mojodec import AbstractSimplicialComplex, IntArray, loadtxt, lsqr
from mojodec.core.utils import int_array, iget, iset, rows_of, real_vector


def main() raises:
    # load graph and edge values: each row is (i, j, omega_ij)
    var data = loadtxt[DType.int]("data.txt")
    var n = rows_of(data)
    var edges = int_array(n, 2)
    var omega = real_vector(n)  # pairwise comparisons
    for r in range(n):
        iset(edges, r, 0, iget(data, r, 0))
        iset(edges, r, 1, iget(data, r, 1))
        omega.unsafe_set(r, Float64(iget(data, r, 2)))

    # abstract simplicial complex from the edges
    var asc = AbstractSimplicialComplex([edges^])
    var B1 = asc.chain_complex()[1].copy()  # boundary matrix

    # least squares problem B1^T alpha ~= omega
    var alpha = lsqr(B1.T(), omega).x.copy()

    # set the minimum to 0
    var lo = alpha.unsafe_get(0)
    for i in range(alpha.size):
        lo = min(lo, alpha.unsafe_get(i))
    for i in range(alpha.size):
        alpha.unsafe_set(i, alpha.unsafe_get(i) - lo)
    for i in range(alpha.size):
        print(i, alpha.unsafe_get(i))
