# Smoke test for the packaged MojoDEC (run by conda.recipe/recipe.yaml):
# imports the installed mojodec.mojoc and numojo.mojoc, builds a small
# complex, and checks its counts and that d d = 0.

from std.testing import assert_equal

import numojo as nm
from mojodec import SimplicialComplex, d


def main() raises:
    # the unit square as two triangles
    var V = nm.fromstring[nm.f64]("[[0,0],[1,0],[1,1],[0,1]]")
    var S = nm.fromstring[DType.int]("[[0,1,3],[1,2,3]]")
    var sc = SimplicialComplex(V, S)
    assert_equal(sc[0].num_simplices, 4)
    assert_equal(sc[1].num_simplices, 5)
    assert_equal(sc[2].num_simplices, 2)
    assert_equal(sc.euler_characteristic(), 1)

    # d1 d0 = 0
    assert_equal((sc[1].d @ sc[0].d).max_abs(), 0.0)
    print("mojodec package OK")
