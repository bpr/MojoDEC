# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
MojoDEC prelude (`mojodec.prelude`).
====================================
The names most programs need: `from mojodec.prelude import *`.
"""

from mojodec.core.sparse import CSRMatrix
from mojodec.core.diagonal import DiagonalOperator
from mojodec.core.primality import Primality
from mojodec.core.subset import Subset
from mojodec.core.cell_set import CellSet
from mojodec.core.simplex import Simplex
from mojodec.core.mesh import SimplicialMesh
from mojodec.core.simplicial_complex import SimplicialComplex
from mojodec.core.cochain import Cochain, CochainBasis, d, star, delta
from mojodec.core.utils import IntArray, RealArray
