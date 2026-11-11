"""PyDEC's side of the cross-library comparison (see README.md here).

Prints one JSON object per line, as mojodec_bench.mojo does. PyDEC computes
geometry lazily; `construct_hodge()` is what forces circumcenters, primal
and dual volumes, and the stars, so that is the geometry phase.

    python pydec_bench.py MESH_DIR REPEATS [MESH...]

PyDEC must be importable (run.py puts it on PYTHONPATH). A phase whose
first run takes more than BUDGET_S seconds is not repeated. Topology,
geometry and build are timed in one loop: build is the complex followed
by `construct_hodge()`.
"""

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from pydec import simplicial_complex, whitney_innerproduct  # noqa: E402
from pydec.io import read_array  # noqa: E402

BUDGET_S = 10.0


def emit(mesh, phase, ms, checks):
    print(json.dumps({"lib": "pydec", "mesh": mesh, "phase": phase,
                      "ms": ms, "check": [float(c) for c in checks]}),
          flush=True)


def best_of(repeats, run):
    """Best time in ms of `run()` and the last result."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t = time.perf_counter()
        result = run()
        elapsed = time.perf_counter() - t
        best = min(best, elapsed * 1e3)
        if elapsed > BUDGET_S:
            break
    return best, result


def bench_mesh(directory, name, repeats):
    V = read_array(f"{directory}/{name}.vertices.arr")
    E = read_array(f"{directory}/{name}.elements.arr")
    n = E.shape[1] - 1

    # one loop for all three: PyDEC's geometry is the slow part
    best = {"topology": float("inf"), "geometry": float("inf"),
            "build": float("inf")}
    for _ in range(repeats):
        t0 = time.perf_counter()
        sc = simplicial_complex((V, E))
        t1 = time.perf_counter()
        sc.construct_hodge()
        t2 = time.perf_counter()
        best["topology"] = min(best["topology"], (t1 - t0) * 1e3)
        best["geometry"] = min(best["geometry"], (t2 - t1) * 1e3)
        best["build"] = min(best["build"], (t2 - t0) * 1e3)
        if t2 - t0 > BUDGET_S:
            break
    emit(name, "topology", best["topology"],
         [sc[k].num_simplices for k in range(n + 1)])
    emit(name, "geometry", best["geometry"],
         [sc[k].star.diagonal().sum() for k in range(n + 1)])
    emit(name, "build", best["build"], [])

    for k in range(n + 1):
        try:
            ms, M = best_of(repeats, lambda: whitney_innerproduct(sc, k))
        except Exception as e:  # e.g. an unpatched PyDEC under NumPy 2
            print(json.dumps({"lib": "pydec", "mesh": name,
                              "phase": f"whitney_{k}",
                              "error": f"{type(e).__name__}: {e}"}),
                  flush=True)
            continue
        M = M.tocsr()
        M.sum_duplicates()
        emit(name, f"whitney_{k}", ms, [np.sqrt((M.data ** 2).sum())])


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: pydec_bench.py MESH_DIR REPEATS [MESH...]")
    directory, repeats = sys.argv[1], int(sys.argv[2])
    names = sys.argv[3:]
    if not names:
        with open(f"{directory}/meshes.txt") as f:
            names = [line.split()[0] for line in f if line.strip()]
    for name in names:
        bench_mesh(directory, name, repeats)


if __name__ == "__main__":
    main()
