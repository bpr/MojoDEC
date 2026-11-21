"""Write the matrices molina's tests use, built from Whitney forms on Kuhn
grids, as text files.

    pixi run mojo run -I . -I build tools/export_molina_fixtures.mojo \
        ../molina/tests/data/matrices

Each file is named `<kind>_<dims>.csr` (`hodge1_8x8.csr`) and holds the
matrix's stored entries in CSR order, explicit zeros included, in the
format molina's `tests/data/matrices/README.md` describes: the distinct
values once each, as the shortest text that reads back to the same
double, then one line per row of `column value-index` pairs (a Kuhn
grid's matrices have a few hundred distinct values; Matrix Market took
3.3 times the space). After writing, every file is read back and
compared bit for bit. The kinds:

- `d<k>`: the coboundary `sc[k].d`.
- `mass<k>`: the Whitney mass `whitney_innerproduct(sc, k)`.
- `hodge<k>`: the Hodge stiffness plus mass,
  `M_k + d_k^T M_{k+1} d_k` (`M_n` alone at the top degree).
- `p1`: the P1 Laplacian `d_0^T M_1 d_0` (Neumann).
- `<kind>_dirichlet`: rows and columns restricted to interior simplices;
  `gradient_dirichlet` is `d_0` from interior vertices to interior edges,
  `curlcurl_dirichlet` is `d_1^T M_2 d_1` on interior edges.
"""

from std.sys import argv
from std.memory import bitcast

from mojodec import SimplicialComplex, kuhn_grid, whitney_innerproduct
from molina.sparse import CSRMatrix


# What molina's tests read, one `kind dims` per entry.
comptime SPECS: List[String] = [
    # test_amg
    "p1_dirichlet 8x8",
    "p1_dirichlet 10x10",
    "p1_dirichlet 16x16",
    "p1_dirichlet 20x20",
    "p1_dirichlet 30x30",
    "p1_dirichlet 32x32",
    "p1_dirichlet 64x64",
    "p1_dirichlet 10x10x10",
    "hodge1_dirichlet 8x8",
    "p1 40x40",
    # test_sparse_ldlt
    "hodge0 10x10",
    "mass1 10x10",
    "d1 10x10",
    "d1 20x20",
    "d0 14x14x14",
    "hodge0 8x8",
    "hodge1 8x8",
    "hodge2 8x8",
    "hodge0 4x4x4",
    "hodge1 4x4x4",
    "hodge2 4x4x4",
    "hodge3 4x4x4",
    "mass1 8x8",
    "d1 8x8",
    "mass1 6x6",
    "d1 6x6",
    "p1 7x7",
    "mass0 7x7",
    "hodge1 9x7",
    "hodge1 4x5x3",
    "hodge0 9x8",
    "hodge1 9x8",
    "hodge2 9x8",
    "hodge0 4x4x3",
    "hodge1 4x4x3",
    "hodge2 4x4x3",
    "hodge3 4x4x3",
    "mass1 7x6",
    "d1 7x6",
    "p1 7x6",
    "mass0 7x6",
    "hodge1 5x5x5",
    "mass2 5x5x5",
    "d2 5x5x5",
    # test_lanczos
    "p1 12x10",
    "mass0 12x10",
    "p1_dirichlet 9x9",
    "mass0_dirichlet 9x9",
    "curlcurl_dirichlet 10x10",
    "mass1_dirichlet 10x10",
    "gradient_dirichlet 10x10",
    # test_parallel
    "hodge0 12x12x12",
    "hodge1 7x7x7",
    "hodge0 90x90",
    "mass1 6x6x6",
    "d1 6x6x6",
]


def _interior(sc: SimplicialComplex, k: Int) raises -> List[Int]:
    var boundary = sc.boundary_cells()[k].copy()
    var out = List[Int]()
    for i in range(sc[k].num_simplices):
        if i not in boundary:
            out.append(i)
    return out^


def _hodge(sc: SimplicialComplex, k: Int) raises -> CSRMatrix:
    var A = whitney_innerproduct(sc, k)
    if k < sc.complex_dimension():
        A = A + sc[k].d.transpose() @ whitney_innerproduct(sc, k + 1) @ sc[k].d
    return A^


def _p1(sc: SimplicialComplex) raises -> CSRMatrix:
    return sc[0].d.transpose() @ whitney_innerproduct(sc, 1) @ sc[0].d


def _build(kind: String, sc: SimplicialComplex) raises -> CSRMatrix:
    var base = kind
    var dirichlet = kind.endswith("_dirichlet")
    if dirichlet:
        base = String(kind[byte = 0 : kind.byte_length() - 10])
    if base == "gradient":
        return sc[0].d.select(_interior(sc, 1), _interior(sc, 0))
    var A: CSRMatrix
    var k: Int
    if base == "p1":
        A = _p1(sc)
        k = 0
    elif base == "curlcurl":
        A = sc[1].d.transpose() @ whitney_innerproduct(sc, 2) @ sc[1].d
        k = 1
    elif base.startswith("hodge"):
        k = atol(base[byte=5:])
        A = _hodge(sc, k)
    elif base.startswith("mass"):
        k = atol(base[byte=4:])
        A = whitney_innerproduct(sc, k)
    elif base.startswith("d"):
        if dirichlet:
            raise Error("no Dirichlet form of " + base)
        return sc[atol(base[byte=1:])].d.copy()
    else:
        raise Error("unknown kind " + kind)
    if dirichlet:
        var keep = _interior(sc, k)
        return A.select(keep, keep)
    return A^


def _write_value(mut s: String, x: Float64):
    # integers as integers (the coboundaries are all +-1); otherwise
    # Mojo's shortest round-trip text, which write_float in
    # mojodec/routines/io/scan.mojo shows is exact below 2^53
    if x == 0 and bitcast[DType.uint64](x) != 0:
        s.write("-0.0")
    elif x == Float64(Int(x)) and abs(x) < 9007199254740992.0:
        s.write(Int(x))
    else:
        s.write(x)


def _write_csr(path: String, A: CSRMatrix, comment: String) raises:
    # distinct values by bit pattern, numbered in order of first use
    var ids = Dict[UInt64, Int]()
    var values = List[Float64]()
    var which = List[Int](capacity=A.nnz())
    for t in range(A.nnz()):
        var key = bitcast[DType.uint64](A.data[t])
        var found = ids.get(key)
        if found:
            which.append(found.value())
        else:
            ids[key] = len(values)
            which.append(len(values))
            values.append(A.data[t])
    var s = String("# ", comment, "\n")
    s.write(A.nrows, " ", A.ncols, " ", A.nnz(), " ", len(values), "\n")
    for x in values:
        _write_value(s, x)
        s.write("\n")
    for i in range(A.nrows):
        for t in range(A.indptr[i], A.indptr[i + 1]):
            if t > A.indptr[i]:
                s.write(" ")
            s.write(A.indices[t], " ", which[t])
        s.write("\n")
    with open(path, "w") as f:
        f.write(s)


def _check_csr(path: String, A: CSRMatrix) raises:
    var lines: List[String]
    with open(path, "r") as f:
        lines = [String(l) for l in f.read().split("\n")]
    var k = 0
    while lines[k].startswith("#"):
        k += 1
    var head = lines[k].split()
    if atol(head[0]) != A.nrows or atol(head[1]) != A.ncols:
        raise Error(path + ": wrong shape")
    if atol(head[2]) != A.nnz():
        raise Error(path + ": wrong entry count")
    var nv = atol(head[3])
    var values = List[Float64]()
    for v in range(nv):
        values.append(atof(lines[k + 1 + v]))
    var first = k + 1 + nv
    for i in range(A.nrows):
        var w = lines[first + i].split()
        if len(w) != 2 * (A.indptr[i + 1] - A.indptr[i]):
            raise Error(String(path, ": row ", i, " differs"))
        for q in range(A.indptr[i], A.indptr[i + 1]):
            var p = 2 * (q - A.indptr[i])
            var x = values[atol(w[p + 1])]
            if atol(w[p]) != A.indices[q] or bitcast[DType.uint64](
                x
            ) != bitcast[DType.uint64](A.data[q]):
                raise Error(String(path, ": row ", i, " differs"))


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: export_molina_fixtures.mojo <output directory>")
    var out = String(args[1])
    var total = 0
    for spec in materialize[SPECS]():
        var parts = spec.split()
        var kind = String(parts[0])
        var dims_text = String(parts[1])
        var dims = List[Int]()
        for n in dims_text.split("x"):
            dims.append(atol(n))
        var sc = SimplicialComplex(kuhn_grid(dims))
        var A = _build(kind, sc)
        var path = String(out, "/", kind, "_", dims_text, ".csr")
        _write_csr(
            path,
            A,
            String(kind, " on kuhn_grid([", dims_text.replace("x", ", "), "])"),
        )
        _check_csr(path, A)
        total += A.nnz()
        print(path, A.nrows, "x", A.ncols, A.nnz())
    print(len(materialize[SPECS]()), "matrices,", total, "entries")
