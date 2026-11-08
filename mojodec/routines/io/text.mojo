# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Text tables (`mojodec.routines.io.text`).
=========================================
`loadtxt` for whitespace-separated tables, as NumPy's default `loadtxt`
reads them: any run of spaces or tabs separates columns and `#` starts a
comment. NuMojo's `loadtxt` splits on a single delimiter string, which the
space-padded tables shipped with PyDEC's examples defeat.
"""

import numojo as nm
from numojo.prelude import *


def loadtxt[
    dtype: DType = DType.float64
](path: String) raises -> NDArray[dtype]:
    """Read a whitespace-separated table into a 2-D array (1-D for one column).

    Blank lines and lines holding only a comment are skipped. Values are
    parsed as `Float64` and truncated toward zero for integral `dtype`s.

    Parameters:
        dtype: The element type of the returned array.

    Args:
        path: The text file to read.

    Returns:
        A `(rows, cols)` array, or a `(rows,)` array when there is one
        column.

    Raises:
        If the file cannot be opened, a token is not a number, or rows have
        different numbers of columns.
    """
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var values = List[Float64]()
    # integers apart, so 64-bit ones do not lose digits through Float64
    var ints = List[Int]()
    var rows = 0
    var cols = -1
    # split on "\n": Mojo 1.2's `splitlines()` also broke lines at tabs
    for line in text.split("\n"):
        var body = line
        var hash = line.find("#")
        if hash >= 0:
            body = line[byte=:hash]
        var n = 0
        for tok in body.split():
            comptime if dtype.is_integral():
                ints.append(_parse_int(tok))
            else:
                values.append(atof(tok))
            n += 1
        if n == 0:
            continue
        if cols >= 0 and n != cols:
            raise Error(
                String(
                    "loadtxt: row ",
                    rows,
                    " has ",
                    n,
                    " columns, expected ",
                    cols,
                )
            )
        cols = n
        rows += 1
    var shape = NDArrayShape(rows, cols) if cols > 1 else NDArrayShape(rows)
    var out = nm.zeros[dtype](shape)
    comptime if dtype.is_integral():
        for i in range(len(ints)):
            out.unsafe_set(i, Scalar[dtype](ints[i]))
    else:
        for i in range(len(values)):
            out.unsafe_set(i, Scalar[dtype](values[i]))
    return out^


def _parse_int(tok: StringSlice) raises -> Int:
    """An integer token exactly, or a real one (`3.0`, `1e3`) truncated."""
    try:
        return atol(tok)
    except:
        return Int(atof(tok))
