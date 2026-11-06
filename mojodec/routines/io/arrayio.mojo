# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
PyDEC array files (`mojodec.routines.io.arrayio`).
==================================================
Reader and writer for the array file format of `pydec.io.arrayio`, so meshes
and matrices written by PyDEC load in MojoDEC and vice versa.

A file is a header (a line holding the number of `key=value` lines, then those
lines) followed by data:

- `format=basic`: a 2-D array as whitespace-separated text, one row per line.
- `type=ndarray`, `format=ascii`: whitespace-separated text.
- `type=ndarray`, `format=binary`: raw little-endian values.
- `type=sparse`: `sptype` is `csr`, `csc`, or `coo`; the three component
  arrays follow as nested array files.

Dense arrays are converted to the requested NuMojo dtype on reading.
"""

from std.memory import bitcast
from std.sys import size_of

import numojo as nm
from numojo.prelude import *

from mojodec.core.sparse import CSRMatrix


def read_array[
    dtype: DType = DType.float64
](path: String) raises -> NDArray[dtype]:
    """Read a dense array written by PyDEC's `write_array`.

    ```mojo
    var V = read_array("mesh.vertices")                  # float64
    var E = read_array[DType.int]("mesh.elements")
    ```

    Parameters:
        dtype: The element type of the returned array; values are converted.

    Args:
        path: The array file to read.

    Returns:
        The array, with the shape recorded in the header's `dims`.

    Raises:
        If the file cannot be opened, holds a sparse matrix, or is
        malformed: a bad header, an unknown `format` or `type`, a `basic`
        array that is not 2-D, an unsupported binary `dtype`, or too little
        data for `dims`.
    """
    var reader = _Reader(path)
    var header = reader.read_header()
    if _get(header, "type") == "sparse":
        raise Error("read_array: file holds a sparse matrix; use read_sparse")
    var parsed = reader.read_dense(header)
    var dims = parsed[0].copy()
    ref values = parsed[1]
    var shape = NDArrayShape(dims) if len(dims) > 0 else NDArrayShape(1)
    var out = nm.zeros[dtype](shape)
    for i in range(len(values)):
        out.unsafe_set(i, Scalar[dtype](values[i]))
    return out^


def read_sparse(path: String) raises -> CSRMatrix:
    """Read a sparse matrix (csr, csc, or coo) written by PyDEC's `write_array`.

    Args:
        path: The array file to read.

    Returns:
        The matrix, converted to CSR.

    Raises:
        If the file cannot be opened, its `dims` are not 2-D, its `sptype`
        is not `csr`, `csc`, or `coo`, or a header or component array is
        malformed or truncated.
    """
    var reader = _Reader(path)
    var header = reader.read_header()
    return reader.read_sparse(header)


def read_header(path: String) raises -> Dict[String, String]:
    """The header of an array file as a dictionary.

    Args:
        path: The array file to read.

    Returns:
        The header's `key=value` lines, keyed by `key`.

    Raises:
        If the file cannot be opened, ends early, or its first line is not a
        line count, or a header line is not `key=value`.
    """
    var reader = _Reader(path)
    return reader.read_header()


def write_array[
    dtype: DType
](path: String, A: NDArray[dtype], format: String = "binary") raises:
    """Write a dense array in PyDEC's format.

    `format` is `basic` (text, rank 1 or 2 only), `ascii`, or `binary`.
    A rank-1 array in `basic` format is written as a single row.

    Parameters:
        dtype: The element type of `A`: `float64`, `float32`, `int64`,
            `int`, `int32`, `int8`, or `uint8`.

    Args:
        path: The file to write; an existing file is replaced.
        A: The array to write.
        format: `basic`, `ascii`, or `binary` (little-endian raw values).

    Raises:
        If `format` is unknown, `basic` is used for rank > 2, `dtype` has no
        NumPy name here, or the file cannot be written.
    """
    var out = List[UInt8]()
    _write_dense(out, A, format)
    _write_file(path, out)


def write_sparse(path: String, A: CSRMatrix, format: String = "binary") raises:
    """Write a sparse matrix as a PyDEC `csr` array file (`ascii` or `binary`).

    Values are written as `float64`, indices and row pointers as `int32`.

    Args:
        path: The file to write; an existing file is replaced.
        A: The matrix to write.
        format: `ascii` or `binary`.

    Raises:
        If `format` is not `ascii` or `binary`, or the file cannot be
        written.
    """
    if format != "ascii" and format != "binary":
        raise Error(
            "write_sparse: sparse matrices require ascii or binary format"
        )
    var out = List[UInt8]()
    var header = Dict[String, String]()
    header["type"] = "sparse"
    header["sptype"] = "csr"
    header["dims"] = String(A.nrows, ",", A.ncols)
    header["format"] = format
    _write_header(out, header^)
    var data = nm.zeros[DType.float64](Shape(A.nnz()))
    var indices = nm.zeros[DType.int32](Shape(A.nnz()))
    var indptr = nm.zeros[DType.int32](Shape(A.nrows + 1))
    for k in range(A.nnz()):
        data.unsafe_set(k, A.data[k])
        indices.unsafe_set(k, Int32(A.indices[k]))
    for i in range(A.nrows + 1):
        indptr.unsafe_set(i, Int32(A.indptr[i]))
    _write_dense(out, data, format)
    _write_dense(out, indices, format)
    _write_dense(out, indptr, format)
    _write_file(path, out)


# ===----------------------------------------------------------------------=== #
# Private helpers
# ===----------------------------------------------------------------------=== #


struct _Reader:
    var bytes: List[UInt8]
    var pos: Int

    def __init__(out self, path: String) raises:
        with open(path, "r") as f:
            self.bytes = f.read_bytes()
        self.pos = 0

    def read_line(mut self) raises -> String:
        if self.pos >= len(self.bytes):
            raise Error("arrayio: unexpected end of file")
        var start = self.pos
        while self.pos < len(self.bytes) and self.bytes[self.pos] != UInt8(
            ord("\n")
        ):
            self.pos += 1
        var line = String(unsafe_from_utf8=Span(self.bytes)[start : self.pos])
        self.pos += 1
        return String(line.strip())

    def read_header(mut self) raises -> Dict[String, String]:
        var first = self.read_line()
        var numlines: Int
        try:
            numlines = Int(first)
        except:
            raise Error("arrayio: invalid header line count [" + first + "]")
        var header = Dict[String, String]()
        for i in range(numlines):
            var line = self.read_line()
            var parts = line.split("=")
            if len(parts) != 2:
                raise Error(
                    String("arrayio: header error: line #", i, " [", line, "]")
                )
            header[String(parts[0])] = String(parts[1])
        return header^

    def read_dense(
        mut self, header: Dict[String, String]
    ) raises -> Tuple[List[Int], List[Float64]]:
        var format = _get(header, "format")
        if format not in ("", "basic", "ascii", "binary"):
            raise Error("arrayio: unknown format [" + format + "]")
        var dims = _split_dims(_get(header, "dims"))
        var dtype = _get(header, "dtype")
        var count = 1
        for x in dims:
            count *= x
        if format == "basic":
            if len(dims) != 2:
                raise Error("arrayio: basic format only supports 2d arrays")
            return (dims^, self.read_text(count))
        if _get(header, "type") != "ndarray":
            raise Error("arrayio: unknown array type")
        if format == "ascii":
            return (dims^, self.read_text(count))
        return (dims^, self.read_binary(dtype, count))

    def read_sparse(mut self, header: Dict[String, String]) raises -> CSRMatrix:
        var dims = _split_dims(_get(header, "dims"))
        if len(dims) != 2:
            raise Error("arrayio: invalid sparse dims")
        var sptype = _get(header, "sptype")
        var a = self.read_dense(self.read_header())[1].copy()
        var b = _to_ints(self.read_dense(self.read_header())[1])
        var c = _to_ints(self.read_dense(self.read_header())[1])
        if sptype == "csr":
            return CSRMatrix(dims[0], dims[1], c^, b^, a^)
        if sptype == "csc":
            return CSRMatrix.from_csc(dims[0], dims[1], c, b, a)
        if sptype == "coo":
            return CSRMatrix.from_coo(dims[0], dims[1], b, c, a)
        raise Error(
            "arrayio: only csr, csc, and coo sparse formats are supported"
        )

    def read_text(mut self, count: Int) raises -> List[Float64]:
        var out = List[Float64](capacity=count)
        while len(out) < count:
            while self.pos < len(self.bytes) and _is_space(
                self.bytes[self.pos]
            ):
                self.pos += 1
            if self.pos >= len(self.bytes):
                raise Error("arrayio: too few values in text array")
            var start = self.pos
            while self.pos < len(self.bytes) and not _is_space(
                self.bytes[self.pos]
            ):
                self.pos += 1
            var tok = String(
                unsafe_from_utf8=Span(self.bytes)[start : self.pos]
            )
            out.append(atof(tok))
        # consume the rest of the current line
        while self.pos < len(self.bytes) and self.bytes[self.pos] != UInt8(
            ord("\n")
        ):
            self.pos += 1
        if self.pos < len(self.bytes):
            self.pos += 1
        return out^

    def read_binary(
        mut self, dtype: String, count: Int
    ) raises -> List[Float64]:
        var size = _dtype_size(dtype)
        if self.pos + size * count > len(self.bytes):
            raise Error("arrayio: binary data truncated")
        var out = List[Float64](capacity=count)
        for _ in range(count):
            var u: UInt64 = 0
            for b in range(size):
                u |= UInt64(self.bytes[self.pos + b]) << UInt64(8 * b)
            self.pos += size
            out.append(_decode(dtype, u))
        return out^


def _get(header: Dict[String, String], key: String) -> String:
    return header.get(key, String(""))


def _split_dims(s: String) raises -> List[Int]:
    var out = List[Int]()
    if s == "":
        return out^
    for p in s.split(","):
        out.append(Int(p))
    return out^


def _to_ints(xs: List[Float64]) -> List[Int]:
    var out = List[Int](capacity=len(xs))
    for x in xs:
        out.append(Int(x))
    return out^


def _is_space(b: UInt8) -> Bool:
    return (
        b == UInt8(ord(" "))
        or b == UInt8(ord("\n"))
        or b == UInt8(ord("\t"))
        or b == UInt8(ord("\r"))
    )


def _dtype_size(dtype: String) raises -> Int:
    if dtype in ("float64", "int64", "uint64"):
        return 8
    if dtype in ("float32", "int32", "uint32"):
        return 4
    if dtype in ("int16", "uint16"):
        return 2
    if dtype in ("int8", "uint8", "bool"):
        return 1
    raise Error("arrayio: unsupported dtype [" + dtype + "]")


def _decode(dtype: String, u: UInt64) raises -> Float64:
    if dtype == "float64":
        return bitcast[DType.float64](u)
    if dtype == "float32":
        return Float64(bitcast[DType.float32](UInt32(u)))
    if dtype == "int64":
        return Float64(bitcast[DType.int64](u))
    if dtype == "int32":
        return Float64(bitcast[DType.int32](UInt32(u)))
    if dtype == "int16":
        return Float64(bitcast[DType.int16](UInt16(u)))
    if dtype == "int8":
        return Float64(bitcast[DType.int8](UInt8(u)))
    return Float64(u)


def _numpy_dtype_name[dtype: DType]() raises -> String:
    comptime if dtype == DType.float64:
        return "float64"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64 or dtype == DType.int:
        return "int64"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.uint8:
        return "uint8"
    else:
        raise Error("arrayio: unsupported dtype for writing")


def _write_header(mut out: List[UInt8], var header: Dict[String, String]):
    header["version"] = "1.0"
    var s = String(len(header), "\n")
    for item in header.items():
        s += item.key + "=" + item.value + "\n"
    out.extend(s.as_bytes())


def _write_dense[
    dtype: DType
](mut out: List[UInt8], A: NDArray[dtype], format: String) raises:
    var header = Dict[String, String]()
    var dims = String()
    for d in range(A.ndim):
        if d > 0:
            dims += ","
        dims += String(Int(A.shape.unsafe_get(d)))
    header["dims"] = dims
    header["dtype"] = _numpy_dtype_name[dtype]()
    if format == "basic":
        if A.ndim > 2:
            raise Error(
                "write_array: basic format only works for rank 1 or 2 arrays"
            )
        header["format"] = "basic"
        var cols = Int(A.shape.unsafe_get(A.ndim - 1))
        if A.ndim == 1:
            header["dims"] = String("1,", cols)
        _write_header(out, header^)
        var s = String()
        for i in range(A.size):
            s += _format_value(A.unsafe_get(i))
            s += "\n" if (i + 1) % cols == 0 else " "
        out.extend(s.as_bytes())
        return
    if format != "ascii" and format != "binary":
        raise Error("write_array: unknown format [" + format + "]")
    header["type"] = "ndarray"
    header["rank"] = String(A.ndim)
    header["format"] = format
    _write_header(out, header^)
    if format == "ascii":
        var s = String()
        for i in range(A.size):
            if i > 0:
                s += " "
            s += _format_value(A.unsafe_get(i))
        if A.size > 0:
            s += "\n"
        out.extend(s.as_bytes())
        return
    comptime size = size_of[dtype]()
    for i in range(A.size):
        var u = _bits(A.unsafe_get(i))
        for b in range(size):
            out.append(UInt8((u >> UInt64(8 * b)) & 0xFF))


def _bits[dtype: DType](x: Scalar[dtype]) -> UInt64:
    comptime if dtype == DType.float64:
        return bitcast[DType.uint64](x)
    elif dtype == DType.float32:
        return UInt64(bitcast[DType.uint32](x))
    elif dtype.is_integral():
        return UInt64(Int64(x))
    else:
        return 0


def _format_value[dtype: DType](x: Scalar[dtype]) -> String:
    comptime if dtype.is_integral():
        return String(Int(x))
    else:
        return String(x)


def _write_file(path: String, bytes: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))
