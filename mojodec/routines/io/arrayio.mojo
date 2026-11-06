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

from std.sys import size_of

import numojo as nm
from numojo.prelude import *

from mojodec.core.sparse import CSRMatrix
from mojodec.routines.io.scan import ByteReader, append_le_array


struct ArrayFormat(Equatable, ImplicitlyCopyable, Writable):
    """How `write_array` and `write_mesh` store values.

    ```mojo
    write_array("v.arr", V, format=.ASCII)
    ```
    """

    var _code: UInt8

    comptime BASIC = ArrayFormat(0)
    """Text, one row per line, with no `type` in the header (rank 1 or 2)."""
    comptime ASCII = ArrayFormat(1)
    """Whitespace-separated text."""
    comptime BINARY = ArrayFormat(2)
    """Raw little-endian values."""

    def __init__(out self, code: UInt8):
        """A format from its code; use the named members instead.

        Args:
            code: 0, 1, or 2.
        """
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        """Whether two formats are the same.

        Args:
            other: The other format.

        Returns:
            True if they are equal.
        """
        return self._code == other._code

    def write_to(self, mut writer: Some[Writer]):
        """Write the name used in the file header.

        Args:
            writer: The destination.
        """
        if self == Self.BASIC:
            writer.write("basic")
        elif self == Self.ASCII:
            writer.write("ascii")
        else:
            writer.write("binary")


struct SparseFormat(Equatable, ImplicitlyCopyable, Writable):
    """How `write_sparse` stores a matrix's arrays."""

    var _code: UInt8

    comptime ASCII = SparseFormat(1)
    """Whitespace-separated text."""
    comptime BINARY = SparseFormat(2)
    """Raw little-endian values."""

    def __init__(out self, code: UInt8):
        """A format from its code; use the named members instead.

        Args:
            code: 1 or 2.
        """
        self._code = code

    def __eq__(self, other: Self) -> Bool:
        """Whether two formats are the same.

        Args:
            other: The other format.

        Returns:
            True if they are equal.
        """
        return self._code == other._code

    def array_format(self) -> ArrayFormat:
        """The format of the component arrays.

        Returns:
            `ArrayFormat.ASCII` or `ArrayFormat.BINARY`.
        """
        return ArrayFormat(self._code)

    def write_to(self, mut writer: Some[Writer]):
        """Write the name used in the file header.

        Args:
            writer: The destination.
        """
        writer.write(self.array_format())


def read_array[
    dtype: DType = DType.float64
](path: String) raises -> NDArray[dtype]:
    """Read a dense array written by PyDEC's `write_array`.

    ```mojo
    var V = read_array("mesh.vertices")                  # float64
    var E = read_array[DType.int]("mesh.elements")
    ```

    Binary values are converted from the file's `dtype` directly, so
    64-bit integers keep every digit; when it is `dtype` itself, the data
    is copied as one block.

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
    var reader = ByteReader.open(path, "arrayio")
    var header = _read_header(reader)
    if _get(header, "type") == "sparse":
        raise Error("read_array: file holds a sparse matrix; use read_sparse")
    return _read_dense[dtype](reader, header)


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
    var reader = ByteReader.open(path, "arrayio")
    var header = _read_header(reader)
    var dims = _split_dims(_get(header, "dims"))
    if len(dims) != 2:
        raise Error("arrayio: invalid sparse dims")
    var sptype = _get(header, "sptype")
    var a = _to_floats(_read_dense[DType.float64](reader, _read_header(reader)))
    var b = _to_ints(_read_dense[DType.int](reader, _read_header(reader)))
    var c = _to_ints(_read_dense[DType.int](reader, _read_header(reader)))
    if sptype == "csr":
        return CSRMatrix(dims[0], dims[1], c^, b^, a^)
    if sptype == "csc":
        return CSRMatrix.from_csc(dims[0], dims[1], c, b, a)
    if sptype == "coo":
        return CSRMatrix.from_coo(dims[0], dims[1], b, c, a)
    raise Error("arrayio: only csr, csc, and coo sparse formats are supported")


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
    var reader = ByteReader.open(path, "arrayio")
    return _read_header(reader)


def write_array[
    dtype: DType
](path: String, A: NDArray[dtype], format: ArrayFormat = .BINARY) raises:
    """Write a dense array in PyDEC's format.

    A rank-1 array in `BASIC` format is written as a single row.

    Parameters:
        dtype: The element type of `A`: `float64`, `float32`, `int64`,
            `int`, `int32`, `int8`, or `uint8`.

    Args:
        path: The file to write; an existing file is replaced.
        A: The array to write.
        format: `BASIC` (text, rank 1 or 2 only), `ASCII`, or `BINARY`
            (little-endian raw values).

    Raises:
        If `BASIC` is used for rank > 2, `dtype` has no NumPy name here,
        or the file cannot be written.
    """
    var out = List[Byte]()
    _write_dense(out, A, format)
    _write_file(path, out)


def write_sparse(
    path: String, A: CSRMatrix, format: SparseFormat = .BINARY
) raises:
    """Write a sparse matrix as a PyDEC `csr` array file.

    Values are written as `float64`, indices and row pointers as `int32`.

    Args:
        path: The file to write; an existing file is replaced.
        A: The matrix to write.
        format: `ASCII` or `BINARY`.

    Raises:
        If the file cannot be written.
    """
    var out = List[Byte]()
    var header = Dict[String, String]()
    header["type"] = "sparse"
    header["sptype"] = "csr"
    header["dims"] = String(A.nrows, ",", A.ncols)
    header["format"] = String(format)
    _write_header(out, header^)
    var data = nm.zeros[DType.float64](Shape(A.nnz()))
    var indices = nm.zeros[DType.int32](Shape(A.nnz()))
    var indptr = nm.zeros[DType.int32](Shape(A.nrows + 1))
    for k in range(A.nnz()):
        data.unsafe_set(k, A.data.unsafe_get(k))
        indices.unsafe_set(k, Int32(A.indices.unsafe_get(k)))
    for i in range(A.nrows + 1):
        indptr.unsafe_set(i, Int32(A.indptr.unsafe_get(i)))
    var fmt = format.array_format()
    _write_dense(out, data, fmt)
    _write_dense(out, indices, fmt)
    _write_dense(out, indptr, fmt)
    _write_file(path, out)


# ===----------------------------------------------------------------------=== #
# Private helpers
# ===----------------------------------------------------------------------=== #


def _read_header(mut reader: ByteReader) raises -> Dict[String, String]:
    if reader.pos >= len(reader.data):
        raise Error("arrayio: unexpected end of file")
    var first = String(reader.line().strip())
    var numlines: Int
    try:
        numlines = Int(first)
    except:
        raise Error("arrayio: invalid header line count [" + first + "]")
    var header = Dict[String, String]()
    for i in range(numlines):
        if reader.pos >= len(reader.data):
            raise Error("arrayio: unexpected end of file")
        var line = String(reader.line().strip())
        var parts = line.split("=")
        if len(parts) != 2:
            raise Error(
                String("arrayio: header error: line #", i, " [", line, "]")
            )
        header[String(parts[0])] = String(parts[1])
    return header^


def _read_dense[
    dtype: DType
](mut reader: ByteReader, header: Dict[String, String]) raises -> NDArray[
    dtype
]:
    var format = _get(header, "format")
    if format not in ("", "basic", "ascii", "binary"):
        raise Error("arrayio: unknown format [" + format + "]")
    var dims = _split_dims(_get(header, "dims"))
    var count = 1
    for x in dims:
        count *= x
    # a value takes at least one byte in any format: check before allocating
    if count < 0 or count > len(reader.data) - reader.pos:
        raise Error("arrayio: too little data for dims " + _get(header, "dims"))
    var shape = NDArrayShape(dims) if len(dims) > 0 else NDArrayShape(1)
    var out = nm.zeros[dtype](shape)
    if format == "basic" and len(dims) != 2:
        raise Error("arrayio: basic format only supports 2d arrays")
    if format != "basic" and _get(header, "type") != "ndarray":
        raise Error("arrayio: unknown array type")
    if format != "binary":
        for i in range(count):
            if not reader.more():
                raise Error("arrayio: too few values in text array")
            out.unsafe_set(i, reader.number[dtype]())
        reader.skip_line()
        return out^
    var name = _get(header, "dtype")
    if name == "float64":
        _read_binary[DType.float64](reader, out, count)
    elif name == "float32":
        _read_binary[DType.float32](reader, out, count)
    elif name == "int64":
        _read_binary[DType.int64](reader, out, count)
    elif name == "uint64":
        _read_binary[DType.uint64](reader, out, count)
    elif name == "int32":
        _read_binary[DType.int32](reader, out, count)
    elif name == "uint32":
        _read_binary[DType.uint32](reader, out, count)
    elif name == "int16":
        _read_binary[DType.int16](reader, out, count)
    elif name == "uint16":
        _read_binary[DType.uint16](reader, out, count)
    elif name == "int8":
        _read_binary[DType.int8](reader, out, count)
    elif name == "uint8" or name == "bool":
        _read_binary[DType.uint8](reader, out, count)
    else:
        raise Error("arrayio: unsupported dtype [" + name + "]")
    return out^


def _read_binary[
    stored: DType, dtype: DType
](mut reader: ByteReader, mut out: NDArray[dtype], count: Int) raises:
    """Read `count` values stored as `stored` into `out`, converting."""
    if reader.pos + size_of[stored]() * count > len(reader.data):
        raise Error("arrayio: binary data truncated")
    comptime if stored == dtype:
        reader.binary_into(out, 0, count)
    elif (
        size_of[stored]() == size_of[dtype]()
        and stored.is_integral()
        and dtype.is_integral()
    ):
        # e.g. int64 in the file and DType.int in memory: copy the block,
        # then convert
        var block = nm.zeros[stored](Shape(count))
        reader.binary_into(block, 0, count)
        for i in range(count):
            out.unsafe_set(i, block.unsafe_get(i).cast[dtype]())
    else:
        for i in range(count):
            out.unsafe_set(i, reader.binary[stored]().cast[dtype]())


def _get(header: Dict[String, String], key: String) -> String:
    return header.get(key, String(""))


def _split_dims(s: String) raises -> List[Int]:
    var out = List[Int]()
    if s == "":
        return out^
    for p in s.split(","):
        out.append(Int(p))
    return out^


def _to_floats(A: NDArray[DType.float64]) -> List[Float64]:
    var out = List[Float64](capacity=A.size)
    for i in range(A.size):
        out.append(A.unsafe_get(i))
    return out^


def _to_ints(A: NDArray[DType.int]) -> List[Int]:
    var out = List[Int](capacity=A.size)
    for i in range(A.size):
        out.append(Int(A.unsafe_get(i)))
    return out^


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


def _write_header(mut out: List[Byte], var header: Dict[String, String]):
    header["version"] = "1.0"
    var s = String(len(header), "\n")
    for item in header.items():
        s += item.key + "=" + item.value + "\n"
    out.extend(s.as_bytes())


def _write_dense[
    dtype: DType
](mut out: List[Byte], A: NDArray[dtype], format: ArrayFormat) raises:
    var header = Dict[String, String]()
    var dims = String()
    for d in range(A.ndim):
        if d > 0:
            dims += ","
        dims += String(Int(A.shape.unsafe_get(d)))
    header["dims"] = dims
    header["dtype"] = _numpy_dtype_name[dtype]()
    if format == ArrayFormat.BASIC:
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
            _write_value(s, A.unsafe_get(i))
            s += "\n" if (i + 1) % cols == 0 else " "
        out.extend(s.as_bytes())
        return
    header["type"] = "ndarray"
    header["rank"] = String(A.ndim)
    header["format"] = String(format)
    _write_header(out, header^)
    if format == ArrayFormat.ASCII:
        var s = String()
        for i in range(A.size):
            if i > 0:
                s += " "
            _write_value(s, A.unsafe_get(i))
        if A.size > 0:
            s += "\n"
        out.extend(s.as_bytes())
        return
    comptime if dtype == DType.int:
        # DType.int is written as int64
        var wide = nm.zeros[DType.int64](Shape(A.size))
        for i in range(A.size):
            wide.unsafe_set(i, Int64(A.unsafe_get(i)))
        append_le_array(out, wide)
    else:
        append_le_array(out, A)


def _write_value[dtype: DType](mut s: String, x: Scalar[dtype]):
    comptime if dtype.is_integral():
        s.write(Int(x))
    else:
        s.write(x)


def _write_file(path: String, bytes: List[Byte]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))
