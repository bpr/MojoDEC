# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# A port of PyDEC (https://github.com/hirani/pydec), BSD 3-Clause License.
# ===----------------------------------------------------------------------=== #
"""
Byte-level reading for the file readers (`mojodec.routines.io.scan`).
=====================================================================
`ByteReader` holds a whole file and a position in it. It parses ASCII
numbers in place, without a `String` per token, and reads binary values of
either byte order, a block at a time when the file's order is the
machine's.
"""

from std.bit import byte_swap
from std.memory import bitcast, unsafe_memcpy
from std.sys import Endian, size_of

import numojo as nm
from numojo.prelude import *


struct ByteReader(Movable):
    """A cursor over the bytes of a file."""

    var data: List[Byte]
    """The file's contents."""
    var pos: Int
    """The offset of the next byte to read."""
    var big_endian: Bool
    """Whether binary values are big-endian."""
    var context: String
    """Prefixes error messages, e.g. `"read_ply"`."""
    var suffix: String
    """Ends error messages, e.g. `" in mesh.msh"`."""

    def __init__(
        out self, var data: List[Byte], context: String, suffix: String = ""
    ):
        """A reader at the start of `data`, for little-endian binary values.

        Args:
            data: The file's contents.
            context: Prefixes error messages.
            suffix: Ends error messages.
        """
        self.data = data^
        self.pos = 0
        self.big_endian = False
        self.context = context
        self.suffix = suffix

    @staticmethod
    def open(path: String, context: String, suffix: String = "") raises -> Self:
        """A reader over the contents of a file.

        Args:
            path: The file.
            context: Prefixes error messages.
            suffix: Ends error messages.

        Returns:
            The reader, at offset 0.

        Raises:
            If the file cannot be read.
        """
        var data: List[Byte]
        with open(path, "r") as f:
            data = f.read_bytes()
        return Self(data^, context, suffix)

    def err(self, message: String) -> Error:
        """An error with this reader's context.

        Args:
            message: What went wrong.

        Returns:
            `"<context>: <message><suffix>"`.
        """
        return Error(String(self.context, ": ", message, self.suffix))

    # ===------------------------------------------------------------------=== #
    # Text
    # ===------------------------------------------------------------------=== #

    def skip_space(mut self):
        """Move past spaces, tabs and line breaks."""
        var p = self.data.unsafe_ptr()
        var n = len(self.data)
        while self.pos < n and _is_space(p[unsafe_offset=self.pos]):
            self.pos += 1

    def more(mut self) -> Bool:
        """Whether anything but white space is left.

        Returns:
            True if a token follows.
        """
        self.skip_space()
        return self.pos < len(self.data)

    def skip_line(mut self):
        """Move past the next line break."""
        var p = self.data.unsafe_ptr()
        var n = len(self.data)
        while self.pos < n and p[unsafe_offset=self.pos] != Byte(ord("\n")):
            self.pos += 1
        if self.pos < n:
            self.pos += 1

    def line(mut self) -> String:
        """The rest of the current line, without its line break (or `\\r`).

        Returns:
            The text up to the line break.
        """
        var start = self.pos
        var p = self.data.unsafe_ptr()
        var n = len(self.data)
        while self.pos < n and p[unsafe_offset=self.pos] != Byte(ord("\n")):
            self.pos += 1
        var end = self.pos
        if self.pos < n:
            self.pos += 1
        if end > start and p[unsafe_offset=end - 1] == Byte(ord("\r")):
            end -= 1
        return String(unsafe_from_utf8=Span(self.data)[start:end])

    def token(mut self) raises -> String:
        """The next whitespace-delimited token.

        Returns:
            The token.

        Raises:
            If only white space is left.
        """
        var r = self._token()
        return String(unsafe_from_utf8=Span(self.data)[r[0] : r[1]])

    def float_(mut self) raises -> Float64:
        """The next token, as a float.

        Returns:
            The value.

        Raises:
            If only white space is left, or the token is not a number.
        """
        var r = self._token()
        try:
            return atof(
                StringSlice(unsafe_from_utf8=Span(self.data)[r[0] : r[1]])
            )
        except:
            raise self.err(String("not a number: ", self._text(r[0], r[1])))

    def int_(mut self) raises -> Int:
        """The next token, as an integer.

        Returns:
            The value.

        Raises:
            If only white space is left, or the token is not an integer.
        """
        var r = self._token()
        var v = self._digits(r[0], r[1])
        if v:
            return v.value()
        try:
            return atol(
                StringSlice(unsafe_from_utf8=Span(self.data)[r[0] : r[1]])
            )
        except:
            raise self.err(String("not an integer: ", self._text(r[0], r[1])))

    def number[dtype: DType](mut self) raises -> Scalar[dtype]:
        """The next token, as a `dtype` value.

        For an integral `dtype`, a token written as an integer is parsed
        exactly; one with a fraction or exponent is truncated toward zero.

        Parameters:
            dtype: The value's type.

        Returns:
            The value.

        Raises:
            If only white space is left, or the token is not a number.
        """
        comptime if dtype.is_integral():
            var r = self._token()
            var v = self._digits(r[0], r[1])
            if v:
                return Scalar[dtype](v.value())
            var text = StringSlice(
                unsafe_from_utf8=Span(self.data)[r[0] : r[1]]
            )
            try:
                if self._integral(r[0], r[1]):
                    # more digits than _digits takes
                    return Scalar[dtype](atol(text))
                return Scalar[dtype](Int(atof(text)))
            except:
                raise self.err(String("not a number: ", text))
        else:
            return Scalar[dtype](self.float_())

    def _integral(self, start: Int, end: Int) -> Bool:
        """Whether `data[start:end]` is an optional sign and digits."""
        var p = self.data.unsafe_ptr()
        var i = start
        if i < end and (
            p[unsafe_offset=i] == Byte(ord("-"))
            or p[unsafe_offset=i] == Byte(ord("+"))
        ):
            i += 1
        if i == end:
            return False
        while i < end:
            var c = p[unsafe_offset=i]
            if c < Byte(ord("0")) or c > Byte(ord("9")):
                return False
            i += 1
        return True

    def _digits(self, start: Int, end: Int) -> Optional[Int]:
        """The integer in `data[start:end]` if it is an optional sign and 1
        to 18 decimal digits, else None."""
        var p = self.data.unsafe_ptr()
        var i = start
        var negative = False
        if i < end and (
            p[unsafe_offset=i] == Byte(ord("-"))
            or p[unsafe_offset=i] == Byte(ord("+"))
        ):
            negative = p[unsafe_offset=i] == Byte(ord("-"))
            i += 1
        if i == end or end - i > 18:
            return None
        var v = 0
        while i < end:
            var c = Int(p[unsafe_offset=i]) - ord("0")
            if c < 0 or c > 9:
                return None
            v = 10 * v + c
            i += 1
        return -v if negative else v

    def _token(mut self) raises -> Tuple[Int, Int]:
        self.skip_space()
        var start = self.pos
        var p = self.data.unsafe_ptr()
        var n = len(self.data)
        while self.pos < n and not _is_space(p[unsafe_offset=self.pos]):
            self.pos += 1
        if start == self.pos:
            raise self.err("file ends early")
        return (start, self.pos)

    def _text(self, start: Int, end: Int) -> String:
        return String(unsafe_from_utf8=Span(self.data)[start:end])

    # ===------------------------------------------------------------------=== #
    # Binary
    # ===------------------------------------------------------------------=== #

    def binary[dtype: DType](mut self) raises -> Scalar[dtype]:
        """The next binary value, in the reader's byte order.

        Parameters:
            dtype: The value's type.

        Returns:
            The value.

        Raises:
            If fewer than `size_of[dtype]()` bytes are left.
        """
        comptime size = size_of[dtype]()
        if self.pos + size > len(self.data):
            raise self.err("file ends early")
        var x = Scalar[dtype](0)
        unsafe_memcpy(
            dest=Pointer(to=x).unsafe_bitcast[Byte](),
            src=self.data.unsafe_ptr().unsafe_offset(self.pos),
            count=size,
        )
        self.pos += size
        if self._swapped():
            return _swap(x)
        return x

    def binary_into[
        dtype: DType
    ](mut self, mut out: NDArray[dtype], start: Int, count: Int) raises:
        """Read `count` binary values into `out[start:start + count]` (flat
        indices).

        Parameters:
            dtype: The values' type.

        Args:
            out: The destination.
            start: The first flat index to write.
            count: The number of values.

        Raises:
            If fewer than `count` values are left, or `out` is too small.
        """
        comptime size = size_of[dtype]()
        if start + count > out.size:
            raise self.err("destination too small")
        if self.pos + size * count > len(self.data):
            raise self.err("file ends early")
        unsafe_memcpy(
            dest=out.unsafe_ptr().unsafe_offset(start).unsafe_bitcast[Byte](),
            src=self.data.unsafe_ptr().unsafe_offset(self.pos),
            count=size * count,
        )
        self.pos += size * count
        if self._swapped():
            for i in range(start, start + count):
                out.unsafe_set(i, _swap(out.unsafe_get(i)))

    def require(self, count: Int) raises:
        """Check that `count` more bytes are left, before allocating room
        for them.

        Args:
            count: The number of bytes.

        Raises:
            If fewer are left, or `count` is negative.
        """
        # pos + count could overflow for a count read from a file
        if count < 0 or count > len(self.data) - self.pos:
            raise self.err("file ends early")

    def require_items(self, count: Int, size: Int) raises:
        """Check that `count` items of `size` bytes each are left, without
        computing `count * size`, which can overflow for a count read from
        a file.

        Args:
            count: The number of items.
            size: The size of each, in bytes (positive).

        Raises:
            If fewer are left, or `count` is negative.
        """
        if count < 0 or count > (len(self.data) - self.pos) // size:
            raise self.err("file ends early")

    def skip(mut self, count: Int) raises:
        """Move past `count` bytes.

        Args:
            count: The number of bytes.

        Raises:
            If fewer than `count` bytes are left, or `count` is negative.
        """
        self.require(count)
        self.pos += count

    def _swapped(self) -> Bool:
        return self.big_endian != (Endian.native() == .big)


def write_float(mut s: String, x: Float64):
    """Append `x` as text that reads back as exactly `x`.

    Mojo's `String(Float64)` drops a digit for about one in five
    consecutive integral doubles between about 1.8e16 and 1e23 (e.g. the
    double after 3e16 prints as `3.000000000000001e+16`, which is the
    double before it); those values are written as exact integers.

    Args:
        s: The destination.
        x: The value.
    """
    var text = String(x)
    var a = abs(x)
    if a >= 9007199254740992.0 and a < 1.0e38:  # 2^53: integral from here
        try:
            if atof(text) != x:
                s.write(Int128(x))
                return
        except:
            pass
    s.write(text)


def native_little_endian() -> Bool:
    """Whether this machine is little-endian.

    Returns:
        True on little-endian machines.
    """
    return Endian.native() == .little


def append_le[dtype: DType](mut out: List[Byte], x: Scalar[dtype]):
    """Append `x` in little-endian byte order.

    Parameters:
        dtype: The value's type.

    Args:
        out: The destination.
        x: The value.
    """
    comptime size = size_of[dtype]()
    var v = x if native_little_endian() else _swap(x)
    var at = len(out)
    out.resize(at + size, 0)
    unsafe_memcpy(
        dest=out.unsafe_ptr().unsafe_offset(at),
        src=Pointer(to=v).unsafe_bitcast[Byte](),
        count=size,
    )


def append_le_array[dtype: DType](mut out: List[Byte], A: NDArray[dtype]):
    """Append the elements of `A`, in flat order and little-endian byte
    order.

    Parameters:
        dtype: The element type.

    Args:
        out: The destination.
        A: The values.
    """
    comptime size = size_of[dtype]()
    if not native_little_endian():
        for i in range(A.size):
            append_le(out, A.unsafe_get(i))
        return
    var at = len(out)
    out.resize(at + size * A.size, 0)
    unsafe_memcpy(
        dest=out.unsafe_ptr().unsafe_offset(at),
        src=A.unsafe_ptr().unsafe_bitcast[Byte](),
        count=size * A.size,
    )


def _swap[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    comptime size = size_of[dtype]()
    comptime if size == 1:
        return x
    elif size == 2:
        return bitcast[dtype](byte_swap(bitcast[DType.uint16](x)))
    elif size == 4:
        return bitcast[dtype](byte_swap(bitcast[DType.uint32](x)))
    else:
        comptime assert size == 8, "unsupported value size"
        return bitcast[dtype](byte_swap(bitcast[DType.uint64](x)))


def _is_space(b: Byte) -> Bool:
    return (
        b == Byte(ord(" "))
        or b == Byte(ord("\n"))
        or b == Byte(ord("\r"))
        or b == Byte(ord("\t"))
    )
