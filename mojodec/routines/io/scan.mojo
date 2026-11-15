# ===----------------------------------------------------------------------=== #
# MojoDEC: Discrete Exterior Calculus in Mojo.
# Largely inspired by PyDEC (https://github.com/hirani/pydec).
# ===----------------------------------------------------------------------=== #
"""
Byte-level reading for the file readers (`mojodec.routines.io.scan`).
=====================================================================
`ByteReader` holds a whole file and a position in it. It parses ASCII
numbers in place, without a `String` per token, and reads binary values of
either byte order, a block at a time when the file's order is the
machine's.
"""

from std.bit import byte_swap, pop_count
from std.memory import bitcast, pack_bits, unsafe_memcpy
from std.os import stat
from std.sys import Endian, size_of

import numojo as nm
from numojo.prelude import *

from mojodec.core.parallel import num_chunks, parallel_ranges


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
            # with the size known, the bytes are read in one call (8x
            # faster than reading to the end); then anything the size did
            # not count, for files that are not regular
            data = f.read_bytes(stat(path).st_size)
            var rest = f.read_bytes()
            if rest:
                data.extend(Span(rest))
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

    def skip_tokens(mut self, count: Int):
        """Move past `count` whitespace-delimited tokens (or to the end).

        Args:
            count: The number of tokens.
        """
        var p = self.data.unsafe_ptr()
        var n = len(self.data)
        for _ in range(count):
            self.skip_space()
            while self.pos < n and not _is_space(p[unsafe_offset=self.pos]):
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
        # file bytes need not be UTF-8: invalid sequences become U+FFFD
        return String(from_utf8_lossy=Span(self.data)[start:end])

    def token(mut self) raises -> String:
        """The next whitespace-delimited token.

        Returns:
            The token.

        Raises:
            If only white space is left.
        """
        var r = self._token()
        return String(from_utf8_lossy=Span(self.data)[r[0] : r[1]])

    def float_(mut self) raises -> Float64:
        """The next token, as a float.

        A decimal with a significand of at most 2^53 and a power of ten
        within 10^±22 is read in one pass: the
        significand and the power are exact doubles, so one multiply or
        divide rounds correctly (Clinger's fast path). Anything else goes
        to `atof`.

        Returns:
            The value.

        Raises:
            If only white space is left, or the token is not a number.
        """
        self.skip_space()
        var n = len(self.data)
        if self.pos < n:
            var t = _scan_number(self.data.unsafe_ptr(), self.pos, n)
            if t[2] != _OTHER:
                self.pos = t[0]
                return _token_float(t[1], t[2])
        var r = self._token()
        try:
            # raises on bytes that are not UTF-8, as on any non-number
            return atof(StringSlice(from_utf8=Span(self.data)[r[0] : r[1]]))
        except:
            raise self.err(String("not a number: ", self._text(r[0], r[1])))

    def int_(mut self) raises -> Int:
        """The next token, as an integer.

        Returns:
            The value.

        Raises:
            If only white space is left, or the token is not an integer.
        """
        self.skip_space()
        var n = len(self.data)
        if self.pos < n:
            var t = _scan_number(self.data.unsafe_ptr(), self.pos, n)
            if t[2] == _INT or t[2] == _MINUS_ZERO:
                self.pos = t[0]
                return Int(Int64(t[1]))
        var r = self._token()
        var v = self._digits(r[0], r[1])
        if v:
            return v.value()
        try:
            return atol(StringSlice(from_utf8=Span(self.data)[r[0] : r[1]]))
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
            try:
                var text = StringSlice(from_utf8=Span(self.data)[r[0] : r[1]])
                if self._integral(r[0], r[1]):
                    # more digits than _digits takes
                    return Scalar[dtype](atol(text))
                return Scalar[dtype](Int(atof(text)))
            except:
                raise self.err(String("not a number: ", self._text(r[0], r[1])))
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
        return String(from_utf8_lossy=Span(self.data)[start:end])

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


comptime _INT: UInt8 = 0
"""A token kind: an integer of magnitude at most 2^53."""
comptime _FLOAT: UInt8 = 1
"""A token kind: a decimal with a fraction or an exponent."""
comptime _MINUS_ZERO: UInt8 = 2
"""A token kind: `-0`, which is 0 as an integer and -0.0 as a float."""
comptime _OTHER: UInt8 = 3
"""A token kind: anything else, left to the serial parser."""


struct NumberTokens(Movable):
    """The numbers in a stretch of text, parsed in parallel by
    `parse_numbers`, then read in order like `ByteReader`'s tokens."""

    var values: List[UInt64]
    """Each token's integer (two's complement) or float bits."""
    var kinds: List[UInt8]
    """Each token's kind: `_INT`, `_FLOAT` or `_MINUS_ZERO`."""
    var next: Int
    """The index of the next token to read."""
    var num_floats: Int
    """The number of `_FLOAT` tokens."""

    def __init__(out self):
        """No tokens."""
        self.values = List[UInt64]()
        self.kinds = List[UInt8]()
        self.next = 0
        self.num_floats = 0

    def __init__(
        out self,
        var values: List[UInt64],
        var kinds: List[UInt8],
        num_floats: Int,
    ):
        """These tokens, at the first.

        Args:
            values: Each token's bits.
            kinds: Each token's kind.
            num_floats: The number of `_FLOAT` tokens.
        """
        self.values = values^
        self.kinds = kinds^
        self.next = 0
        self.num_floats = num_floats

    def done(self) -> Bool:
        """Whether every token has been read.

        Returns:
            True at the end.
        """
        return self.next >= len(self.values)

    @always_inline
    def int_(mut self) raises -> Int:
        """The next token, as an integer.

        Returns:
            The value.

        Raises:
            At the end, or if the token is not an integer.
        """
        var t = self.next
        if t >= len(self.values):
            raise Error("file ends early")
        self.next = t + 1
        if self.kinds.unsafe_get(t) == _FLOAT:
            raise Error("not an integer")
        # `-0` is stored as 0
        return Int(Int64(self.values.unsafe_get(t)))

    def rest_are_ints(self) -> Bool:
        """Whether every token left is an integer.

        Returns:
            False if one is written with a fraction or an exponent.
        """
        return self._ints_in(self.next, len(self.kinds))

    def _ints_in(self, start: Int, end: Int) -> Bool:
        """Whether tokens `start ..< end` are all integers."""
        if self.num_floats == 0:
            return True
        var floats = 0
        for t in range(start, end):
            floats += Int(self.kinds.unsafe_get(t) == _FLOAT)
        return floats == 0

    def ints(
        mut self, mut out: List[Int], rows: Int, skip: Int, keep: Int
    ) raises:
        """Read `rows` rows of `skip + keep` integers, appending the last
        `keep` of each row to `out`.

        Args:
            out: The destination.
            rows: The number of rows.
            skip: The integers to drop at the start of each row.
            keep: The integers to keep after them.

        Raises:
            If fewer tokens are left, or one is not an integer.
        """
        var w = skip + keep
        if rows < 0 or rows > (len(self.values) - self.next) // max(w, 1):
            raise Error("file ends early")
        var t0 = self.next
        if not self._ints_in(t0, t0 + rows * w):
            raise Error("not an integer")
        for r in range(rows):
            var t = t0 + r * w + skip
            for j in range(keep):
                out.append(Int(Int64(self.values.unsafe_get(t + j))))
        self.next = t0 + rows * w

    def floats(mut self, mut out: List[Float64], count: Int) raises:
        """Read `count` numbers, appending them to `out` as floats.

        Args:
            out: The destination.
            count: The number of values.

        Raises:
            If fewer tokens are left.
        """
        if count < 0 or count > len(self.values) - self.next:
            raise Error("file ends early")
        for t in range(self.next, self.next + count):
            out.append(
                _token_float(
                    self.values.unsafe_get(t), self.kinds.unsafe_get(t)
                )
            )
        self.next += count

    @always_inline
    def float_(mut self) raises -> Float64:
        """The next token, as a float.

        Returns:
            The value.

        Raises:
            At the end.
        """
        var t = self.next
        if t >= len(self.values):
            raise Error("file ends early")
        self.next = t + 1
        return _token_float(self.values.unsafe_get(t), self.kinds.unsafe_get(t))


@always_inline
def _token_float(v: UInt64, kind: UInt8) -> Float64:
    if kind == _FLOAT:
        return bitcast[DType.float64](v)
    if kind == _INT:
        # exact: the magnitude is at most 2^53
        return Float64(Int64(v))
    return -0.0


def parse_numbers(
    r: ByteReader, start: Int, end: Int, serial_below: Int
) -> Optional[NumberTokens]:
    """The whitespace-separated numbers in `r.data[start:end]`, parsed in
    parallel, with the values `ByteReader.int_` and `float_` would give.

    `data[end]` must not be white space or a digit (Gmsh's `$` ending a
    section). A token the serial reader might read differently (an
    integer above 2^53, text that is not a decimal, a decimal `atof`
    rejects) makes the result None, as does a range of fewer than
    `serial_below` bytes. On one thread the parse is still faster than
    reading token by token.

    Args:
        r: The reader (its position is not used).
        start: The first byte.
        end: The byte after the last.
        serial_below: Ranges of fewer bytes give None; ranges of more are
            split into one chunk per core.

    Returns:
        The tokens, or None.
    """
    if end - start < serial_below:
        return None
    var C = num_chunks(end - start, serial_below)
    var p = r.data.unsafe_ptr()
    # chunk c takes the tokens starting in bounds[c] ..< bounds[c + 1];
    # each bound but the last follows white space
    var bounds = List[Int](length=C + 1, fill=end)
    for c in range(C):
        var b = start + (end - start) * c // C
        while b > start and b < end and not _is_space(p[unsafe_offset=b - 1]):
            b += 1
        bounds[c] = b
    var counts = List[Int](length=C + 1, fill=0)
    var bp = bounds.unsafe_ptr()
    var cp = counts.unsafe_ptr()

    def count(chunk: Int, lo: Int, hi: Int) {p, bp, cp}:
        # tokens start at a byte that is not white space after one that
        # is (each chunk starts after white space): 32 bytes at a time
        var i = bp[unsafe_offset=lo]
        var stop = bp[unsafe_offset=lo + 1]
        var n = 0
        var after_space = UInt32(1)
        while i + 32 <= stop:
            var v = p.unsafe_offset(i).unsafe_load[width=32]()
            var space = pack_bits(
                v.eq(Byte(ord(" ")))
                | v.eq(Byte(ord("\n")))
                | v.eq(Byte(ord("\r")))
                | v.eq(Byte(ord("\t")))
            )
            n += Int(pop_count(~space & ((space << 1) | after_space)))
            after_space = space >> 31
            i += 32
        var prev = after_space == 1
        while i < stop:
            var space = _is_space(p[unsafe_offset=i])
            n += Int(prev and not space)
            prev = space
            i += 1
        cp[unsafe_offset=lo + 1] = n

    parallel_ranges(C, 1, count)
    for c in range(C):
        counts[c + 1] += counts[c]
    var total = counts[C]
    var values = List[UInt64](unsafe_uninit_length=total)
    var kinds = List[UInt8](unsafe_uninit_length=total)
    # each chunk's float count, or -1 if a token is _OTHER
    var floats = List[Int](length=C, fill=0)
    var vp = values.unsafe_ptr()
    var kp = kinds.unsafe_ptr()
    var fp = floats.unsafe_ptr()

    def scan(chunk: Int, lo: Int, hi: Int) {p, bp, cp, vp, kp, fp, end}:
        var i = bp[unsafe_offset=lo]
        var stop = bp[unsafe_offset=lo + 1]
        var at = cp[unsafe_offset=lo]
        var last = cp[unsafe_offset=lo + 1]
        var nf = 0
        while True:
            while i < stop and _is_space(p[unsafe_offset=i]):
                i += 1
            if i >= stop:
                break
            var t = _scan_number(p, i, end)
            if t[2] == _OTHER or at == last:
                fp[unsafe_offset=lo] = -1
                return
            vp[unsafe_offset=at] = t[1]
            kp[unsafe_offset=at] = t[2]
            nf += Int(t[2] == _FLOAT)
            at += 1
            i = t[0]
        fp[unsafe_offset=lo] = nf if at == last else -1

    parallel_ranges(C, 1, scan)
    _ = bounds^
    _ = counts^
    var num_floats = 0
    for c in range(C):
        if floats[c] < 0:
            return None
        num_floats += floats[c]
    return NumberTokens(values^, kinds^, num_floats)


@always_inline
def _scan_number(
    p: Pointer[Byte, _], start: Int, end: Int
) -> Tuple[Int, UInt64, UInt8]:
    """The token at `p[start]` (not white space), read no further than
    `end`: the offset after it, its bits, and its kind (`_OTHER` when the
    serial reader must judge it)."""
    # first the common forms, short integers and decimals without an
    # exponent; a loop per digit beat eight digits a time (SWAR), whose
    # longer chain of dependent operations each token waits for
    var c0 = p[unsafe_offset=start]
    var negative = c0 == Byte(ord("-"))
    var i = start + Int(negative or c0 == Byte(ord("+")))
    var j = i
    var m = UInt64(0)  # wraps past 19 digits; then not used
    while j < end:
        var c = UInt64(p[unsafe_offset=j]) - UInt64(ord("0"))
        if c > 9:
            break
        m = 10 * m + c
        j += 1
    var digits = j - i
    if j < end and p[unsafe_offset=j] == Byte(ord(".")):
        j += 1
        var f0 = j
        while j < end:
            var c = UInt64(p[unsafe_offset=j]) - UInt64(ord("0"))
            if c > 9:
                break
            m = 10 * m + c
            j += 1
        var f = j - f0
        digits += f
        if (
            digits > 0
            and digits <= 19
            and j < end
            and _is_space(p[unsafe_offset=j])
            and m <= UInt64(1) << 53
            and f <= 22
        ):
            var x = Float64(m) / _pow10(f)
            return (j, bitcast[DType.uint64](-x if negative else x), _FLOAT)
    elif (
        digits > 0
        and digits <= 15  # under 2^53
        and j < end
        and _is_space(p[unsafe_offset=j])
    ):
        if not negative:
            return (j, m, _INT)
        if m == 0:
            return (j, UInt64(0), _MINUS_ZERO)
        return (j, UInt64(-Int64(m)), _INT)
    return _scan_decimal(p, start, end)


def _scan_decimal(
    p: Pointer[Byte, _], start: Int, end: Int
) -> Tuple[Int, UInt64, UInt8]:
    """`_scan_number` for the other forms."""
    var i = start
    var negative = False
    var c0 = p[unsafe_offset=i]
    if c0 == Byte(ord("-")) or c0 == Byte(ord("+")):
        negative = c0 == Byte(ord("-"))
        i += 1
    var m: UInt64 = 0
    var nd = 0
    var s = i
    _digit_run(p, i, end, m, nd)
    var ok = i > s
    var integral = True
    var frac = 0
    if i < end and p[unsafe_offset=i] == Byte(ord(".")):
        integral = False
        i += 1
        var f0 = i
        _digit_run(p, i, end, m, nd)
        frac = i - f0
        ok = ok or frac > 0
    var e = 0
    if i < end and (
        p[unsafe_offset=i] == Byte(ord("e"))
        or p[unsafe_offset=i] == Byte(ord("E"))
    ):
        integral = False
        i += 1
        var e_negative = False
        if i < end and (
            p[unsafe_offset=i] == Byte(ord("-"))
            or p[unsafe_offset=i] == Byte(ord("+"))
        ):
            e_negative = p[unsafe_offset=i] == Byte(ord("-"))
            i += 1
        var e_start = i
        while i < end and i - e_start < 4:
            var c = Int(p[unsafe_offset=i]) - ord("0")
            if c < 0 or c > 9:
                break
            e = 10 * e + c
            i += 1
        ok = ok and i > e_start
        e = -e if e_negative else e
    # the token must end at white space before `end`
    if not (ok and i < end and _is_space(p[unsafe_offset=i])):
        return (i, UInt64(0), _OTHER)
    if integral:
        if nd > 19 or m > UInt64(1) << 53:
            return (i, UInt64(0), _OTHER)
        if negative and m == 0:
            return (i, UInt64(0), _MINUS_ZERO)
        var value = Int64(m)
        return (i, UInt64(-value if negative else value), _INT)
    var p10 = e - frac
    if nd == 0:  # every digit 0, whatever the exponent
        return (i, bitcast[DType.uint64](-0.0 if negative else 0.0), _FLOAT)
    if nd <= 19 and m <= UInt64(1) << 53 and -22 <= p10 and p10 <= 22:
        var x = Float64(m)
        x = x * _pow10(p10) if p10 >= 0 else x / _pow10(-p10)
        return (i, bitcast[DType.uint64](-x if negative else x), _FLOAT)
    return _atof_token(p, start, i)


@no_inline
def _atof_token(
    p: Pointer[Byte, _], start: Int, end: Int
) -> Tuple[Int, UInt64, UInt8]:
    """`_scan_number`'s result for a decimal outside the fast path.

    Values below 1e-307 (and 0) are left to the serial reader: the thread
    pool's workers flush subnormal results to zero, so `atof` there gives
    0 where the calling thread gives a subnormal."""
    try:
        var x = atof(
            StringSlice(
                from_utf8=Span(
                    unsafe_ptr=p.unsafe_offset(start), length=end - start
                )
            )
        )
        if abs(x) < 1.0e-307:
            return (end, UInt64(0), _OTHER)
        return (end, bitcast[DType.uint64](x), _FLOAT)
    except:
        return (end, UInt64(0), _OTHER)


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


@always_inline
def _digit_run(
    p: Pointer[Byte, _], mut i: Int, n: Int, mut m: UInt64, mut nd: Int
):
    """Move `i` past the decimal digits at `p[i]` (reading no further than
    `n`), appending them to `m`; `nd` counts the digits from the first
    that is not zero. Past 19 of those, `m` may have wrapped."""
    while i < n:
        var c = UInt64(p[unsafe_offset=i]) - UInt64(ord("0"))
        if c > 9:
            return
        m = 10 * m + c
        nd += Int(m != 0)
        i += 1


@always_inline
def _pow10(k: Int) -> Float64:
    """10^k, exactly, for k from 0 to 22."""
    # every partial product is a power of ten up to 10^22, so exact
    var a = 10.0 if k & 1 else 1.0
    var b = 100.0 if k & 2 else 1.0
    var c = 1.0e4 if k & 4 else 1.0
    var d = 1.0e8 if k & 8 else 1.0
    var e = 1.0e16 if k & 16 else 1.0
    return ((a * b) * (c * d)) * e


def _is_space(b: Byte) -> Bool:
    return (
        b == Byte(ord(" "))
        or b == Byte(ord("\n"))
        or b == Byte(ord("\r"))
        or b == Byte(ord("\t"))
    )
