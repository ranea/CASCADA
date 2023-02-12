"""Provide additional (secondary) bit-vector operators.

.. autosummary::
   :nosignatures:

    BvMaj
    BvIf
    PopCount
    Reverse
    PopCountSum2
    PopCountSum3
    PopCountDiff
    LeadingZeros
    LutOperation
    MatrixOperation
"""
import functools
import math

from cascada.bitvector import core
from cascada.bitvector import operation


class BvMaj(operation.SecondaryOperation):
    """The bit-wise majority function.

    The bit-wise majority function ``BvMaj(x, y, z)``
    is defined for each bit ``i`` as the most common
    bit in ``[x[i], y[i], z[i]]``.

    Since `BvMaj` is a `SecondaryOperation`, by default it is fully evaluated
    (i.e., `Operation.eval` is used) when all the inputs are constants (see also
    `context.SecondaryOperationEvaluation`). On the other hand, `pre_eval`
    is always called in the evaluation (even with symbolic inputs).

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import BvMaj
        >>> BvMaj(Constant(0b000, 3), Constant(0b001, 3), Constant(0b011, 3))
        0b001
        >>> BvMaj(Variable("x", 4), Variable("y", 4), Variable("z", 4))
        BvMaj(x, y, z)
        >>> BvMaj(Variable("x", 4), Variable("x", 4), Variable("z", 4))
        x
        >>> BvMaj(Variable("x", 4), Variable("y", 4), Variable("z", 4)).doit()
        (x & y) | (x & z) | (y & z)

    """

    arity = [3, 0]
    is_symmetric = True
    is_simple = True

    @classmethod
    def condition(cls, x, y, z):
        return x.width == y.width == z.width

    @classmethod
    def output_width(cls, x, y, z):
        return x.width

    @classmethod
    def pre_eval(cls, x, y, z):
        if x == y or x == z:
            return x
        elif y == z:
            return y

    @classmethod
    def eval(cls, x, y, z):
        return (x & y) | (x & z) | (y & z)


class BvIf(operation.SecondaryOperation):
    """The bit-wise conditional (If) function.

    The bit-wise conditional function ``BvIf(x, y, z)``
    is defined for each bit ``i`` as ``y[i]`` if ``x[i]=1``
    otherwise ``z[i]``.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import BvIf
        >>> BvIf(Constant(0b01, 2), Constant(0b00, 2), Constant(0b11, 2))
        0b10
        >>> BvIf(Variable("x", 4), Variable("y", 4), Variable("z", 4))
        BvIf(x, y, z)
        >>> BvIf(0, Variable("y", 4), Variable("z", 4))
        z
        >>> BvIf(Variable("x", 4), Variable("y", 4), Variable("z", 4)).doit()
        (x & y) | (~x & z)

    """

    arity = [3, 0]
    is_symmetric = False
    is_simple = True

    @classmethod
    def condition(cls, x, y, z):
        return x.width == y.width == z.width

    @classmethod
    def output_width(cls, x, y, z):
        return x.width

    @classmethod
    def pre_eval(cls, x, y, z):
        if x == operation.BvNot(core.Constant(0, x.width)):
            return y
        elif x == core.Constant(0, x.width):
            return z

    @classmethod
    def eval(cls, x, y, z):
        return (x & y) | ((~x) & z)


def _repeat_pattern(pattern, width):
    """Repeat the pattern until obtain a bit-vector of given width."""
    assert width % pattern.width == 0
    return operation.repeat(pattern, width // pattern.width)


def _pattern01(width):
    """Obtain the pattern 0...01...1 with given 0-width."""
    zeroes = core.Constant(0, width)
    return operation.Concat(zeroes, ~zeroes)


class PopCount(operation.SecondaryOperation):
    """Count the number of 1's in the bit-vector.

    This operation is also known as the hamming weight of a bit-vector.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import PopCount
        >>> PopCount(Constant(0b1010, 4))
        0b010
        >>> PopCount(Constant(0b101, 3))
        0b10
        >>> PopCount(Variable("x", 4)).doit()
        (((x - ((x >> 0x1) & 0x5)) & 0x3) + (((x - ((x >> 0x1) & 0x5)) >> 0x2) & 0x3))[2:]
        >>> PopCount(Variable("x", 3)).doit()
        (0b0 :: (x[0])) + (0b0 :: (x[1])) + (0b0 :: (x[2]))

    """

    arity = [1, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x):
        return x.width.bit_length()

    @classmethod
    def eval(cls, bv):
        # Source: Hacker's Delight

        if bv.width < 4:
            w = cls.output_width(bv)
            return sum([operation.zero_extend(bv[i], w - 1) for i in range(bv.width)])

        # extend the bv until power of 2 length
        original_width = bv.width
        while (bv.width & (bv.width - 1)) != 0:
            bv = operation.zero_extend(bv, 1)
        width_log2 = bv.width.bit_length() - 1

        m_cts = []
        for i in range(width_log2):
            m_cts.append(_repeat_pattern(_pattern01(2 ** i), bv.width))

        if bv.width > 32:
            for i, m in enumerate(m_cts):
                bv = (bv & m) + ((bv >> core.Constant(2 ** i, bv.width)) & m)
            return bv[original_width.bit_length() - 1:]

        for i, m in enumerate(m_cts):
            if i == 0:
                bv = bv - ((bv >> core.Constant(1, bv.width)) & m)
            elif i == 1:
                bv = (bv & m) + ((bv >> core.Constant(2 ** i, bv.width)) & m)  # generic case
            elif i == 2:
                bv = (bv + (bv >> core.Constant(4, bv.width))) & m
            elif i == 3:
                bv = bv + (bv >> core.Constant(8, bv.width))
            elif i == 4:
                bv = bv + (bv >> core.Constant(16, bv.width))

        return bv[original_width.bit_length() - 1:]


class Reverse(operation.SecondaryOperation):
    """Reverse the bit-vector.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import PopCount
        >>> Reverse(Constant(0b1010, 4))
        0x5
        >>> Reverse(Constant(0b001, 3))
        0b100
        >>> Reverse(Variable("x", 4)).doit()  # doctest: +NORMALIZE_WHITESPACE
        (((((x & 0x5) << 0x1) | ((x >> 0x1) & 0x5)) & 0x3) << 0x2) |
        (((((x & 0x5) << 0x1) | ((x >> 0x1) & 0x5)) >> 0x2) & 0x3)
        >>> Reverse(Variable("x", 3)).doit()
        (x[0]) :: (x[1]) :: (x[2])

    """

    arity = [1, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, bv):
        # Source: Hacker's Delight

        if bv.width == 1:
            return bv
        elif bv.width == 2:
            return operation.RotateLeft(bv, 1)
        elif bv.width == 3:
            return operation.Concat(operation.Concat(bv[0], bv[1]), bv[2])

        original_width = bv.width
        while (bv.width & (bv.width - 1)) != 0:
            bv = operation.zero_extend(bv, 1)
        width_log2 = bv.width.bit_length() - 1

        m_cts = []
        for i in range(width_log2):
            m_cts.append(_repeat_pattern(_pattern01(2 ** i), bv.width))

        if bv.width > 32:
            for i, m in list(enumerate(m_cts)):
                bv = ((bv & m) << core.Constant(2 ** i, bv.width)) | ((bv >> core.Constant(2 ** i, bv.width)) & m)
            return bv[:bv.width - original_width]

        for i, m in list(enumerate(m_cts))[:3]:
            bv = ((bv & m) << core.Constant(2 ** i, bv.width)) | \
                 ((bv >> core.Constant(2 ** i, bv.width)) & m)  # generic case

        if len(m_cts) == 4:
            bv = ((bv & m_cts[3]) << core.Constant(8, bv.width)) | ((bv >> core.Constant(8, bv.width)) & m_cts[3])
        elif len(m_cts) == 5:
            rol = operation.RotateLeft
            ror = operation.RotateRight
            bv = ror(bv & m_cts[3], 8) | (rol(bv, 8) & m_cts[3])

        return bv[:bv.width - original_width]


class PopCountSum2(operation.SecondaryOperation):
    """Count the number of 1's of two bit-vectors.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import PopCountSum2
        >>> PopCountSum2(Constant(0b000, 3), Constant(0b001, 3))
        0b001
        >>> PopCountSum2(~Constant(0, 15), ~Constant(0, 15))
        0b11110
        >>> PopCountSum2(Variable("x", 4), Variable("y", 4)).doit()  # doctest: +NORMALIZE_WHITESPACE
        ((x - ((x >> 0x1) & 0x5)) & 0x3) + (((x - ((x >> 0x1) & 0x5)) >> 0x2) & 0x3) +
        ((y - ((y >> 0x1) & 0x5)) & 0x3) + (((y - ((y >> 0x1) & 0x5)) >> 0x2) & 0x3)
        >>> PopCountSum2(Variable("x", 3), Variable("y", 3)).doit()  # doctest: +NORMALIZE_WHITESPACE
        (0b0 :: ((0b0 :: (x[0])) + (0b0 :: (x[1])) + (0b0 :: (x[2])))) +
        (0b0 :: ((0b0 :: (y[0])) + (0b0 :: (y[1])) + (0b0 :: (y[2]))))

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        if x.width < 4:
            return PopCount.output_width(x) + 1
        else:
            return (x.width + y.width).bit_length()

    @classmethod
    def eval(cls, x, y):
        # Source: Hacker's Delight

        if x.width < 4:
            # the HW of a 1-bit/2-bit vector requires 1-bit/2-bit (HW(0b1)=0b1, HW(0b11)=0b10)
            # thus, the sum of two HW of these sizes require an extra bit
            # the HW of a 3-bit vector requires 2-bit (Hw(0b111)=0b11)
            # and the sum of two HW of 3-bit also require an extra bit
            return operation.zero_extend(PopCount(x), 1) + operation.zero_extend(PopCount(y), 1)
        elif x.width > 32:
            width = cls.output_width(x, y)
            x = PopCount(x)
            x = operation.zero_extend(x, width - x.width)
            y = PopCount(y)
            y = operation.zero_extend(y, width - y.width)
            return x + y

        orig_x, orig_y = x, y
        while (x.width & (x.width - 1)) != 0:
            x = operation.zero_extend(x, 1)
        while (y.width & (y.width - 1)) != 0:
            y = operation.zero_extend(y, 1)
        width_log2 = x.width.bit_length() - 1

        m_cts = []
        for i in range(width_log2):
            m_cts.append(_repeat_pattern(_pattern01(2 ** i), x.width))

        bv = core.Constant(0, x.width)
        for i, m in enumerate(m_cts):
            if i == 0:
                x = x - ((x >> core.Constant(1, bv.width)) & m)
                y = y - ((y >> core.Constant(1, bv.width)) & m)
                bv = x + y
            elif i == 1:
                x = (x & m) + ((x >> core.Constant(2 ** i, bv.width)) & m)  # generic case
                y = (y & m) + ((y >> core.Constant(2 ** i, bv.width)) & m)
                bv = x + y
            elif i == 2:
                bv = (bv & m) + ((bv >> core.Constant(4, bv.width)) & m)
            elif i == 3:
                bv = bv + (bv >> core.Constant(8, bv.width))
            elif i == 4:
                bv = bv + (bv >> core.Constant(16, bv.width))

        return bv[cls.output_width(orig_x, orig_y) - 1:]


class PopCountSum3(operation.SecondaryOperation):
    """Count the number of 1's of three bit-vectors.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import PopCountSum3
        >>> PopCountSum3(Constant(0b000, 3), Constant(0b001, 3), Constant(0b011, 3))
        0x3
        >>> PopCountSum3(Variable("x", 4), Variable("y", 4), Variable("z", 4)).doit()  # doctest: +NORMALIZE_WHITESPACE
        ((x - ((x >> 0x1) & 0x5)) & 0x3) + (((x - ((x >> 0x1) & 0x5)) >> 0x2) & 0x3) +
        ((y - ((y >> 0x1) & 0x5)) & 0x3) + (((y - ((y >> 0x1) & 0x5)) >> 0x2) & 0x3) +
        ((z - ((z >> 0x1) & 0x5)) & 0x3) + (((z - ((z >> 0x1) & 0x5)) >> 0x2) & 0x3)
        >>> PopCountSum3(Variable("x", 3), Variable("y", 3), Constant(0, 3)).doit()  # doctest: +NORMALIZE_WHITESPACE
        (0b00 :: ((0b0 :: (x[0])) + (0b0 :: (x[1])) + (0b0 :: (x[2])))) +
        (0b00 :: ((0b0 :: (y[0])) + (0b0 :: (y[1])) + (0b0 :: (y[2]))))

    """

    arity = [3, 0]
    is_symmetric = True
    is_simple = True

    @classmethod
    def condition(cls, x, y, z):
        return x.width == y.width == z.width

    @classmethod
    def output_width(cls, x, y, z):
        if x.width < 4:
            offset = 1
            if x.width == 3:
                offset = 2
            return PopCount.output_width(x) + offset
        else:
            return (x.width + y.width + z.width).bit_length()

    @classmethod
    def eval(cls, x, y, z):
        # Source: Hacker's Delight

        if x.width < 4:
            # the HW of a 1-bit/2-bit vector requires 1-bit/2-bit (HW(0b1)=0b1, HW(0b11)=0b10)
            # thus, the sum of three HW of these sizes require an extra bit (3*0b1=0b11, 3*0b10=0b110)
            # the HW of a 3-bit vector requires 2-bit (Hw(0b111)=0b11)
            # but the sum of three HW of 3-bit require two extra bit (3*0b11 = 0b1001)
            offset = 1
            if x.width == 3:
                offset = 2
            x = operation.zero_extend(PopCount(x), offset)
            y = operation.zero_extend(PopCount(y), offset)
            z = operation.zero_extend(PopCount(z), offset)
            return x + y + z
        elif x.width > 32:
            width = cls.output_width(x, y, z)
            x = PopCount(x)
            x = operation.zero_extend(x, width - x.width)
            y = PopCount(y)
            y = operation.zero_extend(y, width - y.width)
            z = PopCount(z)
            z = operation.zero_extend(z, width - z.width)
            return x + y + z

        orig_x, orig_y, orig_z = x, y, z
        while (x.width & (x.width - 1)) != 0:
            x = operation.zero_extend(x, 1)
        while (y.width & (y.width - 1)) != 0:
            y = operation.zero_extend(y, 1)
        while (z.width & (z.width - 1)) != 0:
            z = operation.zero_extend(z, 1)
        width_log2 = x.width.bit_length() - 1

        m_cts = []
        for i in range(width_log2):
            m_cts.append(_repeat_pattern(_pattern01(2 ** i), x.width))

        bv = core.Constant(0, x.width)
        for i, m in enumerate(m_cts):
            if i == 0:
                x = x - ((x >> core.Constant(1, bv.width)) & m)
                y = y - ((y >> core.Constant(1, bv.width)) & m)
                z = z - ((z >> core.Constant(1, bv.width)) & m)
                bv = x + y + z
            elif i == 1:
                x = (x & m) + ((x >> core.Constant(2 ** i, bv.width)) & m)  # generic case
                y = (y & m) + ((y >> core.Constant(2 ** i, bv.width)) & m)
                z = (z & m) + ((z >> core.Constant(2 ** i, bv.width)) & m)
                bv = x + y + z
            elif i == 2:
                bv = (bv & m) + ((bv >> core.Constant(4, bv.width)) & m)
            elif i == 3:
                bv = bv + (bv >> core.Constant(8, bv.width))
            elif i == 4:
                bv = bv + (bv >> core.Constant(16, bv.width))

        return bv[cls.output_width(orig_x, orig_y, orig_z) - 1:]


class PopCountDiff(operation.SecondaryOperation):
    """Compute the difference of the hamming weight of two words.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import PopCountDiff
        >>> PopCountDiff(Constant(0b011, 3), Constant(0b100, 3))
        0b01
        >>> PopCountDiff(Variable("x", 4), Variable("y", 4)).doit()  # doctest: +NORMALIZE_WHITESPACE
        ((((x - ((x >> 0x1) & 0x5)) & 0x3) + (((x - ((x >> 0x1) & 0x5)) >> 0x2) & 0x3) +
        ((~y - ((~y >> 0x1) & 0x5)) & 0x3) + (((~y - ((~y >> 0x1) & 0x5)) >> 0x2) & 0x3)) - 0x4)[2:]
        >>> PopCountDiff(Variable("x", 3), Variable("y", 3)).doit()  # doctest: +NORMALIZE_WHITESPACE
        ((0b0 :: (x[0])) + (0b0 :: (x[1])) + (0b0 :: (x[2]))) -
        ((0b0 :: (y[0])) + (0b0 :: (y[1])) + (0b0 :: (y[2])))

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width.bit_length()

    @classmethod
    def eval(cls, x, y):
        # Source: Hacker's Delight

        if x.width < 4:
            return PopCount(x) - PopCount(y)
        elif x.width > 32:
            return PopCount(x) - PopCount(y)

        orig_x, orig_y = x, y
        while (x.width & (x.width - 1)) != 0:
            x = operation.zero_extend(x, 1)
        while (y.width & (y.width - 1)) != 0:
            y = operation.zero_extend(y, 1)
        width_log2 = x.width.bit_length() - 1

        m_cts = []
        for i in range(width_log2):
            m_cts.append(_repeat_pattern(_pattern01(2 ** i), x.width))

        bv = core.Constant(0, x.width)
        for i, m in enumerate(m_cts):
            if i == 0:
                x = x - ((x >> core.Constant(1, bv.width)) & m)
                y = (~y) - (((~y) >> core.Constant(1, bv.width)) & m)
                bv = x + y
            elif i == 1:
                x = (x & m) + ((x >> core.Constant(2 ** i, bv.width)) & m)  # generic case
                y = (y & m) + ((y >> core.Constant(2 ** i, bv.width)) & m)
                bv = x + y
            elif i == 2:
                bv = (bv & m) + ((bv >> core.Constant(4, bv.width)) & m)
            elif i == 3:
                bv = bv + (bv >> core.Constant(8, bv.width))
            elif i == 4:
                bv = bv + (bv >> core.Constant(16, bv.width))

        return (bv - y.width)[cls.output_width(orig_x, orig_y) - 1:]


class LeadingZeros(operation.SecondaryOperation):
    """Return a bit-vector with the leading zeros set to one and the rest to zero.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import PopCount
        >>> LeadingZeros(Constant(0b11111, 5))
        0b00000
        >>> LeadingZeros(Constant(0b001, 3))
        0b110
        >>> LeadingZeros(Variable("x", 4)).doit()
        ~(x | (x >> 0x1) | ((x | (x >> 0x1)) >> 0x2))
        >>> LeadingZeros(Variable("x", 3)).doit()
        ~(((0b0 :: x) | ((0b0 :: x) >> 0x1) | (((0b0 :: x) | ((0b0 :: x) >> 0x1)) >> 0x2))[2:])

    """

    arity = [1, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, bv):
        # Source: Hacker's Delight

        if bv.width == 1:
            return ~bv

        original_width = bv.width
        while (bv.width & (bv.width - 1)) != 0:
            bv = operation.zero_extend(bv, 1)
        width_log2 = bv.width.bit_length() - 1

        for i in range(width_log2):
            bv = bv | (bv >> core.Constant(2 ** i, bv.width))
        return ~bv[original_width - 1:]


class LutOperation(operation.SecondaryOperation):
    """Base class for LUT-based operations.

    Given a LUT (look-up table) ``T`` with `Constant` elements,
    the (bit-vector) LUT-based operation ``T(x)``
    (with bit-vector input ``x``) returns the bit-vector of ``T``
    in the position given by the integer value of ``x``.

    Since `LutOperation` is a `SecondaryOperation`, it is only evaluated
    by default when the input is a constant value
    (see also `context.SecondaryOperationEvaluation`).

    This class is not meant to be instantiated but to provide a base
    class for LUT-based operations. To define a LUT-based operation,
    subclass `LutOperation` and provide the class attribute `lut`.

    Attributes:
        lut: the table that defines the LUT-based operation given as a
            list of `Constant` objects.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import LutOperation
        >>> class MyInvertibleLut(LutOperation): lut = [Constant(i, 2) for i in reversed(range(2**2))]
        >>> MyInvertibleLut(Constant(0, 2))
        0b11
        >>> MyInvertibleLut(Variable("a", 2)).doit()
        Ite(a == 0b00, 0b11, Ite(a == 0b01, 0b10, Ite(a == 0b10, 0b01, 0b00)))
        >>> class MyNonInvLut(LutOperation): lut = [Constant(0, 8), ~Constant(0, 8)]
        >>> MyNonInvLut(Variable("a", 1)).doit()
        Ite(a == 0b0, 0x00, 0xff)

    """
    arity = [1, 0]
    is_symmetric = False

    @classmethod
    @property
    def _input_width(cls):
        _iw = math.log2(len(cls.lut))
        assert _iw == int(_iw)
        return _iw

    @classmethod
    @property
    def _output_width(cls):
        _ow = cls.lut[0].width
        assert all(elem.width == _ow for elem in cls.lut)
        return _ow

    @classmethod
    def condition(cls, x):
        return x.width == cls._input_width

    @classmethod
    def output_width(cls, x):
        return cls._output_width

    @classmethod
    def eval(cls, x):
        if x.width != cls._input_width:
            raise ValueError(f"{cls} only supports {cls._input_width}-bit inputs")
        if isinstance(x, core.Constant):
            return cls.lut[int(x)]
        else:
            expr = cls.lut[-1]  # backward iteration
            for i in reversed(range(len(cls.lut) - 1)):
                expr = operation.Ite(
                    operation.BvComp(x, core.Constant(i, x.width)),
                    cls.lut[i],
                    expr)
            return expr


class MatrixOperation(operation.SecondaryOperation):
    """Base class for matrix-vector products.

    Given a binary matrix as a 2-dimensional list ``M`` with 1-bit `Constant`
    elements (``M[i][j]`` is the j-th column element in the i-th row),
    the (bit-vector) matrix-vector product ``M(x)``
    (with bit-vector input ``x``) returns the bit-vector of ``y``
    such the i-th bit of ``y`` is given by
    ``y[i] = (M[i][0] & x[0]) ^ (M[i][1] & x[1]) ^ ... ^ (M[i][n-1] & x[n-1])``
    where ``n`` is the number of columns and the width of ``x``.

    Since `MatrixOperation` is a `SecondaryOperation`, it is only evaluated
    by default when the input is a constant value
    (see also `context.SecondaryOperationEvaluation`).

    This class is not meant to be instantiated but to provide a base
    class for matrix-vector products. To define a matrix-vector product,
    subclass `MatrixOperation` and provide the class attribute `matrix`.

    .. note::
        By default, a `MatrixOperation` subclass takes 1 input operand ``x``,
        but `arity` can be changed to support multiple operands.
        In this case, the list of inputs ``(x_1, ..., x_n)`` are concatenated
        into ``x`` as ``x = x_n :: ... :: x_1``
        (``x_1`` denoting the least-significant part and ``x_n``
        denoting the most-significant part).

        In other words, a `MatrixOperation` subclass with ``arity == [n, 0]``
        and input ``(x_1, ..., x_n)`` computes ``M(x_n :: ... :: x_1)``.


    Alternatively, binary matrices can be implemented as a
    sequence of `BvXor` operations
    (e.g., https://eprint.iacr.org/2021/1400).

    Attributes:
        matrix: the binary matrix as a 2D list of 1-bit `Constant` objects.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import MatrixOperation
        >>> matrix = [[0, 1], [1, 0]]  # reverse
        >>> matrix = [[core.Constant(x, 1) for x in row] for row in matrix]
        >>> class MySquareMatrix(MatrixOperation): matrix = matrix
        >>> MySquareMatrix(Constant(0, 2))
        0b00
        >>> MySquareMatrix(Variable("a", 2)).doit()
        (a[0]) :: (a[1])
        >>> matrix = [[1, 0], [0, 1], [1, 0], [0, 1]]  # 2 bit to 4 bit
        >>> matrix = [[core.Constant(x, 1) for x in row] for row in matrix]
        >>> class MyNonSqMatrix(MatrixOperation): arity, matrix = [2, 0], matrix
        >>> MyNonSqMatrix(Variable("a0", 1), Variable("a1", 1)).doit()
        a1 :: a0 :: a1 :: a0


    .. Implementation details:

        Other matrix-2-XOR papers:
        https://doi.org/10.1007/978-3-030-75539-3_25
        https://tches.iacr.org/index.php/TCHES/article/view/8398
        https://tosc.iacr.org/index.php/ToSC/article/view/8671
        https://tosc.iacr.org/index.php/ToSC/article/view/813

    """
    arity = [1, 0]
    is_symmetric = False

    @classmethod
    @property
    def _input_width(cls):
        num_columns = len(cls.matrix[0])
        assert all(len(row) == num_columns for row in cls.matrix)
        return num_columns

    @classmethod
    @property
    def _output_width(cls):
        return len(cls.matrix)

    @classmethod
    def condition(cls, *args):
        return sum(x.width for x in args) == cls._input_width

    @classmethod
    def output_width(cls, *args):
        return cls._output_width

    @classmethod
    def eval(cls, *args):
        if len(args) == 1:
            x = args[0]
        else:
            x = functools.reduce(operation.Concat, reversed(args))  # args[0] LS part
        if x.width != cls._input_width:
            raise ValueError(f"{cls} only supports inputs with total width {cls._input_width}")
        output_bv = [None for _ in range(len(cls.matrix))]
        num_rows = len(cls.matrix)
        num_columns = len(cls.matrix[0])  # == cls._input_width
        for i in range(num_rows):
            for k in range(num_columns):
                if output_bv[i] is None:
                    output_bv[i] = cls.matrix[i][k] & x[k]
                else:
                    output_bv[i] ^= cls.matrix[i][k] & x[k]
        return functools.reduce(operation.Concat, reversed(output_bv))
