"""Simon-like round functions and their XOR and RX differential models.

Source: Observations on the SIMON block cipher family https://eprint.iacr.org/2015/145
"""
import decimal
import math

from cascada.bitvector import core
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import (
    SecondaryOperation, RotateLeft, BvComp, Ite, zero_extend
)
from cascada.bitvector.secondaryop import PopCount

from cascada.differential.opmodel import OpModel
from cascada.differential.difference import XorDiff, RXDiff


class SimonRF(SecondaryOperation):
    """The non-linear part of the round function of Simon.

    This corresponds to ``f(x) = ((x <<< a) & (x <<< b)) ^ (x <<< c)``,
    where ``(a, b, c) = (8, 1, 2)``.
    """
    a = 8
    b = 1
    c = 2

    arity = [1, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, x):
        return (RotateLeft(x, cls.a) & RotateLeft(x, cls.b)) ^ RotateLeft(x, cls.c)


class XorModelSimonRF(OpModel):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `SimonRF`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.primitives.simon_rf import XorModelSimonRF
        >>> alpha = XorDiff(Constant(0, 16))
        >>> f = XorModelSimonRF(alpha)
        >>> print(f.vrepr())
        XorModelSimonRF(XorDiff(Constant(0x0000, width=16)))
        >>> x = Constant(0, 16)
        >>> f.eval_derivative(x)  # f(x + alpha) - f(x)
        XorDiff(0x0000)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (15, 5, 0, 0)

    """

    diff_type = XorDiff
    op = SimonRF

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `XorDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.bitvector.printing import BvWrapPrinter
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.primitives.simon_rf import XorModelSimonRF
            >>> alpha = XorDiff(Constant(0, 16))
            >>> f = XorModelSimonRF(alpha)
            >>> f.validity_constraint(XorDiff(Constant(0, 16)))
            0b1
            >>> u, v = Variable("u", 16), Variable("v", 16)
            >>> f = XorModelSimonRF(XorDiff(u))
            >>> result = f.validity_constraint(XorDiff(v))
            >>> print(BvWrapPrinter().doprint(result))
            Ite(u == 0xffff,
                (PopCount(v ^ (u <<< 2))[0]) == 0b0,
                BvComp(0x0000,
                       BvOr((v ^ (u <<< 2)) & ~((u <<< 8) | (u <<< 1)),
                            (v ^ (u <<< 2) ^ ((v ^ (u <<< 2)) <<< 7)) & (u <<< 1) & ~(u <<< 8) & (u <<< 15)
                       )
                )
            )
            >>> result.xreplace({u: Constant(0, 16), v: Constant(0, 16)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """
        a, b, c = self.op.a, self.op.b, self.op.c
        n = self.input_diff[0].val.width
        assert math.gcd(n, a - b) == 1 and a > b and n % 2 == 0

        alpha = self.input_diff[0].val
        Rol = RotateLeft
        varibits = Rol(alpha, a) | Rol(alpha, b)
        r = (2 * a - b) % n
        doublebits = (Rol(alpha, b) & (~Rol(alpha, a)) & Rol(alpha, r))

        beta = output_diff.val
        gamma = beta ^ Rol(alpha, c)

        def is_even(x):
            return BvComp(x[0], Constant(0, 1))

        case2 = BvComp(
            Constant(0, n),
            (gamma & (~varibits)) | ((gamma ^ Rol(gamma, a - b)) & doublebits)
        )

        condition = Ite(
            BvComp(alpha, ~Constant(0, n)),
            is_even(PopCount(gamma)),
            case2
        )

        return condition

    def pr_one_constraint(self, output_diff):
        """Return the probability-one constraint for a given output `XorDiff`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.primitives.simon_rf import XorModelSimonRF
            >>> alpha = XorDiff(Constant(0, 16))
            >>> f = XorModelSimonRF(alpha)
            >>> f.pr_one_constraint(XorDiff(Constant(0, 16)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        a, b, c = self.op.a, self.op.b, self.op.c
        n = self.input_diff[0].val.width
        assert math.gcd(n, a - b) == 1 and a > b and n % 2 == 0

        alpha = self.input_diff[0].val
        Rol = RotateLeft
        varibits = Rol(alpha, a) | Rol(alpha, b)
        r = (2 * a - b) % n
        doublebits = (Rol(alpha, b) & (~Rol(alpha, a)) & Rol(alpha, r))

        beta = output_diff.val
        gamma = beta ^ Rol(alpha, c)

        case2 = BvComp(
            Constant(0, n),
            (gamma & (~varibits)) | ((gamma ^ Rol(gamma, a - b)) & doublebits)
        )

        hw = varibits ^ doublebits  # no need to PopCount

        condition = ~BvComp(alpha, ~Constant(0, n))
        condition &= case2
        condition &= BvComp(hw, Constant(0, hw.width))

        return condition

    def bv_weight(self, output_diff):
        """Return the bit-vector weight for a given output `XorDiff`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.bitvector.printing import BvWrapPrinter
            >>> from cascada.bitvector.secondaryop import PopCount
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.primitives.simon_rf import XorModelSimonRF
            >>> alpha = XorDiff(Constant(0, 16))
            >>> f = XorModelSimonRF(alpha)
            >>> f.bv_weight(XorDiff(Constant(0, 16)))
            0b00000
            >>> alpha = XorDiff(Variable("u", 16))
            >>> f = XorModelSimonRF(alpha)
            >>> beta = XorDiff(Variable("v", 16))
            >>> print(BvWrapPrinter().doprint(f.bv_weight(beta)))
            Ite(u == 0xffff,
                0b01111,
                PopCount(((u <<< 8) | (u <<< 1)) ^ ((u <<< 1) & ~(u <<< 8) & (u <<< 15)))
            )

        See also `abstractproperty.opmodel.OpModel.bv_weight`.
        """
        a, b = self.op.a, self.op.b
        n = self.input_diff[0].val.width
        assert math.gcd(n, a - b) == 1 and a > b and n % 2 == 0

        alpha = self.input_diff[0].val
        Rol = RotateLeft
        varibits = Rol(alpha, a) | Rol(alpha, b)
        r = (2 * a - b) % n
        doublebits = (Rol(alpha, b) & (~Rol(alpha, a)) & Rol(alpha, r))

        hw = PopCount(varibits ^ doublebits)
        width = max((n - 1).bit_length(), hw.width)

        value = Ite(
            BvComp(alpha, ~Constant(0, n)),
            Constant(n - 1, width),
            zero_extend(hw, width - hw.width)
        )

        return value

    def max_weight(self):
        n = self.input_diff[0].val.width
        return n - 1  # as an integer

    def weight_width(self):
        n = self.input_diff[0].val.width
        return max((n - 1).bit_length(), PopCount.output_width(core.Variable("x", n)))

    def decimal_weight(self, output_diff):
        self._assert_preconditions_decimal_weight(output_diff)
        dw = decimal.Decimal(int(self.bv_weight(output_diff)))  # num_frac_bits = 0
        self._assert_postconditions_decimal_weight(output_diff, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class RXModelSimonRF(XorModelSimonRF):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `SimonRF`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.primitives.simon_rf import RXModelSimonRF
        >>> alpha = RXDiff(Constant(0, 16))
        >>> f = RXModelSimonRF(alpha)
        >>> print(f.vrepr())
        RXModelSimonRF(RXDiff(Constant(0x0000, width=16)))
        >>> x = Constant(0, 16)
        >>> f.eval_derivative(x)  # f(x + alpha) - f(x)
        RXDiff(0x0000)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (15, 5, 0, 0)

    .. Implementation details:

        For any bitwise operation, (w.l.o.g, unary op f(x))
        ``Pr( RXDiff(d) propagates through f to RXDiff(d') ) =
        Pr( Diff(d) propagates through f to Diff(d') )``, where
        ``RXDiff(d) = RXDiff.from_pair(x, RotateLeft(x, 1) ^ d)``

    """

    diff_type = RXDiff
    op = SimonRF


SimonRF.xor_model = XorModelSimonRF
SimonRF.rx_model = RXModelSimonRF
