"""Manipulate differential models of bit-vector operations."""
import functools
import hashlib
import math
import operator
import fractions
import decimal
import warnings

from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import secondaryop

from cascada import abstractproperty

from cascada.differential import difference


zip = functools.partial(zip, strict=True)
log2_decimal = abstractproperty.opmodel.log2_decimal
make_partial_op_model = abstractproperty.opmodel.make_partial_op_model


class OpModel(abstractproperty.opmodel.OpModel):
    """Represent differential models of bit-vector operations.

    A (bit-vector) differential model of a bit-vector `Operation` :math:`f`
    is a set of bit-vector constraints that models the differential
    probability (DP) of :math:`f`. See `Difference`.

    Internally, this class is a subclass of `abstractproperty.opmodel.OpModel`,
    where the `Property` is a `Difference` type, the pair of input and output
    properties (differences) :math:`(\\alpha, \\beta)` is called
    a differential, and the propagation probability is the differential
    probability.

    A differential model is defined for a type of `Difference`,
    such as `XorDiff` or `RXDiff`.

    This class is not meant to be instantiated but to provide a base
    class for creating differential models.

    Attributes:
        diff_type: the type of `Difference` (alias of
            `abstractproperty.opmodel.OpModel.prop_type`).
        input_diff: a list containing the `Difference` of each bit-vector
            operand (alias of `abstractproperty.opmodel.OpModel.input_prop`).

    .. Implementation details:

        New differential models of bit-vector operations can be easily tested
        within ``cascada.differential.tests.test_opmodel.TestOpModelsSmallWidth`.

        The maximum weight of a differential with n-bit input is n
            DP = # { x  : ... } / 2^{n}
            weight = -log2(DP) = - (log2(#x) - n) = n - log2(#x)
            min(log2(#x)) = 0, max(weight) = n

    """
    diff_type = None

    @classmethod
    @property
    def prop_type(cls):
        # @property decorator does not support initial values like
        # class(...): diff_type = ...
        # that is why @property is applied over prop_type
        return cls.diff_type

    @property
    def input_diff(self):
        return self.input_prop

    def eval_derivative(self, *x):
        """Evaluate the derivative of the underlying operation at the point :math:`x`.

        Given a `Difference` operation :math:`-`  and its inverse :math:`+`,
        the derivative of an `Operation` :math:`f` at the input difference
        :math:`\\alpha` is defined as :math:`f_{\\alpha} (x) = f(x + \\alpha) - f(x)`.

        Return:
            `Difference`: the `Difference` :math:`f(x + \\alpha) - f(x)`
        """
        assert all(isinstance(d, core.Term) for d in x)
        f_x = self.__class__.op(*x)
        y = [a_i.get_pair_element(x_i) for x_i, a_i in zip(x, self.input_diff)]
        f_y = self.__class__.op(*y)
        return self.__class__.diff_type.from_pair(f_x, f_y)


class PartialOpModel(abstractproperty.opmodel.PartialOpModel, OpModel):
    """Represent `differential.opmodel.OpModel` of `PartialOperation`.

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.differential.opmodel import XorModelBvAddCt, RXModelBvShlCt, make_partial_op_model
        >>> make_partial_op_model(XorModelBvAddCt, tuple([None, Constant(1, 4)])).__name__
        'XorModelBvAddCt_{·, 0x1}'
        >>> make_partial_op_model(RXModelBvShlCt, tuple([None, Constant(1, 4)])).__name__
        'RXModelBvShlCt_{·, 0x1}'

    See also `abstractproperty.opmodel.PartialOpModel`.
    """
    pass


# --------------------------
# ----- XorDiff models -----
# --------------------------


class XorModelId(abstractproperty.opmodel.ModelIdentity, OpModel):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `BvIdentity`.

    See also `ModelIdentity`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelId
        >>> alpha, beta = XorDiff(Variable("a", 4)), XorDiff(Variable("b", 4))
        >>> f = XorModelId(alpha)
        >>> print(f.vrepr())
        XorModelId(XorDiff(Variable('a', width=4)))
        >>> f.validity_constraint(beta)
        a == b
        >>> x = Constant(0, 4)
        >>> f.eval_derivative(x)  # f(x + alpha) - f(x)
        XorDiff(Id(a))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    diff_type = difference.XorDiff


class XorModelBvAdd(OpModel):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `BvAdd`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelBvAdd
        >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
        >>> f = XorModelBvAdd(alpha)
        >>> print(f.vrepr())
        XorModelBvAdd([XorDiff(Constant(0b0000, width=4)), XorDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (3, 2, 0, 0)

    """

    diff_type = difference.XorDiff
    op = operation.BvAdd

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `XorDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAdd
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvAdd(alpha)
            >>> beta = XorDiff(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 4), Variable("a1", 4), Variable("b", 4)
            >>> alpha = XorDiff(a0), XorDiff(a1)
            >>> f = XorModelBvAdd(alpha)
            >>> beta = XorDiff(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            ((~(a0 << 0x1) ^ (a1 << 0x1)) & (~(a0 << 0x1) ^ (b << 0x1)) & (a0 ^ a1 ^ b ^ (a0 << 0x1))) == 0x0
            >>> result.xreplace({a0: Constant(0, 4), a1: Constant(0, 4), b: Constant(0, 4)})
            0b1
            >>> a1 = Constant(0, 4)
            >>> alpha = XorDiff(a0), XorDiff(a1)
            >>> f = XorModelBvAdd(alpha)
            >>> beta = XorDiff(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            (~(a0 << 0x1) & ~(b << 0x1) & (a0 ^ b)) == 0x0

        See `OpModel.validity_constraint` for more information.
        """
        a, b = [d.val for d in self.input_diff]
        c = output_diff.val

        one = core.Constant(1, a.width)

        def eq(x, y, z):
            if not isinstance(x, core.Constant):
                if isinstance(y, core.Constant):
                    return eq(y, x, z)
                elif isinstance(z, core.Constant):
                    return eq(z, x, y)

            return (~x ^ y) & (~x ^ z)

        def xor_shift(x, y, z):
            if not isinstance(x, core.Constant):
                if isinstance(y, core.Constant):
                    return xor_shift(y, x, z)
                elif isinstance(z, core.Constant):
                    return xor_shift(z, x, y)

            return (x ^ y ^ z ^ (x << one))

        return operation.BvComp(
            eq(a << one, b << one, c << one) & xor_shift(a, b, c),
            core.Constant(0, a.width))

        # # https://doi.org/10.1007/3-540-36231-2_5
        # dx1, dx2 = [d.val for d in self.input_diff]
        # dy = output_diff.val
        # one = core.Constant(1, dx1.width)
        #
        # impossible = ~( ((dx1^dy) << one) | ((dx2^dy) << one) ) & (dx1^dx2^dy^(dx2<<one))
        #
        # return operation.BvComp(impossible, core.Constant(0, dx1.width))

    def pr_one_constraint(self, output_diff):
        """Return the probability-one constraint for a given output `XorDiff`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAdd
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvAdd(alpha)
            >>> f.pr_one_constraint(XorDiff(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        is_possible = self.validity_constraint(output_diff)

        a, b = [d.val for d in self.input_diff]
        c = output_diff.val
        n = a.width

        def eq(x, y, z):
            if not isinstance(x, core.Constant):
                if isinstance(y, core.Constant):
                    return eq(y, x, z)
                elif isinstance(z, core.Constant):
                    return eq(z, x, y)

            return (~x ^ y) & (~x ^ z)

        # not optimized

        return is_possible & operation.BvComp(
            (~eq(a, b, c))[n-2:],
            core.Constant(0, n - 1)
        )

    def bv_weight(self, output_diff):
        """Return the bit-vector weight for a given output `XorDiff`.

        See also `abstractproperty.opmodel.OpModel.bv_weight`.
        """
        a, b = [d.val for d in self.input_diff]
        c = output_diff.val
        n = a.width

        def eq(x, y, z):
            if not isinstance(x, core.Constant):
                if isinstance(y, core.Constant):
                    return eq(y, x, z)
                elif isinstance(z, core.Constant):
                    return eq(z, x, y)

            return (~x ^ y) & (~x ^ z)

        return secondaryop.PopCount((~eq(a, b, c))[n-2:])  # ignore MSB

        # # https://doi.org/10.1007/3-540-36231-2_5
        # dx1, dx2 = [d.val for d in self.input_diff]
        # dy = output_diff.val
        # one = core.Constant(1, dx1.width)
        #
        # w = ((dx1^dy) << one) | ((dx2^dy) << one)
        #
        # return secondaryop.PopCount(w)

    def weight_constraint(self, output_diff, weight_variable):
        """Return the weight constraint for a given output `XorDiff` and weight `Variable`.

        For the modular addition, the probability of a valid differential
        is :math:`2^{-i}` for some :math:`i`, and the weight (-log2)
        is defined as the exponent :math:`i`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAdd
            >>> n = 4
            >>> alpha = XorDiff(Constant(0, n)), XorDiff(Constant(0, n))
            >>> f = XorModelBvAdd(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(XorDiff(Constant(0, n)), w)
            w == 0b00
            >>> a0, a1, b = Variable("a0", n), Variable("a1", n), Variable("b", n)
            >>> alpha = XorDiff(a0), XorDiff(a1)
            >>> f = XorModelBvAdd(alpha)
            >>> result = f.weight_constraint(XorDiff(b), w)
            >>> result
            w == PopCount(~((~a0 ^ a1) & (~a0 ^ b))[2:])
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), b: Constant(0, n)})
            w == 0b00

        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_diff, weight_variable)

    def max_weight(self):
        width = self.input_diff[0].val.width - 1  # MSB is ignored
        return width  # as an integer

    def weight_width(self):
        n = self.input_diff[0].val.width - 1  # due to n-2
        return n.bit_length()

    def decimal_weight(self, output_diff):
        self._assert_preconditions_decimal_weight(output_diff)
        dw = decimal.Decimal(int(self.bv_weight(output_diff)))  # num_frac_bits = 0
        self._assert_postconditions_decimal_weight(output_diff, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class XorModelBvSub(XorModelBvAdd):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `BvSub`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelBvSub
        >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
        >>> f = XorModelBvSub(alpha)
        >>> print(f.vrepr())
        XorModelBvSub([XorDiff(Constant(0b0000, width=4)), XorDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (3, 2, 0, 0)

    The differential model of the modular substraction is the same as `XorModelBvAdd`
    since ``~(x - y) == ~x + y`` (and `BvNot` preserves differences).
    """

    diff_type = difference.XorDiff
    op = operation.BvSub


class XorModelBvOr(OpModel):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `BvOr`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelBvOr
        >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
        >>> f = XorModelBvOr(alpha)
        >>> print(f.vrepr())
        XorModelBvOr([XorDiff(Constant(0b0000, width=4)), XorDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    """

    diff_type = difference.XorDiff
    op = operation.BvOr

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `XorDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvOr
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvOr(alpha)
            >>> beta = XorDiff(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 4), Variable("a1", 4), Variable("b", 4)
            >>> alpha = XorDiff(a0), XorDiff(a1)
            >>> f = XorModelBvOr(alpha)
            >>> beta = XorDiff(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            (~a0 & ~a1 & b) == 0x0

        See `OpModel.validity_constraint` for more information.
        """
        dx, dy = [d.val for d in self.input_diff]
        dz = output_diff.val
        n = dx.width

        # only bad case: (dx, dy) = (0, 0) -> dz = (1)
        bad_case = (~dx) & (~dy) & dz

        return operation.BvComp(bad_case, core.Constant(0, n))

    def pr_one_constraint(self, output_diff):
        """Return the probability-one constraint for a given output `XorDiff`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvOr
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvOr(alpha)
            >>> f.pr_one_constraint(XorDiff(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        dx, dy = [d.val for d in self.input_diff]
        dz = output_diff.val
        n = dx.width

        # (dx, dy) = (0, 0) -> dz =(0)
        pr1_case = (~dx) & (~dy) & (~dz)

        return operation.BvComp(pr1_case, ~core.Constant(0, n))

    def bv_weight(self, output_diff):
        dx, dy = [d.val for d in self.input_diff]
        return secondaryop.PopCount(dx | dy)

    def weight_constraint(self, output_diff, weight_variable):
        """Return the weight constraint for a given output `XorDiff` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvOr
            >>> n = 4
            >>> alpha = XorDiff(Constant(0, n)), XorDiff(Constant(0, n))
            >>> f = XorModelBvOr(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(XorDiff(Constant(0, n)), w)
            w == 0b000
            >>> a0, a1, b = Variable("a0", n), Variable("a1", n), Variable("b", n)
            >>> alpha = XorDiff(a0), XorDiff(a1)
            >>> f = XorModelBvOr(alpha)
            >>> result = f.weight_constraint(XorDiff(b), w)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            w == PopCount(a0 | a1)
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), b: Constant(0, n)})
            w == 0b000

        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_diff, weight_variable)

    def max_weight(self):
        dx, dy = [d.val for d in self.input_diff]
        return dx.width

    def weight_width(self):
        n = self.input_diff[0].val.width
        return n.bit_length()

    def decimal_weight(self, output_diff):
        self._assert_preconditions_decimal_weight(output_diff)
        dw = decimal.Decimal(int(self.bv_weight(output_diff)))
        self._assert_postconditions_decimal_weight(output_diff, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class XorModelBvAnd(XorModelBvOr):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `BvAnd`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelBvAnd
        >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
        >>> f = XorModelBvAnd(alpha)
        >>> print(f.vrepr())
        XorModelBvAnd([XorDiff(Constant(0b0000, width=4)), XorDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    The differential model of BvAnd` is the same as `XorModelBvOr`
    since ``~(x & y) == ~x | ~y`` (and `BvNot` preserves differences).
    """
    op = operation.BvAnd


class XorModelBvIf(OpModel):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `BvIf`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelBvIf
        >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
        >>> f = XorModelBvIf(alpha)
        >>> print(f.vrepr())
        XorModelBvIf([XorDiff(Constant(0b0000, width=4)), XorDiff(Constant(0b0000, width=4)), XorDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    """

    diff_type = difference.XorDiff
    op = secondaryop.BvIf

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `XorDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvIf
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvIf(alpha)
            >>> beta = XorDiff(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, a2, b = Variable("a0", 4), Variable("a1", 4), Variable("a2", 4), Variable("b", 4)
            >>> alpha = XorDiff(a0), XorDiff(a1), XorDiff(a2)
            >>> f = XorModelBvIf(alpha)
            >>> beta = XorDiff(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            (~a0 & (~a1 ^ a2) & (a1 ^ b)) == 0x0

        See `OpModel.validity_constraint` for more information.
        """
        x, y, z = [d.val for d in self.input_diff]
        w = output_diff.val
        n = x.width

        def eq(a, b):
            return ~a ^ b

        bad_case = ~x & eq(y, z) & eq(~y, w)

        return operation.BvComp(bad_case, core.Constant(0, n))

    def pr_one_constraint(self, output_diff):
        """Return the probability-one constraint for a given output `XorDiff`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvIf
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvIf(alpha)
            >>> f.pr_one_constraint(XorDiff(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        x, y, z = [d.val for d in self.input_diff]
        w = output_diff.val
        n = x.width

        def eq(x, y, z):
            return (~x ^ y) & (~x ^ z)

        pr1_case = ~x & eq(y, z, w)

        return operation.BvComp(pr1_case, ~core.Constant(0, n))

    def bv_weight(self, output_diff):
        x, y, z = [d.val for d in self.input_diff]

        def eq(a, b):
            return ~a ^ b

        pr1 = ~x & eq(y, z)

        return secondaryop.PopCount(~pr1)

    def weight_constraint(self, output_diff, weight_variable):
        """Return the weight constraint for a given output `XorDiff` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvIf
            >>> n = 4
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvIf(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(XorDiff(Constant(0, n)), w)
            w == 0b000
            >>> a0, a1, a2, b = Variable("a0", 4), Variable("a1", 4), Variable("a2", 4), Variable("b", 4)
            >>> alpha = XorDiff(a0), XorDiff(a1), XorDiff(a2)
            >>> f = XorModelBvIf(alpha)
            >>> result = f.weight_constraint(XorDiff(b), w)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            w == PopCount(~(~a0 & (~a1 ^ a2)))
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), a2: Constant(0, n), b: Constant(0, n)})
            w == 0b000


        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_diff, weight_variable)

    def max_weight(self):
        x, y, z = [d.val for d in self.input_diff]
        return x.width

    def weight_width(self):
        n = self.input_diff[0].val.width
        return n.bit_length()

    def decimal_weight(self, output_diff):
        self._assert_preconditions_decimal_weight(output_diff)
        dw = decimal.Decimal(int(self.bv_weight(output_diff)))
        self._assert_postconditions_decimal_weight(output_diff, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class XorModelBvMaj(OpModel):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `BvMaj`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelBvMaj
        >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
        >>> f = XorModelBvMaj(alpha)
        >>> print(f.vrepr())
        XorModelBvMaj([XorDiff(Constant(0b0000, width=4)), XorDiff(Constant(0b0000, width=4)), XorDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    """

    diff_type = difference.XorDiff
    op = secondaryop.BvMaj

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `XorDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvMaj
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvMaj(alpha)
            >>> beta = XorDiff(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, a2, b = Variable("a0", 4), Variable("a1", 4), Variable("a2", 4), Variable("b", 4)
            >>> alpha = XorDiff(a0), XorDiff(a1), XorDiff(a2)
            >>> f = XorModelBvMaj(alpha)
            >>> beta = XorDiff(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            ((~a0 ^ a1) & (~a0 ^ a2) & (a0 ^ b)) == 0x0

        See `OpModel.validity_constraint` for more information.
        """
        x, y, z = [d.val for d in self.input_diff]
        w = output_diff.val
        n = x.width

        def eq(x, y, z):
            return (~x ^ y) & (~x ^ z)

        bad_case = eq(x, y, z) & (x ^ w)

        return operation.BvComp(bad_case, core.Constant(0, n))

    def pr_one_constraint(self, output_diff):
        """Return the probability-one constraint for a given output `XorDiff`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvMaj
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvMaj(alpha)
            >>> f.pr_one_constraint(XorDiff(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        x, y, z = [d.val for d in self.input_diff]
        w = output_diff.val
        n = x.width

        def eq(x, y, z, w):
            return (~x ^ y) & (~x ^ z) & (~x ^ w)

        pr1_case = eq(x, y, z, w)

        return operation.BvComp(pr1_case, ~core.Constant(0, n))

    def bv_weight(self, output_diff):
        x, y, z = [d.val for d in self.input_diff]

        def eq(x, y, z):
            return (~x ^ y) & (~x ^ z)

        pr1 = eq(x, y, z)

        return secondaryop.PopCount(~pr1)

    def weight_constraint(self, output_diff, weight_variable):
        """Return the weight constraint for a given output `XorDiff` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvMaj
            >>> n = 4
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XorModelBvMaj(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(XorDiff(Constant(0, n)), w)
            w == 0b000
            >>> a0, a1, a2, b = Variable("a0", 4), Variable("a1", 4), Variable("a2", 4), Variable("b", 4)
            >>> alpha = XorDiff(a0), XorDiff(a1), XorDiff(a2)
            >>> f = XorModelBvMaj(alpha)
            >>> result = f.weight_constraint(XorDiff(b), w)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            w == PopCount(~((~a0 ^ a1) & (~a0 ^ a2)))
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), a2: Constant(0, n), b: Constant(0, n)})
            w == 0b000


        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_diff, weight_variable)

    def max_weight(self):
        x, y, z = [d.val for d in self.input_diff]
        return x.width

    def weight_width(self):
        n = self.input_diff[0].val.width
        return n.bit_length()

    def decimal_weight(self, output_diff):
        self._assert_preconditions_decimal_weight(output_diff)
        dw = decimal.Decimal(int(self.bv_weight(output_diff)))
        self._assert_postconditions_decimal_weight(output_diff, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class XorModelBvAddCt(PartialOpModel):
    """Represent the `XorDiff` `PartialOpModel` of modular addition by a constant.

    This class is not meant to be instantiated but to provide a base
    class for subclasses generated through `make_partial_op_model`.

    The class attribute ``precision`` suggests how many fraction bits
    are used in the bit-vector weight (the actual number of fraction
    bits is controlled by the property `effective_precision`).
    The default value of ``precision`` is ``3``,
    and lower values lead to simpler formulas with higher errors.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelBvAddCt, make_partial_op_model
        >>> XorModelBvAdd_1 = make_partial_op_model(XorModelBvAddCt, (None, Constant(1, 4)))
        >>> alpha = XorDiff(Constant(0, 4))
        >>> f = XorModelBvAdd_1(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(XorModelBvAddCt, (None, Constant(0b0001, width=4)))(XorDiff(Constant(0b0000, width=4)))
        >>> x = Constant(0, 4)
        >>> f.eval_derivative(x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (6, 4, Decimal('0.08607133205593431'), 1)

    """

    diff_type = difference.XorDiff
    base_op = operation.BvAdd
    # op is a set through make_partial_op_model
    precision = 3

    @property
    def ct(self):
        """The constant operand of the modular addition."""
        if self.__class__.op.fixed_args[0] is not None:
            constant = self.__class__.op.fixed_args[0]
        else:
            constant = self.__class__.op.fixed_args[1]
        assert isinstance(constant, core.Constant) and constant != 0
        return constant

    @property
    def _index_first_one(self):
        index_first_one = 0
        for i in range(0, self.ct.width):
            if self.ct[i] == 1:
                index_first_one = i
                break
        return index_first_one

    @property
    def _effective_width(self):
        return self.ct.width - 1 - self._index_first_one

    @property
    def effective_precision(self):
        """The actual precision (number of fraction bits) used."""
        return max(min(self.__class__.precision, self._effective_width - 2), 0)  # 2 found empirically

    # noinspection PyPep8Naming
    def _validity_constraint(self, output_diff, debug=False):
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.ct
        n = a.width

        one = core.Constant(1, n)

        assert self._effective_width == n - 1

        def carry(x, y):
            # carry[i] = Maj(x[i-1], y[i-1], carry[i-1])
            return (x + y) ^ x ^ y

        # # S_0 well defined
        case11_ = (u << one) & (v << one)  # i-bit is True if S_i = 11*
        case00_ = (~(u << one)) & (~(v << one))  # i-bit is True if S_i = 00*
        case__1 = u ^ v
        a = a << one  # i-bit holds a[i-1]

        local_eq_a = ~(a ^ (a << one))  # i-bit is True if a[i-1] == a[i-2]
        case00_prev = case00_ << one  # i-bit is True if S_{i-1} = 00*

        c = carry(local_eq_a & case00_, (~case00_prev))

        case001 = case00_ & case__1
        bad_case11_ = case11_ & ~(case__1 ^ local_eq_a) & (c | ~case00_prev)

        bad_events = (case001 | bad_case11_)
        is_valid = operation.BvComp(bad_events, core.Constant(0, bad_events.width))

        if debug:
            print("\n\n ~~ ")
            print("u:           ", u.bin())
            print("v:           ", v.bin())
            print("a:           ", a.bin())
            print("case11_:     ", case11_.bin())
            print("case00_:     ", case00_.bin())
            print("case__1:     ", case__1.bin())
            print("case001:     ", case001.bin())
            print("case00_prev: ", case00_prev.bin())
            print("local_eq_a:  ", local_eq_a.bin())
            print("c:           ", c.bin())
            print("bad_case11_: ", bad_case11_.bin())
            print("bad_events:  ", bad_events.bin())
            print("is_valid    :", is_valid.bin())
            print("\n\n")

        return is_valid

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `XorDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAddCt, make_partial_op_model
            >>> XorModelBvAdd_1 = make_partial_op_model(XorModelBvAddCt, (None, Constant(1, 4)))
            >>> alpha = XorDiff(Constant(0, 4))
            >>> f = XorModelBvAdd_1(alpha)
            >>> f.validity_constraint(XorDiff(Constant(0, 4)))
            0b1
            >>> u, v = Variable("u", 4), Variable("v", 4)
            >>> f = XorModelBvAdd_1(XorDiff(u))
            >>> result = f.validity_constraint(XorDiff(v))
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            ((~(u << 0x1) & ~(v << 0x1) & (u ^ v)) | ((u << 0x1) & (v << 0x1) & ~(u ^ v ^ 0x9) &
                ((((0x9 & ~(u << 0x1) & ~(v << 0x1)) + ~((~(u << 0x1) & ~(v << 0x1)) << 0x1)) ^ (0x9 & ~(u << 0x1) &
                ~(v << 0x1)) ^ ~((~(u << 0x1) & ~(v << 0x1)) << 0x1)) | ~((~(u << 0x1) & ~(v << 0x1)) << 0x1)))) == 0x0
            >>> result.xreplace({u: Constant(0, 4), v: Constant(0, 4)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.ct

        effective_width = self._effective_width
        index_one = self._index_first_one

        if effective_width == 0:
            return operation.BvComp(u, v)
        elif effective_width == 1:
            return operation.BvComp(u[index_one:], v[index_one:]) & \
                   operation.BvComp(~u[index_one] ^ (u[index_one+1] ^ v[index_one+1]), core.Constant(1, 1))
        else:
            if index_one == 0:
                c = core.Constant(1, 1)
            else:
                c = operation.BvComp(u[index_one-1:], v[index_one-1:])
            u = difference.XorDiff(u[:index_one])
            v = difference.XorDiff(v[:index_one])
            # reduced_model = type(self)([u], a[:index_one])
            reduced_model = make_partial_op_model(XorModelBvAddCt, (None, a[:index_one]))
            reduced_model = reduced_model(u)
            return c & reduced_model._validity_constraint(v)

    def _pr_one_constraint(self, output_diff):
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.ct
        n = a.width

        assert self._effective_width == n - 1

        one = core.Constant(1, n)

        def all_ones(width):
            return ~ core.Constant(0, width)

        case00_ = (~(u << one)) & (~(v << one))  # i-bit is True if S_i = 00*
        case__1 = u ^ v
        case000 = case00_ & (~case__1)

        return operation.BvComp(case000, all_ones(n))

    def pr_one_constraint(self, output_diff):
        """Return the probability-one constraint for a given output `XorDiff`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAddCt, make_partial_op_model
            >>> XorModelBvAdd_1 = make_partial_op_model(XorModelBvAddCt, (None, Constant(1, 4)))
            >>> alpha = XorDiff(Constant(0, 4))
            >>> f = XorModelBvAdd_1(alpha)
            >>> f.pr_one_constraint(XorDiff(Constant(0, 4)))
            0b1
            >>> u, v = Variable("u", 4), Variable("v", 4)
            >>> f = XorModelBvAdd_1(XorDiff(u))
            >>> result = f.pr_one_constraint(XorDiff(v))
            >>> result
            (~(u << 0x1) & ~(v << 0x1) & ~(u ^ v)) == 0xf
            >>> result.xreplace({u: Constant(0, 4), v: Constant(0, 4)})
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.ct

        effective_width = self._effective_width
        index_one = self._index_first_one

        if effective_width == 0:
            return operation.BvComp(u, v)
        elif effective_width == 1:
            return operation.BvComp(u[index_one:], v[index_one:]) & \
                   operation.BvComp(~u[index_one] ^ (u[index_one+1] ^ v[index_one+1]), core.Constant(1, 1))
        else:
            if index_one == 0:
                c = core.Constant(1, 1)
            else:
                c = operation.BvComp(u[index_one-1:], v[index_one-1:])
            u = difference.XorDiff(u[:index_one])
            v = difference.XorDiff(v[:index_one])
            # reduced_model = type(self)([u], a[:index_one])
            reduced_model = make_partial_op_model(XorModelBvAddCt, (None, a[:index_one]))
            reduced_model = reduced_model(u)
            return c & reduced_model._pr_one_constraint(v)

    # noinspection SpellCheckingInspection
    def weight_constraint(self, output_diff, weight_variable, prefix=None, debug=False, version=2):
        """Return the weight constraint for a given output `XorDiff` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.bitvector.printing import BvWrapPrinter
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAddCt, make_partial_op_model
            >>> width = 4
            >>> XorModelBvAdd_1 = make_partial_op_model(XorModelBvAddCt, (None, Constant(1, width)))
            >>> alpha = XorDiff(Constant(0, width))
            >>> f = XorModelBvAdd_1(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(XorDiff(Constant(0, width)), w)
            0x0 == w
            >>> alpha = XorDiff(Variable("u", width))
            >>> f = XorModelBvAdd_1(alpha)
            >>> beta = XorDiff(Variable("v", width))
            >>> weight_constraint = f.weight_constraint(beta, w, prefix="w_tmp")
            >>> BvWrapPrinter.use_symbols, BvWrapPrinter.new_line_right_parenthesis = True, False
            >>> print(BvWrapPrinter().doprint(weight_constraint))
            &(&(&(&(&(==(-(::(PopCountDiff(((u ^ v) << 0x1) | (((~u & ~v) << 0x1) & ~w_tmp_0lz),
                                           ^(^(w_tmp_1rev + 0x1 + (w_tmp_4rev & w_tmp_1rev),
                                               w_tmp_1rev + 0x1),
                                             w_tmp_4rev & w_tmp_1rev)),
                              0b0),
                           ::(0b0,
                              PopCount(&(&(w_tmp_4rev,
                                           ^(^(w_tmp_1rev + 0x1 + (w_tmp_4rev & w_tmp_1rev),
                                               w_tmp_1rev + 0x1),
                                             w_tmp_4rev & w_tmp_1rev)),
                                         ~(<<(^(^(w_tmp_1rev + 0x1 + (w_tmp_4rev & w_tmp_1rev),
                                                  w_tmp_1rev + 0x1),
                                                w_tmp_4rev & w_tmp_1rev),
                                              0x1)))))),
                         w),
                      w_tmp_0lz == LeadingZeros(~((~u & ~v) << 0x1))),
                    w_tmp_1rev == Reverse((~u & ~v) << 0x1)),
                  ==(w_tmp_2rev,
                     Reverse((~(0x2 ^ u ^ v) >> 0x1) & ((~u & ~v) << 0x1) & ~(((~u & ~v) << 0x1) >> 0x1)))),
                w_tmp_3rev == Reverse((w_tmp_1rev + w_tmp_2rev) ^ w_tmp_1rev ^ w_tmp_2rev)),
              ==(w_tmp_4rev,
                 Reverse(|(-((~(0x2 ^ u ^ v) >> 0x1) & ((~u & ~v) << 0x1) & ~(((~u & ~v) << 0x1) >> 0x1),
                             &(+(0x2 & ~((~u & ~v) << 0x1) & (((~u & ~v) << 0x1) >> 0x1),
                                 0x1 & (((~u & ~v) << 0x1) >> 0x1)),
                               |(w_tmp_3rev,
                                 (~(0x2 ^ u ^ v) >> 0x1) & ((~u & ~v) << 0x1) & ~(((~u & ~v) << 0x1) >> 0x1)))),
                           &(+(0x2 & ~((~u & ~v) << 0x1) & (((~u & ~v) << 0x1) >> 0x1),
                               0x1 & (((~u & ~v) << 0x1) >> 0x1)),
                             ~(|(w_tmp_3rev,
                                 (~(0x2 ^ u ^ v) >> 0x1) & ((~u & ~v) << 0x1) & ~(((~u & ~v) << 0x1) >> 0x1))))))))
            >>> weight_constraint = f.weight_constraint(beta, w, prefix="w_tmp")
            >>> r = {Variable("u", width): Constant(0, width), Variable("v", width): Constant(0, width)}
            >>> print(BvWrapPrinter().doprint(weight_constraint.xreplace(r)))
            &(&(&(&(&(==(-(::(PopCountDiff(0xe & ~w_tmp_0lz,
                                           ^(^(w_tmp_1rev + 0x1 + (w_tmp_4rev & w_tmp_1rev),
                                               w_tmp_1rev + 0x1),
                                             w_tmp_4rev & w_tmp_1rev)),
                              0b0),
                           ::(0b0,
                              PopCount(&(&(w_tmp_4rev,
                                           ^(^(w_tmp_1rev + 0x1 + (w_tmp_4rev & w_tmp_1rev),
                                               w_tmp_1rev + 0x1),
                                             w_tmp_4rev & w_tmp_1rev)),
                                         ~(<<(^(^(w_tmp_1rev + 0x1 + (w_tmp_4rev & w_tmp_1rev),
                                                  w_tmp_1rev + 0x1),
                                                w_tmp_4rev & w_tmp_1rev),
                                              0x1)))))),
                         w),
                      w_tmp_0lz == 0xe),
                    w_tmp_1rev == 0x7),
                  w_tmp_2rev == 0x0),
                w_tmp_3rev == Reverse((w_tmp_1rev + w_tmp_2rev) ^ w_tmp_1rev ^ w_tmp_2rev)),
              w_tmp_4rev == Reverse(-(0x1 & w_tmp_3rev) | (0x1 & ~w_tmp_3rev)))
            >>> BvWrapPrinter.use_symbols, BvWrapPrinter.new_line_right_parenthesis = False, True

        See `OpModel.weight_constraint` for more information.
        """
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.ct

        effective_width = self._effective_width
        index_one = self._index_first_one

        if effective_width <= 1:
            bv_weight = core.Constant(0, 1)
            extra_constraints = []
        else:
            u = difference.XorDiff(u[:index_one])
            v = difference.XorDiff(v[:index_one])
            # reduced_model = type(self)([u], a[:index_one])
            reduced_model = make_partial_op_model(XorModelBvAddCt, (None, a[:index_one]))
            reduced_model = reduced_model(u)
            bv_weight, extra_constraints = reduced_model._bvweight_and_extra_constraints(
                v, prefix=prefix, debug=debug, version=version)

        weight_constraint = operation.BvComp(bv_weight, weight_variable)
        for c in extra_constraints:
            weight_constraint &= c

        return weight_constraint

    def external_vars_weight_constraint(self, output_diff, weight_variable, prefix=None, debug=False, version=2):
        constraint = self.weight_constraint(
            output_diff, weight_variable, prefix=prefix, debug=debug, version=version)
        return [v for v in constraint.atoms(core.Variable) if v != weight_variable]

    def _bvweight_and_extra_constraints(self, output_diff, prefix, debug, version):
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.ct
        n = a.width
        one = core.Constant(1, n)

        assert self._effective_width == n - 1

        assert version in [0, 1, 2]  # 0-reference, 1-w/o extra reverse, 2-s_000 and no HW2 in fr

        are_ct_differences = isinstance(u, core.Constant) and isinstance(v, core.Constant)

        if prefix is None and not are_ct_differences:
            h = hashlib.blake2b(digest_size=16)
            h.update((self.vrepr() + output_diff.vrepr()).encode())
            prefix = "_tmp" + h.hexdigest()

        self._i_auxvar = 0
        extra_constraints = []

        def rev(x):
            if are_ct_differences:
                return secondaryop.Reverse(x)
            else:
                aux = core.Variable("{}_{}rev".format(prefix, self._i_auxvar), x.width)
                self._i_auxvar += 1
                extra_constraints.append(operation.BvComp(aux, secondaryop.Reverse(x)))
                return aux

        def lz(x):
            if are_ct_differences:
                return secondaryop.LeadingZeros(x)
            else:
                aux = core.Variable("{}_{}lz".format(prefix, self._i_auxvar), x.width)
                self._i_auxvar += 1
                extra_constraints.append(operation.BvComp(aux, secondaryop.LeadingZeros(x)))
                return aux

        def carry(x, y):
            return (x + y) ^ x ^ y

        def rev_carry(x, y):
            return rev(carry(rev(x), rev(y)))

        if version in [0, 1]:
            s00_old = (~(u << one)) & (~(v << one))  # i-bit is True if S_{i} = 00*
        else:
            s00_old = ((~u) & (~v)) << one
        s00_ = s00_old & (~lz(~s00_old))  # if x is 001*...*, then lz(x) = 1100...0

        if version == 0:
            e_i1 = s00_ & (~ (s00_ >> one))  # e_{i-1}
            e_ili = ~s00_ & (s00_ >> one)  # e_{i-l_i}
        else:
            e_i1 = s00_old & (~ (s00_old >> one))  # e_{i-1}
            e_ili = ~s00_old & (s00_old >> one)  # e_{i-l_i}

        q = ~((a << one) ^ (u ^ v))  # q[i] = ~(a[i-1]^u[i]^v[i])
        q = ((q >> one) & e_i1)  # q[i-1, i-3] = (a[i-1]^u[i]^v[i], 0, 0)

        if version == 0:
            s = ((a << one) & e_ili) + (a & (s00_ >> one))
        else:
            s = ((a << one) & e_ili) + (a & (s00_old >> one))

        if version == 0:
            d = rev_carry(s00_, q) | q
        else:
            rev_s00_old = rev(s00_old)
            d = rev(carry(rev_s00_old, rev(q))) | q

        w = (q - (s & d)) | (s & (~d))

        if version == 0:
            w = w << one
            h = rev_carry(s00_ << one, w & (s00_ << one))
        elif version == 1:
            rev_w = rev(w) >> one
            rev_h = carry((rev_s00_old + one) >> one, rev_w & (rev(s00_)) >> one)
        else:
            rev_w = rev(w)
            rev_h = carry(rev_s00_old + one, rev_w & rev_s00_old)

        sbnegb = (u ^ v) << one  # i-bit is True if S_{i} = (b, \neg b, *)

        if version == 0:
            int = secondaryop.PopCountDiff(sbnegb | s00_, h)   # or hw(sbminb_) + (hw(s00_) - hw(h))
        else:
            int = secondaryop.PopCountDiff(sbnegb | s00_, rev_h)

        def smart_add(x, y):
            # ignores overflow
            if x.width == y.width:
                return x + y
            elif x.width < y.width:
                return operation.zero_extend(x, y.width - x.width) + y
            else:
                return x + operation.zero_extend(y, x.width - y.width)

        def smart_sub(x, y):
            # ignores overflow
            # cannot be replaced by smart_add(x, -y)
            if x.width == y.width:
                return x - y
            elif x.width < y.width:
                return operation.zero_extend(x, y.width - x.width) - y
            else:
                return x - operation.zero_extend(y, x.width - y.width)

        k = self.effective_precision

        if k == 0:
            int_frac = int
        elif k == 1:
            int = operation.Concat(int, core.Constant(0, 1))
            if version == 0:
                f1 = secondaryop.PopCount(w & h & (~(h >> one)))  # each one adds 2^(-1)
            else:
                f1 = secondaryop.PopCount(rev_w & rev_h & (~(rev_h << one)))
            int_frac = smart_sub(int, f1)
        else:
            two = core.Constant(2, n)
            three = core.Constant(3, n)
            four = core.Constant(4, n)

            if version == 0:
                f12 = secondaryop.PopCountSum2(
                    w & h & (~(h >> one)),
                    w & h & ((~(h >> one)) | (~(h >> two)) & (h >> one))
                )  # each one adds 2^(-2), that's why ~(h >> one) need to be counted twice
            elif version == 1:
                f12 = secondaryop.PopCountSum2(
                    rev_w & rev_h & (~(rev_h << one)),
                    rev_w & rev_h & ((~(rev_h << one)) | (~(rev_h << two)) & (rev_h << one))
                )
            else:
                f12 = secondaryop.PopCount(
                    # ( ( rev_w & rev_h & (~(rev_h << one)) ) >> one ) |
                    ( ( (rev_w & rev_h) >> one) & (~rev_h)  ) |
                    (rev_w & rev_h & ((~(rev_h << one)) | (~(rev_h << two)) & (rev_h << one)))
                )

            if k == 2:
                int = operation.Concat(int, core.Constant(0, 2))
                int_frac = smart_sub(int, f12)
            elif k == 3:
                # f3 cannot be included in f12, since ~(h >> one) would need to be counted 4 times
                if version == 0:
                    f3 = secondaryop.PopCount(w & h & (h >> one) & (h >> two) & (~(h >> three)))
                else:
                    f3 = secondaryop.PopCount(rev_w & rev_h & (rev_h << one) & (rev_h << two) & (~(rev_h << three)))
                int = operation.Concat(int, core.Constant(0, 3))
                f12 = operation.Concat(f12, core.Constant(0, 1))
                int_frac = smart_sub(int, smart_add(f12, f3))
            elif k == 4:
                if version == 0:
                    f34 = secondaryop.PopCountSum2(
                        w & h & (h >> one) & (h >> two) & (~(h >> three)),
                        w & h & (h >> one) & (h >> two) & ((~(h >> three)) | (~(h >> four) & (h >> three)))
                    )
                elif version == 1:
                    f34 = secondaryop.PopCountSum2(
                        rev_w & rev_h & (rev_h << one) & (rev_h << two) & (~(rev_h << three)),
                        rev_w & rev_h & (rev_h << one) & (rev_h << two) & ((~(rev_h << three)) | (~(rev_h << four) & (rev_h << three)))
                    )
                else:
                    f34 = secondaryop.PopCount(
                        # ( (rev_w & rev_h & (rev_h << one) & (rev_h << two) & (~(rev_h << three))) >> one ) |
                        ( ((rev_w & rev_h) >> one) & rev_h & (rev_h << one) & (~(rev_h << two))) |
                        (rev_w & rev_h & (rev_h << one) & (rev_h << two) & ((~(rev_h << three)) | (~(rev_h << four) & (rev_h << three))))
                    )
                int = operation.Concat(int, core.Constant(0, 4))
                f12 = operation.Concat(f12, core.Constant(0, 2))
                int_frac = smart_sub(int, smart_add(f12, f34))
            else:
                raise ValueError("precision must be between 0 and 4")

        if debug:
            print("\n\n ~~ ")
            print("u:            ", u.bin())
            print("v:            ", v.bin())
            print("a:            ", a.bin())
            print("s00_:         ", s00_.bin())
            print("e_i1:         ", e_i1.bin())
            print("e_ili1:       ", e_ili.bin())
            print("q:            ", q.bin())
            print("s:            ", s.bin())
            print("d:            ", d.bin())
            print("w:            ", w.bin())
            if version == 0:
                print("h:            ", h.bin())
            else:
                print("rev_w:        ", rev_w.bin())
                print("rev_h:        ", rev_h.bin())
            print("sbnegb:       ", sbnegb.bin())
            print("int:          ", int.bin())
            if k == 1:
                print("f1:           ", f1.bin())
            elif k > 1:
                print("f12:          ", f12.bin())
                if k == 3:
                    print("f3:           ", f3.bin())
                elif k == 4:
                    print("f34:          ", f34.bin())
            print("int_frac:     ", int_frac.bin())

        return int_frac, extra_constraints

    def max_weight(self):
        effective_width = self._effective_width

        if effective_width <= 1:
            return 0
        else:
            return effective_width * (2 ** self.effective_precision)

    def weight_width(self):
        effective_width = self._effective_width
        index_one = self._index_first_one

        if effective_width <= 1:
            return 1
        else:
            u = self.input_diff[0].val[:index_one]
            return u.width.bit_length() + self.effective_precision

    def decimal_weight(self, output_diff):
        """Return the `decimal.Decimal` weight for a given constant output `XorDiff`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAddCt, make_partial_op_model
            >>> XorModelBvAdd_1 = make_partial_op_model(XorModelBvAddCt, (None, Constant(1, 4)))
            >>> alpha = XorDiff(Constant(4, 4))
            >>> f = XorModelBvAdd_1(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(XorDiff(Constant(4, 4)), w), f"w to be divided by 2^{f.num_frac_bits()}"
            (0x1 == w, 'w to be divided by 2^1')
            >>> f.decimal_weight(XorDiff(Constant(4, 4)))
            Decimal('0.4150374992788438185462610560')

        See also `abstractproperty.opmodel.OpModel.decimal_weight`.
        """
        self._assert_preconditions_decimal_weight(output_diff)

        u = self.input_diff[0].val
        v = output_diff.val
        a = self.ct
        n = a.width

        assert isinstance(u, core.Constant) and isinstance(v, core.Constant)

        def probability2weight(pr):
            if pr == 0:
                return math.inf
            elif pr == 1:
                return decimal.Decimal(0)  # avoid 0.0
            else:
                if isinstance(pr, fractions.Fraction):
                    pr = pr.numerator / decimal.Decimal(pr.denominator)
                else:
                    pr = decimal.Decimal(pr)
                my_w =- log2_decimal(pr)
                assert my_w >= 0
                return my_w

        if u[0] != v[0]:
            return probability2weight(0)

        probability = fractions.Fraction(1)
        delta = fractions.Fraction(0)
        for i in range(n - 1):
            if [u[i], v[i], u[i+1] ^ v[i+1]] == [0, 0, 0]:
                delta = fractions.Fraction(int(a[i]) + delta, 2)

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [0, 0, 1]:
                return probability2weight(0)

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [1, 1, 0]:
                probability *= 1 - (int(a[i]) + delta - (2 * int(a[i]) * delta))
                delta = int(a[i])

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [1, 1, 1]:
                probability *= int(a[i]) + delta - (2 * int(a[i]) * delta)
                delta = fractions.Fraction(1, 2)

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [0, 1, 0]:
                probability *= fractions.Fraction(1, 2)
                delta = int(a[i])

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [0, 1, 1]:
                probability *= fractions.Fraction(1, 2)

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [1, 0, 0]:
                probability *= fractions.Fraction(1, 2)
                delta = int(a[i])

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [1, 0, 1]:
                probability *= fractions.Fraction(1, 2)

        dw = probability2weight(probability)

        self._assert_postconditions_decimal_weight(output_diff, dw)
        return dw

    def num_frac_bits(self):
        return self.effective_precision

    # noinspection SpellCheckingInspection
    @classmethod
    @functools.lru_cache()
    def _len2error(cls, ct_width, precision, verbose=False):
        """Return a dictionary mapping difference lengths to their maximum error.

            >>> # docstring for debugging purposes
            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAddCt, make_partial_op_model
            >>> XorModelBvAdd_1 = make_partial_op_model(XorModelBvAddCt, (None, Constant(1, 4)))
            >>> _ = XorModelBvAddCt._len2error(8, 4, True)
            chainlen2biterror: {3: Decimal('0.03')}
            worst_chainlen: 3
            <BLANKLINE>
             > difflen: 4
             > decompositions: [[3]]
             > worst_decomp: [3]
             > difflen2differror[4]: 0.08607133205593431
             > difflen: 5
             > decompositions: [[3]]
             > worst_decomp: [3]
             > difflen2differror[5]: 0.08607133205593431
             > difflen: 7
             > decompositions: [[3, 3], [3]]
             > worst_decomp: [3, 3]
             > difflen2differror[7]: 0.17214266411186862
            <BLANKLINE>
            prime_lens: [3]
            difflen2differror: [0, 0, 0, Decimal('0.09'), Decimal('0.09'), Decimal('0.09'), Decimal('0.17'), Decimal('0.17')]
            <BLANKLINE>
            >>> [round(d, 2) for d in XorModelBvAddCt._len2error(8, 0)]
            [0, 0, 0, Decimal('1.00'), Decimal('1.00'), Decimal('1.00'), Decimal('2.00'), Decimal('2.00')]
            >>> [round(d, 2) for d in XorModelBvAddCt._len2error(8, 1)]
            [0, 0, 0, Decimal('0.09'), Decimal('0.58'), Decimal('0.58'), Decimal('0.58'), Decimal('0.67')]
            >>> [round(d, 2) for d in XorModelBvAddCt._len2error(8, 2)]
            [0, 0, 0, Decimal('0.09'), Decimal('0.09'), Decimal('0.33'), Decimal('0.33'), Decimal('0.33')]
            >>> [round(d, 2) for d in XorModelBvAddCt._len2error(8, 3)]
            [0, 0, 0, Decimal('0.09'), Decimal('0.09'), Decimal('0.09'), Decimal('0.21'), Decimal('0.21')]
            >>> [round(d, 2) for d in XorModelBvAddCt._len2error(8, 4)]
            [0, 0, 0, Decimal('0.09'), Decimal('0.09'), Decimal('0.09'), Decimal('0.17'), Decimal('0.17')]

        """
        n = ct_width

        assert n <= 32

        def get_chainlen2biterror():
            def bit_error(chainlen):
                # given a chain of length l_i,
                # if pi != 2^{-i}, then p_i <= 2^{l_i - 1} - 1
                # and there are (at most) l_i - 2 bits right to the leftmost active bit.
                # (e.g. l_i = 4, p_i <= 7 = 0b111, two "fraction bits")
                k = precision

                if chainlen - 2 <= k:
                    # bound = 1 - (1 + ln(ln(2))) / ln(2) = 0.086...
                    # is the error when all fraction bits are sued
                    return decimal.Decimal("0.08607133205593431") / chainlen

                t = [t_i * decimal.Decimal(2)**(-k) for t_i in range(2**k)]  # t[i] = t_i * 2**(-k)
                chainerror = max(log2_decimal(1 + t_i + decimal.Decimal(2)**(-k)) - t_i for t_i in t)

                return chainerror / chainlen

            d = {}
            worst_chainlen = 2
            worst_chainlen_error = 0

            for chainlen in range(3, n):
                # chainlen = l_i
                chainerror = bit_error(chainlen)

                if chainerror > worst_chainlen_error:
                    worst_chainlen_error = chainerror
                    worst_chainlen = chainlen
                else:
                    if ((chainlen // worst_chainlen) * worst_chainlen) * worst_chainlen_error >= chainlen * chainerror:
                        # assume chainlen = worst_chainlen * k
                        # if error(worst_chainlen) * k) >= error(chainlen), we ignore chainleb
                        continue

                d[chainlen] = chainerror

            return d

        chainlen2biterror = get_chainlen2biterror()

        if len(chainlen2biterror) == 0:
            assert n <= 3
            return [0, 0, 0][:n]

        worst_chainlen = max(chainlen2biterror.items(), key=operator.itemgetter(1))[0]

        def error_chainlen(chainlen):
            return chainlen2biterror[chainlen] * chainlen

        def error_decomposition(decomposition):
            error = 0
            for chainlen in decomposition:
                error += error_chainlen(chainlen)
            return error

        # a prime length correspond to a single chain of maximum error
        # (decomposing the chain in sub-chains do not increase the error)
        prime_lens = [3]
        if worst_chainlen not in prime_lens:
            prime_lens.append(worst_chainlen)

        def get_decompositions(diff_len):
            if diff_len in prime_lens:
                return [[diff_len]]  # trivial decomposition

            diff_decompositions = []
            for m in prime_lens:
                if m <= diff_len:
                    m_decompositions = get_decompositions(diff_len - m)
                    for decomp in m_decompositions:
                        diff_decompositions.append(sorted([m] + decomp))
                    else:
                        diff_decompositions.append([m])

            return diff_decompositions

        difflen2differror = [None for _ in range(n)]
        difflen2differror[:4] = [0, 0, 0, error_chainlen(3)]

        if verbose:
            print("chainlen2biterror:", {k: round(v, 2) for k, v in chainlen2biterror.items()})
            print("worst_chainlen:", worst_chainlen)
            # print("maxchain:", maxchain)
            print()

        for difflen in range(4, n):
            if difflen % worst_chainlen == 0:
                difflen2differror[difflen] = (difflen // worst_chainlen) * error_chainlen(worst_chainlen)
                continue

            decompositions = []

            if difflen in chainlen2biterror:
                decompositions.append([difflen])

            decompositions.extend(get_decompositions(difflen))
            worst_decomp = max(decompositions, key=error_decomposition)

            if worst_decomp[0] == difflen:
                prime_lens.append(difflen)

            # noinspection PyTypeChecker
            difflen2differror[difflen] = error_decomposition(worst_decomp)

            if verbose:
                print(" > difflen:", difflen)
                print(" > decompositions:", decompositions)
                print(" > worst_decomp:", worst_decomp)
                print(" > difflen2differror[{}]: {}".format(difflen, difflen2differror[difflen]))

        if verbose:
            print("\nprime_lens:", prime_lens)
            print("difflen2differror:", [round(d, 2) for d in difflen2differror])
            print()

        return difflen2differror

    def error(self):
        """Return the maximum difference between `OpModel.weight_constraint` and the exact weight.

            >>> # docstring for debugging purposes
            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.opmodel import XorModelBvAddCt, make_partial_op_model
            >>> old_prec = XorModelBvAddCt.precision
            >>> prec2error = {}
            >>> n = 8
            >>> for p in range(0, 8 + 1):
            ...     XorModelBvAddCt.precision = p
            ...     XorModelBvAdd_1 = make_partial_op_model(XorModelBvAddCt, (None, Constant(1, n)))
            ...     alpha = XorDiff(Variable("a", n))
            ...     f = XorModelBvAdd_1(alpha)
            ...     prec2error[p] = round(f.error(), 4)
            >>> XorModelBvAddCt.precision = old_prec
            >>> # error depending on the precision (fractional bits)
            >>> prec2error  # doctest: +NORMALIZE_WHITESPACE
            {0: Decimal('2.0000'), 1: Decimal('0.6710'), 2: Decimal('0.3350'), 3: Decimal('0.2100'),
            4: Decimal('0.1721'), 5: Decimal('0.1721'), 6: Decimal('0.1721'), 7: Decimal('0.1721'), 8: Decimal('0.1721')}

        See also `abstractproperty.opmodel.OpModel.error`.
        """
        effective_width = self._effective_width

        len2error = self._len2error(self.ct.width, self.effective_precision)
        assert effective_width < len(len2error)
        return len2error[effective_width]


class XorModelBvSubCt(XorModelBvAddCt):
    """Represent the `XorDiff` `PartialOpModel` of modular substraction by a constant.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import XorModelBvSubCt, make_partial_op_model
        >>> XorModelBvSub_1 = make_partial_op_model(XorModelBvSubCt, (None, Constant(1, 4)))
        >>> alpha = XorDiff(Constant(0, 4))
        >>> f = XorModelBvSub_1(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(XorModelBvSubCt, (None, Constant(0b0001, width=4)))(XorDiff(Constant(0b0000, width=4)))
        >>> x = Constant(0, 4)
        >>> f.eval_derivative(x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (6, 4, Decimal('0.08607133205593431'), 1)

    The differential model of the modular substraction by a constant
    is the same as `XorModelBvAddCt` since ``~(x - c) = ~x + c``
    and `BvNot` preserves differences.
    """

    diff_type = difference.XorDiff
    base_op = operation.BvSub

    def __init__(self, input_diff):
        if self.op.fixed_args[0] is not None or self.op.fixed_args[1] is None:
            raise ValueError(f"{self.__class__.__name__} only supports the 2nd operand fixed")

        super().__init__(input_diff)


# ---------------------
# ----- RX models -----
# ---------------------


class RXModelId(abstractproperty.opmodel.ModelIdentity, OpModel):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `BvIdentity`.

    See also `ModelIdentity`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelId
        >>> alpha, beta = RXDiff(Variable("a", 4)), RXDiff(Variable("b", 4))
        >>> f = RXModelId(alpha)
        >>> print(f.vrepr())
        RXModelId(RXDiff(Variable('a', width=4)))
        >>> f.validity_constraint(beta)
        a == b
        >>> x = Constant(0, 4)
        >>> f.eval_derivative(x).val.doit()  # f(x + alpha) - f(x)
        Id(a)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    diff_type = difference.RXDiff


class RXModelBvAdd(OpModel):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `BvAdd`.

    The class attribute ``precision`` sets how many fraction bits
    are used in the bit-vector weight. The default value of ``precision``
    is ``3``, and lower values lead to shorter formulas with higher errors.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelBvAdd
        >>> alpha = RXDiff(Constant(0, 16)), RXDiff(Constant(0, 16))
        >>> f = RXModelBvAdd(alpha)
        >>> print(f.vrepr())
        RXModelBvAdd([RXDiff(Constant(0x0000, width=16)), RXDiff(Constant(0x0000, width=16))])
        >>> x = Constant(0, 16), Constant(0, 16)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        RXDiff(0x0000)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (136, 8, Decimal('1.051220727815270622166497531'), 3)

    """

    diff_type = difference.RXDiff
    op = operation.BvAdd
    precision = 3  # 0, 2, 3, 5, 7

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `RXDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.opmodel import RXModelBvAdd
            >>> alpha = RXDiff(Constant(0, 4)), RXDiff(Constant(0, 4))
            >>> f = RXModelBvAdd(alpha)
            >>> beta = RXDiff(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 4), Variable("a1", 4), Variable("b", 4)
            >>> alpha = RXDiff(a0), RXDiff(a1)
            >>> f = RXModelBvAdd(alpha)
            >>> beta = RXDiff(b)
            >>> result = f.validity_constraint(beta)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            (~(((a0[:1]) ^ (a1[:1]) ^ (b[:1]) ^ (((a0[:1]) ^ (a1[:1]) ^ (b[:1])) << 0b001))[:1]) |
            ((((a0[:1]) ^ (b[:1])) | ((a1[:1]) ^ (b[:1])))[1:])) == 0b11
            >>> result.xreplace({a0: Constant(0, 4), a1: Constant(0, 4), b: Constant(0, 4)})
            0b1
            >>> a1 = Constant(0, 4)
            >>> alpha = RXDiff(a0), RXDiff(a1)
            >>> f = RXModelBvAdd(alpha)
            >>> beta = RXDiff(b)
            >>> result = f.validity_constraint(beta)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            (~(((a0[:1]) ^ (b[:1]) ^ (((a0[:1]) ^ (b[:1])) << 0b001))[:1]) |
            ((((a0[:1]) ^ (b[:1])) | (b[:1]))[1:])) == 0b11

        See `OpModel.validity_constraint` for more information.
        """
        # (I ^ SHL)(da ^ db ^ dc)[1:] <= SHL((da ^ dc) OR (db ^ dc))[1:]

        # one = core.Constant(1, self.input_diff[0].val.width)  # alt v1

        alpha, beta = [d.val for d in self.input_diff]
        gamma = output_diff.val
        # da, db, dc = alpha >> one, beta >> one, gamma >> one  # alt v1
        da, db, dc = alpha[:1], beta[:1], gamma[:1]  # ignore LSB

        one = core.Constant(1, da.width)

        # lhs = (da ^ db ^ dc) ^ ((da ^ db ^ dc) << one)  # alt v1
        # rhs = ((da ^ dc) | (db ^ dc)) << one
        if da.width == 1:
            lhs, rhs = core.Constant(0, 1), core.Constant(0, 1)
        else:
            lhs = ((da ^ db ^ dc) ^ ((da ^ db ^ dc) << one))[:1]
            rhs = (((da ^ dc) | (db ^ dc)) << one)[:1]

        def bitwise_implication(x, y):
            return (~x) | y

        # alt v1
        # n = lhs.width
        # return operation.BvComp(
        #     bitwise_implication(lhs[n - 2:1], rhs[n - 2:1]),  # ignore MSB, LSB
        #     ~ core.Constant(0, n - 2))
        return operation.BvComp(bitwise_implication(lhs, rhs), ~core.Constant(0, lhs.width))  # alt v1

    def pr_one_constraint(self, output_diff):
        """Return the probability-one constraint for a given output `RXDiff`.

        For `RXModelBvAdd`, any differential has probability less than 1.
        """
        return core.Constant(0, 1)

    @staticmethod
    def _decimal_weight_rotational_pair_pr(n):
        pr_rotational_pair = 1 + decimal.Decimal(2)**(1 - n) + decimal.Decimal("0.5") + decimal.Decimal(2)**(-n)
        return -(log2_decimal(pr_rotational_pair) - 2)

    def bv_weight(self, output_diff):
        """Return the bit-vector weight for a given output `RXDiff`.

        See also `abstractproperty.opmodel.OpModel.bv_weight`.
        """
        # one = core.Constant(1, self.input_diff[0].val.width)  # alt v1
        # two = core.Constant(2, self.input_diff[0].val.width)

        alpha, beta = [d.val for d in self.input_diff]
        gamma = output_diff.val
        da, db, dc = alpha[:1], beta[:1], gamma[:1]  # alt v1
        # da, db, dc = alpha >> one, beta >> one, gamma >> one  # alt v1

        if da.width == 1:
            rhs = core.Constant(0, 1)
        else:
            rhs = ((da ^ dc) | (db ^ dc))[da.width-2:]  # equiv to shift left
        # rhs = ((da ^ dc) | (db ^ dc)) << two  # alt v1
        hw = secondaryop.PopCount(rhs)

        max_hw = rhs.width
        # max_hw = rhs.width - 1  # alt v1

        # (max_hw + 3) = maximum integer part
        k = self.__class__.precision  # num fraction bits
        max_ct_part = 3  # see below ct_part  (without fraction bits)
        weight_width = (max_hw + max_ct_part).bit_length() + k

        # let lhs = LSB(lhs) = da ^ db ^ dc
        #     rhs = LSB(rhs) = 0
        # case A (w=1.415): lhs => rhs
        # case B (w=3):     lhs ^ 1 => rhs

        # 1.415 = -log2(pr propagation of a rotational pair with offset 1)
        #   this Pr is (1 + 2^(1−n) + 2^(-1) + 2^(−n))/4.
        #   (source Rotational Cryptanalysis in the Presence of Constants)

        n = alpha.width
        w_rotational_pair = self._decimal_weight_rotational_pair_pr(n)
        # take k fraction bits, remove other fraction bits and convert the result to an integer
        w_rotational_pair = int(w_rotational_pair * (2**k))

        def bitwise_implication(x, y):
            return (~x) | y

        ct_part_extend = operation.Ite(
            bitwise_implication(da[0] ^ db[0] ^ dc[0], core.Constant(0, 1)),
            core.Constant(w_rotational_pair, weight_width),
            core.Constant(3, weight_width) << k
        )

        hw_extend = operation.zero_extend(hw, weight_width - hw.width)

        return (hw_extend << k) + ct_part_extend

    def weight_constraint(self, output_diff, weight_variable):
        """Return the weight constraint for a given output `RXDiff` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.opmodel import RXModelBvAdd
            >>> n = 4
            >>> alpha = RXDiff(Constant(0, n)), RXDiff(Constant(0, n))
            >>> f = RXModelBvAdd(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(RXDiff(Constant(0, n)), w)
            w == 0b001001
            >>> a0, a1, b = Variable("a0", n), Variable("a1", n), Variable("b", n)
            >>> alpha = RXDiff(a0), RXDiff(a1)
            >>> f = RXModelBvAdd(alpha)
            >>> result = f.weight_constraint(RXDiff(b), w)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            w == (((0x0 :: PopCount((((a0[:1]) ^ (b[:1])) | ((a1[:1]) ^ (b[:1])))[1:])) << 0b000011) +
            (Ite(~((a0[1]) ^ (a1[1]) ^ (b[1])), 0b001001, 0b011000)))
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), b: Constant(0, n)})
            w == 0b001001

        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_diff, weight_variable)

    def max_weight(self):
        # can be obtained from (hw_extend << k) + ct_part_extend
        alpha_da_width = self.input_diff[0].val.width
        if self.input_diff[0].val.width == 2:
            max_hw = alpha_da_width - 1
        else:
            max_hw = alpha_da_width - 2  # alpha[:1] and [da.width-2:]
        max_ct_part = 3
        k = self.__class__.precision
        return (max_hw + max_ct_part) << k

    def weight_width(self):
        # can be obtained from the variable weight_width in bv_weight
        alpha_da_width = self.input_diff[0].val.width
        if self.input_diff[0].val.width == 2:
            max_hw = alpha_da_width - 1
        else:
            max_hw = alpha_da_width - 2  # alpha[:1] and [da.width-2:]
        k = self.__class__.precision
        max_ct_part = 3
        weight_width = (max_hw + max_ct_part).bit_length() + k
        return weight_width

    def decimal_weight(self, output_diff):
        """Return the `decimal.Decimal` weight for a given constant output `RXDiff`.

        The decimal weight is not equal to the exact weight due to the
        approximation used for the probability of the rotational pair.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.opmodel import RXModelBvAdd
            >>> n = 4
            >>> alpha = RXDiff(Constant(0, n)), RXDiff(Constant(0, n))
            >>> f = RXModelBvAdd(alpha)
            >>> int(f.bv_weight(RXDiff(Constant(0, n)))) / 2**(f.precision)
            1.125
            >>> f.decimal_weight(RXDiff(Constant(0, n)))
            Decimal('1.245112497836531455638783168')

        See also `abstractproperty.opmodel.OpModel.decimal_weight`.
        """
        self._assert_preconditions_decimal_weight(output_diff)

        # in this case, the decimal_weight is not always smaller than the bit-vector weight
        alpha, beta = [d.val for d in self.input_diff]
        gamma = output_diff.val
        da, db, dc = alpha[:1], beta[:1], gamma[:1]

        one = core.Constant(1, da.width)

        rhs = ((da ^ dc) | (db ^ dc)) << one
        # rhs = ((da ^ dc) | (db ^ dc))[da.width-2:]  # alt v1
        hw = secondaryop.PopCount(rhs)

        result = int(hw)

        def bitwise_implication(x, y):
            return (~x) | y

        n = alpha.width
        w_rotational_pair = self._decimal_weight_rotational_pair_pr(n)

        if bitwise_implication(da[0] ^ db[0] ^ dc[0], core.Constant(0, 1)) == core.Constant(1, 1):
            result += w_rotational_pair
        else:
            result += 3

        # print("\n\n ~~ ")
        # print("self:         ", self)
        # print("output diff:  ", output_diff)
        # print("da:           ", da.bin())
        # print("db:           ", db.bin())
        # print("dc:           ", dc.bin())
        # print("rhs:          ", rhs.bin())
        # print("hw:           ", int(hw))
        # print("bw_implies:   ", bitwise_implication(da[0] ^ db[0] ^ dc[0], core.Constant(0, 1)).bin())
        # print("result:       ", result)
        # print("\n\n")

        result = decimal.Decimal(result)
        self._assert_postconditions_decimal_weight(output_diff, result)
        return result

    def num_frac_bits(self):
        return self.__class__.precision

    def error(self, ignore_error_decimal2exact=False):
        """Return the maximum difference between `OpModel.weight_constraint` and the exact weight.

            >>> # docstring for debugging purposes
            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.opmodel import RXModelBvAdd
            >>> old_prec = RXModelBvAdd.precision
            >>> prec2error = {}
            >>> n = 16
            >>> for p in range(0, 8 + 1):
            ...     RXModelBvAdd.precision = p
            ...     alpha = RXDiff(Variable("a", n)), RXDiff(Variable("b", n))
            ...     f = RXModelBvAdd(alpha)
            ...     prec2error[p] = round(float(f.error()), 4)
            >>> RXModelBvAdd.precision = old_prec
            >>> prec2error  # error depending on the precision (fractional bits)
            {0: 1.4262, 1: 1.4262, 2: 1.1762, 3: 1.0512, 4: 1.0512, 5: 1.02, 6: 1.02, 7: 1.0122, 8: 1.0122}

        See also `abstractproperty.opmodel.OpModel.error`.
        """
        n = self.input_diff[0].val.width
        k = self.__class__.precision

        w_rotational_pair = self._decimal_weight_rotational_pair_pr(n)

        w_rotational_pair_approx = int(w_rotational_pair * (2**k)) / decimal.Decimal(2**k)

        # w_rotational_pair_approx < w_rotational_pair
        assert w_rotational_pair_approx < w_rotational_pair
        bv2decimalerror = w_rotational_pair - w_rotational_pair_approx

        if ignore_error_decimal2exact:
            decimal2exact_error = 0
        else:
            # found empirically with test_opmodel.py
            if n <= 2:
                decimal2exact_error = decimal.Decimal("1.584962500721156181453738944")
            elif n <= 3:
                decimal2exact_error = decimal.Decimal("1.321928094887362347870319428")
            elif n <= 4:
                # found for n=4 with 10000 examples
                decimal2exact_error = decimal.Decimal("1.169925001442312362907477888")
            else:
                # found for n=8 with 10000 examples
                decimal2exact_error = decimal.Decimal("1.011227255423254120337880584")

        return bv2decimalerror + decimal2exact_error


class RXModelBvSub(RXModelBvAdd):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `BvSub`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelBvSub
        >>> alpha = RXDiff(Constant(0, 16)), RXDiff(Constant(0, 16))
        >>> f = RXModelBvSub(alpha)
        >>> print(f.vrepr())
        RXModelBvSub([RXDiff(Constant(0x0000, width=16)), RXDiff(Constant(0x0000, width=16))])
        >>> x = Constant(0, 16), Constant(0, 16)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        RXDiff(0x0000)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (136, 8, Decimal('1.051220727815270622166497531'), 3)

    The differential model of the modular substraction is the same as `RXModelBvAdd`
    since ``~(x - y) == ~x + y`` (and `BvNot` preserves RX-differences).
    """

    diff_type = difference.RXDiff
    op = operation.BvSub


class RXModelBvOr(XorModelBvOr):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `BvOr`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelBvOr
        >>> alpha = RXDiff(Constant(0, 4)), RXDiff(Constant(0, 4))
        >>> f = RXModelBvOr(alpha)
        >>> print(f.vrepr())
        RXModelBvOr([RXDiff(Constant(0b0000, width=4)), RXDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        RXDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    .. Implementation details:

        For any bitwise operation, (w.l.o.g, unary op f(x))
        ``Pr( RXDiff(d) propagates through f to RXDiff(d') ) =
        Pr( Diff(d) propagates through f to Diff(d') )``, where
        ``RXDiff(d) = RXDiff.from_pair(x, RotateLeft(x, 1) ^ d)``

        This comes from
        ``Pr[((x & y) <<< 1) ^ beta = ((x <<< 1) ^ alpha1) & ((y <<< 1) ^ alpha2)] =
        Pr[(x & y) ^ beta = (x ^ alpha1) & (y ^ alpha2)]``, where
        ``alpha=d`` is the input diff and beta=d' the output diff

    """

    diff_type = difference.RXDiff
    op = operation.BvOr


class RXModelBvAnd(RXModelBvOr):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `BvAnd`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelBvAnd
        >>> alpha = RXDiff(Constant(0, 4)), RXDiff(Constant(0, 4))
        >>> f = RXModelBvAnd(alpha)
        >>> print(f.vrepr())
        RXModelBvAnd([RXDiff(Constant(0b0000, width=4)), RXDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        RXDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    The differential model of `BvAnd` is the same as `RXModelBvOr`
    since ``~(x & y) == ~x | ~y`` (and `BvNot` preserves differences).
    """
    op = operation.BvAnd


class RXModelBvIf(XorModelBvIf):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `BvIf`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelBvIf
        >>> alpha = RXDiff(Constant(0, 4)), RXDiff(Constant(0, 4)), RXDiff(Constant(0, 4))
        >>> f = RXModelBvIf(alpha)
        >>> print(f.vrepr())
        RXModelBvIf([RXDiff(Constant(0b0000, width=4)), RXDiff(Constant(0b0000, width=4)), RXDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        RXDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    .. Implementation details:

        See `RXModelBvOr`.

    """

    diff_type = difference.RXDiff
    op = secondaryop.BvIf


class RXModelBvMaj(XorModelBvMaj):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `BvMaj`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelBvMaj
        >>> alpha = RXDiff(Constant(0, 4)), RXDiff(Constant(0, 4)), RXDiff(Constant(0, 4))
        >>> f = RXModelBvMaj(alpha)
        >>> print(f.vrepr())
        RXModelBvMaj([RXDiff(Constant(0b0000, width=4)), RXDiff(Constant(0b0000, width=4)), RXDiff(Constant(0b0000, width=4))])
        >>> x = Constant(0, 4), Constant(0, 4), Constant(0, 4)
        >>> f.eval_derivative(*x)  # f(x + alpha) - f(x)
        RXDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    .. Implementation details:

        See `RXModelBvOr`.

    """

    diff_type = difference.RXDiff
    op = secondaryop.BvMaj


class RXModelBvShlCt(PartialOpModel):
    """Represent the `RXDiff` `PartialOpModel` of left shift by a constant.

    This class is not meant to be instantiated but to provide a base
    class for subclasses generated through `make_partial_op_model`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelBvShlCt, make_partial_op_model
        >>> RXModelBvShlCte_1 = make_partial_op_model(RXModelBvShlCt, (None, Constant(1, 4)))
        >>> alpha = RXDiff(Constant(0, 4))
        >>> f = RXModelBvShlCte_1(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(RXModelBvShlCt, (None, Constant(0b0001, width=4)))(RXDiff(Constant(0b0000, width=4)))
        >>> x = Constant(0, 4)
        >>> f.eval_derivative(x)  # f(x + alpha) - f(x)
        RXDiff(0x0)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (2, 2, 0, 0)

    """

    diff_type = difference.RXDiff
    base_op = operation.BvShl
    # op is a set through make_partial_op_model

    def __init__(self, input_diff):
        if self.__class__.op is None:
            raise ValueError(f"{self.__class__.__name__} need to be given to make_partial_op_model"
                             f" to get the OpModel for the particular fixed operand")
        if self.__class__.op.fixed_args[0] is not None or self.__class__.op.fixed_args[1] is None:
            raise ValueError(f"{self.__class__.__name__} only supports the 2nd operand fixed")

        super().__init__(input_diff)

    @property
    def ct(self):
        """The constant operand."""
        my_ct = self.__class__.op.fixed_args[1]
        assert isinstance(my_ct, core.Constant) and 0 < my_ct < my_ct.width
        return my_ct

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `RXDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.opmodel import RXModelBvShlCt, make_partial_op_model
            >>> RXModelBvShlCte_1 = make_partial_op_model(RXModelBvShlCt, (None, Constant(1, 4)))
            >>> alpha = RXDiff(Constant(0, 4))
            >>> f = RXModelBvShlCte_1(alpha)
            >>> f.validity_constraint(RXDiff(Constant(0, 4)))
            0b1
            >>> u, v = Variable("u", 4), Variable("v", 4)
            >>> f = RXModelBvShlCte_1(RXDiff(u))
            >>> result = f.validity_constraint(RXDiff(v))
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            (v[:2]) == (u << 0x1[:2])
            >>> result.xreplace({u: Constant(0, 4), v: Constant(0, 4)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """

        # consider a pair (x, (x <<< 1) ^ d) with RXDiff d
        # if the operation <<_s is applied to the pair we obtain:
        #   x              |-> x << s
        #   (x <<< 1) ^ d  |-> ((x <<< 1) << s) ^ (d << s)
        # and the output RXDiff beta of these two last term is
        #   beta = RXDiff(·, ·) = ((x << s) <<< 1) ^ ((x <<< 1) << s) ^ (d << s)
        # it is easy to see that
        #   ((x << s) <<< 1) ^ ((x <<< 1) << s) = 0 ··· 0 x_{s-1} 0 ··· 0 x_{n-1}
        # where there are (s-1) zeros in the right part, that is,
        # the non-zero bits are in position 0 and position s; thus,
        #   beta = (0 ··· 0 x_{s-1} 0 ··· 0 x_{n-1}) ^ (d << s)
        # meaning that beta is equal to (d << s) except in position 0 and s
        # where it can take arbitrary values

        d = self.input_diff[0].val
        d_s = self.__class__.op(d)  # d << s
        beta = output_diff.val
        s = int(self.ct)
        n = d.width

        true_bv = core.Constant(1, 1)

        if s == 1:
            # no zeros on the right side
            # beta[:n] out of range
            return true_bv if s+1 >= n else operation.BvComp(beta[:s+1], d_s[:s+1])
        elif s == n - 1:
            # no zeros on the left side
            return operation.BvComp(beta[s-1:1], d_s[s-1:1])
        else:
            aux = true_bv if s+1 >= n else operation.BvComp(beta[:s+1], d_s[:s+1])
            return operation.BvComp(beta[s-1:1], d_s[s-1:1]) & aux

    def pr_one_constraint(self, output_diff):
        """Return the probability-one constraint for a given output `RXDiff`.

        For `RXModelBvShlCt`, any differential has probability less than 1.
        """
        return core.Constant(0, 1)

    def bv_weight(self, output_diff):
        """Return the bit-vector weight for a given output `RXDiff`.

        For `RXModelBvShlCt`, any valid differential has weight 2.
        """
        # if the RX differential over <<_s has non-zero probability,
        # then its weight is 2*t where t = min(1, s, n - 1, n - s) = 1
        # (weight is independent of output difference)
        # Source: P.Sokołowski, "Design and Analysis of Cryptographic Hash Functions"
        # (page 35) https:// hdl.handle.net / 10593 / 19267
        return core.Constant(2, width=2)

    def weight_constraint(self, output_diff, weight_variable):
        """Return the weight constraint for a given output `RXDiff` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.opmodel import RXModelBvShlCt, make_partial_op_model
            >>> width = 4
            >>> RXModelBvShlCte_1 = make_partial_op_model(RXModelBvShlCt, (None, Constant(1, width)))
            >>> alpha = RXDiff(Variable("u", width))
            >>> f = RXModelBvShlCte_1(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> beta = RXDiff(Variable("v", width))
            >>> f.weight_constraint(beta, w)
            w == 0b10

        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_diff, weight_variable)

    def max_weight(self):
        return 2

    def weight_width(self):
        return 2

    def decimal_weight(self, output_diff):
        self._assert_preconditions_decimal_weight(output_diff)
        dw = decimal.Decimal(int(self.bv_weight(output_diff)))
        self._assert_postconditions_decimal_weight(output_diff, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class RXModelBvLshrCt(RXModelBvShlCt):
    """Represent the `RXDiff` `PartialOpModel` of right shift by a constant.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import RXModelBvLshrCt, make_partial_op_model
        >>> RXModelBvLshrCte_1 = make_partial_op_model(RXModelBvLshrCt, (None, Constant(1, 8)))
        >>> alpha = RXDiff(Constant(0, 8))
        >>> f = RXModelBvLshrCte_1(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(RXModelBvLshrCt, (None, Constant(0b00000001, width=8)))(RXDiff(Constant(0b00000000, width=8)))
        >>> x = Constant(0, 8)
        >>> f.eval_derivative(x)  # f(x + alpha) - f(x)
        RXDiff(0x00)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (2, 2, 0, 0)

    .. Implementation details:

        See `RXModelBvShlCt`.

    """

    base_op = operation.BvLshr

    def validity_constraint(self, output_diff):
        """Return the validity constraint for a given output `RXDiff` difference.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.opmodel import RXModelBvLshrCt, make_partial_op_model
            >>> RXModelBvLshrCte_1 = make_partial_op_model(RXModelBvLshrCt, (None, Constant(1, 4)))
            >>> alpha = RXDiff(Constant(0, 4))
            >>> f = RXModelBvLshrCte_1(alpha)
            >>> f.validity_constraint(RXDiff(Constant(0, 4)))
            0b1
            >>> u, v = Variable("u", 4), Variable("v", 4)
            >>> f = RXModelBvLshrCte_1(RXDiff(u))
            >>> result = f.validity_constraint(RXDiff(v))
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            (v[2:1]) == (u >> 0x1[2:1])
            >>> result.xreplace({u: Constant(0, 4), v: Constant(0, 4)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """

        # consider a pair (x, (x <<< 1) ^ d) with RXDiff d
        # if the operation >>_s is applied to the pair we obtain:
        #   x              |-> x >> s
        #   (x <<< 1) ^ d  |-> ((x <<< 1) >> s) ^ (d >> s)
        # and the output RXDiff beta of these two last term is
        #   beta = RXDiff(·, ·) = ((x >> s) <<< 1) ^ ((x <<< 1) >> s) ^ (d >> s)
        # it is easy to see that for
        #   ((x >> s) <<< 1) ^ ((x <<< 1) >> s)
        # the non-zero bits are in position 0 and position n-s; thus,
        # beta is equal to (d >> s) except in position 0 and n-s
        # where it can take arbitrary values

        d = self.input_diff[0].val
        d_s = self.__class__.op(d)  # d << s
        beta = output_diff.val
        s = int(self.ct)
        n = d.width

        true_bv = core.Constant(1, 1)

        # similar to RXModelBvShlCt but replacing s by n-s
        if (n-s) == 1:
            # beta[:(n-s)+1] out of range
            return true_bv if (n-s)+1 >= n else operation.BvComp(beta[:(n-s)+1], d_s[:(n-s)+1])
        elif (n-s) == n - 1:
            return operation.BvComp(beta[(n-s)-1:1], d_s[(n-s)-1:1])
        else:
            aux = true_bv if (n-s)+1 >= n else operation.BvComp(beta[:(n-s)+1], d_s[:(n-s)+1])
            return operation.BvComp(beta[(n-s)-1:1], d_s[(n-s)-1:1]) & aux


# --------------------------


@functools.lru_cache(maxsize=None)  # temporary hack to create singletons
def get_weak_model(op, diff_type, nonzero2nonzero_weight, zero2zero_weight=0,
                   zero2nonzero_weight=math.inf, nonzero2zero_weight=math.inf, precision=0):
    """Return the weak model of the given bit-vector operation ``op``.

    Given the `Operation` ``op``, return the
    `WeakModel` of ``op`` for the `Difference` type ``diff_type``
    with given class attributes ``nonzero2nonzero_weight``,
    ``zero2zero_weight``,
    ``zero2nonzero_weight``, ``nonzero2zero_weight`` and
    ``precision`` (see `WeakModel`).

    The returned model is a subclass of `WeakModel` and `OpModel`.

    .. note::
        To link the returned model ``MyModel`` to ``op``
        such that ``MyModel`` is used in ``propagate``,
        set the ``xor_model`` or ``rx_model`` attribute of ``op``
        to ``MyModel`` (e.g., ``op.xor_model = MyModel``).
        See also  `differential.difference.XorDiff.propagate`
        or `differential.difference.RXDiff.propagate`.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import LutOperation
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import get_weak_model
        >>> class MyLut(LutOperation): pass  # a 2-bit function
        >>> XorWeakModelMyLut = get_weak_model(MyLut, XorDiff, decimal.Decimal(1.5), precision=1)
        >>> alpha, beta = XorDiff(Variable("a", 2)), XorDiff(Variable("b", 2))
        >>> f = XorWeakModelMyLut(alpha)
        >>> print(f.vrepr())
        XorWeakModelMyLut(XorDiff(Variable('a', width=2)))
        >>> f.validity_constraint(beta)
        (((a == 0b00) & (b == 0b00)) == 0b1) | ((~(a == 0b00) & ~(b == 0b00)) == 0b1)
        >>> f.bv_weight(beta)
        Ite(((a == 0b00) & (b == 0b00)) == 0b1, 0b00, 0b11)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (3, 2, 0, 1)

    """
    assert issubclass(op, operation.Operation)

    if diff_type == difference.XorDiff:
        prefix = "Xor"
        assert zero2zero_weight == 0
        # for XOR differentials with Pr. 1, an input property propagates to a unique output property
        assert zero2nonzero_weight == math.inf
    elif diff_type == difference.RXDiff:
        prefix = "RX"
    else:
        raise ValueError(f"invalid diff_type {diff_type}")

    _op, _diff_type = op, diff_type
    _zero2zero_weight = zero2zero_weight
    _nonzero2nonzero_weight = nonzero2nonzero_weight
    _zero2nonzero_weight, _nonzero2zero_weight = zero2nonzero_weight, nonzero2zero_weight
    _precision = precision

    class MyWeakModel(abstractproperty.opmodel.WeakModel, OpModel):
        op, diff_type = _op, _diff_type
        zero2zero_weight = _zero2zero_weight
        nonzero2nonzero_weight = _nonzero2nonzero_weight
        zero2nonzero_weight = _zero2nonzero_weight
        nonzero2zero_weight = _nonzero2zero_weight
        precision = _precision

        # def error(self):  # maximum weight of a differential with n-bit input is n
        #     return sum(p.val.width for p in self.input_prop)

    MyWeakModel.__name__ = f"{prefix}{abstractproperty.opmodel.WeakModel.__name__}{op.__name__}"
    return MyWeakModel


@functools.lru_cache(maxsize=None)
def get_branch_number_model(op, diff_type, output_widths, branch_number, nonzero2nonzero_weight, zero2zero_weight=0,
                            zero2nonzero_weight=math.inf, nonzero2zero_weight=math.inf, precision=0):
    """Return the branch-number model of the given bit-vector operation ``op``.

    Given the `Operation` ``op``, return the `BranchNumberModel`
    of ``op`` for the `Difference` type ``diff_type`` with given class
    attributes ``output_widths`` (given as a `tuple`),
    `branch_number`, ``nonzero2nonzero_weight``,
    ``zero2zero_weight``, ``zero2nonzero_weight``,
    ``nonzero2zero_weight`` and ``precision`` (see `BranchNumberModel`).

    The returned model is a subclass of `BranchNumberModel` and `OpModel`.

    .. note::
        To link the returned model ``MyModel`` to ``op``
        such that ``MyModel`` is used in ``propagate``,
        set the ``xor_model`` or ``rx_model`` attribute of ``op``
        to ``MyModel`` (e.g., ``op.xor_model = MyModel``).
        See also  `differential.difference.XorDiff.propagate`
        or `differential.difference.RXDiff.propagate`.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import MatrixOperation
        >>> from cascada.bitvector.printing import BvWrapPrinter
        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.opmodel import get_branch_number_model
        >>> class MyMatrix(MatrixOperation): pass  # a (3, 2) binary matrix
        >>> RXBranchNumberModelMyMatrix = get_branch_number_model(
        ...     MyMatrix, RXDiff, (1, 1, 1), 3, nonzero2nonzero_weight=math.inf,
        ...     zero2zero_weight=math.inf, zero2nonzero_weight=0, nonzero2zero_weight=0)
        >>> alpha, beta = RXDiff(Variable("a", 2)), RXDiff(Variable("b", 3))
        >>> f = RXBranchNumberModelMyMatrix(alpha)
        >>> print(f.vrepr())
        RXBranchNumberModelMyMatrix(RXDiff(Variable('a', width=2)))
        >>> print(BvWrapPrinter().doprint(f.validity_constraint(beta)))
        BvAnd((((a == 0b00) & ~(b == 0b000)) == 0b1) | ((~(a == 0b00) & (b == 0b000)) == 0b1),
              ((0b00 :: ~(a == 0b00)) + (0b00 :: (b[0])) + (0b00 :: (b[1])) + (0b00 :: (b[2]))) >= 0b011
        )
        >>> f.bv_weight(beta)
        0b0
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (1, 1, 0, 0)

    """
    assert issubclass(op, operation.Operation)

    if diff_type == difference.XorDiff:
        prefix = "Xor"
        assert zero2zero_weight == 0
        # for XOR differentials with Pr. 1, an input property propagates to a unique output property
        assert zero2nonzero_weight == math.inf
    elif diff_type == difference.RXDiff:
        prefix = "RX"
    else:
        raise ValueError(f"invalid diff_type {diff_type}")

    _op, _diff_type = op, diff_type
    _branch_number, _output_widths = branch_number, output_widths
    _zero2zero_weight = zero2zero_weight
    _nonzero2nonzero_weight = nonzero2nonzero_weight
    _zero2nonzero_weight, _nonzero2zero_weight = zero2nonzero_weight, nonzero2zero_weight
    _precision = precision

    class MyBranchNumberModel(abstractproperty.opmodel.BranchNumberModel, OpModel):
        op, diff_type= _op, _diff_type
        branch_number, output_widths = _branch_number, _output_widths
        zero2zero_weight = _zero2zero_weight
        nonzero2nonzero_weight = _nonzero2nonzero_weight
        zero2nonzero_weight = _zero2nonzero_weight
        nonzero2zero_weight = _nonzero2zero_weight
        precision = _precision

    MyBranchNumberModel.__name__ = f"{prefix}{abstractproperty.opmodel.BranchNumberModel.__name__}{op.__name__}"
    return MyBranchNumberModel


@functools.lru_cache(maxsize=None)
def get_wdt_model(op, diff_type, weight_distribution_table, loop_rows_then_columns=True, precision=0):
    """Return the WDT-based model of the given bit-vector operation ``op``.

    Given the `Operation` ``op``, return the `WDTModel`
    of ``op`` for the `Difference` type ``diff_type`` with given class
    attributes ``weight_distribution_table``
    (i.e., the Difference Distribution Table (DDT) given as a `tuple` of `tuple`
    of differential weights),
    ``loop_rows_then_columns`` and  ``precision`` (see `WDTModel`).

    The returned model is a subclass of `WDTModel` and `OpModel`.

    .. note::
        To link the returned model ``MyModel`` to ``op``
        such that ``MyModel`` is used in ``propagate``,
        set the ``xor_model`` or ``rx_model`` attribute of ``op``
        to ``MyModel`` (e.g., ``op.xor_model = MyModel``).
        See also  `differential.difference.XorDiff.propagate`
        or `differential.difference.RXDiff.propagate`.

    ::

        >>> # example of a XOR WDTModel for a 3-bit permutation
        >>> from decimal import Decimal
        >>> from math import inf
        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import LutOperation
        >>> from cascada.bitvector.printing import BvWrapPrinter
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.opmodel import log2_decimal, get_wdt_model
        >>> # 3-bit permutation with 4 different weights
        >>> class Sbox3b(LutOperation): lut = [Constant(i, 3) for i in (0, 1, 2, 3, 4, 6, 7, 5)]
        >>> ddt = [(8, 0, 0, 0, 0, 0, 0, 0), (0, 4, 4, 0, 0, 0, 0, 0), (0, 0, 4, 4, 0, 0, 0, 0), (0, 4, 0, 4, 0, 0, 0, 0),
        ...        (0, 0, 0, 0, 2, 2, 2, 2), (0, 0, 0, 0, 2, 2, 2, 2), (0, 0, 0, 0, 2, 2, 2, 2), (0, 0, 0, 0, 2, 2, 2, 2)]
        >>> num_inputs = Decimal(2**3)
        >>> wdt = tuple([tuple(inf if x == 0 else -log2_decimal(x/num_inputs) for x in row) for row in ddt])
        >>> XorWDTModelSbox3b = get_wdt_model(Sbox3b, XorDiff, wdt)
        >>> alpha, beta = XorDiff(Variable("a", 3)), XorDiff(Variable("b", 3))
        >>> f = XorWDTModelSbox3b(alpha)
        >>> print(f.vrepr())
        XorWDTModelSbox3b(XorDiff(Variable('a', width=3)))
        >>> BvWrapPrinter.new_line_right_parenthesis = False
        >>> print(BvWrapPrinter().doprint(f.validity_constraint(beta)))
        Ite(a == 0b011,
            (b == 0b001) | (b == 0b011),
            Ite(a == 0b010,
                (b == 0b010) | (b == 0b011),
                Ite(a == 0b001,
                    (b == 0b001) | (b == 0b010),
                    Ite(a == 0b000, b == 0b000, (b == 0b100) | (b == 0b101) | (b == 0b110) | (b == 0b111)))))
        >>> f.pr_one_constraint(beta)
        Ite(a == 0b000, b == 0b000, 0b0)
        >>> f.bv_weight(beta)
        Ite((a == 0b001) | (a == 0b010) | (a == 0b011), 0b01, Ite(a == 0b000, 0b00, 0b10))
        >>> BvWrapPrinter.new_line_right_parenthesis = True
        >>> x = Constant(0, 3)
        >>> f.eval_derivative(x)  # f(x + alpha) - f(x)
        XorDiff(Sbox3b(a))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (2, 2, 0, 0)

    ::

        >>> # example of a RX WDTModel for a (3,2)-bit function where 0 -> 0 and !=0 -> !=0
        >>> from cascada.differential.difference import RXDiff
        >>> class Lut3to2b(LutOperation): lut = [Constant(0, 3) for i in reversed(range(2 ** 2))]
        >>> w = decimal.Decimal("0.25")
        >>> wdt = tuple([(0, inf, inf, inf)] + [(inf, w, w, w)]*7)
        >>> RXWDTModelLut3to2b = get_wdt_model(Lut3to2b, RXDiff, wdt, precision=2)
        >>> alpha, beta = RXDiff(Variable("a", 3)), RXDiff(Variable("b", 2))
        >>> f = RXWDTModelLut3to2b(alpha)
        >>> print(f.vrepr())
        RXWDTModelLut3to2b(RXDiff(Variable('a', width=3)))
        >>> f.validity_constraint(beta)
        Ite(a == 0b000, b == 0b00, ~(b == 0b00))
        >>> f.pr_one_constraint(beta)
        Ite(a == 0b000, b == 0b00, 0b0)
        >>> f.bv_weight(beta)
        Ite(a == 0b000, 0b0, 0b1)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (1, 1, 0, 2)

    """
    assert issubclass(op, operation.Operation)

    if diff_type == difference.XorDiff:
        prefix = "Xor"
    elif diff_type == difference.RXDiff:
        prefix = "RX"
    else:
        raise ValueError(f"invalid diff_type {diff_type}")

    for row in weight_distribution_table:
        if all(w == math.inf for w in row):
            msg = "weight_distribution_table contains a row with only math.inf values"
            if diff_type == difference.XorDiff:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
    if weight_distribution_table[0][0] != 0:
        msg = "weight_distribution_table[0][0] != 0"
        if diff_type == difference.XorDiff:
            raise ValueError(msg)
        else:
            warnings.warn(msg)

    _op, _diff_type = op, diff_type
    _weight_distribution_table = weight_distribution_table[:]
    _loop_rows_then_columns = loop_rows_then_columns
    _precision = precision

    class MyWDTModel(abstractproperty.opmodel.WDTModel, OpModel):
        op, diff_type = _op, _diff_type
        weight_distribution_table = _weight_distribution_table
        loop_rows_then_columns = _loop_rows_then_columns
        precision = _precision

    MyWDTModel.__name__ = f"{prefix}{abstractproperty.opmodel.WDTModel.__name__}{op.__name__}"
    return MyWDTModel
