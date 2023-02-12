"""Manipulate algebraic models of bit-vector operations.

.. autosummary::
   :nosignatures:

    OpModel
    BitModelId
    WordModelId
    BitModelBvAdd
    BitModelBvSub
"""
import decimal
import functools

from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import context

from cascada import abstractproperty

from cascada.algebraic import value


zip = functools.partial(zip, strict=True)
make_partial_op_model = abstractproperty.opmodel.make_partial_op_model


class OpModel(abstractproperty.opmodel.OpModel):
    """Represent algebraic models of bit-vector operations.

    A (bit-vector) algebraic model of a bit-vector `Operation` :math:`f`
    is a set of bit-vector constraints that models the propagation
    probability of value properties of :math:`f`. See `Value`.

    An algebraic model is defined for a type of `Value`,
    such as `BitValue` or `WordValue`.

    This class is not meant to be instantiated but to provide a base
    class for creating algebraic models.

    .. note::

        For all algebraic models, the ``pr_one_constraint`` method is
        equivalent  to the ``validity_constraint`` method, and the weight of
        a valid propagation is always ``Constant(0, 1)``.
        This class implements the methods ``pr_one_constraint, bv_weight,
        max_weight, weight_width, decimal_weight, num_frac_bits`` and ``error``,
        and subclasses only need to implement ``validity_constraint``.

    Attributes:
        val_type: the type of `Value` (alias of
            `abstractproperty.opmodel.OpModel.prop_type`).
        input_val: a list containing the `Value` of each bit-vector
            operand (alias of `abstractproperty.opmodel.OpModel.input_prop`).

    """
    val_type = None

    @property
    def input_val(self):
        return self.input_prop

    @classmethod
    @property
    def prop_type(cls):
        return cls.val_type

    def pr_one_constraint(self, output_val):
        return self.validity_constraint(output_val)

    def bv_weight(self, output_val):
        return core.Constant(0, width=1)

    def max_weight(self):
        return 0

    def weight_width(self):
        return 1

    def decimal_weight(self, output_val):
        self._assert_preconditions_decimal_weight(output_val)
        dw = decimal.Decimal(int(self.bv_weight(output_val)))  # num_frac_bits = 0
        self._assert_postconditions_decimal_weight(output_val, dw)
        assert dw == 0
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class BitModelId(abstractproperty.opmodel.ModelIdentity, OpModel):
    """Represent the `BitValue` `algebraic.opmodel.OpModel` of `BvIdentity`.

    See also `ModelIdentity`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.algebraic.value import BitValue
        >>> from cascada.algebraic.opmodel import BitModelId
        >>> alpha, beta = BitValue(Variable("a", 4)), BitValue(Variable("b", 4))
        >>> f = BitModelId(alpha)
        >>> print(f.vrepr())
        BitModelId(BitValue(Variable('a', width=4)))
        >>> f.validity_constraint(beta)
        ((a[0]) == (b[0])) & ((a[1]) == (b[1])) & ((a[2]) == (b[2])) & ((a[3]) == (b[3]))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    val_type = value.BitValue

    def validity_constraint(self, output_val):
        assert self.input_val[0].val.width == output_val.val.width
        return functools.reduce(
            operation.BvAnd,
            [operation.BvComp(x, y) for x, y in zip(
                self.input_val[0].get_value(), output_val.get_value()
            )]
        )


class WordModelId(abstractproperty.opmodel.ModelIdentity, OpModel):
    """Represent the `WordValue` `algebraic.opmodel.OpModel` of `BvIdentity`.

    See also `ModelIdentity`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.algebraic.value import WordValue
        >>> from cascada.algebraic.opmodel import WordModelId
        >>> alpha, beta = WordValue(Variable("a", 4)), WordValue(Variable("b", 4))
        >>> f = WordModelId(alpha)
        >>> print(f.vrepr())
        WordModelId(WordValue(Variable('a', width=4)))
        >>> f.validity_constraint(beta)
        a == b
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    val_type = value.WordValue


class BitModelBvAdd(OpModel):
    """Represent the `BitValue` `algebraic.opmodel.OpModel` of `BvAdd`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.algebraic.value import BitValue
        >>> from cascada.algebraic.opmodel import BitModelBvAdd
        >>> alpha = BitValue(Variable("a", 4)), BitValue(Variable("b", 4))
        >>> f = BitModelBvAdd(alpha)
        >>> print(f.vrepr())
        BitModelBvAdd([BitValue(Variable('a', width=4)), BitValue(Variable('b', width=4))])
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    val_type = value.BitValue
    op = operation.BvAdd

    def validity_constraint(self, output_val):
        """Return the validity constraint for a given `BitValue` output.

        This model is based on the fact that the :math:`n`-bit modular addition
        :math:`x \\boxplus y = z` has the following implicit function
        :math:`x \oplus y \oplus z \oplus Q(x \oplus z, y \oplus z)`
        where :math:`Q(x, y) = (0, x_0 \wedge y_0,
        (x_0 \wedge y_0) \oplus (x_1 \wedge y_1), \dots,
        (x_0 \wedge y_0) \oplus (x_1 \wedge y_1) \oplus \dots \oplus
        (x_{n-2} \wedge y_{n-2}))`

        Source: *Implicit White-Box Implementations: White-Boxing ARX Ciphers*.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.opmodel import BitModelBvAdd
            >>> alpha = BitValue(Constant(0, 3)), BitValue(Constant(0, 3))
            >>> f = BitModelBvAdd(alpha)
            >>> beta = BitValue(Constant(0, 3))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 3), Variable("a1", 3), Variable("b", 3)
            >>> alpha = BitValue(a0), BitValue(a1)
            >>> f = BitModelBvAdd(alpha)
            >>> beta = BitValue(b)
            >>> result = f.validity_constraint(beta)
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            (0b0 == ((a0[0]) ^ (a1[0]) ^ (b[0]))) &
            (0b0 == ((a0[1]) ^ (a1[1]) ^ (b[1]) ^ (((a0[0]) ^ (b[0])) & ((a1[0]) ^ (b[0]))))) &
            (0b0 == ((a0[2]) ^ (a1[2]) ^ (b[2]) ^ (((a0[0]) ^ (b[0])) & ((a1[0]) ^ (b[0]))) ^ (((a0[1]) ^ (b[1])) & ((a1[1]) ^ (b[1])))))
            >>> result.xreplace({a0: Constant(0, 3), a1: Constant(0, 3), b: Constant(0, 3)})
            0b1
            >>> a1 = Constant(0, 3)
            >>> alpha = BitValue(a0), BitValue(a1)
            >>> f = BitModelBvAdd(alpha)
            >>> beta = BitValue(b)
            >>> result = f.validity_constraint(beta)
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            (0b0 == ((a0[0]) ^ (b[0]))) &
            (0b0 == ((a0[1]) ^ (b[1]) ^ (((a0[0]) ^ (b[0])) & (b[0])))) &
            (0b0 == ((a0[2]) ^ (b[2]) ^ (((a0[0]) ^ (b[0])) & (b[0])) ^ (((a0[1]) ^ (b[1])) & (b[1]))))

        See `OpModel.validity_constraint` for more information.
        """
        x, y = self.input_val[0].get_value(), self.input_val[1].get_value()
        z = output_val.get_value()

        n = len(x)

        zero_bit = core.Constant(0, 1)

        def q(my_x, my_y):
            res = [None for _ in range(n)]
            res[0] = zero_bit
            for i in range(1, n):
                res[i] = res[i-1] ^ (my_x[i - 1] & my_y[i - 1])
            return res

        def xor_tuples(my_tuple, my_other_tuple):
            assert len(my_tuple) == len(my_other_tuple)
            return tuple(x_i ^ y_i for x_i, y_i in zip(my_tuple, my_other_tuple))

        with context.Simplification(False), context.Validation(False):
            q_x_z_y_z = q(xor_tuples(x, z), xor_tuples(y, z))

            implicit_function = xor_tuples(x, y)
            implicit_function = xor_tuples(implicit_function, z)
            implicit_function = xor_tuples(implicit_function, q_x_z_y_z)

            # true if implicit_function == 0...0
            return functools.reduce(
                operation.BvAnd,
                [operation.BvComp(zero_bit, implicit_function[i]) for i in range(n)])


class BitModelBvSub(OpModel):
    """Represent the `BitValue` `algebraic.opmodel.OpModel` of `BvSub`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.algebraic.value import BitValue
        >>> from cascada.algebraic.opmodel import BitModelBvSub
        >>> alpha = BitValue(Variable("a", 4)), BitValue(Variable("b", 4))
        >>> f = BitModelBvSub(alpha)
        >>> print(f.vrepr())
        BitModelBvSub([BitValue(Variable('a', width=4)), BitValue(Variable('b', width=4))])
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    The algebraic model of the modular substraction is based on `BitModelBvAdd`
    since ``~(x - y) == ~x + y``.
    """
    val_type = value.BitValue
    op = operation.BvSub

    def validity_constraint(self, output_val):
        """Return the validity constraint for a given `BitValue` output.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.opmodel import BitModelBvSub
            >>> alpha = BitValue(Constant(0, 3)), BitValue(Constant(0, 3))
            >>> f = BitModelBvSub(alpha)
            >>> beta = BitValue(Constant(0, 3))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 3), Variable("a1", 3), Variable("b", 3)
            >>> alpha = BitValue(a0), BitValue(a1)
            >>> f = BitModelBvSub(alpha)
            >>> beta = BitValue(b)
            >>> result = f.validity_constraint(beta)
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            (0b0 == (~(a0[0]) ^ (a1[0]) ^ ~(b[0]))) &
            (0b0 == (~(a0[1]) ^ (a1[1]) ^ ~(b[1]) ^ ((~(a0[0]) ^ ~(b[0])) & ((a1[0]) ^ ~(b[0]))))) &
            (0b0 == (~(a0[2]) ^ (a1[2]) ^ ~(b[2]) ^ ((~(a0[0]) ^ ~(b[0])) & ((a1[0]) ^ ~(b[0]))) ^ ((~(a0[1]) ^ ~(b[1])) & ((a1[1]) ^ ~(b[1])))))
            >>> result.xreplace({a0: Constant(0, 3), a1: Constant(0, 3), b: Constant(0, 3)})
            0b1
            >>> a1 = Constant(0, 3)
            >>> alpha = BitValue(a0), BitValue(a1)
            >>> f = BitModelBvSub(alpha)
            >>> beta = BitValue(b)
            >>> result = f.validity_constraint(beta)
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            (0b0 == (~(a0[0]) ^ ~(b[0]))) &
            (0b0 == (~(a0[1]) ^ ~(b[1]) ^ ((~(a0[0]) ^ ~(b[0])) & ~(b[0])))) &
            (0b0 == (~(a0[2]) ^ ~(b[2]) ^ ((~(a0[0]) ^ ~(b[0])) & ~(b[0])) ^ ((~(a0[1]) ^ ~(b[1])) & ~(b[1]))))

        See `OpModel.validity_constraint` for more information.
        """
        # ~(x - y) == ~x + y
        new_input_val = [value.BitValue.propagate(operation.BvNot, self.input_val[0]), self.input_val[1]]
        new_op_model = BitModelBvAdd(new_input_val)
        new_output_val = value.BitValue.propagate(operation.BvNot, output_val)
        return new_op_model.validity_constraint(new_output_val)
