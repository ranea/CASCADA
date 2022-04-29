"""Represent type of value properties."""
import functools
import warnings

from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import secondaryop
from cascada.bitvector import ssa as cascada_ssa

from cascada import abstractproperty


zip = functools.partial(zip, strict=True)


class Value(abstractproperty.property.Property):
    """Represent value properties.

    Given a function :math:`f`,  a `Value` property pair
    :math:`(\\alpha, \\beta)` is a bit-vector `Property` :math:`(\\alpha, \\beta)`
    where the propagation probability
    is simply 1 if :math:`\\beta = f(\\alpha)` otherwise 0, that is,
    the propagation probability is 0 or 1 depending on whether
    the output value is the image of the input value through :math:`f`.

    This class is not meant to be instantiated but to provide a base
    class for the different representations of the value of a bit-vector
    (see `BitValue` or `WordValue`).

    Internally, `Value` is a subclass of `Property` (as `Difference`).
    The `Value` methods inherited from `Property` requiring
    arguments of type `Property` should be called instead with arguments
    of type `Value`.
    """

    def get_value(self):
        """Return the value of the underlying bit-vector.

        Subclasses of `Value` can represent the underlying bit-vector
        in different ways and return its value in different ways
        (i.e., not only as ``self.val``).
        """
        raise NotImplementedError("subclasses need to override this method")


class WordValue(Value):
    """Represent word value properties.

    The word value is a `Value` property where the underlying bit-vector
    value of the property is given as is (i.e., as a bit-vector `Term`).

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.algebraic.value import WordValue
        >>> alpha = WordValue(Constant(0b001, 3))
        >>> alpha, alpha.val
        (WordValue(0b001), 0b001)
        >>> alpha.get_value()
        0b001
        >>> alpha = WordValue(Variable("alpha", 3))
        >>> alpha, alpha.val
        (WordValue(alpha), alpha)

    """

    def get_value(self):
        """Return the value of the underlying bit-vector as ``self.val``."""
        return self.val

    @classmethod
    def propagate(cls, op, input_val):
        """Propagate the given input `WordValue` through the given operation.

        For any operation ``op`` and any
        input value ``input_val``, the output value ``output_val``
        is uniquely determined and its bit-vector value is ``f(input_val)``.

        See `Property.propagate` for more information.

        User-defined or new `Operation` ``op`` can store its associated
        `WordValue` `algebraic.opmodel.OpModel` in ``op.word_model``,
        as this method first checks whether ``op`` has its associated
        `algebraic.opmodel.OpModel` stored in the class attribute ``word_model``.

            >>> from cascada.bitvector.core import Variable, Constant
            >>> from cascada.bitvector.operation import BvXor, BvIdentity
            >>> from cascada.algebraic.value import WordValue
            >>> d1, d2 = WordValue(Variable("d1", 8)), WordValue(Constant(1, 8))
            >>> WordValue.propagate(BvXor, [d1, d2])
            WordValue(d1 ^ 0x01)
            >>> WordValue.propagate(BvIdentity, d1)
            WordModelId(WordValue(d1))

        """
        input_val = abstractproperty.property._tuplify(input_val)

        assert len(input_val) == sum(op.arity)

        msg = "invalid arguments: op={}, input_val={}".format(
            op.__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_val])

        if not all(isinstance(d, cls) for d in input_val):
            raise ValueError(msg)

        if hasattr(op, "_trivial_propagation"):
            return cls(op(*[p.val for p in input_val]))

        if hasattr(op, "word_model"):
            return op.word_model(input_val)

        if op == cascada_ssa.SSAReturn:
            if isinstance(input_val[0].val, core.Constant):
                warnings.warn("constant propagation of output wordval in "
                              f"{cls.__name__}.propagate(op={op.__name__}, "
                              f"input_val={input_val}")
            from cascada.algebraic import opmodel
            return opmodel.WordModelId(input_val)

        if op == operation.BvIdentity:
            if isinstance(input_val[0].val, core.Constant):
                return input_val[0]
            else:
                from cascada.algebraic import opmodel
                return opmodel.WordModelId(input_val)

        return cls(op(*[x.val for x in input_val]))


class BitValue(Value):
    """Represent bit value properties.

    The bit value is a `Value` property where the underlying bit-vector
    value of the property is given as the list of its bits
    (starting from the least significant bit (LSB) and ending in the MSB).

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.algebraic.value import BitValue
        >>> alpha = BitValue(Constant(0b001, 3))
        >>> alpha, alpha.val
        (BitValue(0b001), 0b001)
        >>> alpha.get_value()
        (0b1, 0b0, 0b0)
        >>> alpha = BitValue(Variable("alpha", 3))
        >>> alpha, alpha.val
        (BitValue(alpha), alpha)
        >>> alpha.get_value()
        (alpha[0], alpha[1], alpha[2])

    """

    def get_value(self):
        """Return the value of the underling bit-vector as the
        list of its bits."""
        return tuple(self.val[i] for i in range(self.val.width))

    @classmethod
    def propagate(cls, op, input_val):
        """Propagate the given input `BitValue` through the given operation.

        For any operation ``op`` and any
        input value ``input_val``, the output value
        is uniquely determined and its bit-vector value is ``f(input_val)``.

        For some operations, the output value is given implicitly
        through a system of binary equations or constraints
        (see `BitModelBvAdd`). For these operations,
        this method return an `algebraic.opmodel.OpModel`.

        See `Property.propagate` for more information.

        User-defined or new `Operation` ``op`` can store its associated
        `BitValue` `algebraic.opmodel.OpModel` in ``op.bit_model``, as this method
        first checks whether ``op`` has its associated `algebraic.opmodel.OpModel`
        stored in the class attribute ``bit_model``.

            >>> from cascada.bitvector.core import Variable, Constant
            >>> from cascada.bitvector.operation import BvXor, Concat, BvIdentity
            >>> from cascada.algebraic.value import BitValue
            >>> d1, d2 = BitValue(Variable("d1", 2)), BitValue(Constant(1, 2))
            >>> BitValue.propagate(BvXor, [d1, d2])
            BitValue((d1[1]) :: ~(d1[0]))
            >>> d1, d2 = BitValue(Variable("d1", 2)), BitValue(Variable("d2", 2))
            >>> BitValue.propagate(Concat, [d1, d2])
            BitValue(d1 :: (d2[1]) :: (d2[0]))
            >>> BitValue.propagate(BvIdentity, d1)
            BitValue(Id(d1[1]) :: Id(d1[0]))

        """
        input_val = abstractproperty.property._tuplify(input_val)
        assert len(input_val) == sum(op.arity)

        msg = "invalid arguments: op={}, input_val={}".format(
            op.__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_val])

        if not all(isinstance(d, cls) for d in input_val):
            raise ValueError(msg)

        if hasattr(op, "_trivial_propagation"):
            return cls(op(*[p.val for p in input_val]))

        if hasattr(op, "bit_model"):
            return op.bit_model(input_val)

        # f(x, y) is bitwise if f(x, y) == f(x0, y0) || ... || f(xn-1, yn-1)
        bitwise_ops = [
            operation.BvNot, operation.BvIdentity,  # unary
            operation.BvAnd, operation.BvOr, operation.BvXor,
            secondaryop.BvMaj, secondaryop.BvIf
        ]

        if op in bitwise_ops:
            # l = list of 1-bit inputs
            out_bitlist = [op(*l) for l in zip(*[x.get_value() for x in input_val])]
            return cls(functools.reduce(operation.Concat, reversed(out_bitlist)))  # back to single BV

        if op == operation.BvAdd:
            from cascada.algebraic import opmodel
            return opmodel.BitModelBvAdd(input_val)

        if op == operation.BvSub:
            from cascada.algebraic import opmodel
            return opmodel.BitModelBvSub(input_val)

        if op == operation.BvNeg:
            from cascada.algebraic import opmodel
            zero_val = BitValue(core.Constant(0, input_val[0].val.width))
            return opmodel.BitModelBvSub([zero_val, input_val[0]])

        if op == operation.Concat:
            in_bitlist = list(reversed(input_val[0].get_value()))
            in_bitlist += list(reversed(input_val[1].get_value()))
            return cls(functools.reduce(operation.Concat, in_bitlist))

        if op == cascada_ssa.SSAReturn:
            if isinstance(input_val[0].val, core.Constant):
                warnings.warn("constant propagation of output bitval in "
                              f"{cls.__name__}.propagate(op={op.__name__}, "
                              f"input_val={input_val}")
            from cascada.algebraic import opmodel
            return opmodel.BitModelId(input_val)

        if op == operation.BvIdentity:
            if isinstance(input_val[0].val, core.Constant):
                return input_val[0]
            else:
                from cascada.algebraic import opmodel
                return opmodel.BitModelId(input_val)

        if issubclass(op, operation.PartialOperation):
            if op.base_op in [operation.RotateLeft, operation.RotateRight]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    x, r = input_val[0].get_value(), op.fixed_args[1]
                    if r % len(x) == 0:
                        return input_val[0]
                    if op.base_op == operation.RotateLeft:
                        #               xn-1, xn-2, ..., x1, x0
                        # rl(x, 2) =    xn-3, ...., x1, x0, xn-1, xn-2
                        # reversed(·) = xn-2, xn-1, x0, x1, ..., xn-3
                        out_bitlist = x[len(x) - r:] + x[:len(x) - r]
                    else:
                        #               x-1, xn-2, ..., x1, x0
                        # rr(x, 2) =    x1, x0, xn-1, ...., x2
                        # reversed(·) = x2, ..., xn-1, x0, x1
                        out_bitlist = x[r:] + x[:r]
                    return cls(functools.reduce(operation.Concat, reversed(out_bitlist)))
                else:
                    raise ValueError(msg)

            if op.base_op in [operation.BvShl, operation.BvLshr]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    x, r = input_val[0].get_value(), int(op.fixed_args[1])
                    if r == 0:
                        return input_val[0]
                    if r >= len(x):
                        return cls(core.Constant(0, len(x)))
                    zero_list = tuple(core.Constant(0, 1) for _ in range(r))
                    if op.base_op == operation.BvShl:
                        #               xn-1, xn-2, ..., x1, x0
                        # rs(x, 2) =    xn-3, ...., x1, x0, 0, 0
                        # reversed(·) = 0, 0, x0, x1, ..., xn-3
                        out_bitlist = zero_list + x[:len(x) - r]
                    else:
                        #               x-1, xn-2, ..., x1, x0
                        # rs(x, 2) =    0, 0, xn-1, ...., x2
                        # reversed(·) = x2, ..., xn-1, 0, 0
                        out_bitlist = x[r:] + zero_list
                    return cls(functools.reduce(operation.Concat, reversed(out_bitlist)))
                else:
                    raise ValueError(msg)

            if op.base_op == operation.Extract:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None and op.fixed_args[2] is not None:
                    assert len(input_val) == 1
                    x = input_val[0].get_value()
                    i, j = op.fixed_args[1], op.fixed_args[2]
                    n = len(x)
                    out_bitlist = list(reversed(x))[n - 1 - i:n - j]
                    if i == j:
                        assert len(out_bitlist) == 1
                        return cls(out_bitlist[0])
                    else:
                        return cls(functools.reduce(operation.Concat, out_bitlist))
                else:
                    raise ValueError(msg)

            else:
                new_input_val = []
                counter_non_fixed_args = 0
                for fa in op.fixed_args:
                    if fa is not None:
                        if isinstance(fa, int):
                            raise ValueError(msg)
                        new_input_val.append(cls(fa))
                    else:
                        new_input_val.append(input_val[counter_non_fixed_args])
                        counter_non_fixed_args += 1
                assert len(input_val) == counter_non_fixed_args
                return cls.propagate(op.base_op, new_input_val)

        raise ValueError(msg)

