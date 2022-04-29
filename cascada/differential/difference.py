"""Represent types of difference properties in the context of differential cryptanalysis."""
import warnings

from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import secondaryop
from cascada.bitvector import ssa as cascada_ssa

from cascada import abstractproperty


class Difference(abstractproperty.property.Property):
    """Represent difference properties.

    The *difference* between two `Term` objects :math:`x` and :math:`y`
    is defined as :math:`\\alpha = y - x`
    for some *difference operation* :math:`-` (a bit-vector `Operation`).
    In other words, the pair :math:`(x, x + \\alpha)` has difference
    :math:`\\alpha`, where :math:`+` is the "inverse" of the difference operation.

    Given a function :math:`f`,  a `Difference` property pair
    (also called a differential) :math:`(\\alpha, \\beta)`
    is a bit-vector `Property` :math:`(\\alpha, \\beta)` where the
    propagation probability (also called the differential probability)
    is defined as
    :math:`DP(\\alpha, \\beta) =
    \# \{ x \ : \ f(x + \\alpha) - f(x)  = \\beta \} / 2^{n}`,
    that is, the fraction of input pairs with input difference
    :math:`\\alpha` that lead to output pairs with difference :math:`\\beta`.

    The most common difference used in differential cryptanalysis
    is the XOR difference `XorDiff` (where the difference operation
    is `BvXor`). Other examples are the additive difference
    (where the difference operation is `BvSub`) or the rotational-XOR
    difference `RXDiff`.

    This class is not meant to be instantiated but to provide a base
    class to represent types of differences.

    Internally, `Difference` is a subclass of `Property` (as `LinearMask`).
    The `Difference` methods inherited from `Property` requiring
    arguments of type `Property` should be called instead with arguments
    of type `Difference`.

    Attributes:
        diff_op: the difference `Operation`.
        inv_diff_op: the inverse of the difference operation.

    """
    diff_op = None
    inv_diff_op = None

    def get_pair_element(self, x):
        """Return the `Term` :math:`y` such that :math:`y = \\alpha + x`."""
        assert isinstance(x, core.Term)
        return self.inv_diff_op(x, self.val)

    @classmethod
    def from_pair(cls, x, y):
        """Return the `Difference` :math:`\\alpha = y - x` given two `Term`."""
        assert isinstance(x, core.Term)
        assert isinstance(y, core.Term)
        return cls(cls.diff_op(x, y))  # The order of the operands is important


class XorDiff(Difference):
    """Represent XOR difference properties.

    The XOR difference of two `Term` is given by the XOR
    of the terms. In other words, the *difference operation*
    of `XorDiff` is the `BvXor` (see `Difference`).

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> x, y = Constant(0b000, 3), Constant(0b000, 3)
        >>> alpha = XorDiff.from_pair(x, y)
        >>> alpha
        XorDiff(0b000)
        >>> alpha.get_pair_element(x)
        0b000
        >>> x, y = Constant(0b010, 3), Constant(0b101, 3)
        >>> alpha = XorDiff.from_pair(x, y)
        >>> alpha
        XorDiff(0b111)
        >>> alpha.get_pair_element(x)
        0b101
        >>> k = Variable("k", 8)
        >>> alpha = XorDiff.from_pair(k, k)
        >>> alpha
        XorDiff(0x00)
        >>> alpha.get_pair_element(k)
        k
    """

    diff_op = operation.BvXor
    inv_diff_op = operation.BvXor

    @classmethod
    def propagate(cls, op, input_diff):
        """Propagate the given input difference of type `XorDiff` through the given operation.

        For any operation ``op`` linear with respect to `BvXor` and any
        input difference ``input_diff``, the output difference
        is uniquely determined and its bit-vector value is ``f(input_diff.val)``.

        See `Property.propagate` for more information.

        User-defined or new `Operation` ``op`` can store its associated
        `XorDiff` `differential.opmodel.OpModel` in ``op.xor_model``,
        as this method first checks whether ``op`` has its associated
        `differential.opmodel.OpModel` stored in the class attribute ``xor_model``.

            >>> from cascada.bitvector.core import Variable, Constant
            >>> from cascada.bitvector.operation import BvXor, BvAdd, BvIdentity
            >>> from cascada.bitvector.operation import make_partial_operation
            >>> from cascada.differential.difference import XorDiff
            >>> d1, d2 = XorDiff(Variable("d1", 8)), XorDiff(Variable("d2", 8))
            >>> XorDiff.propagate(BvXor, [d1, d2])
            XorDiff(d1 ^ d2)
            >>> Xor1 = make_partial_operation(BvXor, tuple([None, Constant(1, 8)]))
            >>> XorDiff.propagate(Xor1, d1)
            XorDiff(d1)
            >>> XorDiff.propagate(BvAdd, [d1, d2])
            XorModelBvAdd([XorDiff(d1), XorDiff(d2)])
            >>> Add1 = make_partial_operation(BvAdd, tuple([None, Constant(1, 8)]))
            >>> XorDiff.propagate(Add1, d1)
            XorModelBvAddCt_{·, 0x01}(XorDiff(d1))
            >>> XorDiff.propagate(BvIdentity, d1)
            XorModelId(XorDiff(d1))

        """
        input_diff = abstractproperty.property._tuplify(input_diff)
        assert len(input_diff) == sum(op.arity)

        msg = "invalid arguments: op={}, input_diff={}".format(
            op.__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_diff])

        if not all(isinstance(diff, cls) for diff in input_diff):
            raise ValueError(msg)

        if hasattr(op, "_trivial_propagation"):
            return cls(op(*[p.val for p in input_diff]))

        if hasattr(op, "xor_model"):
            return op.xor_model(input_diff)

        if op == operation.BvNot:
            return input_diff[0]

        if op == operation.BvXor:
            return cls(op(*[d.val for d in input_diff]))

        if op == operation.BvAnd:
            # imported here to avoid circular imports
            from cascada.differential import opmodel
            return opmodel.XorModelBvAnd(input_diff)

        if op == operation.BvOr:
            from cascada.differential import opmodel
            return opmodel.XorModelBvOr(input_diff)

        # RotateLeft, ..., BvLshr only allowed for constant offsets (see below)

        if op == operation.BvAdd:
            from cascada.differential import opmodel
            return opmodel.XorModelBvAdd(input_diff)

        if op == operation.BvSub:
            from cascada.differential import opmodel
            return opmodel.XorModelBvSub(input_diff)

        if op == operation.BvNeg:
            # XorModelBvNeg == XorModelBvAddCt(Ct1) since -x = ~x + 1
            d = input_diff[0]  # BvNot does not change differences
            ct = core.Constant(1, d.val.width)
            from cascada.differential import opmodel
            return opmodel.make_partial_op_model(opmodel.XorModelBvAddCt, (None, ct))(d)

        if op == operation.Concat:
            return cls(op(*[d.val for d in input_diff]))

        if op == cascada_ssa.SSAReturn:
            if isinstance(input_diff[0].val, core.Constant):
                warnings.warn("constant propagation of output difference in "
                              f"{cls.__name__}.propagate(op={op.__name__}, "
                              f"input_diff={input_diff}")
            from cascada.differential import opmodel
            return opmodel.XorModelId(input_diff)

        if op == operation.BvIdentity:
            if isinstance(input_diff[0].val, core.Constant):
                return input_diff[0]
            else:
                from cascada.differential import opmodel
                return opmodel.XorModelId(input_diff)

        if op == secondaryop.BvIf:
            from cascada.differential import opmodel
            return opmodel.XorModelBvIf(input_diff)

        if op == secondaryop.BvMaj:
            from cascada.differential import opmodel
            return opmodel.XorModelBvMaj(input_diff)

        if issubclass(op, operation.PartialOperation):
            if op.base_op in [operation.BvXor, operation.BvOr, operation.BvAnd]:
                d1_val = input_diff[0].val
                val = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                if op.base_op == operation.BvXor:
                    d2_val = cls.from_pair(val, val).val
                    new_op = operation.BvXor
                elif op.base_op == operation.BvOr:
                    d2_val = ~val
                    new_op = operation.BvAnd
                elif op.base_op == operation.BvAnd:
                    d2_val = val
                    new_op = operation.BvAnd
                return cls(new_op(d1_val, d2_val))

            if op.base_op in [operation.RotateLeft, operation.RotateRight]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    return cls(op.base_op(d.val, op.fixed_args[1]))
                else:
                    raise ValueError(msg)

            if op.base_op in [operation.BvShl, operation.BvLshr]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    return cls(op.base_op(d.val, op.fixed_args[1]))
                else:
                    raise ValueError(msg)

            if op.base_op == operation.BvAdd:
                assert len(input_diff) == 1
                d = input_diff[0]
                ct = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                if ct == 0:
                    return d
                else:
                    from cascada.differential import opmodel
                    return opmodel.make_partial_op_model(opmodel.XorModelBvAddCt, op.fixed_args)(d)

            # currently if left op is fixed with a ct, then else code below is executed
            if op.base_op == operation.BvSub and op.fixed_args[0] is None:
                assert len(input_diff) == 1
                d = input_diff[0]
                ct = op.fixed_args[1]
                if ct == 0:
                    return d
                else:
                    from cascada.differential import opmodel
                    return opmodel.make_partial_op_model(opmodel.XorModelBvSubCt, op.fixed_args)(d)

            if op.base_op == operation.Extract:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None and op.fixed_args[2] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    return cls(op.base_op(d.val, op.fixed_args[1], op.fixed_args[2]))
                else:
                    raise ValueError(msg)

            else:
                # e.g., BvIf, BvMaj
                warnings.warn(f"XorDiff OpModel of {op.__name__} is not implemented;"
                              f" instead using OpModel of {op.base_op.__name__} with "
                              f"zero difference for each fixed operand")
                new_input_diff = []
                counter_non_fixed_args = 0
                for fa in op.fixed_args:
                    if fa is not None:
                        if isinstance(fa, int):
                            raise ValueError(msg)
                        new_input_diff.append(cls.from_pair(fa, fa))
                    else:
                        new_input_diff.append(input_diff[counter_non_fixed_args])
                        counter_non_fixed_args += 1
                assert len(input_diff) == counter_non_fixed_args
                return cls.propagate(op.base_op, new_input_diff)

        raise ValueError(msg)


class RXOp(operation.SecondaryOperation):
    """The difference operation of `RXDiff` given by ``(x, y) |--> (x <<< 1) ^ y.``"""

    arity = [2, 0]
    is_symmetric = False
    is_simple = True

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        return operation.RotateLeft(x, 1) ^ y


class RXInvOp(operation.SecondaryOperation):
    """The inverse of the difference operation of `RXDiff` given by ``(x, d) |--> (x <<< 1) ^ d.``"""

    arity = [2, 0]
    is_symmetric = False
    is_simple = True

    @classmethod
    def condition(cls, x, d):
        return x.width == d.width

    @classmethod
    def output_width(cls, x, d):
        return x.width

    @classmethod
    def eval(cls, x, d):
        return operation.RotateLeft(x, 1) ^ d


class RXDiff(Difference):
    """Represent rotational-XOR (RX) difference properties.

    The pair ``(x, (x <<< 1) ^ d)`` has RX difference ``d``.
    In other words,  the RX difference of two `Term` ``x`` and ``y``
    is defined as ``(x <<< 1) ^ y``.

    This definition of rotational-XOR difference is equivalent but
    slightly different to the definitions presented in
    `Rotational Cryptanalysis in the Presence of Constants
    <https://doi.org/10.13154/tosc.v2016.i1.57-70>`_
    and `Rotational-XOR Cryptanalysis of Reduced-round SPECK
    <https://doi.org/10.13154/tosc.v2017.i3.24-36>`_.

    See `Difference` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import RotateLeft
        >>> from cascada.differential.difference import RXDiff
        >>> x, y = Constant(0b000, 3), Constant(0b000, 3)
        >>> alpha = RXDiff.from_pair(x, y)
        >>> alpha
        RXDiff(0b000)
        >>> alpha.get_pair_element(x)
        0b000
        >>> x, y = Constant(0b000, 3), Constant(0b001, 3)
        >>> alpha = RXDiff.from_pair(x, y)
        >>> alpha
        RXDiff(0b001)
        >>> alpha.get_pair_element(x)
        0b001
        >>> x, y, d = Variable("x", 3),  Variable("y", 3), Variable("d", 3)
        >>> RXDiff.from_pair(x, y).val.doit()  # RXOp is a SecondaryOperation
        (x <<< 1) ^ y
        >>> RXDiff.from_pair(x, RotateLeft(x, 1) ^ d).val.doit()
        d
        >>> RXDiff(d).get_pair_element(x).doit()
        (x <<< 1) ^ d

    """

    diff_op = RXOp
    inv_diff_op = RXInvOp

    @classmethod
    def propagate(cls, op, input_diff):
        """Propagate the given input difference of type `RXDiff` through the given operation.

        For any operation ``op`` linear with respect to `RXOp` and any
        input difference ``input_diff``, the output difference
        is uniquely determined and its bit-vector value is ``f(input_diff.val)``.

        See `Property.propagate` for more information.

        User-defined or new `Operation` ``op`` can store its associated `RXDiff`
        `differential.opmodel.OpModel` in ``op.rx_model``, as this method first
        checks whether ``op`` has its associated `differential.opmodel.OpModel`
        stored in the class attribute ``rx_model``.

            >>> from cascada.bitvector.core import Variable, Constant
            >>> from cascada.bitvector.operation import BvAdd, BvXor, BvShl, BvIdentity
            >>> from cascada.bitvector.operation import make_partial_operation
            >>> from cascada.differential.difference import RXDiff
            >>> d1, d2 = RXDiff(Variable("d1", 8)), RXDiff(Variable("d2", 8))
            >>> RXDiff.propagate(BvXor, [d1, d2])
            RXDiff(d1 ^ d2)
            >>> Xor1 = make_partial_operation(BvXor, tuple([None, Constant(1, 8)]))
            >>> RXDiff.propagate(Xor1, d1)
            RXDiff(d1 ^ 0x03)
            >>> RXDiff.propagate(BvAdd, [d1, d2])
            RXModelBvAdd([RXDiff(d1), RXDiff(d2)])
            >>> Shl1 = make_partial_operation(BvShl, tuple([None, Constant(1, 8)]))
            >>> RXDiff.propagate(Shl1, d1)
            RXModelBvShlCt_{·, 0x01}(RXDiff(d1))
            >>> RXDiff.propagate(BvIdentity, d1)
            RXModelId(RXDiff(d1))

        """
        input_diff = abstractproperty.property._tuplify(input_diff)
        assert len(input_diff) == sum(op.arity)

        msg = "invalid arguments: op={}, input_diff={}".format(
            op.__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_diff])

        if not all(isinstance(diff, cls) for diff in input_diff):
            raise ValueError(msg)

        if hasattr(op, "_trivial_propagation"):
            return cls(op(*[p.val for p in input_diff]))

        if hasattr(op, "rx_model"):
            return op.rx_model(input_diff)

        if op == operation.BvNot:
            return input_diff[0]

        if op == operation.BvXor:
            return cls(op(*[d.val for d in input_diff]))

        if op == operation.BvAnd:
            # imported here to avoid circular imports
            from cascada.differential import opmodel
            return opmodel.RXModelBvAnd(input_diff)

        if op == operation.BvOr:
            from cascada.differential import opmodel
            return opmodel.RXModelBvOr(input_diff)

        # RotateLeft, ..., BvLshr only allowed for constant offsets (see below)

        if op == operation.BvAdd:
            from cascada.differential import opmodel
            return opmodel.RXModelBvAdd(input_diff)

        if op == operation.BvSub:
            from cascada.differential import opmodel
            return opmodel.RXModelBvSub(input_diff)

        if op == operation.BvNeg:
            from cascada.differential import opmodel
            warnings.warn(f"RXDiff OpModel of {op.__name__} is not implemented;"
                          f" instead using {opmodel.RXModelBvSub.__name__}")
            d2 = input_diff[0]
            n = d2.val.width
            d1 = cls.from_pair(core.Constant(0, n), core.Constant(0, n))
            return opmodel.RXModelBvSub([d1, d2])

        # Concat RX-propagation is not deterministic

        if op == cascada_ssa.SSAReturn:
            if isinstance(input_diff[0].val, core.Constant):
                warnings.warn("constant propagation of output difference in "
                              f"{cls.__name__}.propagate(op={op.__name__}, "
                              f"input_diff={input_diff}")
            from cascada.differential import opmodel
            return opmodel.RXModelId(input_diff)

        if op == operation.BvIdentity:
            if isinstance(input_diff[0].val, core.Constant):
                return input_diff[0]
            else:
                from cascada.differential import opmodel
                return opmodel.RXModelId(input_diff)

        if op == secondaryop.BvIf:
            from cascada.differential import opmodel
            return opmodel.RXModelBvIf(input_diff)

        if op == secondaryop.BvMaj:
            from cascada.differential import opmodel
            return opmodel.RXModelBvMaj(input_diff)

        if issubclass(op, operation.PartialOperation):
            if op.base_op == operation.BvXor:
                d1 = input_diff[0]
                val = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                d2 = cls.from_pair(val, val)
                input_diff = [d1, d2]
                return cls(op.base_op(*[d.val for d in input_diff]))

            # fixed BvAnd and BvOr non-trivial to propagate

            if op.base_op in [operation.RotateLeft, operation.RotateRight]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    return cls(op.base_op(d.val, op.fixed_args[1]))
                else:
                    raise ValueError(msg)

            if op.base_op in [operation.BvShl, operation.BvLshr]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    ct = int(op.fixed_args[1])
                    if ct == 0:
                        return d
                    elif ct >= d.val.width:
                        return type(d)(core.Constant(0, d.val.width))
                    else:
                        from cascada.differential import opmodel
                        if op.base_op == operation.BvShl:
                            my_op_model = opmodel.RXModelBvShlCt
                        else:
                            my_op_model = opmodel.RXModelBvLshrCt
                        return opmodel.make_partial_op_model(my_op_model, op.fixed_args)(d)
                else:
                    raise ValueError(msg)

            # Extract is non-trivial for some indices

            else:
                # e.g., BvAnd, BvIf
                warnings.warn(f"RXDiff OpModel of {op.__name__} is not implemented;"
                              f" instead using OpModel of {op.base_op.__name__} with "
                              f"constant differences for fixed operands")
                new_input_diff = []
                counter_non_fixed_args = 0
                for fa in op.fixed_args:
                    if fa is not None:
                        if isinstance(fa, int):
                            raise ValueError(msg)
                        new_input_diff.append(cls.from_pair(fa, fa))
                    else:
                        new_input_diff.append(input_diff[counter_non_fixed_args])
                        counter_non_fixed_args += 1
                assert len(input_diff) == counter_non_fixed_args
                return cls.propagate(op.base_op, new_input_diff)

        raise ValueError(msg)
