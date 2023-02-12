"""Represent linear mask properties in the context of linear cryptanalysis.

.. autosummary::
   :nosignatures:

    LinearMask
"""
import warnings

from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import ssa as cascada_ssa

from cascada import abstractproperty


class LinearMask(abstractproperty.property.Property):
    """Represent linear mask properties.

    Given a function :math:`f`,  a `LinearMask` property pair
    (also called a linear approximation) :math:`(\\alpha, \\beta)`
    is a bit-vector `Property` :math:`(\\alpha, \\beta)` where the
    propagation probability is the absolute correlation :math:`| C(\\alpha, \\beta) |`.

    The absolute correlation of a linear approximation with input mask :math:`\\alpha`
    and output mask :math:`\\beta` over a `Operation` :math:`f` is defined as
    :math:`| C(\\alpha, \\beta) | = | 2 \\times \\left(
    \# \{ x \ : \\langle \\alpha, x \\rangle = \\langle \\beta, f(x) \\rangle \}
    / 2^n \\right) \ \ - \ \  1  |`,
    where :math:`n` is the bit-width of the input :math:`x`.
    Other related notions are the bias of an approximation (its correlation
    divided by two) or the linear probability or potential
    (the square of its correlation).

    The inner product
    :math:`\\langle a, b \\rangle` is defined as
    :math:`(a_0 \wedge b_0) \oplus
    (a_1 \wedge b_1) \oplus \dots \oplus (a_{n-1} \wedge b_{n-1})`.

    Internally, `LinearMask` is a subclass of `Property` (as `XorDiff`).
    The `LinearMask` methods inherited from `Property` requiring
    arguments of type `Property` should be called instead with arguments
    of type `LinearMask`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> alpha = LinearMask(Constant(0b001, 3))
        >>> alpha
        LinearMask(0b001)
        >>> alpha.apply(Constant(0b011, 3))
        0b1
        >>> LinearMask(Variable("alpha", 3))
        LinearMask(alpha)

    """

    def apply(self, x):
        """Return the inner product of the mask and x."""
        # assert isinstance(self.val, core.Constant)
        # assert isinstance(x, core.Constant)
        assert isinstance(x, core.Term)
        assert x.width == self.val.width
        result = x[0] & self.val[0]
        for i in range(1, x.width):
            result ^= x[i] & self.val[i]
        return result

    @classmethod
    def propagate(cls, op, input_mask):
        """Propagate the given input `LinearMask` through the given operation.

        For any linear (over the binary finite field)
        operation ``op`` and any input mask ``input_mask``,
        the output mask ``output_mask`` is uniquely determined and its
        bit-vector value satisfies ``input_mask.val == M(output_mask.val)``,
        where :math:`M` is the transpose of the binary matrix representing
        ``op``.

        See `Property.propagate` for more information.

        User-defined or new `Operation` ``op`` can store its associated
        linear `linear.opmodel.OpModel` in ``op.linear_model``, as this method
        first checks whether ``op`` has its associated `linear.opmodel.OpModel`
        stored in the class attribute ``linear_model``.

            >>> from cascada.bitvector.core import Variable, Constant
            >>> from cascada.bitvector.operation import BvXor, RotateLeft, BvIdentity
            >>> from cascada.bitvector.operation import make_partial_operation
            >>> from cascada.linear.mask import LinearMask
            >>> d1, d2 = LinearMask(Variable("d1", 8)), LinearMask(Variable("d2", 8))
            >>> LinearMask.propagate(BvXor, [d1, d2])
            LinearModelBvXor([LinearMask(d1), LinearMask(d2)])
            >>> Xor1 = make_partial_operation(BvXor, tuple([None, Constant(1, 8)]))
            >>> LinearMask.propagate(Xor1, d1)
            LinearMask(d1)
            >>> Rotate1 = make_partial_operation(RotateLeft, tuple([None, 1]))
            >>> LinearMask.propagate(Rotate1, d1)
            LinearMask(d1 <<< 1)
            >>> LinearMask.propagate(BvIdentity, d1)
            LinearModelId(LinearMask(d1))

        """
        input_mask = abstractproperty.property._tuplify(input_mask)
        assert len(input_mask) == sum(op.arity)

        msg = "invalid arguments: op={}, input_mask={}".format(
            op.__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_mask])

        if not all(isinstance(mask, cls) for mask in input_mask):
            raise ValueError(msg)

        if hasattr(op, "_trivial_propagation"):
            return cls(op(*[p.val for p in input_mask]))

        if hasattr(op, "linear_model"):
            return op.linear_model(input_mask)

        # if f(x) = M(x) + ct is an affine operation then
        # (alpha, beta) = (a, b) is a linear approx. with Cr. +-1 if a = M^t · b
        # (ct does not affect the input/output mask, only the correlation sign)

        if op == operation.BvNot:
            # ~x = x ^ 11111 (similar as BvXor with a ct)
            return input_mask[0]

        if op == operation.BvXor:
            # M = [I, I], (a0, a1) = [I I]^T b = (b, b)
            # (need to add extra assertion a0 == a1)
            if input_mask[0] == input_mask[1]:
                return input_mask[0]
            from cascada.linear import opmodel
            return opmodel.LinearModelBvXor(input_mask)

        if op == operation.BvAnd:
            from cascada.linear import opmodel
            return opmodel.LinearModelBvAnd(input_mask)

        if op == operation.BvOr:
            from cascada.linear import opmodel
            return opmodel.LinearModelBvOr(input_mask)

        # RotateLeft, ..., BvLshr only allowed for constant offsets (see below)

        if op == operation.BvAdd:
            from cascada.linear import opmodel
            return opmodel.LinearModelBvAdd(input_mask)

        if op == operation.BvSub:
            from cascada.linear import opmodel
            return opmodel.LinearModelBvSub(input_mask)

        if op == operation.Concat:
            # M = [[I, 0], [0, I]] (identity matrix)
            # (a0, a1) = M^T b = (b0, b1)
            return cls(op(*[m.val for m in input_mask]))

        if op == cascada_ssa.SSAReturn:
            if isinstance(input_mask[0].val, core.Constant):
                warnings.warn("constant propagation of output mask in "
                              f"{cls.__name__}.propagate(op={op.__name__}, "
                              f"input_mask={input_mask}")
            from cascada.linear import opmodel
            return opmodel.LinearModelId(input_mask)

        if op == operation.BvIdentity:
            if isinstance(input_mask[0].val, core.Constant):
                return input_mask[0]
            else:
                from cascada.linear import opmodel
                return opmodel.LinearModelId(input_mask)

        if issubclass(op, operation.PartialOperation):
            if op.base_op == operation.BvXor:
                # M = [I], (a0) = [I]^T b0 = b0
                # XOR by a constant only changes the sign of the correlation
                assert len(input_mask) == 1
                m1 = input_mask[0]
                return m1

            if op.base_op == operation.BvAnd:
                assert len(input_mask) == 1
                m = input_mask[0]
                ct = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                if ct == ~core.Constant(0, ct.width):  # x & 1111 = x
                    return m
                else:
                    from cascada.linear import opmodel
                    return opmodel.make_partial_op_model(opmodel.LinearModelBvAndCt, op.fixed_args)(m)

            if op.base_op == operation.BvOr:
                assert len(input_mask) == 1
                m = input_mask[0]
                ct = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                if ct == core.Constant(0, ct.width):  # x | 0000 = x
                    return m
                else:
                    from cascada.linear import opmodel
                    return opmodel.make_partial_op_model(opmodel.LinearModelBvOrCt, op.fixed_args)(m)

            if op.base_op in [operation.RotateLeft, operation.RotateRight]:
                # RotateLeft and RotateRight matrices are permutation matrices (square binary matrix
                # that has exactly one entry of 1 in each row and each column and 0s elsewhere)
                # Examples:
                #   3-bit, RotateLeft by 1,
                #   RotateLeft1(0bx2x1x0) = 0bx1x0x2, thus RotateLeft1([x0, x1, x2]) = [x2, x0, x1]
                #   M = [0 0 1],
                #       [1 0 0],
                #       [0 1 0]
                #   3-bit, RotateLeft by 2,
                #   RotateLeft2(0bx2x1x0) = 0bx0x2x1, thus RotateLeft2([x0, x1, x2]) = [x1, x2, x0]
                #   M = [0 1 0],
                #       [0 0 1],
                #       [1 0 0]
                # Permutation matrices are orthogonal matrices, thus M^T = M^{-1}
                # (a0) = M^T b0, then, M^(a0) = b0
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_mask) == 1
                    m = input_mask[0]
                    return cls(op.base_op(m.val, op.fixed_args[1]))
                else:
                    raise ValueError(msg)

            if op.base_op in [operation.BvShl, operation.BvLshr]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_mask) == 1
                    m = input_mask[0]
                    if op.fixed_args[1] == 0:  # x << 0 = x
                        return m
                    else:
                        from cascada.linear import opmodel
                        if op.base_op == operation.BvShl:
                            partial_opmodel = opmodel.LinearModelBvShlCt
                        else:
                            assert op.base_op == operation.BvLshr
                            partial_opmodel = opmodel.LinearModelBvLshrCt
                        return opmodel.make_partial_op_model(partial_opmodel, op.fixed_args)(m)
                else:
                    raise ValueError(msg)

            if op.base_op == operation.Extract:
                i, j = op.fixed_args[1], op.fixed_args[2]
                if op.fixed_args[0] is None and i is not None and j is not None:
                    assert len(input_mask) == 1
                    m = input_mask[0]
                    if i == m.val.width - 1 and j == 0:
                        return m
                    else:
                        from cascada.linear import opmodel
                        return opmodel.make_partial_op_model(opmodel.LinearModelExtractCt, op.fixed_args)(m)
                else:
                    raise ValueError(msg)

            else:
                raise ValueError(f"Linear PartialOpModel of {op.__name__} is not implemented")

        raise ValueError(msg)
