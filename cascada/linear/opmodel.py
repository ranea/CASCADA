"""Manipulate linear models of bit-vector operations.

.. autosummary::
   :nosignatures:

    OpModel
    PartialOpModel
    LinearModelId
    LinearModelFreeBranch
    LinearModelBvXor
    LinearModelBvAnd
    LinearModelBvOr
    LinearModelBvAdd
    LinearModelBvSub
    LinearModelBvAndCt
    LinearModelBvOrCt
    LinearModelBvShlCt
    LinearModelBvLshrCt
    LinearModelExtractCt
    get_weak_model
    get_branch_number_model
    get_wdt_model
"""
import functools
import hashlib
import math
import operator
import fractions
import decimal

from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import secondaryop

from cascada import abstractproperty

from cascada.linear.mask import LinearMask


log2_decimal = abstractproperty.opmodel.log2_decimal
make_partial_op_model = abstractproperty.opmodel.make_partial_op_model


class OpModel(abstractproperty.opmodel.OpModel):
    """Represent linear models of bit-vector operations.

    A (bit-vector) linear model of a bit-vector `Operation` :math:`f`
    is a set of bit-vector constraints that models the
    absolute correlation  of :math:`f`. See `LinearMask`.

    Internally, this class is a subclass of `abstractproperty.opmodel.OpModel`,
    where the `Property` is `LinearMask`, the pair of input and output
    properties (masks) :math:`(\\alpha, \\beta)` is called a (linear)
    approximation, and the propagation probability is the absolute correlation.

    For linear models (except `LinearModelFreeBranch`),
    the propagation probability (absolute linear correlation)
    satisfies that for any fixed :math:`\\beta`, the sum of the squares of the
    correlations (linear probabilities) of  :math:`(\\alpha, \\beta)`
    (for all :math:`\\alpha`) is equal to 1.
    Moreover, if the absolute linear correlation is 1, then
    :math:`\\beta` uniquely propagates (backwards) to :math:`\\alpha`.

    This class is not meant to be instantiated but to provide a base
    class for creating linear models.

    .. Implementation details:

        Part of the nomenclature was extracted from
        https://eprint.iacr.org/2005/212

        The capacity or data-complexity of an approximation
        with correlation :math:`c` is roughly :math:`1/c^{2}`.

        The weight of the linear probability ranges between 0 and 2(n − 1),
        max{over abs(C)} -log2(abs(C)^2) = 2(n-1)
        Thus, the weight of the correlation of an n-bit approximation
        ranges between 0 and n − 1 (n - 2 if the approximation
        is over a permutation).

        Proof:
            C = 2 x ( #{ x : ... }/2^n )  -  1  = #x/2^{n-1} - 1  (ignoring abs)
            min(abs(C)) => #x/2^{n-1} closer to 1 ==> #x = 2^{n-1} +- 1  (+- 2 if permutation)
            min(C) = (2^{n-1} +- 1)/2^{n-1} - 1 = 1 +- 1/2^{n-1} - 1 = +-2^{-(n-1)}
            max(weight) = weight(min(C)) = -log2(abs(+-2^{-(n-1)})) = n - 1

        LP is as DP is for differential cryptanalysis.

    """
    prop_type = LinearMask

    @property
    def input_mask(self):
        return self.input_prop


class PartialOpModel(abstractproperty.opmodel.PartialOpModel, OpModel):
    """Represent `linear.opmodel.OpModel` of `PartialOperation`.

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.linear.opmodel import LinearModelBvAndCt, LinearModelBvOrCt, LinearModelBvShlCt
        >>> from cascada.linear.opmodel import LinearModelExtractCt, make_partial_op_model
        >>> make_partial_op_model(LinearModelBvAndCt, tuple([None, Constant(1, 4)])).__name__
        'LinearModelBvAndCt_{·, 0x1}'
        >>> make_partial_op_model(LinearModelBvOrCt, tuple([None, Constant(1, 4)])).__name__
        'LinearModelBvOrCt_{·, 0x1}'
        >>> make_partial_op_model(LinearModelBvShlCt, tuple([None, Constant(1, 4)])).__name__
        'LinearModelBvShlCt_{·, 0x1}'
        >>> make_partial_op_model(LinearModelExtractCt, tuple([None, 2, 1])).__name__
        'LinearModelExtractCt_{·, 2, 1}'

    See also `abstractproperty.opmodel.PartialOpModel`.
    """
    pass


class LinearModelId(abstractproperty.opmodel.ModelIdentity, OpModel):
    """Represent the `linear.opmodel.OpModel` of `BvIdentity`.

    See also `ModelIdentity`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelId
        >>> alpha, beta = LinearMask(Variable("a", 4)), LinearMask(Variable("b", 4))
        >>> f = LinearModelId(alpha)
        >>> print(f.vrepr())
        LinearModelId(LinearMask(Variable('a', width=4)))
        >>> f.validity_constraint(beta)
        a == b
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    pass


class LinearModelFreeBranch(OpModel):
    """Represent the (trivial) linear model of the free branches
    of the forking or branching subroutine.

    The forking or branching subroutine is defined as

    .. code-block:: none

                |--> x
            x --|    .
                |    .
                |    .
                |--> x
                |
                |--> x

    or equivalently :math:`x \mapsto (x, x, \dots, x)`.

    Branching subroutines appears implicitly in bit-vector functions
    anytime a variable is used multiple times.

    When a `BvFunction` contains (implicit) branching subroutines
    and the `SSA` of this function is initialized with ``replace_multiuse_vars``
    set to ``True``, new variables are created to uniquely label each
    branch of the forking subroutine (see also `SSA`).

    .. note::
        For example, a 3-forking subroutine :math:`x \mapsto (x, x, x)` would be
        transformed after this preprocessing to the following 3-forking subroutine

        .. code-block:: none

                    |--> x__0
                x --|
                    |--> x__1
                    |
                    |--> x__1

    If a `linear.chmodel.ChModel` is created for a function containing
    branching subroutines, for every one of them, each right branch
    (except the last one) is assigned to a new free mask, and the last branch
    is assigned to the XOR of all masks.

    .. note::
        Following our previous example, `linear.chmodel.ChModel` would propagate
        the `LinearMask` ``m(x)`` of ``x`` through this (preprocessed)
        branching subroutine by assigning new free masks ``m(x__0), m(x__1)``
        to the first two right branches, and the last branch is set
        to the XOR of ``m(x__0)``, ``m(x__1)`` and ``m(x)``,

        .. code-block:: none

                       |--> m(x__0)
                m(x) --|
                       |--> m(x__1)
                       |
                       |--> m(x) ^ m(x__0) ^ m(x__1)

    These new free masks are associated to `LinearModelFreeBranch` objects
    in the characteristic model so that the characteristic model can keep track
    of these new free masks. In particular, each (output) free mask is associated
    in `linear.chmodel.ChModel.assign_outmask2op_model` to a `LinearModelFreeBranch`
    object with input mask the left branch mask.

    .. note::

        The model `LinearModelFreeBranch` does not represent a linear model of any
        operation. It is instead an auxiliary model to "create" new masks
        in the characteristic model without any constraints.
        For this reason, the validity constraint
        of `LinearModelFreeBranch` is always True and the weight
        is always 0 for any input and output mask.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelFreeBranch
        >>> alpha, beta = LinearMask(Variable("mx", 4)), LinearMask(Variable("mx__0", 4))
        >>> f = LinearModelFreeBranch(alpha)
        >>> print(f.vrepr())
        LinearModelFreeBranch(LinearMask(Variable('mx', width=4)))
        >>> f.validity_constraint(beta)
        0b1
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    op = operation.BvIdentity
    
    def validity_constraint(self, output_mask):
        """Return the validity constraint for a given output `LinearMask`.

        For `LinearModelFreeBranch`, all output masks are valid.
        """
        return core.Constant(1, width=1) # 1 == True

    def pr_one_constraint(self, output_mask):
        """Return the pr. 1 constraint for a given output `LinearMask`.

        For `LinearModelFreeBranch`, all output masks are valid with weight 0.
        """
        return core.Constant(1, width=1) # 1 == True

    def bv_weight(self, output_mask):
        """Return the bit-vector weight for a given output `LinearMask`.

        For `LinearModelFreeBranch`, any output mask has weight 0.
        """
        return core.Constant(0, width=1)

    def max_weight(self):
        return 0

    def weight_width(self):
        return 1

    def decimal_weight(self, output_mask):
        self._assert_preconditions_decimal_weight(output_mask)
        dw = decimal.Decimal(int(self.bv_weight(output_mask)))
        self._assert_postconditions_decimal_weight(output_mask, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class LinearModelBvXor(OpModel):
    """Represent the `linear.opmodel.OpModel` of `BvXor`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvXor
        >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
        >>> f = LinearModelBvXor(alpha)
        >>> print(f.vrepr())
        LinearModelBvXor([LinearMask(Constant(0b0000, width=4)), LinearMask(Constant(0b0000, width=4))])
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    op = operation.BvXor

    def validity_constraint(self, output_mask):
        """Return the validity constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvXor
            >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
            >>> f = LinearModelBvXor(alpha)
            >>> beta = LinearMask(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 4), Variable("a1", 4), Variable("b", 4)
            >>> alpha = LinearMask(a0), LinearMask(a1)
            >>> f = LinearModelBvXor(alpha)
            >>> beta = LinearMask(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            (a0 == a1) & (a0 == b)
            >>> result.xreplace({a0: Constant(0, 4), a1: Constant(0, 4), b: Constant(0, 4)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """
        a0, a1 = [m.val for m in self.input_mask]
        b0 = output_mask.val
        a_i = a1 if isinstance(a1, core.Constant) else a0
        return operation.BvComp(a0, a1) & operation.BvComp(a_i, b0)

    def pr_one_constraint(self, output_mask):
        """Return the probability-one constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvXor
            >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
            >>> f = LinearModelBvXor(alpha)
            >>> f.pr_one_constraint(LinearMask(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        return self.validity_constraint(output_mask)

    def bv_weight(self, output_mask):
        """Return the bit-vector weight for a given output `LinearMask`.

        For `LinearModelBvXor`, any valid linear approximation has weight 0.
        """
        return core.Constant(0, width=1)

    def weight_constraint(self, output_mask, weight_variable):
        """Return the weight constraint for a given output `LinearMask` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvXor
            >>> n = 4
            >>> alpha = LinearMask(Variable("a0", n)), LinearMask(Variable("a1", n))
            >>> f = LinearModelBvXor(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(LinearMask(Variable("b", n)), w)
            w == 0b0

        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_mask, weight_variable)

    def max_weight(self):
        return 0

    def weight_width(self):
        return 1

    def decimal_weight(self, output_mask):
        self._assert_preconditions_decimal_weight(output_mask)
        dw = decimal.Decimal(int(self.bv_weight(output_mask)))
        self._assert_postconditions_decimal_weight(output_mask, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class LinearModelBvAnd(OpModel):
    """Represent the `linear.opmodel.OpModel` of `BvAnd`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvAnd
        >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
        >>> f = LinearModelBvAnd(alpha)
        >>> print(f.vrepr())
        LinearModelBvAnd([LinearMask(Constant(0b0000, width=4)), LinearMask(Constant(0b0000, width=4))])
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    """
    op = operation.BvAnd

    def validity_constraint(self, output_mask):
        """Return the validity constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAnd
            >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
            >>> f = LinearModelBvAnd(alpha)
            >>> beta = LinearMask(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 4), Variable("a1", 4), Variable("b", 4)
            >>> alpha = LinearMask(a0), LinearMask(a1)
            >>> f = LinearModelBvAnd(alpha)
            >>> beta = LinearMask(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            (~b & (a0 | a1)) == 0x0
            >>> result.xreplace({a0: Constant(0, 4), a1: Constant(0, 4), b: Constant(0, 4)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """
        # let f(x) = x0 & x1, and LA=(a,b)=[(a0, a1), b0] 1-bit linear approximation
        # <a, x> = a0x0 + a1x1, ‹f(x), b> = b & (x0 & x1),
        # if LA = [(0, 0), 0],
        #   number of x such that <a, x> = ‹f(x), b>? (right inputs)
        #   0 and 0 are equal in 4/4 right inputs
        #   C(a,b) = |2 x 4/4 - 1| = 1,             weight 0
        # if LA in [(0, 0), 1],
        #   0 and x0 & x1 are equal in 3/4 right inputs ([x0, x1]!=[1,1])
        #   C(a,b) = |2 x 3/4 - 1| = 0.5,           weight 1
        # if LA in [(0, 1), 0], [(1, 0), 0]
        #   for [(0, 1), 0]
        #       x1 and 0 are equal in 2/4 right inputs (x0=*, x1=0)
        #       C(a,b) = |2 x 2/4 - 1| = 0,         weight math.inf
        #   for [(1, 0), 0] is similar
        # if LA in [(u, v), 1], {u, v} = {0, 1},
        #   for [(0, 1), 1]  (other is similar)
        #       x1 and x0 & x1 are equal in 3/4 right inputs ([x0, x1]!=[0,1])
        #       C(a,b) = |2 x 3/4 - 1| = 0.5,       weight 1
        # if LA in [(1, 1), 0],
        #   x0 ^ x1 and 0 are equal in 2/4 right inputs (x0==x1)
        #   C(a,b) = |2 x 2/4 - 1| = 0,             weight math.inf
        # if LA in [(1, 1), 1],
        #   x0 ^ x1 and x0 & x1 are equal in 1/4 right inputs ([x0, x1]==[0,0])
        #   C(a,b) = |2 x 1/4 - 1| = 0.5,           weight 1

        # pseudo code
        """
        a0, a1 = [m.val for m in self.input_mask]
        b0 = output_mask.val
        dw = decimal.Decimal(0)
        for i in range(a0.width):
            la = [(int(a0[i]), int(a1[i])), int(b0[i])]
            if la == [(0, 0), 0]:
                dw += 0
            elif la == [(0, 0), 1]:
                dw += 1
            elif la in [[(0, 1), 0], [(1, 0), 0]]:
                raise ValueError(f"linear approximation ((a0, a1), b0)={la} is not valid")
            elif la in [[(0, 1), 1], [(1, 0), 1]]:
                dw += 1
            elif la == [(1, 1), 0]:
                raise ValueError(f"linear approximation ((a0, a1), b0)={la} is not valid")
            elif la == [(1, 1), 1]:
                dw += 1
            else:
                raise ValueError(f"invalid la = {la}")
        """

        a0, a1 = [m.val for m in self.input_mask]
        b0 = output_mask.val
        # bad_case == bad_case_1 | bad_case_2
        bad_case = (~b0) & (a0 | a1)  # [(0, 1), 0], [(1, 0), 0], [(1, 1), 0]
        return operation.BvComp(bad_case, core.Constant(0, a0.width))
        # bad_case_1 = (~b0) & (a0 ^ a1)  # [(0, 1), 0], [(1, 0), 0]
        # bad_case_2 = (~b0) & a0 & a1  # [(1, 1), 0]
        # return operation.BvComp(bad_case_1 | bad_case_2, core.Constant(0, a0.width))

    def pr_one_constraint(self, output_mask):
        """Return the probability-one constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAnd
            >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
            >>> f = LinearModelBvAnd(alpha)
            >>> f.pr_one_constraint(LinearMask(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        # (a0, a1) = (0, 0) -> b0 = (0)
        a0, a1 = [m.val for m in self.input_mask]
        b0 = output_mask.val
        return operation.BvComp(a0 | a1 | b0, core.Constant(0, a0.width))

    def bv_weight(self, output_mask):
        """Return the bit-vector weight for a given output `LinearMask`.

        See also `abstractproperty.opmodel.OpModel.bv_weight`.
        """
        b0 = output_mask.val
        case = b0  # case == case_1 | case_2 | case_3
        # case_1 = b0 & (~a0) & (~a1)  # [(0, 0), 1]
        # case_2 = b0 & (a0 ^ a1)  # [[(0, 1), 1], [(1, 0), 1]]
        # case_3 = b0 & a0 & a1   # [(1, 1), 1]
        return secondaryop.PopCount(case)

    def weight_constraint(self, output_mask, weight_variable):
        """Return the weight constraint for a given output `LinearMask` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAnd
            >>> n = 4
            >>> alpha = LinearMask(Constant(0, n)), LinearMask(Constant(0, n))
            >>> f = LinearModelBvAnd(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(LinearMask(Constant(0, n)), w)
            w == 0b000
            >>> a0, a1, b = Variable("a0", n), Variable("a1", n), Variable("b", n)
            >>> alpha = LinearMask(a0), LinearMask(a1,)
            >>> f = LinearModelBvAnd(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> result = f.weight_constraint(LinearMask(b), w)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            w == PopCount(b)
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), b: Constant(0, n)})
            w == 0b000

        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_mask, weight_variable)

    def max_weight(self):
        n = self.input_mask[0].val.width
        return n

    def weight_width(self):
        n = self.input_mask[0].val.width
        return n.bit_length()

    def decimal_weight(self, output_mask):
        self._assert_preconditions_decimal_weight(output_mask)
        dw = decimal.Decimal(int(self.bv_weight(output_mask)))
        self._assert_postconditions_decimal_weight(output_mask, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class LinearModelBvOr(LinearModelBvAnd):
    """Represent the `linear.opmodel.OpModel` of `BvOr`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvOr
        >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
        >>> f = LinearModelBvOr(alpha)
        >>> print(f.vrepr())
        LinearModelBvOr([LinearMask(Constant(0b0000, width=4)), LinearMask(Constant(0b0000, width=4))])
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (4, 3, 0, 0)

    The linear model of BvOr` is the same as `LinearModelBvAnd`
    since ``~(x & y) == ~x | ~y`` (and `BvNot` preserves masks).
    """
    op = operation.BvOr


class LinearModelBvAdd(OpModel):
    """Represent the `linear.opmodel.OpModel` of `BvAdd`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvAdd
        >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
        >>> f = LinearModelBvAdd(alpha)
        >>> print(f.vrepr())
        LinearModelBvAdd([LinearMask(Constant(0b0000, width=4)), LinearMask(Constant(0b0000, width=4))])
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (3, 2, 0, 0)

    """
    op = operation.BvAdd

    def external_vars_validity_constraint(self, output_mask, z_name=None, explicit=None):
        v, w = [m.val for m in self.input_mask]
        u = output_mask.val
        n = u.width

        if all(isinstance(x_i, core.Constant) for x_i in [u, v, w]) or explicit is True:
            # explicit definition of z (+50% slower than implicit in SMT problems)
            return []

        # implicit definition of z
        # - definition/statement from https://doi.org/10.1007/978-3-319-39555-5_26
        # - following proof from On CCZ-equivalence of Addition mod 2n
        # z = M^t(u ^ v ^ w) (1), where M = R(I ^ R)^(-1) and R = >> 1
        # then (1) is equivalent to
        # z = R(I ^ R)^(-1)(u ^ v ^ w), that is,
        # (I ^ R)z     = R(u ^ v ^ w), that is,
        # z ^ (z >> 1) = (u ^ v ^ w) >> 1
        # (thus, z[n-1] == 0 is not needed)

        if z_name is None:
            h = hashlib.blake2b(digest_size=16)
            h.update((self.vrepr() + output_mask.vrepr()).encode())
            hash_vwu = h.hexdigest()
            z_name = "_tmp" + hash_vwu

        return [core.Variable(z_name, n)]

    def external_vars_weight_constraint(self, output_mask, weight_variable, z_name=None, explicit=None):
        return self.external_vars_validity_constraint(output_mask, z_name=z_name, explicit=explicit)

    def validity_constraint(self, output_mask, z_name=None, explicit=None):
        """Return the validity constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAdd
            >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
            >>> f = LinearModelBvAdd(alpha)
            >>> beta = LinearMask(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 4), Variable("a1", 4), Variable("b", 4)
            >>> alpha = LinearMask(a0), LinearMask(a1)
            >>> f = LinearModelBvAdd(alpha)
            >>> beta = LinearMask(b)
            >>> result = f.validity_constraint(beta, z_name="z")
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            ((~((b ^ a0) | (b ^ a1)) | z) == 0xf) & ((z ^ (z >> 0x1) ^ ((b ^ a0 ^ a1) >> 0x1)) == 0x0)
            >>> result.xreplace({a0: Constant(0, 4), a1: Constant(0, 4), b: Constant(0, 4)})
            (z ^ (z >> 0x1)) == 0x0

        See `OpModel.validity_constraint` for more information.

        Source: https://doi.org/10.1007/978-3-319-39555-5_26
        """
        v, w = [m.val for m in self.input_mask]
        u = output_mask.val
        n = u.width

        if all(isinstance(x_i, core.Constant) for x_i in [u, v, w]) or explicit is True:
            z = [None for _ in range(n)]
            z[n-1] = core.Constant(0, 1)
            for i in range(n-2, -1, -1):
                z[i] = z[i+1] ^ u[i+1] ^ v[i+1] ^ w[i+1]
            z = functools.reduce(operation.Concat, reversed(z))
        else:
            evs = self.external_vars_validity_constraint(
                output_mask, z_name=z_name, explicit=explicit)
            z = evs[0]

        z_eq = operation.BvComp(z ^ (z >> 1) ^ ((u ^ v ^ w) >> 1), core.Constant(0, z.width))

        # z_eq & (z[n-1] == 0b0 is redundant since
        #  z[n-1] = ((z[n-1]) ^ ((z >> 1)[n-1]) ^ ((u ^ v ^ w) >> 1)[n-1])
        # and right hand size is 0 (due to the shifts)
        # (see also proof in external_vars_validity_constraint)

        # <= is equivalent to implication since both have as bad case (1, 0)
        def bitwise_implication(x, y):
            return (~x) | y

        # c1: u^v <= z
        # c2: u^w <= z
        # c1_2: (u^v|u^w) <= z
        lhs = (u ^ v) | (u ^ w)
        rhs = z
        c1_2 = operation.BvComp(bitwise_implication(lhs, rhs), ~core.Constant(0, lhs.width))

        return c1_2 & z_eq

    def pr_one_constraint(self, output_mask):
        """Return the probability-one constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAdd
            >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
            >>> f = LinearModelBvAdd(alpha)
            >>> f.pr_one_constraint(LinearMask(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        v, w = [m.val for m in self.input_mask]
        u = output_mask.val
        n = u.width

        # if z = 0, then (u ^ v ^ w)[:1] = 0; proof:
        #   0 = z[n-2] = u[n-1] ^ v[n-1] ^ w[n-1] = 0
        #   0 = z[n-1] = u[n-2] ^ v[n-2] ^ w[n-2] = 0
        #   0 = z[0]   = u[1] ^ v[1] ^ w[1]
        # moreover, bitwise_implication(x, 0) = ~x

        lhs = (u ^ v) | (u ^ w)
        c1_2 = operation.BvComp(~lhs, ~core.Constant(0, lhs.width))
        return c1_2 & operation.BvComp((u ^ v ^ w)[:1], core.Constant(0, n-1))

    def weight_constraint(self, output_mask, weight_variable, z_name=None, explicit=None):
        """Return the weight constraint for a given output `LinearMask` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAdd
            >>> n = 4
            >>> alpha = LinearMask(Constant(0, n)), LinearMask(Constant(0, n))
            >>> f = LinearModelBvAdd(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(LinearMask(Constant(0, n)), w)
            0b00 == w
            >>> a0, a1, b = Variable("a0", n), Variable("a1", n), Variable("b", n)
            >>> alpha = LinearMask(a0), LinearMask(a1,)
            >>> f = LinearModelBvAdd(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> result = f.weight_constraint(LinearMask(b), w, z_name="z")
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            (PopCount(z[2:]) == w) & ((z ^ (z >> 0x1) ^ ((b ^ a0 ^ a1) >> 0x1)) == 0x0)
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), b: Constant(0, n)})
            (PopCount(z[2:]) == w) & ((z ^ (z >> 0x1)) == 0x0)

        See `OpModel.weight_constraint` for more information.
        """
        # weight variable name not used in z_name to ensure same name for z
        # used in validity_constraint and weight_constraint
        bv_weight, extra_constraints = self._bvweight_and_extra_constraints(
            output_mask, z_name=z_name, explicit=explicit)

        weight_constraint = operation.BvComp(bv_weight, weight_variable)
        for c in extra_constraints:
            weight_constraint &= c

        return weight_constraint

    def _bvweight_and_extra_constraints(self, output_mask, z_name=None, explicit=None):
        v, w = [m.val for m in self.input_mask]
        u = output_mask.val
        n = u.width

        extra_constraints = []
        if all(isinstance(x_i, core.Constant) for x_i in [u, v, w]) or explicit is True:
            z = [None for _ in range(n)]
            z[n-1] = core.Constant(0, 1)
            for i in range(n-2, -1, -1):
                z[i] = z[i+1] ^ u[i+1] ^ v[i+1] ^ w[i+1]
            z = functools.reduce(operation.Concat, reversed(z))
        else:
            evs = self.external_vars_weight_constraint(
                output_mask, weight_variable=None, z_name=z_name, explicit=explicit)
            z = evs[0]

        z_eq = operation.BvComp(z ^ (z >> 1) ^ ((u ^ v ^ w) >> 1), core.Constant(0, z.width))
        if isinstance(z_eq, core.Constant):
            if z_eq == core.Constant(0, 1):
                raise ValueError(f"OpModel {self} with output mask {output_mask} is not valid")
        else:
            extra_constraints.append(z_eq)

        return secondaryop.PopCount(z[n-2:]), extra_constraints

    def max_weight(self):
        width = self.input_mask[0].val.width - 1  # MSB is ignored
        return width  # as an integer

    def weight_width(self):
        n = self.input_mask[0].val.width - 1  # due to n-2
        return n.bit_length()

    def decimal_weight(self, output_mask):
        self._assert_preconditions_decimal_weight(output_mask)
        bv_weight, extra_constraints = self._bvweight_and_extra_constraints(output_mask)
        assert not extra_constraints
        dw = decimal.Decimal(int(bv_weight))
        self._assert_postconditions_decimal_weight(output_mask, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class LinearModelBvSub(LinearModelBvAdd):
    """Represent the `linear.opmodel.OpModel` of `BvSub`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvSub
        >>> alpha = LinearMask(Constant(0, 4)), LinearMask(Constant(0, 4))
        >>> f = LinearModelBvSub(alpha)
        >>> print(f.vrepr())
        LinearModelBvSub([LinearMask(Constant(0b0000, width=4)), LinearMask(Constant(0b0000, width=4))])
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (3, 2, 0, 0)

    The linear model of BvSub` is the same as `LinearModelBvAdd`
    since ``~(x - y) == ~x + y`` (and `BvNot` preserves masks).
    """
    op = operation.BvSub


class LinearModelBvAndCt(PartialOpModel):
    """Represent the `PartialOpModel` of `BvAnd` by a constant.

    This class is not meant to be instantiated but to provide a base
    class for subclasses generated through `make_partial_op_model`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvAndCt, make_partial_op_model
        >>> LinearModelBvAndCt_1 = make_partial_op_model(LinearModelBvAndCt, (None, Constant(1, 4)))
        >>> alpha = LinearMask(Constant(0, 4))
        >>> f = LinearModelBvAndCt_1(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(LinearModelBvAndCt, (None, Constant(0b0001, width=4)))(LinearMask(Constant(0b0000, width=4)))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    base_op = operation.BvAnd
    # op is a set through make_partial_op_model

    @property
    def ct(self):
        """The constant operand."""
        if self.__class__.op.fixed_args[0] is not None:
            constant = self.__class__.op.fixed_args[0]
        else:
            constant = self.__class__.op.fixed_args[1]
        assert constant is not None and operation.BvNot(constant) != 0, f"{constant}"
        return constant

    def validity_constraint(self, output_mask):
        """Return the validity constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAndCt, make_partial_op_model
            >>> LinearModelBvAndCt_1 = make_partial_op_model(LinearModelBvAndCt, (None, Constant(1, 4)))
            >>> alpha = LinearMask(Constant(0, 4))
            >>> f = LinearModelBvAndCt_1(alpha)
            >>> beta = LinearMask(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a, b = Variable("a", 4), Variable("b", 4)
            >>> alpha = LinearMask(a)
            >>> f = LinearModelBvAndCt_1(alpha)
            >>> beta = LinearMask(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            a == (b & 0x1)
            >>> result.xreplace({a: Constant(0, 4), b: Constant(0, 4)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """
        # let f(x) = x & c, and (a,b) n-bit linear approximation
        # <a, x> = a & x, ‹f(x), b> = x & (c & b),
        # if a == c & b
        #   number of x such that <a, x> = ‹f(x), b>? (right inputs)
        #   x & a == x & (c & b) for all x
        #   2^n/2^n right inputs
        #   C(a,b) = |2 x 2^n/2^n=1 - 1| = 1, weight 0
        # if a != c & b
        #   number of x such that <a, x> = ‹f(x), b>? (right inputs)
        #   for 1-bit linear approximation
        #       x & 0 and x & 1 are equal in 1/2 inputs (for x=0)
        #       1/2 right inputs
        #       C(a,b) = |2 x 1/2 - 1| = 0, weight math.inf
        #   for 2-bit linear approximation
        #       x & 00 and x & 10 are equal in 2/4 inputs (x0=0,x1=*)
        #       x & 00 and x & 11 are equal in 2/4 inputs (x0+x1=0 LHS balanced)
        #       x & 10 and x & 01 are equal in 2/4 inputs (x0=x1)
        #       x & 10 and x & 11 are equal in 2/4 inputs (x0=*,x1=0)
        #       2/4 right inputs
        #       C(a,b) = |2 x 2/4 - 1| = 0, weight math.inf
        a = self.input_mask[0].val
        b = output_mask.val
        ct = self.ct
        return operation.BvComp(a, b & ct)

    def pr_one_constraint(self, output_mask):
        """Return the probability-one constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAndCt, make_partial_op_model
            >>> LinearModelBvAndCt_1 = make_partial_op_model(LinearModelBvAndCt, (None, Constant(1, 4)))
            >>> alpha = LinearMask(Constant(0, 4))
            >>> f = LinearModelBvAndCt_1(alpha)
            >>> f.pr_one_constraint(LinearMask(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        return self.validity_constraint(output_mask)

    def bv_weight(self, output_mask):
        """Return the bit-vector weight for a given output `LinearMask`.

        For `LinearModelBvAndCt`, any valid linear approximation has weight 0.
        """
        return core.Constant(0, width=1)

    def weight_constraint(self, output_mask, weight_variable):
        """Return the weight constraint for a given output `LinearMask` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvAndCt, make_partial_op_model
            >>> LinearModelBvAndCt_1 = make_partial_op_model(LinearModelBvAndCt, (None, Constant(1, 4)))
            >>> n = 4
            >>> alpha = LinearMask(Variable("a0", n))
            >>> f = LinearModelBvAndCt_1(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(LinearMask(Variable("b", n)), w)
            w == 0b0

        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_mask, weight_variable)

    def max_weight(self):
        return 0

    def weight_width(self):
        return 1

    def decimal_weight(self, output_mask):
        self._assert_preconditions_decimal_weight(output_mask)
        dw = decimal.Decimal(int(self.bv_weight(output_mask)))
        self._assert_postconditions_decimal_weight(output_mask, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class LinearModelBvOrCt(LinearModelBvAndCt):
    """Represent the `PartialOpModel` of `BvOr` by a constant.

    This class is not meant to be instantiated but to provide a base
    class for subclasses generated through `make_partial_op_model`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvOrCt, make_partial_op_model
        >>> LinearModelBvOrCt_1 = make_partial_op_model(LinearModelBvOrCt, (None, Constant(1, 4)))
        >>> alpha = LinearMask(Constant(0, 4))
        >>> f = LinearModelBvOrCt_1(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(LinearModelBvOrCt, (None, Constant(0b0001, width=4)))(LinearMask(Constant(0b0000, width=4)))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    The partial linear model of BvOr` is similar as `LinearModelBvAndCt`
    since ``~(x & ct) == ~x | ~ct``. Since `BvNot` preserves masks,
    the only difference is that the constant is negated.
    """
    base_op = operation.BvOr

    @property
    def ct(self):
        """The constant operand (negated to reuse `LinearModelBvAndCt`)."""
        if self.__class__.op.fixed_args[0] is not None:
            constant = self.__class__.op.fixed_args[0]
        else:
            constant = self.__class__.op.fixed_args[1]
        assert constant is not None and constant != 0, f"{constant}"
        return ~constant


class LinearModelBvShlCt(PartialOpModel):
    """Represent the `PartialOpModel` of `BvShl` by a constant.

    This class is not meant to be instantiated but to provide a base
    class for subclasses generated through `make_partial_op_model`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvShlCt, make_partial_op_model
        >>> LinearModelBvShlCt_1 = make_partial_op_model(LinearModelBvShlCt, (None, Constant(1, 4)))
        >>> alpha = LinearMask(Constant(0, 4))
        >>> f = LinearModelBvShlCt_1(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(LinearModelBvShlCt, (None, Constant(0b0001, width=4)))(LinearMask(Constant(0b0000, width=4)))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    base_op = operation.BvShl
    # op is a set through make_partial_op_model

    def __init__(self, input_mask):
        if self.__class__.op is None:
            raise ValueError(f"{self.__class__.__name__} need to be given to make_partial_op_model"
                             f" to get the OpModel for the particular fixed operand")
        if self.__class__.op.fixed_args[0] is not None or self.__class__.op.fixed_args[1] is None:
            raise ValueError(f"{self.__class__.__name__} only supports the 2nd operand fixed")

        super().__init__(input_mask)

    @property
    def ct(self):
        """The constant operand."""
        constant = self.__class__.op.fixed_args[1]
        assert constant is not None and constant != 0, f"{constant}"
        return constant

    def validity_constraint(self, output_mask):
        """Return the validity constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvShlCt, make_partial_op_model
            >>> LinearModelBvShlCt_1 = make_partial_op_model(LinearModelBvShlCt, (None, Constant(1, 4)))
            >>> alpha = LinearMask(Constant(0, 4))
            >>> f = LinearModelBvShlCt_1(alpha)
            >>> beta = LinearMask(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a, b = Variable("a", 4), Variable("b", 4)
            >>> alpha = LinearMask(a)
            >>> f = LinearModelBvShlCt_1(alpha)
            >>> beta = LinearMask(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            (b >> 0x1) == a
            >>> result.xreplace({a: Constant(0, 4), b: Constant(0, 4)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """
        # BvShl is a linear operation
        # Examples:
        # 3-bit, (left shift) BvShl by 1,
        #   BvShl(0bx2x1x0) = 0bx1x0_0, thus BvShl([x0, x1, x2]) = [0, x0, x1]
        #   M = [0 0 0], M^T =  [0 1 0]
        #       [1 0 0],        [0 0 1]
        #       [0 1 0]         [0 0 0]
        #   (a0,a1,a2) = M^T (b0,b1,b2) = (b1, b2, 0) = 0b_0b2b1 = BvLshr_1(0bb2b1b0) = ([b0,b1,b2])
        # 3-bit, (left shift) BvShl by 2,
        #   BvShl(0bx2x1x0) = 0bx0_0_0, thus BvShl([x0, x1, x2]) = [0, 0, x0]
        #   M = [0 0 0], M^T =  [0 0 1]
        #       [0 0 0],        [0 0 0]
        #       [1 0 0]         [0 0 0]
        #   (a0,a1,a2) = M^T (b0,b1,b2) = (b2, 0, 0) = 0b_0_0b2 = BvLshr_2(0bb2b1b0) = ([b0,b1,b2])
        a = self.input_mask[0].val
        b = output_mask.val
        ct = self.ct
        return operation.BvComp(operation.BvLshr(b, ct), a)

    def pr_one_constraint(self, output_mask):
        """Return the probability-one constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvShlCt, make_partial_op_model
            >>> LinearModelBvShlCt_1 = make_partial_op_model(LinearModelBvShlCt, (None, Constant(1, 4)))
            >>> alpha = LinearMask(Constant(0, 4))
            >>> f = LinearModelBvShlCt_1(alpha)
            >>> f.pr_one_constraint(LinearMask(Constant(0, 4)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        return self.validity_constraint(output_mask)

    def bv_weight(self, output_mask):
        """Return the bit-vector weight for a given output `LinearMask`.

        For `LinearModelBvShlCt`, any valid linear approximation has weight 0.
        """
        return core.Constant(0, width=1)

    def weight_constraint(self, output_mask, weight_variable):
        """Return the weight constraint for a given output `LinearMask` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvShlCt, make_partial_op_model
            >>> LinearModelBvShlCt_1 = make_partial_op_model(LinearModelBvShlCt, (None, Constant(1, 4)))
            >>> n = 4
            >>> alpha = LinearMask(Variable("a0", n))
            >>> f = LinearModelBvShlCt_1(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(LinearMask(Variable("b", n)), w)
            w == 0b0

        See `OpModel.weight_constraint` for more information.
        """
        return super().weight_constraint(output_mask, weight_variable)

    def max_weight(self):
        return 0

    def weight_width(self):
        return 1

    def decimal_weight(self, output_mask):
        self._assert_preconditions_decimal_weight(output_mask)
        dw = decimal.Decimal(int(self.bv_weight(output_mask)))
        self._assert_postconditions_decimal_weight(output_mask, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class LinearModelBvLshrCt(LinearModelBvShlCt):
    """Represent the `PartialOpModel` of `BvLshr` by a constant.

    This class is not meant to be instantiated but to provide a base
    class for subclasses generated through `make_partial_op_model`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelBvLshrCt, make_partial_op_model
        >>> LinearModelBvLshrCt_1 = make_partial_op_model(LinearModelBvLshrCt, (None, Constant(1, 4)))
        >>> alpha = LinearMask(Constant(0, 4))
        >>> f = LinearModelBvLshrCt_1(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(LinearModelBvLshrCt, (None, Constant(0b0001, width=4)))(LinearMask(Constant(0b0000, width=4)))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    base_op = operation.BvLshr

    def validity_constraint(self, output_mask):
        """Return the validity constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvLshrCt, make_partial_op_model
            >>> LinearModelBvLshrCt_1 = make_partial_op_model(LinearModelBvLshrCt, (None, Constant(1, 4)))
            >>> alpha = LinearMask(Constant(0, 4))
            >>> f = LinearModelBvLshrCt_1(alpha)
            >>> beta = LinearMask(Constant(0, 4))
            >>> f.validity_constraint(beta)
            0b1
            >>> a, b = Variable("a", 4), Variable("b", 4)
            >>> alpha = LinearMask(a)
            >>> f = LinearModelBvLshrCt_1(alpha)
            >>> beta = LinearMask(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            (b << 0x1) == a
            >>> result.xreplace({a: Constant(0, 4), b: Constant(0, 4)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """
        a = self.input_mask[0].val
        b = output_mask.val
        ct = self.ct
        return operation.BvComp(operation.BvShl(b, ct), a)


class LinearModelExtractCt(PartialOpModel):
    """Represent the `PartialOpModel` of `Extract` with fixed indices.

    This class is not meant to be instantiated but to provide a base
    class for subclasses generated through `make_partial_op_model`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import LinearModelExtractCt, make_partial_op_model
        >>> LinearModelExtractCt_21 = make_partial_op_model(LinearModelExtractCt, (None, 2, 1))
        >>> alpha = LinearMask(Constant(0, 4))
        >>> f = LinearModelExtractCt_21(alpha)
        >>> print(f.vrepr())
        make_partial_op_model(LinearModelExtractCt, (None, 2, 1))(LinearMask(Constant(0b0000, width=4)))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (0, 1, 0, 0)

    """
    base_op = operation.Extract
    # op is a set through make_partial_op_model

    def __init__(self, input_mask):
        if self.__class__.op is None:
            raise ValueError(f"{self.__class__.__name__} need to be given to make_partial_op_model"
                             f" to get the OpModel for the particular fixed operand")
        i, j = self.__class__.op.fixed_args[1], self.__class__.op.fixed_args[2]
        if self.__class__.op.fixed_args[0] is not None or i is None or j is None:
            raise ValueError(f"{self.__class__.__name__} only supports the 2nd and 3rd operands fixed")

        super().__init__(input_mask)

    def validity_constraint(self, output_mask):
        """Return the validity constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelExtractCt, make_partial_op_model
            >>> LinearModelExtractCt_21 = make_partial_op_model(LinearModelExtractCt, (None, 2, 1))
            >>> alpha = LinearMask(Constant(0, 4))
            >>> f = LinearModelExtractCt_21(alpha)
            >>> beta = LinearMask(Constant(0, 2))
            >>> f.validity_constraint(beta)
            0b1
            >>> a, b = Variable("a", 4), Variable("b", 2)
            >>> alpha = LinearMask(a)
            >>> f = LinearModelExtractCt_21(alpha)
            >>> beta = LinearMask(b)
            >>> result = f.validity_constraint(beta)
            >>> result
            a == (0b0 :: b :: 0b0)
            >>> result.xreplace({a: Constant(0, 4), b: Constant(0, 2)})
            0b1

        See `OpModel.validity_constraint` for more information.
        """
        a = self.input_mask[0].val
        b = output_mask.val
        n = a.width
        i, j = self.op.fixed_args[1], self.op.fixed_args[2]

        assert output_mask.val.width == i - j + 1

        # 3-bit, Extract 2nd bit
        #   M = [0 1 0], (a0,a1,a2) = M^T b0 = (0,b0,0)

        m_t_b = b
        if i < n - 1:
            m_t_b = operation.Concat(core.Constant(0, n - 1 - i), m_t_b)
        if j > 0:
            m_t_b = operation.Concat(m_t_b, core.Constant(0, j))

        return operation.BvComp(a, m_t_b)

    def pr_one_constraint(self, output_mask):
        """Return the probability-one constraint for a given output `LinearMask`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelExtractCt, make_partial_op_model
            >>> LinearModelExtractCt_21 = make_partial_op_model(LinearModelExtractCt, (None, 2, 1))
            >>> alpha = LinearMask(Constant(0, 4))
            >>> f = LinearModelExtractCt_21(alpha)
            >>> f.pr_one_constraint(LinearMask(Constant(0, 2)))
            0b1

        See `abstractproperty.opmodel.OpModel.pr_one_constraint` for more information.
        """
        return self.validity_constraint(output_mask)

    def bv_weight(self, output_mask):
        """Return the bit-vector weight for a given output `LinearMask`.

        For `LinearModelBvShlCt`, any valid linear approximation has weight 0.
        """
        assert output_mask.val.width == self.op.fixed_args[1] - self.op.fixed_args[2] + 1
        return core.Constant(0, width=1)

    def weight_constraint(self, output_mask, weight_variable):
        """Return the weight constraint for a given output `LinearMask` and weight `Variable`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.opmodel import LinearModelBvShlCt, make_partial_op_model
            >>> LinearModelExtractCt_21 = make_partial_op_model(LinearModelExtractCt, (None, 2, 1))
            >>> n = 4
            >>> alpha = LinearMask(Variable("a0", n))
            >>> f = LinearModelExtractCt_21(alpha)
            >>> w = Variable("w", f.weight_width())
            >>> f.weight_constraint(LinearMask(Variable("b", 2)), w)
            w == 0b0

        See `OpModel.weight_constraint` for more information.
        """
        assert output_mask.val.width == self.op.fixed_args[1] - self.op.fixed_args[2] + 1
        return super().weight_constraint(output_mask, weight_variable)

    def max_weight(self):
        return 0

    def weight_width(self):
        return 1

    def decimal_weight(self, output_mask):
        assert output_mask.val.width == self.op.fixed_args[1] - self.op.fixed_args[2] + 1
        self._assert_preconditions_decimal_weight(output_mask)
        dw = decimal.Decimal(int(self.bv_weight(output_mask)))
        self._assert_postconditions_decimal_weight(output_mask, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


# --------------------------


@functools.lru_cache(maxsize=None)  # temporary hack to create singletons
def get_weak_model(op, nonzero2nonzero_weight, zero2nonzero_weight=math.inf, precision=0):
    """Return the weak model of the given bit-vector operation ``op``.

    Given the `SecondaryOperation` ``op``, return the
    `WeakModel` of ``op`` with given class attributes
    ``nonzero2nonzero_weight``, ``zero2nonzero_weight`` and  ``precision``
    (see `WeakModel`).
    The attribute ``nonzero2zero_weight`` is set to ``math.inf``,
    and the attribute ``zero2zero_weight`` is set to 0.

    The returned model is a subclass of `WeakModel` and `OpModel`.

    .. note::
        To link the returned model ``MyModel`` to ``op``
        such that ``MyModel`` is used in ``propagate``,
        set the ``linear_model`` attribute of ``op``
        to ``MyModel`` (e.g., ``op.linear_model = MyModel``).
        See also  `linear.mask.LinearMask.propagate`.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import LutOperation
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import get_weak_model
        >>> class MyLut(LutOperation): pass  # a 2-bit function
        >>> LinearWeakModelMyLut = get_weak_model(MyLut, 1, zero2nonzero_weight=2)
        >>> alpha, beta = LinearMask(Variable("a", 2)), LinearMask(Variable("b", 2))
        >>> f = LinearWeakModelMyLut(alpha)
        >>> print(f.vrepr())
        LinearWeakModelMyLut(LinearMask(Variable('a', width=2)))
        >>> f.validity_constraint(beta)
        (((a == 0b00) & (b == 0b00)) == 0b1) | (((a == 0b00) & ~(b == 0b00)) == 0b1) | ((~(a == 0b00) & ~(b == 0b00)) == 0b1)
        >>> f.bv_weight(beta)
        Ite(((a == 0b00) & (b == 0b00)) == 0b1, 0b00, Ite(((a == 0b00) & ~(b == 0b00)) == 0b1, 0b10, 0b01))
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (2, 2, 0, 0)

    """
    assert issubclass(op, operation.SecondaryOperation)

    _op = op
    _nonzero2nonzero_weight = nonzero2nonzero_weight
    _zero2nonzero_weight = zero2nonzero_weight
    _precision = precision

    class MyWeakModel(abstractproperty.opmodel.WeakModel, OpModel):
        op = _op
        zero2zero_weight = 0
        nonzero2nonzero_weight = _nonzero2nonzero_weight
        zero2nonzero_weight = _zero2nonzero_weight
        nonzero2zero_weight = math.inf  # for Pr. 1 hulls, an output propagates back to a unique input
        precision = _precision

        # def error(self):  # maximum weight of a hull with n-bit input is n - 1
        #     return sum(p.val.width for p in self.input_prop) - 1

    MyWeakModel.__name__ = f"Linear{abstractproperty.opmodel.WeakModel.__name__}{op.__name__}"
    return MyWeakModel


@functools.lru_cache(maxsize=None)
def get_branch_number_model(op, output_widths, branch_number, nonzero2nonzero_weight,
                            zero2nonzero_weight=math.inf, precision=0):
    """Return the branch-number model of the given bit-vector operation ``op``.

    Given the `SecondaryOperation` ``op``, return the
    `BranchNumberModel` of ``op`` with given class attributes
    ``output_widths`` (given as a `tuple`),
    `branch_number`, ``nonzero2nonzero_weight``,
    ``zero2nonzero_weight`` and  ``precision`` (see `BranchNumberModel`).
    The attribute ``nonzero2zero_weight`` is set to ``math.inf``,
    and the attribute ``zero2zero_weight`` is set to 0.

    The returned model is a subclass of `BranchNumberModel` and `OpModel`.

    .. note::
        To link the returned model ``MyModel`` to ``op``
        such that ``MyModel`` is used in ``propagate``,
        set the ``linear_model`` attribute of ``op``
        to ``MyModel`` (e.g., ``op.linear_model = MyModel``).
        See also  `linear.mask.LinearMask.propagate`.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import MatrixOperation
        >>> from cascada.bitvector.printing import BvWrapPrinter
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import get_branch_number_model
        >>> class MyMatrix(MatrixOperation): pass  # a (2,3) binary matrix
        >>> LinearBranchNumberModelMyMatrix = get_branch_number_model(
        ...     MyMatrix, (1, 1), 2, decimal.Decimal(1.5), precision=2)
        >>> alpha, beta = LinearMask(Variable("a", 3)), LinearMask(Variable("b", 2))
        >>> f = LinearBranchNumberModelMyMatrix(alpha)
        >>> print(f.vrepr())
        LinearBranchNumberModelMyMatrix(LinearMask(Variable('a', width=3)))
        >>> print(BvWrapPrinter().doprint(f.validity_constraint(beta)))
        BvOr(((a == 0b000) & (b == 0b00)) == 0b1,
             BvAnd((~(a == 0b000) & ~(b == 0b00)) == 0b1,
                   ((0b0 :: ~(a == 0b000)) + (0b0 :: (b[0])) + (0b0 :: (b[1]))) >= 0b10
             )
        )
        >>> f.bv_weight(beta)
        Ite(((a == 0b000) & (b == 0b00)) == 0b1, 0b000, 0b110)
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (6, 3, 0, 2)

    """
    assert issubclass(op, operation.SecondaryOperation)

    _op = op
    _branch_number, _output_widths = branch_number, output_widths
    _nonzero2nonzero_weight = nonzero2nonzero_weight
    _zero2nonzero_weight = zero2nonzero_weight
    _precision = precision

    class MyBranchNumberModel(abstractproperty.opmodel.BranchNumberModel, OpModel):
        op = _op
        branch_number, output_widths = _branch_number, _output_widths
        zero2zero_weight = 0
        nonzero2nonzero_weight = _nonzero2nonzero_weight
        zero2nonzero_weight = _zero2nonzero_weight
        nonzero2zero_weight = math.inf  # for Pr. 1 hulls, an output propagates back to a unique input
        precision = _precision

    MyBranchNumberModel.__name__ = f"Linear{abstractproperty.opmodel.BranchNumberModel.__name__}{op.__name__}"
    return MyBranchNumberModel


@functools.lru_cache(maxsize=None)
def get_wdt(op, input_width, output_width):
    """Return the weight distribution table of the given bit-vector operation ``op`` for WDT-based model.

    Given the `Operation` ``op``, return the weight distribution table
    as a `tuple` of `tuple` of ``op`` from Linear Approximation Table (LAT)
    of correlation weights with given ``input_width`` and ``input_width``.

    The returned table is a `tuple` of `tuple`.

    .. note::
        To link the returned model ``MyModel`` to ``op``
        such that ``MyModel`` is used in ``propagate``,
        set the ``linear_model`` attribute of ``op``
        to ``MyModel`` (e.g., ``op.linear_model = MyModel``).
        See also  `linear.mask.LinearMask.propagate`.

    ::

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.secondaryop import LutOperation
        >>> from cascada.linear.opmodel import get_wdt, get_wdt_model
        >>> class SboxLut(LutOperation):
        >>>     lut = [Constant(x, 4) for x in (6, 5, 12, 10, 1, 14, 7, 9, 11, 0, 3, 13, 8, 15, 4, 2)]
        >>> SboxLut.linear_model = get_wdt_model(SboxLut, get_wdt(SboxLut, 4, 4))

    """
    assert issubclass(op, operation.Operation)

    rows = 2 ** input_width
    cols = 2 ** output_width
    lat = []

    for r in range(rows):
        row = []
        for c in range(cols):
            count = 0
            for i in range(rows):
                input_hw = int(secondaryop.PopCount(core.Constant(i & r, input_width)))
                output_hw = int(secondaryop.PopCount(op.eval(core.Constant(i, input_width)) & core.Constant(c, output_width)))
                count += 1 - ((input_hw - output_hw) & 1)

            row.append((count * 2 - rows) / rows)
        lat.append(row)

    return tuple([tuple(math.inf if x == 0 else -log2_decimal(decimal.Decimal(abs(x))) for x in row) for row in lat])


@functools.lru_cache(maxsize=None)
def get_wdt_model(op, weight_distribution_table, loop_rows_then_columns=False, precision=0):
    """Return the WDT-based model of the given bit-vector operation ``op``.

    Given the `Operation` ``op``, return the `WDTModel`
    of ``op`` with given class attributes ``weight_distribution_table``
    (i.e., the Linear Approximation Table (LAT) given as a `tuple` of `tuple`
    of correlation weights), ``loop_rows_then_columns`` and  ``precision``
    (see `WDTModel`).

    The returned model is a subclass of `WDTModel` and `OpModel`.

    .. note::
        To link the returned model ``MyModel`` to ``op``
        such that ``MyModel`` is used in ``propagate``,
        set the ``linear_model`` attribute of ``op``
        to ``MyModel`` (e.g., ``op.linear_model = MyModel``).
        See also  `linear.mask.LinearMask.propagate`.

    ::

        >>> from decimal import Decimal
        >>> from math import inf
        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import LutOperation
        >>> from cascada.bitvector.printing import BvWrapPrinter
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.opmodel import get_wdt_model
        >>> from cascada.abstractproperty.opmodel import log2_decimal
        >>> # 3-bit permutation with 4 different weights
        >>> class Inversion3b(LutOperation): lut = [Constant(i, 3) for i in (0, 1, 2, 4, 3, 6, 7, 5)]
        >>> lat = [(1, 0, 0, 0, 0, 0, 0, 0), (0, 0, -1/2, 1/2, 1/2, 1/2, 0, 0), (0, 0, 0, 0, 1/2, -1/2, 1/2, 1/2), (0, 0, 1/2, 1/2, 0, 0, -1/2, 1/2),
        ... (0, 1/2, 1/2, 0, 1/2, 0, 0, -1/2), (0, 1/2, 0, -1/2, 0, 1/2, 0, 1/2), (0, -1/2, 1/2, 0, 0, 1/2, 1/2, 0), (0, 1/2, 0, 1/2, -1/2, 0, 1/2, 0)]
        >>> wdt = tuple([tuple(inf if x == 0 else -log2_decimal(Decimal(abs(x))) for x in row) for row in lat])
        >>> LinearWDTModelInversion3b = get_wdt_model(Inversion3b, wdt)
        >>> alpha, beta = LinearMask(Variable("a", 3)), LinearMask(Variable("b", 3))
        >>> f = LinearWDTModelInversion3b(alpha)
        >>> print(f.vrepr())
        LinearWDTModelInversion3b(LinearMask(Variable('a', width=3)))
        >>> BvWrapPrinter.new_line_right_parenthesis = False
        >>> print(BvWrapPrinter().doprint(f.validity_constraint(beta)))
        Ite(b == 0b110,
            (a == 0b010) | (a == 0b011) | (a == 0b110) | (a == 0b111),
            Ite(b == 0b101,
                (a == 0b001) | (a == 0b010) | (a == 0b101) | (a == 0b110),
                Ite(b == 0b100,
                    (a == 0b001) | (a == 0b010) | (a == 0b100) | (a == 0b111),
                    Ite(b == 0b011,
                        (a == 0b001) | (a == 0b011) | (a == 0b101) | (a == 0b111),
                        Ite(b == 0b010,
                            (a == 0b001) | (a == 0b011) | (a == 0b100) | (a == 0b110),
                            Ite(b == 0b001,
                                (a == 0b100) | (a == 0b101) | (a == 0b110) | (a == 0b111),
                                Ite(b == 0b000,
                                    a == 0b000,
                                    (a == 0b010) | (a == 0b011) | (a == 0b100) | (a == 0b101))))))))
        >>> f.pr_one_constraint(beta)
        Ite(b == 0b000, a == 0b000, 0b0)
        >>> f.bv_weight(beta)
        Ite(b == 0b000, 0b0, 0b1)
        >>> BvWrapPrinter.new_line_right_parenthesis = True
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (1, 1, 0, 0)

    """
    assert issubclass(op, operation.Operation)

    num_rows = len(weight_distribution_table)
    num_columns = len(weight_distribution_table[0])
    for j in range(num_columns):
        if all(weight_distribution_table[i][j] == math.inf for i in range(num_rows)):
            raise ValueError("weight_distribution_table contains a column with only math.inf values")
    if weight_distribution_table[0][0] != 0:
        raise ValueError("weight_distribution_table[0][0] != 0")

    _op = op
    _weight_distribution_table = weight_distribution_table[:]
    _loop_rows_then_columns = loop_rows_then_columns
    _precision = precision

    class MyWDTModel(abstractproperty.opmodel.WDTModel, OpModel):
        op = _op
        weight_distribution_table = _weight_distribution_table
        loop_rows_then_columns = _loop_rows_then_columns
        precision = _precision

    MyWDTModel.__name__ = f"Linear{abstractproperty.opmodel.WDTModel.__name__}{op.__name__}"
    return MyWDTModel
