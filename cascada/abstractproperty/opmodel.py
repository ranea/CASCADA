"""Manipulate property models of bit-vector operations (w.r.t an abstract property).

.. autosummary::
   :nosignatures:

    InvalidOpModelError
    OpModel
    PartialOpModel
    make_partial_op_model
    ModelIdentity
    WeakModel
    BranchNumberModel
    WDTModel
"""
import collections
import functools
import decimal
import math
import warnings

from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import context

from cascada import abstractproperty


def _tuplify(seq):
    if isinstance(seq, collections.abc.Sequence):
        return tuple(seq)
    else:
        return tuple([seq])


def log2_decimal(my_decimal):
    first_try = math.log2(my_decimal)
    if first_try.is_integer():
        return decimal.Decimal(first_try)
    else:
        return my_decimal.ln() / decimal.Decimal(2).ln()


class InvalidOpModelError(Exception):
    """Raised when the OpModel is not valid and a valid one was expected."""
    pass


class OpModel(object):
    """Represent property models of bit-vector operations.

    A (bit-vector) property model of a bit-vector `Operation` :math:`f` for
    a particular `Property` is a set of bit-vector constraints that
    models the propagation probability of `Property` over :math:`f`.
    See `Property`.

    .. note::

        For the `Value` and `Difference` `Property` types,
        the propagation probability satisfies that for any
        fixed :math:`\\alpha`, the sum of the probabilities of :math:`\\alpha`
        propagating to :math:`\\beta` (for all :math:`\\beta`) is equal to 1.
        In particular, if the propagation probability of
        :math:`(\\alpha, \\beta)` is 1, then :math:`\\alpha`
        uniquely propagates to :math:`\\beta`.

        For the `LinearMask` property this is different
        (see `linear.opmodel.OpModel`).

    A model of :math:`f` is mainly given by three bit-vector formulas or
    constraints: the `validity_constraint` constraint, the `weight_constraint`
    and the `pr_one_constraint`:

    * The validity constraint (with inputs :math:`(\\alpha, \\beta)`) is True
      if and only if the propagation probability of :math:`(\\alpha, \\beta)`
      is non-zero, that is, :math:`P(\\alpha, \\beta) \\neq 0`.
    * The weight constraint (with inputs :math:`(w, \\alpha, \\beta)` is True
      if and only if the bit-vector :math:`w` is equals to the negative binary
      logarithm (weight) of the propagation probability of :math:`(\\alpha, \\beta)`,
      that is, :math:`w = -log_2(P(\\alpha, \\beta)))`.
    * The probability-one constraint  (with inputs :math:`(\\alpha, \\beta)`)
      is True if and only if the propagation probability of
      :math:`(\\alpha, \\beta)` is 1.

    .. note::
        The weight constraint is only defined for inputs :math:`(w, \\alpha, \\beta)`
        where the propagation probability of :math:`(\\alpha, \\beta)` is non-zero.

        By default, the :math:`n_w`-bit input `Variable` :math:`w`
        of the weight constraint is interpreted as the non-negative integer
        :math:`w[0] + 2 w[1] + \dots + 2^{n_w−1} w[n_w − 1]`.
        However, since the propagation weight can be a non-integer value
        for some properties and functions, :math:`w` can also be interpreted
        as the rational value :math:`2^{−l}(w[0] + 2w[1] + \dots + 2^{n_w−1} w[n_w − 1]`
        for a given fixed number :math:`l` (`num_frac_bits`) of fractional bits.

        Moreover, weight constraints can also be True if
        :math:`w` is equals to weight of the propagation probability up to
        some error bounded by `error`.

    An `OpModel` is defined for a type of `Property` and a bit-vector
    `Operation` :math:`f` . The input property :math:`\\alpha` is used
    when initializing the `OpModel` object, and the output property
    :math:`\\beta` is given in the arguments of the methods
    `OpModel.validity_constraint`, `OpModel.weight_constraint`
    and `OpModel.pr_one_constraint` (among others).

    In some cases, the validity, weight or probability-one constraints
    use auxiliary external (different from the input and output
    properties) variables in the bit-vector expression.
    These external variables can be obtained from
    `external_vars_validity_constraint`,
    `external_vars_weight_constraint` and
    `external_vars_pr_one_constraint` respectively.

    .. note::  `OpModel` of functions with scalar operands are not supported.

    This class is not meant to be instantiated but to provide a base
    class for creating models of some operation w.r.t some property,
    (see `XorModelBvAdd`, `LinearModelBvAdd` or `BitModelBvAdd`
    for some examples).

    .. note::
        It is highly recommended that new models of bit-vector operations
        are tested with ``TestOpModelGeneric``
        (see ``differential/tests/test_opmodel.py`` or
        ``linear/tests/test_opmodel.py`` for some examples).

    Attributes:
        prop_type: the particular `Property` type
        op: the bit-vector `Operation` :math:`f`
        input_prop: a list containing the `Property` associated to
            each bit-vector operand.
    """
    prop_type = None
    op = None

    def __init__(self, input_prop):  # noqa: 102
        input_prop = _tuplify(input_prop)

        if isinstance(self, PartialOpModel) and getattr(self, "op", None) is None:
            raise ValueError(f"{self.__class__.__name__} need to be given to make_partial_op_model"
                             f" to get the OpModel for the particular fixed operand")

        assert self.__class__.op.arity[1] == 0  # no scalars
        if len(input_prop) != sum(self.__class__.op.arity):
            raise ValueError(f"{len(input_prop)} inputs given to {self.__class__.__name__} but "
                             f"arity of {self.__class__.op.__name__} is {self.__class__.op.arity}")
        for p in input_prop:
            if not isinstance(p, self.__class__.prop_type):
                raise ValueError(f"input {p} of {self.__class__.__name__} is not a "
                                 f"{self.__class__.prop_type.__name__} object")

        self.input_prop = input_prop

    def __str__(self):
        input_prop = self.input_prop[0] if len(self.input_prop) == 1 else list(self.input_prop)
        return f"{self.__class__.__name__}({input_prop})"

    __repr__ = __str__

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that ``eval(self.vrepr())``
        and ``self`` have the same content.
        """
        if len(self.input_prop) == 1:
            id_vrepr = self.input_prop[0].vrepr()
        else:
            id_vrepr = f"[{', '.join([d.vrepr() for d in self.input_prop])}]"
        return f"{self.__class__.__name__}({id_vrepr})"

    def validity_constraint(self, output_prop):
        """Return the validity constraint for a given output `Property`.

        The validity constraint is a bit-vector expression that depends on
        the input property `OpModel.input_prop` and the output property
        ``output_prop`` and it is True if and only if the input property
        propagates to the output property with non-zero probability.

        If both the input and output properties are constant values,
        this method returns the `Constant` ``0b1`` or ``0b0`` depending on
        whether the corresponding propagation has non-zero probability.
        Otherwise, this method returns a bit-vector `Term`
        containing the input and output properties as `Variable` objects.
        """
        raise NotImplementedError("subclasses need to override this method")

    def pr_one_constraint(self, output_prop):
        """Return the probability-one constraint for a given output `Property`.

        Return the constraint that evaluates to True if the input `Property`
        propagates to the given output `Property` with probability one.

        This method returns a bit-vector expression that depends on
        the input property `OpModel.input_prop` and the output property
        ``output_prop`` and it is True if and only if the input property
        propagates to the output property with probability 1.

        If both the input and output properties are constant values,
        this method returns the `Constant` ``0b1`` or ``0b0`` depending on
        whether the corresponding propagation has probability 1.
        Otherwise, this method returns a bit-vector `Term`
        containing the input and output properties as `Variable` objects.
        """
        raise NotImplementedError("subclasses need to override this method")

    def weight_constraint(self, output_prop, weight_variable):
        """Return the weight constraint for a given output `Property` and weight `Variable`.

        The weight constraint is a bit-vector expression that depends on
        the input property `OpModel.input_prop`, the output property
        ``output_prop`` and the weight variable ``weight_variable``.
        This expression is True if and only if the weight
        (negative binary logarithm of the probability) of the input property
        propagating to the output property is equals to ``weight_variable``.

        .. note::
            It is assumed that the corresponding propagation has non-zero probability.

        If the input and output properties and the weight variable are constant values,
        this method returns the `Constant` ``0b1`` or ``0b0`` depending on
        whether the corresponding propagation has the given weight.
        Otherwise, this method returns a bit-vector `Term`
        containing the input and output properties and the weight as `Variable` objects.

        Subclasses can either reimplement this method or implement the method ``bv_weight``.
        In the last case, the weight constraint is defined as
        ``BvComp(weight_variable, self.bv_weight(output_prop))``.
        """
        if weight_variable.width != self.weight_width():
            raise ValueError("invalid weight_variable width")
        return operation.BvComp(weight_variable, self.bv_weight(output_prop))

    def bv_weight(self, output_prop):
        """Return the bit-vector weight for a given output `Property`.

        The bit-vector weight is a bit-vector expression that depends on
        the input property `OpModel.input_prop` and the output property
        ``output_prop``. This bit-vector function represents an approximation
        of the negative binary logarithm (weight) of the probability of
        the input property propagating to the output property.

        .. note::
            This method is optional and subclasses of `OpModel` don't need to implement it
            (as opposed to `OpModel.weight_constraint`).

            It is assumed that the corresponding propagation has non-zero probability.

        If the input and output properties are constant values,
        this method returns a `Constant` of bit-size `weight_width`
        denoting the weight of the valid propagation.
        Otherwise, this method returns a bit-vector `Term`
        containing the input and output properties as `Variable` objects.
        """
        raise NotImplementedError("subclasses need to override this method")

    def max_weight(self):
        """Return the maximum value the weight variable can achieve in `OpModel.weight_constraint`."""
        raise NotImplementedError("subclasses need to override this method")

    def weight_width(self):
        """Return the width of the weight variable used `OpModel.weight_constraint`."""
        raise NotImplementedError("subclasses need to override this method")

    def decimal_weight(self, output_prop):
        """Return the `decimal.Decimal` weight for a given constant output `Property`.

        This method returns, as a decimal number, the weight (negative binary logarithm)
        of the probability of the input property propagating to the output property.

        This method only works when the input property and the output property are
        constant values, but provides a better approximation than the bit-vector weight
        from `OpModel.weight_constraint`.
        """
        raise NotImplementedError("subclasses need to override this method")

    def _assert_preconditions_decimal_weight(self, output_prop):
        if any(not isinstance(d.val, core.Constant) for d in self.input_prop + (output_prop, )):
            raise ValueError(f"decimal_weight requires constant input and output properties | "
                             f"{self}(output_prop={output_prop})")
        if self.validity_constraint(output_prop) == core.Constant(0, 1):
            raise InvalidOpModelError(f"decimal_weight requires a valid propagation | "
                                      f"{self}(output_prop={output_prop})")

    def _assert_postconditions_decimal_weight(self, output_prop, decimal_weight):
        assert isinstance(decimal_weight, decimal.Decimal)

        try:
            bv_weight_with_frac_bits = self.bv_weight(output_prop)
        except NotImplementedError:
            return

        bv_weight_with_frac_bits = decimal.Decimal(int(bv_weight_with_frac_bits))
        bv_weight = bv_weight_with_frac_bits / decimal.Decimal(2 ** (self.num_frac_bits()))

        abs_error = (decimal_weight - bv_weight).copy_abs()
        max_abs_error = self.error()

        if abs_error > max_abs_error:
            raise ValueError(f"absolute error between decimal_weight={decimal_weight} "
                             f"and bv_weight={bv_weight} for op_model={self} and "
                             f"output_prop={output_prop} is {abs_error}, which is greater "
                             f"than maximum absolute error given by error()={max_abs_error}")

    def num_frac_bits(self):
        """Return the number of fractional bits used in the weight variable of `OpModel.weight_constraint`.

        If the number of fractional bits is ``k``, then the bit-vector weight variable
        ``w`` of `OpModel.weight_constraint` represents the number ``2^{-k} * bv2int(w)``.
        In particular, if ``k == 0``, then ``w`` represents an integer number.
        Otherwise, the ``k`` least significant bits of ``w`` denote the
        fractional part of the number represented by ``w``.
        """
        raise NotImplementedError("subclasses need to override this method")

    def error(self):
        """Return the maximum difference between `OpModel.weight_constraint` and the exact weight.

        The exact weight is exact value (without error) of the negative binary
        logarithm (weight) of the propagation probability of :math:`(\\alpha, \\beta)`.

        .. note::

            The exact weight can be computed in ``TestOpModelGeneric.get_empirical_weight_slow``.

        This method returns an upper bound (in absolute value) of the maximum difference
        (over all input and output properties) between the bit-vector weight
        from `OpModel.weight_constraint` and the exact weight.

        Note that the exact weight might still differ from `decimal_weight`.
        """
        raise NotImplementedError("subclasses need to override this method")

    def external_vars_validity_constraint(self, output_prop):
        """Return the list of external variables of `validity_constraint`
        for a given output `Property` (an empty list by default)."""
        return []

    def external_vars_pr_one_constraint(self, output_prop):
        """Return the list of external variables of `pr_one_constraint`
        for a given output `Property` (an empty list by default)."""
        return []

    def external_vars_weight_constraint(self, output_prop, weight_variable):
        """Return the list of external variables of `weight_constraint`
        for a given output `Property` and weight `Variable` (an empty list by default)."""
        return []


class PartialOpModel(object):
    """Represent property models of operations with fixed operands.

    This class is not meant to be instantiated but to provide a base
    class for property models of operations with fixed operands generated
    through `make_partial_op_model`.

    Attributes:
        base_op: a subclass of `Operation` denoting the base operator.

    .. Implementation details:

        This class does not subclass `abstractproperty.opmodel.OpModel`,
        but subclasses of this class defined for a particular `Property`
        must subclass the corresponding ``OpModel``.

    """

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that ``eval(self.vrepr())``
        and ``self`` have the same content.
        """
        fa_vrepr = ', '.join([str(a) if not isinstance(a, core.Term) else a.vrepr()
                              for a in self.__class__.op.fixed_args])
        fa_vrepr = f"({fa_vrepr})"

        if len(self.input_prop) == 1:
            id_vrepr = self.input_prop[0].vrepr()
        else:
            id_vrepr = f"[{', '.join([d.vrepr() for d in self.input_prop])}]"

        return "{}({}, {})({})".format(
            make_partial_op_model.__name__,
            self.__class__.__bases__[0].__name__,
            fa_vrepr,
            id_vrepr
        )


@functools.lru_cache(maxsize=None)  # temporary hack to create singletons
def make_partial_op_model(abstract_partial_op_model, fixed_args):
    """Return the partial property model of the given bit-vector partial operation.

    The argument ``abstract_partial_op_model`` is an (abstract) subclass of
    `PartialOpModel` containing the base operator.

    The argument ``fixed_args`` is a `tuple`, with the same length as
    the number of operands of the base operator, containing ``None``,
    scalar or `Constant` elements. If ``fixed_args[i]`` is ``None``,
    the i-th operand is not fixed; otherwise, the i-th operand is
    replaced with ``fixed_args[i]``. See also `make_partial_operation`.

    The resulting class is a subclass of ``abstract_partial_op_model``.
    """
    assert issubclass(abstract_partial_op_model, PartialOpModel)
    assert not hasattr(abstract_partial_op_model, "op") or abstract_partial_op_model.op is None

    partial_op = operation.make_partial_operation(abstract_partial_op_model.base_op, fixed_args)

    fixed_args_str = []
    for arg in partial_op.fixed_args:
        if arg is None:
            fixed_args_str.append("·")
            continue
        fixed_args_str.append(str(arg))

    class MyPartialOpModel(abstract_partial_op_model):
        op = partial_op

    MyPartialOpModel.__name__ = f"{abstract_partial_op_model.__name__}_{{{', '.join(fixed_args_str)}}}"

    return MyPartialOpModel


class ModelIdentity(OpModel):
    """Represent the (trivial) property model of the identity function.

    This model is used to rename a complex intermediate property with a new name.
    """
    op = operation.BvIdentity

    def validity_constraint(self, output_prop):
        return operation.BvComp(self.input_prop[0].val, output_prop.val)

    def pr_one_constraint(self, output_prop):
        return self.validity_constraint(output_prop)

    def bv_weight(self, output_prop):
        return core.Constant(0, width=1)

    def max_weight(self):
        return 0  # as an integer

    def weight_width(self):
        return 1

    def decimal_weight(self, output_prop):
        self._assert_preconditions_decimal_weight(output_prop)
        dw = decimal.Decimal(0)
        self._assert_postconditions_decimal_weight(output_prop, dw)
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class WeakModel(object):
    """Represent *weak models* of bit-vector operations w.r.t some property.

    A *weak model* is an `OpModel` where the propagation weight of a
    `Property` pair :math:`(\\alpha, \\beta)` is one of the following cases:

    - If :math:`\\alpha = 0 = \\beta`, the weight is `zero2zero_weight`.
    - If :math:`\\alpha \\neq 0 \\neq \\beta`, the weight is `nonzero2nonzero_weight`.
    - If :math:`\\alpha = 0 \\neq \\beta`, the weight is `zero2nonzero_weight`.
    - If :math:`\\alpha \\neq 0 = \\beta`, the weight is `nonzero2zero_weight`.

    If any of weights `zero2zero_weight`, `nonzero2nonzero_weight`,
    `zero2nonzero_weight` or `nonzero2zero_weight` is ``math.inf``,
    the propagation probability of the corresponding case is assumed to be 0.

    This class is not meant to be instantiated but to provide a base
    class for weak models of operations (generated for example
    through `differential.opmodel.get_weak_model` or 
    `linear.opmodel.get_weak_model`).

    Attributes:
        zero2zero_weight: a `decimal.Decimal` denoting the weight
            of the zero to zero transitions (``math.inf`` if these
            transitions are invalid)
        nonzero2nonzero_weight: a `decimal.Decimal` denoting the weight
            of the non-zero to non-zero transitions (``math.inf`` if these
            transitions are invalid)
        zero2nonzero_weight: a `decimal.Decimal` denoting the weight
            of the zero to non-zero transitions (``math.inf`` if these
            transitions are invalid)
        nonzero2zero_weight: a `decimal.Decimal` denoting the weight
            of the non-zero to zero transitions (``math.inf`` if these
            transitions are invalid)
        precision: the number of fraction bits used in the bit-vector weight
            (0 if all non-``math.inf`` attributes ``*_weight`` are integer values)

    .. Implementation details:

        This class does not subclass `abstractproperty.opmodel.OpModel`,
        but subclasses of this class defined for a particular `Property`
        must subclass the corresponding ``OpModel``.
        
    """

    @staticmethod
    def _get_1bit_val(my_val): 
        # return 0b0 if my_val == 0 and 0b1 if my_val != 0 (active)
        assert isinstance(my_val, core.Term)
        if my_val.width == 1:
            return my_val
        return operation.BvNot(operation.BvComp(my_val, core.Constant(0, my_val.width)))

    def __init__(self, input_prop):
        super().__init__(input_prop)

        cls = self.__class__
        found_frac_bits = False
        for w, name in zip(
            [cls.zero2zero_weight, cls.nonzero2nonzero_weight, cls.zero2nonzero_weight, cls.nonzero2zero_weight],
            ["zero2zero_weight", "nonzero2nonzero_weight", "zero2nonzero_weight", "nonzero2zero_weight"]
        ):
            if w != math.inf and w != int(w):
                found_frac_bits = True
                if not isinstance(w, decimal.Decimal):
                    raise ValueError(f"non-integer {name} = {w} is not a Decimal object")
                if math.isclose(w, math.ceil(w)):
                    warnings.warn(f"using {name} = {w} that might get approx. to {math.ceil(w)-1} instead of {math.ceil(w)}")
            if 0 != w != math.inf and int(w * (2 ** cls.precision)) == 0:
                raise ValueError(f"{name} {w} is 0 with precision {cls.precision}")
        if not found_frac_bits and cls.precision != 0:
            raise ValueError(f"precision = {cls.precision} != 0 but no non-integer weight was given")

    def validity_constraint(self, output_prop, weight_condition=None):
        alpha_1b = self._get_1bit_val(
            functools.reduce(operation.Concat, reversed([p.val for p in self.input_prop])))
        beta_1b = self._get_1bit_val(output_prop.val)

        if weight_condition is None:
            def weight_condition(w):
                return w != math.inf

        zero2zero_c = core.Constant(0, 1)  # False (False | * = *)
        if weight_condition(self.__class__.zero2zero_weight):
            zero2zero_c = operation.BvComp((~alpha_1b) & (~beta_1b), core.Constant(1, 1))

        zero2nonzero_c = core.Constant(0, 1)
        if weight_condition(self.__class__.zero2nonzero_weight):
            zero2nonzero_c = operation.BvComp((~alpha_1b) & beta_1b, core.Constant(1, 1))

        nonzero2zero_c = core.Constant(0, 1)
        if weight_condition(self.__class__.nonzero2zero_weight):
            nonzero2zero_c = operation.BvComp(alpha_1b & (~beta_1b), core.Constant(1, 1))

        nonzero2nonzero_c = core.Constant(0, 1)
        if weight_condition(self.__class__.nonzero2nonzero_weight):
            nonzero2nonzero_c = operation.BvComp(alpha_1b & beta_1b, core.Constant(1, 1))

        return zero2zero_c | (zero2nonzero_c | nonzero2zero_c | nonzero2nonzero_c)

    def pr_one_constraint(self, output_prop):
        return self.validity_constraint(output_prop, weight_condition=lambda w: w == 0)

    def bv_weight(self, output_prop):
        alpha_1b = self._get_1bit_val(
            functools.reduce(operation.Concat, reversed([p.val for p in self.input_prop])))
        beta_1b = self._get_1bit_val(output_prop.val)

        width = self.weight_width()
        cls = self.__class__

        def parse_dec(my_decimal):
            assert my_decimal != math.inf
            return int(my_decimal * (2 ** cls.precision))

        w2c = {}
        zb = core.Constant(0, 1)

        if cls.zero2zero_weight != math.inf:
            w = parse_dec(cls.zero2zero_weight)
            w2c[w] = w2c.get(w, zb) | operation.BvComp((~alpha_1b) & (~beta_1b), core.Constant(1, 1))

        if cls.zero2nonzero_weight != math.inf:
            w = parse_dec(cls.zero2nonzero_weight)
            w2c[w] = w2c.get(w, zb) | operation.BvComp((~alpha_1b) & beta_1b, core.Constant(1, 1))

        if cls.nonzero2zero_weight != math.inf:
            w = parse_dec(cls.nonzero2zero_weight)
            w2c[w] = w2c.get(w, zb) | operation.BvComp(alpha_1b & (~beta_1b), core.Constant(1, 1))

        if cls.nonzero2nonzero_weight != math.inf:
            w = parse_dec(cls.nonzero2nonzero_weight)
            w2c[w] = w2c.get(w, zb) | operation.BvComp(alpha_1b & beta_1b, core.Constant(1, 1))

        if len(w2c) == 0:
            raise InvalidOpModelError(f"zero2zero_weight, nonzero2nonzero_weight, zero2nonzero_weight and "
                                      f"nonzero2zero_weight of {self} are math.inf")

        w2c = list(w2c.items())

        ite_expr = core.Constant(w2c[-1][0], width)  # last weight
        for my_weight, my_constraint in reversed(w2c[:-1]):
            ite_expr = operation.Ite(
                my_constraint,
                core.Constant(my_weight, width),
                ite_expr
            )
        return ite_expr

    def max_weight(self):
        max_w = 1
        cls = self.__class__
        for w in [cls.zero2zero_weight, cls.nonzero2nonzero_weight, cls.zero2nonzero_weight, cls.nonzero2zero_weight]:
            if w == math.inf:
                continue
            max_w = max(max_w, int(w * (2**cls.precision)))
        return max_w

    def weight_width(self):
        return self.max_weight().bit_length()

    def decimal_weight(self, output_prop):
        self._assert_preconditions_decimal_weight(output_prop)

        alpha_1b = self._get_1bit_val(
            functools.reduce(operation.Concat, reversed([p.val for p in self.input_prop])))
        beta_1b = self._get_1bit_val(output_prop.val)
        if alpha_1b == 0 and beta_1b == 0:
            dw = self.__class__.zero2zero_weight
        elif alpha_1b == 0 and beta_1b == 1:
            dw = self.__class__.zero2nonzero_weight
        elif alpha_1b == 1 and beta_1b == 0:
            dw = self.__class__.nonzero2zero_weight
        elif alpha_1b == 1 and beta_1b == 1:
            dw = self.__class__.nonzero2nonzero_weight
        else:
            raise ValueError(f"invalid input {self.input_prop} or output {output_prop}")

        dw = decimal.Decimal(dw)
        self._assert_postconditions_decimal_weight(output_prop, dw)
        return dw

    def num_frac_bits(self):
        # num_frac_bits() might be larger than weight_width()
        return self.__class__.precision

    def error(self):
        max_error = 0
        cls = self.__class__
        for w in [cls.zero2zero_weight, cls.nonzero2nonzero_weight, cls.zero2nonzero_weight, cls.nonzero2zero_weight]:
            if w == math.inf:
                continue
            # e.g., w = 3/8 = 0.375 = 0.011, prec=2, approx_w = int(1.5)/2^3 = 0.125
            approx_w = int(w * (2**cls.precision)) / decimal.Decimal(2**cls.precision)
            max_error = max(max_error, abs(w - approx_w))
        # if prec = 2, max_error = 0.001···(inf)··1 = 0.01 - 0.00···(inf)···1
        #   ==> max_error > 0.01 = 2^(-prec)
        assert max_error < decimal.Decimal(2)**(-cls.precision)
        return max_error


class BranchNumberModel(WeakModel):
    """Represent property models of bit-vector operations based on the branch number.

    Let :math:`(y_0, y_1, \dots, y_m) = f(x_1, x_2, \dots, x_n)` be a bit-vector
    function with input words :math:`x_i` and output words :math:`y_j`.
    Given a property pair
    :math:`((P_{x_1}, \dots, P_{x_n}), (P_{y_1}, \dots, P_{y_n}))`
    over :math:`f`, we say that the property
    :math:`P_{x_i}` over :math:`x_i` is *active* if :math:`P_{x_i} \\neq 0`
    (similar for :math:`P_{y_j}`).

    A `BranchNumberModel` of an `Operation` :math:`f` is an `OpModel`
    where the propagation weight of a `Property` pair :math:`(\\alpha, \\beta)`
    is one of the four cases of `WeakModel` but with the additional
    constraint that if :math:`(\\alpha, \\beta)` is non-zero and its
    number of active words is strictly less than :math:`B` then it is invalid.

    The number :math:`B` is usually the *branch number* of :math:`f`, that is,
    the minimum number of active words (at the input and at the output)
    among all non-zero property pairs.
    Nevertheless, any :math:`2 \le B \le n + m` can be used.

    Since an `Operation` outputs a single bit-vector, the
    class attribute `output_widths` of `BranchNumberModel` delimits
    the output words of :math:`f`
    (for the counting of active output words).

    .. note::
        To extract the output words from an `Operation` with a
        `BranchNumberModel`, use `PropExtract` instead of `Extract`.

    This class is not meant to be instantiated but to provide a base
    class for branch-number-based models of operations (generated for example
    through `differential.opmodel.get_branch_number_model` or
    `linear.opmodel.get_branch_number_model`).

    Attributes:
        output_widths: a `tuple` containing the widths of the output words
        branch_number: the branch number :math:`B`
        zero2zero_weight: a `decimal.Decimal` denoting the weight
            of the zero to zero transitions (``math.inf`` if all these
            transitions are invalid)
        nonzero2nonzero_weight: a `decimal.Decimal` denoting the weight
            of the non-zero to non-zero transitions (``math.inf`` if all these
            transitions are invalid)
        zero2nonzero_weight: a `decimal.Decimal` denoting the weight
            of the zero to non-zero transitions (``math.inf`` if all these
            transitions are invalid)
        nonzero2zero_weight: a `decimal.Decimal` denoting the weight
            of the non-zero to zero transitions (``math.inf`` if all these
            transitions are invalid)
        precision: the number of fraction bits used in the bit-vector weight
            (0 if all non-``math.inf`` attributes ``*_weight`` are integer values)

    .. Implementation details:

        This class does not subclass `abstractproperty.opmodel.OpModel`,
        but subclasses of this class defined for a particular `Property`
        must subclass the corresponding ``OpModel``.

        Source: The Design of Rijndael: AES - The Advanced Encryption Standard

    """

    def _split(self, output_val):
        assert isinstance(output_val, core.Term)
        assert output_val.width == sum(self.output_widths)
        split_output = []
        offset = 0
        for w in self.output_widths:
            # in Extract both endpoints are included
            split_output.append(output_val[offset + w - 1: offset])
            assert split_output[-1].width == w
            offset += w
        assert output_val == functools.reduce(operation.Concat, reversed(split_output))
        return split_output  # first list element contains LSB, last MSB

    def validity_constraint(self, output_prop, weight_condition=None):
        cls = self.__class__
        assert 2 <= cls.branch_number <= len(self.input_prop) + len(cls.output_widths)

        alpha_1b = self._get_1bit_val(
            functools.reduce(operation.Concat, reversed([p.val for p in self.input_prop])))
        beta_1b = self._get_1bit_val(output_prop.val)

        if weight_condition is None:
            def weight_condition(w):
                return w != math.inf

        zero2zero_c = core.Constant(0, 1)  # False (False | * = *)
        if weight_condition(self.__class__.zero2zero_weight):
            zero2zero_c = operation.BvComp((~alpha_1b) & (~beta_1b), core.Constant(1, 1))

        zero2nonzero_c = core.Constant(0, 1)
        if weight_condition(self.__class__.zero2nonzero_weight):
            zero2nonzero_c = operation.BvComp((~alpha_1b) & beta_1b, core.Constant(1, 1))

        nonzero2zero_c = core.Constant(0, 1)
        if weight_condition(self.__class__.nonzero2zero_weight):
            nonzero2zero_c = operation.BvComp(alpha_1b & (~beta_1b), core.Constant(1, 1))

        nonzero2nonzero_c = core.Constant(0, 1)
        if weight_condition(self.__class__.nonzero2nonzero_weight):
            nonzero2nonzero_c = operation.BvComp(alpha_1b & beta_1b, core.Constant(1, 1))

        #

        alpha_1b_list = [self._get_1bit_val(p.val) for p in self.input_prop]
        beta_1b_list = [self._get_1bit_val(v) for v in self._split(output_prop.val)]

        list_1b = alpha_1b_list + beta_1b_list  # i-th element is 1 if var_i is 1 (active)
        # extend avoid overflow when +
        num_bits = max(len(list_1b).bit_length(), cls.branch_number)
        list_1b = [operation.zero_extend(b, num_bits - b.width) for b in list_1b]
        bn_constraint = operation.BvUge(sum(list_1b), core.Constant(cls.branch_number, num_bits))

        return zero2zero_c | ((zero2nonzero_c | nonzero2zero_c | nonzero2nonzero_c) & bn_constraint)

    def pr_one_constraint(self, output_prop):
        return self.validity_constraint(output_prop, weight_condition=lambda w: w == 0)


class WDTModel(object):
    """Represent WDT-based models of bit-vector operations w.r.t some property.

    A model based on a Weight Distribution Table (WDT) is an `OpModel`
    where the propagation weight is obtained from a 2-dimensional
    table ``WDT`` such that ``WDT[a][b]`` contains the propagation
    weight of the input-output `Property` pair :math:`(a, b)`.

    .. note::
        For the `Difference` (resp. `LinearMask`) property,
        the WDT is also known as the Difference Distribution Table or DDT
        (resp. Linear Approximation Table or LAT).

    This class is not meant to be instantiated but to provide a base
    class for WDT-based models of operations (generated for example
    through `differential.opmodel.get_wdt_model` or
    `linear.opmodel.get_wdt_model`).

    Attributes:
        weight_distribution_table: a 2-dimensional `tuple` containing
            the distribution of the propagations weights,
            where `math.inf` entries correspond to invalid transitions
        loop_rows_then_columns: whether to extract the constraints from
            `weight_distribution_table` by looping first over the rows
            (input properties) and then over the columns (output properties)
            or vice-versa
        precision: the number of fraction bits used in the bit-vector weight
            (0 if all the weights in `weight_distribution_table` are integer values)

    .. Implementation details:

        This class does not subclass `abstractproperty.opmodel.OpModel`,
        but subclasses of this class defined for a particular `Property`
        must subclass the corresponding ``OpModel``.

    """

    @staticmethod
    def _transpose(my_2D_list):
        num_rows = len(my_2D_list)
        num_columns = len(my_2D_list[0])
        assert all(num_columns == len(row) for row in my_2D_list)
        transposed = [[my_2D_list[i][j] for i in range(num_rows)] for j in range(num_columns)]
        num_rows, num_columns = num_columns, num_rows
        assert num_rows == len(transposed) and num_columns == len(transposed[0])
        return transposed, num_rows, num_columns

    def __init__(self, input_prop):
        super().__init__(input_prop)

        if not(self.op.arity == [1, 0] and len(self.input_prop) == 1):
            raise ValueError("WDTModel only supports operations with 1 input operand")

        WDT = self.__class__.weight_distribution_table
        num_rows = len(WDT)
        num_columns = len(WDT[0])

        if num_rows != 2 ** sum(p.val.width for p in self.input_prop):
            raise ValueError(f"2**<input bit-size> = {2 ** sum(p.val.width for p in self.input_prop)} "
                             f"!= {num_rows} = number of rows of WDT")
        if not all(num_columns == len(row) for row in WDT):
            raise ValueError("WDT contains columns with different length")
        if not all(all(elem >= 0 for elem in row) for row in WDT):
            raise ValueError("negative weights found in WDT")

        precision = self.__class__.precision
        found_frac_bits = False
        for row in WDT:
            for weight in row:
                if weight != math.inf and weight != int(weight):
                    found_frac_bits = True
                    if not isinstance(weight, decimal.Decimal):
                        raise ValueError(f"non-integer weight {weight} in weight_distribution_table is"
                                         f" not a Decimal object")
                    if math.isclose(weight, math.ceil(weight)):
                        warnings.warn(
                            f"using weight with value {weight} that might get approx. to "
                            f"{math.ceil(weight) - 1} instead of {math.ceil(weight)}")
                if 0 != weight != math.inf and int(weight * (2 ** precision)) == 0:
                    raise ValueError(f"non-zero weight {weight} in weight_distribution_table is"
                                     f" 0 with precision {precision}")
        if not found_frac_bits and precision != 0:
            raise ValueError(f"precision = {precision} != 0 but no non-integer weight was given")

    def validity_constraint(self, output_prop):
        WDT = self.__class__.weight_distribution_table
        num_rows = len(WDT)
        num_columns = len(WDT[0])
        assert num_columns == 2 ** output_prop.val.width

        alpha = self.input_prop[0].val  # multiple input props could be done by concatenating them
        beta = output_prop.val

        # Ite representation

        civ2oc = collections.OrderedDict()
        oc2list_civ = {}

        if not self.__class__.loop_rows_then_columns:
            alpha, beta = beta, alpha
            WDT, num_rows, num_columns = self._transpose(WDT)

        with context.Simplification(False):
            for ct_input_val in range(num_rows):
                add_invalid = WDT[ct_input_val].count(math.inf) < num_columns / 2

                output_constraint = core.Constant(int(add_invalid), 1)
                for ct_output_val in range(num_columns):
                    if add_invalid == (WDT[ct_input_val][ct_output_val] == math.inf):
                        constraint = operation.BvComp(beta, core.Constant(ct_output_val, beta.width))
                        if add_invalid:
                            output_constraint &= operation.BvNot(constraint)
                        else:
                            output_constraint |= constraint

                # e.g., if for a ct_input_val A, only 2 ct_output_val B1, B2 are invalid (among 4+ Bi)
                # we will have Ite(input == A, output_constraint, ···) where
                #                   output_constraint = ~(output == B1) & ~(output == B2)
                # but if only 3 Bi: output_constraint = output == B3

                civ2oc[ct_input_val] = output_constraint
                oc2list_civ[output_constraint] = oc2list_civ.get(output_constraint, []) + [ct_input_val]

            oc2list_civ = sorted(oc2list_civ.items(), key=lambda x: (len(x[1]), x[1]))  # [1] < [4] < [2, 3]

            if len(oc2list_civ) == 1:
                oc, list_civ = oc2list_civ[0]
                assert len(list_civ) == num_rows
                return oc

            ite = oc2list_civ[-1][0]  # only the OC of the last list_civ is used (and not the large list_civ)
            for output_constraint, list_civ in oc2list_civ[:-1]:
                list_civ = [core.Constant(civ, alpha.width) for civ in list_civ]
                input_c = functools.reduce(operation.BvOr, [operation.BvComp(alpha, civ) for civ in list_civ])
                ite = operation.Ite(
                    input_c,
                    output_constraint,
                    ite
                )

        return ite

        # # sum (disjunction) of minterms representation
        # ite = constant(0, 1)  # False
        # with context.Simplification(False):
        #     for ct_input_val in range(num_rows):
        #         # e.g., if for a ct_input_val A, only 2 ct_output_val B1, B2 are invalid
        #         # we add the constraint [(input == A) & ~(output == B1)] | [(input == A) & ~(output == B2)]
        #         # (equivalent to [(input == A) & (output == B2)] | [(input == A) & output == B3)] | ··· )
        #         add_invalid = WDT[ct_input_val].count(math.inf) < num_columns / 2
        #         input_constraint = operation.BvComp(alpha, core.Constant(ct_input_val, alpha.width))
        #         for ct_output_val in range(num_columns):
        #             if add_invalid == (WDT[int(ct_input_val)][ct_output_val] == math.inf):
        #                 output_constraint = operation.BvComp(beta, core.Constant(ct_output_val, beta.width))
        #                 if add_invalid:
        #                     output_constraint = operation.BvNot(output_constraint)
        #                 assert ite is not None  # ite == False
        #                 ite |= (input_constraint & output_constraint)
        # return ite

    def pr_one_constraint(self, output_prop):
        WDT = self.__class__.weight_distribution_table
        num_rows = len(WDT)
        num_columns = len(WDT[0])
        assert num_columns == 2 ** output_prop.val.width

        alpha = self.input_prop[0].val
        beta = output_prop.val

        civ2oc = collections.OrderedDict()
        oc2list_civ = {}

        if not self.__class__.loop_rows_then_columns:
            alpha, beta = beta, alpha
            WDT, num_rows, num_columns = self._transpose(WDT)

        with context.Simplification(False):
            for ct_input_val in range(num_rows):
                add_non_zero = len([w for w in WDT[ct_input_val] if w != 0]) < num_columns / 2

                output_constraint = core.Constant(int(add_non_zero), 1)
                for ct_output_val in range(num_columns):
                    if add_non_zero == (WDT[ct_input_val][ct_output_val] != 0):
                        constraint = operation.BvComp(beta, core.Constant(ct_output_val, beta.width))
                        if add_non_zero:
                            output_constraint &= operation.BvNot(constraint)
                        else:
                            output_constraint |= constraint

                # e.g., if for a ct_input_val A, only 2 ct_output_val B1, B2 are non-zero (among 4+ Bi)
                # we will have Ite(input == A, output_constraint, ···) where
                #                   output_constraint = ~(output == B1) & ~(output == B2)
                # but if only 3 Bi: output_constraint = output == B3

                civ2oc[ct_input_val] = output_constraint
                oc2list_civ[output_constraint] = oc2list_civ.get(output_constraint, []) + [ct_input_val]

            oc2list_civ = sorted(oc2list_civ.items(), key=lambda x: (len(x[1]), x[1]))  # [1] < [4] < [2, 3]

            if len(oc2list_civ) == 1:
                oc, list_civ = oc2list_civ[0]
                assert len(list_civ) == num_rows
                return oc

            ite = oc2list_civ[-1][0]  # only the OC of the last list_civ is used (and not the large list_civ)
            for output_constraint, list_civ in oc2list_civ[:-1]:
                list_civ = [core.Constant(civ, alpha.width) for civ in list_civ]
                input_c = functools.reduce(operation.BvOr, [operation.BvComp(alpha, civ) for civ in list_civ])
                ite = operation.Ite(
                    input_c,
                    output_constraint,
                    ite
                )

        return ite

    @property
    def _WDT_with_precision_and_loop_order(self):
        if not hasattr(self, "_stored_WDT_with_precision_and_loop_order"):
            if not self.__class__.loop_rows_then_columns:
                WDT, _, _ = self._transpose(self.__class__.weight_distribution_table)
            else:
                WDT = [list(row) for row in self.__class__.weight_distribution_table]
            num_rows = len(WDT)
            num_columns = len(WDT[0])
            for i in range(num_rows):
                for j in range(num_columns):
                    if WDT[i][j] != math.inf:
                        WDT[i][j] = int(WDT[i][j] * (2 ** self.__class__.precision))
                        WDT[i][j] = core.Constant(WDT[i][j], self.weight_width())
            self._stored_WDT_with_precision_and_loop_order = WDT
        return self._stored_WDT_with_precision_and_loop_order

    def bv_weight(self, output_prop):
        WDT_with_p_lo = self._WDT_with_precision_and_loop_order
        num_rows = len(WDT_with_p_lo)
        num_columns = len(WDT_with_p_lo[0])
        if self.__class__.loop_rows_then_columns:
            assert num_columns == 2 ** output_prop.val.width
        else:
            assert num_rows == 2 ** output_prop.val.width

        alpha = self.input_prop[0].val
        beta = output_prop.val

        civ2iw = collections.OrderedDict()
        iw2list_civ = {}

        if not self.__class__.loop_rows_then_columns:
            alpha, beta = beta, alpha
            # WDT_with_p_lo already transposed

        with context.Simplification(False):
            for ct_input_val in range(num_rows):
                non_inf_weights = collections.Counter(w for w in WDT_with_p_lo[ct_input_val] if w != math.inf)
                non_inf_weights = [w for w, _ in non_inf_weights.most_common(None)]  # 1st weight most common

                if len(non_inf_weights) == 0:
                    continue

                ite_weights = non_inf_weights[0]  # if all Ite conditions are False, result is the most common weight
                for bv_w in reversed(non_inf_weights[1:]):  # omitting the 1st weight
                    assert isinstance(bv_w, core.Constant)
                    w_constraint = core.Constant(0, 1)
                    for ct_output_val in range(num_columns):
                        if WDT_with_p_lo[ct_input_val][ct_output_val] == bv_w:
                            w_constraint |= operation.BvComp(beta, core.Constant(ct_output_val, beta.width))
                    ite_weights = operation.Ite(
                        w_constraint,
                        bv_w,
                        ite_weights)

                # e.g., if for a ct_input_val A, only ct_output_vals B1, B2 have weight 0 and B3 weight 1
                # we will have Ite(input == A, ite_weight, ···) where
                #   ite_weights = Ite(output == B3, 1, 0)

                civ2iw[ct_input_val] = ite_weights
                iw2list_civ[ite_weights] = iw2list_civ.get(ite_weights, []) + [ct_input_val]

            iw2list_civ = sorted(iw2list_civ.items(), key=lambda x: (len(x[1]), x[1]))  # [1] < [4] < [2, 3]

            if len(iw2list_civ) == 1:
                iw, list_civ = iw2list_civ[0]
                # # if some row/column is full of math.inf, list_civ might contain fewer elements
                # # e,g, for wdt = [(0, 1), (inf, inf)]], list_civ = [0]
                # assert len(list_civ) == num_rows
                return iw

            main_ite = iw2list_civ[-1][0]  # only the ite_weights of the last list_civ is used
            for ite_ws, list_civ in iw2list_civ[:-1]:
                list_civ = [core.Constant(civ, alpha.width) for civ in list_civ]
                input_c = functools.reduce(operation.BvOr, [operation.BvComp(alpha, civ) for civ in list_civ])
                main_ite = operation.Ite(
                    input_c,
                    ite_ws,
                    main_ite
                )

        return main_ite

    def max_weight(self):
        max_w = 1
        # cannot use self._WDT_with_precision_and_loop_order since it calls max_weight
        for row in self.__class__.weight_distribution_table:
            for w in row:
                if w == math.inf:
                    continue
                max_w = max(max_w, int(w * (2 ** self.__class__.precision)))
        return max_w

    def weight_width(self):
        return self.max_weight().bit_length()

    def decimal_weight(self, output_prop):
        self._assert_preconditions_decimal_weight(output_prop)

        alpha = self.input_prop[0].val
        beta = output_prop.val
        dw = self.__class__.weight_distribution_table[int(alpha)][int(beta)]
        dw = decimal.Decimal(dw)

        self._assert_postconditions_decimal_weight(output_prop, dw)
        return dw

    def num_frac_bits(self):
        return self.__class__.precision

    def error(self):
        max_error = 0
        cls = self.__class__
        for row in cls.weight_distribution_table:
            for w in row:
                if w == math.inf:
                    continue
                approx_w = int(w * (2**cls.precision)) / decimal.Decimal(2**cls.precision)
                max_error = max(max_error, abs(w - approx_w))
        assert max_error < decimal.Decimal(2)**(-cls.precision)
        return max_error
