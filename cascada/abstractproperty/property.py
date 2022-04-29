"""Provide a base class for properties like `Difference`, `LinearMask` or `Value`."""
import collections
import functools

from cascada.bitvector import core, operation


def _tuplify(seq):
    if isinstance(seq, collections.abc.Sequence):
        return tuple(seq)
    else:
        return tuple([seq])


class Property(object):
    """Represent bit-vector properties.

    A (bit-vector) property pair, or simply a property, over a function :math:`f`
    is a pair of bit-vectors :math:`(\\alpha, \\beta)` with
    an associated propagation probability :math:`PP_f(\\alpha, \\beta)`.
    In this case, we also say that the input property :math:`\\alpha` propagates
    to the output property :math:`\\beta` with probability
    :math:`PP_f(\\alpha, \\beta)`.

    Each instance of this class represents an input or output property;
    the underlying bit-vector value of the input or output property is stored
    in the attribute `val`, but the associated function :math:`f`
    is not stored within the `Property` object.

    Note that arithmetic with properties is not supported.
    For example, two `Property` objects ``d1`` and ``d2``
    cannot be XORed, i.e., ``d1 ^ d2``.
    This can be done instead by performing the arithmetic with
    the property values and converting the resulting
    `Term` to a property, that is, ``Property(d1.val ^ d2.val)``

    This class is not meant to be instantiated but to provide a base
    class to define bit-vector properties
    such as `Difference`, `LinearMask` or `Value`.

    Attributes:
        val: a `Term` representing the underlying value of the property.

    """

    def __init__(self, value):
        assert isinstance(value, core.Term)
        self.val = value

    def __str__(self):
        """Return the non-verbose string representation."""
        return "{}({})".format(type(self).__name__, str(self.val))

    __repr__ = __str__

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.val == other.val
        else:
            return False

    def xreplace(self, rule):
        """Replace occurrences of properties within the expression.

        The argument ``rule`` is a dict-like object representing
        the replacement rule.

        This method is similar to SymPy `xreplace
        <https://docs.sympy.org/latest/modules/core.html?highlight=xreplace#
        sympy.core.basic.Basic.xreplace>`_ but with the restriction that
        only `Property` objects are allowed in ``rule``.
        """
        for d in rule:
            assert isinstance(d, type(self)) and isinstance(rule[d], type(self)), \
                f"invalid xreplace rule ({d}, {rule[d]})"

        rule = {d.val: rule[d].val for d in rule}
        return type(self)(self.val.xreplace(rule))

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that the relation
        ``eval(self.vrepr()) == self`` holds.
        """
        return "{}({})".format(type(self).__name__, self.val.vrepr())

    @classmethod
    def propagate(cls, op, input_prop):
        """Propagate the given input property through the given operation.

        Given a function :math:`y = f(x)`,
        an input property :math:`p_x` propagates (over :math:`f`) to
        an output property :math:`p_y` if the propagation probability
        :math:`PP_f(\\alpha, \\beta)` is not zero.

        Note that if :math:`f` has :math:`t` operands,
        the input property is actually a list of :math:`t` properties, that is,
        :math:`p_x = (p_{x_0}, p_{x_1}, \dots, p_{x_{t-1}})`,
        and similar for :math:`p_y`.

        For some functions, any input property propagates to a unique
        output property with probability 1, and the output property
        can be easily computed from the input property.

        Examples of probability-one propagations are the following:

        * For the `Value` property type, for any operation :math:`f` and any
          input property :math:`p_x`, the output property :math:`p_y`
          is uniquely determined and its bit-vector value is ``f(p_x.val)``.
        * For the `Difference` property type with difference operation :math:`-`
          (e.g., XOR),  for any linear (over :math:`-`) operation :math:`f` and any
          input property :math:`p_x`, the output property :math:`p_y`
          is uniquely determined and its bit-vector value is ``f(p_x.val)``.
        * For the `LinearMask` property, for any XOR-linear operation :math:`f`
          and any input property :math:`p_x`,
          the output property :math:`p_y` is uniquely determined and its
          bit-vector value satisfies ``p_x.val == M(p_y.val)``, where :math:`M`
          is the transpose of the binary matrix representing :math:`f`.

        If for the given `Operation` ``op`` any input property propagates
        to a unique output property with probability 1 and the output property
        can be easily computed from the input property, this method
        returns the corresponding output property (for the given
        input property ``input_prop``) as a `Property` object.
        Otherwise, this method returns the
        `abstractproperty.opmodel.OpModel` of the operation ``op``
        with the given input property ``input_prop``.

        .. note::
            Operations ``op`` with scalar operands are not supported,
            but these operands can be removed with `make_partial_operation`
            and partial operations are supported by this method.

        Args:
            op:  the bit-vector operator :math:`f` as an `Operation` subclass
            input_prop: a list containing the input `Property` for each operand of
                ``op`` (or simply the `Property` object if there is only
                one operand)

        """
        raise NotImplementedError("subclasses need to override this method")


class PropConcat(operation.Concat):
    """Subclass of `Concat` that propagates properties naturally.

    The only difference between `Concat` and `PropConcat`
    is that the latter one propagates (see `Property.propagate`)
    an input `Property` list ``p0, p1`` as ``Property(Concat(p0.val, p1.val))``.

     .. Implementation details:

        This propagation is different, for example,
        from  the one of Concat with RX differences.

    """
    _trivial_propagation = True


class PropExtract(operation.Extract):
    """Subclass of ``Extract`` that propagates properties naturally.

    The only difference between `Extract` and `PropExtract`
    is that the latter one (with fixed operands) propagates
    (see `Property.propagate`) an input `Property` ``p`` as
    ``Property(Extract_{·, i, j}(p0.val))``,
    where ``i, j`` are the fixed operands.

     .. Implementation details:

        This propagation is different, for example,
        from the one of Extract with linear masks.

    """
    _trivial_propagation = True


@functools.lru_cache(maxsize=None)  # temporary hack to create singletons
def make_partial_propextract(i, j):
    """Return a `PartialOperation` of `PropExtract` that propagates properties naturally.

    .. Implementation details:

        The different between `make_partial_propextract` and
        ``make_partial_operation(PropExtract, ...)` is that the
        former partial operation propagates properties naturally
        but not the latter one, as some properties do not
        support partial operations without OpModels.

    """
    partial_extract = operation.make_partial_operation(PropExtract, (None, i, j))

    class PartialPropExtract(partial_extract):
        _trivial_propagation = True  # make_partial_operation doesn't preserve attributes

    assert hasattr(PartialPropExtract, "_trivial_propagation")

    PartialPropExtract.__name__ = partial_extract.__name__

    return PartialPropExtract
