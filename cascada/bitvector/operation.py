"""Provide the common bit-vector operators.

.. autosummary::
   :nosignatures:

    Operation
    PrimaryOperation
    SecondaryOperation
    PartialOperation
    make_partial_operation
    BvNot
    BvAnd
    BvOr
    BvXor
    BvComp
    BvUlt
    BvUle
    BvUgt
    BvUge
    BvShl
    BvLshr
    RotateLeft
    RotateRight
    Ite
    Extract
    Concat
    BvNeg
    BvAdd
    BvSub
    BvMul
    BvUdiv
    BvUrem
    BvIdentity
    zero_extend
    repeat
"""
import collections
import functools
import itertools
import math

from sympy.core import cache
from sympy.printing import precedence as sympy_precedence

from cascada.bitvector import context
from cascada.bitvector import core


zip = functools.partial(zip, strict=True)


def _cacheit(func):
    """Cache functions if `CacheContext` is enabled."""
    cfunc = cache.cacheit(func)

    def cached_func(*args, **kwargs):
        if context.Cache.current_context:
            return cfunc(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return cached_func


def _tuplify(seq):
    if isinstance(seq, collections.abc.Sequence):
        return tuple(seq)
    else:
        return tuple([seq])


# noinspection PyUnresolvedReferences
class Operation(core.Term):
    """Represent bit-vector operations.

    A bit-vector operation is a mathematical function that takes some bit-vector
    operands (i.e. `Term`) and some scalar operands (i.e. `int`),
    and returns a single bit-vector term. Often, *operator* is used to
    denote the Python class representing the mathematical function (without operands)
    and *operation* is used to denote the application of an operator to some operands.

    A particular operator (a subclass of `Operation`) can be evaluated
    by instantiating an object with the operands as the object arguments.
    The instantiation internally calls the method `eval`, containing the logic
    of the operator. This behaviour is similar as those of the SymPy classes
    `Add <https://docs.sympy.org/latest/modules/core.html?highlight=add#sympy.core.add.Add>`_
    or `Mul <https://docs.sympy.org/latest/modules/core.html?highlight=mul#sympy.core.mul.Mul>`_.

    Unless the `Simplification` context is disabled (enabled by default),
    many of the operations automatically simplify complex expressions
    by applying basic rules from `Boolean algebra
    <https://en.wikipedia.org/wiki/Bitwise_operation#Boolean_algebra>`_.
    Moreoever, operations are also affected by other context managers
    such as `Cache`, `PrimaryOperationEvaluation`, `SecondaryOperationEvaluation`,
    `Validation` or `Memoization`.

    This class is not meant to be instantiated but to provide a base
    class for the two types of operations: `PrimaryOperation`
    and `SecondaryOperation`.

    .. note::

        New bit-vector operations can be defined by subclassing `SecondaryOperation`
        (see `BvMaj` as example). It is also possible to define new bit-vector
        operations by subclassing `PrimaryOperation`, but this is not recommended.

    Attributes:
        arity: a pair of number specifying the number of bit-vector operands
            (at least one) and scalar operands, respectively.
        is_symmetric: True if the operator is symmetric with respect to
            its operands (a permutation of the inputs does not change the output).
            Operators with scalar operands cannot be symmetric.
        is_simple: True if the operator is *simple*, that is, all its
            operands are bit-vector of the same width. Simple operators allow
            *Automatic Constant Conversion*, that is, instead of passing
            all arguments as bit-vector types, it is possible to pass
            arguments as plain integers.

            ::

                >>> from cascada.bitvector.core import Constant
                >>> (Constant(1, 8) + 1).vrepr()
                'Constant(0b00000010, width=8)'

        operand_types: a list specifying the types of the operands (optional
            if all operands are bit-vectors)
        alt_name: an alternative name used when printing (optional)
        unary_symbol: a symbol used when printing (optional)
        infix_symbol: a symbol used when printing (optional)

    .. Implementation details:

        New operations should be added in the header of test_operation.

    """

    is_Atom = False
    precedence = sympy_precedence.PRECEDENCE["Func"]

    is_symmetric = False
    is_simple = False

    @_cacheit
    def __new__(cls, *args, **options):
        assert issubclass(cls, (PrimaryOperation, SecondaryOperation))

        val_op = options.pop("validate_operands", context.Validation.current_context)

        # evaluate used in _binary_symmetric_simplification
        evaluate = options.pop("evaluate", None)
        pre_evaluate = True if evaluate is None else evaluate
        if evaluate is None:
            # options["evaluate"] overrides context
            if issubclass(cls, PrimaryOperation):
                evaluate = context.PrimaryOperationEvaluation.current_context
            else:
                evaluate = context.SecondaryOperationEvaluation.current_context
            if all(not isinstance(a, (core.Variable, Operation)) for a in args):
                # constant args
                evaluate = True

        simplify = options.pop("simplify", context.Simplification.current_context)
        st = options.pop("state", context.Memoization.current_context)

        if val_op:
            args = cls._parse_args(*args)

        if st is not None:
            found_non_memoized_op_input = False
            newargs = []
            for arg in args:
                if isinstance(arg, Operation):
                    if st.contain_op(arg):
                        newargs.append(st.get_id(arg))
                    else:
                        # non-simple assignments allowed in MemoizationTable
                        found_non_memoized_op_input = True
                        newargs.append(arg)
                else:
                    newargs.append(arg)
            args = newargs

        width = cls.output_width(*args)

        # if cls is secondary op, intermediate primary operations
        # need to be stored in st (but not for primary op)
        with context.Memoization(None if issubclass(cls, PrimaryOperation) else st):
            result = None
            if pre_evaluate and issubclass(cls, SecondaryOperation):
                result = cls.pre_eval(*args)
            if evaluate:
                if result is None:
                    result = cls.eval(*args)

        if result is not None:
            # result is already a Term/Operation (possibly simplified)
            obj = result
        else:
            obj = super().__new__(cls, *args, width=width)

            if isinstance(obj, Operation) and simplify and evaluate:
                with context.Simplification(False), context.Memoization(None):
                    while True:
                        obj, modified = obj._simplify()
                        if not modified or not isinstance(obj, Operation):
                            break

        if isinstance(obj, Operation) and st is not None:
            for arg in obj.args:
                if not found_non_memoized_op_input and isinstance(arg, Operation):
                    raise ValueError("arg {} of {} was not memoized".format(arg, obj))
            if st.contain_op(obj):
                return st.get_id(obj)
            else:
                return st.add_op(obj)

        return obj

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if super().__eq__(other):
            return True
        elif type(self) == type(other) and self.is_symmetric and other.is_symmetric:
            eval_sec_ops = context.SecondaryOperationEvaluation.current_context
            sorted_self = self.doit(eval_sec_ops=eval_sec_ops, sort_symmetric_args=True)
            sorted_other = other.doit(eval_sec_ops=eval_sec_ops, sort_symmetric_args=True)
            return core.Term.__eq__(sorted_self, sorted_other)
        else:
            return False

    @classmethod
    def _parse_args(cls, *args):
        # Automatic Constant Conversion
        if cls.is_simple:
            for a in args:
                if isinstance(a, core.Term):
                    w = a.width
                    break
            else:
                msg = "{} expects at least 1 term operand"
                raise TypeError(msg.format(cls.__name__))

            args = [core.Constant(a, w) if isinstance(a, int) else a for a in args]

        if hasattr(cls, "operand_types"):
            operand_types = cls.operand_types
        else:
            operand_types = [core.Term for _ in args]
        for arg_type, arg in zip(operand_types, args):
            assert isinstance(arg, arg_type)

        num_terms = 0
        num_scalars = 0
        for a in args:
            if isinstance(a, core.Term):
                num_terms += 1
            elif isinstance(a, int):
                num_scalars += 1
            else:
                assert False
        assert tuple(cls.arity) == (num_terms, num_scalars)
        assert num_terms >= 1

        # # disabled sorting symmetric operations by default
        # if cls.is_symmetric:
        #     args = sorted(args, key=sympy.default_sort_key)

        assert cls.condition(*args), "{}.condition({}) did not hold".format(
            cls, [a if not isinstance(a, core.Term) else (a, a.width) for a in args])

        return args

    def _simplify(self):
        """Simplify the bit-vector operation.

        Return the simplified value and a boolean flag depending on
        whether the expression was reduced.
        """
        return self, False

    def _binary_symmetric_simplification(self, compatible_terms):
        """Simplify a binary symmetric operation.

        Replace pairs of *compatible connected terms* by their resulting value.
        * Two terms are connected if they are arguments of the same operator
          node when the bit-vector expression is flattened (e.g. ``z``
          and ``t`` are connected in ``((x ^ y) + z) + t``.
        * Two connected terms are compatible if they can be simplified.
          For example, two constants are always compatible.

        Note that this function assumed the arguments of the operation
        are already simplified.

        Args:
            compatible_terms: a list of lambda functions specifying
            the compatible terms for a particular operator.

        """
        op = type(self)
        assert isinstance(compatible_terms, collections.abc.Sequence)

        # noinspection PyShadowingNames
        def replace_constant(cte, expr):
            modified = False
            newargs = []

            for arg in expr.args:
                if not modified:
                    if isinstance(arg, core.Constant):
                        arg = op(cte, arg)
                        modified = True
                    elif isinstance(arg, op):
                        arg, modified = replace_constant(cte, arg)

                    newargs.append(arg)
                else:
                    newargs.append(arg)

            assert len(newargs) in [1, 2]
            if len(newargs) == 1:
                new_expr = newargs[0]
            elif len(newargs) == 2:
                new_expr = op(*newargs)
            else:
                raise ValueError("invalid newargs length: {}".format(newargs))

            return new_expr, modified

        # noinspection PyShadowingNames
        def replace_term(term, compatible_terms, expr):
            modified = False
            newargs = []

            for arg in expr.args:
                if not modified:
                    if arg in compatible_terms:
                        arg = op(term, arg)
                        modified = True
                    elif isinstance(arg, op):
                        arg, modified = replace_term(term, compatible_terms, arg)

                    newargs.append(arg)
                else:
                    newargs.append(arg)

            assert len(newargs) in [1, 2]
            if len(newargs) == 1:
                new_expr = newargs[0]
            elif len(newargs) == 2:
                new_expr = op(*newargs)
            else:
                raise ValueError("invalid newargs length: {}".format(newargs))

            return new_expr, modified

        x, y = self.args

        modified = False  # modified

        if isinstance(x, core.Constant) and isinstance(y, op):
            new_expr, modified = replace_constant(x, expr=y)
        elif isinstance(y, core.Constant) and isinstance(x, op):
            new_expr, modified = replace_constant(y, expr=x)

        if not modified and not isinstance(x, core.Constant) and isinstance(y, op):
            new_expr, modified = replace_term(x, [f(x) for f in compatible_terms], y)

        if not modified and not isinstance(y, core.Constant) and isinstance(x, op):
            new_expr, modified = replace_term(y, [f(y) for f in compatible_terms], x)

        if not modified and isinstance(x, op) and isinstance(y, op):
            x1, x2 = x.args

            if op(x1, y, evaluate=False) != op(x1, y):
                new_expr = op(x2, op(x1, y))
                modified = True

            if not modified and op(x2, y, evaluate=False) != op(x2, y):
                new_expr = op(x1, op(x2, y))
                modified = True

            if not modified:
                new_expr, modified = op(x1, y)._simplify()
                new_expr = op(x2, new_expr)

            if not modified:
                new_expr, modified = op(x2, y)._simplify()
                new_expr = op(x1, new_expr)

        if modified:
            # noinspection PyUnboundLocalVariable
            return new_expr, True
        else:
            return self, False

    @classmethod
    def condition(cls, *args):
        """Check if the operands verify the restrictions of the operator."""
        return True

    def output_width(*args):
        """Return the bit-width of the resulting bit-vector."""
        raise NotImplementedError("subclasses need to override this method")

    @classmethod
    def eval(cls, *args):
        """Evaluate the operator with given operands.

        This is an internal method that assumes the list ``args`` has been parsed.
        To evaluate a bit-vector operation, instantiate a new object with
        the operands as the object arguments (i.e., use the Python operator ``()``).
        """
        raise NotImplementedError("subclasses need to override this method")

    def formula_size(self):
        """The formula size of the operation."""
        def log2(n):
            return int(math.ceil(math.log2(n)))

        def bin_enc(n):
            return 1 + log2(n + 1)

        size = 1 + bin_enc(self.width)
        for arg in self.args:
            if isinstance(arg, int):
                size += bin_enc(arg)
            else:
                size += arg.formula_size()

        return size

    def memoization_table(self, id_prefix="x", decompose_sec_ops=True):
        """Return a decomposition of the current operation into simple assignments.

        Given an `Operation` object :math:`F(a_1, a_2, \dots)`, this method
        decomposes it into a list of *simple* assignments
        :math:`x_{i+1} \leftarrow f_i(x_i)` such that the last output variable
        :math:`x_{n}` represents the output of :math:`F(a_1, a_2, \dots)`.
        The i-th assignment is given by:

            - the output `Variable` :math:`x_{i+1}` that represents
              the output of :math:`f_i(x_i)`,

            - the `Operation` object :math:`f_i(x_i)` where the input
              :math:`x_i` is a previous output `Variable`,
              an input `Variable` (of :math:`F(a_1, a_2, \dots)`),
              a `Constant`, a scalar, o a list of them
              but not an `Operation` object.

        The list of assignments is given as a `MemoizationTable`,
        and it is obtained by re-evaluating the current operation
        under the `Memoization` context.

        The argument ``id_prefix`` is the string prefix used to name
        intermediate variables and the argument ``decompose_sec_ops``
        determines whether to use the context `SecondaryOperationEvaluation`.
        In other words, if ``decompose_sec_ops`` is ``True``,
        `SecondaryOperation` objects are not allowed in the list of
        assignments and they are replaced by their
        decomposition into  `PrimaryOperation` objects.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.secondaryop import BvMaj
            >>> expr = 1 + BvMaj(2, Variable("a_1", 8), 3 | Variable("a_2", 8))
            >>> expr.memoization_table()  # doctest: +NORMALIZE_WHITESPACE
            MemoizationTable([(x0, a_2 | 0x03), (x1, 0x02 & a_1), (x2, 0x02 & x0),
            (x3, x1 | x2), (x4, a_1 & x0), (x5, x3 | x4), (x6, x5 + 0x01), (x7, Id(x6))])
            >>> expr.memoization_table(decompose_sec_ops=False)
            MemoizationTable([(x0, a_2 | 0x03), (x1, BvMaj(0x02, a_1, x0)), (x2, x1 + 0x01), (x3, Id(x2))])

        .. Implementation details:

            To use this method not only for decomposing secondary operations
            but also for printing complex bit-vector expressions:

             - This method is added to Operation and not to SecondaryOperation,
               to support (1 + BvMaj()).decompose() and printing any complex operation.

             - SSA is not returned, just the table (input/output vars not needed)

             - Constant and scalar inputs are supported (for printing)

        """
        table = context.MemoizationTable(id_prefix=id_prefix)
        with context.Memoization(table):
            output = self.doit(eval_sec_ops=decompose_sec_ops)
        last_assignment = BvIdentity(output, evaluate=False)
        assert not table.contain_op(last_assignment)
        table.add_op(last_assignment)
        return table


class PrimaryOperation(Operation):
    """Represent the primary bit-vector operations.

    The primary bit-vector operations are those *basic*
    operations that are included in the  bit-vector theory of the
    `SMT_LIBv2 <http://smtlib.cs.uiowa.edu/theories-FixedSizeBitVectors.shtml>`_
    format.

    The primary operations are `BvAnd`, `BvOr`, `BvXor`, `BvComp`, `BvUlt`,
    `BvUle`, `BvUgt`, `BvUge`, `BvShl`, `BvLshr`, `RotateLeft`, `RotateRight`,
    `Concat`, `BvAdd`, `BvSub`, `BvMul`, `BvUdiv`, `BvUrem`, `BvNeg`, `BvNot`,
    `Extract`, `Ite` and `BvIdentity`.

    This class is not meant to be instantiated but to provide a base
    class to define primary operators.
    """

    @classmethod
    def class_key(cls):
        return 3, 0, cls.__name__


class SecondaryOperation(Operation):
    """Represent secondary bit-vector operations.

    Secondary bit-vector operations are those bit-vector operations
    that are not primary operations (see `PrimaryOperation`).
    Secondary operations must be defined in terms of primary operations.

    By default, secondary operations are fully evaluated (`Operation.eval`
    is used) if all the operands are scalar or `Constant` objects (see also
    `context.SecondaryOperationEvaluation`). On the other hand, `pre_eval`
    is always called in the evaluation (even with symbolic inputs).

    This class is not meant to be instantiated but to provide a base
    class to define secondary operators.
    """

    @classmethod
    def pre_eval(cls, *args):
        """Evaluate the operator before `Operation.eval`.

        This is an internal method that assumes the list ``args`` has been parsed.
        """
        return None

    @classmethod
    def class_key(cls):
        return 4, 0, cls.__name__

    def formula_size(self):
        return super().doit().formula_size()


class PartialOperation(object):
    """Represent bit-vector operations with fixed operands.

    Given a base operator :math:`(x, y) \mapsto  f(x, y)`,
    a partial operator is a function obtained by fixing some
    of the inputs to constants, e.g., :math:`x \mapsto f(x, y=0)`.

    This class is not meant to be instantiated but to provide a base
    class for bit-vector operations with fixed operands generated
    through `make_partial_operation`.
    `PartialOperation` subclasses generated by `make_partial_operation`
    are also subclasses of `PrimaryOperation` or `SecondaryOperation`
    depending on the type of the base operator.

    Attributes:
        base_op: a subclass of `Operation` denoting the base operator.
        fixed_args: a `tuple` with the same length as the number of operands
            of the base function containing ``None``, scalar or `Constant` elements.
            If ``fixed_args[i]`` is ``None``, the i-th operand is not fixed;
            otherwise, the i-th operand is replaced with ``fixed_args[i]``.

    """

    @classmethod
    def _get_base_op_args(cls, *args):
        full_args = []
        index = 0
        for f_arg in cls.fixed_args:
            if f_arg is None:
                full_args.append(args[index])
                index += 1
            else:
                full_args.append(f_arg)
        return full_args

    @classmethod
    def condition(cls, *args):
        return cls.base_op.condition(*cls._get_base_op_args(*args))

    @classmethod
    def output_width(cls, *args):
        return cls.base_op.output_width(*cls._get_base_op_args(*args))

    @classmethod
    def pre_eval(cls, *args):
        if issubclass(cls.base_op, SecondaryOperation) and hasattr(cls.base_op, "pre_eval"):
            return cls.base_op.pre_eval(*cls._get_base_op_args(*args))

    @classmethod
    def eval(cls, *args):
        base_op_eval = cls.base_op.eval(*cls._get_base_op_args(*args))

        if isinstance(base_op_eval, cls.base_op):
            # trying to convert base_op_eval to a PartialOp object
            non_fixed_args = []
            for b_arg, f_arg in zip(base_op_eval.args, cls.fixed_args):
                if f_arg is None:
                    assert b_arg is not None
                    non_fixed_args.append(b_arg)
                else:
                    if b_arg != f_arg:
                        return base_op_eval  # conversion failed
            # similar as super().__new__ in Operation.__new__
            w = base_op_eval.width
            return core.Term.__new__(cls, *non_fixed_args, width=w)  # conversion succeed
        else:
            return base_op_eval

    def _get_base_op_expr(self):
        return self.__class__.base_op(*self.__class__._get_base_op_args(*self.args))

    def formula_size(self):
        return self._get_base_op_expr().formula_size()

    def memoization_table(self, id_prefix="x", decompose_sec_ops=True):
        return self._get_base_op_expr().memoization_table(
            id_prefix=id_prefix, decompose_sec_ops=decompose_sec_ops)


@functools.lru_cache(maxsize=None)  # temporary hack to create singletons
def make_partial_operation(base_op, fixed_args):
    """Return a new `PartialOperation` subclass with the given base operator and fixed arguments.

    The argument ``fixed_args`` is a `tuple`, with the same length as
    the number of operands of the base operator, containing ``None``,
    scalar or `Constant` elements. If ``fixed_args[i]`` is ``None``,
    the i-th operand is not fixed; otherwise, the i-th operand is
    replaced with ``fixed_args[i]``.

    The resulting class is also a subclass of `PrimaryOperation` or
    `SecondaryOperation`, depending on the type of the base operator.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvAdd, Extract, make_partial_operation
        >>> from cascada.bitvector.secondaryop import BvMaj
        >>> BvAddCte = make_partial_operation(BvAdd, tuple([None, Constant(1, 4)]))
        >>> BvAddCte.__name__
        'BvAdd_{·, 0x1}'
        >>> expr = BvAddCte(Variable("a", 4))
        >>> expr
        a + 0x1
        >>> expr.vrepr()
        "make_partial_operation(BvAdd, (None, Constant(0b0001, width=4)))(Variable('a', width=4))"
        >>> ExtractLSB = make_partial_operation(Extract, tuple([None, 0, 0]))
        >>> ExtractLSB.__name__
        'Extract_{·, 0, 0}'
        >>> ExtractLSB(Extract(Variable("a", 4), 2, 0))  # result is simplified
        a[0]
        >>> BvCteMaj = make_partial_operation(BvMaj, tuple([None, None, Constant(1, 4)]))
        >>> BvCteMaj.__name__
        'BvMaj_{·, ·, 0x1}'
        >>> BvCteMaj(Variable("a", 4), Variable("a", 4))
        a
        >>> expr = BvCteMaj(Variable("a", 4), Variable("b", 4))
        >>> expr
        BvMaj_{·, ·, 0x1}(a, b)
        >>> expr.doit()
        (a & b) | (a & 0x1) | (b & 0x1)
        >>> expr.memoization_table()
        MemoizationTable([(x0, a & b), (x1, a & 0x1), (x2, x0 | x1), (x3, b & 0x1), (x4, x2 | x3), (x5, Id(x4))])
        >>> BvCteMaj_v2 = make_partial_operation(BvCteMaj, tuple([Constant(2, 4), None]))
        >>> BvCteMaj_v2.__name__
        'BvMaj_{0x2, ·, 0x1}'
        >>> BvCteMaj_v2(Variable("a", 4))
        BvMaj_{0x2, ·, 0x1}(a)

    """
    assert issubclass(base_op, Operation)

    assert isinstance(fixed_args, tuple)
    assert len(fixed_args) == sum(base_op.arity)
    assert all(arg is None or isinstance(arg, (int, core.Constant)) for arg in fixed_args)

    # at least one None and one non-None in fixed_args
    assert None in fixed_args
    assert any(arg is not None for arg in fixed_args)

    if issubclass(base_op, PartialOperation):
        assert len(fixed_args) == len([a for a in base_op.fixed_args if a is None])
        combined_fixed_args = list(base_op.fixed_args)
        counter_None = 0
        for i in range(len(combined_fixed_args)):
            if combined_fixed_args[i] is None:
                combined_fixed_args[i] = fixed_args[counter_None]
                counter_None += 1
        assert counter_None == len(fixed_args)
        return make_partial_operation(base_op.base_op, tuple(combined_fixed_args))

    if hasattr(base_op, "operand_types"):
        operand_types = base_op.operand_types
    else:
        operand_types = [core.Term for _ in range(len(fixed_args))]

    num_terms_fixed = 0
    num_scalars_fixed = 0
    free_operand_types = []
    fixed_args_str = []
    for arg, type_arg in zip(fixed_args, operand_types):
        if arg is None:
            free_operand_types.append(type_arg)
            fixed_args_str.append("·")
            continue
        assert isinstance(arg, type_arg)
        if type_arg == int:
            num_scalars_fixed += 1
        elif isinstance(arg, core.Term):
            num_terms_fixed += 1
        else:
            assert False
        fixed_args_str.append(str(arg))

    if issubclass(base_op, PrimaryOperation):
        parent_class = PrimaryOperation
    else:
        assert issubclass(base_op, SecondaryOperation)
        parent_class = SecondaryOperation

    _base_op = base_op
    _fixed_args = fixed_args

    _arity = [base_op.arity[0] - num_terms_fixed, base_op.arity[1] - num_scalars_fixed]
    assert _arity[0] >= 1
    _is_symmetric = base_op.is_symmetric
    _is_simple = base_op.is_simple and sum(_arity) > 1
    _operand_types = free_operand_types

    # avoid subclassing base_op (may introduce side effects)
    class MyPartialOperation(PartialOperation, parent_class):
        base_op = _base_op
        fixed_args = _fixed_args

        arity = _arity
        is_symmetric = _is_symmetric
        is_simple = _is_simple
        operand_types = _operand_types

    if hasattr(base_op, "alt_name"):
        MyPartialOperation.alt_name = f"{base_op.alt_name}_{{{', '.join(fixed_args_str)}}}"

    MyPartialOperation.__name__ = f"{base_op.__name__}_{{{', '.join(fixed_args_str)}}}"

    assert issubclass(parent_class, PrimaryOperation) == \
           issubclass(MyPartialOperation, PrimaryOperation)
    assert issubclass(parent_class, SecondaryOperation) == \
           issubclass(MyPartialOperation, SecondaryOperation)

    return MyPartialOperation


# Bitwise operators

class BvNot(PrimaryOperation):
    """Bitwise negation operation.

    It overrides the operator ~. See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvNot
        >>> BvNot(Constant(0b1010101, 7))
        0b0101010
        >>> ~Constant(0b1010101, 7)
        0b0101010
        >>> ~Variable("x", 8)
        ~x

    """

    arity = [1, 0]
    is_symmetric = False
    unary_symbol = "~"

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, x):
        def doit(x, width):
            """NOT operation when the operand is int."""
            return ~x % (2 ** width)

        if isinstance(x, core.Constant):
            return core.Constant(doit(int(x), x.width), x.width)
        elif isinstance(x, BvNot):
            return x.args[0]
        # # De Morgan's laws (disabled, all op equal precedence)
        # if isinstance(x, BvAnd):
        #     return BvOr(BvNot(x.args[0]), BvNot(x.args[1]))
        # elif isinstance(x, BvOr):
        #     return BvAnd(BvNot(x.args[0]), BvNot(x.args[1]))


class BvAnd(PrimaryOperation):
    """Bitwise AND (logical conjunction) operation.

    It overrides the operator & and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvAnd
        >>> BvAnd(Constant(5, 8), Constant(3, 8))
        0x01
        >>> BvAnd(Constant(5, 8), 3)
        0x01
        >>> Constant(5, 8) & 3
        0x01
        >>> Variable("x", 8) & Variable("y", 8)
        x & y

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True
    infix_symbol = "&"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """AND operation when both operands are int."""
            return x & y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
        elif x == zero or y == zero:
            return zero
        elif x == allones:
            return y
        elif y == allones:
            return x
        elif x == y:
            return x
        elif x == BvNot(y):
            return zero
        elif isinstance(x, BvOr) and y in x.args:  # (... | y) & y = y
            return y
        elif isinstance(y, BvOr) and x in y.args:  # x & (x | ...) = x
            return x

    def _simplify(self, *args, **kwargs):
        # simplify if x and x (or x and ~x) appear in a flattened conjunction
        compatible_terms = [
            lambda x: x,
            lambda x: BvNot(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)


class BvOr(PrimaryOperation):
    """Bitwise OR (logical disjunction) operation.

    It overrides the operator | and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvOr
        >>> BvOr(Constant(5, 8), Constant(3, 8))
        0x07
        >>> BvOr(Constant(5, 8), 3)
        0x07
        >>> Constant(5, 8) | 3
        0x07
        >>> Variable("x", 8) | Variable("y", 8)
        x | y

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True
    infix_symbol = "|"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """OR operation when both operands are int."""
            return x | y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
        elif x == allones or y == allones:
            return allones
        elif x == zero:
            return y
        elif y == zero:
            return x
        elif x == y:
            return x
        elif x == BvNot(y):
            return allones
        elif isinstance(x, BvAnd) and y in x.args:  # (... & y) | y = y
            return y
        elif isinstance(y, BvAnd) and x in y.args:  # x | (x & ...) = x
            return x

    def _simplify(self, *args, **kwargs):
        compatible_terms = [
            lambda x: x,
            lambda x: BvNot(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)


class BvXor(PrimaryOperation):
    """Bitwise XOR (exclusive-or) operation.

    It overrides the operator ^ and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvXor
        >>> BvXor(Constant(5, 8), Constant(3, 8))
        0x06
        >>> BvXor(Constant(5, 8), 3)
        0x06
        >>> Constant(5, 8) ^ 3
        0x06
        >>> Variable("x", 8) ^ Variable("y", 8)
        x ^ y

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True
    infix_symbol = "^"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """XOR operation when both operands are int."""
            return x ^ y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
        elif x == zero:
            return y
        elif y == zero:
            return x
        elif x == allones:
            return BvNot(y)
        elif y == allones:
            return BvNot(x)
        elif x == y:
            return zero
        elif x == BvNot(y):
            return allones

    def _simplify(self, *args, **kwargs):
        compatible_terms = [
            lambda x: x,
            lambda x: BvNot(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)

# Relational operators


class BvComp(PrimaryOperation):
    """Equality operator.

    Provides Automatic Constant Conversion. See `PrimaryOperation` for more
    information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvComp
        >>> BvComp(Constant(1, 8), Constant(2, 8))
        0b0
        >>> BvComp(Constant(1, 8), 2)
        0b0
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> BvComp(Constant(1, 8), Variable("y", 8))
        0x01 == y
        >>> bool(BvComp(x + y, y + x))
        True

    The operator == is used for exact structural equality testing and
    it returns either True or False. On the other hand, BvComp
    performs symbolic equality testing and it leaves the relation unevaluated
    if it cannot prove the objects are equal (or unequal).

        >>> x == y
        False
        >>> BvComp(x, y)  # symbolic equality
        x == y

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True
    infix_symbol = "=="

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if x == y:
            return one
        elif isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val == y.val else zero
        elif getattr(x, "is_symmetric", False) and  getattr(y, "is_symmetric", False):
            eval_sec_ops = context.SecondaryOperationEvaluation.current_context
            sorted_x = x.doit(eval_sec_ops=eval_sec_ops, sort_symmetric_args=True)
            sorted_y = y.doit(eval_sec_ops=eval_sec_ops, sort_symmetric_args=True)
            if sorted_x == sorted_y:
                return one


class BvUlt(PrimaryOperation):
    """Unsigned less than operator.

    It overrides < and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvUlt
        >>> BvUlt(Constant(1, 8), Constant(2, 8))
        0b1
        >>> BvUlt(Constant(1, 8), 2)
        0b1
        >>> Constant(1, 8) < 2
        0b1
        >>> Constant(1, 8) < Variable("y", 8)
        0x01 < y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "<"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val < y.val else zero


class BvUle(PrimaryOperation):
    """Unsigned less than or equal operator.

    It overrides <= and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvUle
        >>> BvUle(Constant(2, 8), Constant(2, 8))
        0b1
        >>> BvUle(Constant(2, 8), 2)
        0b1
        >>> Constant(2, 8) <= 2
        0b1
        >>> Constant(2, 8) <= Variable("y", 8)
        0x02 <= y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "<="

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val <= y.val else zero


class BvUgt(PrimaryOperation):
    """Unsigned greater than operator.

    It overrides > and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvUgt
        >>> BvUgt(Constant(1, 8), Constant(2, 8))
        0b0
        >>> BvUgt(Constant(1, 8), 2)
        0b0
        >>> Constant(1, 8) > 2
        0b0
        >>> Constant(1, 8) > Variable("y", 8)
        0x01 > y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = ">"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val > y.val else zero


class BvUge(PrimaryOperation):
    """Unsigned greater than or equal operator.

    It overrides >= and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvUgt
        >>> BvUge(Constant(2, 8), Constant(2, 8))
        0b1
        >>> BvUge(Constant(2, 8), 2)
        0b1
        >>> Constant(2, 8) >= 2
        0b1
        >>> Constant(2, 8) >= Variable("y", 8)
        0x02 >= y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = ">="

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val >= y.val else zero


# Shifts operators

class BvShl(PrimaryOperation):
    """Shift left operation.

    It overrides << and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvShl
        >>> BvShl(Constant(0b10001, 5), Constant(1, 5))
        0b00010
        >>> BvShl(Constant(0b10001, 5), 1)
        0b00010
        >>> Constant(0b10001, 5) << 1
        0b00010
        >>> Variable("x", 8) << Variable("y", 8)
        x << y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "<<"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Shift left operation when both operands are int."""
            return (x << y) % (2 ** width)

        zero = core.Constant(0, x.width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)
        # y >= x.width doesn't evaluate even if y is a Constant
        elif isinstance(y, core.Constant) and int(y) >= x.width:
            return zero
        elif x == zero or y == zero:
            return x
        elif isinstance(x, BvShl) and isinstance(x.args[1], core.Constant) \
                and isinstance(y, core.Constant):
            # prevent out of bound
            r = min(x.args[0].width, int(x.args[1]) + int(y))
            return BvShl(x.args[0], core.Constant(r, x.width))


class BvLshr(PrimaryOperation):
    """Logical right shift operation.

    It overrides >> and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvLshr
        >>> BvLshr(Constant(0b10001, 5), Constant(1, 5))
        0b01000
        >>> BvLshr(Constant(0b10001, 5), 1)
        0b01000
        >>> Constant(0b10001, 5) >> 1
        0b01000
        >>> Variable("x", 8) >> Variable("y", 8)
        x >> y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = ">>"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """Logical right shift operation when both operands are int."""
            return x >> y

        zero = core.Constant(0, x.width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
        elif isinstance(y, core.Constant) and int(y) >= x.width:
            return zero
        elif x == zero or y == zero:
            return x
        elif isinstance(x, BvLshr) and isinstance(x.args[1], core.Constant) \
                and isinstance(y, core.Constant):
            r = min(x.args[0].width, int(x.args[1]) + int(y))
            return BvLshr(x.args[0], core.Constant(r, x.width))


class RotateLeft(PrimaryOperation):
    """Circular left rotation operation.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import RotateLeft
        >>> RotateLeft(Constant(150, 8), 2)
        0x5a
        >>> RotateLeft(Variable("x", 8), 2)
        x <<< 2

    """

    arity = [1, 1]
    is_symmetric = False
    infix_symbol = "<<<"
    operand_types = [core.Term, int]

    @classmethod
    def condition(cls, x, r):
        return x.width > r >= 0

    @classmethod
    def output_width(cls, x, r):
        return x.width

    @classmethod
    def eval(cls, x, r):
        def doit(val, r, width):
            """Left cyclic rotation operation when both operands are int."""
            mask = 2 ** width - 1
            r = r % width
            # source: hacker's delight 2-15
            # equivalently ((val << r) | (val >> (width - r))) & mask
            return ((val << r) & mask) | ((val & mask) >> (width - r))

        if isinstance(x, core.Constant):
            return core.Constant(doit(int(x), r, x.width), x.width)
        elif r == 0:
            return x
        elif isinstance(x, RotateLeft):
            return RotateLeft(x.args[0], (x.args[1] + r) % x.args[0].width)
        elif isinstance(x, RotateRight):
            return RotateRight(x.args[0], (x.args[1] - r) % x.args[0].width)


class RotateRight(PrimaryOperation):
    """Circular right rotation operation.

    It provides Automatic Constant Conversion. See `PrimaryOperation` for more
    information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import RotateRight
        >>> RotateRight(Constant(150, 8), 3)
        0xd2
        >>> RotateRight(Variable("x", 8), 3)
        x >>> 3

    """

    arity = [1, 1]
    is_symmetric = False
    infix_symbol = ">>>"
    operand_types = [core.Term, int]

    @classmethod
    def condition(cls, x, r):
        return x.width > r >= 0

    @classmethod
    def output_width(cls, x, r):
        return x.width

    @classmethod
    def eval(cls, x, r):
        def doit(val, r, width):
            """Right cyclic rotation operation when both operands are int."""
            mask = 2 ** width - 1
            r = r % width
            # source: hacker's delight 2-15
            # equivalently ((val >> r) | (val << (width - r))) & mask
            return ((val & mask) >> r) | (val << (width - r) & mask)

        if isinstance(x, core.Constant):
            return core.Constant(doit(int(x), r, x.width), x.width)
        elif r == 0:
            return x
        elif isinstance(x, RotateRight):
            return RotateRight(x.args[0], (x.args[1] + r) % x.args[0].width)
        elif isinstance(x, RotateLeft):
            return RotateLeft(x.args[0], (x.args[1] - r) % x.args[0].width)


# Others

class Ite(PrimaryOperation):
    """If-then-else operator.

    ``Ite(b, x, y)`` returns ``x`` if ``b == 0b1`` and ``y`` otherwise.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import Ite
        >>> Ite(Constant(0, 1), Constant(0b11, 2), Constant(0b00, 2))
        0b00
        >>> Ite(Constant(1, 1), Constant(0x1, 4), Constant(0x0, 4))
        0x1

    """

    arity = [3, 0]
    is_symmetric = False

    @classmethod
    def condition(cls, b, x, y):
        return b.width == 1 and x.width == y.width

    @classmethod
    def output_width(cls, b, x, y):
        return x.width

    @classmethod
    def eval(cls, b, x, y):
        if b == core.Constant(1, 1):
            return x
        elif b == core.Constant(0, 1):
            return y


class Extract(PrimaryOperation):
    """Extraction of bits.

    ``Extract(t, i, j)`` extracts the bits from position ``i`` down
    position ``j`` (end points included, position 0 corresponding
    to the least significant bit).

    It overrides the operation [], that is, ``Extract(t, i, j)``
    is equivalent to ``t[i:j]``.

    Note that the indices can be omitted when they point the most
    significant bit or the least significant bit.
    For example, if ``t`` is a bit-vector of length ``n``,
    then ``t[n-1:j] = t[:j]`` and ``t[i:0] = t[i:]``

    Warning:
        In python, given a list ``l``, ``l[i:j]`` denotes the elements
        from position ``i`` up to (but not included) position ``j``.
        Note that with bit-vectors, the order of the arguments is
        swapped and both end points are included.

        For example, for a given list ``l`` and bit-vector ``t``,
        ``l[0:1] == l[0]`` and ``t[1:0] == (t[0], t[1])``.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import Extract
        >>> Extract(Constant(0b11100, 5), 4, 2,)
        0b111
        >>> Constant(0b11100, 5)[4:2]
        0b111
        >>> Variable("x", 8)[4:2]
        x[4:2]
        >>> Variable("x", 8)[7:0]
        x

    """

    arity = [1, 2]
    is_symmetric = False
    operand_types = [core.Term, int, int]

    @classmethod
    def condition(cls, t, i, j):
        return t.width > i >= j

    @classmethod
    def output_width(cls, t, i, j):
        return i - j + 1

    @classmethod
    def eval(cls, x, i, j):
        def doit(x, i, j):
            """Extract from x[i] down x[j] from a constant x."""
            bin_repr = x.bin()
            prefix, value = bin_repr[:2], bin_repr[2:]
            n = x.width
            value = value[n - 1 - i:n - j]  # e.g.: i=n-1, j=0
            return int(prefix + value, 2)

        if isinstance(x, core.Constant):
            return core.Constant(doit(x, i, j), cls.output_width(x, i, j))
        elif i == x.width - 1 and j == 0:
            return x
        elif isinstance(x, Extract):
            # x[3:1][2] = (x3 x2 x1)[2] = x3 = x[3]
            offset = x.args[2]
            return Extract(x.args[0], i + offset, j + offset)
        elif isinstance(x, Concat):
            if i <= x.args[1].width - 1:
                # 4-bit x, y: concat(x, y)[3:] = y[3:]
                return Extract(x.args[1], i, j)
            elif j >= x.args[1].width:
                # 4-bit x, y: concat(x, y)[:5] = x[:1]
                offset = x.args[1].width
                return Extract(x.args[0], i - offset, j - offset)
        elif isinstance(x, (BvShl, RotateLeft)) and \
                isinstance(x.args[1], (int, core.Constant)) and int(x.args[1]) <= j:
            # (x << 1)[:2] = x[n-2: 1]
            offset = int(x.args[1])
            return Extract(x.args[0], i - offset, j - offset)
        elif isinstance(x, (BvLshr, RotateRight)) and \
                isinstance(x.args[1], (int, core.Constant)) and i < x.width - int(x.args[1]):
            # (x >> 1)[n-3:] = x[n-2: 1]
            offset = int(x.args[1])
            return Extract(x.args[0], i + offset, j + offset)


class Concat(PrimaryOperation):
    """Concatenation operation.

    Given the bit-vectors :math:`(x_{n-1}, \dots, x_0)` and
    :math:`(y_{m-1}, \dots, y_0)`, ``Concat(x, y)`` returns the bit-vector
    :math:`(x_{n-1}, \dots, x_0, y_{m-1}, \dots, y_0)`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import Concat
        >>> Concat(Constant(0x12, 8), Constant(0x345, 12))
        0x12345
        >>> Concat(Variable("x", 8), Variable("y", 8))
        x :: y

    """

    arity = [2, 0]
    is_symmetric = False
    infix_symbol = "::"

    @classmethod
    def output_width(cls, x, y):
        return x.width + y.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """Concatenation when both operands are int."""
            return int(x.bin() + y.bin()[2:], 2)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(x, y), cls.output_width(x, y))
        elif isinstance(x, core.Constant) and isinstance(y, Concat) and \
                isinstance(y.args[0], core.Constant):
            return Concat(Concat(x, y.args[0]), y.args[1])
        elif isinstance(y, core.Constant) and isinstance(x, Concat) and \
                isinstance(x.args[1], core.Constant):
            return Concat(x.args[0], Concat(x.args[1], y))
        elif isinstance(x, Extract) and isinstance(y, Extract):
            # x[5:4] concat x[3:2] = x[5:2]
            if x.args[0] == y.args[0] and x.args[2] == y.args[1] + 1:
                return Extract(x.args[0], x.args[1], y.args[2])


# Arithmetic operators

class BvNeg(PrimaryOperation):
    """Unary minus operation.

    It overrides the unary operator -. See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvNeg
        >>> BvNeg(Constant(1, 8))
        0xff
        >>> -Constant(1, 8)
        0xff
        >>> BvNeg(Variable("x", 8))
        -x

    """

    arity = [1, 0]
    is_symmetric = False
    unary_symbol = "-"

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, x):
        def doit(x, width):
            """Unary minus operation when the operand is int."""
            return ((2 ** width) - x) % (2 ** width)

        if isinstance(x, core.Constant):
            return core.Constant(doit(int(x), x.width), x.width)
        elif isinstance(x, BvNeg):
            return x.args[0]
        # # disabled (all op equal precedence)
        # elif isinstance(x, BvAdd):
        #     return BvAdd(BvNeg(x.args[0]), BvNeg(x.args[1]))
        # elif isinstance(x, (BvMul, BvDiv, BvMod)):
        #     return x.func(BvNeg(x.args[0]), x.args[1])


class BvAdd(PrimaryOperation):
    """Modular addition operation.

    It overrides the operator + and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvAdd
        >>> BvAdd(Constant(1, 8), Constant(2, 8))
        0x03
        >>> BvAdd(Constant(1, 8), 2)
        0x03
        >>> Constant(1, 8) + 2
        0x03
        >>> Variable("x", 8) + Variable("y", 8)
        x + y

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True
    infix_symbol = "+"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Modular addition when both operands are integers."""
            return (x + y) % (2 ** width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)

        zero = core.Constant(0, x.width)

        if x == zero:
            return y
        elif y == zero:
            return x
        elif x == BvNeg(y):
            return zero
        elif isinstance(x, BvSub):  # (x0 - x1=y) + y
            if x.args[1] == y:
                return x.args[0]
        elif isinstance(y, BvSub):  # x + (y0 - y1=x)
            if y.args[1] == x:
                return y.args[0]
        elif isinstance(x, BvNeg):  # (-x) + y = y - x
            return y - x.args[0]
        elif isinstance(y, BvNeg):  # x + (-y) = x - y
            return x - y.args[0]

    def _simplify(self, *args, **kwargs):
        # simplify if x and BvNeg(x) appear in a flattened addition
        compatible_terms = [
            lambda x: BvNeg(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)


class BvSub(PrimaryOperation):
    """Modular subtraction operation.

    It overrides the operator - and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvSub
        >>> BvSub(Constant(1, 8), Constant(2, 8))
        0xff
        >>> BvSub(Constant(1, 8), 2)
        0xff
        >>> Constant(1, 8) - 2
        0xff
        >>> Variable("x", 8) - Variable("y", 8)
        x - y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "-"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Modular subtraction when both operands are integers."""
            return (x - y) % (2 ** width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)

        zero = core.Constant(0, x.width)

        if x == zero:
            return BvNeg(y)
        elif y == zero:
            return x
        elif x == y:
            return zero
        elif isinstance(x, BvAdd):  # (x0 + x1) - y, y in [x0, x1]
            if x.args[0] == y:
                return x.args[1]
            elif x.args[1] == y:
                return x.args[0]
        elif isinstance(y, BvAdd):  # x - (y0 + y1), x in [y0, y1]
            if y.args[0] == x:
                return BvNeg(y.args[1])
            elif y.args[1] == x:
                return BvNeg(y.args[0])
        elif isinstance(y, BvNeg):  # x - (-y) = x + y
            return x + y.args[0]


class BvMul(PrimaryOperation):
    """Modular multiplication operation.

    It overrides the operator * and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvMul
        >>> BvMul(Constant(4, 8), Constant(3, 8))
        0x0c
        >>> BvMul(Constant(4, 8), 3)
        0x0c
        >>> Constant(4, 8) * 3
        0x0c
        >>> Variable("x", 8) * Variable("y", 8)
        x * y

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True
    infix_symbol = "*"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Modular multiplication when both operands are int."""
            return (x * y) % (2 ** width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)

        zero = core.Constant(0, x.width)
        one = core.Constant(1, x.width)

        if x == zero or y == zero:
            return zero
        elif x == one:
            return y
        elif y == one:
            return x


class BvUdiv(PrimaryOperation):
    """Unsigned and truncated division operation.

    It overrides the operator / and provides Automatic Constant Conversion.
    See `PrimaryOperation` for more information.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvUdiv
        >>> BvUdiv(Constant(0x0c, 8), Constant(3, 8))
        0x04
        >>> BvUdiv(Constant(0x0c, 8), 3)
        0x04
        >>> Constant(0x0c, 8) / 3
        0x04
        >>> Variable("x", 8) / Variable("y", 8)
        x / y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "/"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """Division operation (truncated) when both operands are int."""
            assert y != 0
            return x // y

        zero = core.Constant(0, x.width)
        one = core.Constant(1, x.width)

        assert y != zero

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
        elif x == y:
            return one
        elif x == zero:
            return zero
        elif y == one:
            return x


class BvUrem(PrimaryOperation):
    """Unsigned remainder (modulus) operation.

    Usage:

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvUrem
        >>> BvUrem(Constant(0x0d, 8), Constant(3, 8))
        0x01
        >>> BvUrem(Constant(0x0d, 8), 3)
        0x01
        >>> Constant(0x0d, 8) % 3
        0x01
        >>> Variable("x", 8) % Variable("y", 8)
        x % y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "%"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """Remainder operation when both operands are int."""
            assert y != 0
            return x % y

        zero = core.Constant(0, x.width)
        one = core.Constant(1, x.width)

        assert y != zero

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
        elif x == y or x == zero or y == one:
            return zero


class BvIdentity(PrimaryOperation):
    """The identity operation.

    Return the same value when the input is constant and
    a `BvIdentity` object when the input is symbolic:

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvIdentity
        >>> BvIdentity(Constant(0x1, 4))
        0x1
        >>> BvIdentity(Variable("x", 8))
        Id(x)

    """
    # 'Identity' is already taken by SymPy

    arity = [1, 0]
    is_symmetric = False
    alt_name = "Id"

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, x):
        if isinstance(x, core.Constant):
            return x


# Shortcuts

def zero_extend(x, i):
    """Extend with zeroes preserving the unsigned value.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import zero_extend
        >>> zero_extend(Constant(0x12, 8), 4)
        0x012
        >>> zero_extend(Variable("x", 8), 4)
        0x0 :: x

    """
    assert isinstance(x, core.Term)
    assert isinstance(i, int) and i >= 0

    if i == 0:
        output = x
    else:
        output = Concat(core.Constant(0, i), x)

    assert x.width + i

    return output


def repeat(x, i):
    """Concatenate a bit-vector with itself a given number of times.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import repeat
        >>> repeat(Constant(0x1, 4), 4)
        0x1111
        >>> repeat(Variable("x", 8), 4)
        x :: x :: x :: x

    """
    assert isinstance(x, core.Term)
    assert isinstance(i, int) and i >= 1

    if i == 1:
        output = x
    else:
        output = functools.reduce(Concat, itertools.repeat(x, i))

    assert output.width == i * x.width

    return output
