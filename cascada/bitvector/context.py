"""Provide context managers to modify the creation and evaluation of bit-vector expressions."""
import collections
import contextlib

import bidict

from cascada.bitvector import core


class StatefulContext(contextlib.AbstractContextManager):
    """Base class for context managers with history."""

    current_context = None

    def __init__(self, new_context):
        """Initialize the context."""
        self.new_context = new_context

    def __enter__(self):
        self.previous_context = type(self).current_context
        type(self).current_context = self.new_context
        # return self.new_context

    def __exit__(self, *args):
        type(self).current_context = self.previous_context


class Cache(StatefulContext):
    """Control the Cache context.

    Control whether or not the cache is used operating with bit-vectors.
    By default, the cache is enabled.

    Note that the Cache context cannot be enabled when the
    `Simplification` or `PrimaryOperationEvaluation` context are disabled,
    or when `SecondaryOperationEvaluation` is enabled.
    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is True:
            assert Simplification.current_context is True
            assert PrimaryOperationEvaluation.current_context is True
            assert SecondaryOperationEvaluation.current_context is False
            assert Memoization.current_context is False
        super().__enter__()


class Simplification(StatefulContext):
    """Control the Simplification context.

    Control whether or not bit-vector expressions are automatically simplified.
    By default, automatic simplification is enabled.

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.bitvector.context import Simplification
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> (x | y) | x
        x | y
        >>> with Simplification(False):
        ...     expr = (x | y) | x
        >>> expr
        x | y | x

    When the Simplification context is disabled, the `Cache` context is
    also disabled.

    Note:
        Disabling `Simplification` and `Validation` speeds up
        non-symbolic computations with bit-vectors.
    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is False:
            self.cache_context = Cache(False)
            self.cache_context.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        if self.new_context is False:
            self.cache_context.__exit__()
        super().__exit__()


class PrimaryOperationEvaluation(StatefulContext):
    """Control the PrimaryOperationEvaluation context.

    Control whether `PrimaryOperation` objects with symbolic inputs
    are evaluated  (True by default).

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.context import PrimaryOperationEvaluation
        >>> Constant(1, 8) - Constant(1, 8) + Variable("x", 8) - Variable("x", 8)
        0x00
        >>> with PrimaryOperationEvaluation(False):
        ...     expr = Constant(1, 8) - Constant(1, 8) + Variable("x", 8) - Variable("x", 8)
        >>> expr
        (0x00 + x) - x
        >>> expr.doit()
        0x00

    When the `PrimaryOperationEvaluation` context is disabled,
    the `Simplification` and `Cache` contexts are also disabled.
    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is False:
            self.simplify_context = Simplification(False)
            self.simplify_context.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        if self.new_context is False:
            self.simplify_context.__exit__()
        super().__exit__()


class SecondaryOperationEvaluation(StatefulContext):
    """Control the SecondaryOperationEvaluation context.

    Control whether `SecondaryOperation` objects with symbolic inputs
    are evaluated (False by default).

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.context import SecondaryOperationEvaluation
        >>> from cascada.bitvector.secondaryop import BvMaj
        >>> BvMaj(Constant(1, 8), Constant(1, 8), Constant(2, 8))
        0x01
        >>> expr = BvMaj(Variable("x", 8), Variable("y", 8), Variable("z", 8))
        >>> expr
        BvMaj(x, y, z)
        >>> expr.doit()
        (x & y) | (x & z) | (y & z)
        >>> with SecondaryOperationEvaluation(True):
        ...     BvMaj(Variable("x", 8), Variable("y", 8), Variable("z", 8))
        (x & y) | (x & z) | (y & z)

    When the `SecondaryOperation` context is enabled, the `Cache` context is disabled.
    """

    current_context = False

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is True:
            self.cache_context = Cache(False)
            self.cache_context.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        if self.new_context is True:
            self.cache_context.__exit__()
        super().__exit__()


class Validation(StatefulContext):
    """Control the Validation context.

    Control whether or not arguments of bit-vector operators are validated
    (e.g., the integer value when creating a `Constant` fits for the width given).
    By default, validation of arguments is enabled.

    Note that when it is disabled,  Automatic Constant Conversion is no longer
    available (see `Operation`).

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.context import Validation
        >>> Constant(1, 8) + 1
        0x02
        >>> with Validation(False):
        ...     Constant(1, 5) + 2
        Traceback (most recent call last):
         ...
        AttributeError: 'int' object has no attribute 'width'

    Note:
        Disabling `Simplification` and `Validation` speeds up
        non-symbolic computations with bit-vectors.
    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)


class Memoization(StatefulContext):
    """Control the Memoization context.

    Control whether or not bit-vector operations are evaluated in the
    *memoization mode*. By default, it is disabled.

    In the memoization mode, the result of each bit-vector operation is
    stored in a table (with a unique identifier). When the same inputs
    occurs again, the result is retrieved from the table. See also
    `Memoization <https://en.wikipedia.org/wiki/Memoization>`_.

    Note that in the memoization mode, bit-vector operations don't return
    the actual values but their identifiers in the memoization table.
    The actual values can be obtained from the `MemoizationTable`.

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.bitvector.context import Memoization, MemoizationTable
        >>> x, y, z = Variable("x", 8), Variable("y", 8), Variable("z", 8),
        >>> ~((x + y) ^ ((z + 1) & y))
        ~((x + y) ^ ((z + 0x01) & y))
        >>> non_memoized_expr = (z + 1)
        >>> lut = MemoizationTable()
        >>> with Memoization(lut):
        ...     expr = ~((x + y) ^ (non_memoized_expr & y))
        >>> expr
        x3
        >>> lut
        MemoizationTable([(x0, x + y), (x1, (z + 0x01) & y), (x2, x0 ^ x1), (x3, ~x2)])

    The Memoization context is useful to efficiently compute large symbolic
    expressions since the identifiers are used instead of the full expressions.

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.bitvector.context import Memoization, MemoizationTable
        >>> x = Variable("x", 8)
        >>> expr = x
        >>> for i in range(3): expr += expr
        >>> expr
        x + x + x + x + x + x + x + x
        >>> lut = MemoizationTable()
        >>> with Memoization(lut):
        ...     expr = x
        ...     for i in range(3): expr += expr
        >>> expr
        x2
        >>> lut  # doctest: +NORMALIZE_WHITESPACE
        MemoizationTable([(x0, x + x), (x1, x0 + x0), (x2, x1 + x1)])

    When the Memoization context is enabled, the `Simplification` and `Cache`
    contexts are disabled.
    """

    current_context = None

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context is None or isinstance(new_context, MemoizationTable)
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is not None:
            self.simplify_context = Simplification(False)
            self.simplify_context.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        if self.new_context is not None:
            self.simplify_context.__exit__()
        super().__exit__()


class MemoizationTable(collections.abc.MutableMapping):
    """Store bit-vector expressions with unique identifiers.

    The MemoizationTable is a dictionary-like structure
    (implementing the usual methods of a dictionary and
    some additional methods, and remembering the order
    entries were added as collections.Counter)
    used for evaluating bit-vector operations
    in the *memoization mode* (see `Memoization`).

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.bitvector.context import Memoization, MemoizationTable
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> lut = MemoizationTable()
        >>> with Memoization(lut):
        ...     expr = ~(x + y)
        >>> lut
        MemoizationTable([(x0, x + y), (x1, ~x0)])
        >>> lut[Variable("x0", 8)]
        x + y
        >>> lut.get_id(x + y)
        x0
        >>> lut.add_op(Variable("x1", 8) & Variable("z", 8))
        x2
        >>> lut
        MemoizationTable([(x0, x + y), (x1, ~x0), (x2, x1 & z)])
        >>> lut.replace_id(Variable("x0", 8), Variable("x_0", 8))
        >>> lut
        MemoizationTable([(x_0, x + y), (x1, ~x_0), (x2, x1 & z)])
    """

    def __init__(self, id_prefix="x"):
        """Initialize an MemoizationTable."""
        self.table = bidict.OrderedBidict()
        self.counter = 0
        self.id_prefix = id_prefix

    @classmethod
    def from_list_of_assignments(cls, assignments, new_id_prefix=None):
        """Returns a new `MemoizationTable` with the given list of assignments.

        The argument ``assignments`` is a list of pairs (`Variable`, `Operation`)
        where the variable represents the output of the operation.
        """
        from cascada.bitvector.operation import Operation
        for v_i, op_i in assignments:
            assert isinstance(v_i, core.Variable) and isinstance(op_i, Operation)

        my_table = MemoizationTable()
        my_table.counter = len(assignments)

        if new_id_prefix is None:
            first_var = assignments[0][0]
            for i, c in enumerate(first_var.name):
                if c.isdigit():
                    index_first_digit = i
                    break
            else:
                index_first_digit = len(first_var.name)
            my_table.id_prefix = first_var.name[:index_first_digit]
        else:
            my_table.id_prefix = new_id_prefix

        for v_i, op_i in assignments:
            if v_i.name.startswith(my_table.id_prefix) and \
                    v_i.name[len(my_table.id_prefix):].isdigit() and \
                    int(v_i.name[len(my_table.id_prefix):]) > my_table.counter:
                msg = "invalid var name {} due to id_prefix {} and counter {}\n{}".format(
                    v_i.name, my_table.id_prefix, my_table.counter, assignments)
                raise ValueError(msg)

        my_table.table = bidict.OrderedBidict(assignments)

        return my_table

    def __getitem__(self, key):
        return self.table.__getitem__(key)

    def __setitem__(self, key, expr):
        raise AttributeError("use add_op and replace_id instead")

    def __delitem__(self, key):
        assert all(key not in op.atoms() for op in self.table.values())
        return self.table.__delitem__(key)

    def __len__(self):
        return self.table.__len__()

    def __iter__(self):
        return self.table.__iter__()

    def __str__(self):
        return '{}({})'.format(type(self).__name__, list(self.table.items()))

    __repr__ = __str__

    def add_op(self, expr):
        """Add an bit-vector expression and return its identifier."""
        from cascada.bitvector import operation
        assert isinstance(expr, operation.Operation)
        assert not self.contain_op(expr)
        name = "{}{}".format(self.id_prefix, self.counter)
        self.counter += 1
        identifier = core.Variable(name, expr.width)
        self.table[identifier] = expr

        return identifier

    def get_id(self, expr):
        """Return the identifier of a bit-vector expression."""
        return self.table.inv[expr]

    def contain_op(self, expr):
        """Check if the bit-vector expression is stored."""
        return expr in self.table.inv

    def replace_id(self, old_id, new_id):
        """Replace the old identifier by the given new identifier."""
        assert isinstance(old_id, core.Variable)
        assert isinstance(new_id, core.Variable)
        assert old_id in self.table and new_id not in self.table

        table = list(self.table.items())

        for i, (key, op) in enumerate(table):
            if key == old_id:
                new_key = new_id
            else:
                new_key = key

            table[i] = (new_key, op.xreplace({old_id: new_id}))

        self.table = bidict.OrderedBidict(table)

    def clear(self):
        """Empty the table."""
        self.__init__()
