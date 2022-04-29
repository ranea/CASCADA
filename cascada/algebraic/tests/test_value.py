"""Tests for the Value module."""
import doctest
import unittest

from hypothesis import given, settings   # , reproduce_failure
from hypothesis.strategies import integers

from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.operation import (
    BvNot, BvXor, RotateLeft, RotateRight, BvShl, BvLshr, Extract, Concat,
    BvAnd, BvOr, BvAdd, make_partial_operation)
from cascada.bitvector.secondaryop import BvIf, BvMaj

from cascada.algebraic import value
from cascada.algebraic.value import BitValue, WordValue  # for vrepr

MIN_SIZE = 2
MAX_SIZE = 32


class TestValue(unittest.TestCase):
    """Tests for the Value class."""

    def test_with_variables(self):
        a = Variable("a", 8)
        one = Constant(1, 8)
        for value_type in [BitValue, WordValue]:
            my_value = value_type(a)
            self.assertEqual(my_value, eval(my_value.vrepr()))
            my_value_xreplaced = my_value.xreplace({value_type(a): value_type(one)})
            self.assertEqual(my_value_xreplaced, value_type(one))
            self.assertEqual(my_value_xreplaced, eval(my_value_xreplaced.vrepr()))

    # noinspection PyPep8Naming
    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    @settings(deadline=None)
    def test_bit_deterministic_op(self, width, x1, x2, x3):
        x1 = Constant(x1 % (2 ** width), width)
        x2 = Constant(x2 % (2 ** width), width)
        x3 = Constant(x3 % (2 ** width), width)

        for op in [BvNot, BvAnd, BvOr, BvXor, BvMaj, BvIf, Concat]:
            assert op.arity[1] == 0
            input_vals = [x1, x2, x3][:op.arity[0]]
            self.assertEqual(
                BitValue(op(*input_vals)),
                BitValue.propagate(op, [BitValue(x_i) for x_i in input_vals]))

        for op in [RotateLeft, RotateRight, BvShl, BvLshr]:
            if op in [RotateLeft, RotateRight]:
                ct = int(x2) % x2.width
            else:
                ct = x2
            op_fix = make_partial_operation(op, tuple([None, ct]))
            self.assertEqual(
                BitValue(op(x1, ct)),
                BitValue.propagate(op_fix, BitValue(x1)),
                msg=f"\nop={op.__name__}\nx1={x1}\nct={ct}\nop(x1, ct)={op(x1, ct)}"
            )

        i = int(x2) % x1.width
        j = int(x3) % (i + 1)
        Extract_fix = make_partial_operation(Extract, tuple([None, i, j]))
        self.assertEqual(
            BitValue(x1[i:j]),
            BitValue.propagate(Extract_fix, BitValue(x1)))


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(value))
    return tests
