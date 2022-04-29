"""Tests for the Difference module."""
import doctest
import unittest

from hypothesis import given, settings   # , reproduce_failure
from hypothesis.strategies import integers

from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.operation import (
    BvNot, BvXor, RotateLeft, RotateRight, BvShl, BvLshr, Extract, Concat,
    BvAnd, BvOr, BvAdd, make_partial_operation)
from cascada.bitvector.secondaryop import BvIf, BvMaj

from cascada.differential import difference
from cascada.differential.difference import XorDiff, RXDiff, RXOp  # for vrepr

MIN_SIZE = 2
MAX_SIZE = 32


class TestDifference(unittest.TestCase):
    """Tests for the Difference class."""

    def test_with_variables(self):
        a = Variable("a", 8)
        b = Variable("b", 8)
        one = Constant(1, 8)
        for diff_type in [XorDiff, RXDiff]:
            diff = diff_type.from_pair(a, b)
            self.assertEqual(diff, eval(diff.vrepr()))
            diff_xreplaced = diff.xreplace({diff_type(b): diff_type(one)})
            self.assertEqual(diff_xreplaced, diff_type.from_pair(a, one))
            self.assertEqual(diff_xreplaced, eval(diff_xreplaced.vrepr()))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_from_pair(self, width, x, y):
        for diff_type in [XorDiff, RXDiff]:
            bvx = Constant(x % (2 ** width), width)
            bvy = Constant(y % (2 ** width), width)
            bvd = diff_type.from_pair(bvx, bvy)
            self.assertEqual(bvy, bvd.get_pair_element(bvx))
            self.assertEqual(bvd, eval(bvd.vrepr()))

    # noinspection PyPep8Naming
    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_xor_deterministic_op(self, width, x1, y1, x2, y2):
        # for unary operation:
        #   input 1: x1       --> f(x1)
        #   input 2: y1=x1^d1 --> f(y1)
        # for binary operations:
        #   input 1: x1,      x2       --> f(x1, x2)
        #   input 2: y1=x1^d1,y2=x1^d2 --> f(y1, y2)

        x1 = Constant(x1 % (2 ** width), width)
        y1 = Constant(y1 % (2 ** width), width)
        d_xy1 = XorDiff.from_pair(x1, y1)
        x2 = Constant(x2 % (2 ** width), width)
        y2 = Constant(y2 % (2 ** width), width)
        d_xy2 = XorDiff.from_pair(x2, y2)

        var = Variable("c", width)
        ct = x2

        self.assertEqual(
            XorDiff.from_pair(~x1, ~y1),
            XorDiff.propagate(BvNot, d_xy1))

        self.assertEqual(
            XorDiff.from_pair(x1 ^ x2, y1 ^ y2),
            XorDiff.propagate(BvXor, [d_xy1, d_xy2]))

        self.assertEqual(
            XorDiff.from_pair(x1 ^ var, y1 ^ var),
            XorDiff.propagate(BvXor, [d_xy1, XorDiff.from_pair(var, var)]))

        self.assertEqual(
            XorDiff.from_pair(Concat(x1, x2), Concat(y1, y2)),
            XorDiff.propagate(Concat, [d_xy1, d_xy2]))

        # without replacing c by a random value, AssertionError is raised since expr like
        # "XorDiff((0b00 :: c) ^ (0b01 :: c)) != XorDiff(0x4)" cannot be simplified
        aux = XorDiff.from_pair(Concat(x1, var), Concat(y1, var))
        aux = XorDiff(aux.val.xreplace(({var: x2})))
        self.assertEqual(
            aux,
            XorDiff.propagate(Concat, [d_xy1, XorDiff.from_pair(var, var)]))

        BvXor_fix = make_partial_operation(BvXor, tuple([None, ct]))
        self.assertEqual(
            XorDiff.from_pair(x1 ^ ct, y1 ^ ct),
            XorDiff.propagate(BvXor_fix, [d_xy1]))

        BvOr_fix = make_partial_operation(BvOr, tuple([None, ct]))
        self.assertEqual(
            XorDiff.from_pair(x1 | ct, y1 | ct),
            XorDiff.propagate(BvOr_fix, [d_xy1]))

        BvAnd_fix = make_partial_operation(BvAnd, tuple([None, ct]))
        self.assertEqual(
            XorDiff.from_pair(x1 & ct, y1 & ct),
            XorDiff.propagate(BvAnd_fix, [d_xy1]))

        r = int(x2) % x1.width
        RotateLeft_fix = make_partial_operation(RotateLeft, tuple([None, r]))
        RotateRight_fix = make_partial_operation(RotateRight, tuple([None, r]))
        self.assertEqual(
            XorDiff.from_pair(RotateLeft(x1, r), RotateLeft(y1, r)),
            XorDiff.propagate(RotateLeft_fix, d_xy1))
        self.assertEqual(
            XorDiff.from_pair(RotateRight(x1, r), RotateRight(y1, r)),
            XorDiff.propagate(RotateRight_fix, d_xy1))

        r = Constant(int(x2) % x1.width, width)
        BvShl_fix = make_partial_operation(BvShl, tuple([None, r]))
        BvLshr_fix = make_partial_operation(BvLshr, tuple([None, r]))
        self.assertEqual(
            XorDiff.from_pair(x1 << r, y1 << r),
            XorDiff.propagate(BvShl_fix, d_xy1))
        self.assertEqual(
            XorDiff.from_pair(x1 >> r, y1 >> r),
            XorDiff.propagate(BvLshr_fix, d_xy1))

        BvAdd_fix = make_partial_operation(BvAdd, tuple([None, Constant(0, width)]))
        self.assertEqual(
            XorDiff.from_pair(x1 + 0, y1 + 0),
            XorDiff.propagate(BvAdd_fix, [d_xy1]))

        i = int(x2) % x1.width
        j = int(y2) % (i + 1)
        Extract_fix = make_partial_operation(Extract, tuple([None, i, j]))
        self.assertEqual(
            XorDiff.from_pair(x1[i:j], y1[i:j]),
            XorDiff.propagate(Extract_fix, d_xy1))

        Concat_fix = make_partial_operation(Concat, tuple([None, ct]))
        self.assertEqual(
            XorDiff.from_pair(Concat(x1, ct), Concat(y1, ct)),
            XorDiff.propagate(Concat_fix, [d_xy1]))

        one, two = Constant(1, width), Constant(2, width)
        for op in [BvIf, BvMaj]:
            Op_fix = make_partial_operation(op, tuple([None, one, two]))
            op_model = XorDiff.propagate(Op_fix, [d_xy1])
            self.assertTrue(op_model.validity_constraint(
                XorDiff.from_pair(op(x1, one, two), op(y1, one, two))
            ))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    @settings(deadline=None)
    def test_rx_deterministic_op(self, width, x1, y1, x2, y2):
        # for unary operation:
        #   input 1: x1       --> f(x1)
        #   input 2: y1=x1^d1 --> f(y1)
        # for binary operations:
        #   input 1: x1,      x2       --> f(x1, x2)
        #   input 2: y1=x1^d1,y2=x1^d2 --> f(y1, y2)

        x1 = Constant(x1 % (2 ** width), width)
        y1 = Constant(y1 % (2 ** width), width)
        d_xy1 = RXDiff.from_pair(x1, y1)
        x2 = Constant(x2 % (2 ** width), width)
        y2 = Constant(y2 % (2 ** width), width)
        d_xy2 = RXDiff.from_pair(x2, y2)

        var = Variable("c", width)
        ct = x2

        self.assertEqual(
            RXDiff.from_pair(~x1, ~y1),
            RXDiff.propagate(BvNot, d_xy1))

        self.assertEqual(
            RXDiff.from_pair(x1 ^ x2, y1 ^ y2),
            RXDiff.propagate(BvXor, [d_xy1, d_xy2]))

        expr1 = RXDiff.from_pair(x1 ^ var, y1 ^ var).val.doit()
        expr2 = RXDiff.propagate(BvXor, [d_xy1, RXDiff.from_pair(var, var)]).val.doit()
        # without replacing c by a random value, AssertionError is raised since expr like
        # "(0b01 ^ c) ^ (c <<< 1) != 0b01 ^ (c ^ (c <<< 1))" cannot be simplified
        self.assertEqual(expr1.xreplace(({var: x2})), expr2.xreplace(({var: x2})))

        BvXor_fix = make_partial_operation(BvXor, tuple([None, ct]))
        self.assertEqual(
            RXDiff.from_pair(x1 ^ ct, y1 ^ ct),
            RXDiff.propagate(BvXor_fix, [d_xy1]))

        r = int(x2) % x1.width
        RotateLeft_fix = make_partial_operation(RotateLeft, tuple([None, r]))
        RotateRight_fix = make_partial_operation(RotateRight, tuple([None, r]))
        self.assertEqual(
            RXDiff.from_pair(RotateLeft(x1, r), RotateLeft(y1, r)),
            RXDiff.propagate(RotateLeft_fix, d_xy1))
        self.assertEqual(
            RXDiff.from_pair(RotateRight(x1, r), RotateRight(y1, r)),
            RXDiff.propagate(RotateRight_fix, d_xy1))

        BvShl_fix = make_partial_operation(BvShl, tuple([None, Constant(0, width)]))
        BvLShr_fix = make_partial_operation(BvLshr, tuple([None, Constant(0, width)]))
        for op in [BvShl_fix, BvLShr_fix]:
            self.assertEqual(
                RXDiff.from_pair(op(x1), op(y1)),
                RXDiff.propagate(op, [d_xy1]))

        r = x1.width + (int(x2) % (2**x1.width - x1.width))  # width <= r < 2**width
        BvShl_fix = make_partial_operation(BvShl, tuple([None, Constant(r, width)]))
        BvLShr_fix = make_partial_operation(BvLshr, tuple([None, Constant(r, width)]))
        for op in [BvShl_fix, BvLShr_fix]:
            self.assertEqual(
                RXDiff.from_pair(op(x1), op(y1)),
                RXDiff.propagate(op, [d_xy1]))

        one = Constant(1, width)
        for op in [BvAnd, BvOr]:
            Op_fix = make_partial_operation(op, tuple([None, one]))
            op_model = RXDiff.propagate(Op_fix, [d_xy1])
            self.assertTrue(op_model.validity_constraint(
                RXDiff.from_pair(op(x1, one), op(y1, one))
            ))

        one, two = Constant(1, width), Constant(2, width)
        for op in [BvIf, BvMaj]:
            Op_fix = make_partial_operation(op, tuple([None, one, two]))
            op_model = RXDiff.propagate(Op_fix, [d_xy1])
            self.assertTrue(op_model.validity_constraint(
                RXDiff.from_pair(op(x1, one, two), op(y1, one, two))
            ))


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(difference))
    return tests
