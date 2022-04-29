"""Tests for the Mask module."""
import doctest
import unittest

from hypothesis import given, settings   # , reproduce_failure
from hypothesis.strategies import integers

from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.operation import (
    BvComp,  BvXor, RotateLeft, RotateRight, Concat, BvNot,
    BvAnd, BvOr, BvShl, BvLshr, make_partial_operation)

from cascada.linear import mask as cascada_mask
from cascada.linear.mask import LinearMask  # for vrepr

MIN_SIZE = 2
MAX_SIZE = 32


class TestMask(unittest.TestCase):
    """Tests for the Mask class."""

    def test_with_variables(self):
        a = Variable("a", 8)
        one = Constant(1, 8)
        my_mask = LinearMask(a)
        self.assertEqual(my_mask, eval(my_mask.vrepr()))
        my_mask_xreplaced = my_mask.xreplace({LinearMask(a): LinearMask(one)})
        self.assertEqual(my_mask_xreplaced, LinearMask(one))
        self.assertEqual(my_mask_xreplaced, eval(my_mask_xreplaced.vrepr()))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
    )
    def test_apply(self, width, x, y):
        bvx = Constant(x % (2 ** width), width)
        bvy = Constant(y % (2 ** width), width)
        self.assertEqual(
            LinearMask(bvx).apply(~bvx),  # inner product <x, BvNot(x)> is always zero
            Constant(0, 1)
        )
        self.assertEqual(
            LinearMask(bvx).apply(bvy),  # commutative property
            LinearMask(bvy).apply(bvx)
        )
        self.assertEqual(LinearMask(bvx), eval(LinearMask(bvx).vrepr()))


    # noinspection PyPep8Naming
    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_linear_deterministic_op(self, width, in1, in2, x1, x2, x3, x4):
        # if (alpha, beta) is a linear approximation of f with probability 1
        # then for all x1,x2 [alpha*x1 = f_k(x1)*beta <=> alpha*x2 = f_k(x2)*beta]

        # words for input mask
        in1 = LinearMask(Constant(in1 % (2 ** width), width))
        in2 = LinearMask(Constant(in2 % (2 ** width), width))

        x1 = Constant(x1 % (2 ** width), width)
        x2 = Constant(x2 % (2 ** width), width)
        x3 = Constant(x3 % (2 ** width), width)
        x4 = Constant(x4 % (2 ** width), width)

        ct = x3

        alpha = in1
        beta = LinearMask.propagate(BvNot, alpha)
        self.assertEqual(
            BvComp(alpha.apply(x1), beta.apply(BvNot(x1))),
            BvComp(alpha.apply(x2), beta.apply(BvNot(x2))),
        )

        alpha = [in1, in1]
        beta = LinearMask.propagate(BvXor, alpha)
        self.assertEqual(
            BvComp(alpha[0].apply(x1) ^ alpha[1].apply(x2), beta.apply(BvXor(x1, x2))),
            BvComp(alpha[0].apply(x3) ^ alpha[1].apply(x4), beta.apply(BvXor(x3, x4))),
        )

        alpha = [in1, in2]
        beta = LinearMask.propagate(Concat, alpha)
        self.assertEqual(
            BvComp(alpha[0].apply(x1) ^ alpha[1].apply(x2), beta.apply(Concat(x1, x2))),
            BvComp(alpha[0].apply(x3) ^ alpha[1].apply(x4), beta.apply(Concat(x3, x4))),
        )

        BvXor_fix = make_partial_operation(BvXor, tuple([None, ct]))
        alpha = in1
        beta = LinearMask.propagate(BvXor_fix, in1)
        self.assertEqual(
            BvComp(alpha.apply(x1), beta.apply(BvXor_fix(x1))),
            BvComp(alpha.apply(x2), beta.apply(BvXor_fix(x2))),
        )

        allones = ~Constant(0, width)
        BvAnd_fix = make_partial_operation(BvAnd, tuple([None, allones]))
        alpha = in1
        beta = LinearMask.propagate(BvAnd_fix, alpha)
        self.assertEqual(
            BvComp(alpha.apply(x1), beta.apply(BvAnd_fix(x1))),
            BvComp(alpha.apply(x2), beta.apply(BvAnd_fix(x2))),
        )

        allzeros = Constant(0, width)
        BvOr_fix = make_partial_operation(BvOr, tuple([None, allzeros]))
        alpha = in1
        beta = LinearMask.propagate(BvOr_fix, alpha)
        self.assertEqual(
            BvComp(alpha.apply(x1), beta.apply(BvOr_fix(x1))),
            BvComp(alpha.apply(x2), beta.apply(BvOr_fix(x2))),
        )

        r = int(x3) % x1.width
        RotateLeft_fix = make_partial_operation(RotateLeft, tuple([None, r]))
        alpha = in1
        beta = LinearMask.propagate(RotateLeft_fix, in1)
        self.assertEqual(
            BvComp(alpha.apply(x1), beta.apply(RotateLeft_fix(x1))),
            BvComp(alpha.apply(x2), beta.apply(RotateLeft_fix(x2))),
        )
        RotateRight_fix = make_partial_operation(RotateRight, tuple([None, r]))
        beta = LinearMask.propagate(RotateRight_fix, in1)
        self.assertEqual(
            BvComp(alpha.apply(x1), beta.apply(RotateRight_fix(x1))),
            BvComp(alpha.apply(x2), beta.apply(RotateRight_fix(x2))),
        )

        alpha = in1
        ct = Constant(0, width)
        for op in [BvShl, BvLshr]:
            op_fix = make_partial_operation(op, tuple([None, ct]))
            beta = LinearMask.propagate(op_fix, alpha)
            self.assertEqual(
                BvComp(alpha.apply(x1), beta.apply(op_fix(x1))),
                BvComp(alpha.apply(x2), beta.apply(op_fix(x2))),
            )


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(cascada_mask))
    return tests
