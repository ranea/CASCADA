"""Test the XOR and RX differential models of Simon-like round functions"""
import doctest
import math
import unittest

from hypothesis import given, example, settings, assume, HealthCheck
from hypothesis.strategies import integers

from cascada.bitvector.core import Constant

from cascada.differential.tests.test_opmodel import TestOpModelGeneric

from cascada.differential.difference import XorDiff, RXDiff

from cascada.primitives import simon_rf


SKIP_LONG_TESTS = True

DIFF_WIDTH = 8
assert DIFF_WIDTH >= 4 and DIFF_WIDTH % 2 == 0

VERBOSE = False
PRINT_DISTRIBUTION_ERROR = False
PRINT_TOP_3_ERRORS = False
PRECISION_DISTRIBUTION_ERROR = 4


class TestOpModelsSmallWidth(TestOpModelGeneric):
    VERBOSE = VERBOSE
    PRINT_DISTRIBUTION_ERROR = PRINT_DISTRIBUTION_ERROR
    PRINT_TOP_3_ERRORS = PRINT_TOP_3_ERRORS
    PRECISION_DISTRIBUTION_ERROR = PRECISION_DISTRIBUTION_ERROR

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_Xor_models_slow")
    @given(
        integers(min_value=0, max_value=DIFF_WIDTH - 1),
        integers(min_value=0, max_value=DIFF_WIDTH - 2),
        integers(min_value=1, max_value=2),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
    )
    @settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
    def test_Xor_models_slow(self, a, b, c, d1, d2):
        width = DIFF_WIDTH
        assume(math.gcd(width, a - b) == 1 and a > b and width % 2 == 0)

        _a, _b, _c = a, b, c
        class MySimonRf(simon_rf.SimonRF):
            a, b, c = _a, _b, _c
        class XorModelMySimonRF(simon_rf.XorModelSimonRF):
            op = MySimonRf
        MySimonRf.xor_model = XorModelMySimonRF

        d1 = XorDiff(Constant(d1 % 2**width, width))
        d2 = XorDiff(Constant(d2 % 2**width, width))

        model = XorModelMySimonRF
        input_diff = [d1]
        output_diff = d2
        self.base_test_op_model(model, input_diff, output_diff)
        self.base_test_pr_one_constraint_slow(model, input_diff, output_diff)
        if output_diff.val == 0:
            self.base_test_op_model_sum_pr_1(model, input_diff)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_RX_models_slow")
    @given(
        integers(min_value=0, max_value=DIFF_WIDTH - 1),
        integers(min_value=0, max_value=DIFF_WIDTH - 2),
        integers(min_value=1, max_value=2),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
    )
    @settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
    def test_RX_models_slow(self, a, b, c, d1, d2):
        width = DIFF_WIDTH
        assume(math.gcd(width, a - b) == 1 and a > b and width % 2 == 0)

        _a, _b, _c = a, b, c
        class MySimonRf(simon_rf.SimonRF):
            a, b, c = _a, _b, _c
        class RXModelMySimonRF(simon_rf.RXModelSimonRF):
            op = MySimonRf
        MySimonRf.rx_model = RXModelMySimonRF

        d1 = RXDiff(Constant(d1 % 2**width, width))
        d2 = RXDiff(Constant(d2 % 2**width, width))

        model = RXModelMySimonRF
        input_diff = [d1]
        output_diff = d2
        self.base_test_op_model(model, input_diff, output_diff)
        self.base_test_pr_one_constraint_slow(model, input_diff, output_diff)
        if output_diff.val == 0:
            self.base_test_op_model_sum_pr_1(model, input_diff)


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(simon_rf))
    return tests
