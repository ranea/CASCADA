"""Tests for the opmodel module."""
import doctest

from hypothesis import given, example, settings  # unlimited, HealthCheck
# useful hypothesis decorator
# @example(arg0=..., arg1=..., ...)
# @settings(max_examples=10000, deadline=None)
from hypothesis.strategies import integers

from cascada.bitvector.core import Constant, Variable

from cascada.abstractproperty.tests.test_opmodel import TestOpModelGeneric as AbstractTestOpModelGeneric

from cascada.algebraic.value import Value, BitValue
from cascada.algebraic import opmodel
from cascada.algebraic.opmodel import BitModelBvAdd, BitModelBvSub


MAX_SMALL_WIDTH = 8

VERBOSE = False
PRINT_DISTRIBUTION_ERROR = False
PRINT_TOP_3_ERRORS = False
PRECISION_DISTRIBUTION_ERROR = 4


class TestOpModelGeneric(AbstractTestOpModelGeneric):
    """Base class for testing OpModel."""
    VERBOSE = VERBOSE
    PRINT_DISTRIBUTION_ERROR = PRINT_DISTRIBUTION_ERROR
    PRINT_TOP_3_ERRORS = PRINT_TOP_3_ERRORS
    PRECISION_DISTRIBUTION_ERROR = PRECISION_DISTRIBUTION_ERROR

    @classmethod
    def is_valid_slow(cls, f, beta):
        assert isinstance(beta, Value) and isinstance(beta.val, Constant)
        input_ct_bvs = [val.val for val in f.input_val]
        output_ct_bv = f.op(*input_ct_bvs)
        assert isinstance(output_ct_bv, Constant)
        return output_ct_bv == beta.val

    @classmethod
    def get_empirical_weight_slow(cls, f, beta):
        return 0


class TestOpModelsSmallWidth(TestOpModelGeneric):
    """Common tests for all models with small width."""

    def test_vrepr(self):
        vals = [BitValue(Variable("d" + str(i), 8)) for i in range(2)]
        for model in [BitModelBvAdd, BitModelBvSub]:
            model_fixed = model(vals)
            # str used since they are not singleton objects
            self.assertEqual(str(model_fixed), str(eval(model_fixed.vrepr())))

    @given(
        integers(min_value=2, max_value=MAX_SMALL_WIDTH),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
    )
    @settings(deadline=None, max_examples=1000)
    def test_models_slow(self, width, d1, d2, d3):
        d1 = BitValue(Constant(d1 % 2**width, width))
        d2 = BitValue(Constant(d2 % 2**width, width))
        d3 = BitValue(Constant(d3 % 2**width, width))
        for model in [BitModelBvAdd, BitModelBvSub]:
            input_val = [d1, d2]
            output_val = d3
            self.base_test_op_model(model, input_val, output_val)
            self.base_test_pr_one_constraint_slow(model, input_val, output_val)
            if output_val.val == 0:
                self.base_test_op_model_sum_pr_1(model, input_val)


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(opmodel))
    return tests
