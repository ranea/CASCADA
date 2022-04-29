"""Tests for the opmodel module."""
import decimal
import doctest
import itertools
import math
import unittest

from hypothesis import given, example, settings, assume  # unlimited, HealthCheck
# useful hypothesis decorator
# @example(arg0=..., arg1=..., ...)
# @settings(max_examples=10000, deadline=None)
from hypothesis.strategies import integers, decimals, booleans

from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.context import Validation, Simplification

from cascada.abstractproperty.opmodel import log2_decimal
from cascada.abstractproperty.tests.test_opmodel import TestOpModelGeneric as AbstractTestOpModelGeneric
from cascada.bitvector.operation import SecondaryOperation

from cascada.differential.tests.test_opmodel import TestOtherOpModels as DifferentialTestOtherOpModels

from cascada.linear.mask import LinearMask
from cascada.linear import opmodel
from cascada.linear.opmodel import (
    LinearModelBvAdd, LinearModelBvSub, LinearModelBvAnd, LinearModelBvXor, LinearModelBvOr,
    LinearModelBvAndCt, LinearModelBvOrCt, LinearModelExtractCt,
    LinearModelBvLshrCt, LinearModelBvShlCt, make_partial_op_model,
    get_weak_model, get_branch_number_model, get_wdt_model
)


SKIP_LONG_TESTS = True

MAX_SMALL_WIDTH = 4

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
    def _count_right_inputs(cls, f, beta):
        widths = [m.val.width for m in f.input_mask]
        iterators = [range(2 ** w) for w in widths]
        num_valid_inputs = 0
        with Simplification(False), Validation(False):
            for x in itertools.product(*iterators):
                x = [Constant(x_i, w) for x_i, w in zip(x, widths)]
                y = f.op(*x)
                # lhs = <a,x>, rhs= <f(x), b>
                lhs = f.input_mask[0].apply(x[0])
                for i in range(1, len(x)):
                    lhs ^= f.input_mask[i].apply(x[i])
                rhs = beta.apply(y)
                assert isinstance(lhs, Constant) and isinstance(rhs, Constant)
                if lhs == rhs:
                    num_valid_inputs += 1
        return num_valid_inputs

    @classmethod
    def is_valid_slow(cls, f, beta):
        assert isinstance(beta, LinearMask) and isinstance(beta.val, Constant)
        num_right_inputs = cls._count_right_inputs(f, beta)
        widths = [m.val.width for m in f.input_mask]
        return num_right_inputs != 2**(sum(widths) - 1)

    @classmethod
    def get_empirical_weight_slow(cls, f, beta):
        assert isinstance(beta, LinearMask) and isinstance(beta.val, Constant)
        num_right_inputs = cls._count_right_inputs(f, beta)
        widths = [m.val.width for m in f.input_mask]
        total_inputs = 2 ** sum(widths)
        assert num_right_inputs != 2**(sum(widths) - 1)
        inner_pr = num_right_inputs / decimal.Decimal(total_inputs)
        correlation = (2 * inner_pr - 1).copy_abs()
        assert correlation != 0
        return - log2_decimal(correlation)

    def get_bv_weight_with_frac_bits(self, f, beta):
        output_mask = beta
        if isinstance(f, LinearModelBvAdd):
            bv_weight, extra_constraints = f._bvweight_and_extra_constraints(beta)
            assert not extra_constraints
            bv_weight_with_frac_bits = decimal.Decimal(int(bv_weight))
        else:
            bv_weight_with_frac_bits = decimal.Decimal(int(f.bv_weight(output_mask)))
        return bv_weight_with_frac_bits

    def base_test_op_model_sum_pr_1(self, op_model_class, input_widths, output_mask, op_model_kwargs=None):
        """Test the propagation probability sums to 1, with output mask fixed and for all input masks.

        Source : https://eprint.iacr.org/2005/212 (equation 8)
        """
        msg = ""
        all_prs = []
        for x in itertools.product(*[range(2 ** w) for w in input_widths]):
            input_mask = [op_model_class.prop_type(Constant(x_i, w)) for x_i, w in zip(x, input_widths)]

            if op_model_kwargs is not None:
                f = op_model_class(input_mask, **op_model_kwargs)
            else:
                f = op_model_class(input_mask)

            if hasattr(f, "precision"):
                aux_str = f"_precision={f.precision}"
            else:
                aux_str = ""
            msg += f"{f}{aux_str}(output_mask={output_mask})\n"

            is_valid = f.validity_constraint(output_mask)
            # msg += "\tis_valid: {}\n".format(is_valid)

            if is_valid:
                if f.error() == 0:
                    weight = f.decimal_weight(output_mask)
                else:
                    weight = self.get_empirical_weight_slow(f, output_mask)
                # msg += "\tweight: {}\n".format(weight)
                all_prs.append((decimal.Decimal(2)**(-weight))**2)  # last **2 from corr to pr.
            else:
                all_prs.append(0)
            msg += "\tprobability: {}\n".format(all_prs[-1])

        self.assertAlmostEqual(sum(all_prs), 1, msg=msg)

        if self.__class__.VERBOSE:
            print(msg)


class TestOpModelsSmallWidth(TestOpModelGeneric):
    """Common tests for all models with small width."""

    def test_vrepr(self):
        masks = [LinearMask(Variable("m" + str(i), 8)) for i in range(4)]
        LinearModelBvAndCt_1r = make_partial_op_model(LinearModelBvAndCt, (None, Constant(1, 8)))
        LinearModelBvAndCt_1l = make_partial_op_model(LinearModelBvAndCt, (Constant(1, 8), None))
        LinearModelBvOrCt_1r = make_partial_op_model(LinearModelBvOrCt, (None, Constant(1, 8)))
        LinearModelBvOrCt_1l = make_partial_op_model(LinearModelBvOrCt, (Constant(1, 8), None))
        LinearModelBvLshrCt_1 = make_partial_op_model(LinearModelBvLshrCt, (None, Constant(1, 8)))
        LinearModelBvShlCt_1 = make_partial_op_model(LinearModelBvShlCt, (None, Constant(1, 8)))
        LinearModelExtractCt_21 = make_partial_op_model(LinearModelExtractCt, (None, 2, 1))
        for model in [LinearModelBvAdd, LinearModelBvSub, LinearModelBvAnd, LinearModelBvOr, LinearModelBvXor,
                      LinearModelBvAndCt_1r, LinearModelBvAndCt_1l, LinearModelBvOrCt_1r, LinearModelBvOrCt_1l,
                      LinearModelBvLshrCt_1, LinearModelBvShlCt_1, LinearModelExtractCt_21]:
            num_input_mask = model.op.arity[0]
            model_fixed = model(masks[:num_input_mask])
            # str used since they are not singleton objects
            self.assertEqual(str(model_fixed), str(eval(model_fixed.vrepr())))

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_Linear_models_slow")
    @given(
        integers(min_value=2, max_value=MAX_SMALL_WIDTH),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
    )
    @settings(deadline=None, max_examples=1000)  # ~1min
    def test_Linear_models_slow(self, width, m1, m2, m3):
        # ct max value is 1 + (2**width - 2) = 2**width - 1
        ct = Constant(1 + (m3 % (2 ** width - 1)), width)

        m1 = LinearMask(Constant(m1 % 2 ** width, width))
        m2 = LinearMask(Constant(m2 % 2 ** width, width))
        m3 = LinearMask(Constant(m3 % 2 ** width, width))

        LinearModelBvAndCt_1r = make_partial_op_model(LinearModelBvAndCt, (None, ct - 1))
        LinearModelBvAndCt_1l = make_partial_op_model(LinearModelBvAndCt, (ct - 1, None))
        LinearModelBvOrCt_1r = make_partial_op_model(LinearModelBvOrCt, (None, ct))
        LinearModelBvOrCt_1l = make_partial_op_model(LinearModelBvOrCt, (ct, None))
        LinearModelBvLshrCt_1 = make_partial_op_model(LinearModelBvLshrCt, (None, ct))
        LinearModelBvShlCt_1 = make_partial_op_model(LinearModelBvShlCt, (None, ct))

        i = int(m2.val) % m1.val.width
        j = int(m3.val) % (i + 1)
        LinearModelExtractCt_21 = make_partial_op_model(LinearModelExtractCt, (None, i, j))

        for model in [LinearModelBvAdd, LinearModelBvSub, LinearModelBvAnd, LinearModelBvOr, LinearModelBvXor,
                      LinearModelBvAndCt_1r, LinearModelBvAndCt_1l, LinearModelBvOrCt_1r, LinearModelBvOrCt_1l,
                      LinearModelBvLshrCt_1, LinearModelBvShlCt_1, LinearModelExtractCt_21]:
            num_input_mask = model.op.arity[0]
            input_mask = [m1, m2][:num_input_mask]
            if model == LinearModelExtractCt_21:
                extract_width = i - j + 1
                output_mask = LinearMask(Constant(int(m3.val) % 2 ** extract_width, extract_width))
            else:
                output_mask = m3
            self.base_test_op_model(model, input_mask, output_mask)
            self.base_test_pr_one_constraint_slow(model, input_mask, output_mask)
            if output_mask.val == 0:
                if all(m.val == 0 for m in input_mask):
                    input_widths = [m.val.width for m in input_mask]
                    self.base_test_op_model_sum_pr_1(model, input_widths, output_mask)


class TestOtherOpModels(DifferentialTestOtherOpModels):
    """Test linear WeakModel and BranchNumberModel."""

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_weak_model")
    @given(
        integers(min_value=1, max_value=3),
        integers(min_value=2, max_value=3),
        decimals(min_value=0, max_value=3),  # 3 converted to inf
        decimals(min_value=0, max_value=3),  # 3 converted to inf
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=1000)
    def test_weak_model(self, num_inputs, width, nonzero2nonzero_weight, zero2nonzero_weight, precision):
        if nonzero2nonzero_weight == 3:
            nonzero2nonzero_weight = math.inf
        if zero2nonzero_weight == 3:
            zero2nonzero_weight = math.inf

        for w in [nonzero2nonzero_weight, zero2nonzero_weight]:
            assume(not (0 != w != math.inf and int(w * (2 ** precision)) == 0))
        assume(not
            (precision != 0 and
                (nonzero2nonzero_weight == math.inf or nonzero2nonzero_weight == int(nonzero2nonzero_weight)) and
                (zero2nonzero_weight == math.inf or zero2nonzero_weight == int(zero2nonzero_weight))
            )
        )

        class MyOp(SecondaryOperation):
            arity = [num_inputs, 0]
            @classmethod
            def condition(cls, *args): return all(a.width == width for a in args)
            @classmethod
            def output_width(cls, *args): return width
            @classmethod
            def eval(cls, *args): return None

        WeakModel = get_weak_model(
            MyOp, nonzero2nonzero_weight,
            zero2nonzero_weight=zero2nonzero_weight, precision=precision
        )

        self._check_weak_model(WeakModel, width, num_inputs, verbose=VERBOSE)  # , write_msg=True)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_branchnumber_model")
    @given(
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=2, max_value=3),
        integers(min_value=2, max_value=5),
        decimals(min_value=0, max_value=3),  # 3 converted to math.inf
        decimals(min_value=0, max_value=3),  # 3 converted to math.inf
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=200)
    def test_branchnumber_model(self, num_inputs, num_outputs, width,
                                branch_number, nonzero2nonzero_weight, zero2nonzero_weight, precision):
        assume(branch_number <= num_inputs + num_outputs)

        if nonzero2nonzero_weight == 3:
            nonzero2nonzero_weight = math.inf
        if zero2nonzero_weight == 3:
            zero2nonzero_weight = math.inf

        for w in [nonzero2nonzero_weight, zero2nonzero_weight]:
            assume(not (0 != w != math.inf and int(w * (2 ** precision)) == 0))
        assume(not
            (precision != 0 and
                (nonzero2nonzero_weight == math.inf or nonzero2nonzero_weight == int(nonzero2nonzero_weight)) and
                (zero2nonzero_weight == math.inf or zero2nonzero_weight == int(zero2nonzero_weight))
            )
        )

        class MyOp(SecondaryOperation):
            arity = [num_inputs, 0]
            @classmethod
            def condition(cls, *args): return all(a.width == width for a in args)
            @classmethod
            def output_width(cls, *args): return num_outputs*width
            @classmethod
            def eval(cls, *args): return None

        BranchNumberModel = get_branch_number_model(
            MyOp, tuple([width for _ in range(num_outputs)]), branch_number,
            nonzero2nonzero_weight, zero2nonzero_weight=zero2nonzero_weight, precision=precision
        )

        self._check_branchnumber_model(BranchNumberModel, num_inputs, verbose=VERBOSE)  # , write_msg=True)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_wdt_model")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=0),
        booleans(),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=10000)
    def test_wdt_model(self, input_width, output_width, seed, lrtc, precision):
        self.__class__.PRNG.seed(seed)

        found_frac_bits = False
        wdt = [[None for _ in range(2**output_width)] for _ in range(2**input_width)]
        for i in range(len(wdt)):
            for j in range(len(wdt[0])):
                if i == 0 == j:
                    wdt[i][j] = 0
                    continue
                # maximum weight of a differential with n-bit input is n
                w = self.__class__.PRNG.uniform(0, 2 ** (input_width + 1))
                if w >= 2 ** input_width:  # 0.5 pr. obtaining math.inf
                    w = math.inf
                elif int(w * (2 ** precision)) == 0:
                    w = decimal.Decimal(0)
                else:
                    w = decimal.Decimal(w)
                    if w != int(w):
                        found_frac_bits = True
                wdt[i][j] = w
        assume(not(not found_frac_bits and precision != 0))
        for j in range(len(wdt[0])):
            assume(not all(wdt[i][j] == math.inf for i in range(len(wdt))))
        wdt = tuple(tuple(row) for row in wdt)

        class MyOp(SecondaryOperation):
            arity = [1, 0]
            @classmethod
            def condition(cls, *args): return args[0].width == input_width
            @classmethod
            def output_width(cls, *args): return output_width
            @classmethod
            def eval(cls, *args): return None

        WDTModel = get_wdt_model(
            MyOp, wdt, loop_rows_then_columns=bool(lrtc), precision=precision
        )

        self._check_wdt_model(WDTModel, seed, verbose=VERBOSE)  # , write_msg=True)


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(opmodel))
    return tests
