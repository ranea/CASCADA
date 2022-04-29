"""Tests for the opmodel module."""
import decimal
import doctest
import importlib
import itertools
import math
import pprint
import random
import tempfile
import unittest

import cffi
from hypothesis import given, example, settings, assume  # unlimited, HealthCheck
# useful hypothesis decorator
# @example(arg0=..., arg1=..., ...)
# @settings(max_examples=10000, deadline=None)
from hypothesis.strategies import integers, decimals, sampled_from, booleans

from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.operation import BvComp, SecondaryOperation
from cascada.bitvector.context import Validation, Simplification

from cascada.abstractproperty.tests.test_opmodel import TestOpModelGeneric as AbstractTestOpModelGeneric

from cascada.differential.difference import XorDiff, RXDiff, Difference
from cascada.differential import opmodel
from cascada.differential.opmodel import (
    XorModelBvAdd, XorModelBvSub, XorModelBvAddCt, XorModelBvSubCt,
    XorModelBvOr, XorModelBvAnd, XorModelBvIf, XorModelBvMaj,
    RXModelBvAdd, RXModelBvSub, RXModelBvOr, RXModelBvAnd,
    RXModelBvIf, RXModelBvMaj, log2_decimal, make_partial_op_model,
    RXModelBvShlCt, RXModelBvLshrCt,
    get_weak_model, get_branch_number_model, get_wdt_model
)

from cascada.differential.tests import preimageXDA
from cascada.differential.tests import preimageRXDA
from cascada.differential.tests import preimageXDAC
from cascada.differential.tests import preimageXDS


SKIP_LONG_TESTS = True

DIFF_WIDTH = 8
assert DIFF_WIDTH <= 8
MAX_SMALL_WIDTH = 4
MAX_PRECISION_RXDA = 4
MAX_PRECISION_XDAC = 4

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
    def find_preimage_slow(cls, f, beta):
        """Return the first preimage found of the given model with given output difference (None otherwise)."""
        assert isinstance(beta, Difference) and isinstance(beta.val, Constant)

        width = f.input_diff[0].val.width
        assert all(d.val.width == width for d in f.input_diff)

        num_input_diff = len(f.input_diff)

        with Simplification(False), Validation(False):
            for input_vals in itertools.product(range(2 ** width), repeat=num_input_diff):
                input_vals = [Constant(val, width) for val in input_vals]
                output_diff = f.eval_derivative(*input_vals)
                assert isinstance(output_diff.val, Constant)
                if output_diff == beta:
                    return input_vals
            else:
                return None

    @classmethod
    def count_preimages_slow(cls, f, beta):
        """Count the number of preimages of the given model with given output difference """
        assert isinstance(beta, Difference) and isinstance(beta.val, Constant)

        width = f.input_diff[0].val.width
        assert all(d.val.width == width for d in f.input_diff)

        num_input_diff = len(f.input_diff)

        num_preimages = 0

        with Simplification(False), Validation(False):
            for input_vals in itertools.product(range(2 ** width), repeat=num_input_diff):
                input_vals = [Constant(val, width) for val in input_vals]
                output_diff = f.eval_derivative(*input_vals)
                assert isinstance(output_diff.val, Constant)
                if output_diff == beta:
                    num_preimages += 1

        return num_preimages

    @classmethod
    def is_valid_slow(cls, f, beta):
        if hasattr(cls, "find_preimage"):
            return cls.find_preimage(f, beta) is not None
        else:
            return cls.find_preimage_slow(f, beta) is not None

    @classmethod
    def get_empirical_weight_slow(cls, f, beta):
        if hasattr(cls, "count_preimages"):
            num_preimages = cls.count_preimages(f, beta)
        else:
            num_preimages = cls.count_preimages_slow(f, beta)

        assert num_preimages != 0
        width = f.input_diff[0].val.width
        assert all(d.val.width == width for d in f.input_diff)
        num_input_diff = len(f.input_diff)
        total_preimages = 2 ** (num_input_diff * width)
        return - log2_decimal(decimal.Decimal(num_preimages) / total_preimages)

    def get_bv_weight_with_frac_bits(self, f, beta):
        """Get the bit-vector weight of the given model with given output difference."""
        output_diff = beta
        if isinstance(f, XorModelBvAddCt):
            u, v = f.input_prop[0].val, output_diff.val
            if f._effective_width <= 1:
                int_frac, extra_constraints = Constant(0, 1), []
            else:
                u, v = XorDiff(u[:f._index_first_one]), XorDiff(v[:f._index_first_one])
                if isinstance(f, XorModelBvSubCt):
                    reduced_model = make_partial_op_model(XorModelBvSubCt, (None, f.ct[:f._index_first_one]))
                else:
                    reduced_model = make_partial_op_model(XorModelBvAddCt, (None, f.ct[:f._index_first_one]))
                reduced_model = reduced_model(u)
                int_frac, extra_constraints = reduced_model._bvweight_and_extra_constraints(
                    v, prefix="_tmp", debug=False, version=2)
            assert len(extra_constraints) == 0
            bv_weight_with_frac_bits = decimal.Decimal(int(int_frac))
        else:
            bv_weight_with_frac_bits = decimal.Decimal(int(f.bv_weight(output_diff)))
        return bv_weight_with_frac_bits


class TestOpModelsSmallWidth(TestOpModelGeneric):
    """Common tests for all differential models with small difference width."""

    def test_vrepr(self):
        diffs = [XorDiff(Variable("d" + str(i), 8)) for i in range(4)]
        XorModelBvAddCte_1r = make_partial_op_model(XorModelBvAddCt, (None, Constant(1, 8)))
        XorModelBvAddCte_1l = make_partial_op_model(XorModelBvAddCt, (Constant(1, 8), None))
        XorModelBvSubCte_1 = make_partial_op_model(XorModelBvSubCt, (None, Constant(1, 8)))
        for model in [XorModelBvAdd, XorModelBvSub, XorModelBvAddCte_1r, XorModelBvAddCte_1l, XorModelBvSubCte_1,
                      XorModelBvOr, XorModelBvAnd, XorModelBvIf, XorModelBvMaj]:
            num_input_diff = model.op.arity[0]
            model_fixed = model(diffs[:num_input_diff])
            # str used since they are not singleton objects
            self.assertEqual(str(model_fixed), str(eval(model_fixed.vrepr())))

        diffs = [RXDiff(Variable("d" + str(i), 8)) for i in range(4)]
        RXModelBvShlCte_1 = make_partial_op_model(RXModelBvShlCt, (None, Constant(1, 8)))
        RXModelBvLshrCte_1 = make_partial_op_model(RXModelBvLshrCt, (None, Constant(1, 8)))
        for model in [RXModelBvAdd, RXModelBvSub,
                      RXModelBvOr, RXModelBvAnd, RXModelBvIf, RXModelBvMaj,
                      RXModelBvShlCte_1, RXModelBvLshrCte_1]:
            num_input_diff = model.op.arity[0]
            model_fixed = model(diffs[:num_input_diff])
            self.assertEqual(str(model_fixed), str(eval(model_fixed.vrepr())))

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_Xor_models_slow")
    @given(
        integers(min_value=2, max_value=MAX_SMALL_WIDTH),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
    )
    @settings(deadline=None, max_examples=1000)  # default 100
    def test_Xor_models_slow(self, width, d1, d2, d3, d4):
        # width = MAX_SMALL_WIDTH

        # ct max value is 1 + (2**width - 2) = 2**width - 1
        ct = Constant(1 + (d3 % (2**width - 1)), width)

        d1 = XorDiff(Constant(d1 % 2**width, width))
        d2 = XorDiff(Constant(d2 % 2**width, width))
        d3 = XorDiff(Constant(d3 % 2**width, width))
        d4 = XorDiff(Constant(d4 % 2**width, width))

        XorModelBvAddCte_ctr = make_partial_op_model(XorModelBvAddCt, (None, ct))
        XorModelBvAddCte_ctl = make_partial_op_model(XorModelBvAddCt, (ct, None))
        XorModelBvSubCte_ct = make_partial_op_model(XorModelBvSubCt, (None, ct))

        for model in [XorModelBvAdd, XorModelBvSub, XorModelBvAddCte_ctr, XorModelBvAddCte_ctl, XorModelBvSubCte_ct,
                      XorModelBvOr, XorModelBvAnd, XorModelBvIf, XorModelBvMaj]:
            num_input_diff = model.op.arity[0]
            input_diff = [d1, d2, d3][:num_input_diff]
            output_diff = d4
            self.base_test_op_model(model, input_diff, output_diff)
            self.base_test_pr_one_constraint_slow(model, input_diff, output_diff)
            if output_diff.val == 0:
                self.base_test_op_model_sum_pr_1(model, input_diff)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_RX_models_slow")
    @given(
        integers(min_value=2, max_value=MAX_SMALL_WIDTH),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
        integers(min_value=0, max_value=2 ** MAX_SMALL_WIDTH - 1),
    )
    @settings(deadline=None, max_examples=1000)
    def test_RX_models_slow(self, width, d1, d2, d3, d4):
        # width = MAX_SMALL_WIDTH

        # ct max value is 1 + (width - 2) = width - 1
        ct = Constant(1 + (d3 % (width - 1)), width)

        d1 = RXDiff(Constant(d1 % 2**width, width))
        d2 = RXDiff(Constant(d2 % 2**width, width))
        d3 = RXDiff(Constant(d3 % 2**width, width))
        d4 = RXDiff(Constant(d4 % 2**width, width))

        RXModelBvShlCte_ct = make_partial_op_model(RXModelBvShlCt, (None, ct))
        RXModelBvLshrCte_ct = make_partial_op_model(RXModelBvLshrCt, (None, ct))

        for model in [RXModelBvAdd, RXModelBvSub,
                      RXModelBvOr, RXModelBvAnd, RXModelBvIf, RXModelBvMaj,
                      RXModelBvShlCte_ct, RXModelBvLshrCte_ct]:
            num_input_diff = model.op.arity[0]
            input_diff = [d1, d2, d3][:num_input_diff]
            output_diff = d4
            self.base_test_op_model(model, input_diff, output_diff)
            self.base_test_pr_one_constraint_slow(model, input_diff, output_diff)
            if output_diff.val == 0:
                self.base_test_op_model_sum_pr_1(model, input_diff)


class TestBvAddModels(TestOpModelGeneric):
    """Tests for the OpModel of BvAdd."""

    @classmethod
    def setUpClass(cls):
        module_name = "_preimageXDA"
        ffibuilderXOR = cffi.FFI()
        ffibuilderXOR.cdef(preimageXDA.header)
        ffibuilderXOR.set_source(module_name, preimageXDA.source)

        cls.tmpdirnameXorDiff = tempfile.TemporaryDirectory()
        lib_path = ffibuilderXOR.compile(tmpdir=cls.tmpdirnameXorDiff.name, verbose=VERBOSE)
        spec = importlib.util.spec_from_file_location(module_name, lib_path)
        lib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib_module)
        cls.libXorDiff = lib_module.lib

        module_name = "_preimageRXDA"
        ffibuilderRX = cffi.FFI()
        ffibuilderRX.cdef(preimageRXDA.header)
        ffibuilderRX.set_source(module_name, preimageRXDA.source)

        cls.tmpdirnameRX = tempfile.TemporaryDirectory()
        lib_path = ffibuilderRX.compile(tmpdir=cls.tmpdirnameRX.name, verbose=VERBOSE)
        spec = importlib.util.spec_from_file_location(module_name, lib_path)
        lib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib_module)
        cls.libRX = lib_module.lib

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdirnameXorDiff.cleanup()
        cls.tmpdirnameRX.cleanup()

        super().tearDownClass()

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_find_preimage")
    @given(
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
    )
    @settings(deadline=None)
    def test_find_preimage_Ccode(self, d1, d2, d3):
        """Check that the C and the python code are equivalent."""
        width = 8

        d1 = Constant(d1, width)
        d2 = Constant(d2, width)
        d3 = Constant(d3, width)

        for diff_type, model_type in zip([XorDiff, RXDiff], [XorModelBvAdd, RXModelBvAdd]):
            alpha = diff_type(d1), diff_type(d2)
            beta = diff_type(d3)
            f = model_type(alpha)

            msg = "{}({} -> {})\n".format(model_type.__name__, alpha, beta)

            if diff_type == XorDiff:
                if width == 8:
                    foo = self.__class__.libXorDiff.find_XOR_preimage_8bit
                elif width == 16:
                    continue
            elif diff_type == RXDiff:
                if width == 8:
                    foo = self.__class__.libRX.find_RX_preimage_8bit
                elif width == 16:
                    foo = self.__class__.libRX.find_RX_preimage_16bit

            result_lib_bool = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            result_lib_bool = result_lib_bool.found

            result_slow = self.__class__.find_preimage_slow(f, beta)

            self.assertEqual(result_lib_bool, result_slow is not None, msg=msg)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_count_preimage")
    @given(
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
    )
    @settings(deadline=None)
    def test_count_preimage_Ccode(self, d1, d2, d3):
        """Check that the C and the python code are equivalent."""
        width = 8

        d1 = Constant(d1, width)
        d2 = Constant(d2, width)
        d3 = Constant(d3, width)

        for diff_type, model_type in zip([XorDiff, RXDiff], [XorModelBvAdd, RXModelBvAdd]):
            alpha = diff_type(d1), diff_type(d2)
            beta = diff_type(d3)
            f = model_type(alpha)

            msg = "{}({} -> {})\n".format(model_type.__name__, alpha, beta)

            if diff_type == XorDiff:
                if width == 8:
                    foo = self.__class__.libXorDiff.count_XOR_preimage_8bit
                elif width == 16:
                    continue
            elif diff_type == RXDiff:
                if width == 8:
                    foo = self.__class__.libRX.count_RX_preimage_8bit
                elif width == 16:
                    foo = self.__class__.libRX.count_RX_preimage_16bit

            result_lib = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            result_lib = int(result_lib)

            result_slow = self.__class__.count_preimages_slow(f, beta)

            self.assertEqual(result_lib, result_slow, msg=msg)

    @classmethod
    def find_preimage(cls, f, beta):
        """Calls C code if width == 8, otherwise find_preimage_slow()."""
        width = f.input_diff[0].val.width

        if width == 8 and f.diff_type == XorDiff:
            foo = cls.libXorDiff.find_XOR_preimage_8bit
        elif width == 8 and f.diff_type == RXDiff:
            foo = cls.libRX.find_RX_preimage_8bit
        elif width == 16 and f.diff_type == RXDiff:
            foo = cls.libRX.find_RX_preimage_16bit
        else:
            foo = None

        if foo is not None:
            result = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            if result.found:
                return result.i, result.j
            else:
                return None
        else:
            return cls.find_preimage_slow(f, beta)

    @classmethod
    def count_preimages(cls, f, beta):
        """Calls C code if width == 8, otherwise count_preimages_slow()."""
        width = f.input_diff[0].val.width

        if width == 8 and f.diff_type == XorDiff:
            foo = cls.libXorDiff.count_XOR_preimage_8bit
        elif width == 8 and f.diff_type == RXDiff:
            foo = cls.libRX.count_RX_preimage_8bit
        elif width == 16 and f.diff_type == RXDiff:
            foo = cls.libRX.count_RX_preimage_16bit
        else:
            foo = None

        if foo is not None:
            result = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            return int(result)
        else:
            return cls.count_preimages_slow(f, beta)

    @given(
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
    )
    @settings(max_examples=1000)
    def test_XDA(self, d1, d2, d3):
        # d1, d2 input differences; d3 output difference
        d1 = XorDiff(Constant(d1, DIFF_WIDTH))
        d2 = XorDiff(Constant(d2, DIFF_WIDTH))
        d3 = XorDiff(Constant(d3, DIFF_WIDTH))

        self.base_test_op_model(XorModelBvAdd, [d1, d2], d3)
        self.base_test_pr_one_constraint_slow(XorModelBvAdd, [d1, d2], d3)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_RXDA")
    @given(
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=MAX_PRECISION_RXDA),
    )
    @settings(deadline=None, max_examples=200)
    def test_RXDA(self, d1, d2, d3, precision):
        old_precision = RXModelBvAdd.precision
        RXModelBvAdd.precision = precision

        # hack to also try w16 when w8
        list_widths = [DIFF_WIDTH]
        if DIFF_WIDTH == 8:
            list_widths.append(16)

        for width in list_widths:
            input_diff = [RXDiff(Constant(d_i, width)) for d_i in [d1, d2]]
            output_diff = RXDiff(Constant(d3, width))

            self.base_test_op_model(RXModelBvAdd, input_diff, output_diff)
            self.base_test_pr_one_constraint_slow(RXModelBvAdd, input_diff, output_diff)

        RXModelBvAdd.precision = old_precision


class TestModelBvSub(TestOpModelGeneric):
    """Tests for the OpModel of BvSub."""

    @classmethod
    def setUpClass(cls):
        module_name = "_preimageXDS"
        ffibuilderXOR = cffi.FFI()
        ffibuilderXOR.cdef(preimageXDS.header)
        ffibuilderXOR.set_source(module_name, preimageXDS.source)

        cls.tmpdirnameXorDiff = tempfile.TemporaryDirectory()
        lib_path = ffibuilderXOR.compile(tmpdir=cls.tmpdirnameXorDiff.name, verbose=VERBOSE)
        spec = importlib.util.spec_from_file_location(module_name, lib_path)
        lib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib_module)
        cls.libXorDiff = lib_module.lib

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdirnameXorDiff.cleanup()

        super().tearDownClass()

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_find_preimage")
    @given(
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
    )
    @settings(deadline=None)
    def test_find_preimage_Ccode(self, d1, d2, d3):
        """Check that the C and the python code are equivalent."""
        width = 8

        d1 = Constant(d1, width)
        d2 = Constant(d2, width)
        d3 = Constant(d3, width)

        for diff_type, model_type in zip([XorDiff], [XorModelBvSub]):
            alpha = diff_type(d1), diff_type(d2)
            beta = diff_type(d3)
            f = model_type(alpha)

            msg = "{}({} -> {})\n".format(model_type.__name__, alpha, beta)

            if diff_type == XorDiff:
                foo = self.__class__.libXorDiff.find_XOR_preimage_8bit
            elif diff_type == RXDiff:
                assert False

            result_lib_bool = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            result_lib_bool = result_lib_bool.found

            result_slow = self.__class__.find_preimage_slow(f, beta)

            self.assertEqual(result_lib_bool, result_slow is not None, msg=msg)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_count_preimage")
    @given(
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
    )
    @settings(deadline=None)
    def test_count_preimage_Ccode(self, d1, d2, d3):
        """Check that the C and the python code are equivalent."""
        width = 8

        d1 = Constant(d1, width)
        d2 = Constant(d2, width)
        d3 = Constant(d3, width)

        for diff_type, model_type in zip([XorDiff], [XorModelBvSub]):
            alpha = diff_type(d1), diff_type(d2)
            beta = diff_type(d3)
            f = model_type(alpha)

            msg = "{}({} -> {})\n".format(model_type.__name__, alpha, beta)

            if diff_type == XorDiff:
                foo = self.__class__.libXorDiff.count_XOR_preimage_8bit
            elif diff_type == RXDiff:
                assert False

            result_lib = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            result_lib = int(result_lib)

            result_slow = self.__class__.count_preimages_slow(f, beta)

            self.assertEqual(result_lib, result_slow, msg=msg)

    @classmethod
    def find_preimage(cls, f, beta):
        """Calls C code if width == 8, otherwise find_preimage_slow()."""
        width = f.input_diff[0].val.width

        if width == 8 and f.diff_type == XorDiff:
            foo = cls.libXorDiff.find_XOR_preimage_8bit
        else:
            foo = None

        if foo is not None:
            result = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            if result.found:
                return result.i, result.j
            else:
                return None
        else:
            return cls.find_preimage_slow(f, beta)

    @classmethod
    def count_preimages(cls, f, beta):
        """Calls C code if width == 8, otherwise count_preimages_slow()."""
        width = f.input_diff[0].val.width

        if width == 8 and f.diff_type == XorDiff:
            foo = cls.libXorDiff.count_XOR_preimage_8bit
        else:
            foo = None

        if foo is not None:
            result = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            return int(result)
        else:
            return cls.count_preimages_slow(f, beta)

    @given(
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
    )
    @settings(max_examples=1000)
    def test_XDS(self, d1, d2, d3):
        # d1, d2 input differences; d3 output difference
        d1 = XorDiff(Constant(d1, DIFF_WIDTH))
        d2 = XorDiff(Constant(d2, DIFF_WIDTH))
        d3 = XorDiff(Constant(d3, DIFF_WIDTH))

        self.base_test_op_model(XorModelBvSub, [d1, d2], d3)
        self.base_test_pr_one_constraint_slow(XorModelBvSub, [d1, d2], d3)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_RXDS")
    @given(
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=MAX_PRECISION_RXDA),
    )
    @settings(deadline=None, max_examples=1000)
    def test_RXDS(self, d1, d2, d3, precision):
        width = DIFF_WIDTH

        old_precision = RXModelBvSub.precision
        RXModelBvSub.precision = precision

        input_diff = [RXDiff(Constant(d_i, width)) for d_i in [d1, d2]]
        output_diff = RXDiff(Constant(d3, width))

        self.base_test_op_model(RXModelBvSub, input_diff, output_diff)
        self.base_test_pr_one_constraint_slow(RXModelBvSub, input_diff, output_diff)

        RXModelBvSub.precision = old_precision


class TestBvAddCteModel(TestOpModelGeneric):
    """Tests for the XorDiff OpModel of BvAddCte."""

    @classmethod
    def setUpClass(cls):
        module_name = "_preimageXDAC"
        ffibuilderXOR = cffi.FFI()
        ffibuilderXOR.cdef(preimageXDAC.header)
        ffibuilderXOR.set_source(module_name, preimageXDAC.source)

        cls.tmpdirnameXorDiff = tempfile.TemporaryDirectory()
        lib_path = ffibuilderXOR.compile(tmpdir=cls.tmpdirnameXorDiff.name, verbose=VERBOSE)
        spec = importlib.util.spec_from_file_location(module_name, lib_path)
        lib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib_module)
        cls.libXorDiff = lib_module.lib

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdirnameXorDiff.cleanup()

        super().tearDownClass()

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_find_preimage")
    @given(
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=1, max_value=2 ** 8 - 1),
    )
    @settings(deadline=None)
    def test_find_preimage_Ccode(self, d1, d2, ct):
        """Check that the C and the python code are equivalent."""
        width = 8

        d1 = Constant(d1, width)
        d2 = Constant(d2, width)
        ct = Constant(ct, width)

        diff_type = XorDiff

        XorModelBvAddCte_ctr = make_partial_op_model(XorModelBvAddCt, (None, ct))
        XorModelBvAddCte_ctl = make_partial_op_model(XorModelBvAddCt, (ct, None))
        for model_type in [XorModelBvAddCte_ctr, XorModelBvAddCte_ctl]:
            alpha = diff_type(d1)
            beta = diff_type(d2)
            f = model_type(alpha)

            msg = "{}({} -> {})\n".format(model_type.__name__, alpha, beta)

            foo = self.__class__.libXorDiff.find_XOR_preimage_8bit

            result_lib_bool = foo(f.input_diff[0].val, beta.val, f.ct)
            result_lib_bool = result_lib_bool.found

            result_slow = self.__class__.find_preimage_slow(f, beta)

            self.assertEqual(result_lib_bool, result_slow is not None, msg=msg)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_count_preimage")
    @given(
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=1, max_value=2 ** 8 - 1),
    )
    @settings(deadline=None)
    def test_count_preimage_Ccode(self, d1, d2, ct):
        """Check that the C and the python code are equivalent."""
        width = 8

        d1 = Constant(d1, width)
        d2 = Constant(d2, width)
        ct = Constant(ct, width)

        diff_type = XorDiff

        XorModelBvAddCte_ctr = make_partial_op_model(XorModelBvAddCt, (None, ct))
        XorModelBvAddCte_ctl = make_partial_op_model(XorModelBvAddCt, (ct, None))
        for model_type in [XorModelBvAddCte_ctr, XorModelBvAddCte_ctl]:
            alpha = diff_type(d1)
            beta = diff_type(d2)
            f = model_type(alpha)

            msg = "{}({} -> {})\n".format(model_type.__name__, alpha, beta)

            foo = self.__class__.libXorDiff.count_XOR_preimage_8bit

            result_lib = foo(f.input_diff[0].val, beta.val, f.ct)
            result_lib = int(result_lib)

            result_slow = self.__class__.count_preimages_slow(f, beta)

            self.assertEqual(result_lib, result_slow, msg=msg)

    @classmethod
    def find_preimage(cls, f, beta):
        """Calls C code if width == 8, otherwise find_preimage_slow()."""
        width = f.input_diff[0].val.width

        if width == 8 and f.diff_type == XorDiff:
            foo = cls.libXorDiff.find_XOR_preimage_8bit
        else:
            foo = None

        if foo is not None:
            result = foo(f.input_diff[0].val, beta.val, f.ct)
            if result.found:
                return result.i
            else:
                return None
        else:
            return cls.find_preimage_slow(f, beta)

    @classmethod
    def count_preimages(cls, f, beta):
        """Calls C code if width == 8, otherwise count_preimages_slow()."""
        width = f.input_diff[0].val.width

        if width == 8 and f.diff_type == XorDiff:
            foo = cls.libXorDiff.count_XOR_preimage_8bit
        else:
            foo = None

        if foo is not None:
            result = foo(f.input_diff[0].val, beta.val, f.ct)
            return int(result)
        else:
            return cls.count_preimages_slow(f, beta)

    @given(
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=1, max_value=2 ** DIFF_WIDTH - 1),
        integers(min_value=0, max_value=MAX_PRECISION_XDAC),
    )
    @example(d1=0x04, d2=0x04, ct=0x01, k=3)
    @example(d1=137, d2=137, ct=138, k=3)
    @example(d1=208, d2=208, ct=1, k=3)
    @example(d1=206, d2=206, ct=1, k=3)
    @example(d1=96, d2=96, ct=97, k=3)
    @example(d1=32, d2=32, ct=31, k=3)
    @example(d1=0x08, d2=0x08, ct=0x7b, k=3)
    @example(d1=8, d2=8, ct=14, k=3)
    @example(d1=64, d2=64, ct=21, k=3)
    @settings(max_examples=1000)
    def test_XDAC(self, d1, d2, ct, k):
        if d1 < 2**DIFF_WIDTH and d2 < 2**DIFF_WIDTH and ct < 2**DIFF_WIDTH:
            precision = k
            old_precision = XorModelBvAddCt.precision
            XorModelBvAddCt.precision = precision

            # d1 input differences; d2 output difference
            d1 = XorDiff(Constant(d1, DIFF_WIDTH))
            d2 = XorDiff(Constant(d2, DIFF_WIDTH))
            ct = Constant(ct, DIFF_WIDTH)

            XorModelBvAddCte_ctr = make_partial_op_model(XorModelBvAddCt, (None, ct))
            XorModelBvAddCte_ctl = make_partial_op_model(XorModelBvAddCt, (ct, None))
            for op_model in [XorModelBvAddCte_ctr, XorModelBvAddCte_ctl]:
                self.base_test_op_model(op_model, [d1], d2)
                self.base_test_pr_one_constraint_slow(op_model, [d1], d2)

            XorModelBvAddCt.precision = old_precision

    @unittest.skip("skipping testing all cases of XorModelBvAddCt")
    def test_all_cases(self):
        for ct in range(1, 2 ** DIFF_WIDTH):
            print("ct:", ct)
            ct = Constant(ct, DIFF_WIDTH)
            om = make_partial_op_model(XorModelBvAddCt, (None, ct))
            for d1, d2 in itertools.product(range(2 ** DIFF_WIDTH), range(2 ** DIFF_WIDTH)):
                d1 = XorDiff(Constant(d1, DIFF_WIDTH))
                d2 = XorDiff(Constant(d2, DIFF_WIDTH))
                self.base_test_op_model(om, [d1], d2)
                self.base_test_pr_one_constraint_slow(om, [d1], d2)
            super().tearDownClass()  # print errors


class TestOtherOpModels(unittest.TestCase):
    """Test differential WeakModel and BranchNumberModel."""

    def _check_weak_model(self, my_model, width, num_inputs, verbose=False, write_msg=False):
        prop_type = my_model.prop_type
        precision = my_model.precision
        zero2zero_weight = my_model.zero2zero_weight
        nonzero2nonzero_weight = my_model.nonzero2nonzero_weight
        nonzero2zero_weight = my_model.nonzero2zero_weight
        zero2nonzero_weight = my_model.zero2nonzero_weight

        if write_msg:
            msg = f"{my_model.__name__}, "
            msg += f"w: {width}, num_inputs:{num_inputs}, "
            msg += f"prec: {precision}, "
            msg += f"zero2zero_weight_w: {zero2zero_weight}, "
            msg += f"nonzero2nonzero_w: {nonzero2nonzero_weight}, "
            msg += f"nonzero2zero_w: {nonzero2zero_weight}, "
            msg += f"zero2nonzero_w: {zero2nonzero_weight}"
            if verbose:
                print(msg)
        else:
            msg = ""

        with Simplification(False):
            for input_prop in itertools.product(range(2 ** width), repeat=num_inputs):
                input_prop = [prop_type(Constant(val, width)) for val in input_prop]
                f = my_model(input_prop)
                for output_prop in range(2 ** width):
                    output_prop = prop_type(Constant(output_prop, width))

                    if write_msg == "debug":
                        msg += f"\n{f}(out={output_prop})"

                    is_valid = f.validity_constraint(output_prop)
                    is_pr_one = f.pr_one_constraint(output_prop)
                    if is_valid:
                        decimal_weight = f.decimal_weight(output_prop)

                    if all(p.val == 0 for p in input_prop):
                        if output_prop.val == 0:
                            self.assertEqual(is_valid, zero2zero_weight != math.inf, msg=msg)
                            self.assertEqual(is_pr_one, zero2zero_weight == 0, msg=msg)
                            if is_valid:
                                self.assertLessEqual(abs(decimal_weight - zero2zero_weight),
                                                     0 if zero2zero_weight == int(zero2zero_weight)
                                                     else 2 ** (-precision), msg=msg)
                        else:
                            self.assertEqual(is_valid, zero2nonzero_weight != math.inf, msg=msg)
                            self.assertEqual(is_pr_one, zero2nonzero_weight == 0, msg=msg)
                            if is_valid:
                                self.assertLessEqual(abs(decimal_weight - zero2nonzero_weight),
                                                     0 if zero2nonzero_weight == int(zero2nonzero_weight)
                                                     else 2 ** (-precision), msg=msg)
                    else:
                        if output_prop.val == 0:
                            self.assertEqual(is_valid, nonzero2zero_weight != math.inf, msg=msg)
                            self.assertEqual(is_pr_one, nonzero2zero_weight == 0, msg=msg)
                            if is_valid:
                                self.assertLessEqual(abs(decimal_weight - nonzero2zero_weight),
                                                     0 if nonzero2zero_weight == int(nonzero2zero_weight)
                                                     else 2 ** (-precision), msg=msg)
                        else:
                            self.assertEqual(is_valid, nonzero2nonzero_weight != math.inf, msg=msg)  # nonzero2nonzero
                            self.assertEqual(is_pr_one, nonzero2nonzero_weight == 0, msg=msg)
                            if is_valid:
                                self.assertLessEqual(abs(decimal_weight - nonzero2nonzero_weight),
                                                     0 if nonzero2nonzero_weight == int(nonzero2nonzero_weight)
                                                     else 2 ** (-precision), msg=msg)

    def _check_branchnumber_model(self, my_model, num_inputs, verbose=False, write_msg=False):
        width = my_model.output_widths[0]
        num_outputs = len(my_model.output_widths)
        prop_type = my_model.prop_type
        branch_number = my_model.branch_number
        precision = my_model.precision
        zero2zero_weight = my_model.zero2zero_weight
        nonzero2nonzero_weight = my_model.nonzero2nonzero_weight
        nonzero2zero_weight = my_model.nonzero2zero_weight
        zero2nonzero_weight = my_model.zero2nonzero_weight

        if write_msg:
            msg = f"{my_model.__name__}, "
            msg += f"w: {width}, num_inputs:{num_inputs}, num_outputs: {num_outputs}, "
            msg += f"branch_number: {branch_number}, "
            msg += f"prec: {precision}, "
            msg += f"zero2zero_weight_w: {zero2zero_weight}, "
            msg += f"nonzero2nonzero_w: {nonzero2nonzero_weight}, "
            msg += f"nonzero2zero_w: {nonzero2zero_weight}, "
            msg += f"zero2nonzero_w: {zero2nonzero_weight}"
            if verbose:
                print(msg)
        else:
            msg = ""

        with Simplification(False):
            for input_prop in itertools.product(range(2 ** width), repeat=num_inputs):
                input_prop = [prop_type(Constant(val, width)) for val in input_prop]
                f = my_model(input_prop)
                for output_prop in range(2 ** (num_outputs * width)):
                    output_prop = prop_type(Constant(output_prop, num_outputs * width))

                    is_valid = f.validity_constraint(output_prop)
                    is_pr_one = f.pr_one_constraint(output_prop)
                    if is_valid:
                        decimal_weight = f.decimal_weight(output_prop)

                    num_active = 0
                    for p in input_prop:
                        if p.val != 0:
                            num_active += 1
                    for v in f._split(output_prop.val):
                        if v != 0:
                            num_active += 1

                    if write_msg == "debug":
                        msg += f"\n{f}(out={output_prop})  // num_active: {num_active}"

                    if all(p.val == 0 for p in input_prop):
                        if output_prop.val == 0:
                            self.assertEqual(is_valid, zero2zero_weight != math.inf, msg=msg)
                            self.assertEqual(is_pr_one, zero2zero_weight == 0, msg=msg)
                            if is_valid:
                                self.assertLessEqual(abs(decimal_weight - zero2zero_weight),
                                                     0 if zero2zero_weight == int(zero2zero_weight)
                                                     else 2 ** (-precision), msg=msg)
                        else:
                            self.assertEqual(is_valid, zero2nonzero_weight != math.inf and num_active >= branch_number, msg=msg)
                            self.assertEqual(is_pr_one, zero2nonzero_weight == 0 and num_active >= branch_number, msg=msg)
                            if is_valid:
                                self.assertLessEqual(abs(decimal_weight - zero2nonzero_weight),
                                                     0 if zero2nonzero_weight == int(zero2nonzero_weight)
                                                     else 2 ** (-precision), msg=msg)
                    else:
                        if output_prop.val == 0:
                            self.assertEqual(is_valid, nonzero2zero_weight != math.inf and num_active >= branch_number, msg=msg)
                            self.assertEqual(is_pr_one, nonzero2zero_weight == 0 and num_active >= branch_number, msg=msg)
                            if is_valid:
                                self.assertLessEqual(abs(decimal_weight - nonzero2zero_weight),
                                                     0 if nonzero2zero_weight == int(nonzero2zero_weight)
                                                     else 2 ** (-precision), msg=msg)
                        else:
                            self.assertEqual(is_valid, nonzero2nonzero_weight != math.inf and num_active >= branch_number, msg=msg)
                            self.assertEqual(is_pr_one, nonzero2nonzero_weight == 0 and num_active >= branch_number, msg=msg)
                            if is_valid:
                                self.assertLessEqual(abs(decimal_weight - nonzero2nonzero_weight),
                                                     0 if nonzero2nonzero_weight == int(nonzero2nonzero_weight)
                                                     else 2 ** (-precision), msg=msg)

    def _check_wdt_model(self, my_model, seed, verbose=False, write_msg=False):
        input_width = int(math.log2(len(my_model.weight_distribution_table)))
        output_width = int(math.log2(len(my_model.weight_distribution_table[0])))
        prop_type = my_model.prop_type
        WDT = my_model.weight_distribution_table
        wdt_str = "\n  ".join("( " + ', '.join([f'{w:.1f}' for w in row])  + " )" for row in WDT)
        loop_rows_then_columns = my_model.loop_rows_then_columns
        precision = my_model.precision

        if write_msg:
            msg = f"{my_model.__name__}, "
            msg += f"input_width:{input_width}, output_width: {output_width}, "
            msg += f"lrtc: {loop_rows_then_columns}, "
            msg += f"prec: {precision}, seed_wdt={seed}, "
            msg += f"wdt:\n  {wdt_str}"
            if verbose:
                print(msg)
        else:
            msg = ""

        with Simplification(False):
            for input_val in range(2 ** input_width):
                input_prop = prop_type(Constant(input_val, input_width))
                f = my_model(input_prop)
                for output_val in range(2 ** output_width):
                    output_prop = prop_type(Constant(output_val, output_width))

                    if write_msg == "debug":
                        msg += f"\n{f}(out={output_prop})"

                    is_valid = f.validity_constraint(output_prop)
                    is_pr_one = f.pr_one_constraint(output_prop)
                    if is_valid:
                        decimal_weight = f.decimal_weight(output_prop)

                    self.assertEqual(is_valid, WDT[input_val][output_val] != math.inf, msg=msg)
                    self.assertEqual(is_pr_one, WDT[input_val][output_val] == 0, msg=msg)
                    if is_valid:
                        w = WDT[input_val][output_val]  # non-approximated decimal value
                        self.assertLessEqual(abs(decimal_weight - w), 0 if w == int(w) else 2 ** (-precision), msg=msg)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_weak_model")
    @given(
        sampled_from([XorDiff, RXDiff]),
        integers(min_value=1, max_value=3),
        integers(min_value=2, max_value=3),
        decimals(min_value=0, max_value=3),  # 3 converted to inf
        decimals(min_value=0, max_value=3),  # 3 converted to inf
        decimals(min_value=0, max_value=3),  # 3 converted to inf
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=1000)
    def test_weak_model(self, diff_type, num_inputs, width, nonzero2nonzero_weight, zero2zero_weight, nonzero2zero_weight, precision):
        assume(not(diff_type == XorDiff and zero2zero_weight != 0))

        if nonzero2nonzero_weight == 3:
            nonzero2nonzero_weight = math.inf
        if zero2zero_weight == 3:
            zero2zero_weight = math.inf
        if nonzero2zero_weight == 3:
            nonzero2zero_weight = math.inf

        for w in [nonzero2nonzero_weight, zero2zero_weight, nonzero2zero_weight]:
            assume(not (0 != w != math.inf and int(w * (2 ** precision)) == 0))
        assume(not
            (precision != 0 and
                (nonzero2nonzero_weight == math.inf or nonzero2nonzero_weight == int(nonzero2nonzero_weight)) and
                (zero2zero_weight == math.inf or zero2zero_weight == int(zero2zero_weight)) and
                (nonzero2zero_weight == math.inf or nonzero2zero_weight == int(nonzero2zero_weight))
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
            MyOp, diff_type, nonzero2nonzero_weight, zero2zero_weight=zero2zero_weight,
            nonzero2zero_weight=nonzero2zero_weight, precision=precision
        )

        self._check_weak_model(WeakModel, width, num_inputs, verbose=VERBOSE)  # , write_msg=True)

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_branchnumber_model")
    @given(
        sampled_from([XorDiff, RXDiff]),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=2, max_value=3),
        integers(min_value=2, max_value=5),
        decimals(min_value=0, max_value=3),  # 3 converted to inf
        decimals(min_value=0, max_value=3),  # 3 converted to inf
        decimals(min_value=0, max_value=3),  # 3 converted to inf
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=100)
    def test_branchnumber_model(self, diff_type, num_inputs, num_outputs, width,
                                branch_number, nonzero2nonzero_weight, zero2zero_weight, nonzero2zero_weight, precision):
        assume(not(diff_type == XorDiff and zero2zero_weight != 0))
        assume(branch_number <= num_inputs + num_outputs)

        if nonzero2nonzero_weight == 3:
            nonzero2nonzero_weight = math.inf
        if zero2zero_weight == 3:
            zero2zero_weight = math.inf
        if nonzero2zero_weight == 3:
            nonzero2zero_weight = math.inf

        for w in [nonzero2nonzero_weight, zero2zero_weight, nonzero2zero_weight]:
            assume(not (0 != w != math.inf and int(w * (2 ** precision)) == 0))
        assume(not
            (precision != 0 and
                (nonzero2nonzero_weight == math.inf or nonzero2nonzero_weight == int(nonzero2nonzero_weight)) and
                (zero2zero_weight == math.inf or zero2zero_weight == int(zero2zero_weight)) and
                (nonzero2zero_weight == math.inf or nonzero2zero_weight == int(nonzero2zero_weight))
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
            MyOp, diff_type, tuple([width for _ in range(num_outputs)]), branch_number,
            nonzero2nonzero_weight, zero2zero_weight=zero2zero_weight,
            nonzero2zero_weight=nonzero2zero_weight, precision=precision
        )

        self._check_branchnumber_model(BranchNumberModel, num_inputs, verbose=VERBOSE)  # , write_msg=True)

    @classmethod
    def setUpClass(cls):
        cls.PRNG = random.Random()

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping test_wdt_model")
    @given(
        sampled_from([XorDiff, RXDiff]),
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=0),
        booleans(),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=10000)
    def test_wdt_model(self, diff_type, input_width, output_width, seed, lrtc, precision):
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
            assume(not (diff_type == XorDiff and all(w == math.inf for w in wdt[i])))
        assume(not(not found_frac_bits and precision != 0))
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
            MyOp, diff_type, wdt, loop_rows_then_columns=bool(lrtc), precision=precision
        )

        self._check_wdt_model(WDTModel, seed, verbose=VERBOSE)  # , write_msg=True)


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(opmodel))
    return tests
