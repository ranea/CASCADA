"""Tests for the chmodel and the characteristic modules."""
import collections
import decimal
import itertools
import math
import random
import unittest
import doctest
import warnings

from wurlitzer import pipes

from hypothesis import given, example, settings, assume
from hypothesis.strategies import integers

from cascada.bitvector import core, operation, secondaryop
from cascada.bitvector import ssa as cascada_ssa
from cascada.differential import difference
from cascada.differential import opmodel as cascada_opmodel
from cascada.differential import chmodel as cascada_chmodel
from cascada.differential import characteristic

from cascada.abstractproperty.tests.test_ch import TestRandomChModelCharacteristicGeneric


SKIP_LONG_TESTS = True
USE_C_CODE = True

VERBOSE = False
WRITE_MSG = False


def _get_extra_models(extra_operations, prop_type, seed):
    assert [op.__name__ for op in extra_operations] == ['RandomLut', 'RandomMatrix', 'Inversion']

    RandomLut, RandomMatrix, Inversion = extra_operations

    width = int(math.log2(len(RandomLut.lut)))

    PRNG = random.Random()
    PRNG.seed(seed)

    nz2nz = decimal.Decimal(PRNG.uniform(0, width))  # diff. max weight width

    if prop_type == difference.XorDiff:
        z2nz = math.inf
    else:
        z2nz = decimal.Decimal(PRNG.uniform(0, width + 1))
        if z2nz >= width:
            z2nz = math.inf

    nz2z = decimal.Decimal(PRNG.uniform(0, width + 1))
    if nz2z >= width:
        nz2z = math.inf

    prec = PRNG.randint(0, 3)
    if int(nz2nz * (2 ** prec)) == 0:
        nz2nz = 0
    if 0 != z2nz != math.inf and int(z2nz * (2 ** prec)) == 0:
        z2nz = 0
    if 0 != nz2z != math.inf and int(nz2z * (2 ** prec)) == 0:
        nz2z = 0
    if all(w == 0 or w == math.inf for w in [nz2nz, z2nz, nz2z]):
        prec = 0

    WeakModelRandomLut = cascada_opmodel.get_weak_model(
        RandomLut, prop_type, nz2nz,
        zero2nonzero_weight=z2nz, nonzero2zero_weight=nz2z, precision=prec)

    output_widths = []
    while sum(output_widths) != width:
        output_widths.append(PRNG.randint(1, width - sum(output_widths)))
    output_widths = tuple(output_widths)
    branch_number = PRNG.randint(2, RandomMatrix.arity[0] + len(output_widths))

    BranchNumberModelRandomMatrix = cascada_opmodel.get_branch_number_model(
        RandomMatrix, prop_type, output_widths, branch_number, nz2nz,
        zero2nonzero_weight=z2nz, nonzero2zero_weight=nz2z, precision=prec)

    if prop_type == difference.RXDiff:
        WDTModelInversion = cascada_opmodel.get_weak_model(
            Inversion, prop_type, nz2nz,
            zero2nonzero_weight=z2nz, nonzero2zero_weight=nz2z, precision=prec)
    else:
        # from sage.crypto.sbox import SBox
        # ddt = list(SBox(inv_lut).difference_distribution_table())
        if width == 2:
            ddt = [(4, 0, 0, 0), (0, 4, 0, 0), (0, 0, 0, 4), (0, 0, 4, 0)]
        else:
            assert width == 3
            ddt = [
                (8, 0, 0, 0, 0, 0, 0, 0),
                (0, 2, 0, 2, 0, 2, 0, 2),
                (0, 0, 0, 0, 2, 2, 2, 2),
                (0, 2, 0, 2, 2, 0, 2, 0),
                (0, 0, 2, 2, 0, 0, 2, 2),
                (0, 2, 2, 0, 0, 2, 2, 0),
                (0, 0, 2, 2, 2, 2, 0, 0),
                (0, 2, 2, 0, 2, 0, 0, 2)
            ]

        l2 = cascada_opmodel.log2_decimal
        num_inputs = decimal.Decimal(2 ** width)
        wdt = tuple([tuple(math.inf if x == 0 else -l2(x / num_inputs) for x in row) for row in ddt])

        lrtc = bool(PRNG.randint(0, 1))

        prec = 0
        for row in wdt:
            for w in row:
                if w != math.inf and w != int(w):
                    prec = PRNG.randint(0, 3)
                    break

        WDTModelInversion = cascada_opmodel.get_wdt_model(
            Inversion, prop_type, wdt,
            loop_rows_then_columns=lrtc, precision=prec)

    return WeakModelRandomLut, BranchNumberModelRandomMatrix, WDTModelInversion


class TestSimpleTests(unittest.TestCase):
    """Simple tests for the ChModel and Characteristic classes."""

    def test_get_differences_for_initialization(self):
        from cascada.primitives import speck
        Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        Speck32_KS.set_num_rounds(2)
        xor_ch_model = cascada_chmodel.ChModel(Speck32_KS, difference.XorDiff, ["dmk0", "dmk1", "dmk2"])
        zd = difference.XorDiff(core.Constant(0, width=16))
        mk0 = operation.RotateLeft(core.Constant(1, width=16), 5)
        rr2 = operation.RotateLeft(core.Constant(1, width=16), 14)
        xor_ch = characteristic.Characteristic([mk0, zd, zd], [zd, zd, rr2],
                                [zd, rr2, zd, zd, rr2], xor_ch_model)
        var2ct = {v.val: d.val for v, d in xor_ch.var_diff2ct_diff.items()}
        self.assertEqual(
            str(characteristic.Characteristic.get_properties_for_initialization(xor_ch_model, var2ct)),
            "([XorDiff(0x0020), XorDiff(0x0000), XorDiff(0x0000)], "
            "[XorDiff(0x0000), XorDiff(0x0000), XorDiff(0x4000)], "
            "[], "
            "[XorDiff(0x0000), XorDiff(0x4000), XorDiff(0x0000), XorDiff(0x0000), XorDiff(0x4000)])"
        )

    def test_compute_empirical_ch_weight_C_code(self):
        from cascada.primitives import speck

        Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        Speck32_KS.set_num_rounds(2)
        rx_ch_model = cascada_chmodel.ChModel(Speck32_KS, difference.RXDiff, ["dmk0", "dmk1", "dmk2"])
        zd = difference.RXDiff(core.Constant(0, width=16))
        td = difference.RXDiff(core.Constant(3, width=16))
        rx_ch = characteristic.Characteristic([zd, zd, zd], [zd, zd, td], [zd, zd, zd, zd, td], rx_ch_model)

        num_input_samples, num_external_samples = rx_ch._get_empirical_data_complexity(None, None)

        self.assertEqual(0, num_external_samples)

        with warnings.catch_warnings():
            # cffi warnings ignored
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            # capturing C-level piped output:
            with pipes() as (out, err):
                num_right_inputs_C_code = rx_ch._get_empirical_ch_weights_C(
                    num_input_samples, num_external_samples, seed=0,
                    num_parallel_processes=None, get_sampled_inputs=True)

        list_input_samples = eval("[" + out.read() + "]")

        self.assertEqual(len(list_input_samples), num_input_samples)
        self.assertEqual(len(list_input_samples[0]), len(rx_ch.input_diff))

        num_right_inputs_Python = rx_ch._get_empirical_ch_weights_Python(
            num_input_samples, num_external_samples, seed=0,
            list_input_samples=list_input_samples)

        self.assertEqual(num_right_inputs_C_code, num_right_inputs_Python)

    def test_Extract_with_BranchNumberModel(self):
        from cascada.abstractproperty.property import make_partial_propextract

        _matrix = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

        class ReverseMatrix(secondaryop.MatrixOperation):
            matrix = [[core.Constant(x, 1) for x in row] for row in _matrix]

        ReverseMatrix.xor_model = cascada_opmodel.get_branch_number_model(
            ReverseMatrix, difference.XorDiff, (1, 1, 1), 2, 1)
        ReverseMatrix.rx_model = cascada_opmodel.get_branch_number_model(
            ReverseMatrix, difference.RXDiff, (1, 1, 1), 2, 1)

        Extract0 = make_partial_propextract(0, 0)
        Extract1 = make_partial_propextract(1, 1)
        Extract2 = make_partial_propextract(2, 2)

        class MyFunc_v1(cascada_ssa.BvFunction):
            input_widths, output_widths = [3], [1, 1, 1]
            @classmethod
            def eval(cls, x):
                y = ReverseMatrix(x)
                return Extract0(y), Extract1(y), Extract2(y)

        class MyFunc_v2(MyFunc_v1):
            @classmethod
            def eval(cls, x):
                y = ReverseMatrix(x)
                return y[0], y[1], y[2]

        for diff_type in [difference.XorDiff, difference.RXDiff]:
            with self.assertRaises((ValueError,)):
                cascada_chmodel.ChModel(MyFunc_v2, diff_type, ["x"], prefix="da")
            cascada_chmodel.ChModel(MyFunc_v1, diff_type, ["x"], prefix="da")


class TestXorChModelCharacteristic(TestRandomChModelCharacteristicGeneric):
    """Test for the ChModel and Characteristic classes."""
    USE_C_CODE = USE_C_CODE
    VERBOSE = VERBOSE
    WRITE_MSG = WRITE_MSG

    prop_type = difference.XorDiff
    prop_prefix = "d"
    operation_set_index = 0
    check_get_empirical_ch_weights = True

    ChModel = cascada_chmodel.ChModel
    EncryptionChModel = cascada_chmodel.EncryptionChModel
    CipherChModel = cascada_chmodel.CipherChModel

    Characteristic = characteristic.Characteristic
    EncryptionCharacteristic = characteristic.EncryptionCharacteristic
    CipherCharacteristic = characteristic.CipherCharacteristic

    eg = {k: v for k, v in itertools.chain(
        TestRandomChModelCharacteristicGeneric.eg.items(),
        vars(difference).items(), vars(cascada_chmodel).items(), vars(characteristic).items()
    ) if not k.startswith('_')}

    #

    _to_obtain_extra_operations = True

    def _obtain_extra_operations(self, width, seed):
        extra_operations = super()._obtain_extra_operations(width, seed)
        if not extra_operations:
            return extra_operations
        extra_models = _get_extra_models(extra_operations, self.__class__.prop_type, seed)
        assert len(extra_operations) == len(extra_models) > 0
        for op, new_model in zip(extra_operations, extra_models):
            if self.__class__.prop_type == difference.XorDiff:
                op.xor_model = new_model
            else:
                assert self.__class__.prop_type == difference.RXDiff
                op.rx_model = new_model
            self.eg[new_model.__name__] = new_model
        return extra_operations

    #

    def _get_to_full_verify(self, ch_model):
        WRITE_MSG = self.__class__.WRITE_MSG

        to_full_verify, msg = super()._get_to_full_verify(ch_model)

        if to_full_verify:
            for _, op in ch_model.ssa.assignments.items():
                # don't allow partial operations that can reduce the weight
                # (e.g., shifts/and/or reduce the weight as they fix part of the difference)
                if ch_model.prop_type == difference.XorDiff and \
                        issubclass(op.__class__, operation.PartialOperation) and \
                        op.base_op in {operation.BvLshr, operation.BvShl, operation.BvAnd, operation.BvOr}:
                    if WRITE_MSG:
                        msg += f"\nfound bad partial operation {op} for to_full_verify"
                    to_full_verify = False
                    break

                # don't allow BvSub with left operand fixed (not accurately implemented)
                if ch_model.prop_type == difference.XorDiff and \
                        issubclass(op.__class__, operation.PartialOperation) and \
                        op.base_op == operation.BvSub and op.fixed_args[0] is not None:
                    if WRITE_MSG:
                        msg += f"\nfound BvSub with left operand fixed ({op}) for to_full_verify"
                    to_full_verify = False
                    break

                # don't allow constant BvAdd, BvSub, BvAnd or BvOr with RXDiff
                if ch_model.prop_type == difference.RXDiff and \
                        issubclass(op.__class__, operation.PartialOperation) and \
                        op.base_op in {operation.BvAdd, operation.BvSub, operation.BvAnd, operation.BvOr}:
                    if WRITE_MSG:
                        msg += f"\nfound bad partial operation {op} for to_full_verify"
                    to_full_verify = False
                    break

                # don't allow BvNeg RXDiff
                if ch_model.prop_type == difference.RXDiff and type(op) == operation.BvNeg:
                    if WRITE_MSG:
                        msg += f"\nfound bad partial operation {op} for to_full_verify"
                    to_full_verify = False
                    break

        return to_full_verify, msg

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping XOR test_random_ch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=20*10 if USE_C_CODE else 200*10)  # ~10 min
    def test_random_ch(self, width, num_inputs, num_outputs, num_assignments,
                       seed, func_type_index, num_rounds):
        return super()._test_random_ch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


class TestRXChModelCharacteristic(TestXorChModelCharacteristic):
    """Test for the ChModel and Characteristic classes."""
    prop_type = difference.RXDiff
    operation_set_index = 1

    def _get_external_var2prop(self, ssa):
        return collections.OrderedDict(
            [v, difference.RXDiff(core.Constant(0, v.width))] for v in ssa.external_vars
        )

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping RXDiff test_random_ch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=20*10 if USE_C_CODE else 200*10)
    def test_random_ch(self, width, num_inputs, num_outputs, num_assignments,
                       seed, func_type_index, num_rounds):
        assume(not func_type_index == 1)  # EncryptionCh doesn't support RXDiff
        return super()._test_random_ch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(cascada_chmodel))
    tests.addTests(doctest.DocTestSuite(characteristic))
    return tests
