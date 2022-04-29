"""Tests for the chmodel and the characteristic modules."""
import decimal
import itertools
import math
import unittest
import doctest
import random

from hypothesis import given, example, settings, assume
from hypothesis.strategies import integers

from cascada.bitvector import operation, secondaryop
from cascada.bitvector import core
from cascada.bitvector import ssa as cascada_ssa
from cascada import abstractproperty
from cascada.linear import mask
from cascada.linear import opmodel as cascada_opmodel
from cascada.linear import chmodel as cascada_chmodel
from cascada.linear import characteristic

from cascada.abstractproperty.tests.test_ch import TestRandomChModelCharacteristicGeneric


SKIP_LONG_TESTS = True
USE_C_CODE = True

VERBOSE = False
WRITE_MSG = False


def _get_extra_models(extra_operations, prop_type, seed):
    assert prop_type == mask.LinearMask
    assert [op.__name__ for op in extra_operations] == ['RandomLut', 'RandomMatrix', 'Inversion']

    RandomLut, RandomMatrix, Inversion = extra_operations

    width = int(math.log2(len(RandomLut.lut)))

    PRNG = random.Random()
    PRNG.seed(seed)

    nz2nz = decimal.Decimal(PRNG.uniform(0, width - 1))  # linear max. weight

    z2nz = decimal.Decimal(PRNG.uniform(0, width))
    if z2nz >= width - 1:
        z2nz = math.inf

    prec = PRNG.randint(0, 3)
    if int(nz2nz * (2 ** prec)) == 0:
        nz2nz = 0
    if 0 != z2nz != math.inf and int(z2nz * (2 ** prec)) == 0:
        z2nz = 0
    if all(w == 0 or w == math.inf for w in [nz2nz, z2nz]):
        prec = 0

    WeakModelRandomLut = cascada_opmodel.get_weak_model(
        RandomLut, nz2nz,
        zero2nonzero_weight=z2nz, precision=prec)

    output_widths = []
    while sum(output_widths) != width:
        output_widths.append(PRNG.randint(1, width - sum(output_widths)))
    output_widths = tuple(output_widths)
    branch_number = PRNG.randint(2, RandomMatrix.arity[0] + len(output_widths))

    BranchNumberModelRandomMatrix = cascada_opmodel.get_branch_number_model(
        RandomMatrix, output_widths, branch_number, nz2nz,
        zero2nonzero_weight=z2nz, precision=prec)

    assert width in [2, 3]

    # from sage.crypto.sbox import SBox
    # lat = list(SBox(inv_lut).linear_approximation_table(scale="correlation"))
    if width == 2:
        lat = [(1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0)]
    else:
        lat = [
            (1, 0, 0, 0, 0, 0, 0, 0),
            (0, -1 / 2, 0, 1 / 2, 0, 1 / 2, 0, 1 / 2),
            (0, 0, 0, 0, 1 / 2, 1 / 2, 1 / 2, -1 / 2),
            (0, 1 / 2, 0, 1 / 2, -1 / 2, 0, 1 / 2, 0),
            (0, 0, 1 / 2, -1 / 2, 0, 0, 1 / 2, 1 / 2),
            (0, 1 / 2, 1 / 2, 0, 0, 1 / 2, -1 / 2, 0),
            (0, 0, 1 / 2, 1 / 2, 1 / 2, -1 / 2, 0, 0),
            (0, 1 / 2, -1 / 2, 0, 1 / 2, 0, 0, 1 / 2)
        ]

    l2 = abstractproperty.opmodel.log2_decimal
    wdt = tuple([tuple(math.inf if x == 0 else -l2(decimal.Decimal(abs(x))) for x in row) for row in lat])

    lrtc = bool(PRNG.randint(0, 1))

    prec = 0
    for row in wdt:
        for w in row:
            if w != math.inf and w != int(w):
                prec = PRNG.randint(0, 3)
                break

    WDTModelInversion = cascada_opmodel.get_wdt_model(
        Inversion, wdt,
        loop_rows_then_columns=lrtc, precision=prec)

    return WeakModelRandomLut, BranchNumberModelRandomMatrix, WDTModelInversion


class TestSimpleTests(unittest.TestCase):
    """Simple tests for the ChModel and Characteristic classes."""

    def test_input_vars_not_used(self):
        class MyFunc(cascada_ssa.BvFunction):
            input_widths, output_widths = [2, 2], [2]
            @classmethod
            def eval(cls, x, y):
                return (x, )

        with self.assertRaises((ValueError)):
            # linear characteristic models of functions with unused input vars is not supported
            cascada_chmodel.ChModel(MyFunc, mask.LinearMask, ["x", "y"], prefix="da")

        # # deprecated
        # zm = mask.LinearMask(core.Constant(0, 2))
        # om = mask.LinearMask(core.Constant(1, 2))
        #
        # ch = characteristic.Characteristic([zm, zm], [zm], [zm], ch_model)
        # ch.compute_empirical_ch_weight(num_input_samples=math.inf, num_external_samples=math.inf, seed=0)
        # for ed in ch.empirical_data_list:
        #     self.assertSequenceEqual(ed.aux_weights, [0] * ed.num_aux_weights)
        #
        # # 2nd input mask is a free prop that cannot be non-zero (invalid ch in this case)
        # ch = characteristic.Characteristic([zm, om], [zm], [zm], ch_model)
        # ch.compute_empirical_ch_weight(num_input_samples=math.inf, num_external_samples=math.inf, seed=0)
        # for ed in ch.empirical_data_list:
        #     self.assertSequenceEqual(ed.aux_weights, [math.inf] * ed.num_aux_weights)


class TestLinearChModelCharacteristic(TestRandomChModelCharacteristicGeneric):
    """Test for the ChModel and Characteristic classes."""
    USE_C_CODE = USE_C_CODE
    VERBOSE = VERBOSE
    WRITE_MSG = WRITE_MSG

    prop_type = mask.LinearMask
    prop_prefix = "m"
    operation_set_index = 2
    check_get_empirical_ch_weights = True

    ChModel = cascada_chmodel.ChModel
    EncryptionChModel = cascada_chmodel.EncryptionChModel
    CipherChModel = type("CipherChModel")

    Characteristic = characteristic.Characteristic
    EncryptionCharacteristic = characteristic.EncryptionCharacteristic
    CipherCharacteristic = type("CipherCharacteristic")

    eg = {k: v for k, v in itertools.chain(
        TestRandomChModelCharacteristicGeneric.eg.items(),
        vars(mask).items(), vars(cascada_chmodel).items(), vars(characteristic).items()
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
            op.linear_model = new_model
            self.eg[new_model.__name__] = new_model
        return extra_operations

    #

    def _valid_ssa(self, ssa):
        if ssa._input_vars_not_used:
            return False
        for expr in ssa.assignments.values():
            if isinstance(expr, operation.BvNeg):
                return False
            if issubclass(expr.__class__, operation.PartialOperation) and \
                    (expr.base_op in [operation.BvAdd, operation.BvSub] or
                     issubclass(expr.base_op, (secondaryop.LutOperation, secondaryop.MatrixOperation))):
                return False
        else:
            return super()._valid_ssa(ssa)

    def _get_to_full_verify(self, ch_model):
        WRITE_MSG = self.__class__.WRITE_MSG

        to_full_verify, msg = super()._get_to_full_verify(ch_model)

        # don't allow partial operations that can reduce the weight
        # (e.g., shifts/and/or reduce the weight as they fix part of the difference)
        if to_full_verify:
            for _, op in ch_model.ssa.assignments.items():
                if issubclass(op.__class__, operation.PartialOperation) and \
                        op.base_op in {operation.BvLshr, operation.BvShl, operation.BvAnd, operation.BvOr}:
                    to_full_verify = False
                    if WRITE_MSG:
                        msg += f"\nfound bad partial operation {op} for to_full_verify"
                    break

        return to_full_verify, msg

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping linear test_random_ch")
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
        assume(not func_type_index == 2)  # CipherCh don't support LinearMask
        return super()._test_random_ch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(cascada_chmodel))
    tests.addTests(doctest.DocTestSuite(characteristic))
    return tests
