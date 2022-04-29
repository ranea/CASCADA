"""Tests for the chmodel and the characteristic modules."""
import itertools
import unittest
import doctest

from hypothesis import given, example, settings, assume
from hypothesis.strategies import integers

from cascada.algebraic import value
from cascada.algebraic import chmodel as cascada_chmodel
from cascada.algebraic import characteristic

from cascada.abstractproperty.tests.test_ch import TestRandomChModelCharacteristicGeneric


SKIP_LONG_TESTS = True

VERBOSE = False
WRITE_MSG = False


class TestBitChModelCharacteristic(TestRandomChModelCharacteristicGeneric):
    """Test for the ChModel and Characteristic classes."""
    USE_C_CODE = False
    VERBOSE = VERBOSE
    WRITE_MSG = WRITE_MSG

    prop_type = value.BitValue
    prop_prefix = "v"
    operation_set_index = 3
    check_get_empirical_ch_weights = False

    ChModel = cascada_chmodel.ChModel
    EncryptionChModel = cascada_chmodel.EncryptionChModel
    CipherChModel = cascada_chmodel.CipherChModel

    Characteristic = characteristic.Characteristic
    EncryptionCharacteristic = characteristic.EncryptionCharacteristic
    CipherCharacteristic = characteristic.CipherCharacteristic

    eg = {k: v for k, v in itertools.chain(
        TestRandomChModelCharacteristicGeneric.eg.items(),
        vars(value).items(), vars(cascada_chmodel).items(), vars(characteristic).items()
    ) if not k.startswith('_')}

    def _get_to_full_verify(self, ch_model):
        return True, ""

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping BitValue test_random_ch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=5000)  # ~10min
    def test_random_ch(self, width, num_inputs, num_outputs, num_assignments,
                       seed, func_type_index, num_rounds):
        return super()._test_random_ch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


class TestWordChModelCharacteristic(TestBitChModelCharacteristic):
    """Test for the ChModel and Characteristic classes."""
    prop_type = value.WordValue

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping WordValue test_random_ch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=5000)  # ~10min
    def test_random_ch(self, width, num_inputs, num_outputs, num_assignments,
                       seed, func_type_index, num_rounds):
        return super()._test_random_ch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(cascada_chmodel))
    tests.addTests(doctest.DocTestSuite(characteristic))
    return tests
