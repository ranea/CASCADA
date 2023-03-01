"""Tests the automated search for characteristic over cryptographic primitives."""
import warnings
from datetime import datetime
import collections
import functools
import os
import unittest

from cascada.differential.difference import XorDiff, RXDiff
from cascada.linear.mask import LinearMask
from cascada.algebraic.value import BitValue, WordValue

from cascada.smt.chsearch import (
    ChModelAssertType, PrintingMode, INCREMENT_NUM_ROUNDS, MissingVarWarning,
    round_based_ch_search, round_based_cipher_ch_search,
    _get_smart_print
)
from cascada.smt.invalidpropsearch import (
    round_based_invalidprop_search, round_based_invalidcipherprop_search
)

from cascada.primitives import blockcipher

from cascada.primitives import aes
from cascada.primitives import cham
from cascada.primitives import chaskey
from cascada.primitives import feal
from cascada.primitives import hight
from cascada.primitives import lea
from cascada.primitives import multi2
from cascada.primitives import noekeon
from cascada.primitives import picipher
from cascada.primitives import rectangle
from cascada.primitives import shacal1
from cascada.primitives import shacal2
from cascada.primitives import speck
from cascada.primitives import simeck
from cascada.primitives import simon
from cascada.primitives import skinny64
from cascada.primitives import skinny128
from cascada.primitives import tea
from cascada.primitives import xtea


SKIP_LONG_TESTS = True
SKIP_INVALID_LONG_TESTS = True
OUTPUT_FILE = False
PRINTING_MODE = PrintingMode.Silent
COMPUTE_EW = True


CHAM64 = cham.get_CHAM_instance(cham.CHAMInstance.CHAM_64_128)
Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
Simeck32 = simeck.get_Simeck_instance(simeck.SimeckInstance.simeck_32_64)
Simon32 = simon.get_Simon_instance(simon.SimonInstance.simon_32_64)


# function can be BvFunction or Cipher
# xor_nr is the maximum number of rounds with probability-one XOR characteristics
# cipher_xor_nr is the maximum number of rounds with probability-one XOR cipher characteristics
# (nr=True means probability-one characteristics for many number of rounds)
# (nr=False means no probability-one characteristics even for the minimum number of rounds)
# (nr=None means no search is done)
SearchParameters = collections.namedtuple(
    'SearchParameters', ['function', 'xor_nr', 'rk_nr', 'linear_nr',
                         'cipher_xor_nr', 'cipher_rx_nr',
                         'ignore_cipher_xor_invalid', 'ignore_cipher_rx_invalid', 'ignore_linear_invalid',
                         'ignore_bitvalue', 'ignore_wordvalue'],
    defaults=[None, None, None, None, None, None, None, None, None, None]  # all except function
)


ListSearchParameters = [
    SearchParameters(aes.AESCipher, xor_nr=False, linear_nr=False, cipher_xor_nr=1, ignore_bitvalue=True, ignore_cipher_rx_invalid=True),
    SearchParameters(aes.AESCipher.key_schedule, xor_nr=3, linear_nr=True, ignore_bitvalue=True),
    SearchParameters(CHAM64, xor_nr=4, rk_nr=None, linear_nr=7, cipher_xor_nr=10, cipher_rx_nr=False),
    SearchParameters(CHAM64.key_schedule, xor_nr=True, rk_nr=True),  # linear_nr=True but unused input vars
    SearchParameters(chaskey.ChaskeyPi, xor_nr=1, rk_nr=False, linear_nr=1),
    SearchParameters(feal.FEALCipher, xor_nr=3, rk_nr=None, linear_nr=None, cipher_xor_nr=3, cipher_rx_nr=None),  # BvAddCt in Enc
    SearchParameters(feal.FEALCipher.key_schedule, xor_nr=1, rk_nr=None, linear_nr=None),  # BvAddCt in KS
    SearchParameters(hight.HightCipher, xor_nr=3, rk_nr=None, linear_nr=4, cipher_xor_nr=7, cipher_rx_nr=False),  # nr+1 due to key-whitening
    SearchParameters(hight.HightCipher.key_schedule, xor_nr=True, rk_nr=1, linear_nr=None),  # BvAddCt
    SearchParameters(lea.LEACipher, xor_nr=2, rk_nr=None, linear_nr=2),  # cipher_xor_nr=2 but 10+min, cipher_rk_nr discards many ch due to EW
    SearchParameters(lea.LEACipher.key_schedule, xor_nr=2, rk_nr=None, linear_nr=None),  # BvAddCt, xor_nr slow, rk discards many ch due to EW
    SearchParameters(multi2.MULTI2Cipher, xor_nr=2, rk_nr=None, linear_nr=3, cipher_xor_nr=2, cipher_rx_nr=False),
    SearchParameters(multi2.MULTI2Cipher.key_schedule, xor_nr=True, rk_nr=False, linear_nr=None),  # BvAddCt
    SearchParameters(noekeon.NOEKEONDirectCipher, xor_nr=False, linear_nr=False, cipher_xor_nr=1, ignore_cipher_rx_invalid=True, ignore_linear_invalid=True, ignore_bitvalue=True),  # (rx requires non-WDT model, and then cipher_rx_nr=2 but 10+min)
    SearchParameters(noekeon.NOEKEONIndirectCipher, xor_nr=False, linear_nr=False, cipher_xor_nr=False, ignore_cipher_rx_invalid=True, ignore_linear_invalid=True, ignore_bitvalue=True),  # (rx requires non-WDT model, and then cipher_rx_nr=1 but 10min)
    SearchParameters(noekeon.NOEKEONIndirectCipher.key_schedule, xor_nr=False, rk_nr=None, linear_nr=False, ignore_bitvalue=True),  # (rx requires non-WDT model)
    SearchParameters(picipher.PiPermutation, xor_nr=False, rk_nr=None, linear_nr=None),  #   # weight rk > 200, BvAddCt
    SearchParameters(rectangle.RECTANGLE80Cipher, xor_nr=False, rk_nr=None, linear_nr=False, cipher_xor_nr=False, cipher_rx_nr=None, ignore_linear_invalid=True, ignore_bitvalue=True),  # cipher_rx_nr=1 but 10min,
    SearchParameters(rectangle.RECTANGLE80Cipher.key_schedule, xor_nr=False, rk_nr=1, linear_nr=False, ignore_bitvalue=True),
    SearchParameters(rectangle.RECTANGLE128Cipher, xor_nr=False, rk_nr=None, linear_nr=False, cipher_xor_nr=False, cipher_rx_nr=None, ignore_linear_invalid=True, ignore_bitvalue=True),  # cipher_rx_nr=1 but 10min,
    SearchParameters(rectangle.RECTANGLE128Cipher.key_schedule, xor_nr=False, rk_nr=1, linear_nr=False, ignore_bitvalue=True),
    SearchParameters(shacal1.SHACAL1Cipher, xor_nr=False, cipher_xor_nr=True),  # xor 1r but mnr=4, BvIf, cipher_xor 16r but slow, rk discards many ch due to EW
    SearchParameters(shacal1.SHACAL1Cipher.key_schedule, xor_nr=True, rk_nr=False, linear_nr=None),  # BvAddCt, xor 19r but slow
    SearchParameters(shacal2.SHACAL2Cipher, xor_nr=False, cipher_xor_nr=True),  # xor 1r but mnr=4, BvIf, cipher_xor 18r but slow, rk discards many ch due to EW
    SearchParameters(shacal2.SHACAL2Cipher.key_schedule, xor_nr=True, rk_nr=False, linear_nr=None),  # BvAddCt, xor 18r but slow
    SearchParameters(Speck32, xor_nr=1, rk_nr=None, linear_nr=2, cipher_xor_nr=2, cipher_rx_nr=False),
    SearchParameters(Speck32.key_schedule, xor_nr=4, rk_nr=False, linear_nr=True),
    SearchParameters(Simeck32, xor_nr=1, rk_nr=None, linear_nr=None, cipher_xor_nr=5, cipher_rx_nr=6, ignore_bitvalue=True),
    SearchParameters(Simeck32.key_schedule, xor_nr=7, rk_nr=8, linear_nr=None, ignore_bitvalue=True),
    SearchParameters(Simon32, xor_nr=1, rk_nr=None, linear_nr=None, cipher_xor_nr=5, cipher_rx_nr=6, ignore_bitvalue=True),
    SearchParameters(Simon32.key_schedule, xor_nr=True, rk_nr=True, linear_nr=None, ignore_bitvalue=True),  # linear ks
    SearchParameters(skinny64.SKINNYCipher, xor_nr=False, linear_nr=False, ignore_bitvalue=True, ignore_cipher_xor_invalid=True, ignore_cipher_rx_invalid=True),  # cipher_xor_nr requires ignore_first_sub_cells
    SearchParameters(skinny64.SKINNYCipher.key_schedule, xor_nr=True, linear_nr=True, ignore_bitvalue=True),
    SearchParameters(skinny128.SKINNYCipher, xor_nr=False, linear_nr=False, ignore_bitvalue=True, ignore_cipher_xor_invalid=True, ignore_cipher_rx_invalid=True),  # cipher_xor_nr requires ignore_first_sub_cells
    SearchParameters(skinny128.SKINNYCipher.key_schedule, xor_nr=True, linear_nr=True, ignore_bitvalue=True),
    SearchParameters(tea.TEACipher, xor_nr=False, rk_nr=None, linear_nr=None, cipher_xor_nr=True, cipher_rx_nr=None, ignore_cipher_xor_invalid=True),  # BvAddCt, rk huge error
    SearchParameters(xtea.XTEACipher, xor_nr=1, rk_nr=None, linear_nr=3, cipher_xor_nr=8, cipher_rx_nr=False, ignore_cipher_xor_invalid=True),
    SearchParameters(xtea.XTEACipher.key_schedule, xor_nr=True, rk_nr=1, linear_nr=None),  # BvAddCt
]

zip = functools.partial(zip, strict=True)


class TestPrimitivesSearch(unittest.TestCase):
    """Test automated characteristic search of the primitives implemented."""

    @classmethod
    def setUpClass(cls):
        if OUTPUT_FILE:
            date_string = datetime.strftime(datetime.now(), '%d-%H-%M')
            filename = "output_test_primitives_smt" + date_string + ".txt"
            assert not os.path.isfile(filename)
            cls.filename = filename
        else:
            cls.filename = None
            warnings.filterwarnings("ignore", category=ImportWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=MissingVarWarning)

    @unittest.skipIf(SKIP_LONG_TESTS, "test_ch_search")
    def test_ch_search(self):
        extra_ch_finder_args = {
            "raise_exception_missing_var": False,
            "exclude_zero_input_prop": True,
            "printing_mode": PRINTING_MODE,
            "filename": self.__class__.filename,
        }

        extra_findnextchweight_args = None
        if COMPUTE_EW:
            extra_findnextchweight_args = {
                "initial_weight": 0,
                "empirical_weight_options": {"C_code": True, "split_by_max_weight": 10}
            }

        if PRINTING_MODE != PrintingMode.Silent:
            smart_print = _get_smart_print(self.__class__.filename)
            smart_print(f"# {self.test_ch_search.__name__}")

        for sp in ListSearchParameters:
            f = sp.function
            for nr, prop_type in zip(
                    [sp.xor_nr, sp.rk_nr, sp.linear_nr, True, True],
                    [XorDiff, RXDiff, LinearMask, BitValue, WordValue]
            ):
                if nr is None:
                    continue

                for at in [ChModelAssertType.ProbabilityOne, ChModelAssertType.ValidityAndWeight]:
                    if prop_type in [BitValue, WordValue] and at == ChModelAssertType.ValidityAndWeight:
                        continue
                    if prop_type == BitValue and sp.ignore_bitvalue is True:
                        continue
                    if prop_type == WordValue and sp.ignore_wordvalue is True:
                        continue

                    msg = f"\n## {f.__name__}, nr: {nr}, prop_type: {prop_type.__name__}, at: {at}"
                    if PRINTING_MODE != PrintingMode.Silent:
                        smart_print(msg)

                    round_ch_iterator = round_based_ch_search(
                        f, nr if (nr is not True and nr is not False) else getattr(f, "_min_num_rounds", 1),
                        nr + 1 if (nr is not True and nr is not False) else getattr(f, "_min_num_rounds", 1) + 1,
                        prop_type, at, "btor",
                        extra_chfinder_args=extra_ch_finder_args,
                        extra_findnextchweight_args=extra_findnextchweight_args
                    )

                    if nr is not False:
                        try:
                            num_rounds_found, ch_found = next(round_ch_iterator)
                        except StopIteration:
                            raise StopIteration(msg)
                        target_nr = nr if nr is not True else getattr(f, "_min_num_rounds", 1)
                        self.assertEqual(num_rounds_found, target_nr, msg=msg)

                        if PRINTING_MODE != PrintingMode.Silent:
                            if PRINTING_MODE == PrintingMode.WeightsAndSrepr:
                                smart_print(ch_found.srepr())
                            else:
                                smart_print(ch_found)
                            smart_print("\n".join(ch_found.get_formatted_logged_msgs()))

                        round_ch_iterator.send(INCREMENT_NUM_ROUNDS)

                        msg += "\n||| probability-one characteristic found"

                    if nr is not True:
                        try:
                            num_rounds_found, ch_found = next(round_ch_iterator)
                        except StopIteration:
                            num_rounds_found, ch_found = None, None

                        if at == ChModelAssertType.ProbabilityOne:
                            self.assertIsNone(num_rounds_found, msg=msg)
                            self.assertIsNone(ch_found, msg=msg)
                        else:
                            if nr is False:
                                # INCREMENT_NUM_ROUNDS not sent
                                target_nr = getattr(f, "_min_num_rounds", 1)
                            else:
                                target_nr = nr + 1
                            self.assertEqual(num_rounds_found, target_nr, msg=msg)
                            self.assertGreater(ch_found.ch_weight, 0, msg=msg)

                            if PRINTING_MODE != PrintingMode.Silent:
                                if PRINTING_MODE == PrintingMode.WeightsAndSrepr:
                                    smart_print(ch_found.srepr())
                                else:
                                    smart_print(ch_found)
                                smart_print("\n".join(ch_found.get_formatted_logged_msgs()))

    @unittest.skipIf(SKIP_LONG_TESTS, "test_cipher_ch_search")
    def test_cipher_ch_search(self):
        extra_cipherchfinder_args = {
            "raise_exception_missing_var": False,
            "ks_exclude_zero_input_prop": False,
            "enc_exclude_zero_input_prop": True,
            "printing_mode": PRINTING_MODE,
            "filename": self.__class__.filename,
        }

        extra_findnextchweight_args = None
        if COMPUTE_EW:
            extra_findnextchweight_args = {
                "initial_weight": 0,
                "ks_empirical_weight_options": {"C_code": True, "split_by_max_weight": 10},
                "enc_empirical_weight_options": {"C_code": True, "split_by_max_weight": 10},
            }

        if PRINTING_MODE != PrintingMode.Silent:
            smart_print = _get_smart_print(self.__class__.filename)
            smart_print(f"# {self.test_cipher_ch_search.__name__}")

        for sp in ListSearchParameters:
            f = sp.function
            for nr, prop_type in zip(
                    [sp.cipher_xor_nr, sp.cipher_rx_nr, True, True],
                    [XorDiff, RXDiff, BitValue, WordValue]
            ):
                if nr is None or not issubclass(f, blockcipher.Cipher):
                    continue

                for at in [ChModelAssertType.ProbabilityOne, ChModelAssertType.ValidityAndWeight]:
                    if prop_type in [BitValue, WordValue] and at == ChModelAssertType.ValidityAndWeight:
                        continue
                    if prop_type == BitValue and sp.ignore_bitvalue is True:
                        continue
                    if prop_type == WordValue and sp.ignore_wordvalue is True:
                        continue

                    msg = f"\n## {f.__name__}, nr: {nr}, prop_type: {prop_type.__name__}, at: {at}"
                    if PRINTING_MODE != PrintingMode.Silent:
                        smart_print(msg)

                    round_ch_iterator = round_based_cipher_ch_search(
                        f, nr if (nr is not True and nr is not False) else getattr(f, "_min_num_rounds", 1),
                        nr + 1 if (nr is not True and nr is not False) else getattr(f, "_min_num_rounds", 1) + 1,
                        prop_type, at, at, "btor",
                        extra_cipherchfinder_args=extra_cipherchfinder_args,
                        extra_findnextchweight_args=extra_findnextchweight_args
                    )

                    if nr is not False:
                        try:
                            num_rounds_found, ch_found = next(round_ch_iterator)
                        except StopIteration:
                            raise StopIteration(msg)
                        target_nr = nr if nr is not True else getattr(f, "_min_num_rounds", 1)
                        self.assertEqual(num_rounds_found, target_nr, msg=msg)

                        if PRINTING_MODE != PrintingMode.Silent:
                            if PRINTING_MODE == PrintingMode.WeightsAndSrepr:
                                smart_print(ch_found.srepr())
                            else:
                                smart_print(ch_found)
                            smart_print("\n".join(ch_found.get_formatted_logged_msgs()))

                        round_ch_iterator.send(INCREMENT_NUM_ROUNDS)

                        msg += "\n||| probability-one characteristic found"

                    if nr is not True:
                        try:
                            num_rounds_found, ch_found = next(round_ch_iterator)
                        except StopIteration:
                            num_rounds_found, ch_found = None, None

                        if at == ChModelAssertType.ProbabilityOne:
                            self.assertIsNone(num_rounds_found, msg=msg)
                            self.assertIsNone(ch_found, msg=msg)
                        else:
                            if nr is False:
                                # INCREMENT_NUM_ROUNDS not sent
                                target_nr = getattr(f, "_min_num_rounds", 1)
                            else:
                                target_nr = nr + 1
                            self.assertEqual(num_rounds_found, target_nr, msg=msg)
                            ch_weight = ch_found.ks_characteristic.ch_weight
                            ch_weight += ch_found.enc_characteristic.ch_weight
                            self.assertGreater(ch_weight, 0, msg=msg)

                            if PRINTING_MODE != PrintingMode.Silent:
                                if PRINTING_MODE == PrintingMode.WeightsAndSrepr:
                                    smart_print(ch_found.srepr())
                                else:
                                    smart_print(ch_found)
                                smart_print("\n".join(ch_found.get_formatted_logged_msgs()))

    @unittest.skipIf(SKIP_INVALID_LONG_TESTS, "test_invalid_prop_search")
    def test_invalid_prop_search(self):
        extra_invalidpropfinder_args = {
            "printing_mode": PRINTING_MODE,
            "filename": self.__class__.filename,
        }

        if PRINTING_MODE != PrintingMode.Silent:
            smart_print = _get_smart_print(self.__class__.filename)
            smart_print(f"# {self.test_invalid_prop_search.__name__}")

        for sp in ListSearchParameters:
            f = sp.function
            min_nr = getattr(f, "_min_num_rounds", 1)

            if not issubclass(f, blockcipher.Cipher):
                continue  # KS are mostly non-permutations

            for prop_type in [XorDiff, RXDiff, LinearMask]:
                if issubclass(f, blockcipher.Cipher) and prop_type == RXDiff:
                    continue  # RXDiff not supported by EncryptionChModel
                if prop_type == LinearMask and (sp.linear_nr is None or sp.ignore_linear_invalid is True):
                    continue  # most probably due to BvAddCt

                msg = f"\n## {f.__name__}, min_nr: {min_nr}, prop_type: {prop_type.__name__}"
                if PRINTING_MODE != PrintingMode.Silent:
                    smart_print(msg)

                iterator = round_based_invalidprop_search(
                    f, 2*min_nr+1, 2*min_nr+1, prop_type, "btor",
                    max_num_skipped_rounds=0,
                    min_num_E0_rounds=min_nr, min_num_E2_rounds=min_nr,
                    exclude_zero_input_prop_E0=True,
                    exclude_zero_input_prop_E2=True,
                    extra_invalidpropfinder_args=extra_invalidpropfinder_args
                )

                try:
                    tuple_rounds, tuple_chs = next(iterator)
                except StopIteration:
                    continue
                except ValueError as e:
                    if "pr_one_assertions() == False" in str(e):  # most probably due to RX
                        continue
                    if str(e) == "linear characteristic models of functions with unused input vars is not supported":
                        continue
                    raise e

                if PRINTING_MODE != PrintingMode.Silent:
                    if PRINTING_MODE == PrintingMode.WeightsAndSrepr:
                        smart_print(tuple_rounds, ', '.join([ch.srepr() for ch in tuple_chs]))
                    else:
                        smart_print(tuple_rounds, tuple_chs)

    @unittest.skipIf(SKIP_INVALID_LONG_TESTS, "test_invalid_cipher_ch_search")
    def test_invalid_cipher_ch_search(self):
        extra_invalidcipherchfinder_args = {
            "printing_mode": PRINTING_MODE,
            "filename": self.__class__.filename,
        }

        if PRINTING_MODE != PrintingMode.Silent:
            smart_print = _get_smart_print(self.__class__.filename)
            smart_print(f"# {self.test_invalid_prop_search.__name__}")

        for sp in ListSearchParameters:
            f = sp.function
            min_nr = getattr(f, "_min_num_rounds", 1)

            if not issubclass(f, blockcipher.Cipher):
                continue

            for prop_type in [XorDiff, RXDiff]:
                if prop_type == XorDiff and sp.ignore_cipher_xor_invalid is True:
                    continue
                if prop_type == RXDiff and sp.ignore_cipher_rx_invalid is True:
                    continue

                msg = f"\n## {f.__name__}, min_nr: {min_nr}, prop_type: {prop_type.__name__}"
                if PRINTING_MODE != PrintingMode.Silent:
                    smart_print(msg)

                iterator = round_based_invalidcipherprop_search(
                    f, 2*min_nr+1, 2*min_nr+1, prop_type, "btor",
                    max_num_skipped_rounds=0,
                    min_num_E0_rounds=min_nr, min_num_E2_rounds=min_nr,
                    exclude_zero_input_prop_E0=True,
                    exclude_zero_input_prop_E2=True,
                    extra_invalidcipherpropfinder_args=extra_invalidcipherchfinder_args
                )

                try:
                    tuple_rounds, tuple_chs = next(iterator)
                except StopIteration:
                    continue
                except ValueError as e:
                    if "pr_one_assertions() == False" in str(e):  # most probably due to RX
                        continue
                    raise e

                if PRINTING_MODE != PrintingMode.Silent:
                    if PRINTING_MODE == PrintingMode.WeightsAndSrepr:
                        smart_print(tuple_rounds, ', '.join([ch.srepr() for ch in tuple_chs]))
                    else:
                        smart_print(tuple_rounds, tuple_chs)
