"""Tests the evaluation of cryptographic primitives."""
import unittest

from cascada.primitives.blockcipher import Cipher

from cascada.primitives import aes
from cascada.primitives import cham
from cascada.primitives import chaskey
from cascada.primitives import feal
from cascada.primitives import hight
from cascada.primitives import lea
from cascada.primitives import multi2
from cascada.primitives import noekeon
from cascada.primitives import picipher
from cascada.primitives import shacal1
from cascada.primitives import shacal2
from cascada.primitives import speck
from cascada.primitives import simeck
from cascada.primitives import simon
from cascada.primitives import skinny64
from cascada.primitives import skinny128
from cascada.primitives import tea
from cascada.primitives import xtea


CHAM64 = cham.get_CHAM_instance(cham.CHAMInstance.CHAM_64_128)
Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
Simeck32 = simeck.get_Simeck_instance(simeck.SimeckInstance.simeck_32_64)
Simon32 = simon.get_Simon_instance(simon.SimonInstance.simon_32_64)


LIST_PRIMITIVES = [
    aes.AESCipher,
    CHAM64,
    chaskey.ChaskeyPi,
    feal.FEALCipher,
    hight.HightCipher,
    lea.LEACipher,
    multi2.MULTI2Cipher,
    noekeon.NOEKEONDirectCipher,
    noekeon.NOEKEONIndirectCipher,
    picipher.PiPermutation,
    shacal1.SHACAL1Cipher,
    shacal2.SHACAL2Cipher,
    Speck32,
    Simeck32,
    Simon32,
    skinny64.SKINNYCipher,
    skinny128.SKINNYCipher,
    tea.TEACipher,
    xtea.XTEACipher,
]


class TestPrimitivesEval(unittest.TestCase):
    """Basic tests for primitives implemented."""

    def setUp(self):
        self.primitive2default_num_rounds = {}
        for f in LIST_PRIMITIVES:
            if getattr(f, "num_rounds", None) is not None:
                self.primitive2default_num_rounds[f] = f.num_rounds

    def tearDown(self):
        for f in LIST_PRIMITIVES:
            if getattr(f, "num_rounds", None) is not None:
                f.set_num_rounds(self.primitive2default_num_rounds[f])
            if issubclass(f, Cipher):
                f.encryption.round_keys = None

    def test_testvectors(self):
        for f in LIST_PRIMITIVES:
            if hasattr(f, "test"):
                f.test()

    def test_split(self):
        for f in LIST_PRIMITIVES:
            if getattr(f, "num_rounds", None) is None:
                continue

            min_num_rounds = getattr(f, "_min_num_rounds", 1)
            for nr in range(min_num_rounds, self.primitive2default_num_rounds[f] + 1):
                f.set_num_rounds(nr)

                if issubclass(f, Cipher):
                    f.set_round_keys(symbolic_prefix="k")
                    assert len(f.encryption.round_keys) == len(f.key_schedule.output_widths)
                    my_f = f.encryption
                else:
                    my_f = f

                ssa = my_f.to_ssa([f"p{i}" for i in range(len(my_f.input_widths))], "x")
                rs = ssa.get_round_separators()
                self.assertEqual(len(rs), nr - 1, msg=f"{my_f.get_name()}, rs:{rs}")

    def test_zero_input(self):
        for f in LIST_PRIMITIVES:
            if getattr(f, "num_rounds", None) is None:
                if issubclass(f, Cipher):
                    key = [0 for _ in f.key_schedule.input_widths]
                    pt = [0 for _ in f.encryption.input_widths]
                    f(pt, key)
                else:
                    f(*[0 for _ in f.input_widths])

            else:
                min_num_rounds = getattr(f, "_min_num_rounds", 1)
                for nr in range(min_num_rounds, self.primitive2default_num_rounds[f] + 1):
                    f.set_num_rounds(nr)
                    if issubclass(f, Cipher):
                        key = [0 for _ in f.key_schedule.input_widths]
                        pt = [0 for _ in f.encryption.input_widths]
                        f(pt, key)
                    else:
                        f(*[0 for _ in f.input_widths])
