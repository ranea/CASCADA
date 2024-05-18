"""GOST block cipher.

Source: https://en.wikipedia.org/wiki/GOST_(block_cipher)

S-boxes: 
* SBOXES_CB (Central Bank of Russian Federation, folklore):
  - https://en.wikipedia.org/wiki/GOST_(block_cipher)
* SBOXES_STB (STB 34.101.50-2019):
  - https://github.com/bcrypto/stb/blob/master/34.101.50/oids.md
  - https://github.com/agievich/GF2/blob/master/test/test.cpp#~:text=testGOST
* SBOXES_CRYPTOPRO_A (id-Gost28147-89-CryptoPro-A-ParamSet): 
  - https://datatracker.ietf.org/doc/html/rfc4357

**Note**. The cipher is defined in the Soviet Union standard GOST 28147-89.
There S-boxes are denoted as `K1, ..., K8` and the key is denoted as `X`. 
We use more conventional notation.
"""

from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft, Extract, Concat
from cascada.bitvector.secondaryop import LutOperation
from cascada.differential.difference import XorDiff
from cascada.differential.opmodel import get_wdt as get_differential_wdt
from cascada.differential.opmodel import get_wdt_model as get_differential_wdt_model
from cascada.linear.opmodel import get_wdt as get_linear_wdt
from cascada.linear.opmodel import get_wdt_model as get_linear_wdt_model
from cascada.primitives.blockcipher import Encryption, Cipher
from cascada.bitvector.ssa import RoundBasedFunction

SBOXES_CB = [
    4, 10, 9, 2, 13, 8, 0, 14, 6, 11, 1, 12, 7, 15, 5, 3,
    14, 11, 4, 12, 6, 13, 15, 10, 2, 3, 8, 1, 0, 7, 5, 9,
    5, 8, 1, 13, 10, 3, 4, 2, 14, 15, 12, 7, 6, 0, 9, 11,
    7, 13, 10, 1, 0, 8, 9, 15, 14, 4, 6, 12, 11, 2, 5, 3,
    6, 12, 7, 1, 5, 15, 13, 8, 4, 10, 9, 14, 0, 3, 11, 2,
    4, 11, 10, 0, 7, 2, 1, 13, 3, 6, 8, 5, 9, 12, 15, 14,
    13, 11, 4, 1, 3, 15, 5, 9, 0, 10, 14, 7, 6, 8, 2, 12,
    1, 15, 13, 0, 5, 7, 10, 4, 9, 2, 3, 14, 6, 11, 8, 12,
]

SBOXES_STB = [
    2, 6, 3, 14, 12, 15, 7, 5, 11, 13, 8, 9, 10, 0, 4, 1,
    8, 12, 9, 6, 10, 7, 13, 1, 3, 11, 14, 15, 2, 4, 0, 5,
    1, 5, 4, 13, 3, 8, 0, 14, 12, 6, 7, 2, 9, 15, 11, 10,
    4, 0, 5, 10, 2, 11, 1, 9, 15, 3, 6, 7, 14, 12, 8, 13,
    7, 9, 6, 11, 15, 10, 8, 12, 4, 14, 1, 0, 5, 3, 13, 2,
    14, 8, 15, 2, 6, 3, 9, 13, 5, 7, 0, 1, 4, 10, 12, 11,
    9, 13, 8, 5, 11, 4, 12, 2, 0, 10, 15, 14, 1, 7, 3, 6,
    11, 15, 10, 8, 1, 14, 3, 6, 9, 0, 4, 5, 13, 2, 7, 12,
]

SBOXES_CRYPTOPRO_A = [
    9, 6, 3, 2, 8, 11, 1, 7, 10, 4, 14, 15, 12, 0, 13, 5,
    3, 7, 14, 9, 8, 10, 15, 0, 5, 2, 6, 12, 11, 4, 13, 1,
    14, 4, 6, 2, 11, 3, 13, 8, 12, 15, 5, 10, 0, 7, 1, 9,
    14, 7, 10, 12, 13, 1, 3, 9, 0, 2, 11, 4, 15, 8, 5, 6,
    11, 5, 1, 9, 8, 13, 15, 0, 14, 4, 2, 3, 12, 7, 10, 6,
    3, 10, 13, 12, 1, 2, 0, 11, 7, 5, 9, 4, 8, 15, 14, 6,
    1, 13, 2, 9, 7, 10, 6, 0, 8, 12, 4, 5, 15, 3, 11, 14,
    11, 10, 15, 5, 0, 12, 14, 8, 6, 2, 3, 9, 1, 7, 13, 4,
]

def get_GOST_instance(_sboxes):

    class GostKeySchedule(RoundBasedFunction):
        num_rounds = 32
        input_widths = [32 for _ in range(8)]
        output_widths = [32 for _ in range(32)]
    
        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds
            cls.input_widths = [32 for _ in range(8)]
            cls.output_widths = [32 for _ in range(cls.num_rounds)]
    
        @classmethod
        def eval(cls, *master_key):
            round_keys = [None for _ in range(cls.num_rounds)]
            for i in range(24):
                round_keys[i] = master_key[i % 8]
            for i in range(24, cls.num_rounds):
                round_keys[i] = master_key[cls.num_rounds - i - 1]
            return round_keys

    class GostSBox(LutOperation):
        @classmethod
        def init(cls, lut):
            cls.lut = [Constant(x, 4) for x in lut]
            # differential wdt
            wdt = get_differential_wdt(cls, XorDiff, 4, 4)
            min_wdt = min(min(wdt[1:], key=lambda row: min(row[1:])))
            assert min_wdt > 0
            prec = 0
            while min_wdt < 1:
               min_wdt = min_wdt * 2
               prec = prec + 1
            cls.xor_model = get_differential_wdt_model(cls, XorDiff, wdt, 
                precision=prec)
            # linear wdt
            wdt = get_linear_wdt(cls, 4, 4)
            min_wdt = min(min(wdt[1:], key=lambda row: min(row[1:])))
            assert min_wdt > 0
            prec = 0
            while min_wdt < 1:
               min_wdt = min_wdt * 2
               prec = prec + 1
            cls.linear_model = get_linear_wdt_model(cls, wdt, precision=prec)
    
    class GostEncryption(Encryption, RoundBasedFunction):
        num_rounds = 32
        input_widths = [32, 32]
        output_widths = [32, 32]
    
        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds
    
        class S1(GostSBox): pass
        class S2(GostSBox): pass
        class S3(GostSBox): pass
        class S4(GostSBox): pass
        class S5(GostSBox): pass
        class S6(GostSBox): pass
        class S7(GostSBox): pass
        class S8(GostSBox): pass
    
        @classmethod
        def roundf(cls, x, k):
            x = x + k
            t1 = cls.S1(Extract(x, 3, 0))
            t2 = cls.S2(Extract(x, 7, 4))
            t3 = cls.S3(Extract(x, 11, 8))
            t4 = cls.S4(Extract(x, 15, 12))
            t5 = cls.S5(Extract(x, 19, 16))
            t6 = cls.S6(Extract(x, 23, 20))
            t7 = cls.S7(Extract(x, 27, 24))
            t8 = cls.S8(Extract(x, 31, 28))
            x = Concat(Concat(Concat(Concat(Concat(Concat(Concat(t8, 
                t7), t6), t5), t4), t3), t2), t1)
            x = RotateLeft(x, 11)
            return x
    
        @classmethod
        def eval(cls, n1, n2):
            K = cls.round_keys
            for i in range(cls.num_rounds):
                n1, n2 = n2 ^ cls.roundf(n1, K[i]), n1
                cls.add_round_outputs(n1, n2)
            if cls.num_rounds == 32:
                n1, n2 = n2, n1
            return n1, n2
    
    GostEncryption.S1.init(_sboxes[0:16])
    GostEncryption.S2.init(_sboxes[16:32])
    GostEncryption.S3.init(_sboxes[32:48])
    GostEncryption.S4.init(_sboxes[48:64])
    GostEncryption.S5.init(_sboxes[64:80])
    GostEncryption.S6.init(_sboxes[80:96])
    GostEncryption.S7.init(_sboxes[96:112])
    GostEncryption.S8.init(_sboxes[112:128])

    class GostCipher(Cipher):
        key_schedule = GostKeySchedule
        encryption = GostEncryption
        sboxes = _sboxes
    
        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.key_schedule.set_num_rounds(new_num_rounds)
            cls.encryption.set_num_rounds(new_num_rounds)

        @classmethod
        def test(cls):
            pt = (0x2AD937E9, 0xF0C18330)
            key = (0x2E27A628, 0x6CD530F7, 0xDB222B74, 0x430B2807, 
                0xADB06D11, 0x11EB4E24, 0x4A1035A0, 0x451831EB)
            if cls.sboxes == SBOXES_CB:
                assert cls(pt, key) == (0x0E526166, 0xDCB8AA21)
            elif cls.sboxes == SBOXES_STB:
                assert cls(pt, key) == (0xD58DF27C, 0x48C58DD4)
            elif cls.sboxes == SBOXES_CRYPTOPRO_A:
                assert cls(pt, key) == (0x010AE060, 0x2FECC9CC)

    return GostCipher
