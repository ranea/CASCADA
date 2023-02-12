"""RECTANGLE cipher."""
from decimal import Decimal
from math import inf

from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft, Concat
from cascada.bitvector.secondaryop import LutOperation
from cascada.abstractproperty.opmodel import log2_decimal
from cascada.differential.difference import XorDiff
from cascada.differential.opmodel import get_wdt_model as get_differential_wdt_model
from cascada.linear.opmodel import get_wdt_model as get_linear_wdt_model
from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher

class SboxLut(LutOperation):
    """The 4-bit S-box of RECTANGLE."""
    lut = [Constant(x, 4) for x in (6, 5, 12, 10, 1, 14, 7, 9, 11, 0, 3, 13, 8, 15, 4, 2)]


ddt = (
    (16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 2, 0, 0, 4, 2, 0, 0, 0, 2, 0, 0, 4, 2),
    (0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 0, 2, 4, 0, 2),
    (0, 0, 0, 2, 0, 0, 2, 0, 2, 4, 2, 2, 2, 0, 0, 0),
    (0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4),
    (0, 2, 0, 0, 4, 2, 0, 0, 4, 2, 0, 0, 0, 2, 0, 0),
    (0, 2, 4, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2),
    (0, 0, 4, 0, 2, 2, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2),
    (0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2),
    (0, 2, 0, 0, 0, 2, 4, 0, 0, 2, 0, 0, 0, 2, 4, 0),
    (0, 0, 0, 0, 0, 4, 2, 2, 2, 0, 2, 0, 2, 0, 0, 2),
    (0, 4, 0, 2, 0, 0, 2, 0, 2, 0, 2, 2, 2, 0, 0, 0),
    (0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 0, 0, 0, 4, 0),
    (0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 4, 0, 0, 2, 4, 0),
    (0, 0, 4, 2, 2, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0),
    (0, 2, 4, 2, 2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0)
)
lat = (
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0,  0.25, 0, -0.25, 0, 0,  0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5),
    (0, 0, 0, 0, 0, 0,  0.25,  0.25, 0, 0,  0.25, -0.25, 0, 0, 0, 0),
    (0, 0, 0, -0.25,  0.25, 0, 0, 0, -0.5,  0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5),
    (0, 0, 0, 0, 0, 0, -0.25,  0.25, 0, 0, 0, 0, 0, 0,  0.25,  0.25),
    (0, 0, -0.25, 0, 0, -0.25, 0, 0, -0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5),
    (0, 0, 0, 0, 0, 0, 0, 0,  0.25,  0.25, 0, 0, -0.25,  0.25, 0, 0),
    (0, 0, -0.25, 0, -0.25, 0, 0, 0, -0.5,  0.5,  0.5,  0.5, -0.5, -0.5,  0.5, -0.5),
    (0, 0, 0, -0.25, -0.5, -0.5,  0.5, -0.5, 0, -0.25, 0, 0, -0.5,  0.5,  0.5,  0.5),
    (0, 0, 0, 0, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.25, 0,  0.25, 0),
    (0, 0, 0, -0.25, -0.5, -0.5, -0.5,  0.5,  0.25, 0, 0, 0,  0.5, -0.5, -0.5, -0.5),
    (0, 0, 0, 0,  0.5, -0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5, 0, -0.25, 0,  0.25),
    (0,  0.25, 0, 0, -0.5,  0.5, -0.5, -0.5, 0, 0, 0, -0.25, -0.5, -0.5, -0.5,  0.5),
    (0,  0.25,  0.25, 0, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5, -0.5,  0.5, 0, 0, 0, 0),
    (0, -0.25, 0, 0, -0.5,  0.5,  0.5,  0.5, 0, 0, -0.25, 0, -0.5, -0.5, -0.5,  0.5),
    (0,  0.25, -0.25, 0,  0.5,  0.5,  0.5,  0.5,  0.5, -0.5, -0.5,  0.5, 0, 0, 0, 0)
)


SboxLut.xor_model = get_differential_wdt_model(
    SboxLut, XorDiff, tuple([tuple(inf if x == 0 else -log2_decimal(x / Decimal(2 ** 4)) for x in row) for row in ddt]))
SboxLut.linear_model = get_linear_wdt_model(
    SboxLut, tuple([tuple(inf if x == 0 else -log2_decimal(Decimal(abs(x))) for x in row) for row in lat]))


def sub_column(x):
    bits = x[0].width

    for i in range(bits):
        s = Concat(Concat(Concat(x[3][i], x[2][i]), x[1][i]), x[0][i])
        s = SboxLut(s)

        for j in range(4):
            if i == 0:
                x[j] = Concat(x[j][:1], s[j])
            elif i == bits - 1:
                x[j] = Concat(s[j], x[j][bits - 2:])
            else:
                x[j] = Concat(Concat(x[j][:i + 1], s[j]), x[j][i - 1:])


def shift_row(x):
    x[1] = RotateLeft(x[1], 1)
    x[2] = RotateLeft(x[2], 12)
    x[3] = RotateLeft(x[3], 13)


def add_round_key(x, k, i):
    x[0] ^= k[i]
    x[1] ^= k[i + 1]
    x[2] ^= k[i + 2]
    x[3] ^= k[i + 3]


RC = (0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16,
      0x0C, 0x19, 0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C,
      0x18, 0x11, 0x03, 0x06, 0x0D, 0x1B, 0x17, 0x0E,
      0x1D)


class RECTANGLEEncryption(Encryption, RoundBasedFunction):
    """RECTANGLE encryption function."""

    num_rounds = 25
    input_widths = [16, 16, 16, 16]
    output_widths = [16, 16, 16, 16]
    round_keys = None

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds

    @classmethod
    def eval(cls, *x):
        x = list(x)

        for i in range(cls.num_rounds):
            add_round_key(x, cls.round_keys, i * 4)
            sub_column(x)
            shift_row(x)
            cls.add_round_outputs(x)

        add_round_key(x, cls.round_keys, cls.num_rounds * 4)

        return x


class RECTANGLE80KeySchedule(RoundBasedFunction):
    """RECTANGLE-80 key schedule function."""

    num_rounds = 25
    input_widths = [16, 16, 16, 16, 16]
    output_widths = [16 for _ in range(4 * num_rounds + 4)]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        assert new_num_rounds <= len(RC)
        cls.num_rounds = new_num_rounds
        cls.output_widths = [16 for _ in range(4 * new_num_rounds + 4)]

    @classmethod
    def eval(cls, *master_key):
        x = list(master_key)
        round_keys = x[:4]

        for i in range(cls.num_rounds):
            t = [x[0][3:], x[1][3:], x[2][3:], x[3][3:]]
            sub_column(t)
            for j in range(4):
                x[j] = Concat(x[j][:4], t[j])

            t = x[0]
            x[0] = RotateLeft(x[0], 8) ^ x[1]
            x[1] = x[2]
            x[2] = x[3]
            x[3] = RotateLeft(x[3], 12) ^ x[4]
            x[4] = t

            x[0] ^= Constant(RC[i], 16)

            round_keys.extend(x[:4])

        return round_keys


class RECTANGLE128KeySchedule(RoundBasedFunction):
    """RECTANGLE-128 key schedule function."""

    num_rounds = 25
    input_widths = [32, 32, 32, 32]
    output_widths = [16 for _ in range(4 * num_rounds + 4)]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        assert new_num_rounds <= len(RC)
        cls.num_rounds = new_num_rounds
        cls.output_widths = [16 for _ in range(4 * new_num_rounds + 4)]

    @classmethod
    def eval(cls, *master_key):
        x = list(master_key)
        round_keys = [w[15:] for w in x]

        for i in range(cls.num_rounds):
            t = [w[7:] for w in x]
            sub_column(t)
            for j in range(4):
                x[j] = Concat(x[j][:8], t[j])

            t = x[0]
            x[0] = RotateLeft(x[0], 8) ^ x[1]
            x[1] = x[2]
            x[2] = RotateLeft(x[2], 16) ^ x[3]
            x[3] = t

            x[0] ^= Constant(RC[i], 32)

            round_keys.extend([w[15:] for w in x])

        return round_keys


class RECTANGLE80Cipher(Cipher):
    """RECTANGLE-80 cipher."""
    key_schedule = RECTANGLE80KeySchedule
    encryption = RECTANGLEEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.key_schedule.set_num_rounds(new_num_rounds)
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        """Test RECTANGLE-80 test vectors."""
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(25)

        plaintext = (0x0000, 0x0000, 0x0000, 0x0000)
        key = (0x0000, 0x0000, 0x0000, 0x0000, 0x0000)
        assert cls(plaintext, key) == (0x2D96, 0xE354, 0xE8B1, 0x0874)

        plaintext = (0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)
        key = (0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)
        assert cls(plaintext, key) == (0x9945, 0xAA34, 0xAE3D, 0x0112)

        cls.set_num_rounds(old_num_rounds)


class RECTANGLE128Cipher(Cipher):
    """RECTANGLE-128 cipher."""
    key_schedule = RECTANGLE128KeySchedule
    encryption = RECTANGLEEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.key_schedule.set_num_rounds(new_num_rounds)
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        """Test RECTANGLE-128 test vectors."""
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(25)

        plaintext = (0x0000, 0x0000, 0x0000, 0x0000)
        key = (0x00000000, 0x00000000, 0x00000000, 0x00000000)
        assert cls(plaintext, key) == (0xAEE6, 0x3613, 0x44A4, 0x99EE)

        plaintext = (0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)
        key = (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        assert cls(plaintext, key) == (0xE83E, 0xEFEE, 0x4A15, 0x7A46)

        cls.set_num_rounds(old_num_rounds)
