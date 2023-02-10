"""RECTANGLE cipher."""
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft, Extract, Concat

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


def sub_column(x):
    t0 = x[2]
    x[2] ^= x[1]
    x[1] = ~x[1]
    t1 = x[0]
    x[0] &= x[1]
    x[1] |= x[3]
    x[3] ^= t0
    x[0] ^= x[3]
    x[1] ^= t1
    x[3] &= x[1]
    x[3] ^= x[2]
    x[2] |= x[0]
    x[2] ^= x[1]
    x[1] ^= t0


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
        round_keys = x[0:4]

        for i in range(cls.num_rounds):
            t = x[0:4]
            sub_column(t)
            x[0] = Concat(Extract(x[0], 15, 4), Extract(t[0], 3, 0))
            x[1] = Concat(Extract(x[1], 15, 4), Extract(t[1], 3, 0))
            x[2] = Concat(Extract(x[2], 15, 4), Extract(t[2], 3, 0))
            x[3] = Concat(Extract(x[3], 15, 4), Extract(t[3], 3, 0))

            t = x[0]
            x[0] = RotateLeft(x[0], 8) ^ x[1]
            x[1] = x[2]
            x[2] = x[3]
            x[3] = RotateLeft(x[3], 12) ^ x[4]
            x[4] = t

            x[0] ^= Constant(RC[i], 16)

            round_keys.append(x[0])
            round_keys.append(x[1])
            round_keys.append(x[2])
            round_keys.append(x[3])

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
        round_keys = []

        for i in range(cls.num_rounds):
            round_keys.append(Extract(x[0], 15, 0))
            round_keys.append(Extract(x[1], 15, 0))
            round_keys.append(Extract(x[2], 15, 0))
            round_keys.append(Extract(x[3], 15, 0))

            t = list(x)
            sub_column(t)
            x[0] = Concat(Extract(x[0], 31, 8), Extract(t[0], 7, 0))
            x[1] = Concat(Extract(x[1], 31, 8), Extract(t[1], 7, 0))
            x[2] = Concat(Extract(x[2], 31, 8), Extract(t[2], 7, 0))
            x[3] = Concat(Extract(x[3], 31, 8), Extract(t[3], 7, 0))

            t = x[0]
            x[0] = RotateLeft(x[0], 8) ^ x[1]
            x[1] = x[2]
            x[2] = RotateLeft(x[2], 16) ^ x[3]
            x[3] = t

            x[0] ^= Constant(RC[i], 32)

        round_keys.append(Extract(x[0], 15, 0))
        round_keys.append(Extract(x[1], 15, 0))
        round_keys.append(Extract(x[2], 15, 0))
        round_keys.append(Extract(x[3], 15, 0))

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
