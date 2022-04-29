"""Speck family of block ciphers."""
import enum

from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft, RotateRight

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


class SpeckInstance(enum.Enum):
    """Represent the available instances of Speck.

    Attributes:
        speck_32_64: Speck32/64
        speck_48_96: Speck48/96
        speck_64_96: Speck64/96
        speck_64_128: Speck64/128

    """
    speck_32_64 = enum.auto()
    speck_48_96 = enum.auto()
    speck_64_96 = enum.auto()
    speck_64_128 = enum.auto()


def get_Speck_instance(speck_instance):
    """Return an instance of the Speck family as a `Cipher`."""
    if speck_instance == SpeckInstance.speck_32_64:
        default_rounds = 22
        n = 16
        m = 4
        alpha = 7
        beta = 2
    elif speck_instance == SpeckInstance.speck_48_96:
        default_rounds = 23
        n = 24
        m = 4
        alpha = 8
        beta = 3
    elif speck_instance == SpeckInstance.speck_64_96:
        default_rounds = 26
        n = 32
        m = 3
        alpha = 8
        beta = 3
    elif speck_instance == SpeckInstance.speck_64_128:
        default_rounds = 27
        n = 32
        m = 4
        alpha = 8
        beta = 3
    else:
        raise ValueError("invalid instance of speck")

    def rf(x, y, k):
        """Round function."""
        x = (RotateRight(x, alpha) + y) ^ k
        y = RotateLeft(y, beta) ^ x
        return x, y

    class SpeckKeySchedule(RoundBasedFunction):
        """Key schedule function."""

        num_rounds = default_rounds - 1
        input_widths = [n for _ in range(m)]
        output_widths = [n for _ in range(default_rounds)]

        @classmethod
        def set_num_rounds(cls, new_num_rounds):  # new_num_rounds = 0 if enc_num_rounds = 1
            cls.num_rounds = new_num_rounds
            cls.input_widths = [n for _ in range(min(m, new_num_rounds + 1))]
            cls.output_widths = [n for _ in range(new_num_rounds + 1)]

        @classmethod
        def eval(cls, *master_key):
            round_keys = [None for _ in range(cls.num_rounds + 1)]
            round_keys[0] = master_key[-1]
            l_values = list(reversed(master_key[:-1]))
            l_values.append(None)

            for i in range(cls.num_rounds):
                result = rf(l_values[i % len(l_values)], round_keys[i], Constant(i, n))
                cls.add_round_outputs(*result)
                l_values[(i + m - 1) % len(l_values)], round_keys[i + 1] = result

            return round_keys

    class SpeckEncryption(Encryption, RoundBasedFunction):
        """Encryption function."""

        num_rounds = default_rounds
        input_widths = [n, n]
        output_widths = [n, n]
        round_keys = None

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds

        @classmethod
        def eval(cls, x, y):
            for i in range(cls.num_rounds):
                x, y = rf(x, y, cls.round_keys[i])
                cls.add_round_outputs(x, y)
            return x, y

    class SpeckCipher(Cipher):
        key_schedule = SpeckKeySchedule
        encryption = SpeckEncryption
        _speck_instance = speck_instance

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.key_schedule.set_num_rounds(new_num_rounds - 1)
            cls.encryption.set_num_rounds(new_num_rounds)

        @classmethod
        def test(cls):
            old_num_rounds = cls.num_rounds
            cls.set_num_rounds(default_rounds)

            if cls._speck_instance == SpeckInstance.speck_32_64:
                plaintext = (0x6574, 0x694c)
                key = (0x1918, 0x1110, 0x0908, 0x0100)
                assert cls(plaintext, key) == (0xa868, 0x42f2)
            elif cls._speck_instance == SpeckInstance.speck_48_96:
                plaintext = (0x6d2073, 0x696874)
                key = (0x1a1918, 0x121110, 0x0a0908, 0x020100)
                assert cls(plaintext, key) == (0x735e10, 0xb6445d)
            elif cls._speck_instance == SpeckInstance.speck_64_96:
                plaintext = (0x74614620, 0x736e6165)
                key = (0x13121110, 0x0b0a0908, 0x03020100)
                assert cls(plaintext, key) == (0x9f7952ec, 0x4175946c)
            elif cls._speck_instance == SpeckInstance.speck_64_128:
                plaintext = (0x3b726574, 0x7475432d)
                key = (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
                assert cls(plaintext, key) == (0x8c6fa548, 0x454e028b)
            else:
                raise ValueError("invalid instance of speck")

            cls.set_num_rounds(old_num_rounds)

    return SpeckCipher
