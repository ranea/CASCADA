"""Simon family of block ciphers."""
import enum

from cascada.bitvector.operation import RotateRight

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher

from cascada.primitives.simon_rf import SimonRF


class SimonInstance(enum.Enum):
    """Represent the available instances of Simon.

    Attributes:
        simon_32_64: Simon32/64
        simon_48_96: Simon48/96
        simon_64_128: Simon64/128

    """
    simon_32_64 = enum.auto()
    simon_48_96 = enum.auto()
    simon_64_128 = enum.auto()


def get_Simon_instance(simon_instance):
    """Return an instance of the Simon family as a `Cipher`."""
    if simon_instance == SimonInstance.simon_32_64:
        default_rounds = 32
        n = 16
        m = 4
        z = "11111010001001010110000111001101111101000100101011000011100110"
    elif simon_instance == SimonInstance.simon_48_96:
        default_rounds = 36
        n = 24
        m = 4
        z = "10001110111110010011000010110101000111011111001001100001011010"
    elif simon_instance == SimonInstance.simon_64_128:
        default_rounds = 44
        n = 32
        m = 4
        z = "11011011101011000110010111100000010010001010011100110100001111"
    else:
        raise ValueError("invalid instance of Simon")

    class SimonKeySchedule(RoundBasedFunction):
        """Key schedule function."""

        num_rounds = default_rounds
        input_widths = [n for _ in range(m)]
        output_widths = [n for _ in range(default_rounds)]

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds
            cls.input_widths = [n for _ in range(min(m, new_num_rounds))]
            cls.output_widths = [n for _ in range(new_num_rounds)]

        @classmethod
        def eval(cls, *master_key):
            if cls.num_rounds <= m:
                return list(reversed(master_key))[:cls.num_rounds]

            k = [None for _ in range(cls.num_rounds)]
            k[:m] = list(reversed(master_key))

            for i in range(m, cls.num_rounds):
                tmp = RotateRight(k[i - 1], 3)
                if m == 4:
                    tmp ^= k[i - 3]
                tmp ^= RotateRight(tmp, 1)
                k[i] = ~k[i - m] ^ tmp ^ int(z[(i - m) % 62]) ^ 3

            return k

    class SimonEncryption(Encryption, RoundBasedFunction):
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
                x, y = (y ^ SimonRF(x) ^ cls.round_keys[i], x)
                cls.add_round_outputs(x, y)
            return x, y

    class SimonCipher(Cipher):
        key_schedule = SimonKeySchedule
        encryption = SimonEncryption
        _simon_instance = simon_instance

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.key_schedule.set_num_rounds(new_num_rounds)
            cls.encryption.set_num_rounds(new_num_rounds)

        @classmethod
        def test(cls):
            old_num_rounds = cls.num_rounds
            cls.set_num_rounds(default_rounds)

            if cls._simon_instance == SimonInstance.simon_32_64:
                plaintext = (0x6565, 0x6877)
                key = (0x1918, 0x1110, 0x0908, 0x0100)
                assert cls(plaintext, key) == (0xc69b, 0xe9bb)
            elif cls._simon_instance == SimonInstance.simon_48_96:
                plaintext = (0x726963, 0x20646e)
                key = (0x1a1918, 0x121110, 0x0a0908, 0x020100)
                assert cls(plaintext, key) == (0x6e06a5, 0xacf156)
            elif cls._simon_instance == SimonInstance.simon_64_128:
                plaintext = (0x656b696c, 0x20646e75)
                key = (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
                assert cls(plaintext, key) == (0x44c8fc20, 0xb9dfa07a)
            else:
                raise ValueError("invalid instance of Simon")

            cls.set_num_rounds(old_num_rounds)

    return SimonCipher
