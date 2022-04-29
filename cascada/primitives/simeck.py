"""Simeck family of block ciphers.

This implementation is based on `simon` and `simon_rf`.
"""
import enum

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher

from cascada.primitives import simon_rf


class SimeckRF(simon_rf.SimonRF):
    """The non-linear part of the round function of Simon.

    This corresponds to ``f(x) = ((x <<< a) & (x <<< b)) ^ (x <<< c)``,
    where ``(a, b, c) = (5, 0, 4)``.
    """
    a = 5
    b = 0
    c = 1


class XorModelSimeckRF(simon_rf.XorModelSimonRF):
    """Represent the `XorDiff` `differential.opmodel.OpModel` of `SimeckRF`."""
    op = SimeckRF


class RXModelSimeckRF(simon_rf.RXModelSimonRF):
    """Represent the `RXDiff` `differential.opmodel.OpModel` of `SimeckRF`."""
    op = SimeckRF


SimeckRF.xor_model = XorModelSimeckRF
SimeckRF.rx_model = RXModelSimeckRF


#


class SimeckInstance(enum.Enum):
    """Represent the available instances of Simeck.

    Attributes:
        simeck_32_64: Simeck32/64
        simeck_48_96: Simeck48/96
        simeck_64_128: Simeck64/128

    """
    simeck_32_64 = enum.auto()
    simeck_48_96 = enum.auto()
    simeck_64_128 = enum.auto()


def get_Simeck_instance(simeck_instance):
    """Return an instance of the Simeck family as a `Cipher`."""
    if simeck_instance == SimeckInstance.simeck_32_64:
        default_rounds = 32
        n = 16
        m = 4
        z = list(reversed(bin(0x9A42BB1F)[2:]))
    elif simeck_instance == SimeckInstance.simeck_48_96:
        default_rounds = 36
        n = 24
        m = 4
        z = list(reversed(bin(0x9A42BB1F)[2:]))
    elif simeck_instance == SimeckInstance.simeck_64_128:
        default_rounds = 44
        n = 32
        m = 4
        z = list(reversed(bin(0x938BCA3083F)[2:]))
    else:
        raise ValueError("invalid instance of Simeck")

    def rf(x, y, k):
        """The round function of Simeck32/64."""
        return y ^ SimeckRF(x) ^ k, x

    class SimeckKeySchedule(RoundBasedFunction):
        """Key schedule function."""

        num_rounds = default_rounds
        input_widths = [n for _ in range(m)]
        output_widths = [n for _ in range(default_rounds)]

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds
            cls.output_widths = [n for _ in range(new_num_rounds)]

        @classmethod
        def eval(cls, *master_key):
            if cls.num_rounds <= m:
                return list(reversed(master_key))[:cls.num_rounds]

            k = [None for _ in range(cls.num_rounds)]
            k[0] = master_key[-1]
            t = list(reversed(master_key[:-1]))
            t.append(None)

            for i in range(cls.num_rounds - 1):
                C = (2 ** n - 4) ^ int(z[i % len(z)])
                result = rf(t[i % 4], k[i], C)
                t[(i + 3) % 4], k[i + 1] = result

            return k

    class SimeckEncryption(Encryption, RoundBasedFunction):
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

    class SimeckCipher(Cipher):
        key_schedule = SimeckKeySchedule
        encryption = SimeckEncryption
        _simeck_instance = simeck_instance

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.key_schedule.set_num_rounds(new_num_rounds)
            cls.encryption.set_num_rounds(new_num_rounds)

        @classmethod
        def test(cls):
            old_num_rounds = cls.num_rounds
            cls.set_num_rounds(default_rounds)

            if cls._simeck_instance == SimeckInstance.simeck_32_64:
                plaintext = (0x6565, 0x6877)
                key = (0x1918, 0x1110, 0x0908, 0x0100)
                assert cls(plaintext, key) == (0x770d, 0x2c76)
            elif cls._simeck_instance == SimeckInstance.simeck_48_96:
                plaintext = (0x726963, 0x20646e)
                key = (0x1a1918, 0x121110, 0x0a0908, 0x020100)
                assert cls(plaintext, key) == (0xf3cf25, 0xe33b36)
            elif cls._simeck_instance == SimeckInstance.simeck_64_128:
                plaintext = (0x656b696c, 0x20646e75)
                key = (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
                assert cls(plaintext, key) == (0x45ce6902, 0x5f7ab7ed)
            else:
                raise ValueError("invalid instance of Simeck")

            cls.set_num_rounds(old_num_rounds)

    return SimeckCipher
