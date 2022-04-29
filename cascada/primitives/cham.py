"""CHAM family of block ciphers."""
import enum

from cascada.bitvector.operation import RotateLeft

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


class CHAMInstance(enum.Enum):
    """Represent the available instances of CHAM.

    Attributes:
        CHAM_64_128: CHAM64/128
        CHAM_128_128: CHAM128/128
        CHAM_128_256: CHAM128/256

    """
    CHAM_64_128 = enum.auto()
    CHAM_128_128 = enum.auto()
    CHAM_128_256 = enum.auto()


def get_CHAM_instance(CHAM_instance):
    """Return an instance of the CHAM family as a `Cipher`."""
    if CHAM_instance == CHAMInstance.CHAM_64_128:
        default_rounds = 80
        n = 64
        kappa = 128
        w = 16
    elif CHAM_instance == CHAMInstance.CHAM_128_128:
        default_rounds = 80
        n = 128
        kappa = 128
        w = 32
    elif CHAM_instance == CHAMInstance.CHAM_128_256:
        default_rounds = 96
        n = 128
        kappa = 256
        w = 32
    else:
        raise ValueError("invalid instance of CHAM")

    class CHAMKeySchedule(RoundBasedFunction):
        """Key schedule function."""

        num_rounds = default_rounds
        input_widths = [w for _ in range(kappa // w)]
        output_widths = [w for _ in range(2 * kappa // w)]

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            # assert new_num_rounds >= kappa // w
            cls.num_rounds = new_num_rounds
            cls.input_widths = [w for _ in range(kappa // w)]
            cls.output_widths = [w for _ in range(min(new_num_rounds, 2 * kappa // w))]

        @classmethod
        def eval(cls, *master_key):
            k = master_key

            num_rk = min(cls.num_rounds, 2 * kappa // w)
            rk = [None for _ in range(num_rk)]

            for i in range(kappa // w):
                if i < num_rk:
                    rk[i] = k[i] ^ RotateLeft(k[i], 1) ^ RotateLeft(k[i], 8)
                if ((i + (kappa // w)) ^ 1) < num_rk:
                    rk[(i + (kappa // w)) ^ 1] = k[i] ^ RotateLeft(k[i], 1) ^ RotateLeft(k[i], 11)

            return rk

    class CHAMEncryption(Encryption, RoundBasedFunction):
        """Encryption function."""

        num_rounds = default_rounds
        input_widths = [w for _ in range(n // w)]
        output_widths = [w for _ in range(n // w)]
        round_keys = None

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds

        @classmethod
        def eval(cls, p0, p1, p2, p3):
            x = [p0, p1, p2, p3]

            for i in range(cls.num_rounds):
                y = [None for _ in range(4)]

                rk = cls.round_keys[i % (2 * kappa // w)]

                if i % 2 == 0:
                    y[3] = RotateLeft((x[0] ^ i) + (RotateLeft(x[1], 1) ^ rk), 8)
                else:
                    y[3] = RotateLeft((x[0] ^ i) + (RotateLeft(x[1], 8) ^ rk), 1)

                for j in range(2 + 1):
                    y[j] = x[j + 1]

                cls.add_round_outputs(*y)
                x = y

            return x

    class CHAMCipher(Cipher):
        key_schedule = CHAMKeySchedule
        encryption = CHAMEncryption
        _CHAM_instance = CHAM_instance

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.key_schedule.set_num_rounds(new_num_rounds)
            cls.encryption.set_num_rounds(new_num_rounds)

        @classmethod
        def test(cls):
            old_num_rounds = cls.num_rounds
            cls.set_num_rounds(default_rounds)

            if cls._CHAM_instance == CHAMInstance.CHAM_64_128:
                plaintext = (0x1100, 0x3322, 0x5544, 0x7766)
                key = (0x0100, 0x0302, 0x0504, 0x0706,
                       0x0908, 0x0b0a, 0x0d0c, 0x0f0e)
                assert cls(plaintext, key) == (0x453c, 0x63bc, 0xdcfa, 0xbf4e)
            elif cls._CHAM_instance == CHAMInstance.CHAM_128_128:
                plaintext = (0x33221100, 0x77665544, 0xbbaa9988, 0xffeeddcc)
                key = (0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c)
                assert cls(plaintext, key) == (0xc3746034, 0xb55700c5, 0x8d64ec32, 0x489332f7)
            elif cls._CHAM_instance == CHAMInstance.CHAM_128_256:
                plaintext = (0x33221100, 0x77665544, 0xbbaa9988, 0xffeeddcc)
                key = (0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
                       0xf3f2f1f0, 0xf7f6f5f4, 0xfbfaf9f8, 0xfffefdfc)
                assert cls(plaintext, key) == (0xa899c8a0, 0xc929d55c, 0xab670d38, 0x0c4f7ac8)
            else:
                raise ValueError("invalid instance of CHAM")

            cls.set_num_rounds(old_num_rounds)

    return CHAMCipher
