"""LEA 128-bit cipher."""
import itertools

from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft as ROL
from cascada.bitvector.operation import RotateRight as ROR

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


class LEAKeySchedule(RoundBasedFunction):
    """Key schedule function."""

    num_rounds = 24
    input_widths = [32, 32, 32, 32]
    output_widths = [32 for i in range(24 * 6)]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [32 for _ in range(new_num_rounds * 6)]

    @classmethod
    def eval(cls, k0, k1, k2, k3):
        round_keys = [None for _ in range(cls.num_rounds)]
        T0, T1, T2, T3 = k0, k1, k2, k3

        delta = [
            Constant(0xc3efe9db, 32), Constant(0x44626b02, 32),
            Constant(0x79e27c8a, 32), Constant(0x78df30ec, 32),
        ]

        for i in range(cls.num_rounds):
            T0 = ROL(T0 + ROL(delta[i % 4], i), 1)
            T1 = ROL(T1 + ROL(delta[i % 4], (i + 1)), 3)
            T2 = ROL(T2 + ROL(delta[i % 4], (i + 2)), 6)
            T3 = ROL(T3 + ROL(delta[i % 4], (i + 3)), 11)
            round_keys[i] = T0, T1, T2, T1, T3, T1
            cls.add_round_outputs(*round_keys[i])

        return tuple(itertools.chain(*round_keys))  # flatten


class LEAEncryption(Encryption, RoundBasedFunction):
    """Encryption function."""

    num_rounds = 24
    input_widths = [32, 32, 32, 32]
    output_widths = [32, 32, 32, 32]
    round_keys = None

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds

    @classmethod
    def eval(cls, p0, p1, p2, p3):
        x0, x1, x2, x3 = p0, p1, p2, p3

        rk = [[None for _ in range(6)] for _ in range(cls.num_rounds)]
        for i, k in enumerate(cls.round_keys):
            rk[i // 6][i % 6] = k

        for i in range(cls.num_rounds):
            k0, k1, k2, k3, k4, k5 = rk[i]
            tmp = x0
            x0 = ROL((x0 ^ k0) + (x1 ^ k1), 9)
            x1 = ROR((x1 ^ k2) + (x2 ^ k3), 5)
            x2 = ROR((x2 ^ k4) + (x3 ^ k5), 3)
            x3 = tmp
            cls.add_round_outputs(x0, x1, x2, x3)

        return x0, x1, x2, x3


class LEACipher(Cipher):
    """LEA-128 cipher."""
    key_schedule = LEAKeySchedule
    encryption = LEAEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.key_schedule.set_num_rounds(new_num_rounds)
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(24)
        key = [
            Constant(0x3c2d1e0f, 32),
            Constant(0x78695a4b, 32),
            Constant(0xb4a59687, 32),
            Constant(0xf0e1d2c3, 32)
        ]
        pt = [
            Constant(0x13121110, 32),
            Constant(0x17161514, 32),
            Constant(0x1b1a1918, 32),
            Constant(0x1f1e1d1c, 32)
        ]
        ct = [
            Constant(0x354ec89f, 32),
            Constant(0x18c6c628, 32),
            Constant(0xa7c73255, 32),
            Constant(0xfd8b6404, 32)
        ]
        assert cls(pt, key) == tuple(ct)
        cls.set_num_rounds(old_num_rounds)
