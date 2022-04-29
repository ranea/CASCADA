"""HIGHT cipher.

While HIGHT is defined for 32 rounds, this implementation considers 34 rounds,
where the first and last round are the initial and final key-whitening.
"""
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft as ROL

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


class HightKeySchedule(RoundBasedFunction):
    """Key schedule function."""

    num_rounds = 34
    input_widths = [8 for _ in range(16)]
    output_widths = [8 for _ in range(4 * 34)]

    constants = [Constant(d_i, 8) for d_i in [
        0x5a, 0x6d, 0x36, 0x1b, 0x0d, 0x06, 0x03, 0x41,
        0x60, 0x30, 0x18, 0x4c, 0x66, 0x33, 0x59, 0x2c,
        0x56, 0x2b, 0x15, 0x4a, 0x65, 0x72, 0x39, 0x1c,
        0x4e, 0x67, 0x73, 0x79, 0x3c, 0x5e, 0x6f, 0x37,
        0x5b, 0x2d, 0x16, 0x0b, 0x05, 0x42, 0x21, 0x50,
        0x28, 0x54, 0x2a, 0x55, 0x6a, 0x75, 0x7a, 0x7d,
        0x3e, 0x5f, 0x2f, 0x17, 0x4b, 0x25, 0x52, 0x29,
        0x14, 0x0a, 0x45, 0x62, 0x31, 0x58, 0x6c, 0x76,
        0x3b, 0x1d, 0x0e, 0x47, 0x63, 0x71, 0x78, 0x7c,
        0x7e, 0x7f, 0x3f, 0x1f, 0x0f, 0x07, 0x43, 0x61,
        0x70, 0x38, 0x5c, 0x6e, 0x77, 0x7b, 0x3d, 0x1e,
        0x4f, 0x27, 0x53, 0x69, 0x34, 0x1a, 0x4d, 0x26,
        0x13, 0x49, 0x24, 0x12, 0x09, 0x04, 0x02, 0x01,
        0x40, 0x20, 0x10, 0x08, 0x44, 0x22, 0x11, 0x48,
        0x64, 0x32, 0x19, 0x0c, 0x46, 0x23, 0x51, 0x68,
        0x74, 0x3a, 0x5d, 0x2e, 0x57, 0x6b, 0x35, 0x5a
    ]]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [8 for _ in range(4 * cls.num_rounds)]

    @classmethod
    def eval(cls, *mk):
        mk = list(reversed(mk))  # mk[i] = mki

        d = cls.constants

        def sk_round_i(round_i):
            assert round_i <= 31
            sk = []
            for i in range(8):
                for j in range(8):
                    if 4*round_i <= 16 * i + j < 4*round_i + 4:
                        sk.append(mk[(j - i) % 8] + d[16 * i + j])
                    elif 4*round_i <= 16 * i + j + 8 < 4*round_i + 4:
                        sk.append(mk[((j - i) % 8) + 8] + d[16 * i + j + 8])
            return sk

        rk = []
        for r in range(cls.num_rounds):
            if r == 0:
                wk0, wk1, wk2, wk3 = [mk[i + 12] for i in range(4)]
                rk.extend([wk0, wk1, wk2, wk3])

            elif r < cls.num_rounds - 1:
                sk0, sk1, sk2, sk3 = sk_round_i(r - 1)
                rk.extend([sk0, sk1, sk2, sk3])

            else:
                assert r == cls.num_rounds - 1
                if r == 33:
                    wk4, wk5, wk6, wk7 = [mk[i - 4] for i in range(4, 8)]
                    rk.extend([wk4, wk5, wk6, wk7])
                else:
                    sk0, sk1, sk2, sk3 = sk_round_i(r - 1)
                    rk.extend([sk0, sk1, sk2, sk3])

            cls.add_round_outputs(*rk[-4:])

        return rk


class HightEncryption(Encryption, RoundBasedFunction):
    """Encryption function."""

    num_rounds = 34
    input_widths = [8 for _ in range(8)]
    output_widths = [8 for _ in range(8)]
    round_keys = None

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds

    @classmethod
    def initial_transformation(cls, p, wk3, wk2, wk1, wk0):
        x = [None for _ in range(len(p))]
        x[0] = p[0] + wk0
        x[1] = p[1]
        x[2] = p[2] ^ wk1
        x[3] = p[3]
        x[4] = p[4] + wk2
        x[5] = p[5]
        x[6] = p[6] ^ wk3
        x[7] = p[7]
        return x

    @classmethod
    def round_function(cls, x, sk3, sk2, sk1, sk0):  # SK4i+3,SK4i+2,SK4i+1,SK4i
        def f0(bv):
            return ROL(bv, 1) ^ ROL(bv, 2) ^ ROL(bv, 7)

        def f1(bv):
            return ROL(bv, 3) ^ ROL(bv, 4) ^ ROL(bv, 6)

        # there is a typo in Section 2.4 of Hight paper;  using Fig. 3 instead
        y = [None for _ in range(len(x))]
        y[1] = x[0]
        y[3] = x[2]
        y[5] = x[4]
        y[7] = x[6]
        y[0] = x[7] ^ (f0(x[6]) + sk3)
        y[2] = x[1] + (f1(x[0]) ^ sk0)  # sk2
        y[4] = x[3] ^ (f0(x[2]) + sk1)
        y[6] = x[5] + (f1(x[4]) ^ sk2)  # sk0
        return y

    @classmethod
    def final_transformation(cls, x, wk7, wk6, wk5, wk4):
        c = [None for _ in range(len(x))]
        c[0] = x[1] + wk4
        c[1] = x[2]
        c[2] = x[3] ^ wk5
        c[3] = x[4]
        c[4] = x[5] + wk6
        c[5] = x[6]
        c[6] = x[7] ^ wk7
        c[7] = x[0]
        return c

    @classmethod
    def eval(cls, *p):  # p7,...,p0
        x = list(reversed(p))
        for r in range(cls.num_rounds):
            if r == 0:
                wk0, wk1, wk2, wk3 = cls.round_keys[4*r: 4*r + 4]
                x = cls.initial_transformation(x, wk3, wk2, wk1, wk0)
            elif r < cls.num_rounds - 1:
                sk0, sk1, sk2, sk3 = cls.round_keys[4*r: 4*r + 4]
                x = cls.round_function(x, sk3, sk2, sk1, sk0)
            else:
                assert r == cls.num_rounds - 1
                if r == 33:
                    wk4, wk5, wk6, wk7 = cls.round_keys[4*r: 4*r + 4]
                    x = cls.final_transformation(x, wk7, wk6, wk5, wk4)
                else:
                    sk0, sk1, sk2, sk3 = cls.round_keys[4*r: 4*r + 4]
                    x = cls.round_function(x, sk3, sk2, sk1, sk0)
            cls.add_round_outputs(*x)

        return list(reversed(x))


class HightCipher(Cipher):
    """HIGHT cipher."""
    key_schedule = HightKeySchedule
    encryption = HightEncryption
    _max_num_rounds = 34

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        assert new_num_rounds <= cls._max_num_rounds
        cls.key_schedule.set_num_rounds(new_num_rounds)
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        """Test Hight with official test vectors."""
        # https://tools.ietf.org/html/draft-kisa-hight-00#section-5
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(34)

        plaintext = (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
        key = (0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
               0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff)
        assert cls(plaintext, key) == (0x00, 0xf4, 0x18, 0xae, 0xd9, 0x4f, 0x03, 0xf2)

        plaintext = (0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77)
        key = (0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00)
        assert cls(plaintext, key) == (0x23, 0xce, 0x9f, 0x72, 0xe5, 0x43, 0xe6, 0xd8)

        cls.set_num_rounds(old_num_rounds)
