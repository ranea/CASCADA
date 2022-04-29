"""Permutation :math:`\pi` of :math:`\pi`-Cipher (16-bit version)."""
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft as ROL

from cascada.bitvector.ssa import RoundBasedFunction


N = 4  # small version with N = 2

PI_MU_CONST_0 = Constant(0xF0E8, 16)
PI_MU_CONST_1 = Constant(0xE4E2, 16)
PI_MU_CONST_2 = Constant(0xE1D8, 16)
PI_MU_CONST_3 = Constant(0xD4D2, 16)

PI_MU_ROT_CONST_0 = 1
PI_MU_ROT_CONST_1 = 4
PI_MU_ROT_CONST_2 = 9
PI_MU_ROT_CONST_3 = 11

PI_NY_CONST_0 = Constant(0xD1CC, 16)
PI_NY_CONST_1 = Constant(0xCAC9, 16)
PI_NY_CONST_2 = Constant(0xC6C5, 16)
PI_NY_CONST_3 = Constant(0xC3B8, 16)

PI_NY_ROT_CONST_0 = 2
PI_NY_ROT_CONST_1 = 5
PI_NY_ROT_CONST_2 = 7
PI_NY_ROT_CONST_3 = 13

PI_CONST = [
    [0xB4B2, 0xB1AC, 0xAAA9, 0xA6A5],
    [0xA39C, 0x9A99, 0x9695, 0x938E],
    [0x8D8B, 0x8778, 0x7472, 0x716C],
    [0x6A69, 0x6665, 0x635C, 0x5A59],
    [0x5655, 0x534E, 0x4D4B, 0x473C],
    [0x3A39, 0x3635, 0x332E, 0x2D2B],
    # 0x271E, 0x1D1B, 0x170F, 0xF0E8,
    # 0xE4E2, 0xE1D8, 0xD4D2, 0xD1CC,
]
PI_CONST = [[Constant(x_i, 16) for x_i in x] for x in PI_CONST]


class PiPermutation(RoundBasedFunction):
    """The permutation :math:`\pi` as a bit-vector function."""
    num_rounds = 3 * 2
    input_widths = [16 for _ in range(N * 4)]
    output_widths = [16 for _ in range(N * 4)]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds

    @classmethod
    def mul(cls, x, y):
        assert len(x) == 4
        assert len(y) == 4

        # mu
        T = [None for _ in range(12)]
        # order?
        aux01 = x[0] + x[1]
        aux23 = x[2] + x[3]
        T[0] = ROL(aux01 + (PI_MU_CONST_0 + x[2]), PI_MU_ROT_CONST_0)
        T[1] = ROL(aux01 + (PI_MU_CONST_1 + x[3]), PI_MU_ROT_CONST_1)
        T[2] = ROL((x[0] + PI_MU_CONST_2) + aux23, PI_MU_ROT_CONST_2)
        T[3] = ROL((x[1] + PI_MU_CONST_3) + aux23, PI_MU_ROT_CONST_3)
        T[4] = T[0] ^ T[1] ^ T[3]
        T[5] = T[0] ^ T[1] ^ T[2]
        T[6] = T[1] ^ T[2] ^ T[3]
        T[7] = T[0] ^ T[2] ^ T[3]

        # ny
        aux01 = y[0] + y[1]
        aux23 = y[2] + y[3]
        T[0] = ROL((PI_NY_CONST_0 + y[0]) + aux23, PI_NY_ROT_CONST_0)
        T[1] = ROL((PI_NY_CONST_1 + y[1]) + aux23, PI_NY_ROT_CONST_1)
        T[2] = ROL(aux01 + (PI_NY_CONST_2 + y[2]), PI_NY_ROT_CONST_2)
        T[3] = ROL(aux01 + (PI_NY_CONST_3 + y[3]), PI_NY_ROT_CONST_3)
        T[8] = T[1] ^ T[2] ^ T[3]
        T[9] = T[0] ^ T[2] ^ T[3]
        T[10] = T[0] ^ T[1] ^ T[3]
        T[11] = T[0] ^ T[1] ^ T[2]

        # sigma
        z3 = T[4] + T[8]
        z0 = T[5] + T[9]
        z1 = T[6] + T[10]
        z2 = T[7] + T[11]

        return z0, z1, z2, z3

    @classmethod
    def e1(cls, c, j):
        j[:4] = cls.mul(c, j[:4])
        if N > 1:
            for i in range(1, N):
                j[4*i:4*i+4] = cls.mul(j[4*i-4:4*i], j[4*i:4*i+4])
        return j

    @classmethod
    def e2(cls, c, i):
        i[-4:] = cls.mul(i[-4:], c)
        if N > 1:
            for j in range(N-2, -1, -1):
                i[4*j:4*j+4] = cls.mul(i[4*j:4*j+4], i[4*j+4:4*j+8])
                # i[-4*j-4:-4*j]= cls.mul(i[-4*j-4:-4*j], i[-4*j:-4*j+4])
        return i

    @classmethod
    def eval(cls, *args):  # p0, p1, ...
        x = list(args)
        for i in range(cls.num_rounds):
            if i % 2 == 0:
                i = i // 2  # due to duplicating the number of rounds
                x = cls.e1(PI_CONST[2*i], x)
            else:
                i = (i - 1) // 2
                x = cls.e2(PI_CONST[2*i + 1], x)
            cls.add_round_outputs(*x)
        return x

    @classmethod
    def test(cls):
        old_num_rounds = cls.num_rounds

        cls.set_num_rounds(1)
        pt = [Constant(0x00000000, 16)] * (N * 4)
        ct = (0xe9f5, 0xcfab, 0x5198, 0x9eec,
              0xcb81, 0x7fb0, 0xd47a, 0x45b7,
              0xa5b5, 0xd8da, 0x2cc5, 0x0aa1,
              0x7bc9, 0x97f0, 0xc515, 0x7224)
        assert cls(*pt) == tuple(ct), "{}".format(cls(*pt))

        cls.set_num_rounds(2)
        pt = [Constant(0x00000000, 16)] * (N * 4)
        ct = (0x4d2f, 0x383a, 0xa84d, 0xe12c,
              0xee36, 0x2b82, 0xb624, 0x1e15,
              0x39f9, 0x2ded, 0x54c3, 0x15c9,
              0x58fe, 0xbff0, 0x44ad, 0x2b57)
        assert cls(*pt) == tuple(ct), "{}".format(cls(*pt))

        cls.set_num_rounds(6)
        pt = [Constant(0x00000000, 16)] * (N * 4)
        ct = (0x0274, 0x2a3d, 0x9d5e, 0x0319,
              0x32b4, 0x2751, 0x745b, 0xa328,
              0xd2d4, 0x1ae9, 0x8e70, 0x0fe6,
              0x9506, 0xe58d, 0x996b, 0x6075)
        assert cls(*pt) == tuple(ct), "{}".format(cls(*pt))

        cls.set_num_rounds(old_num_rounds)