"""FEAL-N cipher."""
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


FEALX = True
"""whether to use `FEALXKeySchedule`."""


def S0(X1, X2):
    return RotateLeft(X1 + X2, 2)


def S1(X1, X2):
    return RotateLeft(X1 + X2 + Constant(1, 8), 2)


def xor(l1, l2):
    assert len(l1) == len(l2)
    return [l1[i] ^ l2[i] for i in range(len(l1))]


class FEALKeySchedule(RoundBasedFunction):
    """Key schedule function."""

    num_rounds = 8
    input_widths = [8 for _ in range(32//8 * 2)]
    output_widths = [8 for i in range(16//8 * (8 + 8))]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [8 for _ in range(16 // 8 * (new_num_rounds + 8))]

    @classmethod
    def fk(cls, a0, a1, a2, a3, b0, b1, b2, b3):
        fk1 = a1 ^ a0
        fk2 = a2 ^ a3
        fk1 = S1(fk1, fk2 ^ b0)
        fk2 = S0(fk2, fk1 ^ b1)
        fk0 = S0(a0, fk1 ^ b2)
        fk3 = S1(a3, fk2 ^ b3)
        return fk0, fk1, fk2, fk3

    @classmethod
    def eval(cls, A00, A01, A02, A03, B00, B01, B02, B03):
        A = A00, A01, A02, A03
        B = B00, B01, B02, B03
        D = [Constant(0, 8) for _ in range(4)]

        K = []
        for r in range(cls.num_rounds//2 + 4):
            D, A, B = A, B, cls.fk(*A, *xor(B, D))
            K.extend(B)

        if cls.num_rounds % 2 != 0:
            D, A, B = A, B, cls.fk(*A, *xor(B, D))
            K.extend(B[:2])

        return K


class FEALXKeySchedule(FEALKeySchedule):
    """Key schedule function."""

    num_rounds = 8
    input_widths = [8 for _ in range(64//8 * 2)]
    output_widths = [8 for i in range(16//8 * (8 + 8))]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [8 for _ in range(16 // 8 * (new_num_rounds + 8))]

    @classmethod
    def eval(cls, KL0, KL1, KL2, KL3, KL4, KL5, KL6, KL7,
             KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7):
        KL = [KL0, KL1, KL2, KL3, KL4, KL5, KL6, KL7]
        KR = [KR0, KR1, KR2, KR3, KR4, KR5, KR6, KR7]
        KR1, KR2 = KR[:4], KR[4:]
        Q = []
        for r in range(cls.num_rounds//2 + 4):
            if (r + 1) % 3 == 1:
                Q.append(xor(KR1, KR2))
            elif (r + 1) % 3 == 2:
                Q.append(KR1)
            elif (r + 1) % 3 == 0:
                Q.append(KR2)

        A = KL[:4]
        B = KL[4:]
        D = [Constant(0, 8) for _ in range(4)]
        K = []
        for r in range(cls.num_rounds//2 + 4):
            D, A, B = A, B, cls.fk(*A, *xor(B, xor(D, Q[r])))
            K.extend(B)

        if cls.num_rounds % 2 != 0:
            if cls.num_rounds % 3 == 1:
                Qr = xor(KR1, KR2)
            elif cls.num_rounds % 3 == 2:
                Qr = KR1
            else:
                assert cls.num_rounds % 3 == 0
                Qr = KR2
            D, A, B = A, B, cls.fk(*A, *xor(B, xor(D, Qr)))
            K.extend(B[:2])

        return K


class FEALEncryption(Encryption, RoundBasedFunction):
    """Encryption function."""

    num_rounds = 8
    input_widths = [8 for _ in range(32//8 * 2)]
    output_widths = [8 for _ in range(32//8 * 2)]
    round_keys = None

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds

    @classmethod
    def f(cls, a0, a1, a2, a3, b0, b1):
        f1 = a1 ^ b0
        f2 = a2 ^ b1
        f1 = f1 ^ a0
        f2 = f2 ^ a3
        f1 = S1(f1, f2)
        f2 = S0(f2, f1)
        f0 = S0(a0, f1)
        f3 = S1(a3, f2)
        return f0, f1, f2, f3

    @classmethod
    def eval(cls, L00, L01, L02, L03, R00, R01, R02, R03):
        L = L00, L01, L02, L03
        R = R00, R01, R02, R03
        K = cls.round_keys

        for r in range(cls.num_rounds):
            if r == 0:
                L, R = xor(L, K[-16:-12]),  xor(R, K[-12:-8])
                L, R = L, xor(R, L)

            R, L = xor(L, cls.f(*R, *K[2*r:2*(r+1)])), R

            if r == cls.num_rounds - 1:
                # L, R = R, L
                R, L = R, xor(L, R)
                R, L = xor(R, K[-8:-4]), xor(L, K[-4:])

            cls.add_round_outputs(*(R+L))

        return R + L

        # # decryption
        # R, L = xor(R, K[-8:-4]), xor(L, K[-4:])
        # R, L = R, xor(L, R)
        #
        # for r in range(cls.num_rounds):
        #     L, R = xor(R, cls.f(*L, *K[-2*(r+1)-16:-2*r-16])), L
        #
        # # L, R = R, L
        # L, R = L, xor(R, L)
        # L, R = xor(L, K[-16:-12]),  xor(R, K[-12:-8])
        #
        # return L + R


class FEALCipher(Cipher):
    """FEAL cipher."""
    key_schedule = FEALXKeySchedule if FEALX else FEALKeySchedule
    encryption = FEALEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.encryption.set_num_rounds(new_num_rounds)
        cls.key_schedule.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        """Test FEAL with official test vectors."""
        # Extracted from "The FEAL Cipher Family"
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(8)

        if not FEALX:
            plaintext = (0, 0, 0, 0, 0, 0, 0, 0)
            key = (0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF)
            ciphertext = (0xCE, 0xEF, 0x2C, 0x86, 0xF2, 0x49, 0x07, 0x52)
            assert cls(plaintext, key) == ciphertext
        else:
            plaintext = (0, 0, 0, 0, 0, 0, 0, 0)
            key = (0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF)
            key += key  # double the size
            ciphertext = (0x92, 0xBE, 0xB6, 0x5D, 0x0E, 0x93, 0x82, 0xFB)
            assert cls(plaintext, key) == ciphertext

        cls.set_num_rounds(old_num_rounds)
