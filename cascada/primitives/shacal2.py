"""SHACAL-2 cipher (based on SHA-256)."""
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateRight as ROR
from cascada.bitvector.secondaryop import BvIf, BvMaj

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


N = 4  # at least 4, at most 16
"""number of key words (N=4 for 128-bit key)."""

assert N >= 4
# N == 16, optimized for default single-key
# N == 4, ref for linear related-key characteristics
# N == 4, optimized for non-linear related-key characteristics

REFERENCE_VERSION = False  # True ==constants added in the encryption

k_cts = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]
k_cts = [Constant(k, 32) for k in k_cts]


class SHACAL2KeySchedule(RoundBasedFunction):
    """Key schedule function."""

    num_rounds = 64
    input_widths = [32 for _ in range(N)]
    output_widths = [32 for _ in range(64 - (16 - N))]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [32 for _ in range(N + max(new_num_rounds - 16, 0))]

    @classmethod
    def eval(cls, *W):  # w0, w1, ...
        rk = list(W)

        for i in range(N, 16):
            rk.append(Constant(0, 32))

        sigma0 = lambda x: ROR(x, 7) ^ ROR(x, 18) ^ (x >> Constant(3, 32))
        sigma1 = lambda x: ROR(x, 17) ^ ROR(x, 19) ^ (x >> Constant(10, 32))

        for i in range(16, cls.num_rounds):
            rk.append(sigma1(rk[i-2]) + rk[i-7] + sigma0(rk[i-15]) + rk[i-16])

        if REFERENCE_VERSION:
            pass
        else:
            for i in range(cls.num_rounds):
                if N <= i <= 15:
                    # rk[:N] + rk[16:cls.num_rounds] does not include rk[N],...,rk[15]
                    continue
                rk[i] += k_cts[i]

        return rk[:N] + rk[16:cls.num_rounds]


class SHACAL2Encryption(Encryption, RoundBasedFunction):
    """Encryption function."""

    num_rounds = 64
    input_widths = [32 for _ in range(8)]
    output_widths = [32 for _ in range(8)]
    round_keys = None

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds

    @classmethod
    def round_function(cls, A, B, C, D, E, F, G, H, i):
        if i < N:
            W = cls.round_keys[i]
        elif N <= i < 16:
            if REFERENCE_VERSION:
                W = Constant(0, 32)
            else:
                W = k_cts[i]
        else:
            W = cls.round_keys[i - (16 - N)]

        delta0 = lambda x: ROR(x, 2) ^ ROR(x, 13) ^ ROR(x, 22)
        delta1 = lambda x: ROR(x, 6) ^ ROR(x, 11) ^ ROR(x, 25)

        if REFERENCE_VERSION:
            T1 = H + delta1(E) + BvIf(E, F, G) + W + k_cts[i]  # ref
        else:
            T1 = H + delta1(E) + BvIf(E, F, G) + W  # optimized
        T2 = delta0(A) + BvMaj(A, B, C)
        return [
            T1 + T2,
            A,
            B,
            C,
            D + T1,
            E,
            F,
            G
        ]

    @classmethod
    def eval(cls, A, B, C, D, E, F, G, H):
        for i in range(cls.num_rounds):
            A, B, C, D, E, F, G, H = cls.round_function(A, B, C, D, E, F, G, H, i)
            cls.add_round_outputs(A, B, C, D, E, F, G, H)

        return A, B, C, D, E, F, G, H


class SHACAL2Cipher(Cipher):
    """SHACAL-2 cipher."""
    key_schedule = SHACAL2KeySchedule
    encryption = SHACAL2Encryption
    _min_num_rounds = N

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        assert new_num_rounds >= cls._min_num_rounds
        cls.key_schedule.set_num_rounds(new_num_rounds)
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        # https://www.cosic.esat.kuleuven.be/nessie/testvectors/
        # key =
        # 80000000000000000000000000000000
        # 00000000000000000000000000000000
        # 00000000000000000000000000000000
        # 00000000000000000000000000000000
        # plain =
        # 00000000000000000000000000000000
        # 00000000000000000000000000000000
        # cipher =
        # 361AB632 2FA9E7A7 BB23818D 839E01BD
        # DAFDF473 05426EDD 297AEDB9 F6202BAE

        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(64)

        global REFERENCE_VERSION

        for ref_v in [True, False]:
            REFERENCE_VERSION = ref_v

            key = [Constant(0x80000000, 32)]
            key.extend([Constant(0, 32) for _ in range(1, N)])
            pt = [Constant(0, 32) for _ in range(8)]
            ct = [
                Constant(0x361AB632, 32),
                Constant(0x2FA9E7A7, 32),
                Constant(0xBB23818D, 32),
                Constant(0x839E01BD, 32),
                Constant(0xDAFDF473, 32),
                Constant(0x05426EDD, 32),
                Constant(0x297AEDB9, 32),
                Constant(0xF6202BAE, 32),
            ]
            assert cls(pt, key) == tuple(ct)

            if N == 16:
                # key =
                # 00010203 04050607 08090A0B 0C0D0E0F
                # 10111213 14151617 18191A1B 1C1D1E1F
                # 20212223 24252627 28292A2B 2C2D2E2F
                # 30313233 34353637 38393A3B 3C3D3E3F
                # plain =
                # 00112233 44556677 8899AABB CCDDEEFF
                # 10213243 54657687 98A9BACB DCEDFE0F
                # cipher =
                # 1A6B234A 20EAD408 C2D83B35 8AC81D7A
                # 648ED25D 01B7C9EC 9CC4C9E2 5CFA813E

                key = [
                    Constant(0x00010203, 32), Constant(0x04050607, 32), Constant(0x08090A0B, 32), Constant(0x0C0D0E0F, 32),
                    Constant(0x10111213, 32), Constant(0x14151617, 32), Constant(0x18191A1B, 32), Constant(0x1C1D1E1F, 32),
                    Constant(0x20212223, 32), Constant(0x24252627, 32), Constant(0x28292A2B, 32), Constant(0x2C2D2E2F, 32),
                    Constant(0x30313233, 32), Constant(0x34353637, 32), Constant(0x38393A3B, 32), Constant(0x3C3D3E3F, 32),
                ]
                pt = [
                    Constant(0x00112233, 32),
                    Constant(0x44556677, 32),
                    Constant(0x8899AABB, 32),
                    Constant(0xCCDDEEFF, 32),
                    Constant(0x10213243, 32),
                    Constant(0x54657687, 32),
                    Constant(0x98A9BACB, 32),
                    Constant(0xDCEDFE0F, 32),
                ]
                ct = [
                    Constant(0x1A6B234A, 32),
                    Constant(0x20EAD408, 32),
                    Constant(0xC2D83B35, 32),
                    Constant(0x8AC81D7A, 32),
                    Constant(0x648ED25D, 32),
                    Constant(0x01B7C9EC, 32),
                    Constant(0x9CC4C9E2, 32),
                    Constant(0x5CFA813E, 32),
                ]
                assert cls(pt, key) == tuple(ct)

        cls.set_num_rounds(old_num_rounds)
