"""NOEKEON cipher."""
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft, RotateRight, BvIdentity

from cascada.bitvector.ssa import BvFunction, RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


def theta(x, k):
    t = x[0] ^ x[2]
    t ^= RotateLeft(t, 8) ^ RotateRight(t, 8)
    x[1] ^= t
    x[3] ^= t

    x[0] ^= k[0]
    x[1] ^= k[1]
    x[2] ^= k[2]
    x[3] ^= k[3]

    t = x[1] ^ x[3]
    t ^= RotateLeft(t, 8) ^ RotateRight(t, 8)
    x[0] ^= t
    x[2] ^= t

    # force the creation of new intermediate properties for the output of theta
    # (to speed up the generation of characteristics)
    x[0] = BvIdentity(x[0])
    x[1] = BvIdentity(x[1])
    x[2] = BvIdentity(x[2])
    x[3] = BvIdentity(x[3])


def gamma(x):
    x[1] ^= ~(x[3] | x[2])
    x[0] ^= x[2] & x[1]

    t = x[3]
    x[3] = x[0]
    x[0] = t
    x[2] ^= x[0] ^ x[1] ^ x[3]

    x[1] ^= ~(x[3] | x[2])
    x[0] ^= x[2] & x[1]


def pi1(x):
    x[1] = RotateLeft(x[1], 1)
    x[2] = RotateLeft(x[2], 5)
    x[3] = RotateLeft(x[3], 2)


def pi2(x):
    x[1] = RotateRight(x[1], 1)
    x[2] = RotateRight(x[2], 5)
    x[3] = RotateRight(x[3], 2)


RC = (0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
      0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
      0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
      0x72, 0xE4, 0xD3, 0xBD, 0x61, 0xC2, 0x9F, 0x25,
      0x4A, 0x94)


class NOEKEONDirectKeySchedule(BvFunction):
    """NOEKEON direct mode key schedule function."""

    input_widths = [32, 32, 32, 32]
    output_widths = [32, 32, 32, 32]

    @classmethod
    def eval(cls, *master_key):
        return master_key


class NOEKEONIndirectKeySchedule(RoundBasedFunction):
    """NOEKEON indirect mode key schedule function."""

    num_rounds = 16
    input_widths = [32, 32, 32, 32]
    output_widths = [32, 32, 32, 32]

    null_vector = (Constant(0, 32), Constant(0, 32), Constant(0, 32), Constant(0, 32))

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        assert new_num_rounds < len(RC)
        cls.num_rounds = new_num_rounds

    @classmethod
    def eval(cls, *master_key):
        x = list(master_key)

        for i in range(cls.num_rounds):
            x[0] ^= Constant(RC[i], 32)
            theta(x, cls.null_vector)
            pi1(x)
            gamma(x)
            pi2(x)

        x[0] ^= Constant(RC[cls.num_rounds], 32)
        theta(x, cls.null_vector)

        return x


class NOEKEONEncryption(Encryption, RoundBasedFunction):
    """NOEKEON encryption function."""

    num_rounds = 16
    input_widths = [32, 32, 32, 32]
    output_widths = [32, 32, 32, 32]
    round_keys = None

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        assert new_num_rounds < len(RC)
        cls.num_rounds = new_num_rounds

    @classmethod
    def eval(cls, *x):
        x = list(x)

        for i in range(cls.num_rounds):
            x[0] ^= Constant(RC[i], 32)
            theta(x, cls.round_keys)
            pi1(x)
            gamma(x)
            pi2(x)
            cls.add_round_outputs(x)

        x[0] ^= Constant(RC[cls.num_rounds], 32)
        theta(x, cls.round_keys)

        return x


class NOEKEONDirectCipher(Cipher):
    """NOEKEON cipher in direct mode."""
    key_schedule = NOEKEONDirectKeySchedule
    encryption = NOEKEONEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        """Test NOEKEON direct mode with NESSIE test vectors."""
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(16)

        # https://www.cosic.esat.kuleuven.be/nessie/testvectors/bc/noekeon/Noekeon-Direct-128-128.verified.test-vectors
        # Set 8, vector#  1
        plaintext = (0x4765F3DA, 0x10CD3D04, 0x73867742, 0xB5E5CC3C)
        key = (0x2BD6459F, 0x82C5B300, 0x952C4910, 0x4881FF48)
        assert cls(plaintext, key) == (0xEA024714, 0xAD5C4D84, 0xEA024714, 0xAD5C4D84)

        cls.set_num_rounds(old_num_rounds)


class NOEKEONIndirectCipher(Cipher):
    """NOEKEON cipher in indirect mode."""
    key_schedule = NOEKEONIndirectKeySchedule
    encryption = NOEKEONEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.key_schedule.set_num_rounds(new_num_rounds)
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        """Test NOEKEON indirect mode with NESSIE test vectors."""
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(16)

        # https://www.cosic.esat.kuleuven.be/nessie/testvectors/bc/noekeon/Noekeon-Indirect-128-128.verified.test-vectors
        # Set 8, vector#  1
        plaintext = (0x023F84AD, 0x917B6F6C, 0x9F9592FD, 0xBAE4EB69)
        key = (0x2BD6459F, 0x82C5B300, 0x952C4910, 0x4881FF48)
        assert cls(plaintext, key) == (0xEA024714, 0xAD5C4D84, 0xEA024714, 0xAD5C4D84)

        cls.set_num_rounds(old_num_rounds)
