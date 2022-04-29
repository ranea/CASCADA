"""XTEA cipher."""
from cascada.bitvector.core import Constant

from cascada.bitvector.ssa import RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


class XTEAKeySchedule(RoundBasedFunction):
    """Key schedule function."""

    num_rounds = 64
    input_widths = [32, 32, 32, 32]
    output_widths = [32 for i in range(64)]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [32 for _ in range(new_num_rounds)]

    @classmethod
    def eval(cls, *master_key):
        mk = list(master_key)
        s = Constant(0, 32)
        delta = Constant(0x9E3779B9, 32)
        k = []
        for i in range(cls.num_rounds):
            if i % 2 == 0:
                k.append(s + mk[int(s & Constant(3, 32))])
                # s += delta
            else:
                k.append(s + mk[int((s >> Constant(11, 32)) & Constant(3, 32))])
            if i % 2 == 0:
                s += delta
            cls.add_round_outputs(k[-1])
        return k


class XTEAEncryption(Encryption, RoundBasedFunction):
    """Encryption function."""

    num_rounds = 64
    input_widths = [32, 32]
    output_widths = [32, 32]
    round_keys = None

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds

    @classmethod
    def eval(cls, x, y):
        v0 = x
        v1 = y
        k = cls.round_keys
        for i in range(cls.num_rounds):
            v0, v1 = v1, v0 + ((((v1 << Constant(4, 32)) ^ (v1 >> Constant(5, 32))) + v1) ^ k[i])
            cls.add_round_outputs(v0, v1)

        return v0, v1


class XTEACipher(Cipher):
    """XTEA cipher."""
    key_schedule = XTEAKeySchedule
    encryption = XTEAEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.encryption.set_num_rounds(new_num_rounds)
        cls.key_schedule.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        """Test XTEA with test vectors."""
        # https://go.googlesource.com/crypto/+/master/xtea/xtea_test.go
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(64)

        plaintext = (0x41424344, 0x45464748)
        key = (0, 0, 0, 0)
        assert cls(plaintext, key) == (0xa0390589, 0xf8b8efa5)

        plaintext = (0x41424344, 0x45464748)
        key = (0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F)
        assert cls(plaintext, key) == (0x497df3d0, 0x72612cb5)

        cls.set_num_rounds(64)
