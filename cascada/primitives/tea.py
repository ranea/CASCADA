"""TEA cipher."""
from cascada.bitvector.core import Constant

from cascada.bitvector.ssa import BvFunction, RoundBasedFunction
from cascada.primitives.blockcipher import Encryption, Cipher


class TEAKeySchedule(BvFunction):
    """Key schedule function."""

    input_widths = [32, 32, 32, 32]
    output_widths = [32, 32, 32, 32]

    @classmethod
    def eval(cls, *master_key):
        return master_key


class TEAEncryption(Encryption, RoundBasedFunction):
    """Encryption function."""

    num_rounds = 32
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
        s = Constant(0, 32)
        delta = Constant(0x9e3779b9, 32)
        k0, k1, k2, k3 = cls.round_keys
        for i in range(cls.num_rounds):
            s += delta
            v0 += ((v1 << Constant(4, 32)) + k0) ^ (v1 + s) ^ ((v1 >> Constant(5, 32)) + k1)
            v1 += ((v0 << Constant(4, 32)) + k2) ^ (v0 + s) ^ ((v0 >> Constant(5, 32)) + k3)
            cls.add_round_outputs(v0, v1)
        return v0, v1


class TEACipher(Cipher):
    """TEA cipher."""
    key_schedule = TEAKeySchedule
    encryption = TEAEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        """Test TEA with official test vectors."""
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(32)

        plaintext = (0, 0)
        key = (0, 0, 0, 0)
        assert cls(plaintext, key) == (0x41EA3A0A, 0x94BAA940)

        plaintext = (0x01020304, 0x05060708)
        key = (0x00112233, 0x44556677, 0x8899AABB, 0xCCDDEEFF)
        assert cls(plaintext, key) == (0xDEB1C0A2, 0x7E745DB3)

        cls.set_num_rounds(old_num_rounds)
