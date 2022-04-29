"""Permutation :math:`\pi` of Chaskey."""
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import RotateLeft

from cascada.bitvector.ssa import RoundBasedFunction


class ChaskeyPi(RoundBasedFunction):
    """The permutation :math:`\pi` of Chaskey as a bit-vector function."""
    num_rounds = 8
    input_widths = [32, 32, 32, 32]
    output_widths = [32, 32, 32, 32]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds

    @classmethod
    def eval(cls, v0, v1, v2, v3):
        for i in range(cls.num_rounds):
            v0 += v1
            v1 = RotateLeft(v1, 5)
            v1 ^= v0
            v0 = RotateLeft(v0, 16)

            v2 += v3
            v3 = RotateLeft(v3, 8)
            v3 ^= v2

            v0 += v3
            v3 = RotateLeft(v3, 13)
            v3 ^= v0

            v2 += v1
            v1 = RotateLeft(v1, 7)
            v1 ^= v2
            v2 = RotateLeft(v2, 16)

            cls.add_round_outputs(v0, v1, v2, v3)

        return v0, v1, v2, v3

    @classmethod
    def test(cls):
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(8)

        pt = [
            Constant(0x00000000, 32)
        ] * 4
        ct = [
            Constant(0x00000000, 32)
        ] * 4
        assert cls(*pt) == tuple(ct)

        cls.set_num_rounds(old_num_rounds)
