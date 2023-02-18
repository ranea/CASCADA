"""SKINNY-64-64.

TK1 version, where the tweakey size is equal to the block size.

This implementation is based on `skinny128`, but a `WDTModel`
is used to model the XOR and linear models of the S-box.
"""
from cascada.bitvector.core import Constant
from cascada.bitvector.secondaryop import LutOperation
from cascada.differential.difference import XorDiff
from cascada.differential.opmodel import get_wdt as get_differential_wdt
from cascada.differential.opmodel import get_wdt_model as get_differential_wdt_model
from cascada.linear.opmodel import get_wdt as get_linear_wdt
from cascada.linear.opmodel import get_wdt_model as get_linear_wdt_model


from cascada.primitives import skinny128


default_num_rounds = 32


class SKINNYTweakeySchedule(skinny128.SKINNYTweakeySchedule):
    """Key schedule of SKINNY-64-64."""
    num_rounds = default_num_rounds
    input_widths = [4 for _ in range(4*4)]
    output_widths = [4 for _ in range(4*2 * default_num_rounds)]  # 2 rows per round

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [4 for _ in range(4*2 * new_num_rounds)]


class SboxLut(LutOperation):
    """The 4-bit S-box of SKINNY-64-64."""
    lut = [Constant(x, 4) for x in [12, 6, 9, 0, 1, 10, 2, 11, 3, 8, 5, 13, 4, 14, 7, 15]]


SboxLut.xor_model = get_differential_wdt_model(SboxLut, XorDiff, get_differential_wdt(SboxLut, 4, 4))
SboxLut.linear_model = get_linear_wdt_model(SboxLut, get_linear_wdt(SboxLut, 4, 4))


class SKINNYEncryption(skinny128.SKINNYEncryption):
    """Encryption function of SKINNY-64-64."""

    num_rounds = default_num_rounds
    cell_width = 4

    ignore_first_sub_cells = False
    sbox = SboxLut

    logging_mode = skinny128.LoggingMode.Silent


class SKINNYCipher(skinny128.SKINNYCipher):
    """The block cipher SKINNY-64-64."""
    key_schedule = SKINNYTweakeySchedule
    encryption = SKINNYEncryption

    @classmethod
    def test(cls):
        old_num_rounds = cls.num_rounds
        old_ignore = cls.encryption.ignore_first_sub_cells
        cls.encryption.ignore_first_sub_cells = False
        cls.set_num_rounds(default_num_rounds)

        """
        0xb, 0x3, 0x9, 0xd, 0xf, 0xb, 0x2, 0x4, 0x2, 0x9, 0xb, 0x8, 0xa, 0xc, 0x7, """

        key = [0xf, 0x5, 0x2, 0x6,
               0x9, 0x8, 0x2, 0x6,
               0xf, 0xc, 0x6, 0x8,
               0x1, 0x2, 0x3, 0x8]
        plaintext = [0x0, 0x6, 0x0, 0x3,
                     0x4, 0xf, 0x9, 0x5,
                     0x7, 0x7, 0x2, 0x4,
                     0xd, 0x1, 0x9, 0xd]
        assert list(cls(plaintext, key)) == [
            0xb, 0xb, 0x3, 0x9,
            0xd, 0xf, 0xb, 0x2,
            0x4, 0x2, 0x9, 0xb,
            0x8, 0xa, 0xc, 0x7]

        cls.set_num_rounds(old_num_rounds)
        cls.encryption.ignore_first_sub_cells = old_ignore
