"""SKINNY-128-128.

TK1 version, where the tweakey size is equal to the block size.

A `WeakModel` is used to model the XOR and linear models of the S-box.
"""
from cascada.bitvector.core import Constant
from cascada.bitvector.operation import BvIdentity
from cascada.bitvector.secondaryop import LutOperation
from cascada.bitvector.ssa import RoundBasedFunction
from cascada.differential.difference import XorDiff
from cascada.differential.opmodel import get_weak_model as get_differential_weak_model
from cascada.linear.opmodel import get_weak_model as get_linear_weak_model
from cascada.primitives.blockcipher import Encryption, Cipher


from cascada.primitives.aes_like import LoggingMode, AESLikeFunction


default_num_rounds = 40


class SKINNYTweakeySchedule(RoundBasedFunction):
    """Key schedule of SKINNY-128-128."""
    num_rounds = default_num_rounds
    input_widths = [8 for _ in range(4*4)]
    output_widths = [8 for _ in range(4*2 * default_num_rounds)]  # 2 rows per round
    _min_num_rounds = 2

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [8 for _ in range(4*2 * new_num_rounds)]

    @classmethod
    def permutation_Pt(cls, input_state):
        # TK1[i] ← TK1[PT[i]]
        Pt = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
        assert len(input_state) == 16 == len(Pt)
        output_state = []
        for i in range(16):
            output_state.append(input_state[Pt[i]])
        return output_state

    @classmethod
    def eval(cls, *master_key):
        tweakey_state = master_key
        list_tweakey_bytes = list(tweakey_state[:4*2])  # first 2 rows

        for i in range(cls.num_rounds - 1):
            # no LFSR in TK1
            tweakey_state = cls.permutation_Pt(tweakey_state)
            list_tweakey_bytes.extend(tweakey_state[:4*2])

        return list_tweakey_bytes


_lut = [
    101, 76, 106, 66, 75, 99, 67, 107, 85, 117, 90, 122, 83, 115, 91, 123, 53, 140,
    58, 129, 137, 51, 128, 59, 149, 37, 152, 42, 144, 35, 153, 43, 229, 204, 232,
    193, 201, 224, 192, 233, 213, 245, 216, 248, 208, 240, 217, 249, 165, 28, 168,
    18, 27, 160, 19, 169, 5, 181, 10, 184, 3, 176, 11, 185, 50, 136, 60, 133, 141,
    52, 132, 61, 145, 34, 156, 44, 148, 36, 157, 45, 98, 74, 108, 69, 77, 100, 68,
    109, 82, 114, 92, 124, 84, 116, 93, 125, 161, 26, 172, 21, 29, 164, 20, 173, 2,
    177, 12, 188, 4, 180, 13, 189, 225, 200, 236, 197, 205, 228, 196, 237, 209,
    241, 220, 252, 212, 244, 221, 253, 54, 142, 56, 130, 139, 48, 131, 57, 150, 38,
    154, 40, 147, 32, 155, 41, 102, 78, 104, 65, 73, 96, 64, 105, 86, 118, 88, 120,
    80, 112, 89, 121, 166, 30, 170, 17, 25, 163, 16, 171, 6, 182, 8, 186, 0, 179,
    9, 187, 230, 206, 234, 194, 203, 227, 195, 235, 214, 246, 218, 250, 211, 243,
    219, 251, 49, 138, 62, 134, 143, 55, 135, 63, 146, 33, 158, 46, 151, 39, 159,
    47, 97, 72, 110, 70, 79, 103, 71, 111, 81, 113, 94, 126, 87, 119, 95, 127, 162,
    24, 174, 22, 31, 167, 23, 175, 1, 178, 14, 190, 7, 183, 15, 191, 226, 202, 238,
    198, 207, 231, 199, 239, 210, 242, 222, 254, 215, 247, 223, 255
]


class SboxLut(LutOperation):
    """The 8-bit S-box of SKINNY-128-128."""
    lut = [Constant(x, 8) for x in _lut]


# weight 1 to count number of active S-boxes
SboxLut.xor_model = get_differential_weak_model(SboxLut, XorDiff, 1)
SboxLut.linear_model = get_linear_weak_model(SboxLut, 1)


class SKINNYEncryption(Encryption, AESLikeFunction):
    """Encryption function of SKINNY-128-128."""

    num_rounds = default_num_rounds
    num_rows, num_columns = 4, 4
    cell_width = 8

    ignore_first_sub_cells = False
    sbox = SboxLut

    name_add_round_constants, name_sub_cells = "AddRoundTweakey", "SubCells"
    logging_mode = LoggingMode.Silent

    round_constants = [
        0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
        0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
        0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04,
        0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28, 0x10, 0x20
    ]

    @classmethod
    def shift_rows(cls, my_matrix_state):
        new_matrix_state = [row[:] for row in my_matrix_state]

        for r in range(cls.num_rows):
            offset = r  # shift offset (0 for r=0, 1 for r=1, ...)
            for c in range(cls.num_columns):
                # rotate to the right
                new_matrix_state[r][c] = my_matrix_state[r][(c - offset) % cls.num_columns]

        if cls.logging_mode not in [LoggingMode.Silent, LoggingMode.RoundOutputs]:
            cls.log_msg(f"output of {getattr(cls, 'name_shift_rows', 'ShiftRows')}:")
            cls.log_activity_matrix_state(new_matrix_state)

        return new_matrix_state

    @classmethod
    def mix_columns(cls, my_matrix_state):
        new_matrix_state = [row[:] for row in my_matrix_state]

        for index_column in range(cls.num_columns):
            input_column = [my_matrix_state[r][index_column] for r in range(cls.num_rows)]

            # M =
            # 1 0 1 1
            # 1 0 0 0
            # 0 1 1 0
            # 1 0 1 0
            output_column = [
                input_column[0] ^ input_column[2] ^ input_column[3],
                input_column[0],
                input_column[1] ^ input_column[2],
                input_column[0] ^ input_column[2]
            ]

            for r in range(cls.num_rows):
                new_matrix_state[r][index_column] = BvIdentity(output_column[r])  # copy with new name

        if cls.logging_mode not in [LoggingMode.Silent, LoggingMode.RoundOutputs]:
            cls.log_msg(f"output of {getattr(cls, 'name_mix_columns', 'MixColumns')}:")
            cls.log_activity_matrix_state(new_matrix_state)

        return new_matrix_state

    @classmethod
    def eval(cls, *plaintext):
        matrix_state = cls.list2matrix(plaintext, column_order=False)

        if cls.logging_mode != LoggingMode.Silent:
            cls.log_msg("\nplaintext:")
            cls.log_activity_matrix_state(matrix_state)

        for r in range(cls.num_rounds):
            if r == 0 and cls.ignore_first_sub_cells is True:
                pass
            else:
                matrix_state = cls.sub_cells(matrix_state)

            # AddConstants
            round_constant = cls.round_constants[r]
            c0 = Constant(round_constant & 0xF, cls.cell_width)
            c1 = Constant(round_constant >> 4, cls.cell_width)
            c2 = Constant(0x2, cls.cell_width)
            matrix_state[0][0] ^= c0
            matrix_state[1][0] ^= c1
            matrix_state[2][0] ^= c2
            if cls.logging_mode not in [LoggingMode.Silent, LoggingMode.RoundOutputs]:
                cls.log_msg("output of AddConstants:")
                cls.log_activity_matrix_state(matrix_state)

            current_round_key = list(cls.round_keys[r*(4*2): (r+1)*(4*2)])  # first 2 rows
            current_round_key += [Constant(0, cls.cell_width) for _ in range(4*2)]  # last 2 rows
            matrix_round_keys = cls.list2matrix(current_round_key, column_order=False)
            matrix_state = cls.add_round_constant(matrix_state, matrix_round_keys)

            matrix_state = cls.shift_rows(matrix_state)
            matrix_state = cls.mix_columns(matrix_state)

            if cls.logging_mode != LoggingMode.Silent:
                cls.log_msg(f"output of round {r+1}:")
                cls.log_activity_matrix_state(matrix_state)
                cls.log_msg("", [])

            cls.add_round_outputs(*cls.matrix2list(matrix_state))

        return tuple(cls.matrix2list(matrix_state, column_order=False))


class SKINNYCipher(Cipher):
    """The block cipher SKINNY-128-128."""
    key_schedule = SKINNYTweakeySchedule
    encryption = SKINNYEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.key_schedule.set_num_rounds(new_num_rounds)
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        old_num_rounds = cls.num_rounds
        old_ignore = cls.encryption.ignore_first_sub_cells
        cls.encryption.ignore_first_sub_cells = False
        cls.set_num_rounds(default_num_rounds)

        key = [0x4f, 0x55, 0xcf, 0xb0,
               0x52, 0x0c, 0xac, 0x52,
               0xfd, 0x92, 0xc1, 0x5f,
               0x37, 0x07, 0x3e, 0x93]
        plaintext = [0xf2, 0x0a, 0xdb, 0x0e,
                     0xb0, 0x8b, 0x64, 0x8a,
                     0x3b, 0x2e, 0xee, 0xd1,
                     0xf0, 0xad, 0xda, 0x14]
        assert list(cls(plaintext, key)) == [
            0x22, 0xff, 0x30, 0xd4,
            0x98, 0xea, 0x62, 0xd7,
            0xe4, 0x5b, 0x47, 0x6e,
            0x33, 0x67, 0x5b, 0x74]

        cls.set_num_rounds(old_num_rounds)
        cls.encryption.ignore_first_sub_cells = old_ignore
