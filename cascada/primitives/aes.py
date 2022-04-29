"""AES-128.

A `WeakModel` is used to model the XOR and linear models of the S-box,
and a `BranchNumberModel` is used to model the XOR and linear models of MixColumns.

AES source: https://csrc.nist.gov/csrc/media/publications/fips/197/final/documents/fips-197.pdf
"""
import itertools

from cascada.bitvector.core import Constant
from cascada.bitvector.secondaryop import LutOperation, MatrixOperation
from cascada.bitvector.ssa import RoundBasedFunction
from cascada.differential.difference import XorDiff
from cascada.differential.opmodel import get_weak_model as get_differential_weak_model
from cascada.differential.opmodel import get_branch_number_model as get_differential_branch_number_model
from cascada.linear.opmodel import get_weak_model as get_linear_weak_model
from cascada.linear.opmodel import get_branch_number_model as get_linear_branch_number_model
from cascada.primitives.blockcipher import Encryption, Cipher


from cascada.primitives.aes_like import LoggingMode, AESLikeFunction


def _hex2byte_list(state):
    """Convert the hexadecimal string to a byte list

        >>> _hex2byte_list("000102030405060708090a0b0c0d0e0f")
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]

    """
    byte_list = []
    for i in range(0, len(state), 2):
        my_byte = int(state[i:i + 2], base=16)
        byte_list.append(Constant(my_byte, 8))
    return byte_list


class AESKeySchedule(RoundBasedFunction):
    """Key schedule of AES-128."""
    num_rounds = 10
    input_widths = [8 for _ in range(16)]
    output_widths = [8 for _ in range(16 * (10 + 1))]
    logging_mode = LoggingMode.Silent

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.output_widths = [8 for _ in range(16 * (new_num_rounds + 1))]

    @classmethod
    def xor_words(cls, my_word, my_other_word):
        return [x ^ y for x, y in zip(my_word, my_other_word, strict=True)]

    @classmethod
    def sub_word(cls, my_word):
        return [SboxLut(x) for x in my_word]

    @classmethod
    def rot_word(cls, my_word):
        # [a0, a1, a2, a3]  ->  [a1, a2, a3, a0]
        return [my_word[(i + 1) % 4] for i in range(4)]

    @classmethod
    def log_round_key_matrices(cls, list_rk):
        class AuxAESLikeFunction(AESLikeFunction):
            num_rows, num_columns = 4, 4
            cell_width = 8
            logging_mode = cls.logging_mode

        for r in range(cls.num_rounds + 1):
            cls.log_msg(f"round keys of round {r+1}:")  # (1 <= {r+1} <= {cls.num_rounds + 1})
            my_matrix = AuxAESLikeFunction.list2matrix(list_rk[r*16:r*16 + 16])
            fs, ffo = AuxAESLikeFunction.log_activity_matrix_state(my_matrix, return_format=True)
            cls.log_msg(fs, ffo)

    @classmethod
    def eval(cls, *master_key):
        Nk = 4  # Number of 32-bit words comprising the Cipher Key
        Nb = 4  # Number of columns (32-bit words) comprising the State

        if Nk == 4:  # AES-128
            Rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
            Rcon = [[Constant(x, 8) for x in word.to_bytes(Nk, 'little')] for word in Rcon]
        else:
            raise ValueError("invalid Nk")

        w = []
        for i in range(Nk):
            w.append([master_key[4*i], master_key[4*i+1], master_key[4*i+2], master_key[4*i+3]])
            # cls.log_msg(f"w[{i}][0] = {{}}", [w[-1][0]])

        for i in range(Nk, Nb * (cls.num_rounds + 1)):
            temp = w[i - 1]
            if i % Nk == 0:
                temp = cls.rot_word(temp)
                temp = cls.xor_words(cls.sub_word(temp), Rcon[(i // Nk) - 1])
            elif Nk > 6 and i % Nk == 4:
                temp = cls.sub_word(temp)
            w.append(cls.xor_words(w[i - Nk], temp))
            # cls.log_msg(f"w[{i}][0] = {{}}", [w[-1][0]])

        assert len(w) == (cls.num_rounds + 1) * 4  # Nr + 1 == num matrix_rk

        # each word is a column of rk bytes that is XORed to the state in the encryption
        list_rk = tuple(itertools.chain.from_iterable(w))  # flatten

        if cls.logging_mode != LoggingMode.Silent:
            cls.log_round_key_matrices(list_rk)

        return list_rk  # ordered by columns

    @classmethod
    def test(cls):
        """Test the key-schedule of AES-128 with official test vectors."""
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(10)

        # https://csrc.nist.gov/csrc/media/publications/fips/197/final/documents/fips-197.pdf
        test_vectors = [  # masterkey, w[i]
            [
                "2b7e151628aed2a6abf7158809cf4f3c",
                ["2b7e1516", "28aed2a6", "abf71588", "09cf4f3c",
                 "a0fafe17", "88542cb1", "23a33939", "2a6c7605",
                 "f2c295f2", "7a96b943", "5935807a", "7359f67f",
                 "3d80477d", "4716fe3e", "1e237e44", "6d7a883b",
                 "ef44a541", "a8525b7f", "b671253b", "db0bad00",
                 "d4d1c6f8", "7c839d87", "caf2b8bc", "11f915bc",
                 "6d88a37a", "110b3efd", "dbf98641", "ca0093fd",
                 "4e54f70e", "5f5fc9f3", "84a64fb2", "4ea6dc4f",
                 "ead27321", "b58dbad2", "312bf560", "7f8d292f",
                 "ac7766f3", "19fadc21", "28d12941", "575c006e",
                 "d014f9a8", "c9ee2589", "e13f0cc8", "b6630ca6"]
            ]
        ]
        for masterkey, list_word_keys in test_vectors:
            masterkey = _hex2byte_list(masterkey)
            round_keys = tuple(itertools.chain.from_iterable([_hex2byte_list(w) for w in list_word_keys]))
            result = cls(*masterkey)
            assert result == round_keys, f"\n{cls.get_name()}({masterkey}):\n{result}\nexpected:\n{round_keys}"

        cls.set_num_rounds(old_num_rounds)


_lut = [
    99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118,
    202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192,
    183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4,
    199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44,
    26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32,
    252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77,
    51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245,
    188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196,
    167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238,
    184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98,
    145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234,
    101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75,
    189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193,
    29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40,
    223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22
]


class SboxLut(LutOperation):
    """The 8-bit S-box of AES."""
    lut = [Constant(x, 8) for x in _lut]


# weight 1 to count number of active S-boxes
SboxLut.xor_model = get_differential_weak_model(SboxLut, XorDiff, 1)
SboxLut.linear_model = get_linear_weak_model(SboxLut, 1)


_matrix = [
    (0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1),
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1),
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1),
    (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
    (0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0)
]


class MixColumnsBitMatrix(MatrixOperation):
    """The (32, 32) binary matrix representing MixColumns."""
    arity = [4, 0]
    matrix = [[Constant(b, 1) for b in row] for row in _matrix]


MixColumnsBitMatrix.xor_model = get_differential_branch_number_model(MixColumnsBitMatrix, XorDiff, (8,)*4, 5, 0)
MixColumnsBitMatrix.linear_model = get_linear_branch_number_model(MixColumnsBitMatrix, (8,)*4, 5, 0)


class AESEncryption(Encryption, AESLikeFunction):
    """Encryption function of AES-128."""

    num_rounds = 10
    num_rows, num_columns = 4, 4
    cell_width = 8

    sbox, mix_columns_bit_matrix = SboxLut, MixColumnsBitMatrix

    name_add_round_constants, name_sub_cells = "AddRoundKey", "SubBytes"
    logging_mode = LoggingMode.Silent

    @classmethod
    def eval(cls, *plaintext):
        matrix_state = cls.list2matrix(plaintext)

        if cls.logging_mode != LoggingMode.Silent:
            cls.log_msg("\nplaintext:")
            cls.log_activity_matrix_state(matrix_state)

        for r in range(cls.num_rounds):
            matrix_round_keys = cls.list2matrix(cls.round_keys[r*16: (r+1)*16])

            if r < cls.num_rounds - 1:
                matrix_state = cls.add_round_constant(matrix_state, matrix_round_keys)
                matrix_state = cls.sub_cells(matrix_state)
                matrix_state = cls.shift_rows(matrix_state)
                matrix_state = cls.mix_columns(matrix_state)
            elif r == cls.num_rounds - 1:
                matrix_state = cls.add_round_constant(matrix_state, matrix_round_keys)
                matrix_state = cls.sub_cells(matrix_state)
                matrix_state = cls.shift_rows(matrix_state)
                last_matrix_round_keys = cls.list2matrix(
                    cls.round_keys[(r + 1) * 16: (r + 2) * 16])
                matrix_state = cls.add_round_constant(matrix_state, last_matrix_round_keys)

            if cls.logging_mode != LoggingMode.Silent:
                cls.log_msg(f"output of round {r+1}:")  # 1 <= {r+1} <= {cls.num_rounds}
                cls.log_activity_matrix_state(matrix_state)
                cls.log_msg("", [])  # new line

            cls.add_round_outputs(*cls.matrix2list(matrix_state))

        return tuple(cls.matrix2list(matrix_state))


class AESCipher(Cipher):
    """The block cipher AES-128."""
    key_schedule = AESKeySchedule
    encryption = AESEncryption

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.key_schedule.set_num_rounds(new_num_rounds)
        cls.encryption.set_num_rounds(new_num_rounds)

    @classmethod
    def test(cls):
        old_num_rounds = cls.num_rounds
        cls.set_num_rounds(10)

        test_vectors = [  # key, plaintext, ciphertext
            # https://csrc.nist.gov/csrc/media/publications/fips/197/final/documents/fips-197.pdf
            ["2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734", "3925841d02dc09fbdc118597196a0b32"],
            ["000102030405060708090a0b0c0d0e0f", "00112233445566778899aabbccddeeff", "69c4e0d86a7b0430d8cdb78070b4c55a"],
            # https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/aes/AESAVS.pdf
            ["00000000000000000000000000000000", "f34481ec3cc627bacd5dc3fb08f273e6", "0336763e966d92595a567cc9ce537f5e"],
            ["00000000000000000000000000000000", "9798c4640bad75c7c3227db910174e72", "a9a1631bf4996954ebc093957b234589"],
            ["10a58869d74be5a374cf867cfb473859", "00000000000000000000000000000000", "6d251e6944b051e04eaa6fb4dbf78465"],
            ["caea65cdbb75e9169ecd22ebe6e54675", "00000000000000000000000000000000", "6e29201190152df4ee058139def610bb"],
        ]

        for key, plaintext, ciphertext in test_vectors:
            key, plaintext, ciphertext = _hex2byte_list(key), _hex2byte_list(plaintext), tuple(_hex2byte_list(ciphertext))
            result = cls(plaintext, key)
            assert result == ciphertext, \
                f"{cls.get_name()}({plaintext}, {key}):\n{result}\nexpected:\n{ciphertext}"

        cls.set_num_rounds(old_num_rounds)
