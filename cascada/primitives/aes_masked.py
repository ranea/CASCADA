"""SMT-based experiments of `A Low-Randomness Second-Order Masked AES
<https://eprint.iacr.org/2021/1111>`_.

To run, call `find_best_key_schedule_characteristic` or `find_best_encryption_characteristic`.
"""
import functools
from decimal import Decimal

from cascada.bitvector.core import Constant, Term
from cascada.bitvector.operation import BvOr, BvComp, Concat, BvIdentity
from cascada.bitvector.secondaryop import LutOperation, PopCount
from cascada.linear.mask import LinearMask
from cascada.linear.opmodel import get_weak_model
from cascada.linear.chmodel import ChModel
from cascada.smt.chsearch import (
    ChModelAssertType, ChFinder, PrintingMode, _get_smart_print
)


from cascada.primitives import aes


class SboxA(aes.SboxLut):
    """Represent a generic S-box with the following absolute correlations:

      - zero -> zero: :math:`1`
      - non-zero -> non-zero: :math:`2^{-3}`
      - zero -> non-zero: :math:`2^{-3.8}`

    """


class SboxB(aes.SboxLut):
    """Represent a generic S-box with the following absolute correlations:

      - zero -> zero: :math:`1`
      - non-zero -> non-zero: :math:`2^{-2.6}`
      - zero -> non-zero: :math:`2^{-4}`

    """


class SboxZeroWeight(aes.SboxLut):
    """Represent a generic S-box with the following absolute correlations:

     - zero -> zero: 1
     - non-zero -> non-zero: 1
     - zero -> non-zero: 1

    """


SboxA.linear_model = get_weak_model(
    SboxA, Decimal(3), zero2nonzero_weight=Decimal(3.8), precision=4)
SboxB.linear_model = get_weak_model(
    SboxB, Decimal(2.6), zero2nonzero_weight=Decimal(4), precision=4)
SboxZeroWeight.linear_model = get_weak_model(
    SboxZeroWeight, 0, zero2nonzero_weight=0)


class AESMaskedKeySchedule(aes.AESKeySchedule):
    """Key schedule of masked AES-128."""
    num_rounds = 10
    input_widths = [8 for _ in range(16)] + [8 for _ in range(2 * 10)]
    output_widths = [8 for _ in range(16)]
    logging_mode = aes.LoggingMode.Silent

    num_extra_cells = 2*10
    ignore_first_last_sbox_weights = True

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        cls.num_rounds = new_num_rounds
        cls.num_extra_cells = 2 * cls.num_rounds
        cls.input_widths = [8 for _ in range(16)] + [8 for _ in range(cls.num_extra_cells)]

    @classmethod
    def _sub_word(cls, my_word, extra_cell, my_sbox):
        a, b, c, d = my_word

        sbox_inputs = [x for x in [a, b, c, d]]
        xor_values = [BvIdentity(xv) for xv in [b, c, d, extra_cell]]  # new copies with != names

        sbox_outputs = [my_sbox(x) for x in sbox_inputs]

        # changing of the guards
        a_ = sbox_outputs[0] ^ xor_values[0]
        b_ = sbox_outputs[1] ^ xor_values[1]
        c_ = sbox_outputs[2] ^ xor_values[2]
        d_ = sbox_outputs[3] ^ xor_values[3]

        new_word = [a_, b_, c_, d_]

        if cls.logging_mode == aes.LoggingMode.Debug:
            # ensuring the correct values are logged (i.e., not intermediate branches) by
            # using different names for sbox_inputs and xor_values
            name_sbox = my_sbox.__name__
            # cls.log_msg(" - word: [{},{},{},{}]\n - sbox_inputs: [{},{},{},{}]\n - xor_values: [{},{},{},{}]",
            #             my_word + sbox_inputs + xor_values)
            for i in range(len(sbox_outputs)):
                cls.log_msg(f" - {name_sbox}({{}}) -> {{}}   XOR   {{}}) -> {{}}",
                            [sbox_inputs[i], sbox_outputs[i], xor_values[i], new_word[i]])

        return new_word

    @classmethod
    def sub_word(cls, my_word):
        output_word = my_word[:]
        r = cls._current_sub_word_round

        if r == 0 and cls.ignore_first_last_sbox_weights:
            my_sbox = SboxZeroWeight
        else:
            my_sbox = SboxA
        output_word = cls._sub_word(output_word, cls._list_extra_cells[2*r + 0], my_sbox)

        if r == cls.num_rounds - 1 and cls.ignore_first_last_sbox_weights:
            my_sbox = SboxZeroWeight
        else:
            my_sbox = SboxB
        output_word = cls._sub_word(output_word, cls._list_extra_cells[2*r + 1], my_sbox)

        cls._current_sub_word_round += 1
        return output_word

    @classmethod
    def eval(cls, *master_key):
        cls._current_sub_word_round = 0
        assert cls.num_extra_cells > 0
        cls._list_extra_cells = master_key[-cls.num_extra_cells:]

        if cls.logging_mode != aes.LoggingMode.Silent:
            format_string = "extra cells:\n(" + ",".join(["{}"]*cls.num_extra_cells) + ")"
            cls.log_msg(format_string, list(cls._list_extra_cells[:]))

        result = super().eval(*master_key)

        assert cls._current_sub_word_round == cls.num_rounds
        return result[-16:]


def get_key_schedule_constraints(ch_model, verbose=False, filename=None):
    """Get the following initial constraints:

      - extra cells not active
      - (at most 2 adjacent cells in the last column of the input) OR (at most 1 row active in the input)
      - (at most 2 adjacent cells in the last column of the output) OR (at most 1 row active in the output)

    """
    initial_constraints = []
    smart_print = _get_smart_print(filename)

    num_rows, num_columns = 4, 4
    num_extra_cells = ch_model.func.num_extra_cells
    assert num_extra_cells > 0

    def to_1b(my_val):
        # return 0b0 if my_val == 0 and 0b1 if my_val != 0 (active)
        assert isinstance(my_val, Term)
        if my_val.width == 1:
            return my_val
        return ~(BvComp(my_val, Constant(0, my_val.width)))

    input_state_list = [m.val for m in ch_model.input_mask]
    output_state_list = [m.val for m in ch_model.output_mask]
    extra_cells_mask = input_state_list[-num_extra_cells:]
    input_state_list = input_state_list[:-num_extra_cells]  # without extra cells

    if verbose:
        smart_print("get_key_schedule_constraints:")

        class AuxAESLikeFunction(aes.AESLikeFunction):
            num_rows, num_columns = 4, 4
            cell_width = 8
            logging_mode = aes.LoggingMode.Debug

        my_matrix = AuxAESLikeFunction.list2matrix(input_state_list)
        fs, ffo = AuxAESLikeFunction.log_activity_matrix_state(my_matrix, return_format=True)
        smart_print("Input masks: \n"+fs.format(*ffo))
        smart_print(f"Extra cell masks: {extra_cells_mask}")
        my_matrix = AuxAESLikeFunction.list2matrix(output_state_list)
        fs, ffo = AuxAESLikeFunction.log_activity_matrix_state(my_matrix, return_format=True)
        smart_print("Output masks:\n"+fs.format(*ffo))

    extra_cells_mask = functools.reduce(BvOr, [m for m in extra_cells_mask])
    initial_constraints.append(BvComp(extra_cells_mask, Constant(0, extra_cells_mask.width)))
    if verbose:
        smart_print("Extra cell constraint:")
        smart_print(" - extra cells == 0 <==>", initial_constraints[-1])

    for index_state, state_list in enumerate([input_state_list, output_state_list]):
        if verbose:
            if index_state == 0:
                smart_print("Input state constraints: (c11 & c12) | c2")
            else:
                smart_print("Output state constraints: (c11 & c12) | c2")

        # - c1

        # state ordered by columns
        last_column_cell_active = [to_1b(cell) for cell in state_list[-4:]]  # last_column_cell_active[i] == 1 if i-th cell in last column is active
        last_column_cell_pair_active = []  # last_column_cell_pair_active[j] == 1 if j-th pair on consecutive column cell is active
        for j in range(num_rows):
            last_column_cell_pair_active.append(last_column_cell_active[j] | last_column_cell_active[(j + 1) % num_rows])
        last_column_cell_pair_active = functools.reduce(Concat, [m for m in last_column_cell_pair_active])
        num_active = PopCount(last_column_cell_pair_active)
        c11 = BvComp(num_active, Constant(2, num_active.width))
        if verbose:
            smart_print(" - c11: 2 == HW( Concat(column_cell_pair_active) = {} ) <==> {}".format(last_column_cell_pair_active, c11))

        # first_columns_cells[i] is the i-th cell (containing all cells in every but last column)
        first_columns_cells = state_list[:-4]
        first_columns_cells = functools.reduce(BvOr, [m for m in first_columns_cells])
        c12 = BvComp(first_columns_cells, Constant(0, first_columns_cells.width))
        if verbose:
            smart_print(" - c12: 0 == Concat(column0,column1,column2) = {} <==> {}".format(first_columns_cells, c12))

        c1 = c11 & c12

        # - c2

        def has_hw_1(x):
            return BvComp(x & (x - 1), Constant(0, x.width))

        row_active = []  # row_active[i] == 1 if i-th row is active
        for row in range(num_rows):
            my_row = [state_list[row + (column * num_rows)] for column in range(num_columns)]
            row_active.append(to_1b(functools.reduce(Concat, my_row)))
        row_active = functools.reduce(Concat, [m for m in row_active])
        c2 = has_hw_1(row_active)
        if verbose:
            smart_print(" - c2: 1 == HW( Concat(row_active) = {} ) <==> {}".format(row_active, c2))

        initial_constraints.append(c1 | c2)

    if verbose:
        smart_print("")

    return initial_constraints


def find_best_key_schedule_characteristic(verbose=False):
    """Find the best trail spanning eight rounds and activating 21 masked S-boxes,
    with total absolute correlation :math:`2^{−63.60}`.

        >>> from cascada.primitives.aes_masked import find_best_key_schedule_characteristic
        >>> found_ch = find_best_key_schedule_characteristic(verbose=False)
        >>> print(found_ch.srepr())  # doctest:+NORMALIZE_WHITESPACE
        Ch(w=63.60, id=00 04 00 00 00 53 00 00 00 24 00 00 00 72 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00,
                    od=00 04 00 00 00 53 00 00 00 24 00 00 00 72 00 00)

    """
    num_rounds = 8
    logging_mode = aes.LoggingMode.Debug if verbose else aes.LoggingMode.Silent
    printing_mode = PrintingMode.Debug if verbose else PrintingMode.Silent
    filename = None

    AESMaskedKeySchedule.logging_mode = logging_mode
    AESMaskedKeySchedule.set_num_rounds(num_rounds)

    input_names = [f"k{i}" for i in range(16)] + [f"r{i}" for i in range(2 * num_rounds)]
    ch_model = ChModel(AESMaskedKeySchedule, LinearMask, input_names, "x")

    initial_constraints = get_key_schedule_constraints(ch_model, verbose=verbose, filename=filename)

    assert_type = ChModelAssertType.ValidityAndWeight
    ch_finder = ChFinder(ch_model, assert_type, "btor", solver_seed=0,
                         initial_constraints=initial_constraints,
                         printing_mode=printing_mode, filename=filename)

    ch_found = next(ch_finder.find_next_ch_increasing_weight(initial_weight=1))

    if verbose:
        smart_print = _get_smart_print(filename)
        smart_print("\nFound trail:", ch_found.srepr(), "\n")
        smart_print("\n".join(ch_found.get_formatted_logged_msgs()))

    return ch_found


class AESMaskedEncryption(aes.AESLikeFunction):
    """Encryption of masked AES-128."""
    num_rounds = 10
    num_rows, num_columns = 4, 5
    cell_width = 8

    mix_columns_bit_matrix = aes.MixColumnsBitMatrix

    ignore_first_subcell = True
    ignore_last_subcell = True
    ignore_first_last_sbox_weights = True

    name_sub_cells = "SubBytes"
    logging_mode = aes.LoggingMode.Silent

    @classmethod
    def sub_cells(cls, my_matrix_state, my_sbox):
        new_matrix_state = [row[:] for row in my_matrix_state]

        for row in range(cls.num_rows):
            a, b, c, d, e = my_matrix_state[row]

            if row == 0:
                xor_values = [b, c, d, e, a]
            elif row == 1:
                xor_values = [c, d, e, a, b]
            elif row == 2:
                xor_values = [d, e, a, b, c]
            elif row == 3:
                xor_values = [e, a, b, c, d]
            else:
                raise ValueError("invalid row")

            sbox_inputs = [x for x in [a, b, c, d]]
            xor_values = [BvIdentity(x) for x in xor_values]  # new copies with != names

            sbox_outputs = [my_sbox(x) for x in sbox_inputs]

            # changing of the guards
            a_ = sbox_outputs[0] ^ xor_values[0]
            b_ = sbox_outputs[1] ^ xor_values[1]
            c_ = sbox_outputs[2] ^ xor_values[2]
            d_ = sbox_outputs[3] ^ xor_values[3]
            e_ = BvIdentity(xor_values[4])  # new copy with != name

            new_matrix_state[row] = [a_, b_, c_, d_, e_]

            if cls.logging_mode == aes.LoggingMode.Debug:
                # ensuring the correct values are logged (i.e., not intermediate branches) by
                #  - using different names for sbox_inputs and xor_values
                #  - using a new name for each output cell
                name_sbox = my_sbox.__name__
                # cls.log_msg(" - row: [{},{},{},{},{}]\n - sbox_inputs: [{},{},{},{}]\n - xor_values: [{},{},{},{}]",
                #             my_matrix_state[row] + sbox_inputs + xor_values)
                cls.log_msg(f" {getattr(cls, 'name_sub_cells', 'SubCells')} row {row}:")
                for i in range(len(sbox_outputs)):
                    cls.log_msg(f" - {name_sbox}({{}}) -> {{}}   XOR   {{}}) -> {{}}",
                                [sbox_inputs[i], sbox_outputs[i], xor_values[i], new_matrix_state[row][i]])

        if cls.logging_mode not in [aes.LoggingMode.Silent, aes.LoggingMode.RoundOutputs]:
            cls.log_msg(f"output of {getattr(cls, 'name_sub_cells', 'SubCells')}:")
            cls.log_activity_matrix_state(new_matrix_state)

        return new_matrix_state

    @classmethod
    def shift_rows(cls, my_matrix_state):
        new_matrix_state = [row[:] for row in my_matrix_state]

        for r in range(4):
            offset = r  # shift offset (0 for r=0, 1 for r=1, ...)
            for c in range(4):
                new_matrix_state[r][c] = my_matrix_state[r][(c + offset) % 4]

        if cls.logging_mode not in [aes.LoggingMode.Silent, aes.LoggingMode.RoundOutputs]:
            cls.log_msg(f"output of {getattr(cls, 'name_shift_rows', 'ShiftRows')}:")
            cls.log_activity_matrix_state(new_matrix_state)

        return new_matrix_state

    @classmethod
    def eval(cls, *plaintext):
        matrix_state = cls.list2matrix(plaintext)

        if cls.logging_mode != aes.LoggingMode.Silent:
            cls.log_msg("\nplaintext:")
            cls.log_activity_matrix_state(matrix_state)

        ign = cls.ignore_first_last_sbox_weights

        for r in range(cls.num_rounds):
            if r == 0:
                if cls.ignore_first_subcell:
                    #
                    matrix_state = cls.sub_cells(matrix_state, my_sbox=SboxB if not ign else SboxZeroWeight)
                else:
                    matrix_state = cls.sub_cells(matrix_state, my_sbox=SboxA if not ign else SboxZeroWeight)
                    matrix_state = cls.sub_cells(matrix_state, my_sbox=SboxB)
                matrix_state = cls.shift_rows(matrix_state)
                matrix_state = cls.mix_columns(matrix_state)
            elif r < cls.num_rounds - 1:
                matrix_state = cls.sub_cells(matrix_state, my_sbox=SboxA)
                matrix_state = cls.sub_cells(matrix_state, my_sbox=SboxB)
                matrix_state = cls.shift_rows(matrix_state)
                matrix_state = cls.mix_columns(matrix_state)
            elif r == cls.num_rounds - 1:
                if cls.ignore_last_subcell:
                    matrix_state = cls.sub_cells(matrix_state, my_sbox=SboxA if not ign else SboxZeroWeight)
                    #
                else:
                    matrix_state = cls.sub_cells(matrix_state, my_sbox=SboxA)
                    matrix_state = cls.sub_cells(matrix_state, my_sbox=SboxB if not ign else SboxZeroWeight)

            if cls.logging_mode != aes.LoggingMode.Silent:
                cls.log_msg(f"output of round {r+1}:")  # 1 <= {r+1} <= {cls.num_rounds}
                cls.log_activity_matrix_state(matrix_state)
                cls.log_msg("", [])  # new line

            cls.add_round_outputs(*cls.matrix2list(matrix_state))

        return tuple(cls.matrix2list(matrix_state))


def get_encryption_constraints(ch_model, verbose=False, filename=None):
    """Get the initial constraint ensuring exactly 1 cell active in the input and 1 cell active in the output."""
    initial_constraints = []
    smart_print = _get_smart_print(filename)

    if verbose:
        smart_print("get_encryption_constraints:")
        my_matrix = ch_model.func.list2matrix([m.val for m in ch_model.input_mask])
        fs, ffo = ch_model.func.log_activity_matrix_state(my_matrix, return_format=True)
        smart_print("Input masks: \n"+fs.format(*ffo))
        my_matrix = ch_model.func.list2matrix([m.val for m in ch_model.output_mask])
        fs, ffo = ch_model.func.log_activity_matrix_state(my_matrix, return_format=True)
        smart_print("Output masks:\n"+fs.format(*ffo))

    def has_hw_1(x):
        return BvComp(x & (x - 1), Constant(0, x.width))

    concat_input_mask = functools.reduce(Concat, [m.val for m in ch_model.input_mask])
    initial_constraints.append(has_hw_1(concat_input_mask))

    concat_output_mask = functools.reduce(Concat, [m.val for m in ch_model.output_mask])
    initial_constraints.append(has_hw_1(concat_output_mask))

    if verbose:
        smart_print(" - 1 == HW( {} ) <==> {}".format(concat_input_mask, initial_constraints[0]))
        smart_print(" - 1 == HW( {} ) <==> {}".format(concat_output_mask, initial_constraints[1]))

    return initial_constraints


def find_best_encryption_characteristic(verbose=False):
    """Find the best trail spanning 3 rounds and activating 21 masked S-boxes,
    with total absolute correlation :math:`2^{-51.60}`.

        >>> from cascada.primitives.aes_masked import find_best_encryption_characteristic
        >>> found_ch = find_best_encryption_characteristic(verbose=False)
        >>> print(found_ch.srepr())  # doctest:+NORMALIZE_WHITESPACE
        Ch(w=52.20, id=00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00,
                    od=00 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00 00 00 00 00)

    """
    num_rounds = 3
    logging_mode = aes.LoggingMode.Debug if verbose else aes.LoggingMode.Silent
    printing_mode = PrintingMode.Debug if verbose else PrintingMode.Silent
    filename = None

    AESMaskedEncryption.logging_mode = logging_mode
    AESMaskedEncryption.set_num_rounds(num_rounds)

    input_names = [f"p{i}" for i in range(4*5)]
    ch_model = ChModel(AESMaskedEncryption, LinearMask, input_names, "x")

    initial_constraints = get_encryption_constraints(ch_model, verbose=verbose, filename=filename)

    assert_type = ChModelAssertType.ValidityAndWeight
    ch_finder = ChFinder(ch_model, assert_type, "btor", solver_seed=0,
                         initial_constraints=initial_constraints,
                         printing_mode=printing_mode, filename=filename)

    ch_found = next(ch_finder.find_next_ch_increasing_weight(initial_weight=1))

    if verbose:
        smart_print = _get_smart_print(filename)
        smart_print("\nFound trail:", ch_found.srepr(), "\n")
        smart_print("\n".join(ch_found.get_formatted_logged_msgs()))

    return ch_found
