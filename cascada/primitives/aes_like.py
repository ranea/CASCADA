"""AES-like functions."""
import enum

from cascada.bitvector.core import Constant, Term
from cascada.bitvector.operation import BvComp, BvNot
from cascada.bitvector.context import Memoization
from cascada.abstractproperty.property import make_partial_propextract

from cascada.bitvector.ssa import RoundBasedFunction


class LoggingMode(enum.Enum):
    """Represent the options available for the information to log.

    Attributes:
        Silent: nothing is logged
        RoundOutputs: the plaintext and the output of each round is logged
        StepOutputs: the plaintext and the output of each step is logged
        Debug: similar as `StepOutputs`, but also logs debugging information

    """
    Silent = enum.auto()
    RoundOutputs = enum.auto()
    StepOutputs = enum.auto()
    Debug = enum.auto()


class AESLikeFunction(RoundBasedFunction):
    """Base class to implemented AES-like functions.

    Subclasses need to set the integer attributes
    ``num_rounds``, ``num_rows``, ``num_columns`` and ``cell_width``.

    Subclasses also need to set ``sbox`` (resp. ``mix_columns_bit_matrix``)
    to use ``sub_cells`` (resp. ``mix_columns``).
    See `AESEncryption` for an example.

    Optionally, subclasses can set the attributes ``logging_mode``
    and ``name*`` to customize the information to log
    (see `LoggingMode`, `BvFunction.get_formatted_logged_msgs` and
    `Characteristic.get_formatted_logged_msgs`).

    By default input/output bit-vector tuples are loaded column-wise.
    """
    # num_rounds
    # num_rows, num_columns
    # cell_width
    # # num_extra_cells

    # sbox, mix_columns_bit_matrix

    # name_add_round_constants, name_sub_cells, name_shift_rows, name_mix_columns
    logging_mode = LoggingMode.Silent

    # - RoundBasedFunction attributes/methods

    @classmethod
    @property
    def input_widths(cls):
        return [cls.cell_width for _ in range(cls.num_rows*cls.num_columns)]  # cls.num_extra_cells

    @classmethod
    @property
    def output_widths(cls):
        return [cls.cell_width for _ in range(cls.num_rows*cls.num_columns)]

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        assert cls.num_rounds is not None
        cls.num_rounds = new_num_rounds

    # - steps of AES-like functions

    @classmethod
    def add_round_constant(cls, my_matrix_state, my_matrix_constant):
        new_matrix_state = [row[:] for row in my_matrix_state]

        for row in range(len(my_matrix_constant)):
            for column in range(len(my_matrix_constant[row])):
                new_matrix_state[row][column] ^= my_matrix_constant[row][column]

        if cls.logging_mode not in [LoggingMode.Silent, LoggingMode.RoundOutputs]:
            cls.log_msg(f"output of {getattr(cls, 'name_add_round_constants', 'AddRoundConstant')}:")
            cls.log_activity_matrix_state(new_matrix_state)

        return new_matrix_state

    @classmethod
    def sub_cells(cls, my_matrix_state):
        new_matrix_state = [row[:] for row in my_matrix_state]

        for row in range(cls.num_rows):
            for column in range(cls.num_columns):
                new_matrix_state[row][column] = cls.sbox(my_matrix_state[row][column])

        if cls.logging_mode not in [LoggingMode.Silent, LoggingMode.RoundOutputs]:
            cls.log_msg(f"output of {getattr(cls, 'name_sub_cells', 'SubCells')}:")
            cls.log_activity_matrix_state(new_matrix_state)

        return new_matrix_state

    @classmethod
    def shift_rows(cls, my_matrix_state):
        new_matrix_state = [row[:] for row in my_matrix_state]

        for r in range(cls.num_rows):
            offset = r  # shift offset (0 for r=0, 1 for r=1, ...)
            for c in range(cls.num_columns):
                new_matrix_state[r][c] = my_matrix_state[r][(c + offset) % cls.num_columns]

        if cls.logging_mode not in [LoggingMode.Silent, LoggingMode.RoundOutputs]:
            cls.log_msg(f"output of {getattr(cls, 'name_shift_rows', 'ShiftRows')}:")
            cls.log_activity_matrix_state(new_matrix_state)

        return new_matrix_state

    @classmethod
    def mix_columns(cls, my_matrix_state):
        new_matrix_state = [row[:] for row in my_matrix_state]

        num_rows = cls.mix_columns_bit_matrix.arity[0]
        assert cls.num_rows >= num_rows

        cw = cls.cell_width
        extract_cell = [make_partial_propextract((r+1)*cw - 1, r*cw) for r in range(num_rows)]

        for index_column in range(cls.num_columns):
            input_column = [my_matrix_state[r][index_column] for r in range(num_rows)]
            output_column = cls.mix_columns_bit_matrix(*input_column)
            for r in range(num_rows):
                new_matrix_state[r][index_column] = extract_cell[r](output_column)

        if cls.logging_mode not in [LoggingMode.Silent, LoggingMode.RoundOutputs]:
            cls.log_msg(f"output of {getattr(cls, 'name_mix_columns', 'MixColumns')}:")
            cls.log_activity_matrix_state(new_matrix_state)

        return new_matrix_state

    # - auxiliary functions

    @classmethod
    def list2matrix(cls, my_list, num_rows=None, num_columns=None, column_order=True):  # , ignore_num_extra_cells=False):
        if num_rows is None:
            num_rows = cls.num_rows
        if num_columns is None:
            num_columns = cls.num_columns
        my_matrix = [[None for _ in range(num_columns)] for _ in range(num_rows)]
        position = 0
        if column_order:
            for column in range(num_columns):  # assuming my_list ordered by columns
                for row in range(num_rows):
                    assert isinstance(my_list[position], Term) and my_list[position].width == cls.cell_width
                    my_matrix[row][column] = my_list[position]
                    position += 1
        else:
            for row in range(num_rows):
                for column in range(num_columns):  # assuming my_list ordered by columns
                    assert isinstance(my_list[position], Term) and my_list[position].width == cls.cell_width
                    my_matrix[row][column] = my_list[position]
                    position += 1
        # if cls.num_extra_cells and ignore_num_extra_cells is not True:
        #     my_matrix.append([None for _ in range(cls.num_extra_cells)])
        #     for cell in range(cls.num_extra_cells):
        #         assert isinstance(my_list[position], Term) and my_list[position].width == cls.cell_width
        #         my_matrix[-1][cell] = my_list[position]
        #         position += 1
        assert position == len(my_list)
        return my_matrix

    @classmethod
    def matrix2list(cls, my_matrix, num_rows=None, num_columns=None, column_order=True):  # , ignore_num_extra_cells=False):
        if num_rows is None:
            num_rows = cls.num_rows
        if num_columns is None:
            num_columns = cls.num_columns
        my_list = []
        if column_order:
            for column in range(num_columns):  # sorting my_list by columns
                for row in range(num_rows):
                    my_list.append(my_matrix[row][column])
        else:
            for row in range(num_rows):
                for column in range(num_columns):
                    my_list.append(my_matrix[row][column])
        # if cls.num_extra_cells and ignore_num_extra_cells is not True:
        #     my_list.extend(my_matrix[-1])
        assert all(isinstance(x, Term) and x.width == cls.cell_width for x in my_list)
        return my_list

    @classmethod
    def log_activity_matrix_state(cls, my_matrix_state, return_format=False):
        def activity(my_bv):
            assert isinstance(my_bv, Term) and my_bv.width == cls.cell_width
            with Memoization(None):
                my_bv = BvNot(BvComp(my_bv, Constant(0, my_bv.width)))
            return my_bv

        # assert len(my_matrix_state) == cls.num_rows + (1 if cls.num_extra_cells else 0)

        format_string = ""
        format_field_objects = []

        for row in range(cls.num_rows):
            if cls.logging_mode == LoggingMode.Debug:
                format_string += "(" + ','.join(["{}"]*cls.num_columns) + ")\t"
                format_field_objects.extend(my_matrix_state[row])
            format_string += "(" + ','.join(["{}"]*cls.num_columns) + ")\n"
            format_field_objects.extend(activity(x) for x in my_matrix_state[row])

        # if cls.num_extra_cells:
        #     assert len(my_matrix_state[-1]) == cls.num_extra_cells
        #     if cls.logging_mode == LoggingMode.Debug:
        #         format_string += "[" + ','.join(["{}"]*cls.num_extra_cells) + "]\t"
        #         format_field_objects.extend(my_matrix_state[-1])
        #     format_string += "[" + ','.join(["{}"]*cls.num_extra_cells) + "]\n"
        #     format_field_objects.extend(activity(x) for x in my_matrix_state[-1])

        format_string = format_string[:-1]  # remove last "\n"

        if return_format:
            return format_string, format_field_objects
        else:
            cls.log_msg(format_string, format_field_objects=format_field_objects)
