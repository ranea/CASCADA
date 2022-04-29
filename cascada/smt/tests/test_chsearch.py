"""Tests for the chsearch module."""
import collections
import datetime
import decimal
import doctest
import itertools
import unittest
import warnings

from hypothesis import given, example, settings, assume
from hypothesis.strategies import integers

from cascada.bitvector import core
from cascada.bitvector import ssa as cascada_ssa
from cascada.primitives import blockcipher

from cascada.algebraic.tests.test_ch import TestBitChModelCharacteristic, TestWordChModelCharacteristic
from cascada.differential.tests.test_ch import TestXorChModelCharacteristic, TestRXChModelCharacteristic
from cascada.linear.tests.test_ch import TestLinearChModelCharacteristic

from cascada.smt.chsearch import (
    ChFinder, CipherChFinder, round_based_ch_search, round_based_cipher_ch_search,
    ChModelAssertType, PrintingMode, INCREMENT_NUM_ROUNDS
)

from cascada.smt import chsearch


SKIP_LONG_TESTS = True
MAX_EXAMPLES = 100  # 1000s
MAX_CH = 64

VERBOSE = False
WRITE_MSG = False


class _FakeFileForPrint(object):
    def __init__(self):
        self.list_msgs = []

    def write(self, string):
        self.list_msgs.append(string)


class TestRandomChSearchGeneric(unittest.TestCase):
    """Base class for testing search for characteristic."""
    USE_C_CODE = False
    VERBOSE = VERBOSE
    WRITE_MSG = WRITE_MSG

    def _invalid_assert_type(self, assert_type, other_assert_type=None):
        return False

    def _obtain_extra_operations(self, width, seed):
        return None

    def _test_random_chsearch(self, width, num_inputs, num_outputs, num_assignments,
                              seed, func_type_index, num_rounds):
        # ignore <frozen importlib._bootstrap>:914: ImportWarning: _SixMetaPathImporter.find_spec()
        # not found; falling back to find_module()
        warnings.filterwarnings("ignore", category=ImportWarning)
        # warnings.filterwarnings("ignore")

        VERBOSE = self.__class__.VERBOSE
        WRITE_MSG = self.__class__.WRITE_MSG

        prop_type = self.__class__.prop_type
        prop_prefix = self.__class__.prop_prefix
        operation_set_index = self.__class__.operation_set_index

        ChModel = self.__class__.ChModel
        EncryptionChModel = self.__class__.EncryptionChModel
        CipherChModel = self.__class__.CipherChModel

        self.assertIn(operation_set_index, [0, 1, 2, 3])

        assume(num_inputs + num_assignments >= num_outputs)
        # avoid errors due to simplification between rounds
        assume(not (num_rounds >= 2 and num_inputs < num_outputs))

        self.__class__.PRNG.seed(seed)

        # ----------------------------------
        # 1 - Generating the random function
        # ----------------------------------

        if num_rounds == 0:
            num_rounds = None

        def get_time():
            now = datetime.datetime.now()
            return "{}-{}:{}".format(now.day, now.hour, now.minute)

        # timestamp messages ignored WRITE_MSG
        msg = f"{get_time()} | test_random_chsearch({width}, {num_inputs}, {num_outputs}, " \
              f"{num_assignments}, {seed}, {func_type_index}, {num_rounds})"
        if VERBOSE:
            print(msg)

        extra_operations = self._obtain_extra_operations(width, seed)

        with warnings.catch_warnings(record=True) as caught_warnings:
            # ignore redundant assignments (get_random_cipher call to_ssa)
            # ignore very slow evaluation warning
            # warnings.filterwarnings("ignore", category=UserWarning)
            if func_type_index == 0:
                MyFoo = cascada_ssa.get_random_bvfunction(
                    width, num_inputs, num_outputs, num_assignments, seed, external_variable_prefix="k",
                    operation_set_index=operation_set_index, num_rounds=num_rounds,
                    extra_operations=extra_operations)
                input_names = [f"{prop_prefix}p" + str(i) for i in range(num_inputs)]
                input_vars = [core.Variable(input_names[i], width) for i in range(num_inputs)]
                output_myfoo = MyFoo(*input_vars, symbolic_inputs=True, simplify=False)
                if WRITE_MSG:
                    msg += f"\nMyFoo({input_vars}) = {output_myfoo}"
                    msg += f"\nMyFoo log: \n\t" + "\n\t".join(MyFoo.get_formatted_logged_msgs())
            elif func_type_index in [1, 2]:
                key_num_inputs = self.__class__.PRNG.randint(1, num_inputs)
                key_num_assignments = self.__class__.PRNG.randint(1, num_assignments)
                enc_num_inputs = num_inputs
                enc_num_outputs = num_outputs
                enc_num_assignments = num_assignments
                del num_inputs, num_outputs, num_assignments
                MyCipher = blockcipher.get_random_cipher(
                    width, key_num_inputs, key_num_assignments,
                    enc_num_inputs, enc_num_outputs, enc_num_assignments, seed,
                    external_variable_prefix=f"{prop_prefix}k",
                    operation_set_index=operation_set_index, num_rounds=num_rounds,
                    extra_operations=extra_operations)
                pt_vars = [core.Variable(f"{prop_prefix}p" + str(i), width) for i in range(enc_num_inputs)]
                mk_vars = [core.Variable(f"{prop_prefix}mk" + str(i), width) for i in range(key_num_inputs)]
                old_round_keys = MyCipher.encryption.round_keys
                MyCipher.encryption.round_keys = [core.Variable(f"k{i}", w)
                                                  for i, w in enumerate(MyCipher.key_schedule.output_widths)]
                if WRITE_MSG:
                    msg += f"\nMyCipher({pt_vars, mk_vars}) = " \
                           f"{MyCipher(pt_vars, mk_vars, symbolic_inputs=True, simplify=False)}"
                    msg += f"\nMyCipher.key_schedule({mk_vars}) = " \
                           f"{MyCipher.key_schedule(*mk_vars, symbolic_inputs=True, simplify=False)}"
                    msg += f"\nMyCipher.round_keys = {MyCipher.encryption.round_keys}"
                    msg += f"\nMyCipher.encryption({pt_vars}) = " \
                           f"{MyCipher.encryption(*pt_vars, symbolic_inputs=True, simplify=False)}"
                    msg += f"\nMyCipher.key_schedule log: \n\t" + \
                           "\n\t".join(MyCipher.key_schedule.get_formatted_logged_msgs())
                    msg += f"\nMyCipher.encryption log: \n\t" + \
                           "\n\t".join(MyCipher.encryption.get_formatted_logged_msgs())
            if WRITE_MSG:
                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

        msg += f"\n# {get_time()} | generated the random function/cipher"

        # ----------------------------
        # 2 - Generating the ch. model
        # ----------------------------

        def __is_valid_ssa(my_foo):
            if func_type_index == 0:
                ssa = my_foo.to_ssa(input_names=input_names, id_prefix="a")
                aux_is_valid = self._valid_ssa(ssa)
            elif func_type_index == 1:
                ssa = my_foo.to_ssa(input_names=[str(x) for x in pt_vars], id_prefix="a")
                aux_is_valid = self._valid_ssa(ssa)
            elif func_type_index == 2:
                ssa1 = my_foo.encryption.to_ssa(input_names=[str(x) for x in pt_vars], id_prefix="a")
                ssa2 = my_foo.key_schedule.to_ssa(input_names=[str(x) for x in mk_vars], id_prefix="a")
                aux_is_valid = self._valid_ssa(ssa1) and self._valid_ssa(ssa2)
            return aux_is_valid

        def is_valid_ssa(my_foo):
            if num_rounds is not None:
                for nr in range(1, num_rounds + 1):
                    my_foo.set_num_rounds(nr)
                    try:
                        stored_is_valid = __is_valid_ssa(my_foo)
                    except ValueError as e:
                        if str(e).startswith("round keys") or str(e).startswith("found external variable"):
                            return False
                        else:
                            raise e
                    else:
                        if stored_is_valid is False:
                            return False
                my_foo.set_num_rounds(num_rounds)
                return True
            else:
                return __is_valid_ssa(my_foo)

        with warnings.catch_warnings(record=True) as caught_warnings:
            # ignore redundant assignments
            # warnings.filterwarnings("ignore", category=UserWarning)
            if func_type_index == 0:
                if not is_valid_ssa(MyFoo):
                    if WRITE_MSG:
                        msg += f"\naborting due to invalid ssa"
                    return
                e_v2d = self._get_external_var2prop(MyFoo.to_ssa(input_names=input_names, id_prefix="a"))
                ch_model = ChModel._prop_new(MyFoo, prop_type, input_names, prefix=f"{prop_prefix}x",
                                             external_var2prop=e_v2d, op_model_class2options=None)
            elif func_type_index == 1:
                if not is_valid_ssa(MyCipher.encryption):
                    if WRITE_MSG:
                        msg += f"\naborting due to invalid ssa"
                    return
                ch_model = EncryptionChModel(MyCipher, prop_type)
            elif func_type_index == 2:
                if not is_valid_ssa(MyCipher):
                    if WRITE_MSG:
                        msg += f"\naborting due to invalid ssa"
                    return
                ch_model = CipherChModel(MyCipher, prop_type)
            if WRITE_MSG:
                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

        if WRITE_MSG:
            msg += f"\nChModel: {ch_model}"
            msg += f"\nChModel log: \n\t" + "\n\t".join(ch_model.get_formatted_logged_msgs())

        msg += f"\n# {get_time()} | generated the characteristic model"

        if type(ch_model) in [ChModel, EncryptionChModel]:
            list_ch_models = [ch_model]
        elif type(ch_model) == CipherChModel:
            list_ch_models = [ch_model.ks_ch_model, ch_model.enc_ch_model]
        else:
            raise ValueError(f"invalid type(ch_model) = {type(ch_model)}")

        for aux_ch_model in list_ch_models:
            aux_max_weight = aux_ch_model.max_weight(truncate=False) / decimal.Decimal(2 ** aux_ch_model.num_frac_bits())
            if 0 < aux_max_weight < 1:
                if func_type_index != 0:
                    MyCipher.encryption.round_keys = old_round_keys  # get_random_cipher outputs are stored in the cached
                if WRITE_MSG:
                    msg += f"\naborting due to 0 < {aux_max_weight} < 1"
                return

        # ---------------------------------
        # 3 - Searching for characteristics
        # ---------------------------------

        printing_mode = PrintingMode.Silent if not self.WRITE_MSG else PrintingMode.WeightsAndVrepr

        CFOptions = collections.namedtuple(
            'CFOptions', ['assert_type'])
        list_cf_opts = [
            CFOptions(ChModelAssertType.Validity),
            CFOptions(ChModelAssertType.ProbabilityOne),
            CFOptions(ChModelAssertType.ValidityAndWeight),
        ]

        msg += "\n#"

        for aux_ch_model in list_ch_models:
            if WRITE_MSG:
                msg += f"\n\t# {aux_ch_model.func.__name__}"

            for cf_options in list_cf_opts:
                if self._invalid_assert_type(cf_options.assert_type):
                    continue

                if aux_ch_model.ssa._input_vars_not_used:
                    exclude_zero_input_prop = False
                else:
                    if cf_options.assert_type == ChModelAssertType.ValidityAndWeight:
                        exclude_zero_input_prop = True
                    else:
                        exclude_zero_input_prop = bool(self.__class__.PRNG.randint(0, 1))
                random_prop = self.__class__.PRNG.choice(list(aux_ch_model.assign_outprop2op_model.items()))
                random_prop = list(random_prop[1].validity_constraint(random_prop[0]).atoms(core.Variable))
                if random_prop and "_tmp" not in str(random_prop):  # avoid implicit vars from OpModels
                    random_prop = self.__class__.PRNG.choice(random_prop)
                    prop_width = random_prop.width
                    random_ct = core.Constant(self.__class__.PRNG.randint(0, 2 ** prop_width - 1), prop_width)
                    var_prop2ct_prop = {random_prop: prop_type(random_ct)}
                else:
                    var_prop2ct_prop = {}

                if WRITE_MSG:
                    msg += f"\n> cf_options: {cf_options}"
                    msg += f"\nexclude, var_prop2ct_prop: {exclude_zero_input_prop}, {var_prop2ct_prop}"

                with warnings.catch_warnings(record=True) as caught_warnings:
                    # avoid warnings found assertion 0 == 1 in chmodel_asserts
                    my_fake_file = _FakeFileForPrint()
                    try:
                        ch_finder = ChFinder(
                            aux_ch_model, cf_options.assert_type, solver_name="btor",
                            exclude_zero_input_prop=exclude_zero_input_prop, var_prop2ct_prop=var_prop2ct_prop,
                            printing_mode=printing_mode, filename=my_fake_file, solver_seed=self.__class__.PRNG.getrandbits(32)
                        )
                    except ValueError as e:
                        if WRITE_MSG:
                            msg += f"\nException raised in ChFinder with {cf_options}"
                        if str(e).startswith("exclude_zero_input_prop ") or str(e).startswith("var "):
                            continue
                        else:
                            if VERBOSE:
                                print(msg)
                            raise e
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                # note that by default, the following tests are done in chsearch.py:
                #  - if ValidityAndWeight, check SMT weight is correct w.r.t last_ch_found.ch_weight
                #  - if Validity, check last_ch_found is valid
                #  - if ProbabilityOne, check last_ch_found has pr.1
                #  - if exclude or var2prop, check that is the case in last_ch_found
                # only test done here is checking that the final ch of find_next_ch_increasing_weight is optimal

                with warnings.catch_warnings(record=True) as caught_warnings:
                    # find ch create Characteristics with free properties which throws warnings
                    all_ch_found = set()
                    if cf_options.assert_type in [ChModelAssertType.Validity, ChModelAssertType.ProbabilityOne]:
                        iterator = ch_finder.find_next_ch()
                    else:
                        weight_set = set()
                        iterator = ch_finder.find_next_ch_increasing_weight(0, yield_weight=True)

                    for i, ch_found in enumerate(iterator):
                        if WRITE_MSG:
                            msg += f"\ncharacteristic found: {ch_found}"
                            if my_fake_file.list_msgs:
                                msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                                my_fake_file.list_msgs = []

                        if cf_options.assert_type == ChModelAssertType.Validity and i == 1:  # search only 2 valid
                            break
                        if cf_options.assert_type == ChModelAssertType.ValidityAndWeight:
                            bv_weight, ch_found = ch_found
                            weight_set.add(ch_found.ch_weight)
                            if bv_weight is None:
                                self.assertTrue(ch_found in all_ch_found, msg=msg)
                                self.assertEqual(ch_found.ch_weight, min(weight_set), msg=msg)
                                break

                        self.assertNotIn(ch_found, all_ch_found, msg=msg)
                        all_ch_found.add(ch_found)

                        if i == MAX_CH - 1:
                            break
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                if num_rounds is not None:
                    if type(aux_ch_model) == EncryptionChModel:
                        func = MyCipher
                    else:
                        func = aux_ch_model.func
                    extra_ch_finder_args = {
                        "exclude_zero_input_prop": exclude_zero_input_prop,
                        # "var_prop2ct_prop": var_prop2ct_prop,  # for less rounds, v2c might contain invalid vars
                        "printing_mode": printing_mode,
                        "filename": my_fake_file,
                        "solver_seed": self.__class__.PRNG.getrandbits(32)
                    }

                    try:
                        round_ch_iterator = round_based_ch_search(
                            func, 1, num_rounds, prop_type, cf_options.assert_type, "btor",
                            extra_chfinder_args=extra_ch_finder_args)
                    except Exception as e:
                        if WRITE_MSG:
                            msg += f"\nException raised in round_based_ch_search with {cf_options}"
                        if VERBOSE:
                            print(msg)
                        raise e

                    prev_num_rounds_found = 0
                    while True:
                        try:
                            with warnings.catch_warnings(record=True) as caught_warnings:
                                # find ch create Characteristics with free properties which throws warnings
                                num_rounds_found, ch_found = next(round_ch_iterator)
                                if WRITE_MSG:
                                    msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)
                        except StopIteration:
                            break
                        except ValueError as e:
                            if WRITE_MSG and my_fake_file.list_msgs:
                                msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                                my_fake_file.list_msgs = []
                            if WRITE_MSG:
                                msg += f"\nException raised in next(round_ch_iterator)"
                            # changing the number of rounds might introduce redundant assignments with external variables
                            if str(e).startswith("round keys") or str(e).startswith("found external variable") or \
                                    str(e).startswith("exclude_zero_input_prop ") or \
                                    str(e).startswith("weight_assertions(..., truncate=True) cannot truncate characteristic weight"):
                                continue
                            else:
                                if VERBOSE:
                                    print(msg)
                                raise e
                        if WRITE_MSG:
                            msg += f"\ncharacteristic found for {num_rounds_found} rounds: {ch_found}"
                            if my_fake_file.list_msgs:
                                msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                                my_fake_file.list_msgs = []
                        self.assertIn(num_rounds_found, range(1, num_rounds + 1), msg=msg)
                        self.assertLess(prev_num_rounds_found, num_rounds_found, msg=msg)
                        prev_num_rounds_found = num_rounds_found
                        round_ch_iterator.send(INCREMENT_NUM_ROUNDS)
                    func.set_num_rounds(num_rounds)
                # end of round_based_ch_search

        if type(ch_model) == CipherChModel:
            CFOptions = collections.namedtuple('CFOptions', ['ks_assert_type', 'enc_assert_type'])
            list_cf_opts = []
            for ks_assert_type, enc_assert_type in itertools.product(
                [ChModelAssertType.Validity, ChModelAssertType.ValidityAndWeight, ChModelAssertType.ProbabilityOne],
                [ChModelAssertType.Validity, ChModelAssertType.ValidityAndWeight, ChModelAssertType.ProbabilityOne]
            ):
                list_cf_opts.append(CFOptions(ks_assert_type, enc_assert_type))

            msg += "\n#"

            for cf_options in list_cf_opts:
                if self._invalid_assert_type(cf_options.ks_assert_type, cf_options.enc_assert_type):
                    continue
                if ch_model.ks_ch_model.ssa._input_vars_not_used:
                    exclude_ks = False
                else:
                    exclude_ks = bool(self.__class__.PRNG.randint(0, 1))
                    if cf_options.ks_assert_type == ChModelAssertType.ValidityAndWeight:
                        exclude_ks = True
                if ch_model.enc_ch_model.ssa._input_vars_not_used:
                    exclude_enc = False
                else:
                    exclude_enc = bool(self.__class__.PRNG.randint(0, 1))
                    if cf_options.enc_assert_type == ChModelAssertType.ValidityAndWeight:
                        exclude_enc = True
                random_ch_model = self.__class__.PRNG.choice([ch_model.ks_ch_model, ch_model.enc_ch_model])
                random_prop = self.__class__.PRNG.choice(list(random_ch_model.assign_outprop2op_model.items()))
                random_prop = list(random_prop[1].validity_constraint(random_prop[0]).atoms(core.Variable))
                if random_prop and "_tmp" not in str(random_prop):  # avoid implicit vars from OpModels
                    random_prop = self.__class__.PRNG.choice(random_prop)
                    prop_width = random_prop.width
                    random_ct = core.Constant(self.__class__.PRNG.randint(0, 2 ** prop_width - 1), prop_width)
                    var_prop2ct_prop = {random_prop: prop_type(random_ct)}
                else:
                    var_prop2ct_prop = {}

                if WRITE_MSG:
                    msg += f"\n> cf_options: {cf_options}"
                    msg += f"\nks-exclude, enc-exclude, var_prop2ct_prop: {exclude_ks}, {exclude_enc}, {var_prop2ct_prop}"

                with warnings.catch_warnings(record=True) as caught_warnings:
                    # avoid warnings found assertion 0 == 1 in chmodel_asserts
                    my_fake_file = _FakeFileForPrint()
                    try:
                        ch_finder = CipherChFinder(
                            ch_model, cf_options.ks_assert_type, cf_options.enc_assert_type, solver_name="btor",
                            ks_exclude_zero_input_prop=exclude_ks, enc_exclude_zero_input_prop=exclude_enc,
                            var_prop2ct_prop=var_prop2ct_prop, printing_mode=printing_mode, filename=my_fake_file,
                            solver_seed=self.__class__.PRNG.getrandbits(32)
                        )
                    except ValueError as e:
                        if WRITE_MSG:
                            msg += f"\nException raised in CipherChFinder with {cf_options}"
                        if str(e).startswith("exclude_zero_input_prop ") or str(e).startswith("var "):
                            continue
                        else:
                            if VERBOSE:
                                print(msg)
                            raise e
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                with warnings.catch_warnings(record=True) as caught_warnings:
                    # find ch create Characteristics with free properties which throws warnings
                    all_ch_found = set()
                    if ch_finder.assert_type in [ChModelAssertType.Validity, ChModelAssertType.ProbabilityOne]:
                        iterator = ch_finder.find_next_ch()
                    else:
                        weight_set = set()
                        iterator = ch_finder.find_next_ch_increasing_weight(0, yield_weight=True)

                    for i, ch_found in enumerate(iterator):
                        if WRITE_MSG:
                            msg += f"\ncharacteristic found: {ch_found}"
                            if my_fake_file.list_msgs:
                                msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                                my_fake_file.list_msgs = []

                        if ch_finder.assert_type == ChModelAssertType.Validity and i == 1:  # search only 2 valid
                            break
                        if ch_finder.assert_type == ChModelAssertType.ValidityAndWeight:
                            bv_weight, ch_found = ch_found
                            ch_found_ch_weight = ch_finder._get_ch_weight(ch_found, add_aux_probability=False)
                            weight_set.add(ch_found_ch_weight)
                            if bv_weight is None:
                                self.assertTrue(ch_found in all_ch_found, msg=msg)
                                self.assertEqual(ch_found_ch_weight, min(weight_set), msg=msg)
                                break

                        self.assertNotIn(ch_found, all_ch_found, msg=msg)
                        all_ch_found.add(ch_found)

                        if i == MAX_CH - 1:
                            break
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                if num_rounds is not None:
                    extra_cipherchfinder_args = {
                        "ks_exclude_zero_input_prop": exclude_ks,
                        "enc_exclude_zero_input_prop": exclude_enc,
                        "printing_mode": printing_mode,
                        "filename": my_fake_file,
                        "solver_seed": self.__class__.PRNG.getrandbits(32)
                    }

                    try:
                        round_ch_iterator = round_based_cipher_ch_search(
                            MyCipher, 1, num_rounds, prop_type,
                            cf_options.ks_assert_type, cf_options.enc_assert_type, "btor",
                            extra_cipherchfinder_args=extra_cipherchfinder_args)
                    except Exception as e:
                        if WRITE_MSG:
                            msg += f"\nException raised in round_based_cipher_ch_search with {cf_options}"
                        if VERBOSE:
                            print(msg)
                        raise e

                    prev_num_rounds_found = 0
                    while True:
                        try:
                            with warnings.catch_warnings(record=True) as caught_warnings:
                            # find ch create Characteristics with free properties which throws warnings
                                num_rounds_found, ch_found = next(round_ch_iterator)
                                if WRITE_MSG:
                                    msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)
                        except StopIteration:
                            break
                        except ValueError as e:
                            if WRITE_MSG and my_fake_file.list_msgs:
                                msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                                my_fake_file.list_msgs = []
                            if WRITE_MSG:
                                msg += f"\nException raised in next(round_ch_iterator) with {cf_options}"
                            if str(e).startswith("round keys") or str(e).startswith("found external variable") or \
                                    str(e).startswith("exclude_zero_input_prop ") or \
                                    str(e).startswith("weight_assertions(..., truncate=True) cannot truncate characteristic weight"):
                                continue
                            else:
                                if VERBOSE:
                                    print(msg)
                                raise e
                        if WRITE_MSG:
                            msg += f"\ncharacteristic found for {num_rounds_found} rounds: {ch_found}"
                            if my_fake_file.list_msgs:
                                msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                                my_fake_file.list_msgs = []
                        self.assertIn(num_rounds_found, range(1, num_rounds + 1), msg=msg)
                        self.assertLess(prev_num_rounds_found, num_rounds_found, msg=msg)
                        prev_num_rounds_found = num_rounds_found
                        round_ch_iterator.send(INCREMENT_NUM_ROUNDS)
                    MyCipher.set_num_rounds(num_rounds)
                # end of round_based_cipher_ch_search

        msg += f"\n# {get_time()} | searched the characteristics"

        # get_random_bvfunc/cipher are stored in the cached
        if func_type_index != 0:
            MyCipher.encryption.round_keys = old_round_keys

        if VERBOSE:
            print(msg)

    def test_random_ch(self, *args, **kwargs):  # disable
        return


class TestBitChSearch(TestRandomChSearchGeneric, TestBitChModelCharacteristic):
    """Test for the ChSearch and CipherChSearch classes."""

    _obtain_extra_operations = TestBitChModelCharacteristic._obtain_extra_operations

    def _invalid_assert_type(self, assert_type, other_assert_type=None):
        if other_assert_type is None:
            other_assert_type = assert_type
        return ChModelAssertType.ValidityAndWeight in [assert_type, other_assert_type]

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping BitValue test_random_chsearch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=MAX_EXAMPLES)
    def test_random_chsearch(self, width, num_inputs, num_outputs, num_assignments,
                             seed, func_type_index, num_rounds):
        return super()._test_random_chsearch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


class TestWordChSearch(TestRandomChSearchGeneric, TestWordChModelCharacteristic):
    """Test for the ChSearch and CipherChSearch classes."""

    _obtain_extra_operations = TestWordChModelCharacteristic._obtain_extra_operations

    def _invalid_assert_type(self, assert_type, other_assert_type=None):
        if other_assert_type is None:
            other_assert_type = assert_type
        return ChModelAssertType.ValidityAndWeight in [assert_type, other_assert_type]

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping WordValue test_random_chsearch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=MAX_EXAMPLES)
    def test_random_chsearch(self, width, num_inputs, num_outputs, num_assignments,
                             seed, func_type_index, num_rounds):
        return super()._test_random_chsearch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


class TestXorChSearch(TestRandomChSearchGeneric, TestXorChModelCharacteristic):
    """Test for the ChSearch and CipherChSearch classes."""

    _obtain_extra_operations = TestXorChModelCharacteristic._obtain_extra_operations

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping XorDiff test_random_chsearch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=MAX_EXAMPLES)
    def test_random_chsearch(self, width, num_inputs, num_outputs, num_assignments,
                             seed, func_type_index, num_rounds):
        return super()._test_random_chsearch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


class TestRXChSearch(TestRandomChSearchGeneric, TestRXChModelCharacteristic):
    """Test for the ChSearch and CipherChSearch classes."""

    _obtain_extra_operations = TestRXChModelCharacteristic._obtain_extra_operations

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping RXDiff test_random_chsearch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=MAX_EXAMPLES)
    def test_random_chsearch(self, width, num_inputs, num_outputs, num_assignments,
                             seed, func_type_index, num_rounds):
        assume(not func_type_index == 1)  # EncryptionCh doesn't support RXDiff
        return super()._test_random_chsearch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


class TestLinearChSearch(TestRandomChSearchGeneric, TestLinearChModelCharacteristic):
    """Test for the ChSearch and CipherChSearch classes."""

    _obtain_extra_operations = TestLinearChModelCharacteristic._obtain_extra_operations

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping linear test_random_chsearch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=MAX_EXAMPLES)
    def test_random_chsearch(self, width, num_inputs, num_outputs, num_assignments,
                             seed, func_type_index, num_rounds):
        assume(not func_type_index == 2)  # CipherCh don't support LinearMask
        return super()._test_random_chsearch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(chsearch))
    return tests
