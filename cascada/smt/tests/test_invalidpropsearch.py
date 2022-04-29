"""Tests for the invalidpropsearch module."""
import collections
import datetime
import decimal
import doctest
import itertools
import math
import unittest
import warnings

from hypothesis import given, example, settings, assume
from hypothesis.strategies import integers, sampled_from

from pysmt.exceptions import SolverReturnedUnknownResultError

from cascada.bitvector import core
from cascada.bitvector import ssa as cascada_ssa
from cascada.primitives import blockcipher

from cascada.abstractproperty.opmodel import ModelIdentity, WeakModel, BranchNumberModel
from cascada.differential.tests.test_ch import TestXorChModelCharacteristic, TestRXChModelCharacteristic
from cascada.linear.tests.test_ch import TestLinearChModelCharacteristic

from cascada.smt.wrappedchmodel import get_wrapped_chmodel, get_wrapped_cipher_chmodel
from cascada.smt.invalidpropsearch import (
    InvalidPropFinder, InvalidCipherPropFinder, round_based_invalidprop_search, round_based_invalidcipherprop_search,
    ActiveBitMode, PrintingMode, INCREMENT_NUM_ROUNDS, _generate_bitvectors
)

from cascada.smt import invalidpropsearch, wrappedchmodel

from cascada.smt.tests.test_chsearch import _FakeFileForPrint


SKIP_LONG_TESTS = True
USE_C_CODE = True
MAX_EXAMPLES = 200  # 1000s
MAX_UNI_INV_CH = 3 if USE_C_CODE else 32

VERBOSE = False
WRITE_MSG = False


class TestSimpleTests(unittest.TestCase):
    """Simple tests for the InvalidPropFinder class."""

    def test_generate_bitvectors(self):
        for length in range(1, 3 + 1):
            for widths in itertools.product(range(1, 3 + 1), repeat=length):
                for total_nab in range(0, sum(widths) + 1):
                    bvs = list(_generate_bitvectors(widths, total_nab, active_bits_mode=ActiveBitMode.Default))
                    # number of n-length bit-vectors with hamming weight k is (n k)
                    expected_num_bvs = math.comb(sum(widths), total_nab)
                    msg = f"widths={widths}, total_nab={total_nab}, " \
                          f"expected_len={expected_num_bvs}, bvs({len(bvs)})={bvs}"
                    self.assertEqual(len(bvs), expected_num_bvs, msg=msg)


class TestRandomInvalidPropSearchGeneric(unittest.TestCase):
    """Base class for testing search for universally-invalid characteristic."""
    USE_C_CODE = USE_C_CODE
    VERBOSE = VERBOSE
    WRITE_MSG = WRITE_MSG

    def _obtain_extra_operations(self, width, seed):
        return None

    def _is_permutation_ssa(self, my_ssa):
        if self._is_permutation:
            input_widths = [v.width for v in my_ssa.input_vars]
            external_widths = [v.width for v in my_ssa.external_vars]
            output_widths = [v.width for v in my_ssa.output_vars]

            if sum(input_widths) != sum(output_widths):
                self._is_permutation = False
                return

            # max complexity 2^16
            if sum(input_widths) + sum(external_widths) > 16:
                self._is_permutation = False
                return

            # input_samples == whole input space
            def get_next_input_pair():
                for input_sample in itertools.product(*[range(2 ** w) for w in input_widths]):
                    yield [core.Constant(p_i, w) for p_i, w in zip(input_sample, input_widths)]

            # external_samples == whole external space
            if len(external_widths) == 0:
                external_space = [None]
            else:
                external_space = itertools.product(*[range(2 ** w) for w in external_widths])

            for external_sample in external_space:
                if external_sample is not None:
                    v2c = collections.OrderedDict()
                    for i_v, v in enumerate(my_ssa.external_vars):
                       v2c[v] = core.Constant(external_sample[i_v], external_widths[i_v])
                    ssa = my_ssa.copy()
                    for outvar in ssa.assignments:
                        ssa.assignments[outvar] = ssa.assignments[outvar].xreplace(v2c)
                    ssa.external_vars = []
                else:
                    ssa = my_ssa

                stored_outputs = set()
                for pt in get_next_input_pair():
                    ct = ssa.eval(pt)
                    if ct in stored_outputs:  # found collision
                        self._is_permutation = False
                        return
                    else:
                        stored_outputs.add(ct)

    def _valid_ssa(self, my_ssa):
        self._is_permutation_ssa(my_ssa)
        return super()._valid_ssa(my_ssa)

    def _test_random_invalidpropsearch(self, width, num_inputs, num_outputs, num_assignments,
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

        self._is_permutation = True  # modified in _is_valid_ssa

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
            msg += f"\nis_permutation: {self._is_permutation}"
            msg += f"\nChModel: {ch_model}"
            msg += f"\nChModel log: \n\t" + "\n\t".join(ch_model.get_formatted_logged_msgs())

        msg += f"\n# {get_time()} | generated the characteristic model"

        if type(ch_model) in [ChModel, EncryptionChModel]:
            list_ch_models = [ch_model]
        elif type(ch_model) == CipherChModel:
            list_ch_models = [ch_model.ks_ch_model, ch_model.enc_ch_model]
        else:
            raise ValueError(f"invalid type(ch_model) = {type(ch_model)}")

        found_weak_bn_model = False
        for aux_ch_model in list_ch_models:
            aux_max_weight = aux_ch_model.max_weight(truncate=False) / decimal.Decimal(2 ** aux_ch_model.num_frac_bits())
            if 0 < aux_max_weight < 1:
                if func_type_index != 0:
                    MyCipher.encryption.round_keys = old_round_keys  # get_random_cipher outputs are stored in the cached
                if WRITE_MSG:
                    msg += f"\naborting due to 0 < {aux_max_weight} < 1"
                return
            for op_model in aux_ch_model.assign_outprop2op_model.values():
                if isinstance(op_model, (WeakModel, BranchNumberModel)):
                    found_weak_bn_model = True

        # -----------------------------------------------------------------
        # 3 - Searching for uni-inv characteristics of bit-vector functions
        # -----------------------------------------------------------------

        printing_mode = PrintingMode.Silent if not self.WRITE_MSG else PrintingMode.WeightsAndVrepr

        CFOptions = collections.namedtuple(
            'CFOptions', ['input_prop_activebitmode', 'output_prop_activebitmode'])
        list_cf_opts = [
            CFOptions(None, None),  # find_next_invalidprop_quantified_logic
            CFOptions(ActiveBitMode.Default, ActiveBitMode.SingleBit),
            CFOptions(ActiveBitMode.MSBit, ActiveBitMode.Zero),
        ]

        with warnings.catch_warnings(record=True) as caught_warnings:
            # avoid warnings wrapping a characteristic model with 1 non-ModelIdentity transition
            list_ch_models += [get_wrapped_chmodel(aux_ch_model) for aux_ch_model in list_ch_models]
            if WRITE_MSG:
                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

        msg += "\n#"

        for aux_ch_model in list_ch_models:
            if WRITE_MSG:
                msg += f"\n\t# {aux_ch_model.func.__name__}"

            non_id_opmodels = 0
            for op_model in aux_ch_model.assign_outprop2op_model.values():
                if not isinstance(op_model, ModelIdentity):
                    non_id_opmodels += 1
            if non_id_opmodels >= 2:
                continue

            with warnings.catch_warnings(record=True) as caught_warnings:
                # avoid warnings found assertion 0 == 1 in chmodel_asserts
                my_fake_file = _FakeFileForPrint()
                try:
                    invalid_prop_finder = InvalidPropFinder(
                        aux_ch_model, solver_name="btor",
                        printing_mode=printing_mode, filename=my_fake_file,
                        solver_seed=self.__class__.PRNG.getrandbits(32)
                    )
                except ValueError as e:
                    if WRITE_MSG:
                        msg += f"\nException raised in InvalidPropFinder: {e}"
                    # if str(e).startswith("exclude_zero_input_prop cannot be True if") or str(e).startswith("var "):
                    #     continue
                    if VERBOSE:
                        print(msg)
                    raise e
                if WRITE_MSG:
                    msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

            for cf_options in list_cf_opts:
                if WRITE_MSG:
                    msg += f"\n> cf_options: {cf_options}"

                if cf_options.input_prop_activebitmode is None:
                    assert cf_options.output_prop_activebitmode is None
                    invalid_prop_finder.solver_name = "z3"
                    iterator = invalid_prop_finder.find_next_invalidprop_quantified_logic()
                else:
                    invalid_prop_finder.solver_name = "btor"
                    iterator = invalid_prop_finder.find_next_invalidprop_activebitmode(
                        1, cf_options.input_prop_activebitmode, cf_options.output_prop_activebitmode
                    )

                for i in range(MAX_UNI_INV_CH):
                    try:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            # find uni-inv ch create Characteristics with free properties which throws warnings
                            uni_inv_ch_found = next(iterator)
                            if WRITE_MSG:
                                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)
                    except StopIteration:
                        break
                    except SolverReturnedUnknownResultError as e:
                        if WRITE_MSG:
                            msg += f"\nException raised in next(iterator): {e}"
                        if cf_options.input_prop_activebitmode is None:
                            continue
                        if VERBOSE:
                            print(msg)
                        raise e
                    except (AssertionError, ValueError) as e:
                        if WRITE_MSG and my_fake_file.list_msgs:
                            msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                            my_fake_file.list_msgs = []
                        if WRITE_MSG:
                            msg += f"\nException raised in next(iterator): {e}"
                        if VERBOSE:
                            print(msg)
                        raise e
                    if WRITE_MSG:
                        msg += f"\nuni-inv characteristic found: {uni_inv_ch_found}"
                        if my_fake_file.list_msgs:
                            msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                            my_fake_file.list_msgs = []
                    #
                    aux_in_w = sum(d.val.width for d in uni_inv_ch_found.input_prop)
                    aux_ext_w = sum(d.val.width for d in uni_inv_ch_found.external_props)
                    if self.__class__.check_get_empirical_ch_weights and USE_C_CODE and (aux_in_w+aux_ext_w) <= 8:
                        ignore_ew = any(isinstance(op_model, (WeakModel, BranchNumberModel))
                                        for op_model in aux_ch_model.assign_outprop2op_model.values())
                        if not ignore_ew and hasattr(aux_ch_model, "_unwrapped_ch_model"):
                            aux_unwrapped_ch_model = aux_ch_model._unwrapped_ch_model
                            ignore_ew = any(isinstance(op_model, (WeakModel, BranchNumberModel))
                                            for op_model in aux_unwrapped_ch_model.assign_outprop2op_model.values())
                        if not ignore_ew:
                            with warnings.catch_warnings(record=True) as caught_warnings:
                                uni_inv_ch_found.compute_empirical_ch_weight(
                                    num_input_samples=2**aux_in_w,
                                    num_external_samples=2**aux_ext_w,
                                    seed=self.__class__.PRNG.random(), C_code=True,
                                )
                                if WRITE_MSG:
                                    msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)
                            ew = uni_inv_ch_found.empirical_ch_weight
                            ew_dl = uni_inv_ch_found.empirical_data_list
                            if WRITE_MSG:
                                msg += f"\n> empirical_ch_weight: {ew}"
                                msg += f"\nempirical_data_list: {[data.vrepr() for data in ew_dl]}"
                            self.assertEqual(ew, math.inf, msg=msg)
                            for ew_dl_i in ew_dl:
                                for ew_j in ew_dl_i.aux_weights:
                                    self.assertEqual(ew_j, math.inf, msg=msg)
                    #

            if num_rounds is not None and num_rounds >= 3 and self._is_permutation and \
                    isinstance(aux_ch_model.func, cascada_ssa.RoundBasedFunction) and \
                    not found_weak_bn_model:
                if type(aux_ch_model) == EncryptionChModel:
                    func = MyCipher
                else:
                    func = aux_ch_model.func
                extra_invalidpropfinder_args = {
                    "printing_mode": printing_mode,
                    "filename": my_fake_file,
                    "solver_seed": self.__class__.PRNG.getrandbits(32)
                }

                try:
                    round_iterator = round_based_invalidprop_search(
                        func, 3, num_rounds, prop_type, "btor",
                        extra_invalidpropfinder_args=extra_invalidpropfinder_args)
                except Exception as e:
                    if WRITE_MSG:
                        msg += f"\nException raised in round_based_invalidprop_search: {e}"
                    if VERBOSE:
                        print(msg)
                    raise e

                prev_num_rounds_E = 1
                while True:
                    try:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            # find uni-inv ch create Characteristics with free properties which throws warnings
                            tuple_rounds, tuple_chs = next(round_iterator)
                            if WRITE_MSG:
                                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)
                    except StopIteration:
                        break
                    except ValueError as e:
                        if WRITE_MSG and my_fake_file.list_msgs:
                            msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                            my_fake_file.list_msgs = []
                        if WRITE_MSG:
                            msg += f"\nException raised in next(round_iterator): {e}"
                        # changing the number of rounds might introduce redundant assignments with external variables
                        if str(e).startswith("round keys") or str(e).startswith("found external variable") or \
                                str(e).startswith("var_separators[") or \
                                str(e).startswith("no non-input/non-external vars in var_separators["):
                            continue
                        else:
                            if VERBOSE:
                                print(msg)
                            raise e
                    if WRITE_MSG:
                        msg += f"\nfound {tuple_rounds} : {', '.join([ch.srepr() for ch in tuple_chs])}"
                        if my_fake_file.list_msgs:
                            msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                            my_fake_file.list_msgs = []
                    self.assertIn(tuple_rounds[0], range(0, num_rounds + 1 - 3), msg=msg)
                    for nr in tuple_rounds[1:]:
                        self.assertIn(nr, range(1, num_rounds + 1), msg=msg)
                    num_rounds_E = sum(tuple_rounds[1:])
                    self.assertLessEqual(prev_num_rounds_E, num_rounds_E, msg=msg)
                    prev_num_rounds_E = num_rounds_E
                    round_iterator.send(INCREMENT_NUM_ROUNDS)

                func.set_num_rounds(num_rounds)
            # end of round_based_invalidprop_search

        # ----------------------------------------------------
        # 4 - Searching for uni-inv characteristics of ciphers
        # ----------------------------------------------------

        if type(ch_model) == CipherChModel:
            msg += "\n#"

            non_id_opmodels = [0, 0]
            for i, cm in enumerate([ch_model.ks_ch_model, ch_model.enc_ch_model]):
                for op_model in cm.assign_outprop2op_model.values():
                    if not isinstance(op_model, ModelIdentity):
                        non_id_opmodels[i] += 1
            if non_id_opmodels[0] <= 1 and non_id_opmodels[1] <= 1:
                wrapped_ch_model = ch_model
            else:
                wrapped_ch_model = get_wrapped_cipher_chmodel(ch_model)

            with warnings.catch_warnings(record=True) as caught_warnings:
                # avoid warnings found assertion 0 == 1 in chmodel_asserts
                my_fake_file = _FakeFileForPrint()
                try:
                    invalid_prop_finder = InvalidCipherPropFinder(
                        wrapped_ch_model, solver_name="btor",
                        printing_mode=printing_mode, filename=my_fake_file,
                        solver_seed=self.__class__.PRNG.getrandbits(32)
                    )
                except ValueError as e:
                    if WRITE_MSG:
                        msg += f"\nException raised in InvalidCipherPropFinder: {e}"
                    # if str(e).startswith("exclude_zero_input_prop cannot be True if") or str(e).startswith("var "):
                    #     continue
                    # else:
                    if VERBOSE:
                        print(msg)
                    raise e
                if WRITE_MSG:
                    msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

            for cf_options in list_cf_opts:
                if WRITE_MSG:
                    msg += f"\n> cf_options: {cf_options}"

                if cf_options.input_prop_activebitmode is None:
                    assert cf_options.output_prop_activebitmode is None
                    invalid_prop_finder.solver_name = "z3"
                    iterator = invalid_prop_finder.find_next_invalidprop_quantified_logic()
                else:
                    invalid_prop_finder.solver_name = "btor"
                    iterator = invalid_prop_finder.find_next_invalidprop_activebitmode(
                        1, cf_options.input_prop_activebitmode, cf_options.output_prop_activebitmode
                    )

                for i in range(MAX_UNI_INV_CH):
                    try:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                        # find uni-inv ch create Characteristics with free properties which throws warnings
                            uni_inv_ch_found = next(iterator)
                            if WRITE_MSG:
                                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)
                    except StopIteration:
                        break
                    except SolverReturnedUnknownResultError as e:
                        if WRITE_MSG:
                            msg += f"\nException raised in next(iterator): {e}"
                        if cf_options.input_prop_activebitmode is None:
                            continue
                        if VERBOSE:
                            print(msg)
                        raise e
                    except (AssertionError, ValueError) as e:
                        if WRITE_MSG and my_fake_file.list_msgs:
                            msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                            my_fake_file.list_msgs = []
                        if WRITE_MSG:
                            msg += f"\nException raised in next(iterator): {e}"
                        if VERBOSE:
                            print(msg)
                        raise e
                    if WRITE_MSG:
                        msg += f"\nuni-inv characteristic found: {uni_inv_ch_found}"
                        if my_fake_file.list_msgs:
                            msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                            my_fake_file.list_msgs = []

            if num_rounds is not None and num_rounds >= 3 and self._is_permutation and not found_weak_bn_model:
                extra_invalidpropfinder_args = {
                    "printing_mode": printing_mode,
                    "filename": my_fake_file,
                    "solver_seed": self.__class__.PRNG.getrandbits(32)
                }

                try:
                    round_iterator = round_based_invalidcipherprop_search(
                        MyCipher, 3, num_rounds, prop_type, "btor",
                        extra_invalidcipherpropfinder_args=extra_invalidpropfinder_args)
                except Exception as e:
                    if WRITE_MSG:
                        msg += f"\nException raised in round_based_invalidcipherprop_search: {e}"
                    if VERBOSE:
                        print(msg)
                    raise e

                prev_num_rounds_E = 1
                while True:
                    try:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                        # find uni-inv ch create Characteristics with free properties which throws warnings
                            tuple_rounds, tuple_chs = next(round_iterator)
                            if WRITE_MSG:
                                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)
                    except StopIteration:
                        break
                    except (AssertionError, ValueError) as e:
                        if WRITE_MSG and my_fake_file.list_msgs:
                            msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                            my_fake_file.list_msgs = []
                        if WRITE_MSG:
                            msg += f"\nException raised in next(round_ch_iterator): {e}"
                        if str(e).startswith("round keys") or str(e).startswith("found external variable")  or \
                                str(e).startswith("var_separators[") or \
                                str(e).startswith("no non-input/non-external vars in var_separators[") or \
                                "pr_one_assertions() == False" in str(e):  # RXDiff
                            continue
                        else:
                            if VERBOSE:
                                print(msg)
                            raise e
                    if WRITE_MSG:
                        msg += f"\nfound {tuple_rounds} : {', '.join([ch.srepr() for ch in tuple_chs])}"
                        if my_fake_file.list_msgs:
                            msg += "\n\t - " + "\n\t - ".join(my_fake_file.list_msgs)
                            my_fake_file.list_msgs = []
                    self.assertIn(tuple_rounds[0], range(0, num_rounds + 1 - 3), msg=msg)
                    for nr in tuple_rounds[1:]:
                        self.assertIn(nr, range(1, num_rounds + 1), msg=msg)
                    num_rounds_E = sum(tuple_rounds[1:4])
                    self.assertLessEqual(prev_num_rounds_E, num_rounds_E, msg=msg)
                    prev_num_rounds_E = num_rounds_E
                    round_iterator.send(INCREMENT_NUM_ROUNDS)

                MyCipher.set_num_rounds(num_rounds)
            # end of round_based_invalidcipherprop_search

        # -------------------
        # 5 - Post-processing
        # -------------------

        msg += f"\n# {get_time()} | searched the uni-inv characteristics"

        # get_random_bvfunc/cipher are stored in the cached
        if func_type_index != 0:
            MyCipher.encryption.round_keys = old_round_keys

        if VERBOSE:
            print(msg)

    def test_random_ch(self, *args, **kwargs):  # disable
        return


class TestXorInvalidPropSearch(TestRandomInvalidPropSearchGeneric, TestXorChModelCharacteristic):
    """Test for the InvalidPropSearch and InvalidCipherPropSearch classes."""

    _obtain_extra_operations = TestXorChModelCharacteristic._obtain_extra_operations

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping XorDiff test_random_invalidpropsearch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        sampled_from([0, 1, 3, 4]),
    )
    @settings(deadline=None, max_examples=MAX_EXAMPLES)
    def test_random_invalidpropsearch(self, width, num_inputs, num_outputs, num_assignments,
                                    seed, func_type_index, num_rounds):
        return super()._test_random_invalidpropsearch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


class TestRXInvalidPropSearch(TestRandomInvalidPropSearchGeneric, TestRXChModelCharacteristic):
    """Test for the InvalidPropSearch and InvalidCipherPropSearch classes."""

    _obtain_extra_operations = TestRXChModelCharacteristic._obtain_extra_operations

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping RXDiff test_random_invalidpropsearch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        sampled_from([0, 1, 3, 4]),
    )
    @settings(deadline=None, max_examples=MAX_EXAMPLES)
    def test_random_invalidpropsearch(self, width, num_inputs, num_outputs, num_assignments,
                                    seed, func_type_index, num_rounds):
        assume(not func_type_index == 1)  # EncryptionCh doesn't support RXDiff
        return super()._test_random_invalidpropsearch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


class TestLinearInvalidPropSearch(TestRandomInvalidPropSearchGeneric, TestLinearChModelCharacteristic):
    """Test for the InvalidPropSearch and InvalidCipherPropSearch classes."""

    _obtain_extra_operations = TestLinearChModelCharacteristic._obtain_extra_operations

    @unittest.skipIf(SKIP_LONG_TESTS, "skipping linear test_random_invalidpropsearch")
    @given(
        integers(min_value=2, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        sampled_from([0, 1, 3, 4]),
    )
    @settings(deadline=None, max_examples=MAX_EXAMPLES)
    def test_random_invalidpropsearch(self, width, num_inputs, num_outputs, num_assignments,
                                    seed, func_type_index, num_rounds):
        assume(not func_type_index == 2)  # CipherCh don't support LinearMask
        return super()._test_random_invalidpropsearch(
            width, num_inputs, num_outputs, num_assignments,
            seed, func_type_index, num_rounds
        )


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(invalidpropsearch))
    tests.addTests(doctest.DocTestSuite(wrappedchmodel))
    return tests
