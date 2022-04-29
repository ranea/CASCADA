"""Tests for the chmodel and the characteristic modules."""
import collections
import datetime
import decimal
import itertools
import math
import pprint
import random
import unittest
import warnings

from wurlitzer import pipes

from hypothesis import assume

from cascada.bitvector import core, operation, secondaryop
from cascada.bitvector import ssa as cascada_ssa
from cascada.primitives import blockcipher

from cascada.abstractproperty.opmodel import ModelIdentity, WeakModel, BranchNumberModel
from cascada.abstractproperty.chmodel import ChModelSigType

from cascada.bitvector.tests.test_others import _get_extra_operations


class TestRandomChModelCharacteristicGeneric(unittest.TestCase):
    """Base class for testing random ChModel and Characteristic."""
    USE_C_CODE = None
    VERBOSE = None
    WRITE_MSG = None

    prop_type = None
    prop_prefix = None
    operation_set_index = None
    check_get_empirical_ch_weights = None

    ChModel = None
    EncryptionChModel = None
    CipherChModel = None

    Characteristic = None
    EncryptionCharacteristic = None
    CipherCharacteristic = None

    # globals for eval (for vrepr)
    eg = {k: v for k, v in itertools.chain(
        vars(core).items(), vars(operation).items(), vars(secondaryop).items(),
        vars(cascada_ssa).items(), vars(blockcipher).items(),
        {"inf": math.inf, "Decimal": decimal.Decimal}.items()
    ) if not k.startswith('_')}

    @classmethod
    def setUpClass(cls):
        cls.PRNG = random.Random()

    _to_obtain_extra_operations = False

    def _obtain_extra_operations(self, width, seed):
        if not self.__class__._to_obtain_extra_operations:
            return None
        extra_operations = _get_extra_operations(width, seed)
        if extra_operations:
            for op in extra_operations:
                self.eg[op.__name__] = op
        return extra_operations

    def _valid_ssa(self, ssa):
        PO = operation.PartialOperation
        for i, (outvar_i, expr_i) in enumerate(ssa.assignments.items()):
            # assuming MatrixOperation are linked with BranchNumberModel objects
            if isinstance(expr_i, secondaryop.MatrixOperation) or \
                    (isinstance(expr_i, PO) and issubclass(expr_i.base_op, secondaryop.MatrixOperation)):
                for j, (outvar_j, expr_j) in enumerate(list(ssa.assignments.items())[i:]):
                    if isinstance(expr_j, PO) and issubclass(expr_j.base_op, operation.Extract) and \
                            expr_j.args[0] == outvar_i:
                        # Extract cannot be used with the output of BranchNumberModel-based operations
                        return False
        return True

    def _get_external_var2prop(self, ssa):
        return None

    def _get_to_full_verify(self, ch_model):
        WRITE_MSG = self.__class__.WRITE_MSG
        msg = ""

        # max 2**16 samples and up to one non-trivial OpModel
        sw = sum(d.val.width for d in ch_model.input_prop)
        sw += sum(d.width for d in ch_model.external_var2prop)
        to_full_verify = 2 ** sw <= 2 ** 16

        # don't allow 2+ non-identity OpModel (even a 2nd OpModel
        # with 0 weight can decrease the weight of the first OpModel)
        to_full_verify &= len([om for om in ch_model.assign_outprop2op_model.values()
                               if not isinstance(om, ModelIdentity)]) <= 1

        # don't allow intermediate identity assignments
        # (they prevent from detecting later dependent inputs to an OpModel)
        if to_full_verify and any(type(op) == operation.BvIdentity
                                  for op in ch_model.ssa.assignments.values()):
            if WRITE_MSG:
                msg += "\nfound intermediate identity assignment"
            to_full_verify = False

        # don't allow Weak or BranchNumber models
        if to_full_verify and any(isinstance(op, (WeakModel, BranchNumberModel))
                                  for op in ch_model.assign_outprop2op_model.values()):
            if WRITE_MSG:
                msg += "\nfound intermediate Weak/BranchNumber Model"
            to_full_verify = False

        # don't allow operations with a Variable appearing multiple times
        # (model with dependant inputs)
        if to_full_verify:
            for _, op in ch_model.ssa.assignments.items():
                if len(op.args) > 1:
                    full_args = [ch_model.var2prop[var].val for var in op.args]
                    full_args_str = "".join(str(a) for a in full_args)
                    for arg in full_args:
                        for v in arg.atoms(core.Variable):
                            if full_args_str.count(str(v)) >= 2:
                                if WRITE_MSG:
                                    msg += f"\nfound operation with duplicate inputs " \
                                           f"{op} | full_args={full_args}"
                                to_full_verify = False

        return to_full_verify, msg

    def _test_random_ch(self, width, num_inputs, num_outputs, num_assignments,
                        seed, func_type_index, num_rounds):
        USE_C_CODE = self.__class__.USE_C_CODE
        VERBOSE = self.__class__.VERBOSE
        WRITE_MSG = self.__class__.WRITE_MSG

        prop_type = self.__class__.prop_type
        prop_prefix = self.__class__.prop_prefix
        operation_set_index = self.__class__.operation_set_index

        ChModel = self.__class__.ChModel
        EncryptionChModel = self.__class__.EncryptionChModel
        CipherChModel = self.__class__.CipherChModel

        Characteristic = self.__class__.Characteristic
        EncryptionCharacteristic = self.__class__.EncryptionCharacteristic
        CipherCharacteristic = self.__class__.CipherCharacteristic

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
        msg = f"{get_time()} | test_random_ch({width}, {num_inputs}, {num_outputs}, " \
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

        with warnings.catch_warnings(record=True) as caught_warnings:
            # ignore redundant assignments
            # warnings.filterwarnings("ignore", category=UserWarning)
            if func_type_index == 0:
                ssa = MyFoo.to_ssa(input_names=input_names, id_prefix="a")
                if not self._valid_ssa(ssa):
                    if WRITE_MSG:
                        msg += f"\naborting due to invalid ssa:\n{ssa}"
                    return
                e_v2d = self._get_external_var2prop(ssa)
                ch_model = ChModel._prop_new(MyFoo, prop_type, input_names, prefix=f"{prop_prefix}x",
                                             external_var2prop=e_v2d, op_model_class2options=None)
            elif func_type_index == 1:
                ssa = MyCipher.encryption.to_ssa(input_names=[str(x) for x in pt_vars], id_prefix="a")
                if not self._valid_ssa(ssa):
                    if WRITE_MSG:
                        msg += f"\naborting due to invalid ssa:\n{ssa}"
                    return
                ch_model = EncryptionChModel(MyCipher, prop_type)
            elif func_type_index == 2:
                ssa = MyCipher.encryption.to_ssa(input_names=[str(x) for x in pt_vars], id_prefix="a")
                if not self._valid_ssa(ssa):
                    if WRITE_MSG:
                        msg += f"\naborting due to invalid ssa:\n{ssa}"
                    return
                ssa = MyCipher.key_schedule.to_ssa(input_names=[str(x) for x in mk_vars], id_prefix="a")
                if not self._valid_ssa(ssa):
                    if WRITE_MSG:
                        msg += f"\naborting due to invalid ssa:\n{ssa}"
                    return
                ch_model = CipherChModel(MyCipher, prop_type)
            if WRITE_MSG:
                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

        if WRITE_MSG:
            msg += f"\nChModel: {ch_model}"
            msg += f"\nChModel log: \n\t" + "\n\t".join(ch_model.get_formatted_logged_msgs())

        for sig_type in [ChModelSigType.InputOutput, ChModelSigType.Unique]:
            ch_model.signature(sig_type)
        if not isinstance(ch_model, CipherChModel):
            for repeat, vrepr_label in itertools.product([True, False], [True, False]):
                ch_model.dotprinting(repeat=repeat, vrepr_label=vrepr_label)

        with warnings.catch_warnings(record=True) as caught_warnings:
            # ignore redundant assignments
            # warnings.filterwarnings("ignore", category=UserWarning)
            self.assertEqual(ch_model.vrepr(), eval(ch_model.vrepr(), self.eg).vrepr(), msg=msg)
            if WRITE_MSG:
                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

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

        # ----------------------
        # 3 - Generating the ch.
        # ----------------------

        try:
            if type(ch_model) == ChModel:
                ch = Characteristic.random(ch_model, seed)
            elif type(ch_model) == EncryptionChModel:
                ch = EncryptionCharacteristic.random(ch_model, seed)
            elif type(ch_model) == CipherChModel:
                ch = CipherCharacteristic.random(ch_model, seed)
            if WRITE_MSG:
                msg += f"\nCharacteristic: {ch}"
                msg += f"\nCharacteristic log: \n\t" + "\n\t".join(ch.get_formatted_logged_msgs())
        except ValueError as e:
            if str(e).startswith("no random characteristic found in"):
                return
            else:
                if WRITE_MSG:
                    msg += f"\nException raised in random()"
                    print(msg)
                raise e

        for sig_type in [ChModelSigType.InputOutput, ChModelSigType.Unique]:
            ch.signature(sig_type)
        if not isinstance(ch, CipherCharacteristic):
            for repeat, vrepr_label in itertools.product([True, False], [True, False]):
                ch.dotprinting(repeat=repeat, vrepr_label=vrepr_label)

        with warnings.catch_warnings(record=True) as caught_warnings:
            # ignore redundant assignments
            # warnings.filterwarnings("ignore", category=UserWarning)
            self.assertEqual(ch.vrepr(), eval(ch.vrepr(), self.eg).vrepr(), msg=msg)
            if WRITE_MSG:
                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

        if type(ch) in [Characteristic, EncryptionCharacteristic]:
            list_ch = [ch]
        elif type(ch) == CipherCharacteristic:
            list_ch = [ch.ks_characteristic, ch.enc_characteristic]

        msg += "\n#"

        for aux_ch_model, aux_ch in zip(list_ch_models, list_ch):
            if WRITE_MSG:
                msg += f"\n\t# {aux_ch_model.func.__name__}"

            var2ct = {v.val: d.val for v, d in aux_ch.var_prop2ct_prop.items()}
            if WRITE_MSG:
                msg += f"\ntuple_assign_outprop2op_model: {pprint.pformat(aux_ch.tuple_assign_outprop2op_model)}"
                msg += f"\nvar2ct: {var2ct}"

            self.assertEqual(set(p.val for p in aux_ch.free_props),
                             set(aux_ch.ch_model.ssa._input_vars_not_used), msg=msg)

            input_prop, output_prop, external_props, assign_outprop_list = \
                Characteristic.get_properties_for_initialization(aux_ch_model, var2ct)
            self.assertSequenceEqual(input_prop, aux_ch.input_prop, msg=msg)
            self.assertSequenceEqual(output_prop, aux_ch.output_prop, msg=msg)
            self.assertSequenceEqual(external_props, aux_ch.external_props, msg=msg)
            self.assertSequenceEqual(assign_outprop_list, [v for v, _ in aux_ch.tuple_assign_outprop2op_model], msg=msg)

            if num_rounds is not None and num_rounds >= 2:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    # ignore redundant assignments
                    # warnings.filterwarnings("ignore", category=UserWarning)
                    try:
                        sub_ch_list = aux_ch.split(aux_ch.get_round_separators())
                    except ValueError as e:
                        if str(e).startswith("var_separators") or str(e).startswith("no non-input/non-external") or \
                                str(e).startswith("if symbolic_inputs, expected no Constant values"):
                            # Random RoundBasedFunction might remove
                            # round output vars in ssa (even w/o simplification ctx)
                            continue
                        else:
                            if WRITE_MSG:
                                msg += f"\nException raised in split(ch.get_round_separators())"
                                print(msg)
                            raise e
                    else:
                        for sub_ch in sub_ch_list:
                            self.assertEqual(sub_ch.vrepr(), eval(sub_ch.vrepr(), self.eg).vrepr(), msg=msg)
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

        if type(ch) == CipherCharacteristic:
            if WRITE_MSG:
                msg += f"\n\t# {ch_model.ks_ch_model.func.__name__} + {ch_model.enc_ch_model.func.__name__}"

            var2ct = {v.val: d.val for v, d in itertools.chain(
                ch.ks_characteristic.var_prop2ct_prop.items(),
                ch.enc_characteristic.var_prop2ct_prop.items())}
            if WRITE_MSG:
                msg += f"\nvar2ct: {var2ct}"

            ks_input_prop, ks_output_prop, ks_assign_outprop_list, \
            enc_input_prop, enc_output_prop, enc_assign_outprop_list = \
                CipherCharacteristic.get_properties_for_initialization(ch_model, var2ct)

            self.assertSequenceEqual(ks_input_prop, ch.ks_characteristic.input_prop, msg=msg)
            self.assertSequenceEqual(ks_output_prop, ch.ks_characteristic.output_prop, msg=msg)
            self.assertSequenceEqual([], ch.ks_characteristic.external_props, msg=msg)
            self.assertSequenceEqual(ks_assign_outprop_list, [v for v, _ in ch.ks_characteristic.tuple_assign_outprop2op_model], msg=msg)

            self.assertSequenceEqual(enc_input_prop, ch.enc_characteristic.input_prop, msg=msg)
            self.assertSequenceEqual(enc_output_prop, ch.enc_characteristic.output_prop, msg=msg)
            self.assertSequenceEqual(ks_output_prop, ch.enc_characteristic.external_props, msg=msg)
            self.assertSequenceEqual(enc_assign_outprop_list, [v for v, _ in ch.enc_characteristic.tuple_assign_outprop2op_model], msg=msg)

        msg += f"\n# {get_time()} | generated the characteristic"

        # ----------------------------------
        # 4 - Computing the empirical weight
        # ----------------------------------

        del ch, ch_model
        for aux_ch_model, aux_ch in zip(list_ch_models, list_ch):
            if WRITE_MSG:
                msg += f"\n\t# {aux_ch_model.func.__name__}"

            ch, ch_model = aux_ch, aux_ch_model

            # ----- 4.1 - Check whether to full verify -----

            num_input_samples = None
            num_external_samples = None

            to_full_verify, aux_msg = self._get_to_full_verify(aux_ch_model)
            msg += aux_msg

            if to_full_verify:
                num_input_samples, num_external_samples = math.inf, math.inf

            if WRITE_MSG:
                msg += f"\nto_full_verify: {to_full_verify}"

            # ----- 4.2 - Run compute_empirical_ch_weight() with different parameters -----

            num_parallel_processes = self.__class__.PRNG.randint(2, 3)
            EWOptions = collections.namedtuple(
                'EWOptions', ['split_by_max_weight', 'split_by_rounds', 'C_code', 'num_parallel_processes'])
            list_ew_opts = [
                EWOptions(None, False, False, None),
                EWOptions(min(ch.assignment_weights), False, False, None),
            ]
            if num_rounds is not None and num_rounds >= 2:
                list_ew_opts += [
                    EWOptions(None, True, False, None),
                ]
            if USE_C_CODE:
                list_ew_opts += [
                    EWOptions(None, False, True, None),
                    EWOptions(max(ch.assignment_weights), False, True, None),
                    # num_parallel_processes requires C_code=True
                    EWOptions(None, False, True, num_parallel_processes),
                    EWOptions(max(ch.assignment_weights), False, True, num_parallel_processes),
                ]
                if num_rounds is not None and num_rounds >= 2:
                    list_ew_opts += [
                        EWOptions(None, True, True, None),
                        EWOptions(None, True, True, num_parallel_processes),
                    ]

            first_ew_py = None
            first_ew_dl_py = None
            for ew_options in list_ew_opts:
                if ew_options.split_by_max_weight is not None and ew_options.split_by_max_weight < ch.ch_weight:
                    continue

                with warnings.catch_warnings(record=True) as caught_warnings:
                    # ignore redundant assignments when splitting
                    # warnings.filterwarnings("ignore", category=UserWarning)
                    # cffi warnings ignored
                    # warnings.filterwarnings("ignore", category=DeprecationWarning)
                    try:
                        ch.compute_empirical_ch_weight(
                            num_input_samples=num_input_samples, num_external_samples=num_external_samples,
                            split_by_max_weight=ew_options.split_by_max_weight,
                            split_by_rounds=ew_options.split_by_rounds,
                            seed=self.__class__.PRNG.random(), C_code=ew_options.C_code,
                            num_parallel_processes=ew_options.num_parallel_processes,
                        )
                    except Exception as e:
                        if num_rounds is not None and num_rounds >= 2 and str(e).startswith("var_separators") or \
                                str(e).startswith("no non-input/non-external vars in var_separators") or \
                                str(e).startswith("if symbolic_inputs, expected no Constant values"):
                            # Random RoundBasedFunction might remove
                            # round output vars in ssa (even w/o simplification ctx)
                            continue
                        else:
                            if WRITE_MSG:
                                msg += f"\nException raised in compute_empirical_ch_weight with {ew_options}"
                                print(msg)
                            raise e
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                ew_py = ch.empirical_ch_weight
                ew_dl_py = ch.empirical_data_list

                if WRITE_MSG:
                    msg += f"\n> empirical options: {ew_options}"
                    msg += f"\nempirical_ch_weight: {ew_py}"
                    msg += f"\nempirical_data_list: {[data.vrepr() for data in ew_dl_py]}"

                for ew_dl in ew_dl_py:
                    self.assertEqual(ew_dl.vrepr(), eval(ew_dl.vrepr(), self.eg).vrepr(), msg=msg)

                # ----- 4.3 - Extra tests when full verify -----

                # C code cannot loop exhaustively over the external values
                if to_full_verify and (ew_options.C_code is False or not ch.external_props):
                    if math.inf in [ew_py, ch.ch_weight] and ew_py != ch.ch_weight:
                        error_emp_decimal = math.inf
                    else:
                        error_emp_decimal = (ew_py - ch.ch_weight).copy_abs()
                    max_error_dec = decimal.Decimal(ch_model.error())

                    if WRITE_MSG:
                        msg += "\nemp_weight - decimal_weight: {}".format(error_emp_decimal)
                        msg += "\nchmodel.error()            : {}".format(max_error_dec)

                    if any(ew_dl.num_inf_aux_weights > 0 for ew_dl in ew_dl_py):
                        # ch.ch_weight is the average among all external samples
                        # but ew_py ignore external samples leading to infinite weight
                        self.assertTrue(len(ch_model.ssa.external_vars) > 0, msg=msg)
                        self.assertLessEqual(ew_py, ch.ch_weight + max_error_dec, msg=msg)
                    else:
                        # error offset '1E-27' found empirically
                        self.assertLessEqual(error_emp_decimal, max_error_dec + decimal.Decimal('1E-27'), msg=msg)

                    if ew_options.split_by_max_weight is not None:
                        # ew_py is closer to ch.weight than first_ew_py (closer to diff weight)
                        # error offset '4E-27' found empirically
                        self.assertGreaterEqual(ew_py + decimal.Decimal('1E-26'), first_ew_py, msg=msg)

                if first_ew_py is None:
                    assert first_ew_dl_py is None
                    first_ew_py = ew_py
                    first_ew_dl_py = ew_dl_py

            # ----- 4.3 - Check _get_empirical_ch_weights C code and that of Python are equivalent -----

            if self.__class__.check_get_empirical_ch_weights and USE_C_CODE and len(ch.external_props) == 0:
                # no need to full verify
                num_input_samples, num_external_samples = ch._get_empirical_data_complexity(None, 0)

                if WRITE_MSG:
                    msg += f"\n> num_input_samples (for C/Py checking): {num_input_samples}"

                my_seed = self.__class__.PRNG.random()

                with warnings.catch_warnings(record=True) as caught_warnings:
                    # cffi warnings ignored
                    # warnings.filterwarnings("ignore", category=DeprecationWarning)
                    with pipes() as (out, err):
                        # capturing C-level piped output:
                        num_right_inputs_C_code = ch._get_empirical_ch_weights_C(
                            num_input_samples, num_external_samples, seed=my_seed,
                            num_parallel_processes=None, get_sampled_inputs=True)
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                captured_text = out.read()
                list_input_samples = eval("(" + captured_text + ")")

                if WRITE_MSG:
                    msg += "\nnum_right_inputs_C_code: {}".format(num_right_inputs_C_code)

                self.assertEqual(len(list_input_samples), num_input_samples, msg=msg)
                self.assertEqual(len(list_input_samples[0]), len(ch.input_prop), msg=msg)

                num_right_inputs_Python = ch._get_empirical_ch_weights_Python(
                    num_input_samples, num_external_samples, seed=my_seed,
                    list_input_samples=list_input_samples)

                if WRITE_MSG:
                    msg += "\nnum_right_inputs_Python: {}".format(num_right_inputs_Python)

                self.assertEqual(num_right_inputs_C_code, num_right_inputs_Python, msg=msg)

        msg += f"\n{get_time()} | computed empirical weights\n\n"

        if func_type_index != 0:
            MyCipher.encryption.round_keys = old_round_keys  # get_random_cipher outputs are stored in the cached

        if VERBOSE:
            print(msg)
