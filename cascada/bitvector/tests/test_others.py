"""Tests for the context, printing and ssa module."""
import datetime
import pprint
import unittest
import doctest
import random
import warnings

from hypothesis import given, example, settings, assume, HealthCheck
from hypothesis.strategies import integers, sampled_from

from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.ssa import BvFunction, get_random_bvfunction, SSAReturn

# for vrepr
from cascada.bitvector.operation import *
from cascada.bitvector.secondaryop import *
from cascada.bitvector.ssa import SSA

import cascada.bitvector.context
import cascada.bitvector.printing
import cascada.bitvector.ssa


USE_C_CODE = True

VERBOSE = False
WRITE_MSG = False


def _get_extra_operations(width, seed):
    if width not in [2, 3]:
        return None

    PRNG = random.Random()
    PRNG.seed(seed)

    class RandomLut(LutOperation):
        lut = [core.Constant(PRNG.randint(0, 2 ** width - 1), width) for i in range(2 ** width)]

    num_inputs = PRNG.randint(1, 3)  # total input_width == num_columns
    rnd_matrix = [[PRNG.randint(0, 1) for _ in range(num_inputs*width)] for _ in range(width)]

    class RandomMatrix(MatrixOperation):
        arity = [num_inputs, 0]
        matrix = [[core.Constant(x, 1) for x in row] for row in rnd_matrix]

    if width == 2:
        inv_lut = [0, 1, 3, 2]
    else:
        assert width == 3
        inv_lut = [0, 1, 5, 6, 7, 2, 3, 4]

    class Inversion(LutOperation):
        lut = [core.Constant(x, width) for x in inv_lut]

    return RandomLut, RandomMatrix, Inversion


class TestSSA(unittest.TestCase):
    """Test for the ssa module."""

    @classmethod
    def setUpClass(cls):
        cls.PRNG = random.Random()

    def test_invalid(self):
        class SimpleBvFunction(BvFunction):
             input_widths = [8, 8]
             output_widths = [8, 8]

             @classmethod
             def eval(cls, x, y):
                 return x ^ y, x

        with self.assertRaises(TypeError):
            SimpleBvFunction(Variable("x", 8), Variable("x", 8))

    def test_copy_function(self):
        class CopyFunction(BvFunction):
             input_widths = [8]
             output_widths = [8, 8, 8]

             @classmethod
             def eval(cls, x):
                 return x, x, x

        ssa = CopyFunction.to_ssa(["x"], "a")
        zero = Constant(0, 8)
        self.assertListEqual([zero, zero, zero], list(ssa.eval(input_vals=[zero])))
        with warnings.catch_warnings():
            # cffi warnings ignored
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.assertListEqual([zero, zero, zero], list(ssa.eval(input_vals=[zero], C_code=True)))

    @unittest.skip("skipping test_random_bvf")
    @given(
        sampled_from([2, 3, 8]),
        integers(min_value=1, max_value=4),
        integers(min_value=1, max_value=4),
        integers(min_value=1, max_value=32),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=6*10 if USE_C_CODE else 60*10,  # ~10min
              suppress_health_check=[HealthCheck.filter_too_much])
    def test_random_bvf(self, width, num_inputs, num_outputs, num_assignments, seed, operation_set_index, num_rounds):
        assume(num_inputs + num_assignments >= num_outputs)

        # avoid errors due to simplification between rounds
        assume(not (num_rounds >= 2 and num_inputs < num_outputs))

        # MyFoo(*input_vals) below does not set external vars to 0 as ssa.eval()
        external_variable_prefix = None

        if num_rounds == 0:
            num_rounds = None

        def get_time():
            now = datetime.datetime.now()
            return "{}-{}:{}".format(now.day, now.hour, now.minute)

        # timestamp messages ignored WRITE_MSG
        msg = f"{get_time()} | test_random_bvf({width}, {num_inputs}, {num_outputs}, " \
              f"{num_assignments}, {seed}, {operation_set_index}, {num_rounds})"
        if VERBOSE:
            print(msg)

        extra_operations = _get_extra_operations(width, seed)
        eg = globals().copy()
        if extra_operations:
            for op in extra_operations:
                eg[op.__name__] = op

        MyFoo = get_random_bvfunction(width, num_inputs, num_outputs, num_assignments, seed,
                                      external_variable_prefix=external_variable_prefix,
                                      operation_set_index=operation_set_index,
                                      num_rounds=num_rounds, extra_operations=extra_operations)

        self.__class__.PRNG.seed(seed)
        input_vals = []
        for _ in range(num_inputs):
            x = Constant(self.__class__.PRNG.randint(0, 2 ** width - 1), width)
            input_vals.append(x)

        output_MyFoo = MyFoo(*input_vals)
        logged_msgs_ct = MyFoo.get_formatted_logged_msgs()

        if WRITE_MSG:
            msg += f"\ninput_vals: {input_vals}"
            msg += f"\noutput_MyFoo: {output_MyFoo}"
            msg += f"\nformatted logged messages (constant): \n\t"
            msg += "\n\t".join(logged_msgs_ct)

        input_names = ["x" + str(i) for i in range(num_inputs)]
        input_vars = [Variable(input_names[i], width) for i in range(num_inputs)]
        output_MyFoo_symbolic = MyFoo(*input_vars, symbolic_inputs=True, simplify=False)
        logged_msgs_symbolic = MyFoo.get_formatted_logged_msgs()

        if WRITE_MSG:
            msg += f"\nMyFoo: {output_MyFoo_symbolic}"
            msg += f"\ninput_names: {input_names}"
            msg += f"\nformatted logged messages (symbolic): \n\t"
            msg += "\n\t".join(logged_msgs_symbolic)

        self.assertNotEqual(id(logged_msgs_ct), id(logged_msgs_symbolic))

        for repeat, vrepr_label in itertools.product([True, False], [True, False]):
            MyFoo.dotprinting(input_names, repeat=repeat, vrepr_label=vrepr_label)

        for replace_multiuse_vars, to_simplify, to_validate in \
                itertools.product([False, True], [False, True], [False, True]):
            # context.Cache cannot be disabled due to Memoization
            with context.Simplification(to_simplify), context.Validation(to_validate):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    # ignore redundant assignments
                    # warnings.filterwarnings("ignore", category=UserWarning)
                    ssa = MyFoo.to_ssa(input_names=input_names, id_prefix="a",
                                    replace_multiuse_vars=replace_multiuse_vars)
                    ssa_from_vrepr = eval(ssa.vrepr(), eg)
                    ssa_copy = ssa.copy()
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                header, body = ssa.get_C_code("eval_ssa")

                if WRITE_MSG:
                    msg += f"\n#\n(replace_multiuse_vars, to_simplify): ({replace_multiuse_vars}, {to_simplify})"
                    msg += f"\nssa:\n{ssa}"
                    msg += f"\nssa C_code:\n{header}\n{body}"

                output_ssa = ssa.eval(input_vals=input_vals)
                if USE_C_CODE:
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        # cffi warnings ignored
                        # warnings.filterwarnings("ignore", category=DeprecationWarning)
                        output_ssa_C = ssa.eval(input_vals=input_vals, C_code=True)
                        if WRITE_MSG:
                            msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                if WRITE_MSG:
                    msg += f"\noutput_ssa: {output_ssa}"
                    if USE_C_CODE:
                        msg += f"\noutput_ssa_C: {output_ssa_C}"

                self.assertEqual(ssa.vrepr(), ssa_from_vrepr.vrepr(), msg=msg)
                self.assertEqual(output_MyFoo, output_ssa, msg=msg)
                if USE_C_CODE:
                    self.assertEqual(output_MyFoo, output_ssa_C, msg=msg)

                # test copy()

                if WRITE_MSG:
                    msg += f"\nssa_copy:\n{ssa_copy}"

                self.assertTupleEqual(ssa.input_vars, ssa_copy.input_vars, msg=msg)
                self.assertTupleEqual(ssa.output_vars, ssa_copy.output_vars, msg=msg)
                self.assertDictEqual(ssa.assignments, ssa_copy.assignments, msg=msg)

                ssa_copy.input_vars = []
                ssa_copy.output_vars = []
                ssa_copy.assignments = []

                self.assertNotEqual(ssa.input_vars, ssa_copy.input_vars, msg=msg)
                self.assertNotEqual(ssa.output_vars, ssa_copy.output_vars, msg=msg)
                self.assertNotEqual(ssa.assignments, ssa_copy.assignments, msg=msg)

                # test _to_bvfunction

                MyFoo_v2 = ssa.to_bvfunction()
                output_MyFoo_v2 = MyFoo_v2(*input_vals)

                if WRITE_MSG:
                    msg += f"\nMyFoo_v2: {MyFoo_v2(*input_vars, symbolic_inputs=True)}"
                    msg += f"\noutput_MyFoo_v2: {output_MyFoo_v2}"

                with warnings.catch_warnings(record=True) as caught_warnings:
                    # ignore redundant assignments
                    # warnings.filterwarnings("ignore", category=UserWarning)
                    ssa_v2 = MyFoo_v2.to_ssa(input_names=input_names, id_prefix="a",
                                             replace_multiuse_vars=replace_multiuse_vars)
                    ssa_v2_from_vrepr = eval(ssa_v2.vrepr(), eg)
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                output_ssa_v2 = ssa_v2.eval(input_vals=input_vals)
                with warnings.catch_warnings(record=True) as caught_warnings:
                    # cffi warnings ignored
                    # warnings.filterwarnings("ignore", category=DeprecationWarning)
                    if USE_C_CODE:
                        output_ssa_v2_C = ssa_v2.eval(input_vals=input_vals, C_code=True)
                    if WRITE_MSG:
                        msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                if WRITE_MSG:
                    msg += f"\nssa_v2:\n{ssa_v2}"
                    msg += f"\noutput_ssa_v2: {output_ssa_v2}"
                    if USE_C_CODE:
                        msg += f"\noutput_ssa_v2_C: {output_ssa_v2_C}"

                self.assertEqual(ssa_v2.vrepr(), ssa_v2_from_vrepr.vrepr(), msg=msg)
                self.assertEqual(output_MyFoo, output_MyFoo_v2, msg=msg)
                self.assertEqual(output_ssa, output_ssa_v2, msg=msg)
                if USE_C_CODE:
                    self.assertEqual(output_ssa, output_ssa_v2_C, msg=msg)

                # test dotprinting

                for repeat, vrepr_label in itertools.product([True, False], [True, False]):
                    ssa.dotprinting(repeat=repeat, vrepr_label=vrepr_label)

                # test SSA.split

                var_separators = [[]]
                i = 0
                for var, expr in ssa.assignments.items():
                    if isinstance(expr, SSAReturn):
                        continue
                    # PRNG.randint(0, len(ssa.assignments) / (n+1))
                    # is 0 n-times if it is called len(ssa.assignments)
                    # e.g., len(ssa.assignments)=20
                    #   randint(0, 20/(4+1)=4) has 1 out of 5 to be 0
                    #   in average, 0 is got for 20/5=4 assignments

                    # 4 sub_ssa in average with 3 assignments in average
                    if self.__class__.PRNG.randint(0, max(2, len(ssa.assignments) // (3*4+1))):
                        var_separators[i].append(var)
                    if self.__class__.PRNG.randint(0, max(3, len(ssa.assignments) // (4+1))) and \
                            len(var_separators[i]) >= 1:
                        i += 1
                        var_separators.append([])
                if len(var_separators[-1]) == 0:
                    del var_separators[-1]

                list_var_separators = [var_separators]
                if num_rounds is not None and num_rounds >= 2:
                    list_var_separators.append(ssa.get_round_separators())

                for index_vs, aux_var_separators in enumerate(list_var_separators):
                    if WRITE_MSG:
                        msg += f"\n\t#\nvar_separators: {aux_var_separators}"

                    if len(aux_var_separators) >= 1:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            # ignore redundant assignments
                            # warnings.filterwarnings("ignore", category=UserWarning)
                            try:
                                sub_ssa_list = ssa.split(aux_var_separators)
                            except ValueError as e:
                                if index_vs == 1 or str(e).startswith("split does not support copies of external variables"):
                                    # Random RoundBasedFunction might remove
                                    # round output vars in ssa (even w/o simplification ctx)
                                    continue
                                else:
                                    print(msg)
                                    raise e
                            if WRITE_MSG:
                                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                        output_vals_prev_subssa = input_vals
                        output_vals_prev_subssa_C = input_vals
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            # cffi warnings ignored
                            # warnings.filterwarnings("ignore", category=DeprecationWarning)
                            for sub_ssa in sub_ssa_list:
                                output_vals_prev_subssa = sub_ssa.eval(input_vals=output_vals_prev_subssa)
                                if USE_C_CODE:
                                    try:
                                        output_vals_prev_subssa_C = sub_ssa.eval(input_vals=output_vals_prev_subssa_C, C_code=True)
                                    except Exception as e:
                                        if WRITE_MSG:
                                            msg += f"\nException raised in the C-evaluation of sub_ssa: {sub_ssa}"
                                        print(msg)
                                        raise e
                            if WRITE_MSG:
                                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

                        if WRITE_MSG:
                            msg += f"\nsub_ssa_list:\n{pprint.pformat(sub_ssa_list)}"
                            msg += f"\noutput_vals_sub_ssa_list: {output_vals_prev_subssa}"
                            if USE_C_CODE:
                                msg += f"\noutput_vals_prev_subssa_C: {output_vals_prev_subssa_C}"

                        self.assertEqual(output_ssa, output_vals_prev_subssa, msg=msg)
                        if USE_C_CODE:
                            self.assertEqual(output_ssa, output_vals_prev_subssa_C, msg=msg)

        if VERBOSE:
            print(msg)


# noinspection PyUnusedLocal,PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(cascada.bitvector.printing))
    tests.addTests(doctest.DocTestSuite(cascada.bitvector.context))
    tests.addTests(doctest.DocTestSuite(cascada.bitvector.ssa))
    return tests
