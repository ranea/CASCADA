"""Tests for the blockcipher module."""
import datetime
import doctest
import pprint
import random
import unittest
import warnings

from hypothesis import given, example, settings
from hypothesis.strategies import integers

from cascada.bitvector.core import Variable, Constant
from cascada.bitvector.ssa import RoundBasedFunction

from cascada.primitives import blockcipher
from cascada.primitives.blockcipher import (
    Encryption, Cipher, get_random_cipher
)

from cascada.bitvector.tests.test_others import _get_extra_operations

TIME_VERBOSE = False
VERBOSE = False


class TestCipher(unittest.TestCase):
    """Tests of the Cipher class."""

    @classmethod
    def setUp(cls):
        class MyKeySchedule(RoundBasedFunction):
            input_widths = [8]
            output_widths = [8 for i in range(2)]
            num_rounds = 2

            @classmethod
            def eval(cls, k):
                return [k + i for i in range(cls.num_rounds)]

            @classmethod
            def set_num_rounds(cls, new_num_rounds):
                cls.num_rounds = new_num_rounds
                cls.output_widths = [8 for _ in range(new_num_rounds)]

        class MyEncryption(Encryption, RoundBasedFunction):
            input_widths = [8]
            output_widths = [8]
            num_rounds = 2
            round_keys = None

            @classmethod
            def eval(cls, x):
                for k_i in cls.round_keys:
                    x ^= k_i
                return tuple([x])

            @classmethod
            def set_num_rounds(cls, new_num_rounds):
                cls.num_rounds = new_num_rounds

        class MyCipher(Cipher):
            key_schedule = MyKeySchedule
            encryption = MyEncryption
            num_rounds = 2

            @classmethod
            def set_num_rounds(cls, new_num_rounds):
                cls.num_rounds = new_num_rounds
                cls.key_schedule.set_num_rounds(new_num_rounds)
                cls.encryption.set_num_rounds(new_num_rounds)

        cls.example_cipher = MyCipher
        cls.PRNG = random.Random()  # for test_random_cipher

    def test_creation(self):
        self.assertEqual(self.example_cipher([0], [0]), (0x01, ))

        with self.assertRaises(TypeError):
            self.example_cipher([Variable("x", 8)], [Variable("y", 8)])

    def test_iterated(self):
        self.example_cipher.set_num_rounds(3)
        self.assertEqual(self.example_cipher([0], [0]), (0x03, ))

    @unittest.skip("skipping test_random_cipher")
    @given(
        integers(min_value=2, max_value=8),
        integers(min_value=1, max_value=3),
        integers(min_value=1, max_value=8),
        integers(min_value=1, max_value=4),
        integers(min_value=1, max_value=4),
        integers(min_value=1, max_value=8*2),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
    )
    @settings(deadline=None, max_examples=1000)  # 1000 examples ~1min
    def test_random_cipher(self, width, key_num_inputs, key_num_assignments,
                           enc_num_inputs, enc_num_outputs, enc_num_assignments,
                           seed, operation_set_index):
        self.__class__.PRNG.seed(seed)

        if not enc_num_inputs + enc_num_assignments >= enc_num_outputs:
            enc_num_assignments = enc_num_outputs - enc_num_inputs

        def get_time():
            now = datetime.datetime.now()
            return "{}-{}:{}".format(now.day, now.hour, now.minute)

        # timestamp messages ignored WRITE_MSG
        msg = f"{get_time()} | test_random_cipher({width}, {key_num_inputs}, {key_num_assignments}, " \
              f"{enc_num_inputs}, {enc_num_outputs}, {enc_num_assignments}, {seed}, {operation_set_index})"
        if TIME_VERBOSE:
            print(msg)

        extra_operations = _get_extra_operations(width, seed)  # no calls to vrepr here

        with warnings.catch_warnings(record=True) as caught_warnings:
            # get_random_cipher calls to_ssa()
            # warnings.filterwarnings("ignore", category=UserWarning)
            MyCipher = get_random_cipher(width, key_num_inputs, key_num_assignments,
                                         enc_num_inputs, enc_num_outputs, enc_num_assignments, seed,
                                         external_variable_prefix="k", operation_set_index=operation_set_index,
                                         extra_operations=extra_operations)
            msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

        pt_vals = []
        mk_vals = []
        for i in range(enc_num_inputs + key_num_inputs):
            x = Constant(self.__class__.PRNG.randint(0, 2 ** width - 1), width)
            if i < enc_num_inputs:
                pt_vals.append(x)
            else:
                mk_vals.append(x)
        output_MyCipher_vals = MyCipher(plaintext=pt_vals, masterkey=mk_vals)

        pt_names = ["dp" + str(i) for i in range(enc_num_inputs)]
        mk_names = ["dmk" + str(i) for i in range(key_num_inputs)]
        pt_vars = [Variable(pt_names[i], width) for i in range(enc_num_inputs)]
        mk_vars = [Variable(mk_names[i], width) for i in range(key_num_inputs)]
        output_MyCipher_symb = MyCipher(plaintext=pt_vars, masterkey=mk_vars, symbolic_inputs=True, simplify=False)

        msg += f"\nMyCipher({pt_vars, mk_vars}) = {output_MyCipher_symb}"
        msg += f"\nMyCipher({pt_vals, mk_vals}) = {output_MyCipher_vals}"

        old_round_keys = MyCipher.encryption.round_keys
        MyCipher.encryption.round_keys = [Variable(f"k{i}", w) for i, w in enumerate(MyCipher.key_schedule.output_widths)]

        msg += f"\nMyCipher.key_schedule({mk_vars}) = {MyCipher.key_schedule(*mk_vars, symbolic_inputs=True, simplify=False)}"
        msg += f"\nMyCipher.round_keys = {MyCipher.encryption.round_keys}"
        msg += f"\nMyCipher.encryption({pt_vars}) = {MyCipher.encryption(*pt_vars, symbolic_inputs=True, simplify=False)}"

        for replace_multiuse_vars in [False, True]:
            with warnings.catch_warnings(record=True) as caught_warnings:
                # ignore redundant assignments
                # warnings.filterwarnings("ignore", category=UserWarning)
                key_ssa = MyCipher.key_schedule.to_ssa(input_names=mk_names, id_prefix="k", replace_multiuse_vars=replace_multiuse_vars)
                enc_ssa = MyCipher.encryption.to_ssa(input_names=pt_names, id_prefix="x", replace_multiuse_vars=replace_multiuse_vars)
                msg += ''.join(f"\nW | {cw.message}" for cw in caught_warnings)

            output_key_ssa = key_ssa.eval(input_vals=mk_vals)
            external_var2val = {var: val for var, val in zip(enc_ssa.external_vars, output_key_ssa)}
            output_enc_ssa = enc_ssa.eval(input_vals=pt_vals, external_var2val=external_var2val)

            msg += f"\nreplace_multiuse_vars: {replace_multiuse_vars}"
            msg += f"\nkey_ssa:\n{pprint.pformat(key_ssa)}"
            msg += f"\nenc_ssa:\n{pprint.pformat(enc_ssa)}"
            msg += f"\nexternal_var2val: {external_var2val}"
            msg += f"\noutput_key_ssa: {output_key_ssa}"
            msg += f"\noutput_enc_ssa      : {output_enc_ssa}"
            msg += f"\noutput_MyCipher_vals: {output_MyCipher_vals}"

            self.assertEqual(output_MyCipher_vals, output_enc_ssa, msg=msg)

        MyCipher.encryption.round_keys = old_round_keys  # get_random_cipher outputs are stored in the cached

        if VERBOSE:
            print(msg)


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(blockcipher))
    return tests
