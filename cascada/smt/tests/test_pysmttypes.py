"""Tests for the Types module."""
import datetime
import doctest
import functools
import random
import unittest

from hypothesis import given, example, settings, assume, HealthCheck
from hypothesis.strategies import integers, sampled_from

from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.ssa import get_random_bvfunction
from cascada.bitvector.operation import (
    BvNot, BvComp, RotateLeft, RotateRight, Concat, zero_extend, repeat, Ite)

import cascada.smt.pysmttypes
from cascada.smt.pysmttypes import bv2pysmt as _bv2pysmt
from cascada.smt.pysmttypes import pysmt_model2bv_model

from pysmt import environment, logics


VERBOSE = False
WRITE_MSG = False


class TestConversion(unittest.TestCase):
    """Tests of the bv2pysmt function."""

    def setUp(self):
        self.env = environment.reset_env()

    def test_bv2pysmt(self):
        bv2pysmt = functools.partial(_bv2pysmt, env=self.env)
        fm = self.env.formula_manager
        tm = self.env.type_manager

        bx, by = Variable("x", 8), Variable("y", 8)
        b1x, b1y = Variable("x1", 1), Variable("y1", 1)
        b6x, b6y = Variable("x6", 6), Variable("y6", 6)
        px, py = bv2pysmt(bx), bv2pysmt(by)
        p1x, p1y = bv2pysmt(b1x, True), bv2pysmt(b1y, True)
        p6x, p6y = bv2pysmt(b6x), bv2pysmt(b6y)

        self.assertEqual(bv2pysmt(Constant(0, 8)), fm.BV(0, 8))
        self.assertEqual(px, fm.Symbol("x", tm.BVType(8)))
        self.assertEqual(p1x, fm.Symbol("x1", tm.BOOL()))

        self.assertEqual(bv2pysmt(~bx), fm.BVNot(px))
        self.assertEqual(bv2pysmt(~b1x, True), fm.Not(p1x))
        self.assertEqual(bv2pysmt(bx & by), fm.BVAnd(px, py))
        self.assertEqual(bv2pysmt(b1x & b1y, True), fm.And(p1x, p1y))
        self.assertEqual(bv2pysmt(bx | by), fm.BVOr(px, py))
        self.assertEqual(bv2pysmt(b1x | b1y, True), fm.Or(p1x, p1y))
        self.assertEqual(bv2pysmt(bx ^ by), fm.BVXor(px, py))
        self.assertEqual(bv2pysmt(b1x ^ b1y, True), fm.Xor(p1x, p1y))

        self.assertEqual(bv2pysmt(BvComp(bx, by)), fm.BVComp(px, py))
        self.assertEqual(bv2pysmt(BvComp(bx, by), True), fm.EqualsOrIff(px, py))
        self.assertEqual(bv2pysmt(BvNot(BvComp(bx, by))),
                         fm.BVNot(fm.BVComp(px, py)))
        self.assertEqual(bv2pysmt(BvNot(BvComp(bx, by)), True),
                         fm.Not(fm.EqualsOrIff(px, py)))

        self.assertEqual(bv2pysmt(bx < by), fm.BVULT(px, py))
        self.assertEqual(bv2pysmt(bx <= by), fm.BVULE(px, py))
        self.assertEqual(bv2pysmt(bx > by), fm.BVUGT(px, py))
        self.assertEqual(bv2pysmt(bx >= by), fm.BVUGE(px, py))

        self.assertEqual(bv2pysmt(bx << by), fm.BVLShl(px, py))
        self.assertEqual(bv2pysmt(bx >> by), fm.BVLShr(px, py))
        self.assertEqual(bv2pysmt(RotateLeft(bx, 1)), fm.BVRol(px, 1))
        self.assertEqual(bv2pysmt(RotateRight(bx, 1)), fm.BVRor(px, 1))

        def zext(pysmt_type, offset):
            # zero_extend reduces to Concat
            return fm.BVConcat(fm.BV(0, offset), pysmt_type)

        self.assertEqual(bv2pysmt(b6x << b6y, parse_shifts_rotations=True),
                         fm.BVExtract(fm.BVLShl(zext(p6x, 2), zext(p6y, 2)), 0, 5))
        self.assertEqual(bv2pysmt(RotateRight(b6x, 1), parse_shifts_rotations=True),
                         fm.BVConcat(fm.BVExtract(p6x, 0, 0), fm.BVExtract(p6x, 1, 5)))

        self.assertEqual(bv2pysmt(bx[4:2]), fm.BVExtract(px, 2, 4))
        self.assertEqual(bv2pysmt(Concat(bx, by)), fm.BVConcat(px, py))

        self.assertEqual(bv2pysmt(zero_extend(bx, 2)), zext(px, 2))
        self.assertEqual(bv2pysmt(repeat(bx, 2)), px.BVRepeat(2))
        self.assertEqual(bv2pysmt(-bx), fm.BVNeg(px))
        self.assertEqual(bv2pysmt(bx + by), fm.BVAdd(px, py))
        # bv_sum reduces to add
        self.assertEqual(bv2pysmt(bx - by), fm.BVSub(px, py))
        self.assertEqual(bv2pysmt(bx * by), fm.BVMul(px, py))
        self.assertEqual(bv2pysmt(bx / by), fm.BVUDiv(px, py))
        self.assertEqual(bv2pysmt(bx % by), fm.BVURem(px, py))

        # cannot reuse Bool and BV{1} variable with the same name
        bxx, byy = Variable("xx", 8), Variable("yy", 8)
        b1xx, b1yy, b1zz = Variable("xx1", 1), Variable("yy1", 1), Variable("zz1", 1)
        pxx, pyy = bv2pysmt(bxx), bv2pysmt(byy)
        p1xx, p1yy, p1zz = bv2pysmt(b1xx, True), bv2pysmt(b1yy, True), bv2pysmt(b1zz, True)
        self.assertEqual(bv2pysmt(Ite(b1xx, bxx, byy)), fm.Ite(p1xx, pxx, pyy))
        self.assertEqual(bv2pysmt(Ite(b1xx, b1yy, b1zz), True), fm.Ite(p1xx, p1yy, p1zz))

    @classmethod
    def setUpClass(cls):
        cls.PRNG = random.Random()

    @unittest.skip("skipping test_random_pysmt_bvf")
    @given(
        sampled_from([2, 3, 8]),
        integers(min_value=1, max_value=4),
        integers(min_value=1, max_value=4),
        integers(min_value=1, max_value=32),
        integers(min_value=0),
        integers(min_value=0, max_value=2),
        integers(min_value=0, max_value=3),
    )
    @settings(deadline=None, max_examples=500,  # ~10min
              suppress_health_check=[HealthCheck.filter_too_much])
    def test_random_pysmt_bvf(self, width, num_inputs, num_outputs, num_assignments,
                              seed, operation_set_index, num_rounds):
        assume(num_inputs + num_assignments >= num_outputs)

        # avoid errors due to simplification between rounds
        # assume(not (num_rounds >= 2 and num_inputs < num_outputs))

        if num_rounds == 0:
            num_rounds = None

        def get_time():
            now = datetime.datetime.now()
            return "{}-{}:{}".format(now.day, now.hour, now.minute)

        # timestamp messages ignored WRITE_MSG
        msg = f"{get_time()} | test_random_pysmt_bvf({width}, {num_inputs}, {num_outputs}, " \
              f"{num_assignments}, {seed}, {operation_set_index}, {num_rounds})"
        if VERBOSE:
            print(msg)

        MyFoo = get_random_bvfunction(
            width, num_inputs, num_outputs, num_assignments, seed, external_variable_prefix=None,
            operation_set_index=operation_set_index, num_rounds=num_rounds)

        self.__class__.PRNG.seed(seed)
        input_vals = []
        for _ in range(num_inputs):
            x = Constant(self.__class__.PRNG.randint(0, 2 ** width - 1), width)
            input_vals.append(x)

        output_vals = MyFoo(*input_vals)
        logged_msgs_ct = MyFoo.get_formatted_logged_msgs()

        if WRITE_MSG:
            msg += f"\ninput_vals: {input_vals}"
            msg += f"\noutput_MyFoo: {output_vals}"
            msg += f"\nformatted logged messages (constant): \n\t"
            msg += "\n\t".join(logged_msgs_ct)

        input_names = ["x" + str(i) for i in range(num_inputs)]
        input_vars = [Variable(input_names[i], width) for i in range(num_inputs)]
        output_symbolic_vals = MyFoo(*input_vars, symbolic_inputs=True, simplify=False)
        logged_msgs_symbolic = MyFoo.get_formatted_logged_msgs()

        if WRITE_MSG:
            msg += f"\nMyFoo: {output_symbolic_vals}"
            msg += f"\ninput_names: {input_names}"
            msg += f"\nformatted logged messages (symbolic): \n\t"
            msg += "\n\t".join(logged_msgs_symbolic)

        self.assertNotEqual(id(logged_msgs_ct), id(logged_msgs_symbolic))

        bv_formula = True
        for i in range(len(input_vars)):
            bv_formula &= BvComp(input_vals[i], input_vars[i])

        output_names = ["y" + str(i) for i in range(num_outputs)]
        output_vars = [Variable(output_names[i], width) for i in range(num_outputs)]
        for i in range(len(output_vars)):
            bv_formula &= BvComp(output_symbolic_vals[i], output_vars[i])

        if WRITE_MSG:
            msg += f"\noutput_vars: {output_vars}"
            msg += f"\nbv_formula: {bv_formula}"

        env = environment.reset_env()
        pysmt_formula = _bv2pysmt(bv_formula, boolean=True,
                                  parse_shifts_rotations=True, env=env)

        if WRITE_MSG:
            msg += f"\npysmt_formula: {pysmt_formula}"

        pysmt_model = env.factory.get_model(pysmt_formula, logic=logics.QF_BV)
        bv_model = pysmt_model2bv_model(pysmt_model)

        if WRITE_MSG:
            msg += f"\npysmt_model: {pysmt_model}"
            msg += f"\nbv_model: {bv_model}"

        other_input_vals = [bv_model[v] for v in input_vars]
        other_output_vals = [bv_model[v] for v in output_vars]

        self.assertSequenceEqual(input_vals, other_input_vals, msg=msg)
        self.assertSequenceEqual(output_vals, other_output_vals, msg=msg)

        if VERBOSE:
            print(msg)


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(cascada.smt.pysmttypes))
    return tests
