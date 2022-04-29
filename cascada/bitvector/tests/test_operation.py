"""Tests for the operation module."""
import doctest
import unittest

from hypothesis import given, settings
from hypothesis.strategies import integers

from cascada.bitvector.core import Constant, Variable, bitvectify
from cascada.bitvector.operation import (
    BvAnd, BvOr, BvXor, BvComp, BvUlt, BvUle, BvUgt, BvUge, BvShl,
    BvLshr, RotateLeft, RotateRight, Concat, BvAdd, BvSub, BvMul, BvUdiv,
    BvUrem, BvNeg, BvNot, Extract, Ite, BvIdentity, zero_extend, repeat,
    Operation, PrimaryOperation, SecondaryOperation, make_partial_operation
)
from cascada.bitvector.secondaryop import (
    PopCount, Reverse, PopCountSum2, PopCountSum3, PopCountDiff, LeadingZeros,
    BvMaj, BvIf, LutOperation, MatrixOperation
)

from cascada.bitvector.context import SecondaryOperationEvaluation, Cache

MIN_SIZE = 2
MAX_SIZE = 32


primary_unary_op = {BvNeg, BvNot, BvIdentity}
primary_binary_op = {
    BvAnd, BvOr, BvXor, BvComp, BvUlt, BvUle, BvUgt, BvUge,
    BvShl, BvLshr, BvAdd, BvSub, BvMul, BvUdiv, Concat, BvUrem
}
primary_ternary_op = {Ite}
primary_scalar_op = {RotateLeft, RotateRight, Extract}

secondary_unary_op = {PopCount, Reverse, LeadingZeros}
secondary_binary_op = {PopCountSum2, PopCountDiff}
secondary_ternary_op = {BvMaj, BvIf, PopCountSum3}


def get_fixed_unary_op(ct, only_secondary=False):
    mpo = make_partial_operation

    if only_secondary:
        fixed_unary_op = set()
        for op in secondary_binary_op:
            fixed_unary_op.add(mpo(op, (ct, None)))
        for op in secondary_ternary_op:
            if op != Ite:
                fixed_unary_op.add(mpo(op, (None, ct, ct + 1)))
        return fixed_unary_op
    else:
        fixed_unary_op = set()
        for op in primary_binary_op | secondary_binary_op:
            fixed_unary_op.add(mpo(op, (ct, None)))
        for op in primary_ternary_op | secondary_ternary_op:
            if op != Ite:
                fixed_unary_op.add(mpo(op, (None, ct, ct + 1)))
        for op in primary_scalar_op:
            if op == Extract:
                fixed_unary_op.add(mpo(op, (None, 1, 0)))
            elif op in [RotateLeft, RotateRight]:
                fixed_unary_op.add(mpo(op, (None, 1)))
            else:
                # avoid problems with repeat
                fixed_unary_op.add(mpo(op, (None, 2)))
        return fixed_unary_op


def get_fixed_binary_op(ct, only_secondary=False):
    mpo = make_partial_operation

    if only_secondary:
        return set(mpo(op, (ct, None, None)) for op in
                   secondary_ternary_op if op != Ite)
    else:
        return set(mpo(op, (ct, None, None)) for op in
                   primary_ternary_op | secondary_ternary_op if op != Ite)


def eval_table(input_var2val, table):
    substitutions = {**input_var2val}

    for var, expr in table.items():
        expr = expr.xreplace(substitutions)
        assert isinstance(expr, Constant)
        substitutions[var] = expr

    # last var expr is the output var
    return expr


class TestOperation(unittest.TestCase):
    """Test for the Operation class and subclasses."""

    def test_initialization_with_variables(self):
        x = Variable("x", 8)
        y = Variable("y", 8)

        ct = Constant(2, 8)

        fixed_unary_op = get_fixed_unary_op(ct)
        for op in primary_unary_op | secondary_unary_op | fixed_unary_op:
            expr = op(x)
            self.assertEqual(expr, bitvectify(expr, op.output_width(x)))
            self.assertFalse(expr.is_Atom)
            self.assertEqual(eval(expr.vrepr()), expr)
            self.assertEqual(expr.func(*expr.args), expr)

        fixed_binary_op = get_fixed_binary_op(ct)
        for op in primary_binary_op | secondary_binary_op | fixed_binary_op:
            expr = op(x, y)
            self.assertEqual(expr, bitvectify(expr, op.output_width(x, y)))
            self.assertFalse(expr.is_Atom)
            self.assertEqual(eval(expr.vrepr()), expr)
            self.assertEqual(expr.func(*expr.args), expr)

        for op in primary_ternary_op | secondary_ternary_op:
            if op == Ite:
                b = Variable("b", 1)
            else:
                b = Variable("b", 8)
            expr = op(b, x, y)
            self.assertEqual(expr, bitvectify(expr, op.output_width(b, x, y)))
            self.assertFalse(expr.is_Atom)
            self.assertEqual(eval(expr.vrepr()), expr)
            self.assertEqual(expr.func(*expr.args), expr)

        for op in primary_scalar_op:
            if op == Extract:
                expr = op(x, 0, 0)
                self.assertEqual(expr, bitvectify(expr, op.output_width(x, 0, 0)))
            else:
                expr = op(x, 2)
                self.assertEqual(expr, bitvectify(expr, op.output_width(x, 2)))
            self.assertFalse(expr.is_Atom)
            self.assertEqual(eval(expr.vrepr()), expr)
            self.assertEqual(expr.func(*expr.args), expr)

    def test_commutativity(self):
        x = Variable("x", 8)
        y = Variable("y", 8)

        ct = Constant(3, 8)
        fixed_binary_op = get_fixed_binary_op(ct)

        for op in primary_binary_op | secondary_binary_op | fixed_binary_op:
            if getattr(op, "is_symmetric", False):
                self.assertEqual(op(x, y), op(y, x))
            else:
                self.assertNotEqual(op(x, y), op(y, x))

        z = Variable("z", 8)
        for op in primary_ternary_op | secondary_ternary_op:
            if op == Ite:
                continue
            if getattr(op, "is_symmetric", False):
                self.assertEqual(op(x, y, z), op(x, z, y))
                self.assertEqual(op(x, y, z), op(y, x, z))
                self.assertEqual(op(x, y, z), op(y, z, x))
                self.assertEqual(op(x, y, z), op(z, x, y))
                self.assertEqual(op(x, y, z), op(z, y, x))
            else:
                self.assertNotEqual(op(x, y, z), op(x, z, y))
                self.assertNotEqual(op(x, y, z), op(y, x, z))
                self.assertNotEqual(op(x, y, z), op(y, z, y))
                self.assertNotEqual(op(x, y, z), op(z, x, y))
                self.assertNotEqual(op(x, y, z), op(z, y, x))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_constant_comparisons(self, width, x, y):
        modulus = 2 ** width
        x, y = x % modulus, y % modulus
        bvx = Constant(x, width)
        bvy = Constant(y, width)

        self.assertEqual((x == y), (bvx == bvy))
        self.assertEqual((bvx == bvy), (bvx == y))

        self.assertEqual((x != y), (bvx != bvy))
        self.assertEqual((bvx != bvy), (bvx != y))

        self.assertEqual((x < y), (bvx < bvy))
        self.assertEqual((bvx < bvy), (bvx < y))

        self.assertEqual((x <= y), (bvx <= bvy))
        self.assertEqual((bvx <= bvy), (bvx <= y))

        self.assertEqual((x > y), (bvx > bvy))
        self.assertEqual((bvx > bvy), (bvx > y))

        self.assertEqual((x >= y), (bvx >= bvy))
        self.assertEqual((bvx >= bvy), (bvx >= y))

    def test_python_operators(self):
        x = Variable("x", 8)
        y = Variable("y", 8)

        self.assertEqual(BvNot(x), ~x)

        expr = x
        expr &= y
        self.assertEqual(BvAnd(x, y), x & y)
        self.assertEqual(x & y, expr)

        expr = x
        expr |= y
        self.assertEqual(BvOr(x, y), x | y)
        self.assertEqual(x | y, expr)

        expr = x
        expr ^= y
        self.assertEqual(BvXor(x, y), x ^ y)
        self.assertEqual(x ^ y, expr)

        self.assertEqual(BvUlt(x, y), (x < y))

        self.assertEqual(BvUle(x, y), (x <= y))

        self.assertEqual(BvUgt(x, y), (x > y))

        self.assertEqual(BvUge(x, y), (x >= y))

        expr = x
        expr <<= y
        self.assertEqual(BvShl(x, y), x << y)
        self.assertEqual(x << y, expr)

        expr = x
        expr >>= y
        self.assertEqual(BvLshr(x, y), x >> y)
        self.assertEqual(x >> y, expr)

        self.assertEqual(Extract(x, 4, 2), x[4:2])
        self.assertEqual(Extract(x, 4, 4), x[4:4])
        self.assertEqual(Extract(x, 7, 1), x[:1])
        self.assertEqual(Extract(x, 6, 0), x[6:])

        self.assertEqual(BvNeg(x), -x)

        expr = x
        expr += y
        self.assertEqual(BvAdd(x, y), x + y)
        self.assertEqual(x + y, expr)

        expr = x
        expr -= y
        self.assertEqual(BvSub(x, y), x - y)
        self.assertEqual(x - y, expr)

        expr = x
        expr *= y
        self.assertEqual(BvMul(x, y), x * y)
        self.assertEqual(x * y, expr)

        expr = x
        expr /= y
        self.assertEqual(BvUdiv(x, y), x / y)
        self.assertEqual(x / y, expr)

        expr = x
        expr %= y
        self.assertEqual(BvUrem(x, y), x % y)
        self.assertEqual(x % y, expr)

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
    )
    @settings(deadline=None)
    def test_pysmt_operations(self, width, x, y):
        try:
            from pysmt import shortcuts as sc
        except ImportError:
            return

        modulus = 2 ** width
        x = x % modulus
        y = y % modulus
        bvx = Constant(x, width)
        bvy = Constant(y, width)
        psx = sc.BV(x, width)
        psy = sc.BV(y, width)

        def eval_pysmt(pysmt_var):
            return pysmt_var.simplify().constant_value()

        self.assertEqual(~bvx, eval_pysmt(sc.BVNot(psx)))
        self.assertEqual(bvx & bvy, eval_pysmt(sc.BVAnd(psx, psy)))
        self.assertEqual(bvx | bvy, eval_pysmt(sc.BVOr(psx, psy)))
        self.assertEqual(bvx ^ bvy, eval_pysmt(sc.BVXor(psx, psy)))

        self.assertEqual(BvComp(bvx, bvy), eval_pysmt(sc.BVComp(psx, psy)))
        self.assertEqual((bvx < bvy), eval_pysmt(sc.BVULT(psx, psy)))
        self.assertEqual((bvx <= bvy), eval_pysmt(sc.BVULE(psx, psy)))
        self.assertEqual((bvx > bvy), eval_pysmt(sc.BVUGT(psx, psy)))
        self.assertEqual((bvx >= bvy), eval_pysmt(sc.BVUGE(psx, psy)))

        r = y % bvx.width
        self.assertEqual(bvx << bvy, eval_pysmt(sc.BVLShl(psx, psy)))
        self.assertEqual(bvx >> bvy, eval_pysmt(sc.BVLShr(psx, psy)))
        self.assertEqual(RotateLeft(bvx, r), eval_pysmt(sc.BVRol(psx, r)))
        self.assertEqual(RotateRight(bvx, r), eval_pysmt(sc.BVRor(psx, r)))

        bvb = Constant(y % 2, 1)
        psb = sc.Bool(bool(bvb))
        self.assertEqual(Ite(bvb, bvx, bvy), eval_pysmt(sc.Ite(psb, psx, psy)))
        j = y % bvx.width
        self.assertEqual(bvx[:j], eval_pysmt(sc.BVExtract(psx, start=j)))
        self.assertEqual(bvx[j:], eval_pysmt(sc.BVExtract(psx, end=j)))
        self.assertEqual(Concat(bvx, bvy), eval_pysmt(sc.BVConcat(psx, psy)))
        self.assertEqual(zero_extend(bvx, j), eval_pysmt(sc.BVZExt(psx, j)))
        self.assertEqual(repeat(bvx, 1 + j), eval_pysmt(psx.BVRepeat(1 + j)))

        self.assertEqual(-bvx, eval_pysmt(sc.BVNeg(psx)))
        self.assertEqual(bvx + bvy, eval_pysmt(sc.BVAdd(psx, psy)))
        self.assertEqual(bvx - bvy, eval_pysmt(sc.BVSub(psx, psy)))
        self.assertEqual(bvx * bvy, eval_pysmt(sc.BVMul(psx, psy)))
        if bvy > 0:
            self.assertEqual(bvx / bvy, eval_pysmt(sc.BVUDiv(psx, psy)))
            self.assertEqual(bvx % bvy, eval_pysmt(sc.BVURem(psx, psy)))

    # noinspection PyTypeChecker
    def test_invalid_operations(self):
        x = Variable("x", 8)
        y = Variable("y", 9)
        b = Variable("b", 1)
        max_value = (2 ** 8) - 1

        with self.assertRaises(TypeError):
            x ** 2
        with self.assertRaises(TypeError):
            x // 2
        with self.assertRaises(TypeError):
            abs(x)

        ct = Constant(4, 8)

        fixed_unary_op = get_fixed_unary_op(ct)
        for op in primary_unary_op | secondary_unary_op | fixed_unary_op:
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op()
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op(-1)
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op(max_value + 1)
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op(1)
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op(x, x)
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op(x, x, x)

        fixed_binary_op = get_fixed_binary_op(ct)
        for op in primary_binary_op | secondary_binary_op | fixed_binary_op:
            with self.assertRaises((AssertionError, TypeError)):
                op()
            with self.assertRaises((AssertionError, TypeError)):
                op(x, -1)  # invalid range
            with self.assertRaises((AssertionError, TypeError)):
                op(x, max_value + 1)  # invalid range
            with self.assertRaises((AssertionError, TypeError)):
                op(2, 3)  # at least 1 Term
            if op != Concat:
                with self.assertRaises((AssertionError, TypeError)):
                    op(x, y)  # != width
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op(x)  # invalid # of args
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op(x, x, x)

        for op in primary_ternary_op | secondary_ternary_op:
            if op == Ite:
                continue
            with self.assertRaises((AssertionError, TypeError)):
                op()
            with self.assertRaises((AssertionError, TypeError)):
                op(x, x, -1)  # invalid range
            with self.assertRaises((AssertionError, TypeError)):
                op(x, x, max_value + 1)  # invalid range
            with self.assertRaises((AssertionError, TypeError)):
                op(1, 2, 3)  # at least 1 Term
            with self.assertRaises((AssertionError, TypeError)):
                op(x, x, y)  # != width
            with self.assertRaises((AssertionError, TypeError)):
                op(x)  # invalid # of args
            with self.assertRaises((AssertionError, TypeError)):
                op(x, x)
            with self.assertRaises((AssertionError, TypeError)):
                op(x, x, x, x)

        for op in primary_scalar_op:
            if op == Extract:
                continue
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op()
            with self.assertRaises((AssertionError, TypeError, ValueError)):
                op(x, -1)

        with self.assertRaises((AssertionError, TypeError)):
            Extract(x, 0, 1)
        with self.assertRaises((AssertionError, TypeError)):
            Extract(x, x, 1)

        with self.assertRaises((AssertionError, TypeError)):
            Ite(b, x, y)
        with self.assertRaises((AssertionError, TypeError)):
            Ite(x, x, x)
        with self.assertRaises((AssertionError, TypeError)):
            Ite(0, x, x)

    # properties that hold for constants and variables
    def _simple_properties_v1(self, x):
        width = x.width
        allones = BvNot(Constant(0, width))

        self.assertTrue( ~~x == x )
        # self.assertTrue( ~(x & y) == (~x) | (~y) )
        # self.assertTrue( ~(x | y) == (~x) & (~y) )

        self.assertTrue( x & 0 == 0 & x == 0 )
        self.assertTrue( x & allones == allones & x == x )
        self.assertTrue( x & x == x )
        self.assertTrue( x & (~x) == 0 )

        self.assertTrue( x | 0 == 0 | x == x )
        self.assertTrue( x | allones == allones | x == allones )
        self.assertTrue( x | x == x )
        self.assertTrue( x | (~x) == allones )

        self.assertTrue( x ^ 0 == 0 ^ x == x )
        self.assertTrue( x ^ allones == allones ^ x == ~x )
        self.assertTrue( x ^ x == 0 )
        self.assertTrue( x ^ (~x) == allones )

        self.assertTrue( x << 0 == x >> 0 == x )
        self.assertTrue( 0 << x == 0 >> x == 0 )
        if isinstance(x, Constant):
            r = min(2 * int(x), x.width)
            self.assertTrue( (x << x) << x == x << r )
            self.assertTrue( (x >> x) >> x == x >> r )
        elif isinstance(x, Variable) and x.width >= 2:
            self.assertTrue( (x << 1) << 1 == x << 2 )
            self.assertTrue( (x >> 1) >> 1 == x >> 2 )

        n = x.width
        self.assertTrue( RotateLeft(x, 0) == RotateRight(x, 0) == x )
        if x.width > 2:
            self.assertTrue( RotateLeft(RotateLeft(x, 1), 1) == RotateLeft(x, 2) )
            self.assertTrue( RotateRight(RotateRight(x, 1), 1) == RotateRight(x, 2) )
        if x.width > 3:
            self.assertTrue( RotateLeft(RotateRight(x, 1), 2) == RotateRight(x, n - 1) )
            self.assertTrue( RotateRight(RotateLeft(x, 1), 2) == RotateLeft(x, n - 1) )

        if isinstance(x, Constant):
            i = int(x) % (width - 1)
        elif isinstance(x, Variable) and x.width >= 2:
            i = width - 2
        else:
            raise ValueError("invalid x: {}".format(x))
        n = x.width
        self.assertTrue( x[:] == x )
        self.assertTrue( x[:i][1:0] == x[i + 1:i] )
        self.assertTrue( Concat(x, x)[n - 1:i] == x[:i] )  # n - 1 <= x.width - 1
        self.assertTrue( Concat(x, x)[n + i:n] == x[i:] )  # n >= x.width
        self.assertTrue( (x << i)[:i] == x[n - 1 - i:] )  # i <= i
        self.assertTrue( RotateLeft(x, i)[:i + 1] == x[n - 1 - i: 1] )  # i <= i + 1
        self.assertTrue( (x >> i)[n - i - 1:] == x[n - 1:i] )  # n - i - 1 < n - i
        self.assertTrue( RotateRight(x, i)[n - i - 1:] == x[n - 1:i] )
        # assert (x & y)[0] == x[0] & y[0]

        if isinstance(x, Constant):
            i = int(x) % (width - 1)
        else:
            self.assertTrue( x.width >= 2 )
            i = width - 2
        self.assertTrue( Concat(x[:i + 1], x[i:]) == x )
        zero_bit = Constant(0, 1)
        self.assertTrue( Concat(zero_bit, Concat(zero_bit, x)) == Concat(Constant(0, 2), x) )
        self.assertTrue( Concat(Concat(x, zero_bit), zero_bit) == Concat(x, Constant(0, 2)) )

        self.assertTrue( -(-x) == x )

        self.assertTrue( x + 0 == 0 + x == x )
        self.assertTrue( x + (-x) == 0 )
        self.assertTrue( (-x) + 1 == 1 - x )
        self.assertTrue( 1 + (-x) == 1 - x )

        self.assertTrue( x - 0 == x )
        self.assertTrue( 0 - x == -x )
        self.assertTrue( x - x == 0 )
        self.assertTrue( 1 - (-x) == 1 + x )

        self.assertTrue( x * 0 == 0 * x == 0 )
        self.assertTrue( x * 1 == 1 * x == x )

        if x != 0:
            self.assertTrue( x / x == 1 )
            self.assertTrue( 0 / x == 0 )
            self.assertTrue( x / 1 == x )

            self.assertTrue( x % x == 0 % x == x % 1 == 0 )

    # properties that hold for constants and variables for multiple inputs
    def _simple_properties_v2(self, x, y, z):
        all_ones = Constant((2 ** x.width) - 1, x.width)
        not_one = ~Constant(1, x.width)

        self.assertEqual(1 + (x + all_ones), x)
        self.assertEqual(1 ^ (x ^ 1), x)
        self.assertEqual(1 & (x & not_one), 0)
        self.assertEqual(1 | (x | not_one), all_ones)

        self.assertEqual(x + (y - x), y)
        self.assertEqual(x ^ (y ^ x), y)
        self.assertEqual(x & (y & x), x & y)
        self.assertEqual(x | (y | x), x | y)

        self.assertEqual((x ^ z) + (y - (x ^ z)), y)
        self.assertEqual((x + z) ^ (y ^ (x + z)), y)

        self.assertTrue( (-x) + y == y - x )
        self.assertTrue( x + (-y) == x - y )
        self.assertTrue( x - (-y) == x + y )

        # self.assertEqual((x + z) + (y - x), z + y)
        self.assertEqual((x ^ z) ^ (y ^ x), z ^ y)
        self.assertEqual((x & z) & (y & (~x)), 0)
        self.assertEqual((x | z) | (y | (~x)), all_ones)

        self.assertEqual(x & (x | y), x)
        self.assertEqual(x | (x & y), x)

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    @settings(deadline=None)
    def test_constant_properties(self, width, x, y, z):
        x = Constant(x % (2 ** width), width)
        y = Constant(y % (2 ** width), width)
        z = Constant(z % (2 ** width), width)
        self._simple_properties_v1(x)
        self._simple_properties_v1(y)
        self._simple_properties_v1(z)
        self._simple_properties_v2(x, y, z)

    def test_variable_properties(self):
        self._simple_properties_v1(Variable("x", 8))
        self._simple_properties_v2(
            Variable("x", 8), Variable("y", 8), Variable("z", 8))

    # properties that hold for constants but not for variables
    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_ct_nonvar_properties(self, width, x, y):
        x = Constant(x % (2 ** width), width)
        y = Constant(y % (2 ** width), width)

        # x - y = \neg (\neg x + y)
        self.assertEqual(x - y, ~(~(x) + y))

        self.assertEqual(-(x + y), -(x) + -(y))
        self.assertEqual(-(x * y), -(x) * y)
        # # not True
        # if y != 0:
        #     self.assertEqual(-(x / y), -(x) / y)
        # self.assertEqual(-(x % y), -(x) % y)

        # hacker's delight p.36
        self.assertEqual( -x, ~x + 1 )
        self.assertEqual( -x, ~(x - 1) )
        self.assertEqual( ~x, -x - 1 )
        self.assertEqual( -~x, x + 1 )
        self.assertEqual( ~-x, x - 1 )
        self.assertEqual(x + y, x - (~y) - 1)
        self.assertEqual(x - y, x + (~y) + 1)
        self.assertEqual(x ^ y, (x | y) - (x & y))
        self.assertEqual(x & (~y), (x | y) - y)
        self.assertEqual(x | y, (x & (~y)) + y)

    def test_speck_like_inverse(self):
        k = Variable("k", 8)

        # noinspection PyShadowingNames
        def f(x, y):
            x = (RotateRight(x, 7) + y) ^ k
            y = RotateLeft(y, 2) ^ x
            return x, y

        # noinspection PyShadowingNames
        def f_inverse(x, y):
            y = RotateRight(x ^ y, 2)
            x = RotateLeft((x ^ k) - y, 7)
            return x, y

        x, y = Variable("x", 8), Variable("y", 8)

        self.assertEqual(f_inverse(*f(x, y)), (x, y))
        self.assertEqual(f(*f_inverse(x, y)), (x, y))

    @given(
        integers(min_value=MIN_SIZE, max_value=2*MAX_SIZE),
        integers(min_value=0),
    )
    # @settings(max_examples=1000000)
    def test_secondary_unary_op(self, width, x):
        bv = Constant(x % (2 ** width), width)
        hw = PopCount(bv)
        rev = Reverse(bv)
        lz = LeadingZeros(bv)

        lz_bf = ""
        for i in bv.bin()[2:]:
            if i == "0":
                lz_bf += "1"
            else:
                break
        lz_bf += "0"*(bv.width - len(lz_bf))

        self.assertEqual(int(hw), bin(int(bv)).count("1"))
        self.assertEqual(rev.bin()[2:], bv.bin()[2:][::-1])
        self.assertEqual(lz.bin()[2:], lz_bf)

    @given(
        integers(min_value=MIN_SIZE, max_value=2*MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    # @settings(max_examples=100000)
    def test_PopCount(self, width, x, y, z):
        bvx = Constant(x % (2 ** width), width)
        hwx = PopCount(bvx)
        bvy = Constant(y % (2 ** width), width)
        hwy = PopCount(bvy)
        bvz = Constant(z % (2 ** width), width)
        hwz = PopCount(bvz)

        popsum2 = PopCountSum2(bvx, bvy)
        popsum3 = PopCountSum3(bvx, bvy, bvz)
        popdiff = PopCountDiff(bvx, bvy)

        self.assertEqual(
            int(hwx) + int(hwy),
            int(popsum2))
        self.assertEqual(
            int(hwx) + int(hwy) + int(hwz),
            int(popsum3))
        self.assertEqual(
            (int(hwx) - int(hwy)) % (2 ** popdiff.width),
            int(popdiff))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    @settings(deadline=None)
    def test_internal_doit(self, width, x, y, z):
        ct = Constant(hash((x, y, z)) % (2 ** width), width)

        x_const = Constant(x % (2 ** width), width)
        x_var = Variable("x", width)

        fixed_unary_op = get_fixed_unary_op(ct)
        for op in primary_unary_op | secondary_unary_op | fixed_unary_op:
            if op in fixed_unary_op and op.base_op in [BvUdiv, BvUrem] and x_const == 0:
                lhs = op(x_var).doit().xreplace({x_var: x_const + 1})
                if isinstance(op(x_var), Operation):
                    # variable has not memoization_table
                    lhs_2 = eval_table({x_var: x_const + 1}, op(x_var).memoization_table())
                rhs = op(x_const + 1)
            else:
                lhs = op(x_var).doit().xreplace({x_var: x_const})
                if isinstance(op(x_var), Operation):
                    lhs_2 = eval_table({x_var: x_const}, op(x_var).memoization_table())
                rhs = op(x_const)
            self.assertEqual(lhs, rhs, msg=f"{lhs} != {rhs}")
            if isinstance(op(x_var), Operation):
                self.assertEqual(lhs_2, rhs, msg=f"{lhs_2} != {rhs}")

        y_const = Constant(y % (2 ** width), width)
        y_var = Variable("y", width)
        fixed_binary_op = get_fixed_binary_op(ct)
        for op in primary_binary_op | secondary_binary_op | fixed_binary_op:
            if op in [BvUdiv, BvUrem] and y_const == 0:
                lhs = op(x_var, y_var).doit().xreplace({x_var: x_const, y_var: y_const + 1})
                if isinstance(op(x_var, y_var), Operation):
                    # variable has not memoization_table
                    lhs_2 = eval_table({x_var: x_const, y_var: y_const + 1}, op(x_var, y_var).memoization_table())
                rhs = op(x_const, y_const + 1)
            else:
                lhs = op(x_var, y_var).doit().xreplace({x_var: x_const, y_var: y_const})
                if isinstance(op(x_var, y_var), Operation):
                    lhs_2 = eval_table({x_var: x_const, y_var: y_const}, op(x_var, y_var).memoization_table())
                rhs = op(x_const, y_const)
            self.assertEqual(lhs, rhs, msg=f"{lhs} != {rhs}")
            if isinstance(op(x_var, y_var), Operation):
                self.assertEqual(lhs_2, rhs, msg=f"{lhs_2} != {rhs}")

        z_const = Constant(z % (2 ** width), width)
        z_var = Variable("z", width)
        for op in primary_ternary_op | secondary_ternary_op:
            if op == Ite:
                continue
            lhs = op(x_var, y_var, z_var).doit().xreplace(
                {x_var: x_const, y_var: y_const, z_var: z_const})
            if isinstance(op(x_var, y_var, z_var), Operation):
                lhs_2 = eval_table({x_var: x_const, y_var: y_const, z_var: z_const},
                                   op(x_var, y_var, z_var).memoization_table())
            rhs = op(x_const, y_const, z_const)
            self.assertEqual(lhs, rhs, msg=f"{lhs} != {rhs}")
            if isinstance(op(x_var, y_var, z_var), Operation):
                self.assertEqual(lhs_2, rhs, msg=f"{lhs_2} != {rhs}")

        b_var = Variable("b", 1)
        b_const = Constant(x % (2 ** 1), 1)
        lhs = Ite(b_var, y_var, z_var).doit().xreplace(
            {b_var: b_const, y_var: y_const, z_var: z_const})
        lhs_2 = eval_table({b_var: b_const, y_var: y_const, z_var: z_const},
                           Ite(b_var, y_var, z_var).memoization_table())
        rhs = Ite(b_const, y_const, z_const)
        self.assertEqual(lhs, rhs, msg=f"{lhs} != {rhs}")
        self.assertEqual(lhs_2, rhs, msg=f"{lhs_2} != {rhs}")

        y = y % width
        for op in primary_scalar_op:
            eval_lhs_2 = False
            if op == Extract:
                lhs = op(x_var, y, z % (y + 1)).xreplace({x_var: x_const})
                if isinstance(op(x_var, y, z % (y + 1)), Operation):
                    lhs_2 = eval_table({x_var: x_const}, op(x_var, y, z % (y + 1)).memoization_table())
                    eval_lhs_2 = True
                rhs = op(x_const, y, z % (y + 1))
            # elif op == Repeat and y == 0:
            #     lhs = op(x_var, y + 1).doit().xreplace({x_var: x_const})
            #     if isinstance(op(x_var, y + 1), Operation):
            #         lhs_2 = eval_table({x_var: x_const}, op(x_var, y + 1).memoization_table())
            #         eval_lhs_2 = True
            #     rhs = op(x_const, y + 1)
            else:
                lhs = op(x_var, y).doit().xreplace({x_var: x_const})
                if isinstance(op(x_var, y), Operation):
                    lhs_2 = eval_table({x_var: x_const}, op(x_var, y).memoization_table())
                    eval_lhs_2 = True
                rhs = op(x_const, y)
            self.assertEqual(lhs, rhs, msg=f"{lhs} != {rhs}")
            if eval_lhs_2:
                self.assertEqual(lhs_2, rhs, msg=f"{lhs_2} != {rhs}")

    def test_symbolic_secondary_op_evaluation(self):
        x = Variable("x", 8)
        y = Variable("y", 8)
        z = Variable("z", 8)

        ct = Constant(5, 8)

        fixed_unary_op = get_fixed_unary_op(ct, only_secondary=True)
        fixed_binary_op = get_fixed_binary_op(ct, only_secondary=True)

        def get_expr():
            if op in secondary_unary_op | fixed_unary_op:
                return op(x)
            elif op in secondary_binary_op | fixed_binary_op:
                return op(x, y)
            elif op in secondary_ternary_op:
                return op(x, y, z)

        with Cache(False):
            for op in secondary_unary_op | secondary_binary_op | secondary_ternary_op | \
                      fixed_unary_op | fixed_binary_op:
                self.assertTrue(isinstance(get_expr(), SecondaryOperation))
                self.assertFalse(isinstance(get_expr().doit(), SecondaryOperation))
                with SecondaryOperationEvaluation(True):
                    self.assertFalse(isinstance(get_expr(), SecondaryOperation))

                for expr in get_expr().memoization_table(decompose_sec_ops=True).values():
                    self.assertTrue(isinstance(expr, PrimaryOperation))

    def test_formula_size(self):
        x = Variable("x", 8)
        y = Variable("y", 8)
        z = Variable("z", 8)

        self.assertEqual(Constant(0, 8).formula_size(), 6)
        self.assertEqual(Constant(1, 8).formula_size(), 7)
        self.assertEqual((~Constant(0, 8)).formula_size(), 14)
        self.assertEqual(x.formula_size(), 6)
        self.assertEqual((x & y).formula_size(), 18)
        self.assertEqual(BvMaj(x, y, z).formula_size(), 66)

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
    )
    def test_singleton_partial_op(self, width, x):
        ct = Constant(x % (2 ** width), width)
        self.assertEqual(get_fixed_unary_op(ct) , get_fixed_unary_op(ct))
        self.assertEqual(get_fixed_binary_op(ct), get_fixed_binary_op(ct))

    @staticmethod
    def _identity_matrix(dimension):
            rows = []
            for i in range(dimension):
                my_row = []
                for j in range(dimension):
                    my_row.append(Constant(1, 1) if i == j else Constant(0, 1))
                rows.append(my_row)
            return rows

    @given(
        integers(min_value=MIN_SIZE, max_value=4),
    )
    def test_lut_matrix(self, width):
        class ReverseLut(LutOperation):  # example permutation
            lut = [Reverse(Constant(i, width)) for i in range(2 ** width)]

        class Repeat3Lut(LutOperation):  # example non-permutation
            lut = [repeat(Constant(i, width), 3) for i in range(2 ** width)]

        id_matrix = self._identity_matrix(width)

        class ROL1Matrix(MatrixOperation):  # example permutation
            matrix = [id_matrix[-1]] + id_matrix[:-1]

        class DuplicateMatrix(MatrixOperation):  # example non-permutation
            matrix = id_matrix + id_matrix

        v = Variable("v", width)
        for op in [ReverseLut, Repeat3Lut, ROL1Matrix, DuplicateMatrix]:
            expr = op(v)
            self.assertEqual(expr, bitvectify(expr, op.output_width(v)))
            self.assertFalse(expr.is_Atom)
            self.assertEqual(eval(expr.vrepr()), expr)
            self.assertEqual(expr.func(*expr.args), expr)

            with self.assertRaises((AssertionError, TypeError)):
                op()
            with self.assertRaises((AssertionError, TypeError)):
                op(-1)
            with self.assertRaises((AssertionError, TypeError)):
                op(1)
            with self.assertRaises((AssertionError, TypeError)):
                op(v, v)

        for x in range(2**width):
            x_const = Constant(x % (2 ** width), width)
            x_var = Variable("x", width)

            for op in [ReverseLut, Repeat3Lut, ROL1Matrix, DuplicateMatrix]:
                out_var_replaced = op(x_var).doit().xreplace({x_var: x_const})
                out_val = op(x_const)
                self.assertEqual(out_var_replaced, out_val)

                if op == ReverseLut:
                    self.assertEqual(out_val, Reverse(x_const), msg=f"{x_const}, {ReverseLut.lut}")
                elif op == Repeat3Lut:
                    self.assertEqual(out_val, repeat(x_const, 3), msg=f"{x_const}, {Repeat3Lut.lut}")
                elif op == ROL1Matrix:
                    self.assertEqual(out_val, RotateLeft(x_const, 1), msg=f"{x_const}, {ROL1Matrix.matrix}")
                elif op == DuplicateMatrix:
                    self.assertEqual(out_val, repeat(x_const, 2), msg=f"{x_const}, {DuplicateMatrix.matrix}")


# noinspection PyUnusedLocal,PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    import cascada.bitvector.operation
    import cascada.bitvector.secondaryop
    tests.addTests(doctest.DocTestSuite(cascada.bitvector.operation))
    tests.addTests(doctest.DocTestSuite(cascada.bitvector.secondaryop))
    return tests
