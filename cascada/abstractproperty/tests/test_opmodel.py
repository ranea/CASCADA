"""Tests for the opmodel module."""
import decimal
import pprint
import unittest

from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.operation import BvComp


class TestOpModelGeneric(unittest.TestCase):
    """Base class for testing OpModel."""
    VERBOSE = False
    PRINT_DISTRIBUTION_ERROR = False
    PRINT_TOP_3_ERRORS = False
    PRECISION_DISTRIBUTION_ERROR = 4

    @classmethod
    def setUpClass(cls):
        cls.opmodel2emp_bv_errors = {}

    @classmethod
    def tearDownClass(cls):
        if cls.PRINT_DISTRIBUTION_ERROR:
            print()
            opmodel2emp_bv_errors_simplified = {}
            for signature, error2counter_msg in cls.opmodel2emp_bv_errors.items():
                # ignore v[1] == msg
                error2counter = {k: v[0] for k, v in error2counter_msg.items()}
                opmodel2emp_bv_errors_simplified[signature] = error2counter
            print(f"opmodel2all_emp_bv_errors:")
            pprint.pprint(opmodel2emp_bv_errors_simplified)
            print()

        if cls.PRINT_TOP_3_ERRORS:
            print()
            opmodel2top_3_emp_bv_errors = {}
            for signature, error2counter_msg in cls.opmodel2emp_bv_errors.items():
                error2counter_msg = sorted(error2counter_msg.items(), reverse=True)  # descending order
                opmodel2top_3_emp_bv_errors[signature] = error2counter_msg[:3]
            print(f"opmodel2top_3_emp_bv_errors:")
            pprint.pprint(opmodel2top_3_emp_bv_errors, width=200)
            print()

    @classmethod
    def is_valid_slow(cls, op_model, output_prop):
        """Compute whether the propagation probability is non-zero empirically
        (by iterating over the whole input space)."""
        raise NotImplementedError("subclasses must override this method")

    @classmethod
    def get_empirical_weight_slow(cls, op_model, output_prop):
        """Compute the exact weight empirically (by iterating over the whole input space).

        The exact weight is exact value (without error) of the negative binary
        logarithm (weight) of the propagation probability of :math:`(\\alpha, \\beta)`.
        """
        raise NotImplementedError("subclasses must override this method")

    def get_bv_weight_with_frac_bits(self, op_model, output_prop):
        """Get the bit-vector weight of the given model with given output property."""
        raise NotImplementedError("subclasses must override this method")

    def base_test_op_model(self, op_model_class, input_prop, output_prop, op_model_kwargs=None):
        """Test the validity and the weight constraint of the OpModel empirically."""
        if op_model_kwargs is not None:
            f = op_model_class(input_prop, **op_model_kwargs)
        else:
            f = op_model_class(input_prop)

        if hasattr(f, "precision"):
            aux_str = f"_precision={f.precision}"
        else:
            aux_str = ""
        msg = f"{f}{aux_str}(output_prop={output_prop})\n"

        is_valid = f.validity_constraint(output_prop)
        msg += "\tis_valid: {}\n".format(is_valid)

        is_valid_slow = self.is_valid_slow(f, output_prop)
        msg += "\tis_valid_slow: {}\n".format(is_valid_slow)

        self.assertEqual(is_valid, is_valid_slow, msg=msg)

        width = f.input_prop[0].val.width
        assert all(d.val.width == width for d in f.input_prop)

        weight_constraint_variable = Variable("w", f.weight_width())
        weight_constraint = f.weight_constraint(output_prop, weight_constraint_variable)
        msg += "\tweight_constraint: {}\n".format(weight_constraint)
        self.assertTrue(weight_constraint_variable in weight_constraint.atoms(Variable), msg=msg)

        if is_valid:
            decimal_weight = f.decimal_weight(output_prop)

            try:
                bv_weight_with_frac_bits = self.get_bv_weight_with_frac_bits(f, output_prop)
            except NotImplementedError as e:
                msg += f"\tignoring get_bv_weight_with_frac_bits | {e}\n"
                bv_weight_with_frac_bits = decimal.Decimal(int(f.bv_weight(output_prop)))

            bv_weight = bv_weight_with_frac_bits / decimal.Decimal(2 ** (f.num_frac_bits()))

            msg += "\tbv weight (with fb)   : {}\n".format(bv_weight_with_frac_bits)
            msg += "\tbv weight (as decimal): {}\n".format(bv_weight)
            msg += "\tdecimal weight        : {}\n".format(decimal_weight)

            self.assertLessEqual(bv_weight_with_frac_bits, f.max_weight(), msg=msg)
            # self.assertLessEqual(bv_weight - decimal_weight, 0, msg=msg)  # not necessarily True
            try:
                self.assertLessEqual((bv_weight - decimal_weight).copy_abs(),
                                     f.error(ignore_error_decimal2exact=True), msg=msg)
            except TypeError as e:
                if 'ignore_error_decimal2exact' not in str(e):
                    raise

            if type(weight_constraint) == BvComp and weight_constraint_variable in weight_constraint.args:
                if isinstance(weight_constraint.args[0], Constant):
                    # bv_weight_with_frac_bits has Decimal type
                    self.assertEqual(bv_weight_with_frac_bits, int(weight_constraint.args[0]), msg=msg)
                if isinstance(weight_constraint.args[1], Constant):
                    self.assertEqual(bv_weight_with_frac_bits, int(weight_constraint.args[1]), msg=msg)

            emp_weight = self.get_empirical_weight_slow(f, output_prop)

            msg += "\temp weight: {}\n".format(emp_weight)

            error_emp_bv = (emp_weight - bv_weight).copy_abs()
            error_emp_decimal = (emp_weight - decimal_weight).copy_abs()

            msg += "\temp_weight - bv_weight     : {}\n".format(error_emp_bv)
            msg += "\temp_weight - decimal_weight: {}\n".format(error_emp_decimal)
            msg += "\tf.error()                  : {}\n".format(f.error())

            if self.__class__.PRINT_DISTRIBUTION_ERROR:
                if hasattr(f, "precision"):
                    signature = (op_model_class.__name__, width, f.precision)
                else:
                    signature = (op_model_class.__name__, width)

                error_emp_bv_rounded = round(error_emp_bv, self.__class__.PRECISION_DISTRIBUTION_ERROR)

                if signature not in self.opmodel2emp_bv_errors:
                    self.opmodel2emp_bv_errors[signature] = {}
                if error_emp_bv_rounded not in self.opmodel2emp_bv_errors[signature]:
                    self.opmodel2emp_bv_errors[signature][error_emp_bv_rounded] = [0, None]

                self.opmodel2emp_bv_errors[signature][error_emp_bv_rounded][0] += 1
                self.opmodel2emp_bv_errors[signature][error_emp_bv_rounded][1] = msg

            # error offset '1E-27' found empirically
            f_error_dec = decimal.Decimal(f.error())
            self.assertLessEqual(error_emp_bv, f_error_dec + decimal.Decimal('1E-27'), msg=msg)
            self.assertLessEqual(error_emp_decimal, f_error_dec + decimal.Decimal('1E-27'), msg=msg)

            try:
                aux_error = f.error(ignore_error_decimal2exact=True)
                msg += "\tf.error(ignore_error_decimal2exact=True): {}\n".format(aux_error)
                self.assertLessEqual(error_emp_decimal, f.error() - aux_error, msg=msg)
            except TypeError as e:
                if 'ignore_error_decimal2exact' not in str(e):
                    raise

        if self.__class__.VERBOSE:
            print(msg)

    def base_test_pr_one_constraint_slow(self, op_model_class, input_prop, output_prop, op_model_kwargs=None):
        """Test the pr_one_constraint constraint of the OpModel empirically."""
        if op_model_kwargs is not None:
            f = op_model_class(input_prop, **op_model_kwargs)
        else:
            f = op_model_class(input_prop)

        if hasattr(f, "precision"):
            aux_str = f"_precision={f.precision}"
        else:
            aux_str = ""
        msg = f"{f}{aux_str}(output_prop={output_prop})\n"

        is_valid = f.validity_constraint(output_prop)

        msg += "\tis_valid: {}\n".format(is_valid)

        has_probability_one = f.pr_one_constraint(output_prop)

        msg += "\thas_probability_one: {}\n".format(has_probability_one)

        if is_valid:
            decimal_weight = f.decimal_weight(output_prop)

            try:
                bv_weight_with_frac_bits = self.get_bv_weight_with_frac_bits(f, output_prop)
            except NotImplementedError as e:
                msg += f"\tignoring get_bv_weight_with_frac_bits | {e}\n"
                bv_weight_with_frac_bits = decimal.Decimal(int(f.bv_weight(output_prop)))

            bv_weight = bv_weight_with_frac_bits / decimal.Decimal(2 ** (f.num_frac_bits()))

            msg += "\tdecimal weight        : {}\n".format(decimal_weight)
            msg += "\tbv weight (as decimal): {}\n".format(bv_weight)

            if has_probability_one:
                self.assertEqual(decimal_weight, 0, msg=msg)
                self.assertEqual(bv_weight, 0, msg=msg)
            else:
                self.assertGreater(decimal_weight, 0, msg=msg)
                self.assertGreater(bv_weight, 0, msg=msg)
        else:
            self.assertFalse(has_probability_one)

    def base_test_op_model_sum_pr_1(self, op_model_class, input_prop, op_model_kwargs=None):
        """Test the propagation probability sums to 1, with input property fixed and for all output properties."""
        if op_model_kwargs is not None:
            f = op_model_class(input_prop, **op_model_kwargs)
        else:
            f = op_model_class(input_prop)

        if hasattr(f, "precision"):
            aux_str = f"_precision={f.precision}"
        else:
            aux_str = ""
        msg = ""

        f_error = f.error()
        output_width = f.op.output_width(*[p.val for p in f.input_prop])

        all_prs = []
        for i in range(2**output_width):
            output_prop = f.prop_type(Constant(i, output_width))

            msg += f"{f}{aux_str}(output_prop={output_prop})\n"

            is_valid = f.validity_constraint(output_prop)
            # msg += "\tis_valid: {}\n".format(is_valid)

            if is_valid:
                if f_error == 0:
                    weight = f.decimal_weight(output_prop)
                else:
                    weight = self.get_empirical_weight_slow(f, output_prop)

                # msg += "\tweight: {}\n".format(weight)

                all_prs.append(decimal.Decimal(2)**(-weight))
            else:
                all_prs.append(0)

            msg += "\tprobability: {}\n".format(all_prs[-1])

        self.assertAlmostEqual(sum(all_prs), 1, msg=msg)

        if self.__class__.VERBOSE:
            print(msg)
