"""Manipulate non-symbolic linear characteristics."""
import decimal
import collections
import functools
import itertools
import math
import multiprocessing
import random

from cascada.bitvector import core
from cascada.bitvector import context
from cascada.bitvector import ssa as cascada_ssa
from cascada.linear import chmodel as cascada_chmodel
from cascada.linear import opmodel as cascada_opmodel

from cascada import abstractproperty


zip = functools.partial(zip, strict=True)
EmpiricalWeightData = abstractproperty.characteristic.EmpiricalWeightData


class Characteristic(abstractproperty.characteristic.Characteristic):
    """Represent linear characteristics over bit-vector functions.

    Internally, this class is a subclass of
    `abstractproperty.characteristic.Characteristic`,
    where the `Property` is a `LinearMask` type.

    As mentioned in `abstractproperty.characteristic.Characteristic`,
    the characteristic probability is defined as the product of the propagation
    probability (absolute correlation) of the `LinearMask` pairs (linear approximations)
    :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})` over :math:`f_i`.
    If :math:`f` has external variables, the characteristic probability
    approximates the absolute correlation of the input-output mask pair
    of the characteristic averaged over the set of all values of the external variables.

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.operation import RotateLeft, RotateRight
        >>> from cascada.linear.mask import LinearMask, LinearMask
        >>> from cascada.linear.chmodel import ChModel
        >>> from cascada.linear.characteristic import Characteristic
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
        >>> zm = core.Constant(0, width=16)
        >>> mk0 = RotateLeft(Constant(1, width=16), 8)  # mk0 >>> 7 == 0b0···010
        >>> mx5 = RotateRight(mk0, 7)  # ~ mk0 >>> 7 == 0b0···010
        >>> mx3__1 = RotateRight(Constant(1, width=16), 1)  # mx3__1 <<< 2 == 0b0···010
        >>> mx3_out = core.Constant(0x8002, 16)
        >>> assign_outmask_list = [zm, zm, zm, zm, mx5, mx3__1, mx5, mx5, zm, mx3_out, mx5]
        >>> ch = Characteristic([mk0, zm, zm], [zm, mx3_out, mx5], assign_outmask_list, ch_model)
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        Characteristic(ch_weight=1, assignment_weights=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            input_mask=[0x0100, 0x0000, 0x0000], output_mask=[0x0000, 0x8002, 0x0002],
            assign_outmask_list=[0x0000, 0x0000, 0x0000, 0x0000, 0x0002, 0x8000, 0x0002, 0x0002, 0x0000, 0x8002, 0x0002])
        >>> list(zip(ch.assignment_weights, ch.tuple_assign_outmask2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (LinearMask(0x0000), LinearModelFreeBranch(LinearMask(0x0000)))),
         (Decimal('0'), (LinearMask(0x0000), LinearModelFreeBranch(LinearMask(0x0000)))),
         (Decimal('0'), (LinearMask(0x0000), LinearModelBvAdd([LinearMask(0x0000), LinearMask(0x0000)]))),
         (Decimal('0'), (LinearMask(0x0000), LinearModelBvXor([LinearMask(0x0000), LinearMask(0x0000)]))),
         (Decimal('0'), (LinearMask(0x0002), LinearModelFreeBranch(LinearMask(0x0000)))),
         (Decimal('0'), (LinearMask(0x8000), LinearModelFreeBranch(LinearMask(0x0000)))),
         (Decimal('1'), (LinearMask(0x0002), LinearModelBvAdd([LinearMask(0x0002), LinearMask(0x0002)]))),
         (Decimal('0'), (LinearMask(0x0002), LinearModelBvXor([LinearMask(0x0002), LinearMask(0x0002)]))),
         (Decimal('0'), (LinearMask(0x0000), LinearModelId(LinearMask(0x0000)))),
         (Decimal('0'), (LinearMask(0x8002), LinearModelId(LinearMask(0x8002)))),
         (Decimal('0'), (LinearMask(0x0002), LinearModelId(LinearMask(0x0002))))]

    Attributes:
        input_mask: a list of `LinearMask` objects containing
            the (constant) input mask (alias of
            `abstractproperty.characteristic.Characteristic.input_prop`).
        output_mask: a list of `LinearMask` objects containing
            the (constant) output mask (alias of
            `abstractproperty.characteristic.Characteristic.output_prop`).
        external_masks: a list containing the (constant) `LinearMask` of
            the external variables of the function (alias of
            `abstractproperty.characteristic.Characteristic.external_props`).
        tuple_assign_outmask2op_model:  a tuple where each element is a pair
            containing: (1) the output (constant) `LinearMask` :math:`\Delta_{x_{i+1}}`
            of the non-trivial assignment  :math:`x_{i+1} \leftarrow f_i(x_i)`
            and (2) the `linear.opmodel.OpModel` of this assignment with
            a (constant) input `LinearMask` :math:`\Delta_{x_{i}}` (alias of
            `abstractproperty.characteristic.Characteristic.tuple_assign_outprop2op_model`).
        free_masks: a list of (symbolic) `LinearMask` objects of
            the `Characteristic.ch_model`, whose values do not affect the characteristic,
            and were replaced by constant masks in `input_mask`,
            `output_mask`, `external_masks` or `tuple_assign_outmask2op_model`
            (alias of `abstractproperty.characteristic.Characteristic.free_props`).
        var_mask2ct_mask: a `collections.OrderedDict` mapping each
            symbolic `LinearMask` in the trail to its constant mask (alias of
            `abstractproperty.characteristic.Characteristic.var_prop2ct_prop`).

    """
    _prop_label = "mask"  # for str and vrepr

    @property
    def input_mask(self):
        return self.input_prop

    @property
    def output_mask(self):
        return self.output_prop

    @property
    def external_masks(self):
        return self.external_props

    @property
    def tuple_assign_outmask2op_model(self):
        return self.tuple_assign_outprop2op_model

    @property
    def free_masks(self):
        return self.free_props

    @property
    def var_mask2ct_mask(self):
        return self.var_prop2ct_prop

    def __init__(self, input_mask, output_mask, assign_outmask_list,
                 ch_model, external_masks=None, free_masks=None,
                 empirical_ch_weight=None, empirical_data_list=None, is_valid=True):
        super().__init__(
            input_prop=input_mask, output_prop=output_mask, assign_outprop_list=assign_outmask_list,
            ch_model=ch_model, external_props=external_masks, free_props=free_masks,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid
        )

    def vrepr(self, ignore_external_masks=False):
        """Return an executable string representation.

        See also `abstractproperty.characteristic.Characteristic.vrepr`.

            >>> from cascada.bitvector.core import Constant
            >>> from cascada.bitvector.operation import RotateLeft, RotateRight
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> zm = core.Constant(0, width=16)
            >>> mk0 = RotateLeft(Constant(1, width=16), 8)  # mk0 >>> 7 == 0b0···010
            >>> mx5 = RotateRight(mk0, 7)  # ~ mk0 >>> 7 == 0b0···010
            >>> mx3__1 = RotateRight(Constant(1, width=16), 1)  # mx3__1 <<< 2 == 0b0···010
            >>> mx3_out = core.Constant(0x8002, 16)
            >>> assign_outmask_list = [zm, zm, zm, zm, mx5, mx3__1, mx5, mx5, zm, mx3_out, mx5]
            >>> ch = Characteristic([mk0, zm, zm], [zm, mx3_out, mx5], assign_outmask_list, ch_model)
            >>> ch.vrepr()  # doctest: +NORMALIZE_WHITESPACE
            "Characteristic(input_mask=[Constant(0x0100, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16)],
                output_mask=[Constant(0x0000, width=16), Constant(0x8002, width=16), Constant(0x0002, width=16)],
                assign_outmask_list=[Constant(0x0000, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16),
                    Constant(0x0000, width=16), Constant(0x0002, width=16), Constant(0x8000, width=16), Constant(0x0002, width=16),
                    Constant(0x0002, width=16), Constant(0x0000, width=16), Constant(0x8002, width=16), Constant(0x0002, width=16)],
                ch_model=ChModel(func=SpeckKeySchedule.set_num_rounds_and_return(2), mask_type=LinearMask,
                    input_mask_names=['mk0', 'mk1', 'mk2'], prefix='mx'))"
            >>> ch.srepr()
            'Ch(w=1, id=0100 0000 0000, od=0000 8002 0002)'

        """
        return super().vrepr(ignore_external_masks)

    def split(self, mask_separators):
        """Split into multiple `Characteristic` objects given the list of mask separators.

        See also `abstractproperty.characteristic.Characteristic.split`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.bitvector.operation import RotateLeft, RotateRight
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.linear.characteristic import Characteristic
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> tuple(ch_model.ssa.assignments.items())  # doctest: +NORMALIZE_WHITESPACE
            ((mx0, mk1 >>> 7), (mk2__0, Id(mk2)), (mk2__1, Id(mk2)), (mk2__2, Id(mk2)),
                (mx1, mx0 + mk2__0), (mx2, mk2__1 <<< 2), (mx3, mx2 ^ mx1),
            (mx4, mk0 >>> 7), (mx3__0, Id(mx3)), (mx3__1, Id(mx3)), (mx3__2, Id(mx3)),
                (mx5, mx4 + mx3__0), (mx6, mx5 ^ 0x0001), (mx7, mx3__1 <<< 2), (mx8, mx7 ^ mx6),
            (mk2_out, Id(mk2__2)), (mx3_out, Id(mx3__2)), (mx8_out, Id(mx8)))
            >>> mask_separators = [ (LinearMask(Variable("mx2", width=16)), LinearMask(Variable("mx3", width=16))), ]
            >>> zm = core.Constant(0, width=16)
            >>> mk0 = RotateLeft(Constant(1, width=16), 8)  # mk0 >>> 7 == 0b0···010
            >>> mx5 = RotateRight(mk0, 7)  # ~ mk0 >>> 7 == 0b0···010
            >>> mx3__1 = RotateRight(Constant(1, width=16), 1)  # mx3__1 <<< 2 == 0b0···010
            >>> mx3_out = core.Constant(0x8002, 16)
            >>> assign_outmask_list = [zm, zm, zm, zm, mx5, mx3__1, mx5, mx5, zm, mx3_out, mx5]
            >>> ch = Characteristic([mk0, zm, zm], [zm, mx3_out, mx5], assign_outmask_list, ch_model)
            >>> for ch in ch.split(mask_separators): print(ch)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0, 0, 0, 0],
                input_mask=[0x0100, 0x0000, 0x0000],
                output_mask=[0x0100, 0x0000, 0x0000],
                assign_outmask_list=[0x0000, 0x0000, 0x0000, 0x0000, 0x0100, 0x0000, 0x0000])
            Characteristic(ch_weight=1, assignment_weights=[0, 0, 1, 0, 0, 0, 0],
                input_mask=[0x0100, 0x0000, 0x0000],
                output_mask=[0x0000, 0x8002, 0x0002],
                assign_outmask_list=[0x0002, 0x8000, 0x0002, 0x0002, 0x0000, 0x8002, 0x0002])

        """
        return super().split(mask_separators)

    @classmethod
    def _sample_outprop_opmodel(cls, mask_type, ct_op_model, outmask_width, get_random_bv):
        if outmask_width > 8:
            raise ValueError("random does not support characteristics with masks with more than 8 bits")
        if isinstance(ct_op_model, cascada_opmodel.LinearModelBvXor):
            if ct_op_model.input_mask[0] != ct_op_model.input_mask[1]:
                return None
            else:
                return ct_op_model.input_mask[0]
        else:
            return super()._sample_outprop_opmodel(mask_type, ct_op_model, outmask_width, get_random_bv)
        # elif isinstance(ct_op_model, cascada_opmodel.LinearModelBvAdd) and outmask_width >= 8:
        #     # LinearModelBvAdd for more than 8-bit too slow
        #     from cascada.smt.types import environment, bv2pysmt, pysmt2bv
        #     var_prop = mask_type(core.Variable("output_mask", outmask_width))
        #     with context.Simplification(False), context.Validation(False):
        #         # explicit=True to ensure only one variable in pysmt_model
        #         bv_expr = ct_op_model.validity_constraint(var_prop, explicit=True)
        #         env = environment.reset_env()
        #         pysmt_expr = bv2pysmt(bv_expr, boolean=True, env=env)
        #         pysmt_model = env.factory.get_model(pysmt_expr)
        #         if pysmt_model is None:
        #             return None
        #         for _, pysmt_val in pysmt_model:
        #             bv_val = pysmt2bv(pysmt_val)
        #             break
        #     return mask_type(bv_val)

    @classmethod
    def random(cls, ch_model, seed, external_masks=None):
        """Return a random `Characteristic` with given `linear.chmodel.ChModel`.

        See also `abstractproperty.characteristic.Characteristic.random`.

            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.linear.characteristic import Characteristic
            >>> from cascada.bitvector.ssa import BvFunction
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [4, 4], [4, 4]
            ...     @classmethod
            ...     def eval(cls, a, b):  return a + b, a
            >>> ch_model = ChModel(MyFoo, LinearMask, ["ma", "mb"])
            >>> Characteristic.random(ch_model, 0)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=2, assignment_weights=[0, 2, 0, 0],
                input_mask=[0x2, 0x6], output_mask=[0x4, 0x5],
                assign_outmask_list=[0x7, 0x4, 0x4, 0x5])

        """
        return super().random(ch_model, seed, external_masks)

    @staticmethod
    def _num_right_inputs2weight(num_right_inputs, num_input_samples):
        if num_right_inputs == num_input_samples / 2:
            return math.inf
        else:
            inner_pr = num_right_inputs / decimal.Decimal(num_input_samples)
            correlation = (2 * inner_pr - 1).copy_abs()
            weight = - abstractproperty.opmodel.log2_decimal(correlation)
            assert weight >= 0
            return weight

    def _get_empirical_ch_weights_C(self, num_input_samples, num_external_samples, seed, num_parallel_processes,
                                    verbose=False, get_sampled_inputs=False):
        """Return a list of empirical weights (one for each ``num_external_samples``)
        by compiling and executing C code.

        See also `differential.characteristic.Characteristic._get_empirical_ch_weights_C`.
        """
        import uuid
        from cascada.bitvector.printing import BvCCodePrinter

        assert not (verbose and get_sampled_inputs)

        assert 1 <= num_input_samples <= 2 ** sum(d.val.width for d in self.input_mask)
        if self.external_masks:
            assert 1 <= num_external_samples <= 2 ** sum(d.val.width for d in self.external_masks)
        else:
            assert num_external_samples == 0

        if num_input_samples.bit_length() > 64:
            raise ValueError("max num_input_samples supported is 2**64 - 1")

        width2C_type = BvCCodePrinter._width2C_type
        get_and_mask_C_code = BvCCodePrinter._get_and_mask_C_code

        ctype_num_right_inputs = width2C_type(num_input_samples.bit_length())

        # 1 - Build the C code

        header = f"{ctype_num_right_inputs} get_num_right_inputs(uint64_t seed);"

        if hasattr(self.ch_model, "_unwrapped_ch_model"):
            # avoid gcc error due to huge line statements
            ssa = self.ch_model._unwrapped_ch_model.ssa
        else:
            ssa = self.ch_model.ssa
        eval_ssa_code_function_name = "eval_ssa"
        _, eval_ssa_code_body = ssa.get_C_code(eval_ssa_code_function_name)  # ignore header

        from pathlib import Path
        from cascada.differential import characteristic as diff_ch_module
        mt19937_filename = Path(diff_ch_module.__file__).parent.resolve() / "mt19937.c"

        def rand(my_width):
            if my_width == 1:
                rand_str = "genrand64_int1"
            elif my_width <= 8:
                rand_str = "genrand64_int8"
            elif my_width <= 16:
                rand_str = "genrand64_int16"
            elif my_width <= 32:
                rand_str = "genrand64_int32"
            elif my_width <= 64:
                rand_str = "genrand64_int64"
            else:
                raise ValueError("random bit-vectors with more than 64 bits are not supported")
            return f"{rand_str}(){get_and_mask_C_code(var.width)}"

        # stdint already in eval_ssa and in mt19937
        body = f'#include "{mt19937_filename}"\n{eval_ssa_code_body}'
        body += f"\n{ctype_num_right_inputs} get_num_right_inputs(uint64_t seed){{"

        to_sample_all_iv = num_input_samples == 2 ** sum(d.val.width for d in self.input_mask)

        if self.external_masks or not to_sample_all_iv:
            body += "\n\tinit_genrand64(seed);"

        body += f"\n\t{ctype_num_right_inputs} num_right_inputs = 0U;"

        # to_sample_all_ev cannot be used since get_num_right_inputs only returns 1 EW
        if ssa.external_vars:
            for i in range(len(ssa.external_vars)):
                var = ssa.external_vars[i]
                body += f"\n\t{width2C_type(var.width)} {var.crepr()} = {rand(var.width)};"
                if verbose:
                    body += f'\n\tprintf("\\nexternal_vars[%u] = %x", {i}U, {var.crepr()});'
            ev_args = ', '.join([v.crepr() for v in ssa.external_vars]) + ", "
        else:
            ev_args = ""

        # start for

        if not to_sample_all_iv:
            body += f"\n\tfor ({ctype_num_right_inputs} i = 0U; i < {num_input_samples}U; ++i) {{"

        for i in range(len(ssa.input_vars)):
            var = ssa.input_vars[i]
            if to_sample_all_iv:
                v = var.crepr()
                body += f"\n\t\tfor ({width2C_type(var.width+1)} {v} = 0U; {v} < {2**(var.width)}U; ++{v}) {{"
            else:
                body += f"\n\t\t{width2C_type(var.width)} {var.crepr()} = {rand(var.width)};"
            if verbose:
                body += f'\n\t\tprintf("\\ninput sample i=%u | input_vars[%u] = %x", i, {i}U, {var.crepr()});'
        if get_sampled_inputs:
            for i in range(len(ssa.input_vars)):
                aux_prefix = "[" if i == 0 else ""
                aux_suffix = "]," if i == len(ssa.input_vars) - 1 else ""
                body += f'\n\t\tprintf("{aux_prefix}0x%x,{aux_suffix}", {ssa.input_vars[i].crepr()});'
        iv_args = ', '.join([v.crepr() for v in ssa.input_vars])

        for i in range(len(ssa.output_vars)):
            var = ssa.output_vars[i]
            # var passed by reference later (no need to declare them as pointers)
            body += f"\n\t\t{width2C_type(var.width)} {var.crepr()};"
        ov_args = ', '.join(["&" + v.crepr() for v in ssa.output_vars])

        body += f"\n\t\t{eval_ssa_code_function_name}({iv_args}, {ev_args}{ov_args});"

        # parity computation

        def parity_func(my_width):
            # __builtin_parity(x) returns the parity of x, the number of 1-bits in x modulo 2.
            # __builtin_parity(a & b) returns <a, b>
            if my_width <= 16:
                return "__builtin_parity"
            elif my_width <= 32:
                return "__builtin_parityl"
            elif my_width <= 64:
                return "__builtin_parityll"
            else:
                raise ValueError("parity of bit-vectors with more than 64 bits is not supported")

        parity_inputs = []
        for i in range(len(ssa.input_vars)):
            parity_inputs.append(ssa.input_vars[i] & self.input_mask[i].val)
        for i in range(len(ssa.output_vars)):
            parity_inputs.append(ssa.output_vars[i] & self.output_mask[i].val)

        parity_outputs = []
        for parity_in in parity_inputs:
            if isinstance(parity_in, core.Constant):  # i.e., mask = 0
                assert parity_in == 0
            else:
                width = parity_in.width
                parity_outputs.append(f"{parity_func(width)}( ({width2C_type(width)})({parity_in.crepr()}) )")
        if not parity_outputs:
            parity_outputs.append("0U")

        # sum_parity = <a,x> ^ <f(x), b>
        sum_parity_c = " ^ ".join(parity_outputs)

        # "1 ^ sum_parity" is equivalent to "if (sum_parity == 0) num_right_inputs++"
        body += f"\n\t\tnum_right_inputs += 1U ^ ({width2C_type(1)})({sum_parity_c});"

        # lhs = self.input_mask[0].apply(ssa.input_vars[0])
        # for i in range(1, len(ssa.input_vars)):
        #     lhs ^= self.input_mask[i].apply(ssa.input_vars[i])
        #
        # rhs = self.output_mask[0].apply(ssa.output_vars[0])
        # for i in range(1, len(ssa.output_vars)):
        #     rhs ^= self.output_mask[i].apply(ssa.output_vars[i])
        #
        # if verbose:
        #     for i in range(len(ssa.output_vars)):
        #         var = ssa.output_vars[i]
        #         body += f'\n\t\tprintf("\\n                 | output_vars[%u] = %x", {i}U, {var.crepr()});'
        #     body += f'\n\t\tprintf("\\n                 | lhs[%u] = %x", {i}U, {lhs.crepr()});'
        #     body += f'\n\t\tprintf("\\n                 | rhs[%u] = %x", {i}U, {rhs.crepr()});'
        #
        # if_condition = f" ({width2C_type(lhs.width)})({lhs.crepr()}) == " \
        #                f" ({width2C_type(rhs.width)})({rhs.crepr()})"
        # body += f"\n\t\tif ( {if_condition} ){{"
        # body += "\n\t\t\tnum_right_inputs += 1U;"
        # body += "\n\t\t}"

        if to_sample_all_iv:
            body += "\n\t\t" + "}"*len(ssa.input_vars)
        else:
            body += "\n\t}"

        # end for

        if verbose:
            body += f'\n\tprintf("\\nnum_right_inputs = %u\\n", num_right_inputs);'

        body += "\n\treturn num_right_inputs;\n}"

        # 2 - Run the C code

        if num_parallel_processes is None or num_external_samples <= 1:
            pymod, tmpdir = cascada_ssa._compile_C_code(header, body, verbose=verbose)

            PRNG = random.Random()
            PRNG.seed(seed)

            aux_empirical_weights = []
            for _ in range(max(1, num_external_samples)):  # num_external_samples can be 0
                num_right_inputs = pymod.lib.get_num_right_inputs(PRNG.randrange(0, 2**64))
                aux_empirical_weights.append(
                    self._num_right_inputs2weight(num_right_inputs, num_input_samples)
                )
        else:
            lib_path, module_name, tmpdir = cascada_ssa._compile_C_code(header, body, return_unloaded=True, verbose=verbose)

            num_parallel_processes = min(num_parallel_processes, num_external_samples)

            PRNG = random.Random()
            PRNG.seed(seed)
            external_seeds = [PRNG.randrange(0, 2**64) for _ in range(num_external_samples)]

            chunk = num_external_samples // num_parallel_processes
            if num_external_samples % num_parallel_processes == 0:
                extra_chunk = 0
            else:
                extra_chunk = num_external_samples % num_parallel_processes
            assert chunk*(num_parallel_processes-1) + chunk + extra_chunk == num_external_samples

            aux_empirical_weights = []
            with multiprocessing.Pool(num_parallel_processes) as pool:
                async_results = [None for _ in range(num_parallel_processes)]
                for ar_index in range(len(async_results)):
                    async_results[ar_index] = pool.apply_async(
                        diff_ch_module._run_C_code_get_num_right_inputs,
                        (
                            external_seeds[ar_index*chunk],
                            module_name,
                            lib_path,
                            chunk+extra_chunk if ar_index == len(async_results) - 1 else chunk
                        )
                    )
                pool.close()
                pool.join()  # blocking call
                for ar_index in range(len(async_results)):
                    # type(process_list[process_index]) == AsyncResult
                    # and AsyncResult.get() blocks until result obtained
                    list_num_right_inputs = async_results[ar_index].get()
                    aux_empirical_weights.extend(
                        [self._num_right_inputs2weight(nri, num_input_samples) for nri in list_num_right_inputs]
                    )

            assert len(aux_empirical_weights) == num_external_samples

        tmpdir.cleanup()

        return aux_empirical_weights

    def _get_empirical_ch_weights_Python(self, num_input_samples, num_external_samples, seed, list_input_samples=None):
        """Return a list of empirical weights (one for each ``num_external_samples``)."""
        assert 1 <= num_input_samples <= 2 ** sum(d.val.width for d in self.input_mask)
        if self.external_masks:
            assert 1 <= num_external_samples <= 2 ** sum(d.val.width for d in self.external_masks)
        else:
            assert num_external_samples == 0

        PRNG = random.Random()
        PRNG.seed(seed)

        def get_random_bv(width):
            return core.Constant(PRNG.randrange(2 ** width), width)

        aux_empirical_weights = []

        with context.Simplification(False), context.Cache(False):
            input_widths = [d.val.width for d in self.input_mask]
            if list_input_samples is not None:
                assert len(list_input_samples) == num_input_samples
                def get_next_input():
                    for pt in list_input_samples:
                        assert len(self.input_mask) == len(pt)
                        yield [core.Constant(p_i, w) for p_i, w in zip(pt, input_widths)]
            elif num_input_samples == 2 ** sum(input_widths):
                # input_samples == whole input space
                def get_next_input():
                    for input_sample in itertools.product(*[range(2**w) for w in input_widths]):
                        yield [core.Constant(p_i, w) for p_i, w in zip(input_sample, input_widths)]
            else:
                def get_next_input():
                    for _ in range(num_input_samples):
                        yield [get_random_bv(mask.val.width) for mask in self.input_mask]

            found_external_vars = len(self.external_masks) > 0
            if found_external_vars:
                external_widths = [d.val.width for d in self.external_masks]
                if num_external_samples == 2 ** sum(external_widths):
                    # external_samples == whole external space
                    external_space = itertools.product(*[range(2**w) for w in external_widths])
                else:
                    external_space = range(num_external_samples)  # num_external_samples > 0

                def get_next_ssa_external_fixed():
                    for external_sample in external_space:
                        ssa = self.ch_model.ssa.copy()
                        v2c = collections.OrderedDict()
                        if isinstance(external_sample, int):
                            for v in ssa.external_vars:
                                v2c[v] = get_random_bv(v.width)
                        else:
                            for i_v, v in enumerate(ssa.external_vars):
                                v2c[v] = core.Constant(external_sample[i_v], external_widths[i_v])
                        for outvar in ssa.assignments:
                            ssa.assignments[outvar] = ssa.assignments[outvar].xreplace(v2c)
                        ssa.external_vars = []
                        # print("# new external samples")
                        # print("v2c:", v2c)
                        # print("external_masks:", self.external_masks)
                        # print("base ssa:", self.ch_model.ssa)
                        # print("ssa:", ssa)
                        yield ssa

                gen_next_ssa_external_fixed = get_next_ssa_external_fixed()

            for _ in range(max(1, num_external_samples)):  # num_external_samples can be 0
                num_right_inputs = 0

                if found_external_vars:
                    ssa = next(gen_next_ssa_external_fixed)
                    # print("> ssa:", ssa)

                for pt in get_next_input():
                    if not found_external_vars:  # func() faster than SSA.eval()
                        try:
                            ct = self.ch_model.func(*pt)
                        except ValueError as e:
                            if str(e).startswith("expected a tuple of Constant values returned"):
                                # func might return redundant symbolic outputs (e.g., k ^ k = 0)
                                ct = self.ch_model.ssa.eval(pt)
                            else:
                                raise e
                    else:
                        ct = ssa.eval(pt)

                    if not all(isinstance(x, core.Constant) for x in ct):
                        raise ValueError(f"found symbolic output in ct={ct}")

                    # lhs = <a,x>, rhs= <f(x), b>
                    lhs = self.input_mask[0].apply(pt[0])
                    for i in range(1, len(pt)):
                        lhs ^= self.input_mask[i].apply(pt[i])

                    rhs = self.output_mask[0].apply(ct[0])
                    for i in range(1, len(ct)):
                        rhs ^= self.output_mask[i].apply(ct[i])

                    if lhs == rhs:
                        num_right_inputs += 1

                    # print("pt:", pt)
                    # print("ct:", ct)
                    # print("lhs:", lhs)
                    # print("rhs:", rhs)
                    # print()

                aux_empirical_weights.append(
                    self._num_right_inputs2weight(num_right_inputs, num_input_samples)
                )

        return aux_empirical_weights

    def _get_empirical_data_complexity(self, num_input_samples, num_external_samples):
        if num_input_samples is None:
            # data complexity = small_ct * c^(-2), c == ch. correlation
            # 2 w = 2 x -log2(c) = log2(c^(-2)) ==> 2**(2 w) = c^(-2)
            p = 2 ** (2 * (self.ch_weight + self.ch_model.error()))
            # small_ct == number of input bits
            small_ct = sum(d.val.width for d in self.ch_model.input_mask)
            num_input_samples = int(math.ceil(small_ct * p))
        num_input_samples = max(
            1, min(num_input_samples, 2**sum(d.val.width for d in self.ch_model.input_mask)))

        if not self.external_masks:
            num_external_samples = 0
        else:
            if num_external_samples is None:
                # an external sample for each external bit
                num_external_samples = sum([d.width for d in self.ch_model.external_var2mask])
            num_external_samples = max(
                1, min(num_external_samples, 2 ** sum(d.width for d in self.ch_model.external_var2mask)))
        return num_input_samples, num_external_samples

    def compute_empirical_ch_weight(self, num_input_samples=None, num_external_samples=None,
                                    split_by_max_weight=None, split_by_rounds=False,
                                    seed=None, C_code=False, num_parallel_processes=None):
        """Compute and store the empirical weight.

        The main description of this method can be read from
        `abstractproperty.characteristic.Characteristic.compute_empirical_ch_weight`,
        simply by replacing `Property` by `LinearMask` and
        input-output pair by hull.

        The basic subroutine in this case consists of computing the
        fraction of right inputs for ``num_input_samples`` sampled inputs.
        An input :math:`x` is a right input if
        :math:`\langle \\alpha, x \\rangle = \\langle \\beta, f(x) \\rangle`,
        where :math:`\\alpha` is `input_mask`, :math:`\\beta` is `output_mask`
        and :math:`f` is the underlying bit-vector function.

            >>> from cascada.bitvector.core import Constant
            >>> from cascada.bitvector.operation import RotateLeft, RotateRight
            >>> from cascada.linear.mask import LinearMask, LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.linear.characteristic import Characteristic
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> zm = core.Constant(0, width=16)
            >>> mk0 = RotateLeft(Constant(1, width=16), 8)  # mk0 >>> 7 == 0b0···010
            >>> mx5 = RotateRight(mk0, 7)  # ~ mk0 >>> 7 == 0b0···010
            >>> mx3__1 = RotateRight(Constant(1, width=16), 1)  # mx3__1 <<< 2 == 0b0···010
            >>> mx3_out = core.Constant(0x8002, 16)
            >>> assign_outmask_list = [zm, zm, zm, zm, mx5, mx3__1, mx5, mx5, zm, mx3_out, mx5]
            >>> ch = Characteristic([mk0, zm, zm], [zm, mx3_out, mx5], assign_outmask_list, ch_model)
            >>> ch.compute_empirical_ch_weight(seed=0)
            >>> ch.empirical_ch_weight
            Decimal('1.093109404391481470675941626')
            >>> for data in ch.empirical_data_list: print(data)  # doctest: +NORMALIZE_WHITESPACE
            EmpiricalWeightData(weight_avg_aux_prs=1.093109404391481470675941626,
                num_aux_weights=1, num_inf_aux_weights=0, num_input_samples=192, seed=0, C_code=False)
            >>> ch.compute_empirical_ch_weight(seed=0, C_code=True)
            >>> ch.empirical_ch_weight
            Decimal('1.061400544664143309159590699')

        """
        super().compute_empirical_ch_weight(
            num_input_samples=num_input_samples, num_external_samples=num_external_samples,
            split_by_max_weight=split_by_max_weight, split_by_rounds=split_by_rounds,
            seed=seed, C_code=C_code, num_parallel_processes=num_parallel_processes)


class EncryptionCharacteristic(abstractproperty.characteristic.EncryptionCharacteristic, Characteristic):
    """Represent linear characteristics over encryption functions.

    Given a `Cipher`, an `EncryptionCharacteristic` is a
    linear characteristic  (see `Characteristic`) over
    the `Cipher.encryption`
    (where the `Cipher.key_schedule` is ignored and round key masks
    are given as constant `LinearMask`).

    The propagation probability of an `EncryptionCharacteristic` is
    similar to the expected linear probability (ELP), but instead of
    multiplying the absolute correlations, the ELP multiplies the
    square of the correlations (see https://eprint.iacr.org/2005/212).

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.operation import RotateRight, RotateLeft
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.chmodel import EncryptionChModel
        >>> from cascada.linear.characteristic import EncryptionCharacteristic
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, LinearMask)
        >>> zm = core.Constant(0, width=16)
        >>> p1 = core.Constant(0x0600, width=16)
        >>> c0 = core.Constant(0x60f0, width=16)
        >>> c1 = core.Constant(0x60c0, width=16)
        >>> k1 = core.Constant(0x0030, width=16)
        >>> bv_1800 = core.Constant(0x1800, width=16)
        >>> assign_outmask_list = [zm, zm, zm, bv_1800, bv_1800, k1, k1, k1, c1, c1, c0, c1]
        >>> ch = EncryptionCharacteristic([zm, p1], [c0, c1], assign_outmask_list, ch_model, [zm, k1])
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        EncryptionCharacteristic(ch_weight=1, assignment_weights=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            input_mask=[0x0000, 0x0600], output_mask=[0x60f0, 0x60c0], external_masks=[0x0000, 0x0030],
            assign_outmask_list=[0x0000, 0x0000, 0x0000, 0x1800, 0x1800, 0x0030,
                                 0x0030, 0x0030, 0x60c0, 0x60c0, 0x60f0, 0x60c0])
        >>> list(zip(ch.assignment_weights, ch.tuple_assign_outmask2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (LinearMask(0x0000), LinearModelFreeBranch(LinearMask(0x0600)))),
         (Decimal('0'), (LinearMask(0x0000), LinearModelBvAdd([LinearMask(0x0000), LinearMask(0x0000)]))),
         (Decimal('0'), (LinearMask(0x0000), LinearModelBvXor([LinearMask(0x0000), LinearMask(0x0000)]))),
         (Decimal('0'), (LinearMask(0x1800), LinearModelFreeBranch(LinearMask(0x0000)))),
         (Decimal('0'), (LinearMask(0x1800), LinearModelBvXor([LinearMask(0x1800), LinearMask(0x1800)]))),
         (Decimal('0'), (LinearMask(0x0030), LinearModelFreeBranch(LinearMask(0x1800)))),
         (Decimal('1'), (LinearMask(0x0030), LinearModelBvAdd([LinearMask(0x0030), LinearMask(0x0030)]))),
         (Decimal('0'), (LinearMask(0x0030), LinearModelBvXor([LinearMask(0x0030), LinearMask(0x0030)]))),
         (Decimal('0'), (LinearMask(0x60c0), LinearModelFreeBranch(LinearMask(0x0030)))),
         (Decimal('0'), (LinearMask(0x60c0), LinearModelBvXor([LinearMask(0x60c0), LinearMask(0x60c0)]))),
         (Decimal('0'), (LinearMask(0x60f0), LinearModelId(LinearMask(0x60f0)))),
         (Decimal('0'), (LinearMask(0x60c0), LinearModelId(LinearMask(0x60c0))))]

    """

    def __init__(self, input_mask, output_mask, assign_outmask_list, ch_model, external_masks=None,
                 free_masks=None, empirical_ch_weight=None, empirical_data_list=None, is_valid=True):
        assert isinstance(ch_model, cascada_chmodel.EncryptionChModel)
        if ch_model.ssa.external_vars:
            assert external_masks is not None
        super().__init__(
            input_mask, output_mask, assign_outmask_list, ch_model, external_props=external_masks, free_props=free_masks,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid)
